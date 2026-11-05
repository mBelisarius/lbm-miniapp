#ifndef LBMINI_OPENCL_PROGRAM_HPP_
#define LBMINI_OPENCL_PROGRAM_HPP_

/**
 * @file Program.hpp
 * @brief Host-side helpers shared by the two OpenCL backends: device
 *        selection, JIT specialisation of the kernel source, program build.
 *
 * ### JIT specialisation
 *
 * Every solver instance compiles its own `cl::Program`. In front of the
 * embedded `kernels.cl` it places a block of `#define`s (`ProgramDefines()`)
 * that turns everything the kernels need besides the buffers into
 * compile-time constants:
 *
 *  - the mesh extents `LBM_NX`, `LBM_NY`, `LBM_N` (index arithmetic by
 *    immediates, division by `ny` strength-reduced, no bounds parameters);
 *  - the fluid constants, the lattice shift and the Sod left/right states,
 *    written as exact hexadecimal floating literals (`%a`), so the device sees
 *    bit-for-bit the values the reference solver uses;
 *  - the lattice velocities and the whole per-direction streaming stencil
 *    (`lbmini::DirectionStencil`: corner offsets, IDW weights, nearest-node
 *    flags) as initialiser lists of `__constant` tables. The kernels index
 *    them only with unrolled loop counters, so every entry folds into an
 *    immediate operand, the nearest-node branch of a regular direction and
 *    every zero-weight corner of a snapped direction become dead code, and a
 *    one-hot weight multiplies by `1.0`, which the compiler drops.
 *
 * The program is built without any relaxed-math option: `exp`, divisions and
 * NaN propagation keep their IEEE semantics, which the equivalence test against
 * `lbmini::plain::LbmTube` (relative 1e-9) depends on.
 */

#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <CL/cl2.hpp>
#include "Data.hpp"
#include "Lbm/Stencil.hpp"
#include "Utils.hpp"

namespace lbmini::opencl::detail {
/// Throws a `std::runtime_error` naming the backend and the failed call.
inline void Check(const cl_int err, const char* what, const std::string& who) {
  if (err != CL_SUCCESS)
    throw std::runtime_error(who + ": " + what + " failed (OpenCL error " + std::to_string(err) + ").");
}

/// Exact OpenCL C literal of a double (hexadecimal floating constant).
inline std::string Literal(const double value, const std::string& who) {
  if (!IsFinite(value))
    throw std::runtime_error(who + ": non-finite solver constant cannot be compiled into the kernels.");
  char buf[64];
  std::snprintf(buf, sizeof(buf), "(%a)", value);
  return buf;
}

/// `{ a, b, ... }` initialiser list of `n` values.
template<typename Value, typename Format>
std::string List(const Value* values, const int n, Format&& format) {
  std::string out = "{ ";
  for (int k = 0; k < n; ++k) {
    out += format(values[k]);
    if (k + 1 < n)
      out += ", ";
  }
  return out + " }";
}

/**
 * @brief `#define` block that specialises `kernels.cl` for one solver.
 *
 * See the file comment; the macro names are the contract with the kernels.
 */
template<typename LatticeType, typename Scalar>
std::string ProgramDefines(
  const FluidData<Scalar>& fluid,
  const MeshData<Scalar, LatticeType::Dim()>& mesh,
  const ControlData<Scalar>& control,
  const std::string& who
) {
  constexpr int kQ = static_cast<int>(LatticeType::Speeds());
  const auto stencil = BuildDirectionStencil<LatticeType>(mesh.size[1], control.U(0), control.U(1), control.idw);

  auto real = [&who](const double v) { return Literal(v, who); };
  auto integer = [](const int v) { return std::to_string(v); };

  int cx[kQ];
  int cy[kQ];
  for (int q = 0; q < kQ; ++q) {
    cx[q] = static_cast<int>(LatticeType::Velocity(q, 0));
    cy[q] = static_cast<int>(LatticeType::Velocity(q, 1));
  }
  double w[kQ * 4];
  for (int q = 0; q < kQ; ++q)
    for (int k = 0; k < 4; ++k)
      w[q * 4 + k] = static_cast<double>(stencil.w[q][k]);
  double cix[kQ];
  double ciy[kQ];
  for (int q = 0; q < kQ; ++q) {
    cix[q] = static_cast<double>(stencil.cix[q]);
    ciy[q] = static_cast<double>(stencil.ciy[q]);
  }

  std::ostringstream d;
  d << "#define LBM_NX " << mesh.size[0] << "\n"
    << "#define LBM_NY " << mesh.size[1] << "\n"
    << "#define LBM_N " << mesh.size[0] * mesh.size[1] << "\n"
    << "#define LBM_Q " << kQ << "\n"
    << "#define LBM_CX " << List(cx, kQ, integer) << "\n"
    << "#define LBM_CY " << List(cy, kQ, integer) << "\n"
    << "#define LBM_CS2 " << real(LatticeType::Cs2) << "\n"
    << "#define LBM_USHIFT_X " << real(control.U(0)) << "\n"
    << "#define LBM_USHIFT_Y " << real(control.U(1)) << "\n"
    << "#define LBM_CV " << real(fluid.specificHeatCv) << "\n"
    << "#define LBM_CP " << real(fluid.specificHeatCp) << "\n"
    << "#define LBM_INV_CV " << real(Scalar{ 1 } / fluid.specificHeatCv) << "\n"
    << "#define LBM_RGAS " << real(fluid.constant) << "\n"
    << "#define LBM_VISCOSITY " << real(fluid.viscosity) << "\n"
    << "#define LBM_CONDUCTIVITY " << real(fluid.conductivity) << "\n"
    << "#define LBM_DENSITY_L " << real(fluid.densityL) << "\n"
    << "#define LBM_PRESSURE_L " << real(fluid.pressureL) << "\n"
    << "#define LBM_DENSITY_R " << real(fluid.densityR) << "\n"
    << "#define LBM_PRESSURE_R " << real(fluid.pressureR) << "\n"
    << "#define LBM_ST_OX " << List(stencil.ox, kQ, integer) << "\n"
    << "#define LBM_ST_OY0 " << List(stencil.oy0, kQ, integer) << "\n"
    << "#define LBM_ST_OY1 " << List(stencil.oy1, kQ, integer) << "\n"
    << "#define LBM_ST_NEAREST " << List(stencil.nearest, kQ, integer) << "\n"
    << "#define LBM_ST_W " << List(w, kQ * 4, real) << "\n"
    << "#define LBM_ST_CIX " << List(cix, kQ, real) << "\n"
    << "#define LBM_ST_CIY " << List(ciy, kQ, real) << "\n";
  return d.str();
}

/**
 * @brief First device of exactly `type` on any platform.
 *
 * Never a cross-type fallback: a CPU-targeted solver must not end up on a GPU
 * (or the reverse) without a single line of output saying so
 * (docs/container.md §6).
 */
inline cl::Device SelectDevice(const cl_device_type type, const std::string& who, const std::string& missing) {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.empty())
    throw std::runtime_error(who + ": no OpenCL platform available.");
  for (auto& platform : platforms) {
    std::vector<cl::Device> devices;
    platform.getDevices(type, &devices);
    if (!devices.empty())
      return devices.front();
  }
  throw std::runtime_error(who + ": " + missing);
}

/**
 * @brief Text of the shared `lbmini/Lbm/OpenCl/kernels.cl`.
 *
 * The kernels are JIT-compiled per solver, so the source is read at run time,
 * relative to the working directory (source tree, build directory one or two
 * levels below it) or from the container install prefix.
 */
inline std::string LoadKernelSource(const std::string& who) {
  const char* candidates[] = {
    "lbmini/Lbm/OpenCl/kernels.cl",
    "../lbmini/Lbm/OpenCl/kernels.cl",
    "../../lbmini/Lbm/OpenCl/kernels.cl",
    "/opt/lbm-miniapp/lbmini/Lbm/OpenCl/kernels.cl"
  };
  for (const char* path : candidates) {
    std::ifstream file(path);
    if (file.is_open())
      return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  }
  throw std::runtime_error(
    who + ": cannot open lbmini/Lbm/OpenCl/kernels.cl "
    "(the path is relative to the working directory)."
  );
}

/**
 * @brief Compiles `defines + source` for `device`.
 *
 * Setting `LBMINI_OPENCL_BUILD_LOG=1` prints the build log (e.g. the ptxas
 * register report requested with `-cl-nv-verbose`) to `stderr`;
 * `LBMINI_OPENCL_BUILD_OPTIONS` replaces the default build options, for
 * tuning experiments.
 */
inline cl::Program BuildProgram(
  const cl::Context& context,
  const cl::Device& device,
  const std::string& source,
  const std::string& defaultOptions,
  const std::string& who
) {
  if (!device.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>())
    throw std::runtime_error(who + ": the OpenCL device has no double-precision support (cl_khr_fp64).");

  const char* optionsEnv = std::getenv("LBMINI_OPENCL_BUILD_OPTIONS");
  const std::string options = optionsEnv ? optionsEnv : defaultOptions;

  cl::Program program(context, cl::Program::Sources{ { source.c_str(), source.length() } });
  const cl_int err = program.build({ device }, options.c_str());
  const std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
  if (err != CL_SUCCESS)
    throw std::runtime_error(who + ": kernel build failed (OpenCL error " + std::to_string(err) + "):\n" + log);
  const char* logEnv = std::getenv("LBMINI_OPENCL_BUILD_LOG");
  if (logEnv && *logEnv && *logEnv != '0')
    std::cerr << who << ": build options \"" << options << "\"\n" << log << std::endl;
  return program;
}
} // namespace lbmini::opencl::detail

#endif // LBMINI_OPENCL_PROGRAM_HPP_
