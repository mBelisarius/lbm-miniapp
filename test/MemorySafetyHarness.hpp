#ifndef LBMINI_TEST_MEMORYSAFETYHARNESS_HPP_
#define LBMINI_TEST_MEMORYSAFETYHARNESS_HPP_

/**
 * @file MemorySafetyHarness.hpp
 * @brief Shared fixtures for the backend-agnostic memory-safety suite.
 *
 * The suite exists for one purpose: proving that **no backend corrupts
 * memory** before any of them is benchmarked. A wrong number is a bug; an
 * out-of-bounds write, a use-after-free or a double-free on a shared cluster
 * node is an incident. See `docs/memory-safety.md` for the verification
 * procedure these tests are part of (they are meant to be executed under
 * AddressSanitizer / UndefinedBehaviorSanitizer / Valgrind /
 * `compute-sanitizer`, where every latent out-of-bounds access becomes a hard
 * failure instead of silent corruption).
 *
 * Everything here is deliberately backend-neutral: a backend is described by a
 * small traits struct (see `memory_safety_test.cpp`) and driven exclusively
 * through the `lbmini::ILbmTube` contract, so a new backend joins the suite by
 * adding one type to a type list.
 */

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Utils.hpp"

#ifdef LBMINI_TEST_KOKKOS
#include <Kokkos_Core.hpp>
#endif

namespace lbmini_test {
using Scalar = double;
using Index = Eigen::Index;

/// Sod shock tube state, mirroring the derived quantities of `ReadYaml`.
inline lbmini::FluidData<Scalar> MakeFluid() {
  lbmini::FluidData<Scalar> fluid{};
  fluid.densityL = 1.0;
  fluid.pressureL = 1.0;
  fluid.densityR = 0.125;
  fluid.pressureR = 0.1;
  fluid.viscosity = 1.0e-2;
  fluid.prandtl = 0.71;
  fluid.gamma = 1.4;
  fluid.specificHeatCv = 1.0 / (fluid.gamma - 1.0);
  fluid.specificHeatCp = fluid.specificHeatCv + 1.0;
  fluid.constant = fluid.specificHeatCp - fluid.specificHeatCv;
  fluid.conductivity = fluid.specificHeatCp * fluid.viscosity / fluid.prandtl;
  return fluid;
}

inline lbmini::MeshData<Scalar, 2> MakeMesh(const Index nx, const Index ny) {
  lbmini::MeshData<Scalar, 2> mesh{};
  mesh.lenght[0] = 1.0;
  mesh.lenght[1] = 1.0;
  mesh.size[0] = nx;
  mesh.size[1] = ny;
  return mesh;
}

inline lbmini::ControlData<Scalar> MakeControl(const Scalar ux = 0.25, const Scalar uy = 0.0) {
  lbmini::ControlData<Scalar> control{};
  control.tmax = 0.2;
  control.Ux = ux;
  control.Uy = uy;
  control.Uz = 0.0;
  control.idw = std::sqrt(1.4);
  control.printStep = 1;
  return control;
}

inline lbmini::PerformanceData MakePerformance(
  const lbmini::BackendEnum backend,
  const lbmini::TargetEnum target = lbmini::TargetEnum::CPU
) {
  lbmini::PerformanceData performance{};
  performance.target = target;
  performance.backend = backend;
  // `cores == 0` means "let the backend decide"; pinning a fixed thread count
  // here would hide races that only appear with the production thread pool.
  performance.cores = 0;
  return performance;
}

/**
 * @brief One mesh geometry exercised by the sweep.
 *
 * The interesting shapes are the degenerate ones: the flat-index arithmetic of
 * every backend contains `i * ny + j`, a `y % ny` wrap and an `i < nx / 2`
 * split, so `nx == 1`, `ny == 1`, odd extents and `nx < 2` are exactly the
 * inputs that turn an index computation into an out-of-bounds access.
 */
struct MeshShape {
  Index nx;
  Index ny;

  [[nodiscard]] std::string Name() const {
    return std::to_string(nx) + "x" + std::to_string(ny);
  }

  [[nodiscard]] Index Cells() const { return nx * ny; }

  /**
   * @brief Whether the shape is a physically meaningful shock tube.
   *
   * Only these shapes get value-level assertions (finiteness, positivity).
   * The degenerate ones are still swept — they are the memory-safety payload —
   * but a 1x1 "tube" has no shock to resolve, so demanding a sane solution
   * there would test physics, not memory.
   */
  [[nodiscard]] bool IsPhysical() const { return nx >= 16 && ny >= 2; }
};

/// Shapes swept by every memory test. Cheap on purpose: the suite is meant to
/// run under sanitizers, which cost 2-20x, and under Valgrind, which costs 50x.
inline const std::vector<MeshShape>& MeshShapes() {
  static const std::vector<MeshShape> shapes = {
    { 64, 4 },   // canonical benchmark shape
    { 63, 5 },   // odd nx and odd ny
    { 32, 3 },   // odd ny
    { 128, 2 },  // minimal ny for a 2-D tube
    { 8, 8 },    // square
    { 4, 4 },    // small
    { 3, 3 },    // odd and small
    { 2, 2 },    // smallest shape with a left/right split
    { 1, 8 },    // single column: nx / 2 == 0, x-clamping degenerates
    { 8, 1 },    // single row: the y-wrap modulus is 1
    { 1, 1 },    // fully degenerate
  };
  return shapes;
}

inline constexpr Index kSweepSteps = 3;

/// Number of `Run()` steps used by the stability checks.
inline constexpr Index kStabilitySteps = 4;

/**
 * @brief Touches every element of every macroscopic field.
 *
 * Reading is the point: under AddressSanitizer / Valgrind a getter that copies
 * from a wrongly-sized buffer, from freed memory, or from a device allocation
 * that was never synchronised back to the host reports here instead of
 * silently producing garbage that only shows up as a "weird benchmark".
 * The accumulated checksum keeps the compiler from optimising the reads away.
 */
template<typename Tube>
Scalar ReadAllFields(const Tube& tube, const MeshShape& shape, const std::string& where) {
  const auto rho = tube.Rho();
  const auto p = tube.P();
  const auto tem = tube.T();
  const auto u = tube.U();

  EXPECT_EQ(rho.dimension(0), shape.nx) << where;
  EXPECT_EQ(rho.dimension(1), shape.ny) << where;
  EXPECT_EQ(p.dimension(0), shape.nx) << where;
  EXPECT_EQ(p.dimension(1), shape.ny) << where;
  EXPECT_EQ(tem.dimension(0), shape.nx) << where;
  EXPECT_EQ(tem.dimension(1), shape.ny) << where;
  EXPECT_EQ(u.dimension(0), shape.nx) << where;
  EXPECT_EQ(u.dimension(1), shape.ny) << where;
  EXPECT_EQ(u.dimension(2), Index{ 2 }) << where;

  Scalar checksum = Scalar{ 0 };
  const bool physical = shape.IsPhysical();
  for (Index i = 0; i < shape.nx; ++i) {
    for (Index j = 0; j < shape.ny; ++j) {
      const Scalar r = rho(i, j);
      const Scalar pr = p(i, j);
      const Scalar t = tem(i, j);
      checksum += r + pr + t;
      for (Index d = 0; d < 2; ++d)
        checksum += u(i, j, d);

      if (physical) {
        if (!lbmini::IsFinite(r) || r <= Scalar{ 0 }) {
          ADD_FAILURE() << where << " invalid rho " << r << " at (" << i << ", " << j << ")";
        }
        if (!lbmini::IsFinite(pr)) {
          ADD_FAILURE() << where << " non-finite p at (" << i << ", " << j << ")";
        }
        if (!lbmini::IsFinite(t)) {
          ADD_FAILURE() << where << " non-finite T at (" << i << ", " << j << ")";
        }
      }
    }
  }
  return checksum;
}

/// Value-level sanity, only meaningful for physically admissible shapes.
template<typename Tube>
void ExpectPhysical(const Tube&, const MeshShape&, const std::string&) {
  // Handled directly inside ReadAllFields to avoid redundant tensor allocations and copies.
}

#ifdef LBMINI_TEST_KOKKOS
/**
 * @brief Boots Kokkos once for the whole test binary.
 *
 * Kokkos cannot be re-initialised after `finalize()`, so the runtime must
 * outlive every solver created by any test case. Registering the environment
 * from several translation units is harmless — both hooks are idempotent.
 */
class KokkosEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    if (Kokkos::is_initialized() || Kokkos::is_finalized())
      return;
    // A Kokkos built with the CUDA backend enumerates devices inside
    // `initialize()` and throws when none answers - which happens on any host
    // without a working driver, and in every container started without
    // `--gpus all` / `--nv`. A global environment that throws aborts the whole
    // binary before the first test: one unusable accelerator would take the
    // entire suite - Plain, OpenMP, OpenCL, RAJA included - down with it, and
    // report it as a test-framework error rather than a missing device. The
    // failure is recorded instead, and the Kokkos-backed cases skip on it.
    try {
      Kokkos::initialize();
    } catch (const std::exception& error) {
      std::cerr
        << "Kokkos::initialize() failed, Kokkos-backed cases will be skipped: "
        << error.what() << std::endl;
    }
  }

  void TearDown() override {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized())
      Kokkos::finalize();
  }

  /// False when the runtime above refused to start; no solver may be created.
  static bool IsUsable() { return Kokkos::is_initialized(); }
};
#endif
} // namespace lbmini_test

#if defined(__SANITIZE_ADDRESS__) || defined(ADDRESS_SANITIZER)
extern "C" const char* __attribute__((weak)) __lsan_default_suppressions() {
  return "leak:libintelocl.so\n"
         "leak:libocl_svml_l9.so\n"
         "leak:libirng.so\n"
         "leak:libcommon_clang.so\n"
         "leak:libigc.so\n"
         "leak:libiga.so\n"
         "leak:libze_loader.so\n"
         "leak:libOpenCL.so\n"
         "leak:libOpenCL.so.1\n"
         "leak:libnvidia-opencl.so\n"
         "leak:libpocl.so\n"
         "leak:libnvomp.so\n"
         "leak:libgomp.so\n"
         "leak:libmpi.so\n"
         "leak:libopen-pal.so\n"
         "leak:libopen-rte.so\n"
         "leak:Intel::OpenCL\n"
         "leak:__ocl_svml\n"
         "leak:dErfInv\n"
         "leak:initThreadInfoTables\n"
         "leak:initThreadId\n"
         "leak:hxiInitThreadId\n";
}
#endif

#endif // LBMINI_TEST_MEMORYSAFETYHARNESS_HPP_
