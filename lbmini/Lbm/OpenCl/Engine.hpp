#ifndef LBMINI_OPENCL_ENGINE_HPP_
#define LBMINI_OPENCL_ENGINE_HPP_

/**
 * @file Engine.hpp
 * @brief Device state and time loop shared by `opencl::cpu::LbmTube` and
 *        `opencl::gpu::LbmTube`.
 *
 * ### Design
 *
 * - **State = post-collision populations.** `f`, `g` live in two ping-pong
 *   SoA buffers each (`f[idc * N + cell]`, `cell = i * ny + j`); a step reads
 *   one set and writes the other, the roles alternate by binding the other
 *   pre-built kernel object — nothing is ever copied. The Newton warm start is
 *   `xi[d * N + cell]`, only the two components that are read.
 * - **One fused kernel per step** (`kernels.cl`): pull streaming through the
 *   per-direction `lbmini::DirectionStencil`, macroscopics, equilibria,
 *   Newton-Raphson closure and collision in registers. No per-cell streaming
 *   tables exist.
 * - **One kernel for `Init()`**: Sod state, seeding, macroscopic recovery and
 *   the first collision on the device; nothing is uploaded from the host.
 * - **Macroscopic fields only on the last step of a `Run()`**
 *   (`stepStoreKernel`), the only point a getter can observe them. All steps
 *   of a `Run()` are enqueued without any host synchronisation and followed by
 *   a single `finish()`; the getters read back on demand.
 * - **JIT specialisation** per solver instance (`Program.hpp`).
 */

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>
#include "Lbm/OpenCl/Program.hpp"

namespace lbmini::opencl::detail {
/// Launch shape and build options chosen by the CPU/GPU front end.
struct LaunchConfig {
  /// 2-D NDRange (`j` fastest) instead of a 1-D range over the flat cell index.
  bool twoD = false;
  /// Work-group size along dimension 0 (`0`: chosen by the runtime).
  std::size_t local0 = 0;
  /// Extra `#define`s / attributes put in front of the kernel source.
  std::string defines;
  /// OpenCL build options (never a relaxed-math option).
  std::string options;
};

template<typename Scalar, typename LatticeType>
class Engine {
public:
  using Index = Eigen::Index;

  static constexpr Index kDim_ = LatticeType::Dim();
  static constexpr Index kQ_ = LatticeType::Speeds();

  static_assert(std::is_same_v<Scalar, double>, "the OpenCL kernels are written for double precision");
  static_assert(kDim_ == 2 && kQ_ == 9, "the OpenCL kernels implement the D2Q9 lattice");

  Engine(
    const FluidData<Scalar>& fluid,
    const MeshData<Scalar, kDim_>& mesh,
    const ControlData<Scalar>& control,
    const cl_device_type deviceType,
    const std::string& missingDevice,
    const LaunchConfig& launch,
    const std::string& who
  )
    : who_(who),
      launch_(launch),
      nx_(mesh.size[0]),
      ny_(mesh.size[1]),
      N_(nx_ * ny_) {
    device_ = SelectDevice(deviceType, who_, missingDevice);
    context_ = cl::Context(device_);
    queue_ = cl::CommandQueue(context_, device_);

    const std::string source = launch_.defines
      + (launch_.twoD ? "#define LBM_2D 1\n" : "")
      + ProgramDefines<LatticeType>(fluid, mesh, control, who_)
      + LoadKernelSource(who_);
    program_ = BuildProgram(context_, device_, source, launch_.options, who_);

    const std::size_t dist = static_cast<std::size_t>(kQ_ * N_) * sizeof(Scalar);
    const std::size_t field = static_cast<std::size_t>(N_) * sizeof(Scalar);
    cl_int err = CL_SUCCESS;
    for (int s = 0; s < 2; ++s) {
      f_[s] = cl::Buffer(context_, CL_MEM_READ_WRITE, dist, nullptr, &err);
      Check(err, "clCreateBuffer(f)", who_);
      g_[s] = cl::Buffer(context_, CL_MEM_READ_WRITE, dist, nullptr, &err);
      Check(err, "clCreateBuffer(g)", who_);
    }
    xi_ = cl::Buffer(context_, CL_MEM_READ_WRITE, kDim_ * field, nullptr, &err);
    Check(err, "clCreateBuffer(xi)", who_);
    rho_ = cl::Buffer(context_, CL_MEM_READ_WRITE, field, nullptr, &err);
    Check(err, "clCreateBuffer(rho)", who_);
    p_ = cl::Buffer(context_, CL_MEM_READ_WRITE, field, nullptr, &err);
    Check(err, "clCreateBuffer(p)", who_);
    tem_ = cl::Buffer(context_, CL_MEM_READ_WRITE, field, nullptr, &err);
    Check(err, "clCreateBuffer(T)", who_);
    u_ = cl::Buffer(context_, CL_MEM_READ_WRITE, kDim_ * field, nullptr, &err);
    Check(err, "clCreateBuffer(u)", who_);

    initKernel_ = MakeKernel("initKernel");
    SetArgs(initKernel_, { &f_[0], &g_[0], &xi_, &rho_, &p_, &tem_, &u_ });
    // [parity][store]: parity 0 reads set 0 and writes set 1.
    for (int parity = 0; parity < 2; ++parity) {
      for (int store = 0; store < 2; ++store) {
        cl::Kernel& k = stepKernel_[parity][store] = MakeKernel(store ? "stepStoreKernel" : "stepKernel");
        SetArgs(k, { &f_[parity], &g_[parity], &f_[1 - parity], &g_[1 - parity], &xi_, &rho_, &p_, &tem_, &u_ });
      }
    }

    // Global range rounded up to whole work-groups; the kernels skip padding.
    // A work-group the kernel cannot run with is left to the runtime instead.
    const std::size_t extent0 = static_cast<std::size_t>(launch_.twoD ? ny_ : N_);
    std::size_t l0 = launch_.local0;
    std::size_t maxLocal = 0;
    Check(
      stepKernel_[0][0].getWorkGroupInfo(device_, CL_KERNEL_WORK_GROUP_SIZE, &maxLocal),
      "clGetKernelWorkGroupInfo",
      who_
    );
    if (l0 > maxLocal)
      l0 = 0;
    const std::size_t g0 = l0 ? (extent0 + l0 - 1) / l0 * l0 : extent0;
    if (launch_.twoD) {
      global_ = cl::NDRange(g0, static_cast<std::size_t>(nx_));
      local_ = l0 ? cl::NDRange(l0, 1) : cl::NullRange;
    } else {
      global_ = cl::NDRange(g0);
      local_ = l0 ? cl::NDRange(l0) : cl::NullRange;
    }
  }

  Engine(const Engine&) = delete;

  Engine& operator=(const Engine&) = delete;

  Engine(Engine&&) = delete;

  Engine& operator=(Engine&&) = delete;

  ~Engine() {
    // Nothing may still be reading or writing a buffer when it is released.
    if (queue_())
      queue_.finish();
  }

  /// Initial condition + first collision; restarts the ping-pong at set 0.
  void Init() {
    Enqueue(initKernel_, "initKernel");
    Check(queue_.finish(), "clFinish(init)", who_);
    parity_ = 0;
  }

  /// `steps` fused steps, one host synchronisation at the end.
  void Run(const Index steps) {
    if (steps <= 0)
      return;
    for (Index s = 0; s < steps; ++s) {
      const int store = (s + 1 == steps) ? 1 : 0;
      Enqueue(stepKernel_[parity_][store], "stepKernel");
      parity_ = 1 - parity_;
    }
    Check(queue_.finish(), "clFinish(run)", who_);
  }

  void CopyRho(Scalar* dst) const { Read(rho_, dst, N_); }

  void CopyP(Scalar* dst) const { Read(p_, dst, N_); }

  void CopyT(Scalar* dst) const { Read(tem_, dst, N_); }

  /// Velocity in the AoS order `(i, j, d)` of the getter `Tensor`.
  void CopyU(Scalar* dst) const {
    std::vector<Scalar> soa(static_cast<std::size_t>(kDim_ * N_));
    Read(u_, soa.data(), kDim_ * N_);
    for (Index cell = 0; cell < N_; ++cell)
      for (Index d = 0; d < kDim_; ++d)
        dst[cell * kDim_ + d] = soa[d * N_ + cell];
  }

private:
  // TEMPORARY tuning hook (removed before submission).
  static LaunchConfig TuneLaunch(LaunchConfig l) {
    if (const char* v = std::getenv("LBMINI_OPENCL_2D"))
      l.twoD = (*v == '1');
    if (const char* v = std::getenv("LBMINI_OPENCL_LOCAL"))
      l.local0 = static_cast<std::size_t>(std::atol(v));
    if (const char* v = std::getenv("LBMINI_OPENCL_DEFINES"))
      l.defines += std::string(v) + "\n";
    return l;
  }

  cl::Kernel MakeKernel(const char* name) {
    cl_int err = CL_SUCCESS;
    cl::Kernel kernel(program_, name, &err);
    Check(err, "clCreateKernel", who_);
    return kernel;
  }

  void SetArgs(cl::Kernel& kernel, std::initializer_list<const cl::Buffer*> buffers) {
    cl_uint arg = 0;
    for (const cl::Buffer* buffer : buffers)
      Check(kernel.setArg(arg++, *buffer), "clSetKernelArg", who_);
  }

  void Enqueue(const cl::Kernel& kernel, const char* what) {
    Check(queue_.enqueueNDRangeKernel(kernel, cl::NullRange, global_, local_), what, who_);
  }

  void Read(const cl::Buffer& buffer, Scalar* dst, const Index count) const {
    Check(
      queue_.enqueueReadBuffer(buffer, CL_TRUE, 0, static_cast<std::size_t>(count) * sizeof(Scalar), dst),
      "clEnqueueReadBuffer",
      who_
    );
  }

  const std::string who_;
  const LaunchConfig launch_;
  const Index nx_;
  const Index ny_;
  const Index N_;

  cl::Device device_;
  cl::Context context_;
  cl::CommandQueue queue_;
  cl::Program program_;

  cl::Buffer f_[2];
  cl::Buffer g_[2];
  cl::Buffer xi_;
  cl::Buffer rho_;
  cl::Buffer p_;
  cl::Buffer tem_;
  cl::Buffer u_;

  cl::Kernel initKernel_;
  cl::Kernel stepKernel_[2][2];

  cl::NDRange global_;
  cl::NDRange local_;
  int parity_ = 0;
};
} // namespace lbmini::opencl::detail

#endif // LBMINI_OPENCL_ENGINE_HPP_
