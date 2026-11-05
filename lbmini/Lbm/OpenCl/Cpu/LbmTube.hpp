#ifndef LBMINI_OPENCL_CPU_LBMTUBE_HPP_
#define LBMINI_OPENCL_CPU_LBMTUBE_HPP_

#include <memory>
#include <stdexcept>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/OpenCl/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenCl/Engine.hpp"

namespace lbmini::opencl::cpu {
// POCL reads POCL_MAX_PTHREAD_COUNT and the process affinity is chosen once per
// process, in `lbmini::Executor`, before any solver exists: the ICD only parses
// those options when the platform is first queried, which happens in the
// constructor below. Doing it from here would reconfigure a process that this
// solver does not own, and would race with any other thread reading the
// environment.

/**
 * @brief OpenCL variant of the compressible LBM tube solver for CPUs.
 *
 * Numerically identical (relative 1e-9) to `lbmini::plain::LbmTube`. The
 * device state, the kernels and the time loop are shared with the GPU variant
 * (`Lbm/OpenCl/Engine.hpp`, `Lbm/OpenCl/kernels.cl`); this class selects an
 * OpenCL CPU device — only `CL_DEVICE_TYPE_CPU`, never a cross-type fallback
 * (docs/container.md §6) — and the CPU launch shape.
 *
 * ### Performance design
 *
 * - **One fused, JIT-specialised kernel per step.** Streaming (pull, through
 *   the per-direction IDW stencil), macroscopics, Newton-Raphson closure and
 *   collision in one pass that reads `f`/`g` once and writes them once. Mesh
 *   extents, fluid constants and the whole stencil are compile-time constants
 *   of the program built for this instance.
 * - **Ping-pong post-collision state**, no copies, no per-cell tables; the
 *   macroscopic fields are written only on the last step of a `Run()`.
 * - **No host synchronisation inside a `Run()`**: all steps are enqueued back
 *   to back on an in-order queue and followed by one `finish()`.
 * - **CPU launch shape.** A 2-D NDRange (`j` fastest) with whole columns of
 *   the tube as work-groups: pocl's work-group vectoriser runs the work-items
 *   of a column in SIMD lanes with unit-stride loads and stores, the index
 *   arithmetic needs no division, and each work-group touches a contiguous
 *   slab of every population array.
 */
template<typename Scalar, typename LatticeType>
class LbmTube : public ILbmTube<Scalar, LatticeType> {
public:
  using Index = Eigen::Index;
  template<typename Type, Index NumIndices>
  using Tensor = Eigen::Tensor<Type, NumIndices, Eigen::RowMajor>;

  static constexpr Index kDim_ = LatticeType::Dim();
  static constexpr Index kQ_ = LatticeType::Speeds();

  LbmTube(const FluidData<Scalar>& fluid, const MeshData<Scalar, kDim_>& mesh, const ControlData<Scalar>& control, const PerformanceData& performance) {
    // Checked before any device object exists: every buffer size and every
    // flat index is derived from these extents, and the kernels index them
    // with no bounds check at all.
    lbmini::ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "opencl::cpu::LbmTube");

    // This class is the CPU instantiation of the OpenCL backend: its launch
    // shape and every performance figure derived from it only describe a run
    // on an OpenCL CPU device. Accepting another target here would report GPU
    // numbers measured on the host (or the reverse), so the mismatch is
    // rejected instead of being silently reinterpreted.
    if (performance.target != lbmini::TargetEnum::CPU)
      throw std::runtime_error(
        "opencl::cpu::LbmTube: this solver only supports the CPU target; "
        "use lbmini::opencl::gpu::LbmTube for the GPU target."
      );

    nx_ = mesh.nx();
    ny_ = mesh.ny();

    detail::LaunchConfig launch;
    launch.twoD = true;
    launch.local0 = static_cast<std::size_t>(ny_);
    engine_ = std::make_unique<Engine>(
      fluid,
      mesh,
      control,
      CL_DEVICE_TYPE_CPU,
      "no OpenCL CPU device available on any platform. "
      "Install an OpenCL CPU runtime ICD (e.g. pocl or the Intel OpenCL CPU runtime) "
      "and make sure its .icd file is visible under /etc/OpenCL/vendors.",
      launch,
      "opencl::cpu::LbmTube"
    );
  }

  // The solver owns the OpenCL context, queue and buffers through `engine_`;
  // solvers are heavy, unique objects held behind `ILbmTubeNd*`, so copy and
  // move are deleted uniformly across every backend. See
  // docs/memory-safety.md §5.2.
  LbmTube(const LbmTube&) = delete;

  LbmTube& operator=(const LbmTube&) = delete;

  LbmTube(LbmTube&&) = delete;

  LbmTube& operator=(LbmTube&&) = delete;

  ~LbmTube() override = default;

  Tensor<Scalar, kDim_> P() const override {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    engine_->CopyP(out.data());
    return out;
  }

  Tensor<Scalar, kDim_> Rho() const override {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    engine_->CopyRho(out.data());
    return out;
  }

  Tensor<Scalar, kDim_> T() const override {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    engine_->CopyT(out.data());
    return out;
  }

  Tensor<Scalar, kDim_ + 1> U() const override {
    Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
    engine_->CopyU(out.data());
    return out;
  }

  void Init() override { engine_->Init(); }

  void Step(const bool save) override { Run(1, save); }

  void Run(const Index steps, bool /*save*/) override { engine_->Run(steps); }

private:
  using Engine = detail::Engine<Scalar, LatticeType>;

  Index nx_ = 0;
  Index ny_ = 0;
  std::unique_ptr<Engine> engine_;
};
} // namespace lbmini::opencl::cpu

#endif
