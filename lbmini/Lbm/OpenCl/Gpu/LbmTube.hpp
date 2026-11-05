#ifndef LBMINI_OPENCL_GPU_LBMTUBE_HPP_
#define LBMINI_OPENCL_GPU_LBMTUBE_HPP_

#include <memory>
#include <stdexcept>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/OpenCl/Gpu/LatticeD2Q9.hpp"
#include "Lbm/OpenCl/Engine.hpp"

namespace lbmini::opencl::gpu {
/**
 * @brief OpenCL variant of the compressible LBM tube solver for GPUs.
 *
 * Numerically identical (relative 1e-9) to `lbmini::plain::LbmTube`. The
 * device state, the kernels and the time loop are shared with the CPU variant
 * (`Lbm/OpenCl/Engine.hpp`, `Lbm/OpenCl/kernels.cl`); this class selects an
 * OpenCL GPU device — only `CL_DEVICE_TYPE_GPU`, never a cross-type fallback
 * (docs/container.md §6) — and the GPU launch shape.
 *
 * ### Performance design
 *
 * - **One fused, JIT-specialised kernel per step.** Streaming (pull, through
 *   the per-direction IDW stencil), macroscopics, Newton-Raphson closure and
 *   collision in one pass that reads `f`/`g` once and writes them once. The
 *   closure is factorised per axis (8 exponentials per cell and step, see
 *   `AxisExp` in `kernels.cl`). Mesh extents, fluid constants and the whole
 *   stencil are compile-time constants of the program built for this instance.
 * - **Ping-pong post-collision state**, no copies, no per-cell tables; the
 *   macroscopic fields are written only on the last step of a `Run()`.
 * - **No host synchronisation inside a `Run()`**: all steps are enqueued back
 *   to back on an in-order queue and followed by one `finish()`.
 * - **GPU launch shape.** A 1-D NDRange over the flat cell index (one
 *   work-item per cell, consecutive cells in a warp: coalesced SoA accesses).
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
    lbmini::ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "opencl::gpu::LbmTube");

    // This class is the GPU instantiation of the OpenCL backend: its launch
    // shape and every performance figure derived from it only describe a run
    // on an OpenCL GPU device. Accepting another target here would report host
    // numbers as GPU ones (or the reverse), so the mismatch is rejected
    // instead of being silently reinterpreted.
    if (performance.target != lbmini::TargetEnum::GPU)
      throw std::runtime_error(
        "opencl::gpu::LbmTube: this solver only supports the GPU target; "
        "use lbmini::opencl::cpu::LbmTube for the CPU target."
      );

    nx_ = mesh.nx();
    ny_ = mesh.ny();

    detail::LaunchConfig launch;
    launch.twoD = false;
    launch.local0 = 128;
    engine_ = std::make_unique<Engine>(
      fluid,
      mesh,
      control,
      CL_DEVICE_TYPE_GPU,
      "no OpenCL GPU device available on any platform. "
      "Install a vendor OpenCL GPU ICD (e.g. the NVIDIA or Intel GPU runtime) "
      "and make sure its .icd file is visible under /etc/OpenCL/vendors.",
      launch,
      "opencl::gpu::LbmTube"
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
} // namespace lbmini::opencl::gpu

#endif
