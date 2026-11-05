#ifndef LBMINI_OPENCL_CPU_LBMTUBE_HPP_
#define LBMINI_OPENCL_CPU_LBMTUBE_HPP_

#include <memory>
#include <stdexcept>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/OpenCl/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenCl/Cpu/Engine.hpp"

namespace lbmini::opencl::cpu {
// POCL reads POCL_MAX_PTHREAD_COUNT and the process affinity is chosen once per
// process, in `lbmini::Executor`, before any solver exists: the ICD only parses
// those options when the platform is first queried, which happens in the
// constructor below. Doing it from here would reconfigure a process that this
// solver does not own, and would race with any other thread reading the
// environment.

/**
 * @brief OpenCL variant of the compressible LBM tube solver for CPUs, SIMD-vectorised.
 *
 * Numerically equivalent (relative 1e-9) to `lbmini::plain::LbmTube`: the same
 * arithmetic per cell, with the energy closure in its factorised form (two scalar
 * Newton steps from 2 exponentials per evaluation, shared divisions: see
 * `Lbm/HostKernel.hpp`), which agrees with the reference to rounding error; `exp`
 * (a vector polynomial in OpenCL C, see `kernels_cpu.cl`) and the contraction of
 * `a * b + c` into fused multiply-adds differ in the last bits as well.
 *
 * Unlike the GPU variant
 * (`Lbm/OpenCl/Engine.hpp`, per-cell `kernels.cl`), the CPU path is organised
 * around the vector kernel of `Lbm/OpenCl/kernels_cpu.cl`, which is the OpenCL C
 * image of the shared host kernel (`Lbm/HostKernel.hpp`) and is driven by
 * `Lbm/OpenCl/Cpu/Engine.hpp`. This class selects an OpenCL CPU device — only
 * `CL_DEVICE_TYPE_CPU`, never a cross-type fallback (docs/container.md §6).
 *
 * ### State and data layout
 *
 * Structure of arrays, **x-fastest**, `cell = j * pitch + i` (`i` = x is the
 * contiguous direction, `pitch` is `nx` rounded up to a multiple of 16 columns,
 * see `host::Geometry`): `f`/`g` are `f[idc * plane + cell]` in two ping-pong
 * `cl::Buffer`s each, `xi` (Newton warm start) and `u` have two planes, `rho`,
 * `p`, `T` one. Sizes, the staggered plane stride and the export into the
 * row-major `(nx, ny)` getter tensors are those of the host kernel
 * (`host::MakeGeometry`, `host::ExportField`). The state is the set of
 * *post*-collision populations: `Init()` seeds the equilibria, recovers the
 * macroscopic fields and runs the first collision on the device; every step
 * pulls the nine directions of a block of cells from their four inverse-distance
 * source cells of the previous state (unaligned vector loads from the two
 * wrapped source rows, lane-wise clamping only for the first and last block of
 * a row), evaluates macroscopics, equilibria, Newton-Raphson closure and
 * BGK/Knudsen collision for the whole block, and writes the other ping-pong set.
 *
 * ### Parallel organisation
 *
 *  - **Vectorisation is explicit.** One work-item processes a *block* of
 *    `VW = 2 x CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE` neighbouring cells of one
 *    row in OpenCL `doubleVW` vectors (8 lanes = two AVX2 registers: the fused
 *    step is latency-bound, and two independent registers keep the FMA ports
 *    busy). Branches become lane selects; the `solverOk` branch of the Newton
 *    closure is taken per lane unless every lane agrees.
 *  - **Work is dealt out dynamically at the granularity of a row segment.**
 *    The cells are cut into row segments (`host::Partition`: ~256 cells of one
 *    row, at least eight per worker thread); consecutive segments are consecutive
 *    rows of the same column chunk, so the five source rows of a segment stay in
 *    cache. Each launch is one work-group of one work-item per worker thread, and
 *    the work-items take segments from an atomic queue until it is empty: pocl's
 *    pthread device would deal out `remaining / threads` work-groups at a time,
 *    which for a few hundred short work-groups is a static schedule and idles
 *    the other threads whenever one core is slower (hybrid E-cores, SMT siblings,
 *    time-sliced vCPUs). The worker count is `POCL_MAX_PTHREAD_COUNT` (set from
 *    `cores` by `lbmini::Executor`), read back as `CL_DEVICE_MAX_COMPUTE_UNITS`.
 *  - **One launch per step, no host synchronisation in a `Run()`**: every step is
 *    enqueued on the in-order queue, ping-pong by binding the other pre-built
 *    kernel object, and a single `finish()` ends the `Run()` (one more per 4096
 *    steps bounds the commands in flight). The macroscopic fields are written
 *    only by the last step (a kernel argument).
 *  - **The one-time JIT cost is paid in the constructor**: the program is built
 *    per instance (mesh, fluid and stencil are compile-time constants) and every
 *    kernel is launched once on a single segment, because pocl generates the
 *    native work-group function on the first launch.
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

    engine_ = std::make_unique<Engine>(
      fluid,
      mesh,
      control,
      "no OpenCL CPU device available on any platform. "
      "Install an OpenCL CPU runtime ICD (e.g. pocl or the Intel OpenCL CPU runtime) "
      "and make sure its .icd file is visible under /etc/OpenCL/vendors.",
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
  using Engine = detail::CpuEngine<Scalar, LatticeType>;

  Index nx_ = 0;
  Index ny_ = 0;
  std::unique_ptr<Engine> engine_;
};
} // namespace lbmini::opencl::cpu

#endif
