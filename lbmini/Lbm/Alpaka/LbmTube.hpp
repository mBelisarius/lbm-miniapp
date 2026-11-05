#ifndef LBMINI_ALPAKA_LBMTUBE_HPP_
#define LBMINI_ALPAKA_LBMTUBE_HPP_

#include <memory>

#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/Alpaka/LatticeD2Q9.hpp"

namespace lbmini::alpaka {
namespace detail {
/**
 * @brief The alpaka-compiled part of the solver, behind a plain-C++ interface.
 *
 * Declared here, defined in `LbmTubeImpl.hpp` and explicitly instantiated in
 * `Cpu/LbmTube.cpp` / `Gpu/LbmTube.cpp`. Its interface deliberately carries
 * only scalars and raw host pointers - no Eigen object is ever created or
 * destroyed on the alpaka side. The GPU flavour is compiled by the CUDA
 * compiler with different target flags (no `-march=native`), so Eigen's
 * alignment configuration, and with it the allocator behind `Eigen::Tensor`,
 * differs from the one of the code that owns the returned tensors; a tensor
 * allocated there and freed here is a heap corruption. All tensors are
 * therefore built by `LbmTube` below, in the caller's translation unit.
 */
template<typename Scalar, typename LatticeType, TargetEnum Target>
class Engine {
public:
  using Index = Eigen::Index;

  Engine(
    const FluidData<Scalar>& fluid,
    const MeshData<Scalar, LatticeType::Dim()>& mesh,
    const ControlData<Scalar>& control
  );

  Engine(const Engine&) = delete;

  Engine& operator=(const Engine&) = delete;

  Engine(Engine&&) = delete;

  Engine& operator=(Engine&&) = delete;

  ~Engine();

  void Init();

  void Run(Index steps);

  /// Copy `N` values of a scalar field into `dst` (row-major `i * ny + j`).
  void CopyRho(Scalar* dst) const;

  void CopyP(Scalar* dst) const;

  void CopyT(Scalar* dst) const;

  /// Copy the velocity into `dst` as `[cell][d]` (AoS, the Eigen RowMajor layout).
  void CopyU(Scalar* dst) const;

private:
  struct State;

  std::unique_ptr<State> state_;
};
} // namespace detail
/**
 * @brief Alpaka3 variant of the compressible LBM tube solver.
 *
 * Numerically equivalent to `lbmini::plain::LbmTube` (same equilibria, same
 * Newton-Raphson energy closure, same BGK + Knudsen-sensor collision, same IDW
 * shifted-lattice streaming), written against alpaka3 and compiled for
 * `TargetEnum::CPU` (alpaka `api::host` + `exec::cpuOmpBlocks`, i.e. OpenMP)
 * and `TargetEnum::GPU` (alpaka `api::cuda` + `exec::gpuCuda`).
 *
 * ### Why this header does not include alpaka
 *
 * alpaka requires every translation unit that includes it to belong to a
 * CMake target processed by `alpaka_finalize()`, which for the CUDA flavour
 * also turns every C++ source of that target into a CUDA source. `Executor.cpp`
 * and the test binary instantiate *every* backend, so they must not see a
 * single alpaka header. The solver is therefore a thin, header-only front end
 * over `detail::Engine`, whose implementation (`LbmTubeImpl.hpp`) is
 * explicitly instantiated in `Cpu/LbmTube.cpp` (a host-only alpaka target) and
 * in `Gpu/LbmTube.cpp` (an alpaka CUDA target). The virtual dispatch through
 * `ILbmTubeNd` already costs one indirection per call, the engine adds a
 * second one per `Run()` — never per cell or per step.
 *
 * ### Performance design (see `LbmTubeImpl.hpp` and `Cpu/LbmTubeImpl.hpp`)
 *
 * The two targets share the numerics and the public interface, but not the
 * kernel: a GPU wants one thread per cell, a CPU core wants whole SIMD vectors
 * of neighbouring cells.
 *
 * - **CPU target: the shared SIMD host kernel.** The step is `host::Kernel` of
 *   `Lbm/HostKernel.hpp`, vectorised over runs of cells of one row, on the
 *   x-fastest padded layout (`cell = j * pitch + i`). alpaka allocates the
 *   buffers and launches one block per OpenMP thread; the blocks balance
 *   the row segments dynamically through a shared counter. The getters
 *   transpose the layout into the row-major tensors of the interface.
 * - **GPU target: one fused kernel per time step.** Pull-scheme streaming,
 *   macroscopic reduction, equilibria, Newton-Raphson closure and collision run
 *   in a single pass, one thread per cell. Each step reads the post-collision
 *   `f`/`g` once and writes them once (plus the two-component Newton warm
 *   start): about half the DRAM traffic of the two-kernel scheme of the Kokkos /
 *   RAJA / CUDA backends. The distributions are SoA, direction-major
 *   (`f[idc * N + cell]`, `cell = i * ny + j`), which coalesces the accesses.
 * - **No streaming tables (both targets).** The lattice shift `U` is uniform, so
 *   the four IDW source offsets and weights of a direction are the same for
 *   every cell. They are evaluated once on the host and handed to the kernel as
 *   plain data; per cell only a clamp in `x` and a wrap in `y`
 *   remain. This removes 432 bytes/cell of table reads per step.
 * - **Macroscopic fields are only stored when observable** (both targets), i.e.
 *   on the last step of every `Run()` call: they are consumed from registers by
 *   the collision of the same pass, and no getter can run between two steps of
 *   one `Run()`.
 */
template<typename Scalar, typename LatticeType, TargetEnum Target>
class LbmTube : public ILbmTube<Scalar, LatticeType> {
public:
  using Index = Eigen::Index;

  template<typename Type, Index NumIndices>
  using Tensor = Eigen::Tensor<Type, NumIndices, Eigen::RowMajor>;

  static constexpr Index kDim_ = LatticeType::Dim();
  static constexpr Index kQ_ = LatticeType::Speeds();

  LbmTube(
    const FluidData<Scalar>& fluid,
    const MeshData<Scalar, kDim_>& mesh,
    const ControlData<Scalar>& control,
    const PerformanceData& /*performance*/
  ) {
    // Before any device is touched or any byte allocated: every buffer size and
    // every flat index derives from these extents (and the device kernels index
    // cells with `std::int32_t`).
    ValidateMeshExtents(
      mesh.size[0], mesh.size[1], kQ_, Target == TargetEnum::GPU ? "alpaka::gpu" : "alpaka::cpu"
    );
    nx_ = mesh.size[0];
    ny_ = mesh.size[1];
    engine_ = std::make_unique<Engine>(fluid, mesh, control);
  }

  // The engine owns alpaka device buffers and a queue; solvers are
  // heavy, unique objects held behind `ILbmTubeNd*`, so the operations are
  // deleted rather than implemented, uniformly across every backend.
  // See docs/memory-safety.md §5.2.
  LbmTube(const LbmTube&) = delete;

  LbmTube& operator=(const LbmTube&) = delete;

  LbmTube(LbmTube&&) = delete;

  LbmTube& operator=(LbmTube&&) = delete;

  ~LbmTube() override = default;

  /**
   * @name Macroscopic field getters
   * @brief Copy-out accessors; the engine fills the returned Eigen tensor
   *        directly (the CPU target transposes its x-fastest layout, the GPU
   *        target copies device buffers and repacks the velocity SoA -> AoS).
   * @{
   */
  [[nodiscard]] Tensor<Scalar, kDim_> P() const override {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    engine_->CopyP(out.data());
    return out;
  }

  [[nodiscard]] Tensor<Scalar, kDim_> Rho() const override {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    engine_->CopyRho(out.data());
    return out;
  }

  [[nodiscard]] Tensor<Scalar, kDim_> T() const override {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    engine_->CopyT(out.data());
    return out;
  }

  [[nodiscard]] Tensor<Scalar, kDim_ + 1> U() const override {
    Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
    engine_->CopyU(out.data());
    return out;
  }

  /** @} */

  void Init() override { engine_->Init(); }

  void Step(const bool save) override { Run(1, save); }

  void Run(const Index steps, bool /*save*/) override { engine_->Run(steps); }

private:
  using Engine = detail::Engine<Scalar, LatticeType, Target>;

  Index nx_ = 0;
  Index ny_ = 0;
  std::unique_ptr<Engine> engine_;
};

namespace gpu {
/**
 * @brief True when the alpaka CUDA flavour of this backend was compiled in.
 *
 * Without it there is no accelerator implementation at all; `Executor` uses
 * the flag to refuse a "GPU" run instead of failing at link time or, worse,
 * quietly running somewhere else.
 */
inline constexpr bool kHasDeviceBackend =
#ifdef USE_ALPAKA_GPU
  true;
#else
  false;
#endif

#ifdef USE_ALPAKA_GPU
/// Number of CUDA devices alpaka can see (0 when the driver has none).
int DeviceCount();
#endif
} // namespace gpu
} // namespace lbmini::alpaka

#endif // LBMINI_ALPAKA_LBMTUBE_HPP_
