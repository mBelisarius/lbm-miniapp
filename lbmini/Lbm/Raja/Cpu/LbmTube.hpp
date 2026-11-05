#ifndef LBMINI_RAJA_CPU_LBMTUBE_HPP_
#define LBMINI_RAJA_CPU_LBMTUBE_HPP_

#include <cstddef>
#include <cstdint>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <RAJA/RAJA.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/HostKernel.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/Simd.hpp"

namespace lbmini::raja::cpu {
/**
 * @brief RAJA multi-core, SIMD-vectorised variant of `lbmini::plain::LbmTube`.
 *
 * Numerically equivalent to the plain reference (same operations in the same
 * order per cell; only `exp` differs in the last bit, see `Lbm/Simd.hpp`), but
 * organised around the single fused vector kernel of `Lbm/HostKernel.hpp`,
 * which this class drives with `RAJA::forall`. It is the host-only sibling of
 * the portable per-cell template `lbmini::raja::LbmTube` (which the GPU target
 * keeps using): RAJA only supplies the loop construct here, the physics and the
 * data layout are the ones shared by every CPU backend.
 *
 * ### State and data layout
 *
 * Structure of arrays, **x-fastest**, with `cell = j * pitch + i` (`i` = x is
 * the contiguous direction, `pitch` is `nx` rounded up to a cache line, see
 * `host::Geometry`):
 *
 * | buffer        | size      | layout                       | role                               |
 * | :------------ | :-------- | :--------------------------- | :--------------------------------- |
 * | `fA_/fB_`     | `Q * pl`  | `f[idc * plane + cell]`      | post-collision density (ping-pong) |
 * | `gA_/gB_`     | `Q * pl`  | `g[idc * plane + cell]`      | post-collision energy (ping-pong)  |
 * | `xi_`         | `D * pl`  | `xi[d * plane + cell]`       | Newton-Raphson warm start          |
 * | `rho_/p_/tem_`| `plane`   | `x[cell]`                    | macroscopic output                 |
 * | `u_`          | `D * pl`  | `u[d * plane + cell]`        | macroscopic output                 |
 *
 * The solver state is the set of *post*-collision populations: `Init()` seeds
 * the equilibria, recovers the macroscopic fields (what the getters see after
 * `Init()`) and already runs the first collision; every time step then
 *
 *  1. pulls the 9 directions of each cell from their four inverse-distance
 *     source cells of the previous post-collision state (unaligned vector loads),
 *  2. reduces them to `rho`, `u`, `T`, `p`,
 *  3. evaluates the equilibria, the Newton-Raphson energy closure and the
 *     BGK / Knudsen collision for `W` cells at once, and
 *  4. writes the new post-collision populations into the other ping-pong
 *     buffer (the pointers are swapped, never copied).
 *
 * Per cell and step that is ~320 B of memory traffic against ~850 flops and 8
 * exponentials (the per-axis Newton closure of `HostKernel.hpp`); the macroscopic fields are written only by the last step of a
 * `Run()`, the only point where a getter can observe them.
 *
 * ### Threading
 *
 *  - `Init()` and each time step are one `RAJA::forall` over the *row
 *    segments* (`host::Partition`), whose body is a `[=]` lambda capturing
 *    only plain data (the `host::Params`, the geometry, the partition and raw
 *    pointers). The step policy is `RAJA::omp_parallel_for_dynamic_exec<1>`,
 *    i.e. `#pragma omp parallel for schedule(dynamic, 1)`: with a static
 *    schedule one slow core (a hybrid CPU's E-core, an SMT sibling, a
 *    time-sliced vCPU) idles the other threads at the end of every step, and
 *    the fine-grained segments make dynamic scheduling cheap. The fork/join of
 *    the parallel region is the only synchronisation per step. Without OpenMP
 *    in RAJA both policies degrade to `RAJA::seq_exec`.
 *  - The unit of work is a run of vector blocks of one row, not a row: the work
 *    is balanced to a fraction of a block even for `ny = 1`, where a row
 *    partition leaves most cores idle.
 *  - `Init()` uses `RAJA::omp_parallel_for_static_exec<>` over the same
 *    segments, so the distributions are first touched by one thread per
 *    contiguous run of segments (NUMA locality for a static consumer).
 *  - The thread count the partition is sized for is `omp_get_max_threads()`
 *    at construction; the count (`cores` from `config.yaml`, or the
 *    performance-core count on hybrid Intel hosts when `cores == 0`) and the
 *    `OMP_PLACES=cores` / `OMP_PROC_BIND=close` affinity are process-global
 *    settings owned by `lbmini::Executor`; this class only consumes them.
 */
template<typename Scalar, typename LatticeType>
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
    const PerformanceData& performance
  );

  // The solver caches raw pointers into its own buffers (`fCur_`/`fAlt_`,
  // `gCur_`/`gAlt_`); a copy would leave them dangling into the source's
  // buffers, and a second destruction would free the same storage twice.
  // Solvers are heavy, unique objects held behind `ILbmTubeNd*`, so the
  // operations are deleted rather than implemented, uniformly across every
  // backend. See docs/memory-safety.md §5.2.
  LbmTube(const LbmTube&) = delete;

  LbmTube& operator=(const LbmTube&) = delete;

  LbmTube(LbmTube&&) = delete;

  LbmTube& operator=(LbmTube&&) = delete;

  /**
   * @name Macroscopic field getters
   * @brief Copy-out accessors for pressure / density / temperature / velocity.
   *
   * They return the fields of the last `Init()` or of the last step of the
   * last `Run()`, transposed from the x-fastest storage into the row-major
   * `(nx, ny)` tensors of the interface.
   * @{
   */
  [[nodiscard]] Tensor<Scalar, kDim_> P() const override;

  [[nodiscard]] Tensor<Scalar, kDim_> Rho() const override;

  [[nodiscard]] Tensor<Scalar, kDim_> T() const override;

  [[nodiscard]] Tensor<Scalar, kDim_ + 1> U() const override;

  /** @} */

  void Init() override;

  void Step(bool save) override;

  void Run(Index steps, bool save) override;

private:
  static constexpr int kLanes = simd::kLanes;
  using Kernel = host::Kernel<Scalar, LatticeType, kLanes>;
  using Fields = host::Fields<Scalar>;

#if defined(RAJA_ENABLE_OPENMP)
  using InitPolicy = RAJA::omp_parallel_for_static_exec<>;
  using StepPolicy = RAJA::omp_parallel_for_dynamic_exec<1>;
#else
  using InitPolicy = RAJA::seq_exec;
  using StepPolicy = RAJA::seq_exec;
#endif

  static_assert(kDim_ == 2, "raja::cpu::LbmTube implements the 2-D closure");

  /// One time step: a `RAJA::forall` over the row segments.
  ///
  /// @tparam kStoreMacro write `rho`, `p`, `T`, `u` (last step of a `Run()`).
  template<bool kStoreMacro>
  void step(const Fields& fields) const;

  [[nodiscard]] Tensor<Scalar, kDim_> exportField(const lbmini::DeviceBuffer<Scalar>& field) const;

  const FluidData<Scalar> kFluid_;
  const MeshData<Scalar, kDim_> kMesh_;
  const ControlData<Scalar> kControl_;
  const PerformanceData kPerformance_;

  Index nx_;
  Index ny_;

  host::Geometry geo_;
  host::Params<Scalar> params_;
  host::Partition part_;

  FirstTouchBuffer<Scalar> fA_; // Q * plane
  FirstTouchBuffer<Scalar> fB_; // swap partner of fA_
  FirstTouchBuffer<Scalar> gA_; // Q * plane
  FirstTouchBuffer<Scalar> gB_; // swap partner of gA_
  FirstTouchBuffer<Scalar> xi_; // D * plane (Newton warm start)

  lbmini::DeviceBuffer<Scalar> rho_; // plane
  lbmini::DeviceBuffer<Scalar> p_;   // plane
  lbmini::DeviceBuffer<Scalar> tem_; // plane
  lbmini::DeviceBuffer<Scalar> u_;   // D * plane

  // Ping-pong roles: `fCur_`/`gCur_` hold the current post-collision state.
  Scalar* fCur_ = nullptr;
  Scalar* fAlt_ = nullptr;
  Scalar* gCur_ = nullptr;
  Scalar* gAlt_ = nullptr;
};

template<typename Scalar, typename LatticeType>
LbmTube<Scalar, LatticeType>::LbmTube(
  const FluidData<Scalar>& fluid,
  const MeshData<Scalar, kDim_>& mesh,
  const ControlData<Scalar>& control,
  const PerformanceData& performance
)
  : kFluid_(fluid), kMesh_(mesh), kControl_(control), kPerformance_(performance) {
  // Everything below derives from these two extents, so they are checked
  // before the first allocation: a zero extent yields an empty buffer the
  // kernels still index, and the flat indices must not overflow.
  ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "raja::cpu::LbmTube");

  nx_ = mesh.size[0];
  ny_ = mesh.size[1];
  geo_ = host::MakeGeometry(nx_, ny_);
  params_ = host::MakeParams<Scalar, LatticeType>(fluid, mesh, control);
#ifdef _OPENMP
  const int threads = omp_get_max_threads();
#else
  const int threads = 1;
#endif
  part_ = host::MakePartition(geo_.pitch, geo_.ny, kLanes, threads);

  // Distributions and warm start are first written by `Init()` (see
  // `FirstTouchAllocator`); the macroscopic fields are zero until then.
  const auto plane = static_cast<std::size_t>(geo_.plane);
  fA_.resize(kQ_ * plane);
  fB_.resize(kQ_ * plane);
  gA_.resize(kQ_ * plane);
  gB_.resize(kQ_ * plane);
  xi_.resize(kDim_ * plane);

  rho_.assign(plane, Scalar{ 0 });
  p_.assign(plane, Scalar{ 0 });
  tem_.assign(plane, Scalar{ 0 });
  u_.assign(kDim_ * plane, Scalar{ 0 });

  fCur_ = fA_.data();
  fAlt_ = fB_.data();
  gCur_ = gA_.data();
  gAlt_ = gB_.data();
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::exportField(const lbmini::DeviceBuffer<Scalar>& field) const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(nx_, ny_);
  host::ExportField(field.data(), geo_, host::Index{ 0 }, out.data());
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::P() const -> Tensor<Scalar, kDim_> {
  return exportField(p_);
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::Rho() const -> Tensor<Scalar, kDim_> {
  return exportField(rho_);
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::T() const -> Tensor<Scalar, kDim_> {
  return exportField(tem_);
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::U() const -> Tensor<Scalar, kDim_ + 1> {
  // SoA d-major, x-fastest -> RowMajor [i][j][d] Eigen tensor.
  Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
  host::ExportVector(u_.data(), u_.data() + geo_.plane, geo_, host::Index{ 0 }, out.data());
  return out;
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Init() {
  // The buffers are only ever written by kernels, so a re-Init() simply
  // restarts from the initial condition, in the current ping-pong buffer.
  Fields fields;
  fields.fOut = fCur_;
  fields.gOut = gCur_;
  fields.xi = xi_.data();
  fields.rho = rho_.data();
  fields.p = p_.data();
  fields.tem = tem_.data();
  fields.u = u_.data();

  const host::Params<Scalar> params = params_;
  const host::Geometry geo = geo_;
  const host::Partition part = part_;
  const auto nx = static_cast<host::Index>(nx_);

  // Static partition of the same segments the steps use: NUMA first touch.
  RAJA::forall<InitPolicy>(
    RAJA::TypedRangeSegment<host::Index>(0, part.Segments()),
    [=](const host::Index s) {
      const host::Index i0 = part.FirstBlock(s) * kLanes;
      Kernel::InitRow(params, geo, fields, nx, i0, part.Row(s), i0, part.BlockCount(s));
    }
  );
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Step(const bool save) {
  Run(1, save);
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Run(const Index steps, bool /*save*/) {
  if (steps <= 0)
    return;

  for (Index t = 0; t < steps; ++t) {
    Fields fields;
    fields.fIn = fCur_;
    fields.gIn = gCur_;
    fields.fOut = fAlt_;
    fields.gOut = gAlt_;
    fields.xi = xi_.data();
    fields.rho = rho_.data();
    fields.p = p_.data();
    fields.tem = tem_.data();
    fields.u = u_.data();
    if (t + 1 < steps)
      step<false>(fields);
    else
      step<true>(fields);
    std::swap(fCur_, fAlt_);
    std::swap(gCur_, gAlt_);
  }
}

template<typename Scalar, typename LatticeType>
template<bool kStoreMacro>
void LbmTube<Scalar, LatticeType>::step(const Fields& fields) const {
  const host::Params<Scalar> params = params_;
  const host::Geometry geo = geo_;
  const host::Partition part = part_;
  const host::Index xHi = geo.nx - 1;

  RAJA::forall<StepPolicy>(
    RAJA::TypedRangeSegment<host::Index>(0, part.Segments()),
    [=](const host::Index s) {
      Kernel::template StepRow<kStoreMacro>(
        params, geo, fields, 0, xHi, part.Row(s), part.FirstBlock(s) * kLanes, part.BlockCount(s)
      );
    }
  );
}
} // namespace lbmini::raja::cpu

#endif // LBMINI_RAJA_CPU_LBMTUBE_HPP_
