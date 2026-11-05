#ifndef LBMINI_OPENMP_CPU_LBMTUBE_HPP_
#define LBMINI_OPENMP_CPU_LBMTUBE_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <omp.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/HostKernel.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/Simd.hpp"

namespace lbmini::openmp::cpu {
/**
 * @brief OpenMP multi-core, SIMD-vectorised variant of `lbmini::plain::LbmTube`.
 *
 * Numerically equivalent to the plain reference (same operations in the same
 * order per cell; only `exp` differs in the last bit, see `Lbm/Simd.hpp`), but
 * organised around the single fused vector kernel of `Lbm/HostKernel.hpp`,
 * which this class drives with `#pragma omp for`.
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
 *  - One `#pragma omp parallel` region per `Run()`; each step is one
 *    `#pragma omp for schedule(runtime)` over *row segments* (`host::Partition`),
 *    whose implicit barrier is the only synchronisation per step. Every thread
 *    swaps its own copy of the ping-pong pointers, so no `single` is needed.
 *    `lbmini::Executor` makes the runtime schedule `dynamic,1` unless the user
 *    sets `OMP_SCHEDULE`: with a static schedule one slow core (a hybrid CPU's
 *    E-core, an SMT sibling, a time-sliced vCPU) idles the other threads at the
 *    barrier of every step, and the fine-grained segments make dynamic cheap.
 *    Use `OMP_SCHEDULE=static` on NUMA nodes whose first-touch placement matters.
 *  - The unit of work is a run of vector blocks of one row, not a row: the work
 *    is balanced to a fraction of a block even for `ny = 1`, where a row
 *    partition leaves most cores idle.
 *  - `Init()` uses the same static partition, so the distributions are first
 *    touched by the threads that update them (NUMA locality).
 *  - The thread count (`cores` from `config.yaml`, or the performance-core
 *    count on hybrid Intel hosts when `cores == 0`) and the
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

  static_assert(kDim_ == 2, "openmp::cpu::LbmTube implements the 2-D closure");

  /// One time step over the row segments of the calling thread.
  ///
  /// Must be called by every thread of the enclosing parallel region (it
  /// contains the work-sharing `omp for` and its implicit barrier).
  ///
  /// @tparam kStoreMacro write `rho`, `p`, `T`, `u` (last step of a `Run()`).
  template<bool kStoreMacro>
  void step(const Fields& fields);

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
  ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "openmp::cpu::LbmTube");

  // Thread count and affinity (`omp_set_num_threads`, `OMP_PLACES`,
  // `OMP_PROC_BIND`) are process-global state and are configured once by
  // `lbmini::Executor`. Setting them here would reconfigure every solver that
  // is already live in the process and would race with any other thread
  // reading the environment.

  nx_ = mesh.size[0];
  ny_ = mesh.size[1];
  geo_ = host::MakeGeometry(nx_, ny_);
  params_ = host::MakeParams<Scalar, LatticeType>(fluid, mesh, control);
  part_ = host::MakePartition(geo_.pitch, geo_.ny, kLanes, omp_get_max_threads());

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

  const host::Index segments = part_.Segments();

  // Static partition of the same segments the steps use: NUMA first touch.
  #pragma omp parallel for schedule(static)
  for (host::Index s = 0; s < segments; ++s) {
    const host::Index i0 = part_.FirstBlock(s) * kLanes;
    Kernel::InitRow(params_, geo_, fields, nx_, i0, part_.Row(s), i0, part_.BlockCount(s));
  }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Step(const bool save) {
  Run(1, save);
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Run(const Index steps, bool /*save*/) {
  if (steps <= 0)
    return;

  #pragma omp parallel default(shared)
  {
    // Thread-private ping-pong roles, swapped identically by every thread:
    // the implicit barrier of each step's `omp for` is the only
    // synchronisation. The members are read here, before that first barrier,
    // and written back after the last one.
    Scalar* fIn = fCur_;
    Scalar* fOut = fAlt_;
    Scalar* gIn = gCur_;
    Scalar* gOut = gAlt_;
    for (Index t = 0; t < steps; ++t) {
      Fields fields;
      fields.fIn = fIn;
      fields.gIn = gIn;
      fields.fOut = fOut;
      fields.gOut = gOut;
      fields.xi = xi_.data();
      fields.rho = rho_.data();
      fields.p = p_.data();
      fields.tem = tem_.data();
      fields.u = u_.data();
      if (t + 1 < steps)
        step<false>(fields);
      else
        step<true>(fields);
      std::swap(fIn, fOut);
      std::swap(gIn, gOut);
    }
    #pragma omp single nowait
    {
      fCur_ = fIn;
      fAlt_ = fOut;
      gCur_ = gIn;
      gAlt_ = gOut;
    }
  }
}

template<typename Scalar, typename LatticeType>
template<bool kStoreMacro>
void LbmTube<Scalar, LatticeType>::step(const Fields& fields) {
  const host::Index segments = part_.Segments();
  const host::Index xHi = geo_.nx - 1;

  #pragma omp for schedule(runtime)
  for (host::Index s = 0; s < segments; ++s)
    Kernel::template StepRow<kStoreMacro>(
      params_, geo_, fields, 0, xHi, part_.Row(s), part_.FirstBlock(s) * kLanes, part_.BlockCount(s)
    );
}
} // namespace lbmini::openmp::cpu

#endif // LBMINI_OPENMP_CPU_LBMTUBE_HPP_
