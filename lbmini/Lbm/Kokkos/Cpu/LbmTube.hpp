#ifndef LBMINI_KOKKOS_CPU_LBMTUBE_HPP_
#define LBMINI_KOKKOS_CPU_LBMTUBE_HPP_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <Kokkos_Core.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/HostKernel.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/Kokkos/Runtime.hpp"
#include "Lbm/Simd.hpp"

namespace lbmini::kokkos::cpu {
namespace detail {
/// Sod initial state of the row segment `s`; the body of the first-touch `parallel_for`.
template<typename Scalar, typename LatticeType>
struct InitFunctor {
  using Kernel = host::Kernel<Scalar, LatticeType, simd::kLanes>;

  host::Params<Scalar> params;
  host::Geometry geo;
  host::Partition part;
  host::Fields<Scalar> fields;

  // Host-only on purpose: the kernel is host code, and the functor only ever
  // runs in the host execution space, even when Kokkos has a device backend.
  void operator()(const host::Index s) const {
    const host::Index i0 = part.FirstBlock(s) * simd::kLanes;
    Kernel::InitRow(params, geo, fields, geo.nx, i0, part.Row(s), i0, part.BlockCount(s));
  }
};

/// One time step of the row segment `s`; the body of the per-step `parallel_for`.
template<typename Scalar, typename LatticeType, bool kStoreMacro>
struct StepFunctor {
  using Kernel = host::Kernel<Scalar, LatticeType, simd::kLanes>;

  host::Params<Scalar> params;
  host::Geometry geo;
  host::Partition part;
  host::Fields<Scalar> fields;

  void operator()(const host::Index s) const {
    Kernel::template StepRow<kStoreMacro>(
      params, geo, fields, 0, geo.nx - 1, part.Row(s), part.FirstBlock(s) * simd::kLanes, part.BlockCount(s)
    );
  }
};
} // namespace detail

/**
 * @brief Kokkos multi-core, SIMD-vectorised variant of `lbmini::plain::LbmTube`.
 *
 * The host-space (`Kokkos::DefaultHostExecutionSpace`: OpenMP, Threads or
 * Serial) solver of the Kokkos backend; it runs on the host even when Kokkos
 * was built with a device backend, so the CPU and GPU measurements can be taken
 * from a single binary. Numerically equivalent to the plain reference (same
 * operations in the same order per cell; only `exp` differs in the last bit,
 * see `Lbm/Simd.hpp`), but organised around the single fused vector kernel of
 * `Lbm/HostKernel.hpp`, which this class drives with `Kokkos::parallel_for`.
 *
 * It is deliberately *not* an instantiation of the single-source
 * `lbmini::kokkos::LbmTube`: that template keeps the scalar per-cell functors
 * and the cell-major layout that suit the GPU target (`lbmini::kokkos::gpu`),
 * and cannot be vectorised on a CPU.
 *
 * ### State and data layout
 *
 * Structure of arrays, **x-fastest**, with `cell = j * pitch + i` (`i` = x is
 * the contiguous direction, `pitch` is `nx` rounded up to a cache line, see
 * `host::Geometry`). Every buffer is a `Kokkos::View<Scalar*, Kokkos::HostSpace>`
 * that the kernels index through its raw pointer:
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
 *  - Each time step is one `Kokkos::parallel_for` over the *row segments* of
 *    `host::Partition` (`detail::InitFunctor` / `detail::StepFunctor`, which
 *    hand raw pointers and plain-data parameters to the kernel, so nothing in
 *    the hot loop goes through a `View`). The step policy is
 *    `Kokkos::Schedule<Kokkos::Dynamic>` with chunk size 1: with a static
 *    schedule one slow core (a hybrid CPU's E-core, an SMT sibling, a
 *    time-sliced vCPU) idles the other threads at the end of every step, and
 *    the fine-grained segments make dynamic cheap.
 *  - The unit of work is a run of vector blocks of one row, not a row: the work
 *    is balanced to a fraction of a block even for `ny = 1`, where a row
 *    partition leaves most cores idle.
 *  - The distributions and the warm start are allocated with
 *    `Kokkos::WithoutInitializing`, so their first write is `Init()`, a static
 *    `parallel_for` over the same segments the steps use (NUMA first touch);
 *    the macroscopic fields are value-initialised by Kokkos.
 *  - The thread count (`cores` from `config.yaml`; 0 lets Kokkos decide) is
 *    applied once per process by `Runtime`, which this class owns as its first
 *    member; the partition is sized for
 *    `Kokkos::DefaultHostExecutionSpace().concurrency()`. Kernels run in order
 *    on the default instance, so the host only fences where it reads the
 *    buffers (the getters) or hands control back (`Init()`, `Run()`).
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
  // `gCur_`/`gAlt_`); a copy would leave them pointing into the source's
  // buffers while its own ping-pong roles stay unswapped. Solvers are heavy,
  // unique objects held behind `ILbmTubeNd*`, so the operations are deleted
  // rather than implemented, uniformly across every backend. See
  // docs/memory-safety.md §5.2.
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
  using ExecSpace = Kokkos::DefaultHostExecutionSpace;
  using View = Kokkos::View<Scalar*, Kokkos::HostSpace>;
  using Fields = host::Fields<Scalar>;

  using InitPolicy = Kokkos::RangePolicy<ExecSpace, Kokkos::Schedule<Kokkos::Static>, Kokkos::IndexType<host::Index>>;
  using StepPolicy = Kokkos::RangePolicy<ExecSpace, Kokkos::Schedule<Kokkos::Dynamic>, Kokkos::IndexType<host::Index>>;

  static_assert(kDim_ == 2, "kokkos::cpu::LbmTube implements the 2-D closure");

  /// One time step over all row segments, then the ping-pong swap.
  ///
  /// @tparam kStoreMacro write `rho`, `p`, `T`, `u` (last step of a `Run()`).
  template<bool kStoreMacro>
  void step();

  /// The pointers that do not take part in the ping-pong.
  [[nodiscard]] Fields fixedFields() const;

  [[nodiscard]] Tensor<Scalar, kDim_> exportField(const View& field) const;

  // Must stay the first member: the runtime has to be up before the first
  // `View` is allocated (see `Runtime`).
  Runtime runtime_;

  const FluidData<Scalar> kFluid_;
  const MeshData<Scalar, kDim_> kMesh_;
  const ControlData<Scalar> kControl_;
  const PerformanceData kPerformance_;

  Index nx_;
  Index ny_;

  host::Geometry geo_;
  host::Params<Scalar> params_;
  host::Partition part_;

  View fA_; // Q * plane
  View fB_; // swap partner of fA_
  View gA_; // Q * plane
  View gB_; // swap partner of gA_
  View xi_; // D * plane (Newton warm start)

  View rho_; // plane
  View p_;   // plane
  View tem_; // plane
  View u_;   // D * plane

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
  : runtime_(static_cast<int>(performance.cores)),
    kFluid_(fluid), kMesh_(mesh), kControl_(control), kPerformance_(performance) {
  // Everything below derives from these two extents, so they are checked
  // before the first allocation: a zero extent yields an empty buffer the
  // kernels still index, and the flat indices must not overflow.
  ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "kokkos::cpu::LbmTube");

  nx_ = mesh.size[0];
  ny_ = mesh.size[1];
  geo_ = host::MakeGeometry(nx_, ny_);
  params_ = host::MakeParams<Scalar, LatticeType>(fluid, mesh, control);
  part_ = host::MakePartition(geo_.pitch, geo_.ny, kLanes, ExecSpace().concurrency());

  // Distributions and warm start are first written by `Init()` (first touch by
  // the threads that update them); the macroscopic fields are zero until then.
  const auto plane = static_cast<std::size_t>(geo_.plane);
  const auto q = static_cast<std::size_t>(kQ_);
  const auto d = static_cast<std::size_t>(kDim_);
  fA_ = View(Kokkos::view_alloc(Kokkos::WithoutInitializing, "lbmini::kokkos::cpu::fA"), q * plane);
  fB_ = View(Kokkos::view_alloc(Kokkos::WithoutInitializing, "lbmini::kokkos::cpu::fB"), q * plane);
  gA_ = View(Kokkos::view_alloc(Kokkos::WithoutInitializing, "lbmini::kokkos::cpu::gA"), q * plane);
  gB_ = View(Kokkos::view_alloc(Kokkos::WithoutInitializing, "lbmini::kokkos::cpu::gB"), q * plane);
  xi_ = View(Kokkos::view_alloc(Kokkos::WithoutInitializing, "lbmini::kokkos::cpu::xi"), d * plane);

  rho_ = View("lbmini::kokkos::cpu::rho", plane);
  p_ = View("lbmini::kokkos::cpu::p", plane);
  tem_ = View("lbmini::kokkos::cpu::tem", plane);
  u_ = View("lbmini::kokkos::cpu::u", d * plane);

  fCur_ = fA_.data();
  fAlt_ = fB_.data();
  gCur_ = gA_.data();
  gAlt_ = gB_.data();
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::exportField(const View& field) const -> Tensor<Scalar, kDim_> {
  Kokkos::fence("lbmini::kokkos::cpu::LbmTube getter");
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
  Kokkos::fence("lbmini::kokkos::cpu::LbmTube getter");
  // SoA d-major, x-fastest -> RowMajor [i][j][d] Eigen tensor.
  Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
  host::ExportVector(u_.data(), u_.data() + geo_.plane, geo_, host::Index{ 0 }, out.data());
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::fixedFields() const -> Fields {
  Fields fields;
  fields.xi = xi_.data();
  fields.rho = rho_.data();
  fields.p = p_.data();
  fields.tem = tem_.data();
  fields.u = u_.data();
  return fields;
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Init() {
  // The buffers are only ever written by kernels, so a re-Init() simply
  // restarts from the initial condition, in the current ping-pong buffer.
  Fields fields = fixedFields();
  fields.fOut = fCur_;
  fields.gOut = gCur_;

  // Static partition of the same segments the steps use: NUMA first touch.
  const detail::InitFunctor<Scalar, LatticeType> fn{ params_, geo_, part_, fields };
  Kokkos::parallel_for("lbmini::kokkos::cpu::Init", InitPolicy(0, part_.Segments()), fn);
  Kokkos::fence("lbmini::kokkos::cpu::Init");
}

template<typename Scalar, typename LatticeType>
template<bool kStoreMacro>
void LbmTube<Scalar, LatticeType>::step() {
  Fields fields = fixedFields();
  fields.fIn = fCur_;
  fields.gIn = gCur_;
  fields.fOut = fAlt_;
  fields.gOut = gAlt_;

  const detail::StepFunctor<Scalar, LatticeType, kStoreMacro> fn{ params_, geo_, part_, fields };
  Kokkos::parallel_for("lbmini::kokkos::cpu::Step", StepPolicy(0, part_.Segments()).set_chunk_size(1), fn);

  std::swap(fCur_, fAlt_);
  std::swap(gCur_, gAlt_);
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
    if (t + 1 < steps)
      step<false>();
    else
      step<true>();
  }
  Kokkos::fence("lbmini::kokkos::cpu::Run");
}
} // namespace lbmini::kokkos::cpu

#endif // LBMINI_KOKKOS_CPU_LBMTUBE_HPP_
