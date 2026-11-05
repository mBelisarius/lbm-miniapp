#ifndef LBMINI_OPENMP_CPU_LBMTUBE_HPP_
#define LBMINI_OPENMP_CPU_LBMTUBE_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>
#include <omp.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/Stencil.hpp"

/// Force inlining of the per-cell physics into the cell loops: the populations
/// of one cell then live in registers and every lattice lookup folds into an
/// immediate once the direction loops are unrolled.
#if defined(__GNUC__) || defined(__clang__)
#define LBMINI_OMP_CPU_INLINE        [[gnu::always_inline]] inline
#define LBMINI_OMP_CPU_LAMBDA_INLINE __attribute__((always_inline))
#else
#define LBMINI_OMP_CPU_INLINE inline
#define LBMINI_OMP_CPU_LAMBDA_INLINE
#endif

namespace lbmini::openmp::cpu {
namespace detail {
/**
 * @brief `AlignedAllocator` that default-initialises instead of zero-filling.
 *
 * `resize()` then only reserves address space: the first write to every page
 * happens in `Init()` / the first time step, from the thread that owns the
 * page under the `schedule(static)` partition (NUMA first touch), instead of
 * from the constructing thread.
 */
template<typename T>
struct FirstTouchAllocator : lbmini::AlignedAllocator<T, 64> {
  template<class U>
  struct rebind {
    using other = FirstTouchAllocator<U>;
  };

  FirstTouchAllocator() noexcept = default;

  template<class U>
  explicit FirstTouchAllocator(const FirstTouchAllocator<U>&) noexcept {}

  template<class U>
  void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
    ::new (static_cast<void*>(p)) U;
  }

  template<class U, class... Args>
  void construct(U* p, Args&&... args) {
    ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
  }
};

template<typename T>
using FirstTouchBuffer = std::vector<T, FirstTouchAllocator<T>>;
} // namespace detail

/**
 * @brief OpenMP multi-core variant of `lbmini::plain::LbmTube`.
 *
 * Numerically equivalent to the plain reference (same operations in the same
 * order per cell), but organised around a single fused kernel.
 *
 * ### State and data layout
 *
 * | buffer        | size    | layout              | role                               |
 * | :------------ | :------ | :------------------ | :--------------------------------- |
 * | `fA_/fB_`     | `Q * N` | `f[idc * N + cell]` | post-collision density (ping-pong) |
 * | `gA_/gB_`     | `Q * N` | `g[idc * N + cell]` | post-collision energy (ping-pong)  |
 * | `xi_`         | `D * N` | `xi[d * N + cell]`  | Newton-Raphson warm start          |
 * | `rho_/p_/tem_`| `N`     | `x[cell]`           | macroscopic output                 |
 * | `u_`          | `D * N` | `u[d * N + cell]`   | macroscopic output                 |
 *
 * with `cell = i * ny + j`. The solver state is the set of *post*-collision
 * populations: `Init()` seeds the equilibria, recovers the macroscopic fields
 * (what the getters see after `Init()`) and already runs the first collision;
 * every time step then
 *
 *  1. pulls the 9 directions of each cell from their four inverse-distance
 *     source cells of the previous post-collision state,
 *  2. reduces them to `rho`, `u`, `T`, `p`,
 *  3. evaluates the equilibria, the Newton-Raphson energy closure and the
 *     BGK / Knudsen collision in registers, and
 *  4. writes the new post-collision populations into the other ping-pong
 *     buffer (the pointers are swapped, never copied).
 *
 * Per cell and step that is 2 x 9 x 8 B of streamed reads (the neighbouring
 * rows are reused from cache), 2 x 9 x 8 B of writes and 2 x 2 x 8 B for the
 * warm start, ~320 B in total. The reference's warm-start component 0
 * (`log(scale / rho) - smax`) is never read and is therefore not kept. The
 * macroscopic fields are written only by the last step of a `Run()`, the only
 * point where a getter can observe them.
 *
 * ### Streaming
 *
 * With a uniform lattice shift the four source cells and weights of direction
 * `idc` are the same for every cell up to a clamp of the source column to the
 * tube and a periodic wrap of the source row, so the kernel uses the shared
 * per-direction `lbmini::DirectionStencil` (`Lbm/Stencil.hpp`) and resolves the
 * sources exactly like `lbmini::GatherSources`. The clamped source columns
 * depend on the row `i` only and are hoisted out of the inner `j` loop; the
 * wrap only matters in the first / last few columns `j`, which are peeled so
 * that the interior `j` loop reads four contiguous runs per direction without
 * any index arithmetic.
 *
 * ### Threading
 *
 *  - One `#pragma omp parallel` region per `Run()`; each step is one
 *    `#pragma omp for schedule(static)` over the rows `i`, whose implicit
 *    barrier is the only synchronisation per step. Every thread swaps its own
 *    copy of the ping-pong pointers, so no `single` is needed.
 *  - `Init()` uses the same static row partition, so the distributions are
 *    first touched by the threads that update them (NUMA locality).
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
   * last `Run()`.
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
  static constexpr int kQ = static_cast<int>(kQ_);
  static constexpr int kD = static_cast<int>(kDim_);
  static constexpr Scalar kTiny_ = Scalar{ 1.0e-12 };
  static constexpr Scalar kMaxExp_ = Scalar{ 700.0 };
  static constexpr int kMaxIter_ = 3;

  static_assert(kD == 2, "openmp::cpu::LbmTube implements the 2-D closure");

  /// Fluid and lattice constants of the per-cell physics.
  struct Constants {
    Scalar ushiftX;
    Scalar ushiftY;
    Scalar cv;
    Scalar cp;
    Scalar invCv;
    Scalar rgas;
    Scalar viscosity;
    Scalar conductivity;
  };

  /// Density equilibrium, factorised product form.
  LBMINI_OMP_CPU_INLINE static void Feq(
    Scalar rhoc,
    Scalar ux,
    Scalar uy,
    Scalar temc,
    const Constants& c,
    Scalar (&feq)[kQ]
  );

  /// `rho`, `u`, `T`, `p` from the populations of one cell.
  LBMINI_OMP_CPU_INLINE static void Macroscopic(
    const Scalar (&f)[kQ],
    const Scalar (&g)[kQ],
    const Constants& c,
    Scalar& rho,
    Scalar& ux,
    Scalar& uy,
    Scalar& tem,
    Scalar& pres
  );

  /**
   * @brief Equilibria + Newton-Raphson energy closure + BGK collision, in place.
   *
   * `f`/`g` enter as the pre-collision populations of the cell and leave as
   * the post-collision ones; `xi` is the Newton warm start (read and updated).
   * The Newton loop runs a fixed number of iterations with clamped reciprocals
   * and a sticky `solverOk` predicate, exactly as the reference.
   */
  LBMINI_OMP_CPU_INLINE static void Collide(
    const Constants& c,
    Scalar rhoc,
    Scalar ux,
    Scalar uy,
    Scalar temc,
    Scalar (&f)[kQ],
    Scalar (&g)[kQ],
    Scalar (&xi)[kD]
  );

  /**
   * @brief One time step over the rows of the calling thread.
   *
   * Must be called by every thread of the enclosing parallel region (it
   * contains the work-sharing `omp for` and its implicit barrier).
   *
   * @tparam kStoreMacro write `rho`, `p`, `T`, `u` (last step of a `Run()`).
   */
  template<bool kStoreMacro>
  void step(const Scalar* fIn, const Scalar* gIn, Scalar* fOut, Scalar* gOut);

  const FluidData<Scalar> kFluid_;
  const MeshData<Scalar, kDim_> kMesh_;
  const ControlData<Scalar> kControl_;
  const PerformanceData kPerformance_;

  Index nx_;
  Index ny_;
  Index N_; // nx_ * ny_

  Constants constants_;
  lbmini::DirectionStencil<Scalar, kQ> stencil_;
  bool anyNearest_ = false; // some direction uses the nearest-node fallback

  // Columns `j` in `[jLo_, jHi_)` read row `j + sy0_[idc]` / `j + sy1_[idc]`
  // of every direction without a periodic wrap (empty if `jHi_ <= jLo_`).
  Index jLo_ = 0;
  Index jHi_ = 0;
  Index sy0_[kQ] = {};
  Index sy1_[kQ] = {};

  detail::FirstTouchBuffer<Scalar> fA_; // size Q * N, idc-major
  detail::FirstTouchBuffer<Scalar> fB_; // size Q * N  (swap partner of fA_)
  detail::FirstTouchBuffer<Scalar> gA_; // size Q * N, idc-major
  detail::FirstTouchBuffer<Scalar> gB_; // size Q * N  (swap partner of gA_)
  detail::FirstTouchBuffer<Scalar> xi_; // size D * N, d-major (Newton warm start)

  lbmini::DeviceBuffer<Scalar> rho_; // size N
  lbmini::DeviceBuffer<Scalar> p_;   // size N
  lbmini::DeviceBuffer<Scalar> tem_; // size N
  lbmini::DeviceBuffer<Scalar> u_;   // size D * N, d-major

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
  N_ = nx_ * ny_;

  constants_.ushiftX = control.U(0);
  constants_.ushiftY = control.U(1);
  constants_.cv = fluid.specificHeatCv;
  constants_.cp = fluid.specificHeatCp;
  constants_.invCv = Scalar{ 1 } / fluid.specificHeatCv;
  constants_.rgas = fluid.constant;
  constants_.viscosity = fluid.viscosity;
  constants_.conductivity = fluid.conductivity;

  stencil_ = lbmini::BuildDirectionStencil<LatticeType>(ny_, control.U(0), control.U(1), control.idw);

  // Corner row `(j + o) mod ny` with `o` in `[0, ny)` wraps for `j >= ny - o`.
  // A small `o` is a forward offset (valid for `j < ny - o`), a large one a
  // backward offset `o - ny` (valid for `j >= ny - o`).
  jLo_ = 0;
  jHi_ = ny_;
  auto classify = [this](const Index o) -> Index {
    if (2 * o <= ny_) {
      jHi_ = std::min(jHi_, ny_ - o);
      return o;
    }
    jLo_ = std::max(jLo_, ny_ - o);
    return o - ny_;
  };
  for (int idc = 0; idc < kQ; ++idc) {
    anyNearest_ = anyNearest_ || (stencil_.nearest[idc] != 0);
    sy0_[idc] = classify(stencil_.oy0[idc]);
    sy1_[idc] = classify(stencil_.oy1[idc]);
  }
  jHi_ = std::max(jHi_, jLo_);
  if (anyNearest_) {
    // Degenerate weights: every cell takes the generic gather.
    jLo_ = ny_;
    jHi_ = ny_;
  }

  // Distributions and warm start are first written by `Init()` (see
  // `FirstTouchAllocator`); the macroscopic fields are zero until then.
  fA_.resize(N_ * kQ_);
  fB_.resize(N_ * kQ_);
  gA_.resize(N_ * kQ_);
  gB_.resize(N_ * kQ_);
  xi_.resize(N_ * kDim_);

  rho_.assign(N_, Scalar{ 0 });
  p_.assign(N_, Scalar{ 0 });
  tem_.assign(N_, Scalar{ 0 });
  u_.assign(N_ * kDim_, Scalar{ 0 });

  fCur_ = fA_.data();
  fAlt_ = fB_.data();
  gCur_ = gA_.data();
  gAlt_ = gB_.data();
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::P() const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(p_.begin(), p_.end(), out.data());
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::Rho() const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(rho_.begin(), rho_.end(), out.data());
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::T() const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(tem_.begin(), tem_.end(), out.data());
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::U() const -> Tensor<Scalar, kDim_ + 1> {
  // SoA d-major -> RowMajor [i][j][d] Eigen tensor.
  Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
  Scalar* dst = out.data();
  for (Index cell = 0; cell < N_; ++cell)
    for (Index d = 0; d < kDim_; ++d)
      dst[cell * kDim_ + d] = u_[d * N_ + cell];
  return out;
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Feq(
  const Scalar rhoc,
  const Scalar ux,
  const Scalar uy,
  const Scalar temc,
  const Constants& c,
  Scalar (&feq)[kQ]
) {
  LBMINI_UNROLL(9)
  for (int idc = 0; idc < kQ; ++idc) {
    Scalar phi = rhoc;
    LBMINI_UNROLL(2)
    for (int d = 0; d < kD; ++d) {
      const auto vi = LatticeType::Velocity(idc, d);
      const Scalar uia = (d == 0 ? ux : uy) - (d == 0 ? c.ushiftX : c.ushiftY);
      const Scalar uia2t = uia * uia + temc;
      Scalar pf;
      if (vi == 0)
        pf = Scalar{ 1 } - uia2t;
      else if (vi == 1)
        pf = Scalar{ 0.5 } * (uia + uia2t);
      else
        pf = Scalar{ 0.5 } * (-uia + uia2t);
      phi *= pf;
    }
    feq[idc] = phi;
  }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Macroscopic(
  const Scalar (&f)[kQ],
  const Scalar (&g)[kQ],
  const Constants& c,
  Scalar& rho,
  Scalar& ux,
  Scalar& uy,
  Scalar& tem,
  Scalar& pres
) {
  Scalar rhoSum = Scalar{ 0 };
  Scalar nrg = Scalar{ 0 };
  Scalar momX = Scalar{ 0 };
  Scalar momY = Scalar{ 0 };
  LBMINI_UNROLL(9)
  for (int idc = 0; idc < kQ; ++idc) {
    rhoSum += f[idc];
    nrg += g[idc];
    momX += LatticeType::Cshift(idc, 0, c.ushiftX) * f[idc];
    momY += LatticeType::Cshift(idc, 1, c.ushiftY) * f[idc];
  }
  const Scalar invRho = Scalar{ 1 } / rhoSum;
  ux = momX * invRho;
  uy = momY * invRho;
  const Scalar kin = Scalar{ 0.5 } * (ux * ux + uy * uy);
  tem = (Scalar{ 0.5 } * nrg * invRho - kin) * c.invCv;
  rho = rhoSum;
  pres = c.rgas * rhoSum * tem;
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Collide(
  const Constants& c,
  const Scalar rhoc,
  const Scalar ux,
  const Scalar uy,
  const Scalar temc,
  Scalar (&f)[kQ],
  Scalar (&g)[kQ],
  Scalar (&xi)[kD]
) {
  using std::exp;
  using std::fabs;

  const Scalar ushiftX = c.ushiftX;
  const Scalar ushiftY = c.ushiftY;
  const Scalar u2 = ux * ux + uy * uy;

  Scalar feqLocal[kQ];
  Feq(rhoc, ux, uy, temc, c, feqLocal);

  // Geq via Newton-Raphson on the normalised energy flux.
  const Scalar E = c.cv * temc + Scalar{ 0.5 } * u2;
  const Scalar targetE = Scalar{ 2 } * rhoc * E;
  Scalar targetM[kD];
  LBMINI_UNROLL(2)
  for (int d = 0; d < kD; ++d)
    targetM[d] = Scalar{ 2 } * rhoc * (d == 0 ? ux : uy) * (E + temc) / targetE;

  Scalar alpha = Scalar{ 1 };
  Scalar si[kQ];
  Scalar e[kQ];
  Scalar Z = Scalar{ 0 };
  Scalar smax = Scalar{ 0 };
  bool solverOk = true;

  LBMINI_UNROLL(3)
  for (int iter = 0; iter < kMaxIter_; ++iter) {
    smax = -kMaxExp_;
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      Scalar s = Scalar{ 0 };
      LBMINI_UNROLL(2)
      for (int d = 0; d < kD; ++d)
        s += xi[d] * LatticeType::Cshift(idc, d, (d == 0 ? ushiftX : ushiftY));
      si[idc] = s;
      smax = (s > smax) ? s : smax;
    }
    smax = (smax < kMaxExp_) ? smax : kMaxExp_;

    Scalar S1[kD] = { Scalar{ 0 }, Scalar{ 0 } };
    Scalar S2[kD][kD] = { { Scalar{ 0 }, Scalar{ 0 } }, { Scalar{ 0 }, Scalar{ 0 } } };
    Z = Scalar{ 0 };
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      Scalar expo = si[idc] - smax;
      expo = (expo < kMaxExp_) ? expo : kMaxExp_;
      expo = (expo > -kMaxExp_) ? expo : -kMaxExp_;
      const Scalar wev = LatticeType::Weights(idc, temc) * exp(expo);
      e[idc] = wev;
      Z += wev;
      LBMINI_UNROLL(2)
      for (int a = 0; a < kD; ++a) {
        const Scalar cia = LatticeType::Cshift(idc, a, (a == 0 ? ushiftX : ushiftY));
        S1[a] += cia * wev;
        LBMINI_UNROLL(2)
        for (int b = 0; b < kD; ++b)
          S2[a][b] += cia * LatticeType::Cshift(idc, b, (b == 0 ? ushiftX : ushiftY)) * wev;
      }
    }

    const Scalar Zsafe = (Z > kTiny_) ? Z : kTiny_;
    solverOk = solverOk && (Z > kTiny_);
    const Scalar invZ = Scalar{ 1 } / Zsafe;

    Scalar J[kD][kD];
    LBMINI_UNROLL(2)
    for (int a = 0; a < kD; ++a)
      LBMINI_UNROLL(2)
      for (int b = 0; b < kD; ++b)
        J[a][b] = (S2[a][b] - S1[a] * S1[b] * invZ) * invZ;
    J[0][0] += kTiny_;
    J[1][1] += kTiny_;

    const Scalar detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    const Scalar absDet = fabs(detJ);
    const Scalar detSafe = (absDet > kTiny_) ? detJ : ((detJ >= Scalar{ 0 }) ? kTiny_ : -kTiny_);
    solverOk = solverOk && (absDet > kTiny_);
    const Scalar invDet = Scalar{ 1 } / detSafe;

    const Scalar r0 = S1[0] * invZ - targetM[0];
    const Scalar r1 = S1[1] * invZ - targetM[1];
    xi[0] -= alpha * (J[1][1] * r0 - J[0][1] * r1) * invDet;
    xi[1] -= alpha * (-J[1][0] * r0 + J[0][0] * r1) * invDet;
    alpha *= Scalar{ 0.5 };
  }

  Scalar geqLocal[kQ];
  if (solverOk) {
    // Recompute e[], Z with the final xi.
    smax = -kMaxExp_;
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      Scalar s = Scalar{ 0 };
      LBMINI_UNROLL(2)
      for (int d = 0; d < kD; ++d)
        s += xi[d] * LatticeType::Cshift(idc, d, (d == 0 ? ushiftX : ushiftY));
      si[idc] = s;
      smax = (s > smax) ? s : smax;
    }
    smax = (smax < kMaxExp_) ? smax : kMaxExp_;
    Z = Scalar{ 0 };
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      Scalar expo = si[idc] - smax;
      expo = (expo < kMaxExp_) ? expo : kMaxExp_;
      expo = (expo > -kMaxExp_) ? expo : -kMaxExp_;
      e[idc] = LatticeType::Weights(idc, temc) * exp(expo);
      Z += e[idc];
    }
    const Scalar scale = targetE / Z;
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc)
      geqLocal[idc] = scale * e[idc];
  } else {
    // Fallback: normalised-Wi distribution, cold restart of the warm start.
    xi[0] = Scalar{ 0 };
    xi[1] = Scalar{ 0 };
    Scalar sumW = Scalar{ 0 };
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc)
      sumW += LatticeType::Weights(idc, temc);
    if (sumW <= kTiny_) {
      const Scalar uni = targetE / Scalar(kQ);
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc)
        geqLocal[idc] = uni;
    } else {
      const Scalar sc = targetE / sumW;
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc)
        geqLocal[idc] = LatticeType::Weights(idc, temc) * sc;
    }
  }

  // Collision (BGK + Knudsen sensor + thermal-relaxation correction).
  const Scalar tau = c.viscosity / (rhoc * temc) + Scalar{ 0.5 };
  const Scalar omega = Scalar{ 1 } / tau;
  const Scalar diffusivity = c.conductivity / (rhoc * c.cp);
  const Scalar tauT = diffusivity / temc + Scalar{ 0.5 };
  Scalar omegaT = Scalar{ 1 } / tauT;

  Scalar eps = Scalar{ 0 };
  LBMINI_UNROLL(9)
  for (int idc = 0; idc < kQ; ++idc) {
    const Scalar diff = f[idc] - feqLocal[idc];
    const Scalar den = (feqLocal[idc] > kTiny_) ? feqLocal[idc] : kTiny_;
    eps += fabs(diff) / den;
  }
  eps /= Scalar(kQ);
  Scalar sigma = Scalar{ 1 };
  if (eps >= Scalar{ 1 })
    sigma = omega;
  else if (eps >= Scalar{ 0.1 })
    sigma = Scalar{ 1.35 };
  else if (eps >= Scalar{ 0.01 })
    sigma = Scalar{ 1.05 };

  Scalar omegaL = omega / sigma;
  omegaL = (omegaL > Scalar{ 1 }) ? omegaL : Scalar{ 1 };
  omegaL = (omegaL < (Scalar{ 2 } - Scalar{ 1e-7 })) ? omegaL : (Scalar{ 2 } - Scalar{ 1e-7 });
  omegaT = (omegaT > Scalar{ 1 }) ? omegaT : Scalar{ 1 };
  omegaT = (omegaT < (Scalar{ 2 } - Scalar{ 1e-7 })) ? omegaT : (Scalar{ 2 } - Scalar{ 1e-7 });

  // L[a] = sum_i 2 * (u . c_i) * c_i_a * (f - feq)   (c_i unshifted)
  Scalar L[kD] = { Scalar{ 0 }, Scalar{ 0 } };
  LBMINI_UNROLL(9)
  for (int idc = 0; idc < kQ; ++idc) {
    const Scalar cix0 = static_cast<Scalar>(LatticeType::Velocity(idc, 0));
    const Scalar ciy0 = static_cast<Scalar>(LatticeType::Velocity(idc, 1));
    const Scalar uvi = ux * cix0 + uy * ciy0;
    const Scalar aux = Scalar{ 2 } * uvi * (f[idc] - feqLocal[idc]);
    L[0] += aux * cix0;
    L[1] += aux * ciy0;
  }

  const Scalar invT = Scalar{ 1 } / temc;
  LBMINI_UNROLL(9)
  for (int idc = 0; idc < kQ; ++idc) {
    const Scalar fOld = f[idc];
    f[idc] = fOld + omegaL * (feqLocal[idc] - fOld);
    const Scalar cidotL = L[0] * LatticeType::Cshift(idc, 0, ushiftX) + L[1] * LatticeType::Cshift(idc, 1, ushiftY);
    const Scalar Wi = LatticeType::Weights(idc, temc);
    const Scalar gDiff = Wi * cidotL * invT;
    const Scalar gOld = g[idc];
    g[idc] = gOld + omegaL * (geqLocal[idc] - gOld) + (omegaL - omegaT) * gDiff;
  }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Init() {
  // The buffers are only ever written by kernels, so a re-Init() simply
  // restarts from the initial condition, in the current ping-pong buffer.
  const Constants c = constants_;
  const Index nx = nx_;
  const Index ny = ny_;
  const Index N = N_;
  const Scalar densityL = kFluid_.densityL;
  const Scalar pressureL = kFluid_.pressureL;
  const Scalar densityR = kFluid_.densityR;
  const Scalar pressureR = kFluid_.pressureR;
  Scalar* __restrict__ pF = fCur_;
  Scalar* __restrict__ pG = gCur_;
  Scalar* __restrict__ pXi = xi_.data();
  Scalar* __restrict__ pRho = rho_.data();
  Scalar* __restrict__ pP = p_.data();
  Scalar* __restrict__ pT = tem_.data();
  Scalar* __restrict__ pU = u_.data();

  // Same `schedule(static)` row partition as `step()`: NUMA first touch.
  #pragma omp parallel for schedule(static)
  for (Index i = 0; i < nx; ++i) {
    // Sod-like left/right state (reference `Init()`).
    const bool left = i < nx / 2;
    const Scalar rho0 = left ? densityL : densityR;
    const Scalar p0 = left ? pressureL : pressureR;
    const Scalar tem0 = LatticeType::Cs2 * p0 / (rho0 * c.rgas);
    const Scalar ux0 = Scalar{ 0 };
    const Scalar uy0 = Scalar{ 0 };

    for (Index j = 0; j < ny; ++j) {
      const Index cell = i * ny + j;

      // seedEquilibria(): f = feq, g = Wi(T)-normalised geq (xi == 0).
      Scalar f[kQ];
      Scalar g[kQ];
      Feq(rho0, ux0, uy0, tem0, c, f);
      const Scalar u2 = ux0 * ux0 + uy0 * uy0;
      const Scalar E = c.cv * tem0 + Scalar{ 0.5 } * u2;
      const Scalar targetE = Scalar{ 2 } * rho0 * E;
      Scalar sumW = Scalar{ 0 };
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc)
        sumW += LatticeType::Weights(idc, tem0);
      const Scalar sc = (sumW > kTiny_) ? targetE / sumW : targetE / Scalar(kQ);
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc) {
        const Scalar wi = (sumW > kTiny_) ? LatticeType::Weights(idc, tem0) : Scalar{ 1 };
        g[idc] = wi * sc;
      }

      // computeMacroscopic(): the state every getter sees after Init().
      Scalar rho;
      Scalar ux;
      Scalar uy;
      Scalar tem;
      Scalar pres;
      Macroscopic(f, g, c, rho, ux, uy, tem, pres);
      pRho[cell] = rho;
      pP[cell] = pres;
      pT[cell] = tem;
      pU[0 * N + cell] = ux;
      pU[1 * N + cell] = uy;

      // First collision, cold Newton start.
      Scalar xi[kD] = { Scalar{ 0 }, Scalar{ 0 } };
      Collide(c, rho, ux, uy, tem, f, g, xi);

      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc) {
        pF[idc * N + cell] = f[idc];
        pG[idc * N + cell] = g[idc];
      }
      pXi[0 * N + cell] = xi[0];
      pXi[1 * N + cell] = xi[1];
    }
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
      if (t + 1 < steps)
        step<false>(fIn, gIn, fOut, gOut);
      else
        step<true>(fIn, gIn, fOut, gOut);
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
void LbmTube<Scalar, LatticeType>::step(
  const Scalar* const fInPtr,
  const Scalar* const gInPtr,
  Scalar* const fOutPtr,
  Scalar* const gOutPtr
) {
  const Scalar* __restrict__ fIn = fInPtr;
  const Scalar* __restrict__ gIn = gInPtr;
  Scalar* __restrict__ fOut = fOutPtr;
  Scalar* __restrict__ gOut = gOutPtr;
  Scalar* __restrict__ pXi = xi_.data();
  Scalar* __restrict__ pRho = rho_.data();
  Scalar* __restrict__ pP = p_.data();
  Scalar* __restrict__ pT = tem_.data();
  Scalar* __restrict__ pU = u_.data();

  // Thread-local copies: nothing below can alias them, so they stay in
  // registers / on the stack across the stores of the cell loop.
  const Constants c = constants_;
  const Index nx = nx_;
  const Index ny = ny_;
  const Index N = N_;
  const Index jLo = jLo_;
  const Index jHi = jHi_;
  const lbmini::DirectionStencil<Scalar, kQ> st = stencil_;
  Index sy0[kQ];
  Index sy1[kQ];
  for (int idc = 0; idc < kQ; ++idc) {
    sy0[idc] = sy0_[idc];
    sy1[idc] = sy1_[idc];
  }

  auto clampX = [nx](const Index x) -> Index { return (x < 0) ? Index{ 0 } : ((x >= nx) ? nx - 1 : x); };

  #pragma omp for schedule(static)
  for (Index i = 0; i < nx; ++i) {
    // Flat offsets of the two (clamped) source columns of every direction.
    Index rowA[kQ];
    Index rowB[kQ];
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      const Index xa = i + st.ox[idc];
      rowA[idc] = idc * N + clampX(xa) * ny;
      rowB[idc] = idc * N + clampX(xa + 1) * ny;
    }

    // Shared tail of both gathers: macroscopics, collision, stores.
    auto relax = [&](const Index cell, Scalar (&f)[kQ], Scalar (&g)[kQ]) LBMINI_OMP_CPU_LAMBDA_INLINE {
      Scalar rho;
      Scalar ux;
      Scalar uy;
      Scalar tem;
      Scalar pres;
      Macroscopic(f, g, c, rho, ux, uy, tem, pres);
      if constexpr (kStoreMacro) {
        pRho[cell] = rho;
        pP[cell] = pres;
        pT[cell] = tem;
        pU[0 * N + cell] = ux;
        pU[1 * N + cell] = uy;
      }

      Scalar xi[kD] = { pXi[0 * N + cell], pXi[1 * N + cell] };
      Collide(c, rho, ux, uy, tem, f, g, xi);

      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc) {
        fOut[idc * N + cell] = f[idc];
        gOut[idc * N + cell] = g[idc];
      }
      pXi[0 * N + cell] = xi[0];
      pXi[1 * N + cell] = xi[1];
    };

    // Generic gather (`lbmini::GatherSources`): periodic wrap of the rows and
    // the nearest-node fallback; only the peeled edge columns take it.
    auto generic = [&](const Index j) {
      Scalar f[kQ];
      Scalar g[kQ];
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc) {
        if (st.nearest[idc]) {
          const auto xN = static_cast<Index>(std::round(static_cast<Scalar>(i) - st.cix[idc]));
          const auto yN = static_cast<Index>(std::round(static_cast<Scalar>(j) - st.ciy[idc]));
          Index yW = yN % ny;
          yW += (yW < 0) ? ny : 0;
          const Index src = idc * N + ((xN < 0 || xN >= nx) ? i * ny + j : xN * ny + yW);
          f[idc] = fIn[src];
          g[idc] = gIn[src];
        } else {
          Index ya = j + st.oy0[idc];
          ya -= (ya >= ny) ? ny : 0;
          Index yb = j + st.oy1[idc];
          yb -= (yb >= ny) ? ny : 0;
          const Scalar w0 = st.w[idc][0];
          const Scalar w1 = st.w[idc][1];
          const Scalar w2 = st.w[idc][2];
          const Scalar w3 = st.w[idc][3];
          const Index a = rowA[idc];
          const Index b = rowB[idc];
          f[idc] = w0 * fIn[a + ya] + w1 * fIn[b + ya] + w2 * fIn[a + yb] + w3 * fIn[b + yb];
          g[idc] = w0 * gIn[a + ya] + w1 * gIn[b + ya] + w2 * gIn[a + yb] + w3 * gIn[b + yb];
        }
      }
      relax(i * ny + j, f, g);
    };

    for (Index j = 0; j < jLo; ++j)
      generic(j);

    // Interior: four contiguous runs per direction, no index arithmetic.
    for (Index j = jLo; j < jHi; ++j) {
      Scalar f[kQ];
      Scalar g[kQ];
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc) {
        const Index a0 = rowA[idc] + sy0[idc] + j;
        const Index b0 = rowB[idc] + sy0[idc] + j;
        const Index a1 = rowA[idc] + sy1[idc] + j;
        const Index b1 = rowB[idc] + sy1[idc] + j;
        const Scalar w0 = st.w[idc][0];
        const Scalar w1 = st.w[idc][1];
        const Scalar w2 = st.w[idc][2];
        const Scalar w3 = st.w[idc][3];
        f[idc] = w0 * fIn[a0] + w1 * fIn[b0] + w2 * fIn[a1] + w3 * fIn[b1];
        g[idc] = w0 * gIn[a0] + w1 * gIn[b0] + w2 * gIn[a1] + w3 * gIn[b1];
      }
      relax(i * ny + j, f, g);
    }

    for (Index j = jHi; j < ny; ++j)
      generic(j);
  }
}
} // namespace lbmini::openmp::cpu

#endif // LBMINI_OPENMP_CPU_LBMTUBE_HPP_
