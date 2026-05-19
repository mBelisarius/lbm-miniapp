#ifndef LBMINI_OPENMP_CPU_LBMTUBE_HPP_
#define LBMINI_OPENMP_CPU_LBMTUBE_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Utils.hpp"

namespace lbmini::openmp::cpu {
/**
 * @brief OpenMP multi-core/SIMD variant of `lbmini::plain::LbmTube`.
 *
 * Numerically identical to the plain reference; the data layout, kernel
 * bodies, and `DeviceBuffer<Scalar>` storage policy are shared verbatim
 * (see `lbmini::plain::LbmTube` for the full design rationale on SoA
 * velocity-major layout, kernel fission, stack-local Newton-Raphson
 * workspaces, and `LBMINI_UNROLL(N)` scalarisation of `kQ_`-sized arrays).
 *
 * OpenMP-specific design decisions:
 *  - The `#pragma omp parallel` region is **hoisted around the `Run()`
 *    time loop**, so the thread pool forks once per `Run(steps)`; each
 *    kernel inside uses `#pragma omp for simd schedule(static)` to amortise
 *    fork/join and to vectorise the cell loop: every thread receives a
 *    contiguous block of cells and sweeps it in SIMD lanes. The O(1) buffer
 *    swap between kernels is guarded by `#pragma omp single`, whose implicit
 *    barrier orders the ping-pong.
 *  - The cell-loop bodies are branch-free (selects instead of `if`), so the
 *    compiler can map them onto SIMD lanes without masking around divergent
 *    paths. The per-cell arithmetic is the plain reference's, so results
 *    agree with it up to round-off (contractions into FMA may differ).
 *  - The collision kernel is tiled (`kTile_` cells at a time, `omp for` over
 *    tiles) with the direction/iteration loops outside flat `omp simd` loops
 *    over the tile's cells. GCC does not vectorize a cell loop that nests
 *    those loops, so `exp`/`log` would stay scalar; in the tiled form they
 *    become libmvec calls under `-ffast-math` (`LBMINI_FAST_MATH`).
 *  - The thread count (`cores` from `config.yaml`, or the performance-core
 *    count on hybrid Intel hosts when `cores == 0`) and the
 *    `OMP_PLACES=cores` / `OMP_PROC_BIND=close` affinity are process-global
 *    settings owned by `lbmini::Executor`; this class only consumes them.
 *    They are applied once per process, before any solver exists, so the
 *    first parallel region here already sees the final configuration and
 *    first-touch allocations are NUMA-local.
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
   * @brief Lazy copy-out accessors for pressure / density / temperature / velocity.
   *
   * Macroscopic fields (`rho`, `p`, `tem`, `u`) are kept in a small, canonical
   * RowMajor Eigen layout so these getters can return by value cheaply.
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

protected:
  // `Index` (64-bit) throughout: a mesh with `nx_ * ny_` beyond 2^31 would
  // wrap a 32-bit product to a negative flat index and gather from a wild
  // address. `ValidateMeshExtents()` rejects such meshes, and this signature
  // keeps the arithmetic itself out of the narrow type.
  [[nodiscard]] Index cellIndex(const Index i, const Index j) const { return i * ny_ + j; }

  /**
   * @brief Recomputes macroscopic fields (rho, p, T, u) from f_ and g_.
   *
   * Worksharing SIMD loop over cells (`#pragma omp for simd`, static
   * schedule). Called from `Init()`, outside any parallel region, so the
   * calling thread sweeps every cell on its own.
   */
  void computeMacroscopic();

  /**
   * @brief Initializes f_ and g_ using local equilibria.
   *
   * Evaluated once during Init. Uses OpenMP parallelization over cells.
   */
  void seedEquilibria();

  /**
   * @brief Fused BGK collision and equilibrium computation.
   *
   * Processes the cells in tiles of `kTile_` (worksharing over tiles). Each
   * stage is a flat `omp simd` loop over the lanes of a tile, so every `exp`
   * and `log` is vectorized through the compiler's SIMD math library (libmvec
   * with `-ffast-math`) instead of being a scalar call per cell. Tile scratch
   * lives on the stack; thread locality is maintained across steps.
   */
  void collisionAndEquilibria();

  /**
   * @brief Combined streaming and macroscopic update.
   *
   * Calls stream() followed by computeMacroscopic() to complete a step.
   */
  void streamAndMacroscopic();

  // Deliberately `protected` rather than `private`: the memory-safety suite
  // (`test/streaming_table_test.cpp`) derives from this class to assert that
  // every precomputed gather index stays inside `[0, N_)`. The streaming
  // kernel dereferences `streamIdx_` without any bounds check, so that table
  // is the one place where a geometry mistake becomes an out-of-bounds read.
protected:
  /**
   * @brief Precompute per-(cell, idc) streaming neighbor indices and weights.
   *
   * Allocated on device, but static across time steps. Evaluated on host.
   */
  void buildStreamingTables();

  /**
   * @brief Performs the IDW streaming gather into the alternate buffers.
   *
   * Every thread walks the `kQ_` velocity planes in the same order and the
   * cells of each plane are shared out by a worksharing SIMD loop, so the
   * innermost loop is unit-stride in the output and gathers from the source
   * plane. Data locality is preserved by OMP thread binding.
   */
  void stream();

  static constexpr Scalar kTiny_ = Scalar{ 1.0e-12 };
  static constexpr Scalar kMaxExp_ = Scalar{ 700.0 };

  // Cells per tile in `collisionAndEquilibria()`: a multiple of every SIMD width
  // (8 doubles / 16 floats at 512 bits) and small enough that the per-tile
  // working set stays in L1.
  static constexpr Index kTile_ = 32;

  const FluidData<Scalar> kFluid_;
  const MeshData<Scalar, kDim_> kMesh_;
  const ControlData<Scalar> kControl_;
  const PerformanceData kPerformance_;

  // Cached mesh extents (promoted to primitive ints for register residency).
  Index nx_;
  Index ny_;
  Index N_;        // nx_ * ny_
  Index uSize_;    // N_ * Dim
  Index distSize_; // N_ * Q

  // Flat SoA storage (all contiguous, 64B-aligned by std::vector on most libs).
  lbmini::DeviceBuffer<Scalar> rho_;    // size N_
  lbmini::DeviceBuffer<Scalar> p_;      // size N_
  lbmini::DeviceBuffer<Scalar> tem_;    // size N_
  lbmini::DeviceBuffer<Scalar> u_;      // size uSize_,   d-major
  lbmini::DeviceBuffer<Scalar> f_;      // size distSize_, idc-major
  lbmini::DeviceBuffer<Scalar> g_;      // size distSize_, idc-major
  lbmini::DeviceBuffer<Scalar> faux_;   // size distSize_  (swap partner of f_)
  lbmini::DeviceBuffer<Scalar> gaux_;   // size distSize_  (swap partner of g_)
  lbmini::DeviceBuffer<Scalar> lastGx_; // size N_ * (Dim + 1)   (Newton warm-start)

  // Ping-pong aliases: swapped each step in Run() without moving data.
  Scalar* fCur_ = nullptr; // points to active distribution (f_ or faux_)
  Scalar* fAlt_ = nullptr; // points to alternate buffer   (faux_ or f_)
  Scalar* gCur_ = nullptr;
  Scalar* gAlt_ = nullptr;

  // Precomputed streaming tables (static across time steps).
  // For each (cell, idc) we store up to 4 source cell indices + 4 weights.
  // The tables are allocated in [idc][cell][k] order (k=0..3) so the streaming
  // kernel accesses them coalesced across cells for fixed idc.
  lbmini::DeviceBuffer<std::int32_t> streamIdx_; // size distSize_ * 4
  lbmini::DeviceBuffer<Scalar> streamW_;         // size distSize_ * 4
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
  // kernels still index, and a cell count beyond int32 wraps the streaming
  // tables into negative (wild) indices.
  ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "openmp::cpu::LbmTube");

  // Thread count and affinity (`omp_set_num_threads`, `OMP_PLACES`,
  // `OMP_PROC_BIND`) are process-global state and are configured once by
  // `lbmini::Executor`. Setting them here would reconfigure every solver that
  // is already live in the process and would race with any other thread
  // reading the environment.

  nx_ = mesh.size[0];
  ny_ = mesh.size[1];
  N_ = nx_ * ny_;
  uSize_ = N_ * kDim_;
  distSize_ = N_ * kQ_;

  rho_.assign(N_, Scalar{ 0 });
  p_.assign(N_, Scalar{ 0 });
  tem_.assign(N_, Scalar{ 0 });
  u_.assign(uSize_, Scalar{ 0 });

  f_.assign(distSize_, Scalar{ 0 });
  g_.assign(distSize_, Scalar{ 0 });
  faux_.assign(distSize_, Scalar{ 0 });
  gaux_.assign(distSize_, Scalar{ 0 });

  lastGx_.assign(N_ * (kDim_ + 1), Scalar{ 0 });

  fCur_ = f_.data();
  fAlt_ = faux_.data();
  gCur_ = g_.data();
  gAlt_ = gaux_.data();

  buildStreamingTables();
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
void LbmTube<Scalar, LatticeType>::Init() {
  // Initialize macroscopic fields with the Sod-like left/right state.
  // NUMA first-touch: the same `schedule(static)` partition also binds
  // each thread's pages to its local NUMA node so the hot kernels later
  // find their data on the right side of the interconnect.
  #pragma omp parallel for schedule(static)
  for (Index i = 0; i < nx_; ++i) {
    for (Index j = 0; j < ny_; ++j) {
      const Index cell = cellIndex(i, j);
      u_[0 * N_ + cell] = Scalar{ 0 };
      u_[1 * N_ + cell] = Scalar{ 0 };
      if (i < nx_ / 2) {
        rho_[cell] = kFluid_.densityL;
        p_[cell] = kFluid_.pressureL;
      } else {
        rho_[cell] = kFluid_.densityR;
        p_[cell] = kFluid_.pressureR;
      }
      tem_[cell] = LatticeType::Cs2 * p_[cell] / (rho_[cell] * kFluid_.constant);
      for (Index d = 0; d < kDim_ + 1; ++d)
        lastGx_[cell * (kDim_ + 1) + d] = Scalar{ 0 };
    }
  }

  // Seed f = feq, g = geq. The fused collision kernel is idempotent when
  // f == feq and g == geq (omegaL*(feq-f) == 0), so calling it twice gives
  // the same result as a dedicated "compute equilibria only" path while
  // avoiding duplicating ~200 lines of NR logic.
  // First call: f/g are zero -> writes relaxed BGK = feq (since omegaL*(feq-0)+0 = omegaL*feq,
  // which is NOT exactly feq). So we do a plain feq seed instead:
  seedEquilibria();
  computeMacroscopic();
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Step(const bool save) {
  Run(1, save);
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Run(const Index steps, bool /*save*/) {
  #pragma omp parallel default(shared)
  {
    for (Index t = 0; t < steps; ++t) {
      collisionAndEquilibria();
      streamAndMacroscopic();
      #pragma omp single
      {
        std::swap(fCur_, fAlt_);
        std::swap(gCur_, gAlt_);
      }
      // implicit barrier after `single` synchronises all threads before the
      // next collisionAndEquilibria reads the freshly swapped fCur_/gCur_.
    }
  }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::computeMacroscopic() {
  const Scalar Ushift[2] = { kControl_.U(0), kControl_.U(1) };
  const Scalar invCv = Scalar{ 1 } / kFluid_.specificHeatCv;
  const Scalar Rgas = kFluid_.constant;
  const Scalar* __restrict__ pF = fCur_;
  const Scalar* __restrict__ pG = gCur_;
  Scalar* __restrict__ pRho = rho_.data();
  Scalar* __restrict__ pP = p_.data();
  Scalar* __restrict__ pT = tem_.data();
  Scalar* __restrict__ pU = u_.data();

  // `for simd` (not `parallel for`): callable from inside a hoisted parallel
  // region without nested parallelism. The caller (`Run()`) provides the
  // surrounding `#pragma omp parallel`.
  // No `nowait`: the implicit barrier at the end of `omp for` protects the
  // downstream kernel (next time-step's collisionAndEquilibria) from reading
  // rho_/u_/tem_ before all threads have finished writing them.
  #pragma omp for simd schedule(static)
  for (Index cell = 0; cell < N_; ++cell) {
    Scalar rho = Scalar{ 0 };
    Scalar nrg = Scalar{ 0 };
    Scalar mom[kDim_] = { Scalar{ 0 }, Scalar{ 0 } };

    for (Index idc = 0; idc < kQ_; ++idc) {
      const Scalar fi = pF[idc * N_ + cell];
      const Scalar gi = pG[idc * N_ + cell];
      rho += fi;
      nrg += gi;
      mom[0] += LatticeType::Cshift(idc, 0, Ushift[0]) * fi;
      mom[1] += LatticeType::Cshift(idc, 1, Ushift[1]) * fi;
    }

    pRho[cell] = rho;

    const Scalar invRho = Scalar{ 1 } / rho;
    const Scalar ux = mom[0] * invRho;
    const Scalar uy = mom[1] * invRho;
    pU[0 * N_ + cell] = ux;
    pU[1 * N_ + cell] = uy;

    const Scalar kin = Scalar{ 0.5 } * (ux * ux + uy * uy);
    const Scalar T = (Scalar{ 0.5 } * nrg * invRho - kin) * invCv;
    pT[cell] = T;
    pP[cell] = Rgas * rho * T;
  }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::seedEquilibria() {
  // Compute feq / geq directly from (rho, u, T) and write into f_ / g_ .
  // Used only from Init() where u == 0, so NR solution is trivial (xi == 0)
  // and geq_i = Wi(T) * (2*rho*E) / sum(Wi). Called outside the hoisted
  // `Run()` parallel region, so we fork our own.
  const Scalar Ushift[2] = { kControl_.U(0), kControl_.U(1) };
  Scalar* __restrict__ pF = f_.data();
  Scalar* __restrict__ pG = g_.data();
  const Scalar* __restrict__ pRho = rho_.data();
  const Scalar* __restrict__ pT = tem_.data();
  const Scalar* __restrict__ pU = u_.data();

  #pragma omp parallel for simd schedule(static)
  for (Index cell = 0; cell < N_; ++cell) {
    const Scalar rho = pRho[cell];
    const Scalar T = pT[cell];
    const Scalar ux = pU[0 * N_ + cell];
    const Scalar uy = pU[1 * N_ + cell];
    const Scalar u2 = ux * ux + uy * uy;

    // Feq (product form).
    for (Index idc = 0; idc < kQ_; ++idc) {
      Scalar phi = rho;
      for (Index d = 0; d < kDim_; ++d) {
        const int vi = LatticeType::Velocity(idc, d);
        const Scalar uia = (d == 0 ? ux : uy) - Ushift[d];
        const Scalar uia2t = uia * uia + T;
        const Scalar pf = (vi == 0)   ? (Scalar{ 1 } - uia2t)
                          : (vi == 1) ? (Scalar{ 0.5 } * (uia + uia2t))
                                      : (Scalar{ 0.5 } * (-uia + uia2t));
        phi *= pf;
      }
      pF[idc * N_ + cell] = phi;
    }

    // Geq: normalized-Wi distribution (xi == 0 -> e_i = Wi).
    const Scalar E = kFluid_.specificHeatCv * T + Scalar{ 0.5 } * u2;
    const Scalar targetE = Scalar{ 2 } * rho * E;
    Scalar sumW = Scalar{ 0 };
    for (Index idc = 0; idc < kQ_; ++idc)
      sumW += LatticeType::Weights(idc, T);
    const Scalar sc = (sumW > kTiny_) ? targetE / sumW : targetE / Scalar(kQ_);
    for (Index idc = 0; idc < kQ_; ++idc) {
      const Scalar wi = (sumW > kTiny_) ? LatticeType::Weights(idc, T) : Scalar{ 1 };
      pG[idc * N_ + cell] = wi * sc;
    }
  }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::collisionAndEquilibria() {
  static_assert(kDim_ == 2, "The tiled collision kernel is written for two-dimensional lattices.");

  // `for` (not `parallel for`): see `computeMacroscopic()`. The caller owns
  // the parallel region; we just schedule the tiles.
  //
  // Cells are processed in tiles of `kTile_`. Every stage below is a flat
  // `omp simd` loop over the lanes of one tile, with the `kQ_` directions
  // outermost. A single per-cell loop holding the nested direction and
  // Newton-Raphson loops is not vectorized by GCC at all, so its `exp`/`log`
  // stay scalar; here each of them sits in a plain loop that the compiler maps
  // onto the SIMD math library (libmvec under `-ffast-math`). The per-cell
  // arithmetic and its evaluation order are unchanged.
  const Scalar Ushift[2] = { kControl_.U(0), kControl_.U(1) };
  const Scalar cv = kFluid_.specificHeatCv;
  const Scalar cp = kFluid_.specificHeatCp;
  const Scalar mu = kFluid_.viscosity;
  const Scalar lambda = kFluid_.conductivity;
  constexpr Index maxIter = 3;

  Scalar* __restrict__ pF = fCur_;
  Scalar* __restrict__ pG = gCur_;
  Scalar* __restrict__ pLastGx = lastGx_.data();
  const Scalar* __restrict__ pRho = rho_.data();
  const Scalar* __restrict__ pT = tem_.data();
  const Scalar* __restrict__ pU = u_.data();

  // Precompute cshift table once (runtime constant).
  Scalar cshift[kQ_][kDim_];
  for (Index idc = 0; idc < kQ_; ++idc)
    for (Index d = 0; d < kDim_; ++d)
      cshift[idc][d] = LatticeType::Cshift(idc, d, Ushift[d]);

  using std::exp;
  using std::log;
  using std::fabs;

  // Factorized product form of the equilibrium, one lattice component.
  const auto eqFactor = [](const Index vi, const Scalar uia, const Scalar T) {
    const Scalar uia2t = uia * uia + T;
    return (vi == 0)   ? (Scalar{ 1 } - uia2t)
           : (vi == 1) ? (Scalar{ 0.5 } * (uia + uia2t))
                       : (Scalar{ 0.5 } * (-uia + uia2t)); // vi == -1
  };

  const Index nTiles = (N_ + kTile_ - 1) / kTile_;

  // No `nowait`: downstream `stream()` reads the freshly relaxed f_/g_ and
  // requires them fully written by all threads first.
  #pragma omp for schedule(static)
  for (Index tile = 0; tile < nTiles; ++tile) {
    const Index base = tile * kTile_;
    const Index n = std::min<Index>(kTile_, N_ - base);
    const Scalar* rho = pRho + base;
    const Scalar* tem = pT + base;
    const Scalar* ux = pU + base;
    const Scalar* uy = pU + N_ + base;

    Scalar feq[kQ_][kTile_];
    Scalar wgt[kQ_][kTile_];
    Scalar ew[kQ_][kTile_]; // weighted exponentials, then geq
    Scalar si[kQ_][kTile_];
    Scalar tE[kTile_], tM0[kTile_], tM1[kTile_];
    Scalar xi0[kTile_], xi1[kTile_];
    Scalar smax[kTile_], Z[kTile_];
    Scalar S1x[kTile_], S1y[kTile_], S2xx[kTile_], S2xy[kTile_], S2yy[kTile_];
    Scalar ok[kTile_]; // solverOk as 1/0: once degenerate, stays degenerate
    Scalar sumW[kTile_];

    // Targets and warm start.
    #pragma omp simd
    for (Index l = 0; l < n; ++l) {
      const Scalar r = rho[l];
      const Scalar T = tem[l];
      const Scalar vx = ux[l];
      const Scalar vy = uy[l];
      const Scalar u2 = vx * vx + vy * vy;
      const Scalar E = cv * T + Scalar{ 0.5 } * u2;
      const Scalar targetE = Scalar{ 2 } * r * E;
      tE[l] = targetE;
      tM0[l] = Scalar{ 2 } * r * vx * (E + T) / targetE;
      tM1[l] = Scalar{ 2 } * r * vy * (E + T) / targetE;
      ok[l] = Scalar{ 1 };
      sumW[l] = Scalar{ 0 };
    }
    // Strided in the cell index, so kept out of the simd loops.
    for (Index l = 0; l < n; ++l) {
      xi0[l] = pLastGx[(base + l) * (kDim_ + 1) + 1];
      xi1[l] = pLastGx[(base + l) * (kDim_ + 1) + 2];
    }

    for (Index idc = 0; idc < kQ_; ++idc) {
      #pragma omp simd
      for (Index l = 0; l < n; ++l) {
        const Scalar w = LatticeType::Weights(idc, tem[l]);
        wgt[idc][l] = w;
        sumW[l] += w;
      }
    }

    // Feq (Eq. factorized product form)
    for (Index idc = 0; idc < kQ_; ++idc) {
      const Index vx = LatticeType::Velocity(idc, 0);
      const Index vy = LatticeType::Velocity(idc, 1);
      #pragma omp simd
      for (Index l = 0; l < n; ++l)
        feq[idc][l] = rho[l] * eqFactor(vx, ux[l] - Ushift[0], tem[l])
                      * eqFactor(vy, uy[l] - Ushift[1], tem[l]);
    }

    // Geq via Newton-Raphson on normalized energy flux.
    // Fixed-iteration Newton-Raphson: no early exit, no data-dependent branch
    // on Z or detJ. Z and detJ are clamped to `kTiny_` magnitude to keep the
    // reciprocals finite; `ok` records whether any iteration degenerated and
    // drives the fallback selection at geq assembly time. Evaluated on exit
    // (no mid-loop break) so control flow is uniform across a warp / SIMD lane
    // set — the GPU-critical property.
    Scalar alpha = Scalar{ 1 };
    for (Index iter = 0; iter < maxIter; ++iter) {
      #pragma omp simd
      for (Index l = 0; l < n; ++l) {
        smax[l] = -kMaxExp_;
        Z[l] = Scalar{ 0 };
        S1x[l] = Scalar{ 0 };
        S1y[l] = Scalar{ 0 };
        S2xx[l] = Scalar{ 0 };
        S2xy[l] = Scalar{ 0 };
        S2yy[l] = Scalar{ 0 };
      }

      for (Index idc = 0; idc < kQ_; ++idc) {
        const Scalar cx = cshift[idc][0];
        const Scalar cy = cshift[idc][1];
        #pragma omp simd
        for (Index l = 0; l < n; ++l) {
          Scalar s = Scalar{ 0 };
          s += xi0[l] * cx;
          s += xi1[l] * cy;
          si[idc][l] = s;
          smax[l] = (s > smax[l]) ? s : smax[l];
        }
      }
      #pragma omp simd
      for (Index l = 0; l < n; ++l)
        smax[l] = (smax[l] < kMaxExp_) ? smax[l] : kMaxExp_;

      for (Index idc = 0; idc < kQ_; ++idc) {
        const Scalar cx = cshift[idc][0];
        const Scalar cy = cshift[idc][1];
        const Scalar cxx = cx * cx;
        const Scalar cxy = cx * cy;
        const Scalar cyy = cy * cy;
        #pragma omp simd
        for (Index l = 0; l < n; ++l) {
          Scalar expo = si[idc][l] - smax[l];
          expo = (expo < kMaxExp_) ? expo : kMaxExp_;
          expo = (expo > -kMaxExp_) ? expo : -kMaxExp_;
          const Scalar wev = wgt[idc][l] * exp(expo);
          Z[l] += wev;
          S1x[l] += cx * wev;
          S1y[l] += cy * wev;
          S2xx[l] += cxx * wev;
          S2xy[l] += cxy * wev;
          S2yy[l] += cyy * wev;
        }
      }

      #pragma omp simd
      for (Index l = 0; l < n; ++l) {
        // Clamp Z magnitude (branchless) -> safe reciprocal.
        const Scalar Zl = Z[l];
        const Scalar Zsafe = (Zl > kTiny_) ? Zl : kTiny_;
        bool good = (ok[l] > Scalar{ 0 }) && (Zl > kTiny_);
        const Scalar invZ = Scalar{ 1 } / Zsafe;

        // J is symmetric: S2 and S1 (x) S1 are.
        // Branchless ridge on the diagonal (was: if (... <= kTiny_) {...}).
        const Scalar J00 = (S2xx[l] - S1x[l] * S1x[l] * invZ) * invZ + kTiny_;
        const Scalar J01 = (S2xy[l] - S1x[l] * S1y[l] * invZ) * invZ;
        const Scalar J11 = (S2yy[l] - S1y[l] * S1y[l] * invZ) * invZ + kTiny_;

        const Scalar detJ = J00 * J11 - J01 * J01;
        // Clamp |detJ| magnitude -> safe reciprocal, preserving sign.
        const Scalar absDet = fabs(detJ);
        const Scalar detSafe = (absDet > kTiny_)
                                 ? detJ
                                 : ((detJ >= Scalar{ 0 }) ? kTiny_ : -kTiny_);
        good = good && (absDet > kTiny_);
        ok[l] = good ? Scalar{ 1 } : Scalar{ 0 };
        const Scalar invDet = Scalar{ 1 } / detSafe;

        const Scalar r0 = S1x[l] * invZ - tM0[l];
        const Scalar r1 = S1y[l] * invZ - tM1[l];
        xi0[l] -= alpha * (J11 * r0 - J01 * r1) * invDet;
        xi1[l] -= alpha * (-J01 * r0 + J00 * r1) * invDet;
      }
      alpha *= Scalar{ 0.5 };
    }
    // Final equilibrium. Both candidates are evaluated for every cell and one
    // of them is selected, so the tile loops have no data-dependent branch and
    // map onto SIMD lanes: the Newton-Raphson result while the solver stayed
    // healthy, the normalized-Wi distribution (uniform if the weights vanish)
    // otherwise, with the warm start reset in the latter case.
    #pragma omp simd
    for (Index l = 0; l < n; ++l) {
      smax[l] = -kMaxExp_;
      Z[l] = Scalar{ 0 };
    }
    for (Index idc = 0; idc < kQ_; ++idc) {
      const Scalar cx = cshift[idc][0];
      const Scalar cy = cshift[idc][1];
      #pragma omp simd
      for (Index l = 0; l < n; ++l) {
        Scalar s = Scalar{ 0 };
        s += xi0[l] * cx;
        s += xi1[l] * cy;
        si[idc][l] = s;
        smax[l] = (s > smax[l]) ? s : smax[l];
      }
    }
    #pragma omp simd
    for (Index l = 0; l < n; ++l)
      smax[l] = (smax[l] < kMaxExp_) ? smax[l] : kMaxExp_;
    for (Index idc = 0; idc < kQ_; ++idc) {
      #pragma omp simd
      for (Index l = 0; l < n; ++l) {
        Scalar expo = si[idc][l] - smax[l];
        expo = (expo < kMaxExp_) ? expo : kMaxExp_;
        expo = (expo > -kMaxExp_) ? expo : -kMaxExp_;
        const Scalar wev = wgt[idc][l] * exp(expo);
        ew[idc][l] = wev;
        Z[l] += wev;
      }
    }

    Scalar scale[kTile_], sc[kTile_], uni[kTile_], vanish[kTile_], chi[kTile_];
    #pragma omp simd
    for (Index l = 0; l < n; ++l) {
      scale[l] = tE[l] / Z[l];
      const bool weightsVanish = sumW[l] <= kTiny_;
      sc[l] = tE[l] / (weightsVanish ? Scalar{ 1 } : sumW[l]);
      uni[l] = tE[l] / Scalar(kQ_);
      vanish[l] = weightsVanish ? Scalar{ 1 } : Scalar{ 0 };
      chi[l] = log(scale[l] / rho[l]) - smax[l];
    }

    // geq, overwriting the weighted exponentials in place.
    for (Index idc = 0; idc < kQ_; ++idc) {
      #pragma omp simd
      for (Index l = 0; l < n; ++l) {
        const Scalar fallback = (vanish[l] > Scalar{ 0 }) ? uni[l] : wgt[idc][l] * sc[l];
        ew[idc][l] = (ok[l] > Scalar{ 0 }) ? scale[l] * ew[idc][l] : fallback;
      }
    }

    // Strided in the cell index, so kept out of the simd loops.
    for (Index l = 0; l < n; ++l) {
      const bool good = ok[l] > Scalar{ 0 };
      Scalar* gx = pLastGx + (base + l) * (kDim_ + 1);
      gx[0] = good ? chi[l] : Scalar{ 0 };
      gx[1] = good ? xi0[l] : Scalar{ 0 };
      gx[2] = good ? xi1[l] : Scalar{ 0 };
    }

    // Collision (BGK + Knudsen sensor + thermal-relaxation correction)
    Scalar epsAcc[kTile_], L0[kTile_], L1[kTile_];
    Scalar omegaL[kTile_], omegaT[kTile_], invT[kTile_];
    #pragma omp simd
    for (Index l = 0; l < n; ++l) {
      epsAcc[l] = Scalar{ 0 };
      L0[l] = Scalar{ 0 };
      L1[l] = Scalar{ 0 };
    }
    for (Index idc = 0; idc < kQ_; ++idc) {
      const Scalar cix0 = static_cast<Scalar>(LatticeType::Velocity(idc, 0));
      const Scalar ciy0 = static_cast<Scalar>(LatticeType::Velocity(idc, 1));
      const Scalar* fRow = pF + idc * N_ + base;
      #pragma omp simd
      for (Index l = 0; l < n; ++l) {
        const Scalar d = fRow[l] - feq[idc][l];
        const Scalar den = (feq[idc][l] > kTiny_) ? feq[idc][l] : kTiny_;
        epsAcc[l] += fabs(d) / den;

        // L[a] = sum_i 2 * (u . c_i) * c_i_a * (f - feq)   (c_i unshifted)
        const Scalar uvi = ux[l] * cix0 + uy[l] * ciy0;
        const Scalar aux = Scalar{ 2 } * uvi * d;
        L0[l] += aux * cix0;
        L1[l] += aux * ciy0;
      }
    }

    #pragma omp simd
    for (Index l = 0; l < n; ++l) {
      const Scalar r = rho[l];
      const Scalar T = tem[l];
      const Scalar tau = mu / (r * T) + Scalar{ 0.5 };
      const Scalar omega = Scalar{ 1 } / tau;
      const Scalar diffusivity = lambda / (r * cp);
      const Scalar tauT = diffusivity / T + Scalar{ 0.5 };
      Scalar oT = Scalar{ 1 } / tauT;

      const Scalar eps = epsAcc[l] / Scalar(kQ_);
      const Scalar sigma = (eps >= Scalar{ 1 })      ? omega
                           : (eps >= Scalar{ 0.1 })  ? Scalar{ 1.35 }
                           : (eps >= Scalar{ 0.01 }) ? Scalar{ 1.05 }
                                                     : Scalar{ 1 };

      Scalar oL = omega / sigma;
      oL = (oL > Scalar{ 1 }) ? oL : Scalar{ 1 };
      oL = (oL < (Scalar{ 2 } - Scalar{ 1e-7 })) ? oL : (Scalar{ 2 } - Scalar{ 1e-7 });
      oT = (oT > Scalar{ 1 }) ? oT : Scalar{ 1 };
      oT = (oT < (Scalar{ 2 } - Scalar{ 1e-7 })) ? oT : (Scalar{ 2 } - Scalar{ 1e-7 });
      omegaL[l] = oL;
      omegaT[l] = oT;
      invT[l] = Scalar{ 1 } / T;
    }

    for (Index idc = 0; idc < kQ_; ++idc) {
      const Scalar cx = cshift[idc][0];
      const Scalar cy = cshift[idc][1];
      Scalar* fRow = pF + idc * N_ + base;
      Scalar* gRow = pG + idc * N_ + base;
      #pragma omp simd
      for (Index l = 0; l < n; ++l) {
        const Scalar fOld = fRow[l];
        fRow[l] = fOld + omegaL[l] * (feq[idc][l] - fOld);
        const Scalar cidotL = L0[l] * cx + L1[l] * cy;
        const Scalar gDiff = wgt[idc][l] * cidotL * invT[l];
        const Scalar gOld = gRow[l];
        gRow[l] = gOld + omegaL[l] * (ew[idc][l] - gOld) + (omegaL[l] - omegaT[l]) * gDiff;
      }
    }
  }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::streamAndMacroscopic() {
  // Stream fCur_/gCur_ into fAlt_/gAlt_ via precomputed IDW tables.
  stream();

  // Compute macroscopic from the just-streamed fAlt_/gAlt_.
  const Scalar Ushift[2] = { kControl_.U(0), kControl_.U(1) };
  const Scalar invCv = Scalar{ 1 } / kFluid_.specificHeatCv;
  const Scalar Rgas = kFluid_.constant;
  const Scalar* __restrict__ pF = fAlt_;
  const Scalar* __restrict__ pG = gAlt_;
  Scalar* __restrict__ pRho = rho_.data();
  Scalar* __restrict__ pP = p_.data();
  Scalar* __restrict__ pT = tem_.data();
  Scalar* __restrict__ pU = u_.data();

  #pragma omp for simd schedule(static)
  for (Index cell = 0; cell < N_; ++cell) {
    Scalar rho = Scalar{ 0 };
    Scalar nrg = Scalar{ 0 };
    Scalar mom[kDim_] = { Scalar{ 0 }, Scalar{ 0 } };

    for (Index idc = 0; idc < kQ_; ++idc) {
      const Scalar fi = pF[idc * N_ + cell];
      const Scalar gi = pG[idc * N_ + cell];
      rho += fi;
      nrg += gi;
      mom[0] += LatticeType::Cshift(idc, 0, Ushift[0]) * fi;
      mom[1] += LatticeType::Cshift(idc, 1, Ushift[1]) * fi;
    }

    const Scalar invRho = Scalar{ 1 } / rho;
    const Scalar ux = mom[0] * invRho;
    const Scalar uy = mom[1] * invRho;
    pU[0 * N_ + cell] = ux;
    pU[1 * N_ + cell] = uy;
    const Scalar kin = Scalar{ 0.5 } * (ux * ux + uy * uy);
    const Scalar T = (Scalar{ 0.5 } * nrg * invRho - kin) * invCv;
    pRho[cell] = rho;
    pT[cell] = T;
    pP[cell] = Rgas * rho * T;
  }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::buildStreamingTables() {
  streamIdx_.assign(distSize_ * 4, 0);
  streamW_.assign(distSize_ * 4, Scalar{ 0 });

  auto clampX = [&](const Index x) -> Index {
    if (x < 0)
      return 0;
    if (x >= nx_)
      return nx_ - 1;
    return x;
  };
  auto wrapY = [&](Index y) -> Index {
    y = y % ny_;
    if (y < 0)
      y += ny_;
    return y;
  };

  // One-off per-simulation cost — parallelise over the (idc, i) plane.
  #pragma omp parallel for collapse(2) schedule(static)
  for (Index idc = 0; idc < kQ_; ++idc) {
    for (Index i = 0; i < nx_; ++i) {
      const Scalar cix = LatticeType::Cshift(idc, 0, kControl_.U(0));
      const Scalar ciy = LatticeType::Cshift(idc, 1, kControl_.U(1));
      for (Index j = 0; j < ny_; ++j) {
        const Index cell = cellIndex(i, j);
        const Index base = (idc * N_ + cell) * 4;
        const Scalar xSrc = static_cast<Scalar>(i) - cix;
        const Scalar ySrc = static_cast<Scalar>(j) - ciy;

        const Index x0 = static_cast<Index>(std::floor(xSrc));
        const Index y0 = static_cast<Index>(std::floor(ySrc));
        const Index x1 = x0 + 1;
        const Index y1 = y0 + 1;
        const Index cx0 = clampX(x0), cx1 = clampX(x1);
        const Index wy0 = wrapY(y0), wy1 = wrapY(y1);

        const Index c00 = cx0 * ny_ + wy0;
        const Index c10 = cx1 * ny_ + wy0;
        const Index c01 = cx0 * ny_ + wy1;
        const Index c11 = cx1 * ny_ + wy1;


        // General bilinear-ish IDW gather (original semantic preserved).
        const Scalar dx00 = xSrc - static_cast<Scalar>(x0);
        const Scalar dy00 = ySrc - static_cast<Scalar>(y0);
        const Scalar dx10 = xSrc - static_cast<Scalar>(x1);
        const Scalar dy10 = dy00;
        const Scalar dx01 = dx00;
        const Scalar dy01 = ySrc - static_cast<Scalar>(y1);
        const Scalar dx11 = dx10;
        const Scalar dy11 = dy01;
        const Scalar d00 = std::sqrt(dx00 * dx00 + dy00 * dy00);
        const Scalar d10 = std::sqrt(dx10 * dx10 + dy10 * dy10);
        const Scalar d01 = std::sqrt(dx01 * dx01 + dy01 * dy01);
        const Scalar d11 = std::sqrt(dx11 * dx11 + dy11 * dy11);

        auto snap = [&](Scalar d, Index cAny) -> bool {
          if (d >= kTiny_)
            return false;
          streamIdx_[base + 0] = static_cast<std::int32_t>(cAny);
          streamIdx_[base + 1] = streamIdx_[base + 2] = streamIdx_[base + 3] = static_cast<std::int32_t>(cAny);
          streamW_[base + 0] = Scalar{ 1 };
          streamW_[base + 1] = streamW_[base + 2] = streamW_[base + 3] = Scalar{ 0 };
          return true;
        };
        if (snap(d00, c00))
          continue;
        if (snap(d10, c10))
          continue;
        if (snap(d01, c01))
          continue;
        if (snap(d11, c11))
          continue;

        const Scalar idw = kControl_.idw;
        const Scalar w00 = Scalar{ 1 } / std::pow(d00, idw);
        const Scalar w10 = Scalar{ 1 } / std::pow(d10, idw);
        const Scalar w01 = Scalar{ 1 } / std::pow(d01, idw);
        const Scalar w11 = Scalar{ 1 } / std::pow(d11, idw);
        const Scalar wsum = w00 + w10 + w01 + w11;

        if (!IsFinite(wsum) || wsum < kTiny_) {
          // Degenerate fallback: nearest neighbor, bounce-back on x out-of-range.
          Index xN = static_cast<Index>(std::round(xSrc));
          Index yN = static_cast<Index>(std::round(ySrc));
          Index cell2;
          if (xN < 0 || xN >= nx_) {
            // Bounce back: use *own* cell but opposite direction; streaming
            // kernel can't re-read opposite idc without a branch, so we fall
            // back to exact self-copy of the current cell.
            cell2 = cell;
          } else {
            cell2 = xN * ny_ + wrapY(yN);
          }
          streamIdx_[base + 0] = static_cast<std::int32_t>(cell2);
          streamIdx_[base + 1] = streamIdx_[base + 2] = streamIdx_[base + 3] = static_cast<std::int32_t>(cell2);
          streamW_[base + 0] = Scalar{ 1 };
          streamW_[base + 1] = streamW_[base + 2] = streamW_[base + 3] = Scalar{ 0 };
          continue;
        }
        const Scalar invSum = Scalar{ 1 } / wsum;
        streamIdx_[base + 0] = static_cast<std::int32_t>(c00);
        streamIdx_[base + 1] = static_cast<std::int32_t>(c10);
        streamIdx_[base + 2] = static_cast<std::int32_t>(c01);
        streamIdx_[base + 3] = static_cast<std::int32_t>(c11);
        streamW_[base + 0] = w00 * invSum;
        streamW_[base + 1] = w10 * invSum;
        streamW_[base + 2] = w01 * invSum;
        streamW_[base + 3] = w11 * invSum;
      }
    }
  }

  // One-off proof that the gather table only addresses existing cells. This is
  // what licenses `stream()` to dereference `streamIdx_` with no bounds check
  // whatsoever: the cost is a single linear pass at construction, against
  // thousands of time steps of unguarded, full-speed gathering.
  VerifyIndexTable(streamIdx_, N_, "openmp::cpu::LbmTube");
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::stream() {
  // Pure streaming: branch-free 4-way IDW gather from f_/g_ into faux_/gaux_.
  // Intentionally NOT fused with the macroscopic reduction — see the class
  // `stream()` docstring for the kernel-fission rationale (GPU occupancy).
  // `for simd` (not `parallel for`): the caller (`Run()`) owns the hoisted
  // parallel region. Every thread walks the planes in the same order and the
  // cells of each plane are shared out in contiguous, vectorised chunks.
  const Scalar* __restrict__ pF = fCur_;
  const Scalar* __restrict__ pG = gCur_;
  Scalar* __restrict__ pFaux = fAlt_;
  Scalar* __restrict__ pGaux = gAlt_;
  const std::int32_t* __restrict__ pSI = streamIdx_.data();
  const Scalar* __restrict__ pSW = streamW_.data();

  for (Index idc = 0; idc < kQ_; ++idc) {
    const Scalar* __restrict__ fplane = pF + idc * N_;
    const Scalar* __restrict__ gplane = pG + idc * N_;
    Scalar* __restrict__ fOut = pFaux + idc * N_;
    Scalar* __restrict__ gOut = pGaux + idc * N_;
    const std::int32_t* __restrict__ idxPlane = pSI + idc * N_ * 4;
    const Scalar* __restrict__ wPlane = pSW + idc * N_ * 4;

    // `nowait`: the planes are independent, so a thread moves on to the next
    // one without waiting for the others.
    #pragma omp for simd schedule(static) nowait
    for (Index cell = 0; cell < N_; ++cell) {
      const std::int32_t i0 = idxPlane[cell * 4 + 0];
      const std::int32_t i1 = idxPlane[cell * 4 + 1];
      const std::int32_t i2 = idxPlane[cell * 4 + 2];
      const std::int32_t i3 = idxPlane[cell * 4 + 3];
      const Scalar w0 = wPlane[cell * 4 + 0];
      const Scalar w1 = wPlane[cell * 4 + 1];
      const Scalar w2 = wPlane[cell * 4 + 2];
      const Scalar w3 = wPlane[cell * 4 + 3];
      fOut[cell] = w0 * fplane[i0] + w1 * fplane[i1] + w2 * fplane[i2] + w3 * fplane[i3];
      gOut[cell] = w0 * gplane[i0] + w1 * gplane[i1] + w2 * gplane[i2] + w3 * gplane[i3];
    }
  }

  // The macroscopic reduction that follows reads the `kQ_` planes of a cell,
  // which other threads may have written (OpenMP does not guarantee the same
  // iteration-to-thread assignment for SIMD loops), hence the barrier.
  #pragma omp barrier
}
} // namespace lbmini::openmp::cpu

#endif // LBMINI_OPENMP_CPU_LBMTUBE_HPP_
