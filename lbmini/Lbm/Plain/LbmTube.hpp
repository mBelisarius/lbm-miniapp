#ifndef LBMINI_PLAIN_LBMTUBE_HPP_
#define LBMINI_PLAIN_LBMTUBE_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILbmTube.hpp"

namespace lbmini::plain {
/**
 * @brief GPU-oriented, scalability-tuned reference implementation of the
 *        compressible LBM tube solver.
 *
 * Serves as the single-threaded reference on CPU, but its data layout,
 * control flow, and helper decorations are chosen so the same kernel bodies
 * port to CUDA / HIP / SYCL / OpenMP-target offload without a rewrite.
 *
 * Flat velocity-major SoA distributions are stored in
 * `lbmini::DeviceBuffer<Scalar>` (64-byte aligned, `std::vector`-compatible)
 * indexed as `idx(idc, cell) = idc*N + cell`, so a warp or SIMD lane set
 * loading a per-`idc` plane gets fully coalesced accesses. `DeviceBuffer<T>`
 * is the storage-policy seam that GPU backends specialise to
 * `cudaMalloc` / `hipMalloc` / `omp_target_alloc` without touching this
 * class.
 *
 * The per-step pipeline is intentionally split into three independent
 * per-cell kernels — `collisionAndEquilibria()`, `stream()`,
 * `computeMacroscopic()` — dispatched sequentially from `Run()`. Each
 * kernel is an embarrassingly parallel loop over cells and is the natural
 * unit of GPU-kernel launch. No persistent `feq`/`geq` buffers exist: both
 * live as `kQ_`-sized stack arrays inside `collisionAndEquilibria`, are
 * scalarised into registers by `LBMINI_UNROLL(kQ_)`, and the relaxed
 * distributions are written straight into `f_`/`g_`.
 *
 * @section design_future Possible future improvements
 *  - **A-A single-population scheme (Bailey/Geier)** — would halve the
 *    distribution footprint on both CPU cache and GPU HBM, but requires a
 *    bijection between source and destination cells. Our streaming is a
 *    convex 4-way IDW gather over the shifted lattice (weighted combination
 *    of up to 4 neighbours, not invertible), so A-A is incompatible with
 *    the current scheme without breaking mass/momentum/energy conservation.
 *  - **Ghost-cell padding** — the branch-free gather is currently emulated
 *    by `streamIdx_`/`streamW_`; a true ghost halo would remove the
 *    per-(cell, idc) index table at the cost of a small boundary kernel.
 *  - **Per-idc cell-major transpose** — beneficial only for very small
 *    `ny`; not worth the complexity on the current mesh sizes.
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

  /**
   * @name Macroscopic field getters
   * @brief Lazy, copy-out accessors for pressure / density / temperature /
   *        velocity.
   *
   * Macroscopic fields (`rho`, `p`, `tem`, `u`) are kept in a small, canonical
   * RowMajor Eigen layout so these getters can return by value cheaply — they
   * are 2-D / 3-D fields, not full distributions. The heavy `f_` / `g_`
   * distributions intentionally never leave the flat SoA buffer.
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
  // Flat-index helpers.
  //   Macro fields:        rho[i*ny + j]
  //   Velocity field:      u[d * N + i*ny + j]       (SoA, d-major)
  //   Distribution fields: f[idc * N + i*ny + j]     (SoA, idc-major)
  [[nodiscard]] Index cellIndex(const Index i, const Index j) const { return i * ny_ + j; }

  [[nodiscard]] Index distIndex(const Index idc, const Index cell) const { return idc * N_ + cell; }

  [[nodiscard]] Index uIndex(const Index d, const Index cell) const { return d * N_ + cell; }

  /**
   * @brief Precompute per-(cell, idc) streaming neighbor indices and weights.
   *
   * Streaming depends only on mesh geometry and the control shift, yet the
   * original code evaluated `floor`, clamps and IDW `pow`/`sqrt` per cell per
   * step. This one-off pass fills `streamIdx_` (four source cells) and
   * `streamW_` (four IDW weights) per (cell, idc), so `stream()` degenerates
   * to the branch-free gather described above — the only truly
   * parallel-friendly way to express shifted-lattice streaming on GPU.
   */
  void buildStreamingTables();

  /**
   * @name Per-step kernels
   * @brief Each one is an embarrassingly parallel loop over all cells and is
   *        the natural unit of GPU-kernel dispatch in a future port.
   * @{
   */
  /// Recompute `rho`, `u`, `p`, `tem` from the current `f_`, `g_` SoA buffers.
  /// Used only on `Init()`; inside the time loop, macroscopics are produced
  /// by the fused `streamAndMacroscopic()` pass for free.
  void computeMacroscopic();

  /**
   * @brief Seed `f_` and `g_` with the equilibrium distributions built from
   *        the current `rho_`, `u_`, `tem_` fields.
   *
   * Called once from `Init()`. Mirrors the Feq/Geq math of
   * `collisionAndEquilibria` without the BGK relaxation or Newton-Raphson
   * loop (with `u == 0` the NR solution is `xi == 0`, so `geq_i =
   * Wi(T) * targetE / sum(Wi)`).
   */
  void seedEquilibria();

  /**
   * @brief Fused equilibria + BGK collision + thermal correction.
   *
   * Single per-cell pass that evaluates `feq` and `geq`, applies BGK
   * relaxation, and writes the relaxed distributions straight into
   * `f_`/`g_`. All scratch (`feq[kQ_]`, `geq[kQ_]`, Newton-Raphson
   * workspaces `xi`, `residual`, `Jacobian`) is stack-local so each
   * thread owns it in registers / local memory.
   *
   * The inner Newton-Raphson solver runs a **fixed** number of iterations
   * (no early exit), and the reciprocals of `Z` and `detJ` are taken on
   * clamped magnitudes so control flow is warp-uniform. A post-loop
   * `solverOk` predicate selects between the NR result and the
   * `Wi(T)`-normalised fallback. All `idc` loops and the outer NR loop
   * are annotated with `LBMINI_UNROLL(N)` to scalarise the `kQ_`-sized
   * arrays into registers on both CPU and GPU.
   */
  void collisionAndEquilibria();

  /**
   * @brief Branch-free 4-way gather streaming into `faux_`/`gaux_`.
   *
   * Independent per-cell kernel, kept separate from `computeMacroscopic()`
   * to match the GPU-optimal dispatch granularity (one kernel per logical
   * stage, maximising occupancy and enabling asynchronous overlap on
   * streams). The trade-off is one extra DRAM read of `f_`/`g_` per step
   * on CPU.
   */
  void stream();

  /**
   * @brief Combined streaming and macroscopic computation step.
   *
   * In this reference implementation, this merely calls `stream()` followed by
   * `computeMacroscopic()` to unify the interface across all backends.
   */
  void streamAndMacroscopic();

  /** @} */

private:
  static constexpr Scalar kTiny_ = Scalar{ 1.0e-12 };
  static constexpr Scalar kMaxExp_ = Scalar{ 700.0 };

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
  for (Index t = 0; t < steps; ++t) {
    collisionAndEquilibria();
    stream();
    // O(1) buffer swap (no element-wise copy).
    f_.swap(faux_);
    g_.swap(gaux_);
    computeMacroscopic();
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

  for (Index idc = 0; idc < kQ_; ++idc) {
    const Scalar cix = LatticeType::Cshift(idc, 0, kControl_.U(0));
    const Scalar ciy = LatticeType::Cshift(idc, 1, kControl_.U(1));

    for (Index i = 0; i < nx_; ++i) {
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

        if (!std::isfinite(wsum) || wsum < kTiny_) {
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
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::computeMacroscopic() {
  const Scalar Ushift[2] = { kControl_.U(0), kControl_.U(1) };
  const Scalar invCv = Scalar{ 1 } / kFluid_.specificHeatCv;
  const Scalar Rgas = kFluid_.constant;
  const Scalar* __restrict__ pF = f_.data();
  const Scalar* __restrict__ pG = g_.data();
  Scalar* __restrict__ pRho = rho_.data();
  Scalar* __restrict__ pP = p_.data();
  Scalar* __restrict__ pT = tem_.data();
  Scalar* __restrict__ pU = u_.data();

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
void LbmTube<Scalar, LatticeType>::collisionAndEquilibria() {
  const Scalar Ushift[2] = { kControl_.U(0), kControl_.U(1) };
  constexpr Index maxIter = 3;

  Scalar* __restrict__ pF = f_.data();
  Scalar* __restrict__ pG = g_.data();
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

  for (Index cell = 0; cell < N_; ++cell) {
    const Scalar rho = pRho[cell];
    const Scalar T = pT[cell];
    const Scalar ux = pU[0 * N_ + cell];
    const Scalar uy = pU[1 * N_ + cell];
    const Scalar u2 = ux * ux + uy * uy;

    // Feq (Eq. factorized product form)
    Scalar feqLocal[kQ_];
    for (Index idc = 0; idc < kQ_; ++idc) {
      Scalar phi = rho;
      for (Index d = 0; d < kDim_; ++d) {
        const int vi = LatticeType::Velocity(idc, d);
        const Scalar uia = (d == 0 ? ux : uy) - Ushift[d];
        const Scalar uia2t = uia * uia + T;
        Scalar pf;
        if (vi == 0)
          pf = Scalar{ 1 } - uia2t;
        else if (vi == 1)
          pf = Scalar{ 0.5 } * (uia + uia2t);
        else /* vi == -1*/
          pf = Scalar{ 0.5 } * (-uia + uia2t);
        phi *= pf;
      }
      feqLocal[idc] = phi;
    }

    // Geq via Newton-Raphson on normalized energy flux
    const Scalar E = kFluid_.specificHeatCv * T + Scalar{ 0.5 } * u2;
    const Scalar targetE = Scalar{ 2 } * rho * E;
    Scalar targetM[kDim_];
    for (Index d = 0; d < kDim_; ++d)
      targetM[d] = Scalar{ 2 } * rho * (d == 0 ? ux : uy) * (E + T) / targetE;

    Scalar xi[kDim_];
    for (Index d = 0; d < kDim_; ++d)
      xi[d] = pLastGx[cell * (kDim_ + 1) + (d + 1)];

    Scalar alpha = Scalar{ 1 };
    Scalar si[kQ_];
    Scalar e[kQ_];
    Scalar Z = Scalar{ 0 };
    Scalar smax = Scalar{ 0 };
    // Branchless solverOk accumulator: once degenerate, stays degenerate.
    // Evaluated on exit (no mid-loop break) so control flow is uniform across
    // a warp / SIMD lane set — the GPU-critical property.
    bool solverOk = true;

    // Fixed-iteration Newton-Raphson: no early exit, no data-dependent branch
    // on Z or detJ. Z and detJ are clamped to `kTiny_` magnitude to keep the
    // reciprocals finite; the `solverOk` flag is updated after the loop and
    // drives the fallback selection at geq assembly time.
    for (Index iter = 0; iter < maxIter; ++iter) {
      smax = -kMaxExp_;
      for (Index idc = 0; idc < kQ_; ++idc) {
        Scalar s = Scalar{ 0 };
        for (Index d = 0; d < kDim_; ++d)
          s += xi[d] * cshift[idc][d];
        si[idc] = s;
        smax = (s > smax) ? s : smax;
      }
      smax = (smax < kMaxExp_) ? smax : kMaxExp_;

      Scalar S1[kDim_] = { Scalar{ 0 }, Scalar{ 0 } };
      Scalar S2[kDim_][kDim_] = { { Scalar{ 0 }, Scalar{ 0 } }, { Scalar{ 0 }, Scalar{ 0 } } };
      Z = Scalar{ 0 };
      for (Index idc = 0; idc < kQ_; ++idc) {
        Scalar expo = si[idc] - smax;
        expo = (expo < kMaxExp_) ? expo : kMaxExp_;
        expo = (expo > -kMaxExp_) ? expo : -kMaxExp_;
        const Scalar wev = LatticeType::Weights(idc, T) * exp(expo);
        e[idc] = wev;
        Z += wev;
        for (Index a = 0; a < kDim_; ++a) {
          S1[a] += cshift[idc][a] * wev;
          for (Index b = 0; b < kDim_; ++b)
            S2[a][b] += cshift[idc][a] * cshift[idc][b] * wev;
        }
      }

      // Clamp Z magnitude (branchless) -> safe reciprocal.
      const Scalar Zsafe = (Z > kTiny_) ? Z : kTiny_;
      solverOk = solverOk && (Z > kTiny_);
      const Scalar invZ = Scalar{ 1 } / Zsafe;

      Scalar J[kDim_][kDim_];
      for (Index a = 0; a < kDim_; ++a)
        for (Index b = 0; b < kDim_; ++b)
          J[a][b] = (S2[a][b] - S1[a] * S1[b] * invZ) * invZ;
      // Branchless ridge on the diagonal (was: if (... <= kTiny_) {...}).
      J[0][0] += kTiny_;
      J[1][1] += kTiny_;

      const Scalar detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
      // Clamp |detJ| magnitude -> safe reciprocal, preserving sign.
      const Scalar absDet = fabs(detJ);
      const Scalar detSafe = (absDet > kTiny_)
                               ? detJ
                               : ((detJ >= Scalar{ 0 }) ? kTiny_ : -kTiny_);
      solverOk = solverOk && (absDet > kTiny_);
      const Scalar invDet = Scalar{ 1 } / detSafe;

      const Scalar r0 = S1[0] * invZ - targetM[0];
      const Scalar r1 = S1[1] * invZ - targetM[1];
      xi[0] -= alpha * (J[1][1] * r0 - J[0][1] * r1) * invDet;
      xi[1] -= alpha * (-J[1][0] * r0 + J[0][0] * r1) * invDet;
      alpha *= Scalar{ 0.5 };
    }

    Scalar geqLocal[kQ_];
    if (solverOk) {
      // Recompute e[], Z with final xi.
      smax = -kMaxExp_;
      for (Index idc = 0; idc < kQ_; ++idc) {
        Scalar s = Scalar{ 0 };
        for (Index d = 0; d < kDim_; ++d)
          s += xi[d] * cshift[idc][d];
        si[idc] = s;
        smax = (s > smax) ? s : smax;
      }
      smax = (smax < kMaxExp_) ? smax : kMaxExp_;
      Z = Scalar{ 0 };
      for (Index idc = 0; idc < kQ_; ++idc) {
        Scalar expo = si[idc] - smax;
        expo = (expo < kMaxExp_) ? expo : kMaxExp_;
        expo = (expo > -kMaxExp_) ? expo : -kMaxExp_;
        e[idc] = LatticeType::Weights(idc, T) * exp(expo);
        Z += e[idc];
      }
      const Scalar scale = targetE / Z;
      for (Index idc = 0; idc < kQ_; ++idc) {
        geqLocal[idc] = scale * e[idc];
      }
      pLastGx[cell * (kDim_ + 1) + 0] = log(scale / rho) - smax;
      for (Index d = 0; d < kDim_; ++d)
        pLastGx[cell * (kDim_ + 1) + (d + 1)] = xi[d];
    } else {
      // Fallback: normalized-Wi distribution.
      for (Index d = 0; d < kDim_ + 1; ++d)
        pLastGx[cell * (kDim_ + 1) + d] = Scalar{ 0 };
      Scalar sumW = Scalar{ 0 };
      for (Index idc = 0; idc < kQ_; ++idc)
        sumW += LatticeType::Weights(idc, T);
      if (sumW <= kTiny_) {
        const Scalar uni = targetE / Scalar(kQ_);
        for (Index idc = 0; idc < kQ_; ++idc) { geqLocal[idc] = uni; }
      } else {
        const Scalar sc = targetE / sumW;
        for (Index idc = 0; idc < kQ_; ++idc) {
          geqLocal[idc] = LatticeType::Weights(idc, T) * sc;
        }
      }
    }

    // Collision (BGK + Knudsen sensor + thermal-relaxation correction)
    const Scalar tau = kFluid_.viscosity / (rho * T) + Scalar{ 0.5 };
    const Scalar omega = Scalar{ 1 } / tau;
    const Scalar diffusivity = kFluid_.conductivity / (rho * kFluid_.specificHeatCp);
    const Scalar tauT = diffusivity / T + Scalar{ 0.5 };
    Scalar omegaT = Scalar{ 1 } / tauT;

    Scalar eps = Scalar{ 0 };
    for (Index idc = 0; idc < kQ_; ++idc) {
      const Scalar fi = pF[idc * N_ + cell];
      const Scalar d = fi - feqLocal[idc];
      const Scalar den = (feqLocal[idc] > kTiny_) ? feqLocal[idc] : kTiny_;
      eps += fabs(d) / den;
    }
    eps /= Scalar(kQ_);
    Scalar sigma = Scalar{ 1 };
    if (eps >= Scalar{ 1 })
      sigma = omega;
    else if (eps >= Scalar{ 0.1 })
      sigma = Scalar{ 1.35 };
    else
      if (eps >= Scalar{ 0.01 })
        sigma = Scalar{ 1.05 };

    Scalar omegaL = omega / sigma;
    omegaL = (omegaL > Scalar{ 1 }) ? omegaL : Scalar{ 1 };
    omegaL = (omegaL < (Scalar{ 2 } - Scalar{ 1e-7 })) ? omegaL : (Scalar{ 2 } - Scalar{ 1e-7 });
    omegaT = (omegaT > Scalar{ 1 }) ? omegaT : Scalar{ 1 };
    omegaT = (omegaT < (Scalar{ 2 } - Scalar{ 1e-7 })) ? omegaT : (Scalar{ 2 } - Scalar{ 1e-7 });

    // L[a] = sum_i 2 * (u . c_i) * c_i_a * (f - feq)   (c_i unshifted)
    Scalar L[kDim_] = { Scalar{ 0 }, Scalar{ 0 } };
    for (Index idc = 0; idc < kQ_; ++idc) {
      const Scalar fi = pF[idc * N_ + cell];
      const Scalar cix0 = static_cast<Scalar>(LatticeType::Velocity(idc, 0));
      const Scalar ciy0 = static_cast<Scalar>(LatticeType::Velocity(idc, 1));
      const Scalar uvi = ux * cix0 + uy * ciy0;
      const Scalar aux = Scalar{ 2 } * uvi * (fi - feqLocal[idc]);
      L[0] += aux * cix0;
      L[1] += aux * ciy0;
    }

    const Scalar invT = Scalar{ 1 } / T;
    for (Index idc = 0; idc < kQ_; ++idc) {
      const Index fi = idc * N_ + cell;
      const Scalar fOld = pF[fi];
      pF[fi] = fOld + omegaL * (feqLocal[idc] - fOld);
      const Scalar cidotL = L[0] * cshift[idc][0] + L[1] * cshift[idc][1];
      const Scalar Wi = LatticeType::Weights(idc, T);
      const Scalar gDiff = Wi * cidotL * invT;
      const Scalar gOld = pG[fi];
      pG[fi] = gOld + omegaL * (geqLocal[idc] - gOld) + (omegaL - omegaT) * gDiff;
    }
  }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::seedEquilibria() {
  // Compute feq / geq directly from (rho, u, T) and write into f_ / g_ .
  // Used only from Init() where u == 0, so NR solution is trivial (xi == 0)
  // and geq_i = Wi(T) * (2*rho*E) / sum(Wi).
  const Scalar Ushift[2] = { kControl_.U(0), kControl_.U(1) };
  Scalar* __restrict__ pF = f_.data();
  Scalar* __restrict__ pG = g_.data();
  const Scalar* __restrict__ pRho = rho_.data();
  const Scalar* __restrict__ pT = tem_.data();
  const Scalar* __restrict__ pU = u_.data();

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
        Scalar pf;
        if (vi == 0)
          pf = Scalar{ 1 } - uia2t;
        else if (vi == 1)
          pf = Scalar{ 0.5 } * (uia + uia2t);
        else
          pf = Scalar{ 0.5 } * (-uia + uia2t);
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
void LbmTube<Scalar, LatticeType>::stream() {
  // Pure streaming: branch-free 4-way IDW gather from f_/g_ into faux_/gaux_.
  // Intentionally NOT fused with the macroscopic reduction — see the class
  // `stream()` docstring for the kernel-fission rationale (GPU occupancy).
  const Scalar* __restrict__ pF = f_.data();
  const Scalar* __restrict__ pG = g_.data();
  Scalar* __restrict__ pFaux = faux_.data();
  Scalar* __restrict__ pGaux = gaux_.data();
  const std::int32_t* __restrict__ pSI = streamIdx_.data();
  const Scalar* __restrict__ pSW = streamW_.data();

  for (Index idc = 0; idc < kQ_; ++idc) {
    const Scalar* __restrict__ fplane = pF + idc * N_;
    const Scalar* __restrict__ gplane = pG + idc * N_;
    Scalar* __restrict__ fOut = pFaux + idc * N_;
    Scalar* __restrict__ gOut = pGaux + idc * N_;
    const std::int32_t* __restrict__ idxPlane = pSI + idc * N_ * 4;
    const Scalar* __restrict__ wPlane = pSW + idc * N_ * 4;

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
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::streamAndMacroscopic() {
  stream();
  computeMacroscopic();
}
} // namespace lbmini::plain

#endif // LBMINI_PLAIN_LBMTUBE_HPP_
