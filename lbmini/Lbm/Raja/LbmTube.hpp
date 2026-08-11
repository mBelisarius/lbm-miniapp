#ifndef LBMINI_RAJA_LBMTUBE_HPP_
#define LBMINI_RAJA_LBMTUBE_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include <RAJA/RAJA.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/Raja/Backends.hpp"

namespace lbmini::raja {
/**
 * @brief RAJA variant of the compressible LBM tube solver.
 *
 * Numerically identical to `lbmini::plain::LbmTube`, but every per-cell loop is
 * a `RAJA::forall` and every buffer lives in the memory space selected by the
 * backend tag. The class is templated on that tag, so the *same* source
 * produces the CPU solver (`HostBackend` — OpenMP or sequential) and the GPU
 * solver (`DeviceBackend` — CUDA or HIP) with no `#ifdef` in the numerics and
 * no second code path. That single-source portability is precisely the property
 * this backend is benchmarked for.
 *
 * ### RAJA design decisions
 *
 * **Loops are abstracted, memory is not.** RAJA provides no container: it
 * parallelises loop *bodies* and leaves data placement to the application
 * (raw pointers, Umpire or CHAI). The solver therefore keeps the flat raw
 * pointers of the CUDA backend and routes every allocation and transfer through
 * `Backend::Memory` (see `Backends.hpp`), which is the one place where a host
 * build differs from a device build. Contrast the Kokkos backend, where
 * `Kokkos::View` is reference counted and needs no destructor at all — here the
 * nine buffers are released explicitly in `~LbmTube()`.
 *
 * **`[=] RAJA_HOST_DEVICE` closures, never `this`.** All kernel state is hoisted
 * into local variables before each `RAJA::forall`, so the closure captures only
 * scalars and raw pointers. Capturing `this` would dereference a host object
 * from device code; hoisting also keeps the CPU version free of one indirection
 * per access.
 *
 * **1-D flat SoA layout.** Distributions keep the `idc`-major flat indexing
 * `f[idc * N + cell]` of the reference backends so the numerics, the streaming
 * tables and the macroscopic reductions are bit-for-bit comparable across
 * frameworks; `cell` remains the fastest-running index, which is what makes the
 * device loads coalesce.
 *
 * **Precomputed streaming tables.** As in the plain, OpenMP and Kokkos
 * backends, the shifted-lattice gather is reduced to four source cells plus
 * four inverse-distance weights per `(cell, idc)`, built once on the host and
 * copied to the device. The table is laid out as `(idc * 4 + k) * N + cell` —
 * `cell` fastest — so all four gathers stay perfectly coalesced across a warp,
 * unlike the `(idc * N + cell) * 4 + k` interleaving used by the CPU backends.
 *
 * **Two kernels per time step.** `collisionAndEquilibria()` is followed by the
 * fused `streamAndMacroscopic()`: the streamed distributions are still in
 * registers when the macroscopic reduction consumes them, which removes one
 * full read of `f`/`g` per step. Consecutive `RAJA::forall` calls on the same
 * (default) resource are ordered, so no synchronisation is needed between them;
 * `Run()` synchronises once at the end so the benchmark timer in `Executor`
 * measures completed device work rather than an enqueue.
 */
template<typename Scalar, typename LatticeType, typename Backend = DeviceBackend>
class LbmTube : public ILbmTube<Scalar, LatticeType> {
public:
  using Index = Eigen::Index;

  template<typename Type, Index NumIndices>
  using Tensor = Eigen::Tensor<Type, NumIndices, Eigen::RowMajor>;

  static constexpr Index kDim_ = LatticeType::Dim();
  static constexpr Index kQ_ = LatticeType::Speeds();

  using ExecPolicy = typename Backend::ExecPolicy;
  using Memory = typename Backend::Memory;
  using RangeSegment = RAJA::TypedRangeSegment<Index>;

  LbmTube(
    const FluidData<Scalar>& fluid,
    const MeshData<Scalar, kDim_>& mesh,
    const ControlData<Scalar>& control,
    const PerformanceData& performance
  );

  LbmTube(const LbmTube&) = delete;

  LbmTube& operator=(const LbmTube&) = delete;

  /// Releases every buffer obtained from `Backend::Memory`.
  ~LbmTube() override;

  /**
   * @name Macroscopic field getters
   * @brief Copy-out accessors for pressure / density / temperature / velocity.
   *
   * Each one copies the corresponding buffer back to a persistent host staging
   * array (a plain `memcpy` when host and device are the same memory) and
   * repacks it into the canonical RowMajor Eigen layout expected by
   * `Executor::OutputData`. The heavy `f_` / `g_` distributions never leave the
   * device.
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
  //   Newton warm start:   lastGx[d * N + i*ny + j]  (SoA, d-major)
  [[nodiscard]] Index cellIndex(const Index i, const Index j) const { return i * ny_ + j; }

  [[nodiscard]] Index distIndex(const Index idc, const Index cell) const { return idc * N_ + cell; }

  [[nodiscard]] Index uIndex(const Index d, const Index cell) const { return d * N_ + cell; }

  /// Streaming table entry `k` (0..3) of direction `idc` for `cell`.
  [[nodiscard]] Index streamIndex(const Index idc, const Index k, const Index cell) const {
    return (idc * 4 + k) * N_ + cell;
  }

  /**
   * @brief Precompute per-(cell, idc) streaming neighbor indices and weights.
   *
   * Streaming depends only on mesh geometry and the control shift, so the
   * `floor` / clamp / IDW `pow` evaluation is hoisted out of the time loop into
   * this one-off host pass. The tables are filled in host staging arrays and
   * copied to the solver's memory space once, after which
   * `streamAndMacroscopic()` degenerates to a branch-free four-way gather.
   */
  void buildStreamingTables();

  /**
   * @name Per-step kernels
   * @brief Each one is a single `RAJA::forall` over all cells.
   * @{
   */
  /// Recompute `rho`, `u`, `p`, `tem` from the current `f_`, `g_` buffers. Used
  /// only on `Init()`; inside the time loop the macroscopics are produced for
  /// free by the fused `streamAndMacroscopic()` pass.
  void computeMacroscopic();

  /**
   * @brief Seed `f_` and `g_` with the equilibrium distributions built from the
   *        current `rho_`, `u_`, `tem_` fields.
   *
   * Called once from `Init()`. Mirrors the Feq/Geq math of
   * `collisionAndEquilibria` without the BGK relaxation or Newton-Raphson loop
   * (with `u == 0` the NR solution is `xi == 0`, so `geq_i = Wi(T) * targetE /
   * sum(Wi)`).
   */
  void seedEquilibria();

  /**
   * @brief Fused equilibria + BGK collision + thermal correction.
   *
   * One iteration per cell. All scratch (`feq[kQ_]`, `geq[kQ_]`, the
   * Newton-Raphson workspaces) is declared inside the lambda body, so it lives
   * in registers / thread-private memory on every execution policy. The inner
   * Newton-Raphson solver runs a fixed number of iterations with no early exit
   * and clamped reciprocals, keeping control flow warp-uniform; a post-loop
   * `solverOk` predicate selects between the NR result and the
   * `Wi(T)`-normalised fallback.
   */
  void collisionAndEquilibria();

  /**
   * @brief Fused four-way gather streaming + macroscopic reduction.
   *
   * Reads `f_`/`g_`, writes the streamed populations into `faux_`/`gaux_` and,
   * while they are still in registers, accumulates `rho`, `u`, `p` and `tem`
   * for the same cell. Equivalent to the plain backend's `stream()` followed by
   * `computeMacroscopic()` on the swapped buffers, minus one round trip to
   * memory.
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

  // Cached mesh extents (promoted to locals before every kernel launch so the
  // device closure never dereferences `this`).
  Index nx_;
  Index ny_;
  Index N_;        // nx_ * ny_
  Index uSize_;    // N_ * Dim
  Index distSize_; // N_ * Q

  // Flat SoA storage owned by `Backend::Memory`.
  Scalar* rho_ = nullptr;    // size N_
  Scalar* p_ = nullptr;      // size N_
  Scalar* tem_ = nullptr;    // size N_
  Scalar* u_ = nullptr;      // size uSize_,    d-major
  Scalar* f_ = nullptr;      // size distSize_, idc-major
  Scalar* g_ = nullptr;      // size distSize_, idc-major
  Scalar* faux_ = nullptr;   // size distSize_  (swap partner of f_)
  Scalar* gaux_ = nullptr;   // size distSize_  (swap partner of g_)
  Scalar* lastGx_ = nullptr; // size N_ * (Dim + 1), d-major (Newton warm start)

  // Precomputed streaming tables (static across time steps), laid out as
  // (idc * 4 + k) * N_ + cell so that `cell` is the fastest-running index.
  std::int32_t* streamIdx_ = nullptr; // size distSize_ * 4
  Scalar* streamW_ = nullptr;         // size distSize_ * 4

  // Persistent host staging arrays used by `Init()` and the copy-out getters.
  mutable lbmini::DeviceBuffer<Scalar> rhoHost_;
  mutable lbmini::DeviceBuffer<Scalar> pHost_;
  mutable lbmini::DeviceBuffer<Scalar> temHost_;
  mutable lbmini::DeviceBuffer<Scalar> uHost_;
};

template<typename Scalar, typename LatticeType, typename Backend>
LbmTube<Scalar, LatticeType, Backend>::LbmTube(
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

  // `Memory::Allocate` zero-fills, so the buffers start exactly like the
  // `assign(size, Scalar{0})` of the host backends.
  rho_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(N_));
  p_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(N_));
  tem_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(N_));
  u_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(uSize_));

  f_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(distSize_));
  g_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(distSize_));
  faux_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(distSize_));
  gaux_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(distSize_));

  lastGx_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(N_ * (kDim_ + 1)));

  streamIdx_ = Memory::template Allocate<std::int32_t>(static_cast<std::size_t>(distSize_ * 4));
  streamW_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(distSize_ * 4));

  rhoHost_.assign(N_, Scalar{ 0 });
  pHost_.assign(N_, Scalar{ 0 });
  temHost_.assign(N_, Scalar{ 0 });
  uHost_.assign(uSize_, Scalar{ 0 });

  buildStreamingTables();
}

template<typename Scalar, typename LatticeType, typename Backend>
LbmTube<Scalar, LatticeType, Backend>::~LbmTube() {
  Memory::Release(streamW_);
  Memory::Release(streamIdx_);
  Memory::Release(lastGx_);
  Memory::Release(gaux_);
  Memory::Release(faux_);
  Memory::Release(g_);
  Memory::Release(f_);
  Memory::Release(u_);
  Memory::Release(tem_);
  Memory::Release(p_);
  Memory::Release(rho_);
}

template<typename Scalar, typename LatticeType, typename Backend>
auto LbmTube<Scalar, LatticeType, Backend>::P() const -> Tensor<Scalar, kDim_> {
  Memory::CopyOut(pHost_.data(), p_, static_cast<std::size_t>(N_));
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(pHost_.begin(), pHost_.end(), out.data());
  return out;
}

template<typename Scalar, typename LatticeType, typename Backend>
auto LbmTube<Scalar, LatticeType, Backend>::Rho() const -> Tensor<Scalar, kDim_> {
  Memory::CopyOut(rhoHost_.data(), rho_, static_cast<std::size_t>(N_));
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(rhoHost_.begin(), rhoHost_.end(), out.data());
  return out;
}

template<typename Scalar, typename LatticeType, typename Backend>
auto LbmTube<Scalar, LatticeType, Backend>::T() const -> Tensor<Scalar, kDim_> {
  Memory::CopyOut(temHost_.data(), tem_, static_cast<std::size_t>(N_));
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(temHost_.begin(), temHost_.end(), out.data());
  return out;
}

template<typename Scalar, typename LatticeType, typename Backend>
auto LbmTube<Scalar, LatticeType, Backend>::U() const -> Tensor<Scalar, kDim_ + 1> {
  Memory::CopyOut(uHost_.data(), u_, static_cast<std::size_t>(uSize_));
  // SoA d-major -> RowMajor [i][j][d] Eigen tensor.
  Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
  Scalar* dst = out.data();
  for (Index cell = 0; cell < N_; ++cell)
    for (Index d = 0; d < kDim_; ++d)
      dst[cell * kDim_ + d] = uHost_[d * N_ + cell];
  return out;
}

template<typename Scalar, typename LatticeType, typename Backend>
void LbmTube<Scalar, LatticeType, Backend>::Init() {
  // Initialize macroscopic fields with the Sod-like left/right state on the
  // host, then push them to the solver's memory space in four transfers.
  for (Index i = 0; i < nx_; ++i) {
    for (Index j = 0; j < ny_; ++j) {
      const Index cell = cellIndex(i, j);
      uHost_[uIndex(0, cell)] = Scalar{ 0 };
      uHost_[uIndex(1, cell)] = Scalar{ 0 };
      if (i < nx_ / 2) {
        rhoHost_[cell] = kFluid_.densityL;
        pHost_[cell] = kFluid_.pressureL;
      } else {
        rhoHost_[cell] = kFluid_.densityR;
        pHost_[cell] = kFluid_.pressureR;
      }
      temHost_[cell] = LatticeType::Cs2 * pHost_[cell] / (rhoHost_[cell] * kFluid_.constant);
    }
  }

  Memory::CopyIn(rho_, rhoHost_.data(), static_cast<std::size_t>(N_));
  Memory::CopyIn(p_, pHost_.data(), static_cast<std::size_t>(N_));
  Memory::CopyIn(tem_, temHost_.data(), static_cast<std::size_t>(N_));
  Memory::CopyIn(u_, uHost_.data(), static_cast<std::size_t>(uSize_));

  // Newton-Raphson warm start; the buffer is already zero-filled, but Init()
  // may be called more than once on the same solver.
  const lbmini::DeviceBuffer<Scalar> zeros(N_ * (kDim_ + 1), Scalar{ 0 });
  Memory::CopyIn(lastGx_, zeros.data(), static_cast<std::size_t>(N_ * (kDim_ + 1)));

  // Seed f = feq, g = geq. The fused collision kernel is not idempotent for
  // f == 0, so a dedicated equilibrium seeding pass is used instead.
  seedEquilibria();
  computeMacroscopic();
  Memory::Synchronize();
}

template<typename Scalar, typename LatticeType, typename Backend>
void LbmTube<Scalar, LatticeType, Backend>::Step(const bool save) {
  Run(1, save);
}

template<typename Scalar, typename LatticeType, typename Backend>
void LbmTube<Scalar, LatticeType, Backend>::Run(const Index steps, bool /*save*/) {
  for (Index t = 0; t < steps; ++t) {
    collisionAndEquilibria();
    streamAndMacroscopic();
    // O(1) pointer swap (no element-wise copy).
    std::swap(f_, faux_);
    std::swap(g_, gaux_);
  }
  // Device kernel dispatch is asynchronous: synchronize so the caller's
  // wall-clock timing covers the work and not just the enqueue.
  Memory::Synchronize();
}

template<typename Scalar, typename LatticeType, typename Backend>
void LbmTube<Scalar, LatticeType, Backend>::buildStreamingTables() {
  lbmini::DeviceBuffer<std::int32_t> idxHost(distSize_ * 4, 0);
  lbmini::DeviceBuffer<Scalar> wHost(distSize_ * 4, Scalar{ 0 });

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

        auto setUniform = [&](const Index cAny) {
          for (Index k = 0; k < 4; ++k) {
            idxHost[streamIndex(idc, k, cell)] = static_cast<std::int32_t>(cAny);
            wHost[streamIndex(idc, k, cell)] = (k == 0) ? Scalar{ 1 } : Scalar{ 0 };
          }
        };
        auto snap = [&](const Scalar d, const Index cAny) -> bool {
          if (d >= kTiny_)
            return false;
          setUniform(cAny);
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
          // Degenerate fallback: nearest neighbor, self-copy on x out-of-range.
          const Index xN = static_cast<Index>(std::round(xSrc));
          const Index yN = static_cast<Index>(std::round(ySrc));
          const Index cell2 = (xN < 0 || xN >= nx_) ? cell : (xN * ny_ + wrapY(yN));
          setUniform(cell2);
          continue;
        }
        const Scalar invSum = Scalar{ 1 } / wsum;
        idxHost[streamIndex(idc, 0, cell)] = static_cast<std::int32_t>(c00);
        idxHost[streamIndex(idc, 1, cell)] = static_cast<std::int32_t>(c10);
        idxHost[streamIndex(idc, 2, cell)] = static_cast<std::int32_t>(c01);
        idxHost[streamIndex(idc, 3, cell)] = static_cast<std::int32_t>(c11);
        wHost[streamIndex(idc, 0, cell)] = w00 * invSum;
        wHost[streamIndex(idc, 1, cell)] = w10 * invSum;
        wHost[streamIndex(idc, 2, cell)] = w01 * invSum;
        wHost[streamIndex(idc, 3, cell)] = w11 * invSum;
      }
    }
  }

  Memory::CopyIn(streamIdx_, idxHost.data(), static_cast<std::size_t>(distSize_ * 4));
  Memory::CopyIn(streamW_, wHost.data(), static_cast<std::size_t>(distSize_ * 4));
}

template<typename Scalar, typename LatticeType, typename Backend>
void LbmTube<Scalar, LatticeType, Backend>::computeMacroscopic() {
  constexpr Index Q = kQ_;
  const Index N = N_;
  const Scalar ushiftX = kControl_.U(0);
  const Scalar ushiftY = kControl_.U(1);
  const Scalar invCv = Scalar{ 1 } / kFluid_.specificHeatCv;
  const Scalar Rgas = kFluid_.constant;

  const Scalar* f = f_;
  const Scalar* g = g_;
  Scalar* rho = rho_;
  Scalar* p = p_;
  Scalar* tem = tem_;
  Scalar* u = u_;

  RAJA::forall<ExecPolicy>(
    RangeSegment(0, N),
    [=] RAJA_HOST_DEVICE(const Index cell) {
      Scalar rhoSum = Scalar{ 0 };
      Scalar nrg = Scalar{ 0 };
      Scalar momX = Scalar{ 0 };
      Scalar momY = Scalar{ 0 };

      for (Index idc = 0; idc < Q; ++idc) {
        const Scalar fi = f[idc * N + cell];
        const Scalar gi = g[idc * N + cell];
        rhoSum += fi;
        nrg += gi;
        momX += LatticeType::Cshift(idc, 0, ushiftX) * fi;
        momY += LatticeType::Cshift(idc, 1, ushiftY) * fi;
      }

      const Scalar invRho = Scalar{ 1 } / rhoSum;
      const Scalar ux = momX * invRho;
      const Scalar uy = momY * invRho;
      u[0 * N + cell] = ux;
      u[1 * N + cell] = uy;
      const Scalar kin = Scalar{ 0.5 } * (ux * ux + uy * uy);
      const Scalar temp = (Scalar{ 0.5 } * nrg * invRho - kin) * invCv;
      rho[cell] = rhoSum;
      tem[cell] = temp;
      p[cell] = Rgas * rhoSum * temp;
    }
  );
}

template<typename Scalar, typename LatticeType, typename Backend>
void LbmTube<Scalar, LatticeType, Backend>::seedEquilibria() {
  // Compute feq / geq directly from (rho, u, T) and write into f_ / g_ .
  // Used only from Init() where u == 0, so the NR solution is trivial
  // (xi == 0) and geq_i = Wi(T) * (2*rho*E) / sum(Wi).
  constexpr Index Q = kQ_;
  constexpr Index D = kDim_;
  const Index N = N_;
  const Scalar ushiftX = kControl_.U(0);
  const Scalar ushiftY = kControl_.U(1);
  const Scalar cv = kFluid_.specificHeatCv;
  const Scalar tiny = kTiny_;

  Scalar* f = f_;
  Scalar* g = g_;
  const Scalar* rho = rho_;
  const Scalar* tem = tem_;
  const Scalar* u = u_;

  RAJA::forall<ExecPolicy>(
    RangeSegment(0, N),
    [=] RAJA_HOST_DEVICE(const Index cell) {
      const Scalar rhoc = rho[cell];
      const Scalar temc = tem[cell];
      const Scalar ux = u[0 * N + cell];
      const Scalar uy = u[1 * N + cell];
      const Scalar u2 = ux * ux + uy * uy;

      // Feq (product form).
      for (Index idc = 0; idc < Q; ++idc) {
        Scalar phi = rhoc;
        for (Index d = 0; d < D; ++d) {
          const auto vi = LatticeType::Velocity(idc, d);
          const Scalar uia = (d == 0 ? ux : uy) - (d == 0 ? ushiftX : ushiftY);
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
        f[idc * N + cell] = phi;
      }

      // Geq: normalized-Wi distribution (xi == 0 -> e_i = Wi).
      const Scalar E = cv * temc + Scalar{ 0.5 } * u2;
      const Scalar targetE = Scalar{ 2 } * rhoc * E;
      Scalar sumW = Scalar{ 0 };
      for (Index idc = 0; idc < Q; ++idc)
        sumW += LatticeType::Weights(idc, temc);
      const Scalar sc = (sumW > tiny) ? targetE / sumW : targetE / Scalar(Q);
      for (Index idc = 0; idc < Q; ++idc) {
        const Scalar wi = (sumW > tiny) ? LatticeType::Weights(idc, temc) : Scalar{ 1 };
        g[idc * N + cell] = wi * sc;
      }
    }
  );
}

template<typename Scalar, typename LatticeType, typename Backend>
void LbmTube<Scalar, LatticeType, Backend>::collisionAndEquilibria() {
  constexpr Index Q = kQ_;
  constexpr Index D = kDim_;
  constexpr Index maxIter = 3;
  const Index N = N_;
  const Scalar ushiftX = kControl_.U(0);
  const Scalar ushiftY = kControl_.U(1);
  const Scalar tiny = kTiny_;
  const Scalar maxExp = kMaxExp_;
  const Scalar cv = kFluid_.specificHeatCv;
  const Scalar cp = kFluid_.specificHeatCp;
  const Scalar viscosity = kFluid_.viscosity;
  const Scalar conductivity = kFluid_.conductivity;

  Scalar* f = f_;
  Scalar* g = g_;
  Scalar* lastGx = lastGx_;
  const Scalar* rho = rho_;
  const Scalar* tem = tem_;
  const Scalar* u = u_;

  RAJA::forall<ExecPolicy>(
    RangeSegment(0, N),
    [=] RAJA_HOST_DEVICE(const Index cell) {
      const Scalar rhoc = rho[cell];
      const Scalar temc = tem[cell];
      const Scalar ux = u[0 * N + cell];
      const Scalar uy = u[1 * N + cell];
      const Scalar u2 = ux * ux + uy * uy;

      // Feq (Eq. factorized product form)
      Scalar feqLocal[Q];
      for (Index idc = 0; idc < Q; ++idc) {
        Scalar phi = rhoc;
        for (Index d = 0; d < D; ++d) {
          const auto vi = LatticeType::Velocity(idc, d);
          const Scalar uia = (d == 0 ? ux : uy) - (d == 0 ? ushiftX : ushiftY);
          const Scalar uia2t = uia * uia + temc;
          Scalar pf;
          if (vi == 0)
            pf = Scalar{ 1 } - uia2t;
          else if (vi == 1)
            pf = Scalar{ 0.5 } * (uia + uia2t);
          else /* vi == -1 */
            pf = Scalar{ 0.5 } * (-uia + uia2t);
          phi *= pf;
        }
        feqLocal[idc] = phi;
      }

      // Geq via Newton-Raphson on normalized energy flux
      const Scalar E = cv * temc + Scalar{ 0.5 } * u2;
      const Scalar targetE = Scalar{ 2 } * rhoc * E;
      Scalar targetM[D];
      for (Index d = 0; d < D; ++d)
        targetM[d] = Scalar{ 2 } * rhoc * (d == 0 ? ux : uy) * (E + temc) / targetE;

      Scalar xi[D];
      for (Index d = 0; d < D; ++d)
        xi[d] = lastGx[(d + 1) * N + cell];

      Scalar alpha = Scalar{ 1 };
      Scalar si[Q];
      Scalar e[Q];
      Scalar Z = Scalar{ 0 };
      Scalar smax = Scalar{ 0 };
      // Branchless solverOk accumulator: once degenerate, stays degenerate.
      // Evaluated on exit (no mid-loop break) so control flow is uniform across
      // a warp / SIMD lane set.
      bool solverOk = true;

      for (Index iter = 0; iter < maxIter; ++iter) {
        smax = -maxExp;
        for (Index idc = 0; idc < Q; ++idc) {
          Scalar s = Scalar{ 0 };
          for (Index d = 0; d < D; ++d)
            s += xi[d] * LatticeType::Cshift(idc, d, (d == 0 ? ushiftX : ushiftY));
          si[idc] = s;
          smax = (s > smax) ? s : smax;
        }
        smax = (smax < maxExp) ? smax : maxExp;

        Scalar S1[D] = { Scalar{ 0 }, Scalar{ 0 } };
        Scalar S2[D][D] = { { Scalar{ 0 }, Scalar{ 0 } }, { Scalar{ 0 }, Scalar{ 0 } } };
        Z = Scalar{ 0 };
        for (Index idc = 0; idc < Q; ++idc) {
          Scalar expo = si[idc] - smax;
          expo = (expo < maxExp) ? expo : maxExp;
          expo = (expo > -maxExp) ? expo : -maxExp;
          const Scalar wev = LatticeType::Weights(idc, temc) * std::exp(expo);
          e[idc] = wev;
          Z += wev;
          for (Index a = 0; a < D; ++a) {
            const Scalar cia = LatticeType::Cshift(idc, a, (a == 0 ? ushiftX : ushiftY));
            S1[a] += cia * wev;
            for (Index b = 0; b < D; ++b)
              S2[a][b] += cia * LatticeType::Cshift(idc, b, (b == 0 ? ushiftX : ushiftY)) * wev;
          }
        }

        // Clamp Z magnitude (branchless) -> safe reciprocal.
        const Scalar Zsafe = (Z > tiny) ? Z : tiny;
        solverOk = solverOk && (Z > tiny);
        const Scalar invZ = Scalar{ 1 } / Zsafe;

        Scalar J[D][D];
        for (Index a = 0; a < D; ++a)
          for (Index b = 0; b < D; ++b)
            J[a][b] = (S2[a][b] - S1[a] * S1[b] * invZ) * invZ;
        // Branchless ridge on the diagonal.
        J[0][0] += tiny;
        J[1][1] += tiny;

        const Scalar detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
        // Clamp |detJ| magnitude -> safe reciprocal, preserving sign.
        const Scalar absDet = std::fabs(detJ);
        const Scalar detSafe = (absDet > tiny)
                                 ? detJ
                                 : ((detJ >= Scalar{ 0 }) ? tiny : -tiny);
        solverOk = solverOk && (absDet > tiny);
        const Scalar invDet = Scalar{ 1 } / detSafe;

        const Scalar r0 = S1[0] * invZ - targetM[0];
        const Scalar r1 = S1[1] * invZ - targetM[1];
        xi[0] -= alpha * (J[1][1] * r0 - J[0][1] * r1) * invDet;
        xi[1] -= alpha * (-J[1][0] * r0 + J[0][0] * r1) * invDet;
        alpha *= Scalar{ 0.5 };
      }

      Scalar geqLocal[Q];
      if (solverOk) {
        // Recompute e[], Z with final xi.
        smax = -maxExp;
        for (Index idc = 0; idc < Q; ++idc) {
          Scalar s = Scalar{ 0 };
          for (Index d = 0; d < D; ++d)
            s += xi[d] * LatticeType::Cshift(idc, d, (d == 0 ? ushiftX : ushiftY));
          si[idc] = s;
          smax = (s > smax) ? s : smax;
        }
        smax = (smax < maxExp) ? smax : maxExp;
        Z = Scalar{ 0 };
        for (Index idc = 0; idc < Q; ++idc) {
          Scalar expo = si[idc] - smax;
          expo = (expo < maxExp) ? expo : maxExp;
          expo = (expo > -maxExp) ? expo : -maxExp;
          e[idc] = LatticeType::Weights(idc, temc) * std::exp(expo);
          Z += e[idc];
        }
        const Scalar scale = targetE / Z;
        for (Index idc = 0; idc < Q; ++idc)
          geqLocal[idc] = scale * e[idc];
        lastGx[0 * N + cell] = std::log(scale / rhoc) - smax;
        for (Index d = 0; d < D; ++d)
          lastGx[(d + 1) * N + cell] = xi[d];
      } else {
        // Fallback: normalized-Wi distribution.
        for (Index d = 0; d < D + 1; ++d)
          lastGx[d * N + cell] = Scalar{ 0 };
        Scalar sumW = Scalar{ 0 };
        for (Index idc = 0; idc < Q; ++idc)
          sumW += LatticeType::Weights(idc, temc);
        if (sumW <= tiny) {
          const Scalar uni = targetE / Scalar(Q);
          for (Index idc = 0; idc < Q; ++idc)
            geqLocal[idc] = uni;
        } else {
          const Scalar sc = targetE / sumW;
          for (Index idc = 0; idc < Q; ++idc)
            geqLocal[idc] = LatticeType::Weights(idc, temc) * sc;
        }
      }

      // Collision (BGK + Knudsen sensor + thermal-relaxation correction)
      const Scalar tau = viscosity / (rhoc * temc) + Scalar{ 0.5 };
      const Scalar omega = Scalar{ 1 } / tau;
      const Scalar diffusivity = conductivity / (rhoc * cp);
      const Scalar tauT = diffusivity / temc + Scalar{ 0.5 };
      Scalar omegaT = Scalar{ 1 } / tauT;

      Scalar eps = Scalar{ 0 };
      for (Index idc = 0; idc < Q; ++idc) {
        const Scalar fi = f[idc * N + cell];
        const Scalar diff = fi - feqLocal[idc];
        const Scalar den = (feqLocal[idc] > tiny) ? feqLocal[idc] : tiny;
        eps += std::fabs(diff) / den;
      }
      eps /= Scalar(Q);
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
      Scalar L[D] = { Scalar{ 0 }, Scalar{ 0 } };
      for (Index idc = 0; idc < Q; ++idc) {
        const Scalar fi = f[idc * N + cell];
        const Scalar cix0 = static_cast<Scalar>(LatticeType::Velocity(idc, 0));
        const Scalar ciy0 = static_cast<Scalar>(LatticeType::Velocity(idc, 1));
        const Scalar uvi = ux * cix0 + uy * ciy0;
        const Scalar aux = Scalar{ 2 } * uvi * (fi - feqLocal[idc]);
        L[0] += aux * cix0;
        L[1] += aux * ciy0;
      }

      const Scalar invT = Scalar{ 1 } / temc;
      for (Index idc = 0; idc < Q; ++idc) {
        const Index fi = idc * N + cell;
        const Scalar fOld = f[fi];
        f[fi] = fOld + omegaL * (feqLocal[idc] - fOld);
        const Scalar cidotL = L[0] * LatticeType::Cshift(idc, 0, ushiftX)
          + L[1] * LatticeType::Cshift(idc, 1, ushiftY);
        const Scalar Wi = LatticeType::Weights(idc, temc);
        const Scalar gDiff = Wi * cidotL * invT;
        const Scalar gOld = g[fi];
        g[fi] = gOld + omegaL * (geqLocal[idc] - gOld) + (omegaL - omegaT) * gDiff;
      }
    }
  );
}

template<typename Scalar, typename LatticeType, typename Backend>
void LbmTube<Scalar, LatticeType, Backend>::streamAndMacroscopic() {
  constexpr Index Q = kQ_;
  const Index N = N_;
  const Scalar ushiftX = kControl_.U(0);
  const Scalar ushiftY = kControl_.U(1);
  const Scalar invCv = Scalar{ 1 } / kFluid_.specificHeatCv;
  const Scalar Rgas = kFluid_.constant;

  const Scalar* f = f_;
  const Scalar* g = g_;
  Scalar* faux = faux_;
  Scalar* gaux = gaux_;
  const std::int32_t* sIdx = streamIdx_;
  const Scalar* sW = streamW_;
  Scalar* rho = rho_;
  Scalar* p = p_;
  Scalar* tem = tem_;
  Scalar* u = u_;

  RAJA::forall<ExecPolicy>(
    RangeSegment(0, N),
    [=] RAJA_HOST_DEVICE(const Index cell) {
      Scalar rhoSum = Scalar{ 0 };
      Scalar nrg = Scalar{ 0 };
      Scalar momX = Scalar{ 0 };
      Scalar momY = Scalar{ 0 };

      for (Index idc = 0; idc < Q; ++idc) {
        const Index table = (idc * 4) * N + cell;
        const Index i0 = static_cast<Index>(sIdx[table]);
        const Index i1 = static_cast<Index>(sIdx[table + N]);
        const Index i2 = static_cast<Index>(sIdx[table + 2 * N]);
        const Index i3 = static_cast<Index>(sIdx[table + 3 * N]);
        const Scalar w0 = sW[table];
        const Scalar w1 = sW[table + N];
        const Scalar w2 = sW[table + 2 * N];
        const Scalar w3 = sW[table + 3 * N];

        const Index plane = idc * N;
        const Scalar fi = w0 * f[plane + i0] + w1 * f[plane + i1]
          + w2 * f[plane + i2] + w3 * f[plane + i3];
        const Scalar gi = w0 * g[plane + i0] + w1 * g[plane + i1]
          + w2 * g[plane + i2] + w3 * g[plane + i3];
        faux[plane + cell] = fi;
        gaux[plane + cell] = gi;

        rhoSum += fi;
        nrg += gi;
        momX += LatticeType::Cshift(idc, 0, ushiftX) * fi;
        momY += LatticeType::Cshift(idc, 1, ushiftY) * fi;
      }

      const Scalar invRho = Scalar{ 1 } / rhoSum;
      const Scalar ux = momX * invRho;
      const Scalar uy = momY * invRho;
      u[0 * N + cell] = ux;
      u[1 * N + cell] = uy;
      const Scalar kin = Scalar{ 0.5 } * (ux * ux + uy * uy);
      const Scalar temp = (Scalar{ 0.5 } * nrg * invRho - kin) * invCv;
      rho[cell] = rhoSum;
      tem[cell] = temp;
      p[cell] = Rgas * rhoSum * temp;
    }
  );
}
} // namespace lbmini::raja

#endif // LBMINI_RAJA_LBMTUBE_HPP_
