#ifndef LBMINI_MPI_CPU_LBMTUBE_HPP_
#define LBMINI_MPI_CPU_LBMTUBE_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mpi.h>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILbmTube.hpp"

namespace lbmini::mpi::cpu {
/**
 * @brief MPI-based distributed memory implementation of the compressible LBM tube solver.
 *
 * This backend parallelizes the simulation using a 1D domain decomposition along the X-axis.
 * It distributes the global domain across MPI ranks, allocating `localNx` rows per rank
 * plus a halo (ghost) region of `ghosts_` cells on both the left and right boundaries.
 *
 * During the simulation step, the halo exchange is overlapped with inner domain calculations
 * to hide communication latency. Non-blocking `MPI_Isend` and `MPI_Irecv` are initiated via
 * `startExchangeGhosts()` after processing the boundary cells. The inner domain is then processed,
 * followed by `finishExchangeGhosts()` to complete the synchronization before streaming.
 *
 * Global fields are gathered transparently via `MPI_Allgatherv` when calling getters
 * such as `P()`, `Rho()`, `T()`, and `U()`, allowing the root rank to output data as if
 * it were a single continuous domain.
 *
 * @section design_future Possible future improvements
 *  - **2D/3D Cartesian Topology** — Currently using 1D decomposition. `MPI_Cart_create`
 *    and multi-dimensional halo exchanges could improve scalability for domains where
 *    both `nx` and `ny` are large.
 *  - **GPU Integration** — Combining MPI with CUDA/OpenACC/OpenMP offloading.
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

  int Rank() const override { return rank_; }

  void BroadcastOutputPath(std::string& path) const override {
    int pathLen = path.size();
    MPI_Bcast(&pathLen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank_ != 0)
      path.resize(pathLen);
    MPI_Bcast(path.data(), pathLen, MPI_CHAR, 0, MPI_COMM_WORLD);
  }

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
  void collisionAndEquilibria(Index iStart, Index iEnd);

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

  int rank_{ 0 };
  int numRanks_{ 1 };
  Index globalNx_;
  Index localNxActive_;
  Index offsetX_;
  Index ghosts_;

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

  lbmini::DeviceBuffer<Scalar> sendLeftBuf_;
  lbmini::DeviceBuffer<Scalar> sendRightBuf_;
  lbmini::DeviceBuffer<Scalar> recvLeftBuf_;
  lbmini::DeviceBuffer<Scalar> recvRightBuf_;
  MPI_Request reqs_[4];

  void startExchangeGhosts();

  void finishExchangeGhosts();

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
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks_);

  globalNx_ = mesh.size[0];
  ny_ = mesh.size[1];

  Index base_nx = globalNx_ / numRanks_;
  Index rem = globalNx_ % numRanks_;
  localNxActive_ = base_nx + (rank_ < rem ? 1 : 0);
  offsetX_ = rank_ * base_nx + std::min(static_cast<Index>(rank_), rem);

  ghosts_ = 2;
  nx_ = localNxActive_ + 2 * ghosts_; // local padded dimension
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

  Index bufSize = ghosts_ * ny_ * kQ_ * 2;
  sendLeftBuf_.assign(bufSize, Scalar{ 0 });
  sendRightBuf_.assign(bufSize, Scalar{ 0 });
  recvLeftBuf_.assign(bufSize, Scalar{ 0 });
  recvRightBuf_.assign(bufSize, Scalar{ 0 });

  buildStreamingTables();
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::P() const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(globalNx_, ny_);
  std::vector<Scalar> sendBuf(localNxActive_ * ny_);
  for (typename LbmTube<Scalar, LatticeType>::Index i = 0; i < localNxActive_; ++i)
    for (typename LbmTube<Scalar, LatticeType>::Index j = 0; j < ny_; ++j)
      sendBuf[i * ny_ + j] = p_[(i + ghosts_) * ny_ + j];

  if (numRanks_ == 1) {
    std::copy(sendBuf.begin(), sendBuf.end(), out.data());
    return out;
  }

  std::vector<int> recvcounts(numRanks_);
  std::vector<int> displs(numRanks_);
  int count = static_cast<int>(localNxActive_ * ny_);
  MPI_Allgather(&count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  displs[0] = 0;
  for (int i = 1; i < numRanks_; ++i)
    displs[i] = displs[i - 1] + recvcounts[i - 1];

  MPI_Datatype mpiType = (sizeof(Scalar) == 8) ? MPI_DOUBLE : MPI_FLOAT;
  MPI_Allgatherv(
    sendBuf.data(),
    count,
    mpiType,
    out.data(),
    recvcounts.data(),
    displs.data(),
    mpiType,
    MPI_COMM_WORLD
  );
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::Rho() const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(globalNx_, ny_);
  std::vector<Scalar> sendBuf(localNxActive_ * ny_);
  for (typename LbmTube<Scalar, LatticeType>::Index i = 0; i < localNxActive_; ++i)
    for (typename LbmTube<Scalar, LatticeType>::Index j = 0; j < ny_; ++j)
      sendBuf[i * ny_ + j] = rho_[(i + ghosts_) * ny_ + j];

  if (numRanks_ == 1) {
    std::copy(sendBuf.begin(), sendBuf.end(), out.data());
    return out;
  }

  std::vector<int> recvcounts(numRanks_);
  std::vector<int> displs(numRanks_);
  int count = static_cast<int>(localNxActive_ * ny_);
  MPI_Allgather(&count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  displs[0] = 0;
  for (int i = 1; i < numRanks_; ++i)
    displs[i] = displs[i - 1] + recvcounts[i - 1];

  MPI_Datatype mpiType = (sizeof(Scalar) == 8) ? MPI_DOUBLE : MPI_FLOAT;
  MPI_Allgatherv(
    sendBuf.data(),
    count,
    mpiType,
    out.data(),
    recvcounts.data(),
    displs.data(),
    mpiType,
    MPI_COMM_WORLD
  );
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::T() const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(globalNx_, ny_);
  std::vector<Scalar> sendBuf(localNxActive_ * ny_);
  for (typename LbmTube<Scalar, LatticeType>::Index i = 0; i < localNxActive_; ++i)
    for (typename LbmTube<Scalar, LatticeType>::Index j = 0; j < ny_; ++j)
      sendBuf[i * ny_ + j] = tem_[(i + ghosts_) * ny_ + j];

  if (numRanks_ == 1) {
    std::copy(sendBuf.begin(), sendBuf.end(), out.data());
    return out;
  }

  std::vector<int> recvcounts(numRanks_);
  std::vector<int> displs(numRanks_);
  int count = static_cast<int>(localNxActive_ * ny_);
  MPI_Allgather(&count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  displs[0] = 0;
  for (int i = 1; i < numRanks_; ++i)
    displs[i] = displs[i - 1] + recvcounts[i - 1];

  MPI_Datatype mpiType = (sizeof(Scalar) == 8) ? MPI_DOUBLE : MPI_FLOAT;
  MPI_Allgatherv(
    sendBuf.data(),
    count,
    mpiType,
    out.data(),
    recvcounts.data(),
    displs.data(),
    mpiType,
    MPI_COMM_WORLD
  );
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::U() const -> Tensor<Scalar, kDim_ + 1> {
  Tensor<Scalar, kDim_ + 1> out(globalNx_, ny_, kDim_);
  std::vector<Scalar> sendBuf(localNxActive_ * ny_ * kDim_);
  for (typename LbmTube<Scalar, LatticeType>::Index i = 0; i < localNxActive_; ++i) {
    for (typename LbmTube<Scalar, LatticeType>::Index j = 0; j < ny_; ++j) {
      for (typename LbmTube<Scalar, LatticeType>::Index d = 0; d < kDim_; ++d) {
        sendBuf[(i * ny_ + j) * kDim_ + d] = u_[d * N_ + (i + ghosts_) * ny_ + j];
      }
    }
  }

  if (numRanks_ == 1) {
    std::copy(sendBuf.begin(), sendBuf.end(), out.data());
    return out;
  }

  std::vector<int> recvcounts(numRanks_);
  std::vector<int> displs(numRanks_);
  int count = static_cast<int>(localNxActive_ * ny_ * kDim_);
  MPI_Allgather(&count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  displs[0] = 0;
  for (int i = 1; i < numRanks_; ++i)
    displs[i] = displs[i - 1] + recvcounts[i - 1];

  MPI_Datatype mpiType = (sizeof(Scalar) == 8) ? MPI_DOUBLE : MPI_FLOAT;
  MPI_Allgatherv(
    sendBuf.data(),
    count,
    mpiType,
    out.data(),
    recvcounts.data(),
    displs.data(),
    mpiType,
    MPI_COMM_WORLD
  );
  return out;
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Init() {
  // Initialize macroscopic fields with the Sod-like left/right state.
  for (typename LbmTube<Scalar, LatticeType>::Index i = 0; i < nx_; ++i) {
    for (typename LbmTube<Scalar, LatticeType>::Index j = 0; j < ny_; ++j) {
      const typename LbmTube<Scalar, LatticeType>::Index cell = cellIndex(i, j);
      u_[0 * N_ + cell] = Scalar{ 0 };
      u_[1 * N_ + cell] = Scalar{ 0 };

      const typename LbmTube<Scalar, LatticeType>::Index globalI = offsetX_ + i - ghosts_;
      if (globalI < globalNx_ / 2) {
        rho_[cell] = kFluid_.densityL;
        p_[cell] = kFluid_.pressureL;
      } else {
        rho_[cell] = kFluid_.densityR;
        p_[cell] = kFluid_.pressureR;
      }
      tem_[cell] = LatticeType::Cs2 * p_[cell] / (rho_[cell] * kFluid_.constant);
      for (typename LbmTube<Scalar, LatticeType>::Index d = 0; d < kDim_ + 1; ++d)
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
void LbmTube<Scalar, LatticeType>::Run(const typename LbmTube<Scalar, LatticeType>::Index steps, bool /*save*/) {
  for (typename LbmTube<Scalar, LatticeType>::Index t = 0; t < steps; ++t) {
    if (numRanks_ > 1) {
      // 1. Compute collision on boundary active cells
      collisionAndEquilibria(ghosts_, 2 * ghosts_);
      collisionAndEquilibria(nx_ - 2 * ghosts_, nx_ - ghosts_);

      // 2. Start non-blocking halo exchange of f_ and g_
      startExchangeGhosts();

      // 3. Compute collision on inner domain while communication is in flight
      collisionAndEquilibria(2 * ghosts_, nx_ - 2 * ghosts_);

      // 4. Wait for communication to finish
      finishExchangeGhosts();
    } else {
      // Single-rank execution: compute all active cells at once
      collisionAndEquilibria(ghosts_, nx_ - ghosts_);
    }

    // 5. Stream the entire active domain (plus ghosts logic internally handled)
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

  auto clampGlobalX = [&](const Index x) -> Index {
    if (x < 0)
      return 0;
    if (x >= globalNx_)
      return globalNx_ - 1;
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

        // Active domain only? Wait, we can evaluate for all i, but if streamIdx is out of bounds,
        // we map it to our own ghost zones!
        const Scalar globalX = static_cast<Scalar>(offsetX_ + i - ghosts_);
        const Scalar xSrc = globalX - cix;
        const Scalar ySrc = static_cast<Scalar>(j) - ciy;

        const Index x0 = static_cast<Index>(std::floor(xSrc));
        const Index y0 = static_cast<Index>(std::floor(ySrc));
        const Index x1 = x0 + 1;
        const Index y1 = y0 + 1;
        const Index cx0 = clampGlobalX(x0), cx1 = clampGlobalX(x1);
        const Index wy0 = wrapY(y0), wy1 = wrapY(y1);

        const Index lx0 = cx0 - offsetX_ + ghosts_;
        const Index lx1 = cx1 - offsetX_ + ghosts_;

        const Index c00 = lx0 * ny_ + wy0;
        const Index c10 = lx1 * ny_ + wy0;
        const Index c01 = lx0 * ny_ + wy1;
        const Index c11 = lx1 * ny_ + wy1;

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
          if (xN < 0 || xN >= globalNx_) {
            cell2 = cell;
          } else {
            Index lxN = xN - offsetX_ + ghosts_;
            cell2 = lxN * ny_ + wrapY(yN);
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
void LbmTube<Scalar, LatticeType>::collisionAndEquilibria(Index iStart, Index iEnd) {
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

  const Index cellStart = iStart * ny_;
  const Index cellEnd = iEnd * ny_;

  for (Index cell = cellStart; cell < cellEnd; ++cell) {
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

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::startExchangeGhosts() {
  if (numRanks_ <= 1)
    return;

  const int leftRank = (rank_ > 0) ? rank_ - 1 : MPI_PROC_NULL;
  const int rightRank = (rank_ < numRanks_ - 1) ? rank_ + 1 : MPI_PROC_NULL;

  Scalar* pF = f_.data();
  Scalar* pG = g_.data();
  Scalar* sL = sendLeftBuf_.data();
  Scalar* sR = sendRightBuf_.data();
  Scalar* rL = recvLeftBuf_.data();
  Scalar* rR = recvRightBuf_.data();

  typename LbmTube<Scalar, LatticeType>::Index idxL = 0;
  typename LbmTube<Scalar, LatticeType>::Index idxR = 0;

  // Pack send buffers
  for (typename LbmTube<Scalar, LatticeType>::Index idc = 0; idc < kQ_; ++idc) {
    for (typename LbmTube<Scalar, LatticeType>::Index i = 0; i < ghosts_; ++i) {
      for (typename LbmTube<Scalar, LatticeType>::Index j = 0; j < ny_; ++j) {
        // Send to Left: pack cells [ghosts_, 2*ghosts_ - 1]
        typename LbmTube<Scalar, LatticeType>::Index cellL = (i + ghosts_) * ny_ + j;
        sL[idxL] = pF[idc * N_ + cellL];
        sL[idxL + 1] = pG[idc * N_ + cellL];
        idxL += 2;

        // Send to Right: pack cells [nx_ - 2*ghosts_, nx_ - ghosts_ - 1]
        typename LbmTube<Scalar, LatticeType>::Index cellR = (nx_ - 2 * ghosts_ + i) * ny_ + j;
        sR[idxR] = pF[idc * N_ + cellR];
        sR[idxR + 1] = pG[idc * N_ + cellR];
        idxR += 2;
      }
    }
  }

  const int count = static_cast<int>(ghosts_ * ny_ * kQ_ * 2);
  MPI_Datatype mpiType = (sizeof(Scalar) == 8) ? MPI_DOUBLE : MPI_FLOAT;

  int reqCount = 0;

  // Send to neighbors
  if (leftRank != MPI_PROC_NULL) {
    MPI_Isend(sL, count, mpiType, leftRank, 0, MPI_COMM_WORLD, &reqs_[reqCount++]);
    MPI_Irecv(rL, count, mpiType, leftRank, 1, MPI_COMM_WORLD, &reqs_[reqCount++]);
  }
  if (rightRank != MPI_PROC_NULL) {
    MPI_Isend(sR, count, mpiType, rightRank, 1, MPI_COMM_WORLD, &reqs_[reqCount++]);
    MPI_Irecv(rR, count, mpiType, rightRank, 0, MPI_COMM_WORLD, &reqs_[reqCount++]);
  }

  // To avoid uninitialized MPI_Request elements causing MPI_Waitall to fail,
  // we initialize unused requests to MPI_REQUEST_NULL.
  for (int i = reqCount; i < 4; ++i) {
    reqs_[i] = MPI_REQUEST_NULL;
  }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::finishExchangeGhosts() {
  if (numRanks_ <= 1)
    return;

  const int leftRank = (rank_ > 0) ? rank_ - 1 : MPI_PROC_NULL;
  const int rightRank = (rank_ < numRanks_ - 1) ? rank_ + 1 : MPI_PROC_NULL;

  // Wait for all non-blocking communications to complete
  MPI_Waitall(4, reqs_, MPI_STATUSES_IGNORE);

  Scalar* pF = f_.data();
  Scalar* pG = g_.data();
  Scalar* rL = recvLeftBuf_.data();
  Scalar* rR = recvRightBuf_.data();

  // Unpack recv buffers
  typename LbmTube<Scalar, LatticeType>::Index idxL = 0;
  typename LbmTube<Scalar, LatticeType>::Index idxR = 0;
  for (typename LbmTube<Scalar, LatticeType>::Index idc = 0; idc < kQ_; ++idc) {
    for (typename LbmTube<Scalar, LatticeType>::Index i = 0; i < ghosts_; ++i) {
      for (typename LbmTube<Scalar, LatticeType>::Index j = 0; j < ny_; ++j) {
        // Recv from Left: unpack into [0, ghosts_ - 1]
        typename LbmTube<Scalar, LatticeType>::Index cellL = i * ny_ + j;
        if (leftRank != MPI_PROC_NULL) {
          pF[idc * N_ + cellL] = rL[idxL];
          pG[idc * N_ + cellL] = rL[idxL + 1];
        }
        idxL += 2;

        // Recv from Right: unpack into [nx_ - ghosts_, nx_ - 1]
        typename LbmTube<Scalar, LatticeType>::Index cellR = (nx_ - ghosts_ + i) * ny_ + j;
        if (rightRank != MPI_PROC_NULL) {
          pF[idc * N_ + cellR] = rR[idxR];
          pG[idc * N_ + cellR] = rR[idxR + 1];
        }
        idxR += 2;
      }
    }
  }
}
} // namespace lbmini::mpi::cpu

#endif // LBMINI_MPI_CPU_LBMTUBE_HPP_
