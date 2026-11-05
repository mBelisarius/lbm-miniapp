#ifndef LBMINI_MPI_CPU_LBMTUBE_HPP_
#define LBMINI_MPI_CPU_LBMTUBE_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/OpenAcc/CellPhysics.hpp"

namespace lbmini::mpi::cpu {
/** X-decomposed D2Q9 solver. The current planes always contain post-collision populations. */
template<typename Scalar, typename LatticeType>
class LbmTube : public ILbmTube<Scalar, LatticeType> {
public:
  using Index = Eigen::Index;
  template<typename Type, Index NumIndices>
  using Tensor = Eigen::Tensor<Type, NumIndices, Eigen::RowMajor>;
  static constexpr Index kDim_ = LatticeType::Dim();
  static constexpr Index kQ_ = LatticeType::Speeds();

  LbmTube(const FluidData<Scalar>& fluid, const MeshData<Scalar, kDim_>& mesh,
          const ControlData<Scalar>& control, const PerformanceData& /*performance*/) {
    static_assert(kDim_ == directive::kD && kQ_ == directive::kQ, "D2Q9 only");
    ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "MPI global mesh");
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks_);
    globalNx_ = mesh.size[0];
    ny_ = mesh.size[1];
    const Index base = globalNx_ / numRanks_;
    const Index rem = globalNx_ % numRanks_;
    localNxActive_ = base + (rank_ < rem ? 1 : 0);
    offsetX_ = rank_ * base + std::min(static_cast<Index>(rank_), rem);
    params_ = directive::MakeCellParams<Scalar, LatticeType>(fluid, mesh, control);

    // A shifted gather can reach two or more columns beyond a partition.
    ghosts_ = 2;
    for (int q = 0; q < kQ_; ++q) {
      ghosts_ = std::max(ghosts_, static_cast<Index>(std::abs(params_.stencil.ox[q])));
      ghosts_ = std::max(ghosts_, static_cast<Index>(std::abs(params_.stencil.ox[q] + 1)));
      ghosts_ = std::max(ghosts_, static_cast<Index>(std::abs(std::round(params_.stencil.cix[q]))));
    }
    nx_ = localNxActive_ + 2 * ghosts_;
    ValidateMeshExtents(nx_, ny_, kQ_, "MPI local mesh with halos");
    N_ = nx_ * ny_;
    const Index distSize = N_ * kQ_;
    fA_.assign(distSize, Scalar{ 0 });
    fB_.assign(distSize, Scalar{ 0 });
    gA_.assign(distSize, Scalar{ 0 });
    gB_.assign(distSize, Scalar{ 0 });
    xi_.assign(2 * N_, Scalar{ 0 });
    rho_.assign(N_, Scalar{ 0 });
    p_.assign(N_, Scalar{ 0 });
    tem_.assign(N_, Scalar{ 0 });
    u_.assign(2 * N_, Scalar{ 0 });
    fCur_ = fA_.data();
    fNext_ = fB_.data();
    gCur_ = gA_.data();
    gNext_ = gB_.data();

    const Index bufSize = ghosts_ * ny_ * kQ_ * 2;
    sendLeftBuf_.resize(bufSize);
    sendRightBuf_.resize(bufSize);
    recvLeftBuf_.resize(bufSize);
    recvRightBuf_.resize(bufSize);
    for (int r = 0; r < numRanks_; ++r) {
      const Index count = (base + (r < rem ? 1 : 0)) * ny_;
      counts_.push_back(static_cast<int>(count));
      displs_.push_back(static_cast<int>((r * base + std::min(static_cast<Index>(r), rem)) * ny_));
    }
    // A one-column rank cannot provide a two-column neighbor halo. For these
    // tiny domains gather the full post-collision planes instead of using
    // uninitialized / wrong-rank ghost cells.
    if (base < ghosts_) {
      globalF_.resize(globalNx_ * ny_ * kQ_);
      globalG_.resize(globalNx_ * ny_ * kQ_);
    }
  }

  LbmTube(const LbmTube&) = delete;
  LbmTube& operator=(const LbmTube&) = delete;
  LbmTube(LbmTube&&) = delete;
  LbmTube& operator=(LbmTube&&) = delete;

  [[nodiscard]] Tensor<Scalar, kDim_> P() const override { return gatherField(p_); }
  [[nodiscard]] Tensor<Scalar, kDim_> Rho() const override { return gatherField(rho_); }
  [[nodiscard]] Tensor<Scalar, kDim_> T() const override { return gatherField(tem_); }
  [[nodiscard]] Tensor<Scalar, kDim_ + 1> U() const override {
    Tensor<Scalar, kDim_ + 1> out(globalNx_, ny_, kDim_);
    std::vector<Scalar> send(localNxActive_ * ny_ * kDim_);
    for (Index i = 0; i < localNxActive_; ++i)
      for (Index j = 0; j < ny_; ++j)
        for (Index d = 0; d < kDim_; ++d)
          send[(i * ny_ + j) * kDim_ + d] = u_[d * N_ + (i + ghosts_) * ny_ + j];
    if (numRanks_ == 1)
      std::copy(send.begin(), send.end(), out.data());
    else {
      std::vector<int> counts(numRanks_), displs(numRanks_);
      for (int r = 0; r < numRanks_; ++r) {
        counts[r] = counts_[r] * kDim_;
        displs[r] = displs_[r] * kDim_;
      }
      MPI_Allgatherv(send.data(), static_cast<int>(send.size()), mpiType(), out.data(),
                     counts.data(), displs.data(), mpiType(), MPI_COMM_WORLD);
    }
    return out;
  }

  void Init() override {
    fCur_ = fA_.data();
    fNext_ = fB_.data();
    gCur_ = gA_.data();
    gNext_ = gB_.data();
    // Initialize all local columns; exchanged ghosts overwrite the remote ones.
    for (Index i = 0; i < nx_; ++i) {
      const auto globalI = static_cast<std::int32_t>(std::clamp(offsetX_ + i - ghosts_, Index{ 0 }, globalNx_ - 1));
      for (Index j = 0; j < ny_; ++j) {
        const Index cell = i * ny_ + j;
        Scalar f[kQ_], g[kQ_], xi[kDim_], rho, ux, uy, tem, pres;
        directive::Cell<Scalar>::Init(params_, globalI, f, g, xi, rho, ux, uy, tem, pres);
        rho_[cell] = rho;
        p_[cell] = pres;
        tem_[cell] = tem;
        u_[cell] = ux;
        u_[N_ + cell] = uy;
        xi_[cell] = xi[0];
        xi_[N_ + cell] = xi[1];
        for (int q = 0; q < kQ_; ++q) {
          fCur_[q * N_ + cell] = f[q];
          gCur_[q * N_ + cell] = g[q];
        }
      }
    }
  }

  void Step(bool save) override { Run(1, save); }

  void Run(Index steps, bool /*save*/) override {
    for (Index t = 0; t < steps; ++t) {
      const bool storeMacro = t + 1 == steps;
      if (!globalF_.empty()) {
        exchangeGlobal();
        computeRange(ghosts_, nx_ - ghosts_, storeMacro, true);
      } else if (numRanks_ > 1) {
        // MPI reads the current (post-collision) buffer only. Disjoint interior
        // output can be computed while receives fill the input halo.
        startExchangeGhosts();
        const Index innerStart = std::min(nx_ - ghosts_, 2 * ghosts_);
        const Index innerEnd = std::max(innerStart, nx_ - 2 * ghosts_);
        computeRange(innerStart, innerEnd, storeMacro, false);
        finishExchangeGhosts();
        computeRange(ghosts_, innerStart, storeMacro, false);
        computeRange(innerEnd, nx_ - ghosts_, storeMacro, false);
      } else {
        computeRange(ghosts_, nx_ - ghosts_, storeMacro, false);
      }
      std::swap(fCur_, fNext_);
      std::swap(gCur_, gNext_);
    }
  }

  int Rank() const override { return rank_; }
  void BroadcastOutputPath(std::string& path) const override {
    int length = static_cast<int>(path.size());
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank_ != 0)
      path.resize(length);
    MPI_Bcast(path.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);
  }

private:
  [[nodiscard]] MPI_Datatype mpiType() const { return sizeof(Scalar) == 8 ? MPI_DOUBLE : MPI_FLOAT; }

  [[nodiscard]] Tensor<Scalar, kDim_> gatherField(const lbmini::DeviceBuffer<Scalar>& field) const {
    Tensor<Scalar, kDim_> out(globalNx_, ny_);
    const Scalar* active = field.data() + ghosts_ * ny_;
    if (numRanks_ == 1)
      std::copy_n(active, localNxActive_ * ny_, out.data());
    else
      MPI_Allgatherv(active, counts_[rank_], mpiType(), out.data(), counts_.data(),
                     displs_.data(), mpiType(), MPI_COMM_WORLD);
    return out;
  }

  void exchangeGlobal() {
    const Index globalCells = globalNx_ * ny_;
    for (int q = 0; q < kQ_; ++q) {
      MPI_Allgatherv(fCur_ + q * N_ + ghosts_ * ny_, counts_[rank_], mpiType(),
                     globalF_.data() + q * globalCells, counts_.data(), displs_.data(), mpiType(), MPI_COMM_WORLD);
      MPI_Allgatherv(gCur_ + q * N_ + ghosts_ * ny_, counts_[rank_], mpiType(),
                     globalG_.data() + q * globalCells, counts_.data(), displs_.data(), mpiType(), MPI_COMM_WORLD);
    }
  }

  void computeRange(Index first, Index last, bool storeMacro, bool globalInput) {
    const Scalar* fIn = globalInput ? globalF_.data() : fCur_;
    const Scalar* gIn = globalInput ? globalG_.data() : gCur_;
    const Index inCells = globalInput ? globalNx_ * ny_ : N_;
    const auto& st = params_.stencil;
#pragma omp parallel for schedule(static)
    for (Index i = first; i < last; ++i) {
      const Index globalI = offsetX_ + i - ghosts_;
      for (Index j = 0; j < ny_; ++j) {
        const Index cell = i * ny_ + j;
        Scalar f[kQ_], g[kQ_];
        LBMINI_UNROLL(9)
        for (int q = 0; q < kQ_; ++q) {
          const Scalar* fp = fIn + q * inCells;
          const Scalar* gp = gIn + q * inCells;
          Index src;
          if (st.nearest[q]) {
            const Index xn = static_cast<Index>(std::round(static_cast<Scalar>(globalI) - st.cix[q]));
            const Index yn = static_cast<Index>(std::round(static_cast<Scalar>(j) - st.ciy[q]));
            Index yw = yn % ny_;
            yw += (yw < 0) ? ny_ : 0;
            src = (xn < 0 || xn >= globalNx_) ? globalI * ny_ + j : xn * ny_ + yw;
            if (!globalInput)
              src += (ghosts_ - offsetX_) * ny_;
            f[q] = fp[src];
            g[q] = gp[src];
          } else {
            const Index xa = std::clamp(globalI + st.ox[q], Index{ 0 }, globalNx_ - 1);
            const Index xb = std::clamp(globalI + st.ox[q] + 1, Index{ 0 }, globalNx_ - 1);
            const Index rowA = (xa + (globalInput ? 0 : ghosts_ - offsetX_)) * ny_;
            const Index rowB = (xb + (globalInput ? 0 : ghosts_ - offsetX_)) * ny_;
            Index ya = j + st.oy0[q];
            ya -= (ya >= ny_) ? ny_ : 0;
            Index yb = j + st.oy1[q];
            yb -= (yb >= ny_) ? ny_ : 0;
            f[q] = st.w[q][0] * fp[rowA + ya] + st.w[q][1] * fp[rowB + ya]
                 + st.w[q][2] * fp[rowA + yb] + st.w[q][3] * fp[rowB + yb];
            g[q] = st.w[q][0] * gp[rowA + ya] + st.w[q][1] * gp[rowB + ya]
                 + st.w[q][2] * gp[rowA + yb] + st.w[q][3] * gp[rowB + yb];
          }
        }
        Scalar rho, ux, uy, tem, pres;
        directive::Cell<Scalar>::Macroscopic(f, g, params_, rho, ux, uy, tem, pres);
        if (storeMacro) {
          rho_[cell] = rho;
          p_[cell] = pres;
          tem_[cell] = tem;
          u_[cell] = ux;
          u_[N_ + cell] = uy;
        }
        Scalar xi[kDim_] = { xi_[cell], xi_[N_ + cell] };
        directive::Cell<Scalar>::Collide(params_, rho, ux, uy, tem, f, g, xi);
        xi_[cell] = xi[0];
        xi_[N_ + cell] = xi[1];
        for (int q = 0; q < kQ_; ++q) {
          fNext_[q * N_ + cell] = f[q];
          gNext_[q * N_ + cell] = g[q];
        }
      }
    }
  }

  void startExchangeGhosts() {
    Index left = 0, right = 0;
    for (int q = 0; q < kQ_; ++q)
      for (Index i = 0; i < ghosts_; ++i)
        for (Index j = 0; j < ny_; ++j) {
          const Index cellL = (ghosts_ + i) * ny_ + j;
          const Index cellR = (nx_ - 2 * ghosts_ + i) * ny_ + j;
          sendLeftBuf_[left++] = fCur_[q * N_ + cellL];
          sendLeftBuf_[left++] = gCur_[q * N_ + cellL];
          sendRightBuf_[right++] = fCur_[q * N_ + cellR];
          sendRightBuf_[right++] = gCur_[q * N_ + cellR];
        }
    const int count = static_cast<int>(ghosts_ * ny_ * kQ_ * 2);
    const int leftRank = rank_ ? rank_ - 1 : MPI_PROC_NULL;
    const int rightRank = rank_ + 1 < numRanks_ ? rank_ + 1 : MPI_PROC_NULL;
    MPI_Irecv(recvLeftBuf_.data(), count, mpiType(), leftRank, 1, MPI_COMM_WORLD, &reqs_[0]);
    MPI_Irecv(recvRightBuf_.data(), count, mpiType(), rightRank, 0, MPI_COMM_WORLD, &reqs_[1]);
    MPI_Isend(sendLeftBuf_.data(), count, mpiType(), leftRank, 0, MPI_COMM_WORLD, &reqs_[2]);
    MPI_Isend(sendRightBuf_.data(), count, mpiType(), rightRank, 1, MPI_COMM_WORLD, &reqs_[3]);
  }

  void finishExchangeGhosts() {
    MPI_Waitall(4, reqs_, MPI_STATUSES_IGNORE);
    const bool leftRank = rank_ > 0, rightRank = rank_ + 1 < numRanks_;
    Index left = 0, right = 0;
    for (int q = 0; q < kQ_; ++q)
      for (Index i = 0; i < ghosts_; ++i)
        for (Index j = 0; j < ny_; ++j) {
          const Index cellL = i * ny_ + j;
          const Index cellR = (nx_ - ghosts_ + i) * ny_ + j;
          if (leftRank) {
            fCur_[q * N_ + cellL] = recvLeftBuf_[left];
            gCur_[q * N_ + cellL] = recvLeftBuf_[left + 1];
          }
          if (rightRank) {
            fCur_[q * N_ + cellR] = recvRightBuf_[right];
            gCur_[q * N_ + cellR] = recvRightBuf_[right + 1];
          }
          left += 2;
          right += 2;
        }
  }

  int rank_{ 0 }, numRanks_{ 1 };
  Index globalNx_, localNxActive_, offsetX_, ghosts_, nx_, ny_, N_;
  directive::CellParams<Scalar> params_{};
  lbmini::DeviceBuffer<Scalar> rho_, p_, tem_, u_, fA_, fB_, gA_, gB_, xi_;
  Scalar *fCur_{ nullptr }, *fNext_{ nullptr }, *gCur_{ nullptr }, *gNext_{ nullptr };
  lbmini::DeviceBuffer<Scalar> sendLeftBuf_, sendRightBuf_, recvLeftBuf_, recvRightBuf_;
  MPI_Request reqs_[4]{};
  std::vector<int> counts_, displs_;
  lbmini::DeviceBuffer<Scalar> globalF_, globalG_;
};
} // namespace lbmini::mpi::cpu

#endif // LBMINI_MPI_CPU_LBMTUBE_HPP_