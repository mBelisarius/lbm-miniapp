#ifndef LBMINI_MPI_CPU_LBMTUBE_HPP_
#define LBMINI_MPI_CPU_LBMTUBE_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/HostKernel.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/Simd.hpp"

namespace lbmini::mpi::cpu {
/**
 * @brief MPI x-decomposed, SIMD-vectorised variant of `lbmini::plain::LbmTube`.
 *
 * Numerically equivalent to the plain reference (same operations in the same
 * order per cell; only `exp` differs in the last bit, see `Lbm/Simd.hpp`).
 * Every rank advances its slab of columns with the fused vector kernel of
 * `Lbm/HostKernel.hpp` and exchanges halos with its two neighbours; the
 * getters return the *global* fields on every rank.
 *
 * ### Decomposition and layout
 *
 * The `nx` columns of the tube are split evenly (`nx / ranks` each, the first
 * `nx % ranks` ranks one more). Each rank stores its columns in the x-fastest
 * padded layout of `host::Geometry` (`cell = j * pitch + i`, see
 * `OpenMp/Cpu/LbmTube.hpp` for the buffers), with `G` ghost columns on both
 * sides of the owned ones:
 *
 *     | pad | G ghosts | owned columns ... nb * W | G ghosts |
 *
 * `G` is the widest source offset of the gather (at least 2), `W =
 * simd::kLanes` and `nb = ceil(owned / W)`. The vector blocks of a step start
 * at the first owned column (a multiple of 8, so the stores are aligned) with
 * stride `W`; when the owned width is not a multiple of `W` the last block
 * spills up to `W - 1` lanes past it. Those lanes compute throw-away cells in
 * the right ghost / padding columns of the *output* buffer, which nothing reads
 * before they are rewritten: the next halo exchange refreshes the ghost columns,
 * the others are out of reach of the gather of any owned column, and a rank at
 * the right end of the tube clamps its source columns at the last column.
 * Source columns are clamped into the tube by `Kernel::StepRow`, given in local
 * indices, so only the first and the last rank ever clamp.
 *
 * ### Communication
 *
 * Every step packs the `G` outermost owned columns of every population plane
 * (per plane and row, contiguous), posts `MPI_Irecv` / `MPI_Isend` with the two
 * neighbours and, while the messages are in flight, computes the *inner* blocks,
 * those whose gather window contains no ghost column. After `MPI_Waitall` the
 * ghosts are unpacked and the left and right edge blocks are computed. The split
 * is by whole blocks, so no block straddles two ranges.
 *
 * A rank with fewer columns than `G` (`nx / ranks < G`) cannot supply a
 * neighbour's halo. Such tiny domains keep full-width arrays instead and
 * `MPI_Allgatherv` the owned columns of every plane at the start of each step;
 * a rank computes only its own columns (the lanes spilling into other ranks'
 * columns are rewritten by that exchange).
 *
 * ### Threading
 *
 * MPI is only called from the calling thread, outside parallel regions. Each
 * range of blocks is one `#pragma omp parallel for schedule(runtime)` over row
 * segments (`host::Partition`), so the schedule that `lbmini::Executor` selects
 * (`dynamic,1` unless `OMP_SCHEDULE` is set) balances slow cores; the thread
 * count of a rank is `cores / ranks`, set by the executor. `Init()` uses a
 * static partition of the same segments, so the distributions are first touched
 * by the threads that update them.
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

  // The solver caches raw pointers into its own buffers; a copy would leave
  // them dangling and a move is not needed (see docs/memory-safety.md §5.2).
  LbmTube(const LbmTube&) = delete;

  LbmTube& operator=(const LbmTube&) = delete;

  LbmTube(LbmTube&&) = delete;

  LbmTube& operator=(LbmTube&&) = delete;

  /**
   * @name Macroscopic field getters
   * @brief Collective accessors: every rank must call them, and every rank
   *        receives the global `(nx, ny)` field of the last `Init()` or of the
   *        last step of the last `Run()`.
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
    int length = static_cast<int>(path.size());
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank_ != 0)
      path.resize(length);
    MPI_Bcast(path.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);
  }

private:
  static constexpr int kLanes = simd::kLanes;
  using Kernel = host::Kernel<Scalar, LatticeType, kLanes>;
  using Fields = host::Fields<Scalar>;

  static_assert(kDim_ == 2, "mpi::cpu::LbmTube implements the 2-D closure");

  /// Consecutive vector blocks of every row, split into row segments.
  struct BlockRange {
    host::Index first = 0;  ///< first block; block `b` starts at column `firstCol_ + b * kLanes`
    host::Partition part{}; ///< no segments for an empty range
  };

  [[nodiscard]] BlockRange makeRange(host::Index first, host::Index count) const;

  /// One time step of this rank (halo exchange and compute).
  template<bool kStoreMacro>
  void advance();

  /// The row segments of the ranges `a` and `b`, shared by the threads of this rank.
  template<bool kStoreMacro>
  void compute(const Fields& fields, const BlockRange& a, const BlockRange& b);

  void startHaloExchange();

  void finishHaloExchange();

  void exchangeColumns();

  [[nodiscard]] Tensor<Scalar, kDim_> gatherField(const lbmini::DeviceBuffer<Scalar>& field) const;

  int rank_ = 0;
  int numRanks_ = 1;

  host::Index nxGlobal_ = 0; ///< columns of the tube
  host::Index ny_ = 0;
  host::Index nxActive_ = 0; ///< columns owned by this rank
  host::Index offsetX_ = 0;  ///< global index of the first owned column
  host::Index ghosts_ = 0;   ///< ghost columns on each side (`G`)
  host::Index nb_ = 0;       ///< vector blocks over the owned columns
  host::Index firstCol_ = 0; ///< local index of the first owned column
  host::Index xLo_ = 0;      ///< local index of tube column 0
  host::Index xHi_ = 0;      ///< local index of tube column `nx - 1`
  bool fullWidth_ = false;   ///< tiny-domain mode: full-width arrays, `MPI_Allgatherv` per step
  int threads_ = 1;

  host::Geometry geo_;
  host::Params<Scalar> params_;
  host::Partition initPart_;
  BlockRange left_;  ///< blocks that read the left ghosts
  BlockRange inner_; ///< blocks that read no ghost (all blocks without halo exchange)
  BlockRange right_; ///< blocks that read the right ghosts

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
  Scalar* fNext_ = nullptr;
  Scalar* gCur_ = nullptr;
  Scalar* gNext_ = nullptr;

  // Per-rank column counts/offsets, and the `MPI_Allgatherv` counts of a scalar / 2-vector field.
  std::vector<host::Index> nxOf_;
  std::vector<host::Index> offsetOf_;
  std::vector<int> counts_;
  std::vector<int> displs_;
  std::vector<int> countsU_;
  std::vector<int> displsU_;

  lbmini::DeviceBuffer<Scalar> sendLeft_;
  lbmini::DeviceBuffer<Scalar> sendRight_;
  lbmini::DeviceBuffer<Scalar> recvLeft_;
  lbmini::DeviceBuffer<Scalar> recvRight_;
  lbmini::DeviceBuffer<Scalar> allColumns_; ///< one plane of the whole tube (tiny domains)
  MPI_Request reqs_[4]{};
};

template<typename Scalar, typename LatticeType>
LbmTube<Scalar, LatticeType>::LbmTube(
  const FluidData<Scalar>& fluid,
  const MeshData<Scalar, kDim_>& mesh,
  const ControlData<Scalar>& control,
  const PerformanceData& /*performance*/
) {
  ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "MPI global mesh");
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks_);
#ifdef _OPENMP
  threads_ = omp_get_max_threads();
#endif

  nxGlobal_ = mesh.size[0];
  ny_ = mesh.size[1];
  const host::Index base = nxGlobal_ / numRanks_;
  const host::Index rem = nxGlobal_ % numRanks_;
  for (int r = 0; r < numRanks_; ++r) {
    const host::Index nx = base + (r < rem ? 1 : 0);
    const host::Index offset = r * base + std::min<host::Index>(r, rem);
    nxOf_.push_back(nx);
    offsetOf_.push_back(offset);
    counts_.push_back(static_cast<int>(nx * ny_));
    displs_.push_back(static_cast<int>(offset * ny_));
    countsU_.push_back(static_cast<int>(nx * ny_ * kDim_));
    displsU_.push_back(static_cast<int>(offset * ny_ * kDim_));
  }
  nxActive_ = nxOf_[rank_];
  offsetX_ = offsetOf_[rank_];
  params_ = host::MakeParams<Scalar, LatticeType>(fluid, mesh, control);

  // A shifted gather reaches `ox` ... `ox + 1` columns away (also the nearest-node fallback).
  ghosts_ = 2;
  for (int q = 0; q < kQ_; ++q) {
    ghosts_ = std::max<host::Index>(ghosts_, std::abs(params_.stencil.ox[q]));
    ghosts_ = std::max<host::Index>(ghosts_, std::abs(params_.stencil.ox[q] + 1));
  }

  // A rank with fewer columns than the ghost width cannot provide a neighbour's halo.
  fullWidth_ = numRanks_ > 1 && base < ghosts_;
  nb_ = (nxActive_ + kLanes - 1) / kLanes;
  host::Index columns = 0;
  if (fullWidth_) {
    firstCol_ = offsetX_;
    columns = nxGlobal_ + kLanes; // room for the lanes spilling past the last owned column
    xLo_ = 0;
  } else {
    firstCol_ = host::RoundUp(ghosts_, host::Index{ 8 });
    columns = firstCol_ + nb_ * kLanes + ghosts_;
    xLo_ = firstCol_ - offsetX_;
  }
  xHi_ = xLo_ + nxGlobal_ - 1;
  ValidateMeshExtents(columns, ny_, kQ_, "MPI local mesh with halos");
  geo_ = host::MakeGeometry(nxActive_, ny_, columns);

  // Blocks whose gather window leaves the owned columns need the ghosts; the others
  // are computed while the halo is in flight.
  host::Index leftEnd = 0;
  host::Index rightStart = nb_;
  if (!fullWidth_ && numRanks_ > 1) {
    leftEnd = std::min<host::Index>(nb_, (std::max<host::Index>(0, -params_.minOx) + kLanes - 1) / kLanes);
    const host::Index reach = nxActive_ - kLanes + 1 - params_.maxOx1;
    rightStart = std::clamp<host::Index>(reach > 0 ? (reach + kLanes - 1) / kLanes : 0, leftEnd, nb_);
  }
  left_ = makeRange(0, leftEnd);
  inner_ = makeRange(leftEnd, rightStart - leftEnd);
  right_ = makeRange(rightStart, nb_ - rightStart);
  initPart_ = host::MakePartition(geo_.pitch, ny_, kLanes, threads_);

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
  fNext_ = fB_.data();
  gCur_ = gA_.data();
  gNext_ = gB_.data();

  if (fullWidth_) {
    allColumns_.resize(static_cast<std::size_t>(nxGlobal_ * ny_));
  } else if (numRanks_ > 1) {
    const auto halo = static_cast<std::size_t>(2 * kQ_ * ny_ * ghosts_);
    sendLeft_.resize(halo);
    sendRight_.resize(halo);
    recvLeft_.resize(halo);
    recvRight_.resize(halo);
  }
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::makeRange(const host::Index first, const host::Index count) const -> BlockRange {
  BlockRange range;
  if (count > 0) {
    range.first = first;
    range.part = host::MakePartition(count * kLanes, ny_, kLanes, threads_);
  }
  return range;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::gatherField(const lbmini::DeviceBuffer<Scalar>& field) const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(nxGlobal_, ny_);
  host::ExportField(field.data(), geo_, firstCol_, out.data() + offsetX_ * ny_);
  if (numRanks_ > 1)
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, out.data(), counts_.data(), displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::P() const -> Tensor<Scalar, kDim_> {
  return gatherField(p_);
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::Rho() const -> Tensor<Scalar, kDim_> {
  return gatherField(rho_);
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::T() const -> Tensor<Scalar, kDim_> {
  return gatherField(tem_);
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::U() const -> Tensor<Scalar, kDim_ + 1> {
  // SoA d-major, x-fastest -> RowMajor [i][j][d] Eigen tensor, gathered in place.
  Tensor<Scalar, kDim_ + 1> out(nxGlobal_, ny_, kDim_);
  host::ExportVector(u_.data(), u_.data() + geo_.plane, geo_, firstCol_, out.data() + offsetX_ * ny_ * kDim_);
  if (numRanks_ > 1)
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, out.data(), countsU_.data(), displsU_.data(), MPI_DOUBLE, MPI_COMM_WORLD);
  return out;
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Init() {
  // Both ping-pong buffers are seeded: the lanes spilling past the owned columns
  // read the other buffer's pad and ghost columns in the next step.
  fCur_ = fA_.data();
  fNext_ = fB_.data();
  gCur_ = gA_.data();
  gNext_ = gB_.data();
  Fields current;
  current.fOut = fCur_;
  current.gOut = gCur_;
  current.xi = xi_.data();
  current.rho = rho_.data();
  current.p = p_.data();
  current.tem = tem_.data();
  current.u = u_.data();
  Fields other = current;
  other.fOut = fNext_;
  other.gOut = gNext_;

  const host::Index segments = initPart_.Segments();

  // Every local column of every row is initialised, ghosts included (the exchanged
  // ones are overwritten before they are read); a local column `i` is global `i - xLo_`.
  #pragma omp parallel for schedule(static)
  for (host::Index s = 0; s < segments; ++s) {
    const host::Index i0 = initPart_.FirstBlock(s) * kLanes;
    Kernel::InitRow(params_, geo_, current, nxGlobal_, i0 - xLo_, initPart_.Row(s), i0, initPart_.BlockCount(s));
    Kernel::InitRow(params_, geo_, other, nxGlobal_, i0 - xLo_, initPart_.Row(s), i0, initPart_.BlockCount(s));
  }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Step(const bool save) {
  Run(1, save);
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Run(const Index steps, bool /*save*/) {
  for (Index t = 0; t < steps; ++t) {
    if (t + 1 < steps)
      advance<false>();
    else
      advance<true>();
    std::swap(fCur_, fNext_);
    std::swap(gCur_, gNext_);
  }
}

template<typename Scalar, typename LatticeType>
template<bool kStoreMacro>
void LbmTube<Scalar, LatticeType>::advance() {
  Fields fields;
  fields.fIn = fCur_;
  fields.gIn = gCur_;
  fields.fOut = fNext_;
  fields.gOut = gNext_;
  fields.xi = xi_.data();
  fields.rho = rho_.data();
  fields.p = p_.data();
  fields.tem = tem_.data();
  fields.u = u_.data();

  if (fullWidth_) {
    exchangeColumns();
    compute<kStoreMacro>(fields, inner_, BlockRange{});
  } else if (numRanks_ > 1) {
    // The step reads the current buffer only, so the inner blocks can be computed
    // while the receives fill its ghost columns.
    startHaloExchange();
    compute<kStoreMacro>(fields, inner_, BlockRange{});
    finishHaloExchange();
    compute<kStoreMacro>(fields, left_, right_);
  } else {
    compute<kStoreMacro>(fields, inner_, BlockRange{});
  }
}

template<typename Scalar, typename LatticeType>
template<bool kStoreMacro>
void LbmTube<Scalar, LatticeType>::compute(const Fields& fields, const BlockRange& a, const BlockRange& b) {
  const host::Index segmentsA = a.part.Segments();
  const host::Index segments = segmentsA + b.part.Segments();

  #pragma omp parallel for schedule(runtime)
  for (host::Index s = 0; s < segments; ++s) {
    const BlockRange& range = s < segmentsA ? a : b;
    const host::Index k = s < segmentsA ? s : s - segmentsA;
    const host::Index i0 = firstCol_ + (range.first + range.part.FirstBlock(k)) * kLanes;
    Kernel::template StepRow<kStoreMacro>(
      params_, geo_, fields, xLo_, xHi_, range.part.Row(k), i0, range.part.BlockCount(k)
    );
  }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::startHaloExchange() {
  // Per plane and row, the `G` outermost owned columns, contiguous: f planes first, then g.
  const Scalar* const dist[2] = { fCur_, gCur_ };
  const std::size_t width = static_cast<std::size_t>(ghosts_) * sizeof(Scalar);
  const host::Index rightCol = firstCol_ + nxActive_ - ghosts_;
  Scalar* left = sendLeft_.data();
  Scalar* right = sendRight_.data();
  for (const Scalar* d : dist)
    for (int q = 0; q < kQ_; ++q)
      for (host::Index j = 0; j < ny_; ++j) {
        const Scalar* row = d + q * geo_.plane + j * geo_.pitch;
        std::memcpy(left, row + firstCol_, width);
        std::memcpy(right, row + rightCol, width);
        left += ghosts_;
        right += ghosts_;
      }

  const int count = static_cast<int>(2 * kQ_ * ny_ * ghosts_);
  const int leftRank = rank_ > 0 ? rank_ - 1 : MPI_PROC_NULL;
  const int rightRank = rank_ + 1 < numRanks_ ? rank_ + 1 : MPI_PROC_NULL;
  MPI_Irecv(recvLeft_.data(), count, MPI_DOUBLE, leftRank, 1, MPI_COMM_WORLD, &reqs_[0]);
  MPI_Irecv(recvRight_.data(), count, MPI_DOUBLE, rightRank, 0, MPI_COMM_WORLD, &reqs_[1]);
  MPI_Isend(sendLeft_.data(), count, MPI_DOUBLE, leftRank, 0, MPI_COMM_WORLD, &reqs_[2]);
  MPI_Isend(sendRight_.data(), count, MPI_DOUBLE, rightRank, 1, MPI_COMM_WORLD, &reqs_[3]);
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::finishHaloExchange() {
  MPI_Waitall(4, reqs_, MPI_STATUSES_IGNORE);

  // The tube ends are not ghosts: a rank at either end clamps its source columns instead.
  Scalar* const dist[2] = { fCur_, gCur_ };
  const std::size_t width = static_cast<std::size_t>(ghosts_) * sizeof(Scalar);
  const bool hasLeft = rank_ > 0;
  const bool hasRight = rank_ + 1 < numRanks_;
  const Scalar* left = recvLeft_.data();
  const Scalar* right = recvRight_.data();
  for (Scalar* d : dist)
    for (int q = 0; q < kQ_; ++q)
      for (host::Index j = 0; j < ny_; ++j) {
        Scalar* row = d + q * geo_.plane + j * geo_.pitch;
        if (hasLeft)
          std::memcpy(row + firstCol_ - ghosts_, left, width);
        if (hasRight)
          std::memcpy(row + firstCol_ + nxActive_, right, width);
        left += ghosts_;
        right += ghosts_;
      }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::exchangeColumns() {
  // The owned columns of every plane, gathered in place and copied into the
  // columns of the other ranks.
  Scalar* const dist[2] = { fCur_, gCur_ };
  Scalar* const gathered = allColumns_.data();
  for (Scalar* d : dist)
    for (int q = 0; q < kQ_; ++q) {
      Scalar* plane = d + q * geo_.plane;
      for (host::Index j = 0; j < ny_; ++j)
        std::memcpy(
          gathered + displs_[rank_] + j * nxActive_,
          plane + j * geo_.pitch + firstCol_,
          static_cast<std::size_t>(nxActive_) * sizeof(Scalar)
        );
      MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, gathered, counts_.data(), displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);
      for (int r = 0; r < numRanks_; ++r) {
        if (r == rank_)
          continue;
        for (host::Index j = 0; j < ny_; ++j)
          std::memcpy(
            plane + j * geo_.pitch + offsetOf_[r],
            gathered + displs_[r] + j * nxOf_[r],
            static_cast<std::size_t>(nxOf_[r]) * sizeof(Scalar)
          );
      }
    }
}
} // namespace lbmini::mpi::cpu

#endif // LBMINI_MPI_CPU_LBMTUBE_HPP_