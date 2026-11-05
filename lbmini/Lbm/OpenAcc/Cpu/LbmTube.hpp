#ifndef LBMINI_OPENACC_CPU_LBMTUBE_HPP_
#define LBMINI_OPENACC_CPU_LBMTUBE_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <type_traits>
#include <utility>
#include <openacc.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/HostKernel.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/OpenAcc/Cpu/LaneKernel.hpp"

namespace lbmini::openacc::cpu {
/**
 * @brief OpenACC multi-core variant of the compressible LBM tube solver.
 *
 * Numerically equivalent to `lbmini::plain::LbmTube` (to ~1e-12: the exponential is
 * the inline polynomial of `LaneKernel.hpp`) and laid out like every CPU backend
 * (`Lbm/HostKernel.hpp`): structure of arrays, **x-fastest** padded rows,
 * `x[plane_index * plane + j * pitch + i]`, so that a vector loop over `i` runs over
 * contiguous memory whatever `ny` is.
 *
 * ### State and data layout
 *
 * | buffer      | size      | role                                |
 * | :---------- | :-------- | :---------------------------------- |
 * | `fA/fB`     | `Q * pl`  | post-collision density (ping-pong)  |
 * | `gA/gB`     | `Q * pl`  | post-collision energy (ping-pong)   |
 * | `xi`        | `2 * pl`  | Newton-Raphson warm start           |
 * | `rho/p/tem` | `plane`   | macroscopic output                  |
 * | `u`         | `2 * pl`  | macroscopic output                  |
 *
 * ### Kernels
 *
 *  - One time step is one `acc parallel` region. Its gangs (one per core) pull row
 *    segments (`host::Partition`: a run of `kVL`-cell blocks of one row) from a shared
 *    counter with `acc atomic capture`, i.e. a dynamic schedule: a slower core simply
 *    takes fewer segments, where the static gang distribution of the compiler left
 *    25-35 % of every step idle on a hybrid CPU.
 *  - Each block is processed by a chain of `acc routine vector` phases over a
 *    gang-private `Block` (pull, macroscopics and equilibria, three Newton
 *    iterations, energy equilibrium, collision; see `LaneKernel.hpp` for why the code
 *    is written as lane loops: `nvc++` vectorises those and nothing bigger).
 *  - `Init()` runs the same phases over the Sod state with a static gang loop, so each
 *    thread first touches the pages of its own segments. The macroscopic fields are
 *    written on the last step of a `Run()` only; the ping-pong swap is a host pointer swap.
 *
 * ### Data mapping and device selection
 *
 * The test and benchmark binaries are built with both `-acc=multicore` and
 * `-acc=gpu`, so every compute region exists in a host and in a GPU version and the
 * *runtime* current device type selects one. This solver owns plain host memory, so every
 * member that runs a region does so under a scope that selects `acc_device_host`
 * (restoring the previous type on exit): on the host device the host pointers are the
 * device pointers and `deviceptr` is exact - no present-table lookup, no implicit copy.
 *
 * Thread count and thread/memory affinity are process-global settings owned by
 * `lbmini::Executor` (`cores`, `ACC_NUM_CORES`); the constructor touches none of it.
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

  // The solver caches raw ping-pong pointers into its own buffers; a copy
  // would alias the source's storage and a move would leave them dangling.
  // See docs/memory-safety.md §5.2.
  LbmTube(const LbmTube&) = delete;

  LbmTube& operator=(const LbmTube&) = delete;

  LbmTube(LbmTube&&) = delete;

  LbmTube& operator=(LbmTube&&) = delete;

  ~LbmTube() override = default;

  [[nodiscard]] Tensor<Scalar, kDim_> P() const override;

  [[nodiscard]] Tensor<Scalar, kDim_> Rho() const override;

  [[nodiscard]] Tensor<Scalar, kDim_> T() const override;

  [[nodiscard]] Tensor<Scalar, kDim_ + 1> U() const override;

  void Init() override;

  void Step(bool save) override;

  void Run(Index steps, bool save) override;

private:
  static constexpr int kVL = lane::kVL;
  static constexpr int kQ = lane::kQ;

  static_assert(std::is_same_v<Scalar, double>, "the OpenACC CPU lane kernel is double precision");
  static_assert(kDim_ == 2 && kQ_ == kQ, "D2Q9 only");
  static_assert(host::kPadColumns % kVL == 0, "rows are padded to a whole number of blocks");

  /// Selects the host device for the current thread, restores the previous type.
  class HostScope {
  public:
    HostScope() : previous_(acc_get_device_type()) {
      if (previous_ != acc_device_host)
        acc_set_device_type(acc_device_host);
    }

    HostScope(const HostScope&) = delete;

    HostScope& operator=(const HostScope&) = delete;

    ~HostScope() {
      if (previous_ != acc_device_host && previous_ != acc_device_none)
        acc_set_device_type(previous_);
    }

  private:
    acc_device_t previous_;
  };

  /// One fused stream + macroscopic + collision step.
  template<bool kStoreMacro>
  void step();

  [[nodiscard]] Tensor<Scalar, kDim_> exportField(const lbmini::DeviceBuffer<Scalar>& field) const;

  /// Cores of the run: `ACC_NUM_CORES` (set by the executor) or every hardware thread.
  static int GangCount() {
    if (const char* value = std::getenv("ACC_NUM_CORES")) {
      if (const int n = std::atoi(value); n > 0)
        return n;
    }
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  }

  const FluidData<Scalar> kFluid_;
  const MeshData<Scalar, kDim_> kMesh_;
  const ControlData<Scalar> kControl_;
  const PerformanceData kPerformance_;

  Index nx_;
  Index ny_;
  int gangs_;

  host::Geometry geo_;
  host::Params<Scalar> params_;
  host::Partition part_;

  FirstTouchBuffer<Scalar> fA_;
  FirstTouchBuffer<Scalar> fB_;
  FirstTouchBuffer<Scalar> gA_;
  FirstTouchBuffer<Scalar> gB_;
  FirstTouchBuffer<Scalar> xi_;
  lbmini::DeviceBuffer<Scalar> rho_;
  lbmini::DeviceBuffer<Scalar> p_;
  lbmini::DeviceBuffer<Scalar> tem_;
  lbmini::DeviceBuffer<Scalar> u_;

  /// Next segment of the running step; touched only through `acc atomic`.
  alignas(64) host::Index next_ = 0;

  // Ping-pong roles of the distribution buffers (swapped, never copied).
  Scalar* fCur_ = nullptr;
  Scalar* fNext_ = nullptr;
  Scalar* gCur_ = nullptr;
  Scalar* gNext_ = nullptr;
};

template<typename Scalar, typename LatticeType>
LbmTube<Scalar, LatticeType>::LbmTube(
  const FluidData<Scalar>& fluid,
  const MeshData<Scalar, kDim_>& mesh,
  const ControlData<Scalar>& control,
  const PerformanceData& performance
)
  : kFluid_(fluid), kMesh_(mesh), kControl_(control), kPerformance_(performance) {
  ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "openacc::cpu::LbmTube");

  nx_ = mesh.size[0];
  ny_ = mesh.size[1];
  gangs_ = GangCount();
  geo_ = host::MakeGeometry(nx_, ny_);
  params_ = host::MakeParams<Scalar, LatticeType>(fluid, mesh, control);
  part_ = host::MakePartition(geo_.pitch, geo_.ny, kVL, gangs_);

  // Distributions and warm start are first written by `Init()`; the macroscopic fields are zero until then.
  const auto plane = static_cast<std::size_t>(geo_.plane);
  fA_.resize(kQ * plane);
  fB_.resize(kQ * plane);
  gA_.resize(kQ * plane);
  gB_.resize(kQ * plane);
  xi_.resize(2 * plane);
  rho_.assign(plane, Scalar{ 0 });
  p_.assign(plane, Scalar{ 0 });
  tem_.assign(plane, Scalar{ 0 });
  u_.assign(2 * plane, Scalar{ 0 });

  fCur_ = fA_.data();
  fNext_ = fB_.data();
  gCur_ = gA_.data();
  gNext_ = gB_.data();
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
  Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
  host::ExportVector(u_.data(), u_.data() + geo_.plane, geo_, host::Index{ 0 }, out.data());
  return out;
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Init() {
  HostScope scope;
  // The buffers are only ever written by kernels, so there is no stale state
  // to clear: a re-Init() simply restarts from the initial condition.
  fCur_ = fA_.data();
  fNext_ = fB_.data();
  gCur_ = gA_.data();
  gNext_ = gB_.data();

  const host::Params<Scalar>* __restrict__ prm = &params_;
  Scalar* __restrict__ fOut = fCur_;
  Scalar* __restrict__ gOut = gCur_;
  Scalar* __restrict__ xiOut = xi_.data();
  Scalar* __restrict__ rhoOut = rho_.data();
  Scalar* __restrict__ pOut = p_.data();
  Scalar* __restrict__ temOut = tem_.data();
  Scalar* __restrict__ uOut = u_.data();
  const host::Index segments = part_.Segments();
  const host::Index rows = part_.rows;
  const host::Index chunkBlocks = part_.chunkBlocks;
  const host::Index blocksPerRow = part_.blocksPerRow;
  const host::Index nx = nx_;
  const host::Index plane = geo_.plane;
  const host::Index pitch = geo_.pitch;

  // Static gang distribution of the same segments the steps use: first touch by the owner.
  #pragma acc parallel loop gang deviceptr(prm, fOut, gOut, xiOut, rhoOut, pOut, temOut, uOut)
  for (host::Index s = 0; s < segments; ++s) {
    lane::Block B;
    const host::Index j = s % rows;
    const host::Index first = (s / rows) * chunkBlocks;
    const host::Index count = (blocksPerRow - first < chunkBlocks) ? blocksPerRow - first : chunkBlocks;
    for (host::Index b = 0; b < count; ++b) {
      const host::Index i0 = (first + b) * kVL;
      const host::Index cell = j * pitch + i0;
      lane::Seed(*prm, nx, i0, B);
      lane::Prepare(*prm, xiOut, cell, plane, true, B);
      lane::StoreMacro(cell, plane, rhoOut, pOut, temOut, uOut, B);
      lane::Newton(*prm, B);
      lane::Equilibrium(B);
      lane::Collide(*prm, B);
      lane::StoreState(cell, plane, fOut, gOut, xiOut, B);
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
  HostScope scope;
  for (Index t = 0; t < steps; ++t) {
    if (t + 1 == steps)
      step<true>();
    else
      step<false>();
    std::swap(fCur_, fNext_);
    std::swap(gCur_, gNext_);
  }
}

template<typename Scalar, typename LatticeType>
template<bool kStoreMacro>
void LbmTube<Scalar, LatticeType>::step() {
  const host::Params<Scalar>* __restrict__ prm = &params_;
  const Scalar* __restrict__ fIn = fCur_;
  const Scalar* __restrict__ gIn = gCur_;
  Scalar* __restrict__ fOut = fNext_;
  Scalar* __restrict__ gOut = gNext_;
  Scalar* __restrict__ xiIo = xi_.data();
  Scalar* __restrict__ rhoOut = rho_.data();
  Scalar* __restrict__ pOut = p_.data();
  Scalar* __restrict__ temOut = tem_.data();
  Scalar* __restrict__ uOut = u_.data();
  host::Index* const counter = &next_;
  *counter = 0;

  const int gangs = gangs_;
  const host::Index segments = part_.Segments();
  const host::Index rows = part_.rows;
  const host::Index chunkBlocks = part_.chunkBlocks;
  const host::Index blocksPerRow = part_.blocksPerRow;
  const host::Index plane = geo_.plane;
  const host::Index pitch = geo_.pitch;
  const host::Geometry geo = geo_;
  const host::Index xLo = 0;
  const host::Index xHi = geo_.nx - 1;
  // A block is interior if every direction's gather window stays inside the tube.
  const host::Index interiorLo = xLo - params_.minOx;
  const host::Index interiorHi = xHi - (kVL - 1) - params_.maxOx1;

  #pragma acc parallel num_gangs(gangs) deviceptr(prm, fIn, gIn, fOut, gOut, xiIo, rhoOut, pOut, temOut, uOut, counter)
  {
    #pragma acc loop gang
    for (int gang = 0; gang < gangs; ++gang) {
      lane::Block B;
      for (;;) {
        host::Index s;
        #pragma acc atomic capture
        s = (*counter)++;
        if (s >= segments)
          break;
        const host::Index j = s % rows;
        const host::Index first = (s / rows) * chunkBlocks;
        const host::Index count = (blocksPerRow - first < chunkBlocks) ? blocksPerRow - first : chunkBlocks;
        lane::Row row;
        lane::BeginRow(*prm, geo, xLo, xHi, j, row);
        for (host::Index b = 0; b < count; ++b) {
          const host::Index i0 = (first + b) * kVL;
          const host::Index cell = j * pitch + i0;
          const bool interior = i0 >= interiorLo && i0 <= interiorHi;
          lane::Pull(*prm, fIn, gIn, row, i0, interior, B);
          lane::Prepare(*prm, xiIo, cell, plane, false, B);
          if constexpr (kStoreMacro)
            lane::StoreMacro(cell, plane, rhoOut, pOut, temOut, uOut, B);
          lane::Newton(*prm, B);
          lane::Equilibrium(B);
          lane::Collide(*prm, B);
          lane::StoreState(cell, plane, fOut, gOut, xiIo, B);
        }
      }
    }
  }
}
} // namespace lbmini::openacc::cpu

#endif // LBMINI_OPENACC_CPU_LBMTUBE_HPP_
