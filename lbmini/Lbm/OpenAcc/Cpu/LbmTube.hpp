#ifndef LBMINI_OPENACC_CPU_LBMTUBE_HPP_
#define LBMINI_OPENACC_CPU_LBMTUBE_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <openacc.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/OpenAcc/CellPhysics.hpp"

namespace lbmini::openacc::cpu {
/**
 * @brief OpenACC multi-core variant of the compressible LBM tube solver.
 *
 * Numerically equivalent to `lbmini::plain::LbmTube`: the per-cell arithmetic
 * is the one of `lbmini::directive::Cell` (`Lbm/OpenAcc/CellPhysics.hpp`),
 * shared operation for operation with `openacc::gpu` and `openmp::gpu`, and
 * the streaming uses the compact `lbmini::DirectionStencil` instead of
 * per-cell index/weight tables.
 *
 * ### State and data layout
 *
 * The state after `n` steps is the set of *post*-collision populations in
 * host memory, structure-of-arrays, `cell = i * ny + j`:
 *
 * | buffer      | size    | layout              | role                                |
 * | :---------- | :------ | :------------------ | :---------------------------------- |
 * | `fA/fB`     | `Q * N` | `f[idc * N + cell]` | post-collision density (ping-pong)  |
 * | `gA/gB`     | `Q * N` | `g[idc * N + cell]` | post-collision energy (ping-pong)   |
 * | `xi`        | `2 * N` | `xi[d * N + cell]`  | Newton-Raphson warm start           |
 * | `rho/p/tem` | `N`     | `x[cell]`           | macroscopic output                  |
 * | `u`         | `2 * N` | `u[d * N + cell]`   | macroscopic output                  |
 *
 * ### Kernels
 *
 * - `Init()`: one gang-vector loop computes the Sod state, the equilibrium
 *   seed, the macroscopic fields and the first collision of every cell.
 * - One time step is one fused gang-vector loop per cell: pull-stream the 9
 *   directions from their 4 IDW source cells, reduce to the macroscopic
 *   fields, evaluate the equilibria, the Newton-Raphson closure and the
 *   BGK/Knudsen collision in registers and write the post-collision
 *   populations to the other ping-pong buffer. The macroscopic fields are
 *   written on the last step of a `Run()` only. The ping-pong swap is a
 *   host-side pointer swap; nothing is copied and nothing is allocated.
 *
 * ### Data mapping and device selection
 *
 * The test and benchmark binaries are built with both `-acc=multicore` and
 * `-acc=gpu`, so every compute region exists in a host and in a GPU version
 * and the *runtime* current device type selects one. This solver owns plain
 * host memory, so every member that runs a region does so under a scope that
 * selects `acc_device_host` (restoring the previous type on exit): on the host
 * device the host pointers are the device pointers and `deviceptr` is exact —
 * no present-table lookup, no implicit copy, and no way for the region to see
 * an unmapped (uninitialised) buffer, which is what used to surface as `-nan`
 * when the pointer members were captured by a region that ran on the GPU.
 *
 * Thread count and thread/memory affinity are process-global settings owned
 * by `lbmini::Executor` (`cores`, `ACC_NUM_CORES`); the constructor touches
 * none of it.
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
  using Params = directive::CellParams<Scalar>;
  using Cell = directive::Cell<Scalar>;
  static constexpr int kQ = directive::kQ;
  static constexpr int kD = directive::kD;

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

  const FluidData<Scalar> kFluid_;
  const MeshData<Scalar, kDim_> kMesh_;
  const ControlData<Scalar> kControl_;
  const PerformanceData kPerformance_;

  Index nx_;
  Index ny_;
  Index N_;

  Params params_;

  lbmini::DeviceBuffer<Scalar> fA_;
  lbmini::DeviceBuffer<Scalar> fB_;
  lbmini::DeviceBuffer<Scalar> gA_;
  lbmini::DeviceBuffer<Scalar> gB_;
  lbmini::DeviceBuffer<Scalar> xi_;
  lbmini::DeviceBuffer<Scalar> rho_;
  lbmini::DeviceBuffer<Scalar> p_;
  lbmini::DeviceBuffer<Scalar> tem_;
  lbmini::DeviceBuffer<Scalar> u_;

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
  N_ = nx_ * ny_;
  params_ = directive::MakeCellParams<Scalar, LatticeType>(fluid, mesh, control);

  const auto n = static_cast<std::size_t>(N_);
  fA_.assign(kQ * n, Scalar{ 0 });
  fB_.assign(kQ * n, Scalar{ 0 });
  gA_.assign(kQ * n, Scalar{ 0 });
  gB_.assign(kQ * n, Scalar{ 0 });
  xi_.assign(kD * n, Scalar{ 0 });
  rho_.assign(n, Scalar{ 0 });
  p_.assign(n, Scalar{ 0 });
  tem_.assign(n, Scalar{ 0 });
  u_.assign(kD * n, Scalar{ 0 });

  fCur_ = fA_.data();
  fNext_ = fB_.data();
  gCur_ = gA_.data();
  gNext_ = gB_.data();
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::P() const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::memcpy(out.data(), p_.data(), static_cast<std::size_t>(N_) * sizeof(Scalar));
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::Rho() const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::memcpy(out.data(), rho_.data(), static_cast<std::size_t>(N_) * sizeof(Scalar));
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::T() const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::memcpy(out.data(), tem_.data(), static_cast<std::size_t>(N_) * sizeof(Scalar));
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::U() const -> Tensor<Scalar, kDim_ + 1> {
  // SoA d-major -> RowMajor [i][j][d] Eigen tensor.
  Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
  Scalar* dst = out.data();
  for (Index cell = 0; cell < N_; ++cell)
    for (Index d = 0; d < kD; ++d)
      dst[cell * kD + d] = u_[static_cast<std::size_t>(d * N_ + cell)];
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

  const Params* __restrict__ prm = &params_;
  Scalar* __restrict__ fOut = fCur_;
  Scalar* __restrict__ gOut = gCur_;
  Scalar* __restrict__ xiOut = xi_.data();
  Scalar* __restrict__ rhoOut = rho_.data();
  Scalar* __restrict__ pOut = p_.data();
  Scalar* __restrict__ temOut = tem_.data();
  Scalar* __restrict__ uOut = u_.data();
  const int nx = static_cast<int>(nx_);
  const int ny = static_cast<int>(ny_);
  const int N = static_cast<int>(N_);

  #pragma acc parallel loop gang vector collapse(2) deviceptr(prm, fOut, gOut, xiOut, rhoOut, pOut, temOut, uOut)
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      const int cell = i * ny + j;
      Scalar f[kQ];
      Scalar g[kQ];
      Scalar xi[kD];
      Scalar rho;
      Scalar ux;
      Scalar uy;
      Scalar tem;
      Scalar pres;
      Cell::Init(*prm, i, f, g, xi, rho, ux, uy, tem, pres);
      rhoOut[cell] = rho;
      pOut[cell] = pres;
      temOut[cell] = tem;
      uOut[0 * N + cell] = ux;
      uOut[1 * N + cell] = uy;
      for (int idc = 0; idc < kQ; ++idc) {
        fOut[idc * N + cell] = f[idc];
        gOut[idc * N + cell] = g[idc];
      }
      xiOut[0 * N + cell] = xi[0];
      xiOut[1 * N + cell] = xi[1];
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
  const Params* __restrict__ prm = &params_;
  const Scalar* __restrict__ fIn = fCur_;
  const Scalar* __restrict__ gIn = gCur_;
  Scalar* __restrict__ fOut = fNext_;
  Scalar* __restrict__ gOut = gNext_;
  Scalar* __restrict__ xiIo = xi_.data();
  Scalar* __restrict__ rhoOut = rho_.data();
  Scalar* __restrict__ pOut = p_.data();
  Scalar* __restrict__ temOut = tem_.data();
  Scalar* __restrict__ uOut = u_.data();
  const int nx = static_cast<int>(nx_);
  const int ny = static_cast<int>(ny_);
  const int N = static_cast<int>(N_);

  #pragma acc parallel loop gang vector collapse(2) deviceptr(prm, fIn, gIn, fOut, gOut, xiIo, rhoOut, pOut, temOut, uOut)
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      const int cell = i * ny + j;

      // Pull the post-collision populations of the previous step.
      Scalar f[kQ];
      Scalar g[kQ];
      Cell::Gather(*prm, fIn, gIn, i, j, f, g);

      Scalar rho;
      Scalar ux;
      Scalar uy;
      Scalar tem;
      Scalar pres;
      Cell::Macroscopic(f, g, *prm, rho, ux, uy, tem, pres);
      if constexpr (kStoreMacro) {
        rhoOut[cell] = rho;
        pOut[cell] = pres;
        temOut[cell] = tem;
        uOut[0 * N + cell] = ux;
        uOut[1 * N + cell] = uy;
      }

      Scalar xi[kD] = { xiIo[0 * N + cell], xiIo[1 * N + cell] };
      Cell::Collide(*prm, rho, ux, uy, tem, f, g, xi);

      for (int idc = 0; idc < kQ; ++idc) {
        fOut[idc * N + cell] = f[idc];
        gOut[idc * N + cell] = g[idc];
      }
      xiIo[0 * N + cell] = xi[0];
      xiIo[1 * N + cell] = xi[1];
    }
  }
}
} // namespace lbmini::openacc::cpu

#endif // LBMINI_OPENACC_CPU_LBMTUBE_HPP_
