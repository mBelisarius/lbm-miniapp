#ifndef LBMINI_OPENACC_GPU_LBMTUBE_HPP_
#define LBMINI_OPENACC_GPU_LBMTUBE_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <openacc.h>
#include <stdexcept>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/OpenAcc/CellPhysics.hpp"

namespace lbmini::openacc::gpu {
/**
 * @brief OpenACC (GPU) variant of the compressible LBM tube solver.
 *
 * Numerically equivalent to `lbmini::plain::LbmTube` (same arithmetic,
 * operation for operation, see `Lbm/OpenAcc/CellPhysics.hpp`); the design is
 * the one of `lbmini::alpaka::LbmTube`.
 *
 * ### State and data layout
 *
 * The state after `n` steps is kept as *post*-collision populations in raw
 * device memory (`acc_malloc`), structure-of-arrays, `cell = i * ny + j`:
 *
 * | buffer      | size    | layout              | role                                |
 * | :---------- | :------ | :------------------ | :---------------------------------- |
 * | `fA/fB`     | `Q * N` | `f[idc * N + cell]` | post-collision density (ping-pong)  |
 * | `gA/gB`     | `Q * N` | `g[idc * N + cell]` | post-collision energy (ping-pong)   |
 * | `xi`        | `2 * N` | `xi[d * N + cell]`  | Newton-Raphson warm start           |
 * | `rho/p/tem` | `N`     | `x[cell]`           | macroscopic output                  |
 * | `u`         | `2 * N` | `u[d * N + cell]`   | macroscopic output                  |
 *
 * A direction plane is contiguous, so consecutive threads of a warp touch
 * consecutive addresses for every load and store. The kernel parameters and
 * the per-direction streaming stencil (`lbmini::DirectionStencil`, built
 * once on the host) are copied once into device memory and passed by pointer.
 *
 * ### Kernels
 *
 * - `Init()`: one kernel computes the Sod left/right state, the equilibrium
 *   seed, the macroscopic fields (visible to the getters) and the first
 *   collision with a cold warm start. Nothing large crosses PCIe.
 * - One time step is one fused kernel per cell: pull-stream the 9 directions
 *   from their 4 IDW source cells, reduce to the macroscopic fields, evaluate
 *   the equilibria, the Newton-Raphson energy closure and the BGK/Knudsen
 *   collision in registers, and write the post-collision populations to the
 *   other ping-pong buffer. The macroscopic fields are written on the last
 *   step of a `Run()` only — the only point where a getter can observe them.
 *
 * All kernels of a `Run()` go to one asynchronous queue: the host enqueues
 * the steps back to back and waits once at the end, so launch latency is
 * hidden and no device <-> host transfer happens inside `Run()`. The ping-pong
 * swap is a host-side pointer swap. Getters copy device -> host on demand.
 *
 * ### Device selection
 *
 * The OpenACC current-device type is per host thread and set by its owner
 * (`lbmini::Executor`). This solver must run on the GPU even when the
 * process also uses the OpenACC multicore backend (both are built into one
 * binary), so every member that touches the device runs under a scope that
 * selects `acc_device_nvidia` and restores the previous type on exit.
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

  // The solver owns raw device allocations: a copy would free them twice.
  LbmTube(const LbmTube&) = delete;

  LbmTube& operator=(const LbmTube&) = delete;

  LbmTube(LbmTube&&) = delete;

  LbmTube& operator=(LbmTube&&) = delete;

  ~LbmTube() override;

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

  /// Asynchronous queue of every kernel of this solver.
  static constexpr int kQueue = 1;
  /// Threads per gang: 128 gives the best occupancy / register balance on sm_89.
  static constexpr int kVectorLength = 128;

  /// Selects the GPU for the current host thread, restores the previous type.
  class DeviceScope {
  public:
    DeviceScope() : previous_(acc_get_device_type()) {
      if (previous_ != acc_device_nvidia)
        acc_set_device_type(acc_device_nvidia);
      // Binds the device context to *this* thread: a thread that did not
      // create the context (a worker, next to a main thread that already
      // used the GPU) has none current and every transfer would fail with
      // CUDA_ERROR_INVALID_CONTEXT.
      acc_set_device_num(acc_get_device_num(acc_device_nvidia), acc_device_nvidia);
    }

    DeviceScope(const DeviceScope&) = delete;

    DeviceScope& operator=(const DeviceScope&) = delete;

    ~DeviceScope() {
      if (previous_ != acc_device_nvidia && previous_ != acc_device_none)
        acc_set_device_type(previous_);
    }

  private:
    acc_device_t previous_;
  };

  /// One fused stream + macroscopic + collision step, enqueued on `kQueue`.
  template<bool kStoreMacro>
  void enqueueStep();

  /// Device -> host copy of `count` scalars (synchronous).
  void copyToHost(Scalar* dst, const Scalar* src, Index count) const;

  /// Allocates `count` scalars of device memory (throws on failure).
  static Scalar* allocate(Index count);

  void release() noexcept;

  const FluidData<Scalar> kFluid_;
  const MeshData<Scalar, kDim_> kMesh_;
  const ControlData<Scalar> kControl_;
  const PerformanceData kPerformance_;

  Index nx_;
  Index ny_;
  Index N_;

  Params params_;

  // Device-resident buffers (acc_malloc).
  Params* paramsDev_ = nullptr;
  Scalar* fA_ = nullptr;
  Scalar* fB_ = nullptr;
  Scalar* gA_ = nullptr;
  Scalar* gB_ = nullptr;
  Scalar* xi_ = nullptr;
  Scalar* rho_ = nullptr;
  Scalar* p_ = nullptr;
  Scalar* tem_ = nullptr;
  Scalar* u_ = nullptr;

  // Ping-pong roles of the distribution buffers (swapped, never copied).
  Scalar* fCur_ = nullptr;
  Scalar* fNext_ = nullptr;
  Scalar* gCur_ = nullptr;
  Scalar* gNext_ = nullptr;

  // Host staging for the SoA -> AoS conversion of U().
  mutable lbmini::DeviceBuffer<Scalar> uHost_;
};

template<typename Scalar, typename LatticeType>
LbmTube<Scalar, LatticeType>::LbmTube(
  const FluidData<Scalar>& fluid,
  const MeshData<Scalar, kDim_>& mesh,
  const ControlData<Scalar>& control,
  const PerformanceData& performance
)
  : kFluid_(fluid), kMesh_(mesh), kControl_(control), kPerformance_(performance) {
  ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "openacc::gpu::LbmTube");

  nx_ = mesh.size[0];
  ny_ = mesh.size[1];
  N_ = nx_ * ny_;
  params_ = directive::MakeCellParams<Scalar, LatticeType>(fluid, mesh, control);

  DeviceScope scope;
  try {
    paramsDev_ = static_cast<Params*>(acc_malloc(sizeof(Params)));
    if (!paramsDev_)
      throw std::runtime_error("openacc::gpu::LbmTube: acc_malloc failed for the kernel parameters.");
    fA_ = allocate(kQ * N_);
    fB_ = allocate(kQ * N_);
    gA_ = allocate(kQ * N_);
    gB_ = allocate(kQ * N_);
    xi_ = allocate(kD * N_);
    rho_ = allocate(N_);
    p_ = allocate(N_);
    tem_ = allocate(N_);
    u_ = allocate(kD * N_);
    acc_memcpy_to_device(paramsDev_, &params_, sizeof(Params));
    uHost_.assign(static_cast<std::size_t>(kD * N_), Scalar{ 0 });
  } catch (...) {
    release();
    throw;
  }

  fCur_ = fA_;
  fNext_ = fB_;
  gCur_ = gA_;
  gNext_ = gB_;
}

template<typename Scalar, typename LatticeType>
LbmTube<Scalar, LatticeType>::~LbmTube() {
  DeviceScope scope;
  // Nothing may still be reading or writing a buffer when it is released.
  #pragma acc wait(kQueue)
  release();
}

template<typename Scalar, typename LatticeType>
Scalar* LbmTube<Scalar, LatticeType>::allocate(const Index count) {
  auto* ptr = static_cast<Scalar*>(acc_malloc(static_cast<std::size_t>(count) * sizeof(Scalar)));
  if (!ptr)
    throw std::runtime_error(
      "openacc::gpu::LbmTube: acc_malloc of " + std::to_string(count) + " values failed (out of device memory?)."
    );
  return ptr;
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::release() noexcept {
  for (Scalar** buf : { &fA_, &fB_, &gA_, &gB_, &xi_, &rho_, &p_, &tem_, &u_ }) {
    if (*buf)
      acc_free(*buf);
    *buf = nullptr;
  }
  if (paramsDev_)
    acc_free(paramsDev_);
  paramsDev_ = nullptr;
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::copyToHost(Scalar* dst, const Scalar* src, const Index count) const {
  DeviceScope scope;
  #pragma acc wait(kQueue)
  acc_memcpy_from_device(dst, const_cast<Scalar*>(src), static_cast<std::size_t>(count) * sizeof(Scalar));
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::P() const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(nx_, ny_);
  copyToHost(out.data(), p_, N_);
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::Rho() const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(nx_, ny_);
  copyToHost(out.data(), rho_, N_);
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::T() const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(nx_, ny_);
  copyToHost(out.data(), tem_, N_);
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::U() const -> Tensor<Scalar, kDim_ + 1> {
  copyToHost(uHost_.data(), u_, kD * N_);
  // SoA d-major -> RowMajor [i][j][d] Eigen tensor.
  Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
  Scalar* dst = out.data();
  for (Index cell = 0; cell < N_; ++cell)
    for (Index d = 0; d < kD; ++d)
      dst[cell * kD + d] = uHost_[d * N_ + cell];
  return out;
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Init() {
  DeviceScope scope;
  // The buffers are only ever written by kernels, so there is no stale state
  // to clear: a re-Init() simply restarts from the initial condition.
  fCur_ = fA_;
  fNext_ = fB_;
  gCur_ = gA_;
  gNext_ = gB_;

  const Params* __restrict__ prm = paramsDev_;
  Scalar* __restrict__ fOut = fCur_;
  Scalar* __restrict__ gOut = gCur_;
  Scalar* __restrict__ xiOut = xi_;
  Scalar* __restrict__ rhoOut = rho_;
  Scalar* __restrict__ pOut = p_;
  Scalar* __restrict__ temOut = tem_;
  Scalar* __restrict__ uOut = u_;
  const int nx = static_cast<int>(nx_);
  const int ny = static_cast<int>(ny_);
  const int N = static_cast<int>(N_);

  #pragma acc wait(kQueue)
  #pragma acc parallel loop gang vector collapse(2) vector_length(kVectorLength) async(kQueue) \
    deviceptr(prm, fOut, gOut, xiOut, rhoOut, pOut, temOut, uOut)
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
  #pragma acc wait(kQueue)
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Step(const bool save) {
  Run(1, save);
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Run(const Index steps, bool /*save*/) {
  if (steps <= 0)
    return;
  DeviceScope scope;
  for (Index t = 0; t < steps; ++t) {
    if (t + 1 == steps)
      enqueueStep<true>();
    else
      enqueueStep<false>();
    std::swap(fCur_, fNext_);
    std::swap(gCur_, gNext_);
  }
  // One synchronisation per Run(): the caller's wall clock covers the device
  // work, and the getters see the final state.
  #pragma acc wait(kQueue)
}

template<typename Scalar, typename LatticeType>
template<bool kStoreMacro>
void LbmTube<Scalar, LatticeType>::enqueueStep() {
  const Params* __restrict__ prm = paramsDev_;
  const Scalar* __restrict__ fIn = fCur_;
  const Scalar* __restrict__ gIn = gCur_;
  Scalar* __restrict__ fOut = fNext_;
  Scalar* __restrict__ gOut = gNext_;
  Scalar* __restrict__ xiIo = xi_;
  Scalar* __restrict__ rhoOut = rho_;
  Scalar* __restrict__ pOut = p_;
  Scalar* __restrict__ temOut = tem_;
  Scalar* __restrict__ uOut = u_;
  const int nx = static_cast<int>(nx_);
  const int ny = static_cast<int>(ny_);
  const int N = static_cast<int>(N_);

  #pragma acc parallel loop gang vector collapse(2) vector_length(kVectorLength) async(kQueue) \
    deviceptr(prm, fIn, gIn, fOut, gOut, xiIo, rhoOut, pOut, temOut, uOut)
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
} // namespace lbmini::openacc::gpu

#endif // LBMINI_OPENACC_GPU_LBMTUBE_HPP_
