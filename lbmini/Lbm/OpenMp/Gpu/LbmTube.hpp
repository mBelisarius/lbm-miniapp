#ifndef LBMINI_OPENMP_GPU_LBMTUBE_HPP_
#define LBMINI_OPENMP_GPU_LBMTUBE_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/OpenAcc/CellPhysics.hpp"

namespace lbmini::openmp::gpu {
/**
 * @brief OpenMP-target (GPU) variant of the compressible LBM tube solver.
 *
 * Numerically equivalent to `lbmini::plain::LbmTube` (same arithmetic,
 * operation for operation, see `Lbm/OpenAcc/CellPhysics.hpp`); the design is
 * the one of `lbmini::alpaka::LbmTube` and `lbmini::openacc::gpu::LbmTube`.
 *
 * ### State and data layout
 *
 * The state after `n` steps is kept as *post*-collision populations in raw
 * device memory (`omp_target_alloc`), structure-of-arrays, `cell = i * ny + j`:
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
 * once on the host) are copied once into device memory and passed by pointer
 * via `is_device_ptr` — no per-launch `map(...)` of the ~0.6 kB block.
 *
 * ### Kernels
 *
 * - `Init()`: one target region computes the Sod left/right state, the
 *   equilibrium seed, the macroscopic fields (visible to the getters) and
 *   the first collision with a cold warm start.
 * - One time step is one fused target region per cell: pull-stream the 9
 *   directions from their 4 IDW source cells (`DirectionStencil`), reduce to
 *   the macroscopic fields, evaluate the equilibria, the Newton-Raphson
 *   energy closure and the BGK/Knudsen collision in registers, and write the
 *   post-collision populations to the other ping-pong buffer. Macroscopic
 *   fields are written on the last step of a `Run()` only.
 *
 * The ping-pong swap is a host-side pointer swap. Getters copy device -> host
 * on demand. There is no host-side `#pragma omp parallel` region: the host
 * thread count is left to `lbmini::Executor`.
 *
 * ### Launch geometry
 *
 * `thread_limit(128)`; the fused kernel uses ~192-200 regs/thread (no spills)
 * on sm_89, so only 2 teams fit per SM (256/1536 threads, ~17% occupancy).
 * NF-2 (<=128 regs) is unmet. A `maxregcount:128` cap (experimental, not
 * enabled in CMake) spills ~280B and was ~2.5% slower at 800k cells (within
 * noise at 16k cells). `num_teams` is one team per 128-cell tile.
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

  // The solver owns raw `omp_target_alloc` device pointers: a copy would
  // `omp_target_free` them twice. See docs/memory-safety.md §5.2.
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
  static constexpr int kThreadLimit = 128;

  /// One fused stream + macroscopic + collision step.
  template<bool kStoreMacro>
  void step();

  /// Device -> host copy of `count` scalars (synchronous).
  void copyToHost(Scalar* dst, const Scalar* src, Index count) const;

  /// Allocates `count` scalars of device memory (throws on failure).
  Scalar* allocate(Index count);

  void release() noexcept;

  const FluidData<Scalar> kFluid_;
  const MeshData<Scalar, kDim_> kMesh_;
  const ControlData<Scalar> kControl_;
  const PerformanceData kPerformance_;

  Index nx_;
  Index ny_;
  Index N_;
  int dev_;
  int host_;

  Params params_;

  // Device-resident buffers (omp_target_alloc).
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
  ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "openmp::gpu::LbmTube");

  nx_ = mesh.size[0];
  ny_ = mesh.size[1];
  N_ = nx_ * ny_;
  params_ = directive::MakeCellParams<Scalar, LatticeType>(fluid, mesh, control);
  dev_ = omp_get_default_device();
  host_ = omp_get_initial_device();

  try {
    paramsDev_ = static_cast<Params*>(omp_target_alloc(sizeof(Params), dev_));
    if (!paramsDev_)
      throw std::runtime_error("openmp::gpu::LbmTube: omp_target_alloc failed for the kernel parameters.");
    fA_ = allocate(kQ * N_);
    fB_ = allocate(kQ * N_);
    gA_ = allocate(kQ * N_);
    gB_ = allocate(kQ * N_);
    xi_ = allocate(kD * N_);
    rho_ = allocate(N_);
    p_ = allocate(N_);
    tem_ = allocate(N_);
    u_ = allocate(kD * N_);

    if (omp_target_memcpy(paramsDev_, &params_, sizeof(Params), 0, 0, dev_, host_) != 0)
      throw std::runtime_error("openmp::gpu::LbmTube: omp_target_memcpy of kernel parameters failed.");

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
  release();
}

template<typename Scalar, typename LatticeType>
Scalar* LbmTube<Scalar, LatticeType>::allocate(const Index count) {
  auto* ptr = static_cast<Scalar*>(
    omp_target_alloc(static_cast<std::size_t>(count) * sizeof(Scalar), dev_)
  );
  if (!ptr)
    throw std::runtime_error(
      "openmp::gpu::LbmTube: omp_target_alloc of " + std::to_string(count)
      + " values failed on device " + std::to_string(dev_) + " (out of device memory?)."
    );
  return ptr;
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::release() noexcept {
  for (Scalar** buf : { &fA_, &fB_, &gA_, &gB_, &xi_, &rho_, &p_, &tem_, &u_ }) {
    if (*buf)
      omp_target_free(*buf, dev_);
    *buf = nullptr;
  }
  if (paramsDev_)
    omp_target_free(paramsDev_, dev_);
  paramsDev_ = nullptr;
  fCur_ = fNext_ = gCur_ = gNext_ = nullptr;
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::copyToHost(Scalar* dst, const Scalar* src, const Index count) const {
  if (omp_target_memcpy(
    dst,
    const_cast<Scalar*>(src),
    static_cast<std::size_t>(count) * sizeof(Scalar),
    0,
    0,
    host_,
    dev_
  ) != 0) {
    throw std::runtime_error("openmp::gpu::LbmTube: omp_target_memcpy from device failed.");
  }
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
  Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
  Scalar* dst = out.data();
  for (Index cell = 0; cell < N_; ++cell)
    for (Index d = 0; d < kD; ++d)
      dst[cell * kD + d] = uHost_[d * N_ + cell];
  return out;
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Init() {
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
  const int teams = (N + kThreadLimit - 1) / kThreadLimit;

  #pragma omp target teams distribute parallel for collapse(2) \
    num_teams(teams) thread_limit(kThreadLimit) \
    is_device_ptr(prm, fOut, gOut, xiOut, rhoOut, pOut, temOut, uOut) \
    firstprivate(nx, ny, N)
  for (int ci = 0; ci < nx; ++ci) {
    for (int cj = 0; cj < ny; ++cj) {
      const int cell = ci * ny + cj;
      Scalar f[kQ];
      Scalar g[kQ];
      Scalar xi[kD];
      Scalar rho;
      Scalar ux;
      Scalar uy;
      Scalar tem;
      Scalar pres;
      Cell::Init(*prm, ci, f, g, xi, rho, ux, uy, tem, pres);
      rhoOut[cell] = rho;
      pOut[cell] = pres;
      temOut[cell] = tem;
      uOut[0 * N + cell] = ux;
      uOut[1 * N + cell] = uy;
      LBMINI_UNROLL(9)
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
  const int teams = (N + kThreadLimit - 1) / kThreadLimit;

  #pragma omp target teams distribute parallel for collapse(2) \
    num_teams(teams) thread_limit(kThreadLimit) \
    is_device_ptr(prm, fIn, gIn, fOut, gOut, xiIo, rhoOut, pOut, temOut, uOut) \
    firstprivate(nx, ny, N)
  for (int ci = 0; ci < nx; ++ci) {
    for (int cj = 0; cj < ny; ++cj) {
      const int cell = ci * ny + cj;

      Scalar f[kQ];
      Scalar g[kQ];
      Cell::Gather(*prm, fIn, gIn, ci, cj, f, g);

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

      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc) {
        fOut[idc * N + cell] = f[idc];
        gOut[idc * N + cell] = g[idc];
      }
      xiIo[0 * N + cell] = xi[0];
      xiIo[1 * N + cell] = xi[1];
    }
  }
}
} // namespace lbmini::openmp::gpu

#endif // LBMINI_OPENMP_GPU_LBMTUBE_HPP_
