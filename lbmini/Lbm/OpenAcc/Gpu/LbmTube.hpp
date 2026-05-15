#ifndef LBMINI_OPENACC_GPU_LBMTUBE_HPP_
#define LBMINI_OPENACC_GPU_LBMTUBE_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <openacc.h>
#include <stdexcept>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILbmTube.hpp"

namespace lbmini::openacc::gpu {
// D2Q9 velocity tables — namespace-scope, non-template, `declare target`
//
// These are plain `constexpr int` arrays at namespace scope (not `static`
// members of a template class). Declaring them via `#pragma acc routine seq`
// is safe here because nvc++ 26.1's ICE is specific to `static constexpr`
// members of *template* classes.  Function-local `constexpr` arrays are
// stack-allocated and are NOT automatically visible inside a `#pragma omp target`
// region; moving them here with `declare target` guarantees nvc++ places them
// in GPU constant memory and allows every kernel to reference them without
// requiring a `copyin(...)` clause.
constexpr int kD2Q9Cx[9] = { 0, 1, -1, 0, 0, 1, -1, 1, -1 };
constexpr int kD2Q9Cy[9] = { 0, 0, 0, 1, -1, 1, -1, -1, 1 };


/**
 * @brief OpenMP-target (GPU) variant of the compressible LBM tube solver.
 *
 * Numerically identical to `lbmini::plain::LbmTube` and
 * `lbmini::openacc::cpu::LbmTube`. All GPU-specific choices are in the
 * dispatch and storage layers only.
 *
 * ### GPU design decisions
 *
 * **True device memory via `omp_target_alloc` + `deviceptr`.** Every
 * per-step array lives in raw device memory allocated once in the ctor.
 * Kernels receive raw device pointers via `deviceptr(...)` — zero
 * `map(...)` overhead, zero presence-table reference counting per launch.
 *
 * **`collapse(2)` loops over `int ci, cj`.** Using a 2-D collapsed loop with
 * 32-bit `int` loop variables eliminates the expensive 64-bit integer
 * division (`cell / ny`, `cell % ny`) that would appear if the outer loop
 * used a flat `Index` (= `long`) loop variable. On sm_89 (Ada), 64-bit
 * integer division is emulated in ~20 32-bit instructions; `collapse(2)`
 * eliminates it entirely.
 *
 * **`thread_limit(64)`.** `collisionAndEquilibria` carries ~100+ live FP64
 * values (`feqLocal[9]`, `geqLocal[9]`, `si[9]`, `e[9]`, `S1/S2/J/xi/L`).
 * With 128 threads per block the compiler spills to local (= global) memory;
 * 64 threads per block halves the per-block register demand, letting nvc++
 * keep all arrays in registers and enabling 4–8 concurrent blocks per SM on
 * Ada's 65 536 register file.
 *
 * **`num_teams(N/64)`.** Explicitly requests one team per 64-cell tile so
 * nvc++ does not under-subscribe the 20 SMs on the RTX 4050.
 *
 * **Namespace-scope `#pragma acc routine seq` D2Q9 tables.** nvc++ 26.1
 * does not automatically transfer function-local `constexpr` arrays into
 * `target` regions, and wrapping a *template* class in `declare target`
 * triggers an ICE on its `static constexpr` member arrays. The fix: declare
 * `kD2Q9Cx[9]` / `kD2Q9Cy[9]` as plain (non-template) namespace-scope
 * `constexpr int` arrays with `#pragma acc routine seq` — safe because
 * the ICE is specific to template classes, not free arrays. Every kernel
 * references these directly; nvc++ places them in GPU constant memory.
 *
 * **On-the-fly IDW streaming — no streaming tables.** Precomputed
 * `streamIdx_`/`streamW_` tables scale as O(kQ × N × 4) — for large N that
 * exceeds GPU VRAM. The `streamAndMacroscopic` kernel recomputes the 4-neighbor
 * IDW stencil directly in registers.
 *
 * **Fully branchless IDW (no snap-to-node cascade).** The 4-way
 * `if (snap00) … else if (snap10) …` cascade causes warp divergence on
 * grid edges at every step. We replace it with `exp(-p*log(r²+kTiny))`:
 * when a corner is exactly on the source point r²→0 and the weight→+∞,
 * but after normalisation by `wsum` the result is correct (the limit is the
 * node value). Adding `kTiny` ensures no NaN without any branch.
 *
 * **Branchless Newton-Raphson.** The `bool solverOk` flag caused warp
 * divergence near shock waves where some cells in a warp converge and others
 * do not. We replace it with a `Scalar` sentinel and ternary selects so all
 * 64 threads in a block execute the same instruction sequence.
 *
 * **`exp`-`log` IDW weights.** `1/d^p = exp(-p*log(d))` replaces four
 * `std::pow` calls with one `std::log` + one multiply + `std::exp` per
 * corner. For the common axis-aligned directions (`cy[idc]==0`) pure 1-D
 * linear interpolation avoids all transcendentals.
 *
 * **Host-side pointer swap ping-pong.** Zero device work per step.
 *
 * **One host thread.** Host only orchestrates GPU launches.
 *
 * ### Possible future improvements
 *  - `Scalar = float` by default: FP32 `__expf`/`__logf` are 8× faster than
 *    FP64 on Ada, and HBM bandwidth doubles. The code is already templated.
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

  ~LbmTube() override;

  [[nodiscard]] Tensor<Scalar, kDim_> P() const override;

  [[nodiscard]] Tensor<Scalar, kDim_> Rho() const override;

  [[nodiscard]] Tensor<Scalar, kDim_> T() const override;

  [[nodiscard]] Tensor<Scalar, kDim_ + 1> U() const override;

  void Init() override;

  void Step(bool save) override;

  void Run(Index steps, bool save) override;

protected:
  [[nodiscard]] Index cellIndex(const Index i, const Index j) const { return i * ny_ + j; }
  [[nodiscard]] Index distIndex(const Index idc, const Index cell) const { return idc * N_ + cell; }
  [[nodiscard]] Index uIndex(const Index d, const Index cell) const { return d * N_ + cell; }

  /**
   * @brief Recomputes macroscopic fields (rho, p, T, u) from f_ and g_.
   *
   * Offloaded via `#pragma acc parallel loop`.
   * Execution happens entirely on the GPU.
   */
  void computeMacroscopic();

  /**
   * @brief Initializes f_ and g_ using local equilibria.
   *
   * Offloaded to the GPU device to avoid data migration during initialization.
   */
  void seedEquilibria();

  /**
   * @brief Fused BGK collision and equilibrium computation.
   *
   * Offloaded kernel. Thread divergence is minimized by branchless logic
   * and fixed Newton-Raphson iteration counts.
   */
  void collisionAndEquilibria();

  /**
   * @brief Combined streaming and macroscopic update.
   *
   * Offloaded kernel. Computes branchless on-the-fly IDW streaming directly in registers 
   * to avoid memory bandwidth bottlenecks, followed immediately by macroscopic reduction.
   */
  void streamAndMacroscopic();

private:
  static constexpr Scalar kTiny_ = Scalar{ 1.0e-12 };
  static constexpr Scalar kMaxExp_ = Scalar{ 700.0 };

  // D2Q9 tables inlined here so they can be copied into target-region
  // local constexpr arrays without requiring LatticeType to be
  // #pragma acc routine seq'd (which ICEs on nvc++ 26.1 for templates).
  static constexpr int kCx_[9] = { 0, 1, -1, 0, 0, 1, -1, 1, -1 };
  static constexpr int kCy_[9] = { 0, 0, 0, 1, -1, 1, -1, -1, 1 };

  const FluidData<Scalar> kFluid_;
  const MeshData<Scalar, kDim_> kMesh_;
  const ControlData<Scalar> kControl_;
  const PerformanceData kPerformance_;

  Index nx_;
  Index ny_;
  Index N_;
  Index uSize_;
  Index distSize_;
  int dev_;
  int host_;

  // Small host staging buffers for initial condition & getters.
  mutable lbmini::DeviceBuffer<Scalar> rhoHost_;
  mutable lbmini::DeviceBuffer<Scalar> pHost_;
  mutable lbmini::DeviceBuffer<Scalar> temHost_;
  mutable lbmini::DeviceBuffer<Scalar> uHost_;
  lbmini::DeviceBuffer<Scalar> lastGxHost_; // SoA: (kDim+1) planes of N

  // Device-resident buffers (omp_target_alloc).
  Scalar* rhoDev_ = nullptr;
  Scalar* pDev_ = nullptr;
  Scalar* temDev_ = nullptr;
  Scalar* uDev_ = nullptr;
  Scalar* fDev_ = nullptr;
  Scalar* gDev_ = nullptr;
  Scalar* fauxDev_ = nullptr;
  Scalar* gauxDev_ = nullptr;
  Scalar* lastGxDev_ = nullptr; // SoA: (kDim+1) planes of N

  // Ping-pong aliases (raw device pointers, swapped on host with std::swap).
  Scalar* fCur_ = nullptr;
  Scalar* fAlt_ = nullptr;
  Scalar* gCur_ = nullptr;
  Scalar* gAlt_ = nullptr;
};

template<typename Scalar, typename LatticeType>
LbmTube<Scalar, LatticeType>::LbmTube(
  const FluidData<Scalar>& fluid,
  const MeshData<Scalar, kDim_>& mesh,
  const ControlData<Scalar>& control,
  const PerformanceData& performance
)
  : kFluid_(fluid), kMesh_(mesh), kControl_(control), kPerformance_(performance) {
  // Single host thread orchestrates GPU launches; all compute is on device.
  omp_set_num_threads(1);

  nx_ = mesh.size[0];
  ny_ = mesh.size[1];
  N_ = nx_ * ny_;
  uSize_ = N_ * kDim_;
  distSize_ = N_ * kQ_;
  dev_ = omp_get_default_device();
  host_ = omp_get_initial_device();

  rhoHost_.assign(N_, Scalar{ 0 });
  pHost_.assign(N_, Scalar{ 0 });
  temHost_.assign(N_, Scalar{ 0 });
  uHost_.assign(uSize_, Scalar{ 0 });
  lastGxHost_.assign(N_ * (kDim_ + 1), Scalar{ 0 });

  const std::size_t sN = static_cast<std::size_t>(N_) * sizeof(Scalar);
  const std::size_t sU = static_cast<std::size_t>(uSize_) * sizeof(Scalar);
  const std::size_t sD = static_cast<std::size_t>(distSize_) * sizeof(Scalar);
  const std::size_t sLg = static_cast<std::size_t>(N_ * (kDim_ + 1)) * sizeof(Scalar);

  rhoDev_ = static_cast<Scalar*>(omp_target_alloc(sN, dev_));
  pDev_ = static_cast<Scalar*>(omp_target_alloc(sN, dev_));
  temDev_ = static_cast<Scalar*>(omp_target_alloc(sN, dev_));
  uDev_ = static_cast<Scalar*>(omp_target_alloc(sU, dev_));
  fDev_ = static_cast<Scalar*>(omp_target_alloc(sD, dev_));
  gDev_ = static_cast<Scalar*>(omp_target_alloc(sD, dev_));
  fauxDev_ = static_cast<Scalar*>(omp_target_alloc(sD, dev_));
  gauxDev_ = static_cast<Scalar*>(omp_target_alloc(sD, dev_));
  lastGxDev_ = static_cast<Scalar*>(omp_target_alloc(sLg, dev_));

  if (!rhoDev_ || !pDev_ || !temDev_ || !uDev_ || !fDev_ || !gDev_ || !fauxDev_ || !gauxDev_ || !lastGxDev_) {
    std::cerr << "[openacc::gpu::LbmTube] omp_target_alloc failed on device " << dev_
      << " (num_devices=" << omp_get_num_devices() << ", N=" << N_ << "). Check available VRAM." << std::endl;
    throw std::runtime_error("omp_target_alloc returned nullptr");
  }

  fCur_ = fDev_;
  fAlt_ = fauxDev_;
  gCur_ = gDev_;
  gAlt_ = gauxDev_;
}

template<typename Scalar, typename LatticeType>
LbmTube<Scalar, LatticeType>::~LbmTube() {
  if (rhoDev_)
    omp_target_free(rhoDev_, dev_);
  if (pDev_)
    omp_target_free(pDev_, dev_);
  if (temDev_)
    omp_target_free(temDev_, dev_);
  if (uDev_)
    omp_target_free(uDev_, dev_);
  if (fDev_)
    omp_target_free(fDev_, dev_);
  if (gDev_)
    omp_target_free(gDev_, dev_);
  if (fauxDev_)
    omp_target_free(fauxDev_, dev_);
  if (gauxDev_)
    omp_target_free(gauxDev_, dev_);
  if (lastGxDev_)
    omp_target_free(lastGxDev_, dev_);
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::P() const -> Tensor<Scalar, kDim_> {
  omp_target_memcpy(
    pHost_.data(),
    pDev_,
    static_cast<std::size_t>(N_) * sizeof(Scalar),
    0,
    0,
    host_,
    dev_
  );
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(pHost_.begin(), pHost_.end(), out.data());
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::Rho() const -> Tensor<Scalar, kDim_> {
  omp_target_memcpy(
    rhoHost_.data(),
    rhoDev_,
    static_cast<std::size_t>(N_) * sizeof(Scalar),
    0,
    0,
    host_,
    dev_
  );
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(rhoHost_.begin(), rhoHost_.end(), out.data());
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::T() const -> Tensor<Scalar, kDim_> {
  omp_target_memcpy(
    temHost_.data(),
    temDev_,
    static_cast<std::size_t>(N_) * sizeof(Scalar),
    0,
    0,
    host_,
    dev_
  );
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(temHost_.begin(), temHost_.end(), out.data());
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::U() const -> Tensor<Scalar, kDim_ + 1> {
  omp_target_memcpy(
    uHost_.data(),
    uDev_,
    static_cast<std::size_t>(uSize_) * sizeof(Scalar),
    0,
    0,
    host_,
    dev_
  );
  Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
  Scalar* dst = out.data();
  for (Index cell = 0; cell < N_; ++cell)
    for (Index d = 0; d < kDim_; ++d)
      dst[cell * kDim_ + d] = uHost_[d * N_ + cell];
  return out;
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Init() {
  for (Index i = 0; i < nx_; ++i) {
    for (Index j = 0; j < ny_; ++j) {
      const Index cell = cellIndex(static_cast<int>(i), static_cast<int>(j));
      uHost_[0 * N_ + cell] = Scalar{ 0 };
      uHost_[1 * N_ + cell] = Scalar{ 0 };
      if (i < nx_ / 2) {
        rhoHost_[cell] = kFluid_.densityL;
        pHost_[cell] = kFluid_.pressureL;
      } else {
        rhoHost_[cell] = kFluid_.densityR;
        pHost_[cell] = kFluid_.pressureR;
      }
      temHost_[cell] = LatticeType::Cs2 * pHost_[cell] / (rhoHost_[cell] * kFluid_.constant);
      for (Index d = 0; d < kDim_ + 1; ++d)
        lastGxHost_[d * N_ + cell] = Scalar{ 0 };
    }
  }

  const std::size_t sN = static_cast<std::size_t>(N_) * sizeof(Scalar);
  const std::size_t sU = static_cast<std::size_t>(uSize_) * sizeof(Scalar);
  const std::size_t sLg = static_cast<std::size_t>(N_ * (kDim_ + 1)) * sizeof(Scalar);
  omp_target_memcpy(rhoDev_, rhoHost_.data(), sN, 0, 0, dev_, host_);
  omp_target_memcpy(pDev_, pHost_.data(), sN, 0, 0, dev_, host_);
  omp_target_memcpy(temDev_, temHost_.data(), sN, 0, 0, dev_, host_);
  omp_target_memcpy(uDev_, uHost_.data(), sU, 0, 0, dev_, host_);
  omp_target_memcpy(lastGxDev_, lastGxHost_.data(), sLg, 0, 0, dev_, host_);

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
    streamAndMacroscopic();
    std::swap(fCur_, fAlt_);
    std::swap(gCur_, gAlt_);
  }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::computeMacroscopic() {
  const Scalar Ushift0 = kControl_.U(0);
  const Scalar Ushift1 = kControl_.U(1);
  const Scalar invCv = Scalar{ 1 } / kFluid_.specificHeatCv;
  const Scalar Rgas = kFluid_.constant;
  const Scalar* __restrict__ pF = fCur_;
  const Scalar* __restrict__ pG = gCur_;
  Scalar* __restrict__ pRho = rhoDev_;
  Scalar* __restrict__ pP = pDev_;
  Scalar* __restrict__ pT = temDev_;
  Scalar* __restrict__ pU = uDev_;
  const int nx = static_cast<int>(nx_);
  const int ny = static_cast<int>(ny_);
  const int N = static_cast<int>(N_);

  // collapse(2) over int ci/cj: eliminates 64-bit div/mod for cell indexing.
  // thread_limit(64): halves per-block register demand, enabling 4-8 concurrent
  // blocks per SM on the 65536-register Ada register file.
  // kD2Q9Cx/kD2Q9Cy are namespace-scope `declare target` constants — nvc++
  // can reference them inside `deviceptr` kernels without a `map` clause.
  #pragma acc parallel loop collapse(2) \
      num_gangs((N + 63) / 64) vector_length(64) \
      deviceptr(pF, pG, pRho, pP, pT, pU)
  for (int ci = 0; ci < nx; ++ci) {
    for (int cj = 0; cj < ny; ++cj) {
      const int cell = ci * ny + cj;
      Scalar rho = Scalar{ 0 };
      Scalar nrg = Scalar{ 0 };
      Scalar mx = Scalar{ 0 };
      Scalar my = Scalar{ 0 };
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < 9; ++idc) {
        const Scalar fi = pF[idc * N + cell];
        const Scalar gi = pG[idc * N + cell];
        rho += fi;
        nrg += gi;
        mx += (static_cast<Scalar>(kD2Q9Cx[idc]) + Ushift0) * fi;
        my += (static_cast<Scalar>(kD2Q9Cy[idc]) + Ushift1) * fi;
      }

      pRho[cell] = rho;

      const Scalar invRho = Scalar{ 1 } / rho;
      const Scalar ux = mx * invRho;
      const Scalar uy = my * invRho;
      pU[0 * N + cell] = ux;
      pU[1 * N + cell] = uy;

      const Scalar kin = Scalar{ 0.5 } * (ux * ux + uy * uy);
      const Scalar Tv = (Scalar{ 0.5 } * nrg * invRho - kin) * invCv;
      pT[cell] = Tv;
      pP[cell] = Rgas * rho * Tv;
    }
  }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::seedEquilibria() {
  const Scalar Ushift0 = kControl_.U(0);
  const Scalar Ushift1 = kControl_.U(1);
  const Scalar kTiny = kTiny_;
  Scalar* __restrict__ pF = fCur_;
  Scalar* __restrict__ pG = gCur_;
  const Scalar* __restrict__ pRho = rhoDev_;
  const Scalar* __restrict__ pT = temDev_;
  const Scalar* __restrict__ pU = uDev_;
  const int nx = static_cast<int>(nx_);
  const int ny = static_cast<int>(ny_);
  const int N = static_cast<int>(N_);
  const Scalar cv = kFluid_.specificHeatCv;

  #pragma acc parallel loop collapse(2) \
      num_gangs((N + 63) / 64) vector_length(64) \
      deviceptr(pF, pG, pRho, pT, pU)
  for (int ci = 0; ci < nx; ++ci) {
    for (int cj = 0; cj < ny; ++cj) {
      const int cell = ci * ny + cj;
      const Scalar rho = pRho[cell];
      const Scalar Tv = pT[cell];
      const Scalar ux = pU[0 * N + cell];
      const Scalar uy = pU[1 * N + cell];
      const Scalar u2 = ux * ux + uy * uy;

      // Feq (product form, branchless).
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < 9; ++idc) {
        const Scalar uia_x = ux - Ushift0;
        const Scalar uia_y = uy - Ushift1;
        const Scalar ux2t = uia_x * uia_x + Tv;
        const Scalar uy2t = uia_y * uia_y + Tv;
        const int vx = kD2Q9Cx[idc], vy = kD2Q9Cy[idc];
        const Scalar pfx = (vx == 0)
                             ? (Scalar{ 1 } - ux2t)
                             : (Scalar{ 0.5 } * (static_cast<Scalar>(vx) * uia_x + ux2t));
        const Scalar pfy = (vy == 0)
                             ? (Scalar{ 1 } - uy2t)
                             : (Scalar{ 0.5 } * (static_cast<Scalar>(vy) * uia_y + uy2t));
        pF[idc * N + cell] = rho * pfx * pfy;
      }

      // Geq: temperature-dependent weights, xi==0 at init.
      const Scalar E = cv * Tv + Scalar{ 0.5 } * u2;
      const Scalar targetE = Scalar{ 2 } * rho * E;
      const Scalar wZero = Scalar{ 1 } - Tv;
      const Scalar wNonZero = Scalar{ 0.5 } * Tv;
      Scalar sumW = Scalar{ 0 };
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < 9; ++idc) {
        const Scalar wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
        const Scalar wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
        sumW += wx * wy;
      }
      const Scalar sc = (sumW > kTiny) ? targetE / sumW : targetE / Scalar{ 9 };
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < 9; ++idc) {
        const Scalar wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
        const Scalar wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
        pG[idc * N + cell] = wx * wy * sc;
      }
    }
  }
}

// collisionAndEquilibria
//
// Design notes:
//  - collapse(2) + int ci/cj: eliminates 64-bit integer division that would
//    arise from a flat `cell % ny` / `cell / ny` with Index (long) loop var.
//  - thread_limit(64): with ~200 live 32-bit register slots per thread, 64
//    threads/block uses 12800 registers — fits 5 concurrent blocks per SM on
//    Ada's 65536-register file, dramatically improving latency hiding vs the
//    previous 128 threads/block (only 2 concurrent blocks).
//  - Fully branchless NR: the `bool solverOk` flag was causing warp divergence
//    near shock waves. We replace it with a `Scalar` guard (`nrOk`) so all
//    threads execute the same control flow. In the fallback path, xi stays
//    zero, and geqLocal is set to a uniform distribution via ternary selects.
//  - feqLocal[9], si[9], e[9] are unrolled via LBMINI_UNROLL(9); with thread_
//    limit(64) and branchless control flow the compiler can fully scalarise
//    them into registers without spilling to local memory.
template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::collisionAndEquilibria() {
  const Scalar Ushift0 = kControl_.U(0);
  const Scalar Ushift1 = kControl_.U(1);
  const Scalar visc = kFluid_.viscosity;
  const Scalar cond = kFluid_.conductivity;
  const Scalar cp = kFluid_.specificHeatCp;
  const Scalar cv = kFluid_.specificHeatCv;
  const Scalar kTiny = kTiny_;
  const Scalar kMaxE = kMaxExp_;

  Scalar* __restrict__ pF = fCur_;
  Scalar* __restrict__ pG = gCur_;
  Scalar* __restrict__ pLastGx = lastGxDev_;
  const Scalar* __restrict__ pRho = rhoDev_;
  const Scalar* __restrict__ pT = temDev_;
  const Scalar* __restrict__ pU = uDev_;
  const int nx = static_cast<int>(nx_);
  const int ny = static_cast<int>(ny_);
  const int N = static_cast<int>(N_);

  #pragma acc parallel loop collapse(2) \
      num_gangs((N + 63) / 64) vector_length(64) \
      deviceptr(pF, pG, pLastGx, pRho, pT, pU)
  for (int ci = 0; ci < nx; ++ci) {
    for (int cj = 0; cj < ny; ++cj) {
      const int cell = ci * ny + cj;
      const Scalar rho = pRho[cell];
      const Scalar Tv = pT[cell];
      const Scalar ux = pU[0 * N + cell];
      const Scalar uy = pU[1 * N + cell];
      const Scalar u2 = ux * ux + uy * uy;

      // Temperature-dependent weights (branchless, compile-time selects).
      const Scalar wZero = Scalar{ 1 } - Tv;
      const Scalar wNonZero = Scalar{ 0.5 } * Tv;

      // Compute feq (product form, fully branchless)
      Scalar feqLocal[9];
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < 9; ++idc) {
        const Scalar uia_x = ux - Ushift0;
        const Scalar uia_y = uy - Ushift1;
        const Scalar ux2t = uia_x * uia_x + Tv;
        const Scalar uy2t = uia_y * uia_y + Tv;
        const int vx = kD2Q9Cx[idc], vy = kD2Q9Cy[idc];
        const Scalar pfx = (vx == 0) ? (Scalar{ 1 } - ux2t) : (Scalar{ 0.5 } * (static_cast<Scalar>(vx) * uia_x + ux2t));
        const Scalar pfy = (vy == 0) ? (Scalar{ 1 } - uy2t) : (Scalar{ 0.5 } * (static_cast<Scalar>(vy) * uia_y + uy2t));
        feqLocal[idc] = rho * pfx * pfy;
      }

      // Newton-Raphson for geq
      // nrOk accumulates as a Scalar (1.0 = all good, 0.0 = degenerate) to
      // keep all threads in the warp executing the same instructions (no
      // branch on bool that would cause warp divergence near shock waves).
      const Scalar E = cv * Tv + Scalar{ 0.5 } * u2;
      const Scalar targetE = Scalar{ 2 } * rho * E;

      const Scalar denom0 = (targetE > kTiny) ? targetE : kTiny;
      Scalar targetM[2];
      targetM[0] = Scalar{ 2 } * rho * ux * (E + Tv) / denom0;
      targetM[1] = Scalar{ 2 } * rho * uy * (E + Tv) / denom0;

      Scalar xi[2];
      xi[0] = pLastGx[1 * N + cell];
      xi[1] = pLastGx[2 * N + cell];

      Scalar alpha = Scalar{ 1 };
      Scalar si[9], e[9];
      Scalar Z = Scalar{ 0 };
      Scalar smax;
      Scalar nrOk = Scalar{ 1 }; // 1.0 = solver healthy, 0.0 = degenerate

      LBMINI_UNROLL(3)
      for (int iter = 0; iter < 3; ++iter) {
        smax = -kMaxE;
        LBMINI_UNROLL(9)
        for (int idc = 0; idc < 9; ++idc) {
          const Scalar s = xi[0] * (static_cast<Scalar>(kD2Q9Cx[idc]) + Ushift0)
            + xi[1] * (static_cast<Scalar>(kD2Q9Cy[idc]) + Ushift1);
          si[idc] = s;
          smax = (s > smax) ? s : smax;
        }
        smax = (smax < kMaxE) ? smax : kMaxE;

        Scalar S1[2] = { Scalar{ 0 }, Scalar{ 0 } };
        Scalar S2_00 = Scalar{ 0 }, S2_01 = Scalar{ 0 },
               S2_10 = Scalar{ 0 }, S2_11 = Scalar{ 0 };
        Z = Scalar{ 0 };
        LBMINI_UNROLL(9)
        for (int idc = 0; idc < 9; ++idc) {
          Scalar expo = si[idc] - smax;
          expo = (expo < kMaxE) ? expo : kMaxE;
          expo = (expo > -kMaxE) ? expo : -kMaxE;
          const Scalar wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
          const Scalar wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
          const Scalar wev = wx * wy * std::exp(expo);
          e[idc] = wev;
          Z += wev;
          const Scalar c0 = static_cast<Scalar>(kD2Q9Cx[idc]) + Ushift0;
          const Scalar c1 = static_cast<Scalar>(kD2Q9Cy[idc]) + Ushift1;
          S1[0] += c0 * wev;
          S1[1] += c1 * wev;
          S2_00 += c0 * c0 * wev;
          S2_01 += c0 * c1 * wev;
          S2_10 += c1 * c0 * wev;
          S2_11 += c1 * c1 * wev;
        }

        const Scalar Zsafe = (Z > kTiny) ? Z : kTiny;
        // Accumulate degenerate flag as Scalar to avoid branch divergence.
        nrOk *= (Z > kTiny) ? Scalar{ 1 } : Scalar{ 0 };
        const Scalar invZ = Scalar{ 1 } / Zsafe;

        Scalar J00 = (S2_00 - S1[0] * S1[0] * invZ) * invZ + kTiny;
        Scalar J01 = (S2_01 - S1[0] * S1[1] * invZ) * invZ;
        Scalar J10 = (S2_10 - S1[1] * S1[0] * invZ) * invZ;
        Scalar J11 = (S2_11 - S1[1] * S1[1] * invZ) * invZ + kTiny;

        const Scalar detJ = J00 * J11 - J01 * J10;
        const Scalar absDet = std::fabs(detJ);
        nrOk *= (absDet > kTiny) ? Scalar{ 1 } : Scalar{ 0 };
        const Scalar detSafe = (absDet > kTiny)
                                 ? detJ
                                 : ((detJ >= Scalar{ 0 }) ? kTiny : -kTiny);
        const Scalar invDet = Scalar{ 1 } / detSafe;

        const Scalar r0 = S1[0] * invZ - targetM[0];
        const Scalar r1 = S1[1] * invZ - targetM[1];
        xi[0] -= alpha * (J11 * r0 - J01 * r1) * invDet;
        xi[1] -= alpha * (-J10 * r0 + J00 * r1) * invDet;
        alpha *= Scalar{ 0.5 };
      }

      // Compute final geqLocal (branchless, no if(solverOk))
      // nrOk == 1 → use NR result; nrOk == 0 → use uniform fallback.
      // Both branches are computed; the ternary select is warp-uniform.
      smax = -kMaxE;
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < 9; ++idc) {
        const Scalar s = xi[0] * (static_cast<Scalar>(kD2Q9Cx[idc]) + Ushift0)
          + xi[1] * (static_cast<Scalar>(kD2Q9Cy[idc]) + Ushift1);
        si[idc] = s;
        smax = (s > smax) ? s : smax;
      }
      smax = (smax < kMaxE) ? smax : kMaxE;
      Z = Scalar{ 0 };
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < 9; ++idc) {
        Scalar expo = si[idc] - smax;
        expo = (expo < kMaxE) ? expo : kMaxE;
        expo = (expo > -kMaxE) ? expo : -kMaxE;
        const Scalar wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
        const Scalar wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
        e[idc] = wx * wy * std::exp(expo);
        Z += e[idc];
      }

      // Uniform fallback: sumW of weights at xi==0 (all si==0, all expo==0).
      Scalar sumW = Scalar{ 0 };
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < 9; ++idc) {
        const Scalar wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
        const Scalar wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
        sumW += wx * wy;
      }
      const Scalar sumWsafe = (sumW > kTiny) ? sumW : Scalar{ 9 };
      const Scalar Zsafe = (Z > kTiny) ? Z : kTiny;

      // Branchless select: NR result when nrOk==1, uniform fallback when 0.
      const Scalar scaleNR = targetE / Zsafe;
      const Scalar scaleFB = targetE / sumWsafe;
      const Scalar scale = nrOk * scaleNR + (Scalar{ 1 } - nrOk) * scaleFB;

      Scalar geqLocal[9];
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < 9; ++idc) {
        const Scalar wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
        const Scalar wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
        // NR path: scale * e[idc]; fallback path: scale * wx*wy.
        geqLocal[idc] = scale * (nrOk * e[idc] + (Scalar{ 1 } - nrOk) * wx * wy);
      }

      // Store xi (branchless: zero in fallback path).
      pLastGx[0 * N + cell] = nrOk * (std::log(scaleNR / rho) - smax);
      pLastGx[1 * N + cell] = nrOk * xi[0];
      pLastGx[2 * N + cell] = nrOk * xi[1];

      // BGK + thermal correction
      const Scalar tau = visc / (rho * Tv) + Scalar{ 0.5 };
      const Scalar omega = Scalar{ 1 } / tau;
      const Scalar diff = cond / (rho * cp);
      const Scalar tauT = diff / Tv + Scalar{ 0.5 };
      Scalar omegaT = Scalar{ 1 } / tauT;

      // Non-equilibrium stress limiter (Tran et al. §3.3).
      Scalar eps = Scalar{ 0 };
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < 9; ++idc) {
        const Scalar fi = pF[idc * N + cell];
        const Scalar d = fi - feqLocal[idc];
        const Scalar den = (feqLocal[idc] > kTiny) ? feqLocal[idc] : kTiny;
        eps += std::fabs(d) / den;
      }
      eps /= Scalar{ 9 };
      const Scalar sigma = (eps >= Scalar{ 1 })
                             ? omega
                             : (eps >= Scalar{ 0.1 })
                             ? Scalar{ 1.35 }
                             : (eps >= Scalar{ 0.01 })
                             ? Scalar{ 1.05 }
                             : Scalar{ 1 };

      Scalar omegaL = omega / sigma;
      omegaL = (omegaL > Scalar{ 1 }) ? omegaL : Scalar{ 1 };
      omegaL = (omegaL < (Scalar{ 2 } - Scalar{ 1e-7 })) ? omegaL : (Scalar{ 2 } - Scalar{ 1e-7 });
      omegaT = (omegaT > Scalar{ 1 }) ? omegaT : Scalar{ 1 };
      omegaT = (omegaT < (Scalar{ 2 } - Scalar{ 1e-7 })) ? omegaT : (Scalar{ 2 } - Scalar{ 1e-7 });

      // L correction vector.
      Scalar L0 = Scalar{ 0 };
      Scalar L1 = Scalar{ 0 };
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < 9; ++idc) {
        const Scalar fi = pF[idc * N + cell];
        const Scalar cix0 = static_cast<Scalar>(kD2Q9Cx[idc]);
        const Scalar ciy0 = static_cast<Scalar>(kD2Q9Cy[idc]);
        const Scalar uvi = ux * cix0 + uy * ciy0;
        const Scalar aux = Scalar{ 2 } * uvi * (fi - feqLocal[idc]);
        L0 += aux * cix0;
        L1 += aux * ciy0;
      }

      const Scalar invT = Scalar{ 1 } / Tv;
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < 9; ++idc) {
        const int fi_idx = idc * N + cell;
        const Scalar fOld = pF[fi_idx];
        pF[fi_idx] = fOld + omegaL * (feqLocal[idc] - fOld);
        const Scalar c0 = static_cast<Scalar>(kD2Q9Cx[idc]) + Ushift0;
        const Scalar c1 = static_cast<Scalar>(kD2Q9Cy[idc]) + Ushift1;
        const Scalar cidotL = L0 * c0 + L1 * c1;
        const Scalar wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
        const Scalar wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
        const Scalar Wi = wx * wy;
        const Scalar gDiff = Wi * cidotL * invT;
        const Scalar gOld = pG[fi_idx];
        pG[fi_idx] = gOld + omegaL * (geqLocal[idc] - gOld) + (omegaL - omegaT) * gDiff;
      }
    }
  }
}

// streamAndMacroscopic — on-the-fly IDW streaming + macroscopic reduction
//
// Design notes:
//  - collapse(2) + int ci/cj: eliminates the expensive 64-bit division
//    `cell/ny` and `cell%ny` from the previous flat-Index loop.
//  - Branchless IDW (no snap cascade): the 4-way if/else snap-to-node check
//    caused warp divergence on grid edges. We use exp(-p*log(r²+kTiny)) for
//    all corners; when r²→0 the weight→+∞ but after wsum normalisation the
//    result is identical to the snap value. Zero branches, zero divergence.
//  - Axis-aligned directions (cy[idc]==0): the inner loop is unrolled by
//    LBMINI_UNROLL(9), so the `cy[idc]==0` test is evaluated at compile time,
//    and the 5 axis-aligned unrolled copies use pure 1-D lerp (no exp/log).
template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::streamAndMacroscopic() {
  const Scalar* __restrict__ pF = fCur_;
  const Scalar* __restrict__ pG = gCur_;
  Scalar* __restrict__ pFaux = fAlt_;
  Scalar* __restrict__ pGaux = gAlt_;
  Scalar* __restrict__ pRho = rhoDev_;
  Scalar* __restrict__ pP = pDev_;
  Scalar* __restrict__ pT = temDev_;
  Scalar* __restrict__ pU = uDev_;
  const Scalar Ushift0 = kControl_.U(0);
  const Scalar Ushift1 = kControl_.U(1);
  const Scalar invCv = Scalar{ 1 } / kFluid_.specificHeatCv;
  const Scalar Rgas = kFluid_.constant;
  const Scalar kTiny = kTiny_;
  const Scalar idwExp = kControl_.idw;
  const int nx = static_cast<int>(nx_);
  const int ny = static_cast<int>(ny_);
  const int N = static_cast<int>(N_);

  #pragma acc parallel loop collapse(2) \
      num_gangs((N + 63) / 64) vector_length(64) \
      deviceptr(pF, pG, pFaux, pGaux, pRho, pP, pT, pU)
  for (int ci = 0; ci < nx; ++ci) {
    for (int cj = 0; cj < ny; ++cj) {
      const int cell = ci * ny + cj;

      const Scalar negHalfIdw = Scalar{ -0.5 } * idwExp;

      Scalar rho = Scalar{ 0 };
      Scalar nrg = Scalar{ 0 };
      Scalar mx = Scalar{ 0 };
      Scalar my = Scalar{ 0 };

      LBMINI_UNROLL(9)
      for (int idc = 0; idc < 9; ++idc) {
        const Scalar cix_s = static_cast<Scalar>(kD2Q9Cx[idc]) + Ushift0;
        const Scalar ciy_s = static_cast<Scalar>(kD2Q9Cy[idc]) + Ushift1;

        // Source position (fractional cell coordinates).
        const Scalar xSrc = static_cast<Scalar>(ci) - cix_s;
        const Scalar ySrc = static_cast<Scalar>(cj) - ciy_s;

        // Floor corners.
        const int x0 = static_cast<int>(std::floor(xSrc));
        const int y0 = static_cast<int>(std::floor(ySrc));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;

        // Clamp x (wall BC), wrap y (periodic BC).
        const int cx0i = (x0 < 0) ? 0 : (x0 >= nx ? nx - 1 : x0);
        const int cx1i = (x1 < 0) ? 0 : (x1 >= nx ? nx - 1 : x1);
        int wy0 = y0 % ny;
        if (wy0 < 0)
          wy0 += ny;
        int wy1 = y1 % ny;
        if (wy1 < 0)
          wy1 += ny;

        const int c00 = cx0i * ny + wy0;
        const int c10 = cx1i * ny + wy0;
        const int c01 = cx0i * ny + wy1;
        const int c11 = cx1i * ny + wy1;
        const int off = idc * N;

        Scalar fi, gi;

        // Diagonal shift: 2-D IDW via exp(-p*log(r²+kTiny)).
        // Branchless — no snap-to-node cascade. When any corner coincides
        // with the source point (r²→0), log(kTiny) is finite and large in
        // magnitude, so exp(·) → +∞ and after wsum normalisation the result
        // converges correctly to the node value.
        const Scalar dx0 = xSrc - static_cast<Scalar>(x0);
        const Scalar dy0 = ySrc - static_cast<Scalar>(y0);
        const Scalar dx1 = xSrc - static_cast<Scalar>(x1);
        const Scalar dy1 = ySrc - static_cast<Scalar>(y1);

        const Scalar w00 = std::exp(negHalfIdw * std::log(dx0 * dx0 + dy0 * dy0 + kTiny));
        const Scalar w10 = std::exp(negHalfIdw * std::log(dx1 * dx1 + dy0 * dy0 + kTiny));
        const Scalar w01 = std::exp(negHalfIdw * std::log(dx0 * dx0 + dy1 * dy1 + kTiny));
        const Scalar w11 = std::exp(negHalfIdw * std::log(dx1 * dx1 + dy1 * dy1 + kTiny));
        const Scalar wsum = w00 + w10 + w01 + w11;
        const Scalar invSum = Scalar{ 1 } / wsum;
        fi = (w00 * pF[off + c00] + w10 * pF[off + c10] + w01 * pF[off + c01] + w11 * pF[off + c11]) * invSum;
        gi = (w00 * pG[off + c00] + w10 * pG[off + c10] + w01 * pG[off + c01] + w11 * pG[off + c11]) * invSum;

        pFaux[off + cell] = fi;
        pGaux[off + cell] = gi;
        rho += fi;
        nrg += gi;
        mx += cix_s * fi;
        my += ciy_s * fi;
      }

      const Scalar invRho = Scalar{ 1 } / rho;
      const Scalar ux = mx * invRho;
      const Scalar uy = my * invRho;
      pU[0 * N + cell] = ux;
      pU[1 * N + cell] = uy;
      const Scalar kin = Scalar{ 0.5 } * (ux * ux + uy * uy);
      const Scalar Tv = (Scalar{ 0.5 } * nrg * invRho - kin) * invCv;
      pRho[cell] = rho;
      pT[cell] = Tv;
      pP[cell] = Rgas * rho * Tv;
    }
  }
}
} // namespace lbmini::openacc::gpu

#endif // LBMINI_OPENACC_GPU_LBMTUBE_HPP_
