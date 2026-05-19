#ifndef LBMINI_CUDA_LBMTUBE_HPP_
#define LBMINI_CUDA_LBMTUBE_HPP_

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/Cuda/Gpu/LatticeD2Q9.hpp"

namespace lbmini::cuda::gpu {

using Index = Eigen::Index;

namespace internal {
inline void CheckCuda(const cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template<typename Scalar>
void launchComputeMacroscopic(
  const Scalar* pF,
  const Scalar* pG,
  Scalar* pRho,
  Scalar* pP,
  Scalar* pT,
  Scalar* pU,
  Scalar Ushift0,
  Scalar Ushift1,
  Scalar invCv,
  Scalar Rgas,
  int nx,
  int ny,
  int N
);

template<typename Scalar>
void launchSeedEquilibria(
  Scalar* pF,
  Scalar* pG,
  const Scalar* pRho,
  const Scalar* pT,
  const Scalar* pU,
  Scalar Ushift0,
  Scalar Ushift1,
  Scalar kTiny,
  int nx,
  int ny,
  int N,
  Scalar cv
);

template<typename Scalar>
void launchCollisionAndEquilibria(
  Scalar* pF,
  Scalar* pG,
  Scalar* pLastGx,
  const Scalar* pRho,
  const Scalar* pT,
  const Scalar* pU,
  Scalar Ushift0,
  Scalar Ushift1,
  Scalar visc,
  Scalar cond,
  Scalar cp,
  Scalar cv,
  Scalar kTiny,
  Scalar kMaxE,
  int nx,
  int ny,
  int N
);

template<typename Scalar>
void launchStreamAndMacroscopic(
  const Scalar* pF,
  const Scalar* pG,
  Scalar* pFaux,
  Scalar* pGaux,
  Scalar* pRho,
  Scalar* pP,
  Scalar* pT,
  Scalar* pU,
  Scalar Ushift0,
  Scalar Ushift1,
  Scalar invCv,
  Scalar Rgas,
  Scalar kTiny,
  Scalar idwExp,
  int nx,
  int ny,
  int N
);
} // namespace internal

/**
 * @brief CUDA variant of the compressible LBM tube solver.
 *
 * Numerically identical to `lbmini::plain::LbmTube`. All GPU-specific choices 
 * are handled via custom CUDA kernels and direct device memory management.
 *
 * ### CUDA design decisions
 *
 * **Raw device pointers.** Device memory is allocated via `cudaMalloc` and 
 * managed via raw pointers (`fDev_`, `gDev_`, etc.) to interface cleanly with 
 * CUDA kernel launches without overhead.
 *
 * **Kernel fission & fusion.** The streaming and macroscopic steps are fused 
 * into a single `streamAndMacroscopicKernel` to avoid an extra device memory 
 * round-trip and maximize HBM bandwidth utilization. 
 *
 * **On-the-fly streaming.** Precomputed index lookup tables are eliminated. 
 * Shifted-lattice streaming performs branchless on-the-fly fractional cell 
 * Inverse Distance Weighting (IDW) directly in registers, removing memory 
 * indirection and saving memory bandwidth.
 *
 * **Branchless warp execution.** Divergence in the BGK relaxation and 
 * Newton-Raphson solver is minimized by using fixed iteration counts and 
 * branchless ternary logic (`nrOk`, temperature weights).
 *
 * **Explicit host synchronization.** Macroscopic getters (`P()`, `Rho()`, etc.) 
 * trigger explicit `cudaMemcpy` from device to host and synchronize the 
 * device before returning the Eigen tensors.
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

  // The solver owns raw `cudaMalloc` handles freed in its destructor and
  // caches `fCur_`/`fAlt_`/`gCur_`/`gAlt_` pointers into them; a copy would
  // double-free the device memory and leave the survivor dangling. Solvers are
  // heavy, unique objects held behind `ILbmTubeNd*`, so the operations are
  // deleted rather than implemented, uniformly across every backend.
  // See docs/memory-safety.md §5.2.
  LbmTube(const LbmTube&) = delete;

  LbmTube& operator=(const LbmTube&) = delete;

  LbmTube(LbmTube&&) = delete;

  LbmTube& operator=(LbmTube&&) = delete;

  ~LbmTube() override;

  [[nodiscard]] Tensor<Scalar, kDim_> P() const override {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    internal::CheckCuda(
      cudaMemcpy(out.data(), pDev_, static_cast<std::size_t>(N_) * sizeof(Scalar), cudaMemcpyDeviceToHost),
      "cudaMemcpy pressure to host"
    );
    return out;
  }

  [[nodiscard]] Tensor<Scalar, kDim_> Rho() const override {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    internal::CheckCuda(
      cudaMemcpy(out.data(), rhoDev_, static_cast<std::size_t>(N_) * sizeof(Scalar), cudaMemcpyDeviceToHost),
      "cudaMemcpy density to host"
    );
    return out;
  }

  [[nodiscard]] Tensor<Scalar, kDim_> T() const override {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    internal::CheckCuda(
      cudaMemcpy(out.data(), temDev_, static_cast<std::size_t>(N_) * sizeof(Scalar), cudaMemcpyDeviceToHost),
      "cudaMemcpy temperature to host"
    );
    return out;
  }

  [[nodiscard]] Tensor<Scalar, kDim_ + 1> U() const override {
    internal::CheckCuda(
      cudaMemcpy(uHost_.data(), uDev_, static_cast<std::size_t>(uSize_) * sizeof(Scalar), cudaMemcpyDeviceToHost),
      "cudaMemcpy velocity to host"
    );
    Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
    Scalar* dst = out.data();
    for (Index cell = 0; cell < N_; ++cell)
      for (Index d = 0; d < kDim_; ++d)
        dst[cell * kDim_ + d] = uHost_[d * N_ + cell];
    return out;
  }

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
   * Launches `computeMacroscopicKernel` on the CUDA device. Uses optimized
   * grid/block dimensioning for maximum warp occupancy.
   */
  void computeMacroscopic();

  /**
   * @brief Initializes f_ and g_ using local equilibria.
   *
   * Launches `seedEquilibriaKernel` on the CUDA device.
   */
  void seedEquilibria();

  /**
   * @brief Fused BGK collision and equilibrium computation.
   *
   * Launches `collisionAndEquilibriaKernel`. Minimizes memory traffic by
   * performing BGK collision and local operations using registers.
   */
  void collisionAndEquilibria();

  /**
   * @brief Combined streaming and macroscopic update.
   *
   * Launches `streamAndMacroscopicKernel`. Performs IDW streaming using
   * on-the-fly index calculations without lookup tables, fused with the
   * macroscopic reduction to avoid an extra device memory round-trip.
   */
  void streamAndMacroscopic();

private:
  static constexpr Scalar kTiny_ = Scalar{ 1.0e-12 };
  static constexpr Scalar kMaxExp_ = Scalar{ 700.0 };

  const FluidData<Scalar> kFluid_;
  const MeshData<Scalar, kDim_> kMesh_;
  const ControlData<Scalar> kControl_;
  const PerformanceData kPerformance_;

  Index nx_;
  Index ny_;
  Index N_;
  Index uSize_;
  Index distSize_;

  mutable lbmini::DeviceBuffer<Scalar> rhoHost_;
  mutable lbmini::DeviceBuffer<Scalar> pHost_;
  mutable lbmini::DeviceBuffer<Scalar> temHost_;
  mutable lbmini::DeviceBuffer<Scalar> uHost_;
  lbmini::DeviceBuffer<Scalar> lastGxHost_;

  Scalar* rhoDev_ = nullptr;
  Scalar* pDev_ = nullptr;
  Scalar* temDev_ = nullptr;
  Scalar* uDev_ = nullptr;
  Scalar* fDev_ = nullptr;
  Scalar* gDev_ = nullptr;
  Scalar* fauxDev_ = nullptr;
  Scalar* gauxDev_ = nullptr;
  Scalar* lastGxDev_ = nullptr;

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
  ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "CUDA GPU");
  // Single host thread orchestrates GPU launches; all compute is on device.
  nx_ = mesh.size[0];
  ny_ = mesh.size[1];
  N_ = nx_ * ny_;
  uSize_ = N_ * kDim_;
  distSize_ = N_ * kQ_;

  rhoHost_.assign(N_, Scalar{ 0 });
  pHost_.assign(N_, Scalar{ 0 });
  temHost_.assign(N_, Scalar{ 0 });
  uHost_.assign(uSize_, Scalar{ 0 });
  lastGxHost_.assign(N_ * (kDim_ + 1), Scalar{ 0 });

  const std::size_t sN = static_cast<std::size_t>(N_) * sizeof(Scalar);
  const std::size_t sU = static_cast<std::size_t>(uSize_) * sizeof(Scalar);
  const std::size_t sD = static_cast<std::size_t>(distSize_) * sizeof(Scalar);
  const std::size_t sLg = static_cast<std::size_t>(N_ * (kDim_ + 1)) * sizeof(Scalar);

  try {
    internal::CheckCuda(cudaMalloc(&rhoDev_, sN), "cudaMalloc rho");
    internal::CheckCuda(cudaMalloc(&pDev_, sN), "cudaMalloc pressure");
    internal::CheckCuda(cudaMalloc(&temDev_, sN), "cudaMalloc temperature");
    internal::CheckCuda(cudaMalloc(&uDev_, sU), "cudaMalloc velocity");
    internal::CheckCuda(cudaMalloc(&fDev_, sD), "cudaMalloc f");
    internal::CheckCuda(cudaMalloc(&gDev_, sD), "cudaMalloc g");
    internal::CheckCuda(cudaMalloc(&fauxDev_, sD), "cudaMalloc f scratch");
    internal::CheckCuda(cudaMalloc(&gauxDev_, sD), "cudaMalloc g scratch");
    internal::CheckCuda(cudaMalloc(&lastGxDev_, sLg), "cudaMalloc lastGx");
  } catch (...) {
    if (rhoDev_)
      cudaFree(rhoDev_);
    if (pDev_)
      cudaFree(pDev_);
    if (temDev_)
      cudaFree(temDev_);
    if (uDev_)
      cudaFree(uDev_);
    if (fDev_)
      cudaFree(fDev_);
    if (gDev_)
      cudaFree(gDev_);
    if (fauxDev_)
      cudaFree(fauxDev_);
    if (gauxDev_)
      cudaFree(gauxDev_);
    if (lastGxDev_)
      cudaFree(lastGxDev_);
    throw;
  }

  fCur_ = fDev_;
  fAlt_ = fauxDev_;
  gCur_ = gDev_;
  gAlt_ = gauxDev_;
}

template<typename Scalar, typename LatticeType>
LbmTube<Scalar, LatticeType>::~LbmTube() {
  if (rhoDev_)
    cudaFree(rhoDev_);
  if (pDev_)
    cudaFree(pDev_);
  if (temDev_)
    cudaFree(temDev_);
  if (uDev_)
    cudaFree(uDev_);
  if (fDev_)
    cudaFree(fDev_);
  if (gDev_)
    cudaFree(gDev_);
  if (fauxDev_)
    cudaFree(fauxDev_);
  if (gauxDev_)
    cudaFree(gauxDev_);
  if (lastGxDev_)
    cudaFree(lastGxDev_);
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Init() {
  for (Index i = 0; i < nx_; ++i) {
    for (Index j = 0; j < ny_; ++j) {
      const Index cell = cellIndex(i, j);
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
  internal::CheckCuda(cudaMemcpy(rhoDev_, rhoHost_.data(), sN, cudaMemcpyHostToDevice),
                      "cudaMemcpy density to device");
  internal::CheckCuda(cudaMemcpy(pDev_, pHost_.data(), sN, cudaMemcpyHostToDevice),
                      "cudaMemcpy pressure to device");
  internal::CheckCuda(cudaMemcpy(temDev_, temHost_.data(), sN, cudaMemcpyHostToDevice),
                      "cudaMemcpy temperature to device");
  internal::CheckCuda(cudaMemcpy(uDev_, uHost_.data(), sU, cudaMemcpyHostToDevice),
                      "cudaMemcpy velocity to device");
  internal::CheckCuda(cudaMemcpy(lastGxDev_, lastGxHost_.data(), sLg, cudaMemcpyHostToDevice),
                      "cudaMemcpy lastGx to device");

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
  // Kernel launches are asynchronous: block until the whole chunk has run so
  // that the caller's timer measures the device work, not only the launches.
  internal::CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after Run");
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::computeMacroscopic() {
  const Scalar Ushift0 = kControl_.U(0);
  const Scalar Ushift1 = kControl_.U(1);
  const Scalar invCv = Scalar{ 1 } / kFluid_.specificHeatCv;
  const Scalar Rgas = kFluid_.constant;
  internal::launchComputeMacroscopic<Scalar>(
    fCur_,
    gCur_,
    rhoDev_,
    pDev_,
    temDev_,
    uDev_,
    Ushift0,
    Ushift1,
    invCv,
    Rgas,
    static_cast<int>(nx_),
    static_cast<int>(ny_),
    static_cast<int>(N_)
  );
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::seedEquilibria() {
  const Scalar Ushift0 = kControl_.U(0);
  const Scalar Ushift1 = kControl_.U(1);
  internal::launchSeedEquilibria<Scalar>(
    fCur_,
    gCur_,
    rhoDev_,
    temDev_,
    uDev_,
    Ushift0,
    Ushift1,
    kTiny_,
    static_cast<int>(nx_),
    static_cast<int>(ny_),
    static_cast<int>(N_),
    kFluid_.specificHeatCv
  );
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::collisionAndEquilibria() {
  const Scalar Ushift0 = kControl_.U(0);
  const Scalar Ushift1 = kControl_.U(1);
  internal::launchCollisionAndEquilibria<Scalar>(
    fCur_,
    gCur_,
    lastGxDev_,
    rhoDev_,
    temDev_,
    uDev_,
    Ushift0,
    Ushift1,
    kFluid_.viscosity,
    kFluid_.conductivity,
    kFluid_.specificHeatCp,
    kFluid_.specificHeatCv,
    kTiny_,
    kMaxExp_,
    static_cast<int>(nx_),
    static_cast<int>(ny_),
    static_cast<int>(N_)
  );
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::streamAndMacroscopic() {
  const Scalar Ushift0 = kControl_.U(0);
  const Scalar Ushift1 = kControl_.U(1);
  const Scalar invCv = Scalar{ 1 } / kFluid_.specificHeatCv;
  const Scalar Rgas = kFluid_.constant;
  internal::launchStreamAndMacroscopic<Scalar>(
    fCur_,
    gCur_,
    fAlt_,
    gAlt_,
    rhoDev_,
    pDev_,
    temDev_,
    uDev_,
    Ushift0,
    Ushift1,
    invCv,
    Rgas,
    kTiny_,
    kControl_.idw,
    static_cast<int>(nx_),
    static_cast<int>(ny_),
    static_cast<int>(N_)
  );
}

} // namespace lbmini::cuda::gpu

#endif // LBMINI_CUDA_LBMTUBE_HPP_
