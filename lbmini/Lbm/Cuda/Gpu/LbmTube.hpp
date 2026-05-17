#ifndef LBMINI_CUDA_LBMTUBE_HPP_
#define LBMINI_CUDA_LBMTUBE_HPP_

#include <unsupported/Eigen/CXX11/Tensor>
#include "Data.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/Cuda/Gpu/LatticeD2Q9.hpp"

namespace lbmini::cuda::gpu {
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

  ~LbmTube() override;

  [[nodiscard]] Tensor<Scalar, kDim_> P() const override {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    getP(out.data());
    return out;
  }

  [[nodiscard]] Tensor<Scalar, kDim_> Rho() const override {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    getRho(out.data());
    return out;
  }

  [[nodiscard]] Tensor<Scalar, kDim_> T() const override {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    getT(out.data());
    return out;
  }

  [[nodiscard]] Tensor<Scalar, kDim_ + 1> U() const override {
    Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
    getU(out.data());
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
  /**
   * @brief Helper to copy pressure field from device to host memory.
   * Synchronizes CUDA device before returning.
   */
  void getP(Scalar* dst) const;

  /**
   * @brief Helper to copy density field from device to host memory.
   * Synchronizes CUDA device before returning.
   */
  void getRho(Scalar* dst) const;

  /**
   * @brief Helper to copy temperature field from device to host memory.
   * Synchronizes CUDA device before returning.
   */
  void getT(Scalar* dst) const;

  /**
   * @brief Helper to copy velocity field from device to host memory.
   * Synchronizes CUDA device before returning.
   */
  void getU(Scalar* dst) const;

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
} // namespace lbmini::cuda::gpu

#endif // LBMINI_CUDA_LBMTUBE_HPP_
