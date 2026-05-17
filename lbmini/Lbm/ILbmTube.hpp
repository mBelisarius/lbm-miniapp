#ifndef LBMINI_ILBMTUBE_HPP_
#define LBMINI_ILBMTUBE_HPP_

#include <unsupported/Eigen/CXX11/Tensor>

namespace lbmini {
/**
 * @class ILbmTube
 * @brief Base class for Lattice-Boltzmann fluid simulations of high-speed compressible flows.
 *
 * Implements the core LBM equations and macroscopic field recovery (density, pressure,
 * temperature, velocity) following the methodology described by Tran et al. (2022).
 */
template<typename Scalar, typename LatticeType>
class ILbmTube {
public:
  using Index = Eigen::Index;

  template<typename Type, Index NumIndices>
  using Tensor = Eigen::Tensor<Type, NumIndices, Eigen::RowMajor>;

  virtual ~ILbmTube() = default;

  /** @brief Returns a copy of the macroscopic pressure field (P). */
  virtual Tensor<Scalar, LatticeType::Dim()> P() const = 0;

  /** @brief Returns a copy of the macroscopic density field (Rho). */
  virtual Tensor<Scalar, LatticeType::Dim()> Rho() const = 0;

  /** @brief Returns a copy of the macroscopic temperature field (T). */
  virtual Tensor<Scalar, LatticeType::Dim()> T() const = 0;

  /** @brief Returns a copy of the macroscopic velocity field (U). */
  virtual Tensor<Scalar, LatticeType::Dim() + 1> U() const = 0;

  /** @brief Initializes the simulation distributions and macroscopic fields. */
  virtual void Init() = 0;

  /** @brief Advances the simulation by a single time step. */
  virtual void Step(bool save) = 0;

  /** @brief Advances the simulation by multiple time steps. */
  virtual void Run(Index steps, bool save) = 0;

  /** @brief Returns the rank of the current process (0 for non-MPI backends). */
  virtual int Rank() const { return 0; }

  /** @brief Broadcasts the output path across MPI ranks (no-op for non-MPI backends). */
  virtual void BroadcastOutputPath(std::string& /*path*/) const {}

protected:
  ILbmTube() = default;
};
} // namespace lbmini

#endif // LBMINI_ILBMTUBE_HPP_
