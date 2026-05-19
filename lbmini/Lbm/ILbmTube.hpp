#ifndef LBMINI_ILBMTUBE_HPP_
#define LBMINI_ILBMTUBE_HPP_

#include <string>

#include <unsupported/Eigen/CXX11/Tensor>

namespace lbmini {
/**
 * @class ILbmTubeNd
 * @brief Lattice-independent virtual API of every LBM solver.
 *
 * The whole polymorphic contract lives here and depends on the spatial
 * dimension **only** — deliberately not on the lattice type. `lbmini::Executor`
 * has to hold any backend behind a single pointer, while every backend
 * instantiates `ILbmTube` with its *own* lattice (`plain::LatticeD2Q9`,
 * `cuda::gpu::LatticeD2Q9`, ...), so no `ILbmTube<Scalar, SomeLattice>` is a
 * base class of more than one solver and none of them can serve as that
 * pointer type. Keeping the virtual API in a dimension-only base makes
 * `ILbmTubeNd<Scalar, 2>` a genuine base class of every 2-D solver, so the
 * executor stores them through an ordinary implicit upcast — a
 * `reinterpret_cast` between unrelated interface instantiations would make
 * every virtual call and the final `delete` undefined behaviour.
 */
template<typename Scalar, Eigen::Index Dim_>
class ILbmTubeNd {
public:
  using Index = Eigen::Index;

  template<typename Type, Index NumIndices>
  using Tensor = Eigen::Tensor<Type, NumIndices, Eigen::RowMajor>;

  static constexpr Index Dim() { return Dim_; }

  virtual ~ILbmTubeNd() = default;

  /** @brief Returns a copy of the macroscopic pressure field (P). */
  virtual Tensor<Scalar, Dim_> P() const = 0;

  /** @brief Returns a copy of the macroscopic density field (Rho). */
  virtual Tensor<Scalar, Dim_> Rho() const = 0;

  /** @brief Returns a copy of the macroscopic temperature field (T). */
  virtual Tensor<Scalar, Dim_> T() const = 0;

  /** @brief Returns a copy of the macroscopic velocity field (U). */
  virtual Tensor<Scalar, Dim_ + 1> U() const = 0;

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
  ILbmTubeNd() = default;

  // A polymorphic base that owns nothing must not be copied through: slicing a
  // solver would duplicate the raw device handles its concrete type owns.
  ILbmTubeNd(const ILbmTubeNd&) = default;

  ILbmTubeNd& operator=(const ILbmTubeNd&) = default;
};

/**
 * @class ILbmTube
 * @brief Base class for Lattice-Boltzmann fluid simulations of high-speed compressible flows.
 *
 * Implements the core LBM equations and macroscopic field recovery (density, pressure,
 * temperature, velocity) following the methodology described by Tran et al. (2022).
 *
 * The declaration point of the backends (`class LbmTube : public
 * ILbmTube<Scalar, LatticeType>`), so the lattice stays visible where it is
 * meaningful; the virtual API itself comes from `ILbmTubeNd<Scalar, Dim>`,
 * which is what the executor stores.
 */
template<typename Scalar, typename LatticeType>
class ILbmTube : public ILbmTubeNd<Scalar, LatticeType::Dim()> {
protected:
  ILbmTube() = default;
};
} // namespace lbmini

#endif // LBMINI_ILBMTUBE_HPP_
