#ifndef LBMINI_ILATTICE_HPP_
#define LBMINI_ILATTICE_HPP_

#include "Data/ControlData.hpp"
#include "Data/FluidData.hpp"

namespace lbmini {
template<typename Scalar, Eigen::Index Dim_, Eigen::Index Speeds_>
class ILattice {
public:
  using Index = Eigen::Index;

  // This class only defines an interface and should not be created as an object
  ILattice() = delete;

  ~ILattice() = delete;

  static constexpr Index Dim() { return Dim_; }

  static constexpr Index Speeds() { return Speeds_; }

  static constexpr Index Velocity(Index index, Index dir);

  static constexpr Index Opposite(Index index);

  static Scalar Weights(Index idc, Scalar tem);

  static void Init(
    const Scalar u0[Dim()],
    Scalar* rho0,
    Scalar* p0,
    Scalar* rho,
    Scalar* p,
    Scalar* tem,
    Scalar* u,
    Scalar* f,
    Scalar* feq,
    Scalar* g,
    Scalar* geq,
    Scalar* lastGx,
    const FluidData<Scalar>& pFluid,
    const ControlData<Scalar>& pControl
  );

  static void ComputeMacroscopic(
    Scalar* rho,
    Scalar* p,
    Scalar* tem,
    Scalar* u,
    Scalar* f,
    Scalar* feq,
    Scalar* g,
    Scalar* geq,
    Scalar* lastGx,
    const FluidData<Scalar>& pFluid,
    const ControlData<Scalar>& pControl
  );

  static void Collision(
    Scalar* rho,
    Scalar* p,
    Scalar* tem,
    Scalar* u,
    Scalar* f,
    Scalar* feq,
    Scalar* g,
    Scalar* geq,
    Scalar* lastGx,
    const FluidData<Scalar>& pFluid,
    const ControlData<Scalar>& pControl
  );
};
} // namespace lbmini

#endif // LBMINI_ILATTICE_HPP_
