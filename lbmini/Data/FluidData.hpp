#ifndef LBMINI_FLUID_DATA_HPP_
#define LBMINI_FLUID_DATA_HPP_

namespace lbmini {

template <typename Scalar_>
struct FluidData {
  Scalar_ densityL;      // Left state density
  Scalar_ pressureL;     // Left state pressure
  Scalar_ temperatureL;  // Left state temperature
  Scalar_ densityR;      // Right state density
  Scalar_ pressureR;     // Right state pressure
  Scalar_ temperatureR;  // Right state temperature
  Scalar_ viscosity;     // Kinematic viscosity (nu)
  Scalar_ specificHeat;  // Specific heat
  Scalar_ gamma;         // Adiabatic heat capacity ratio (γ)
  Scalar_ constant;      // Molar gas constant
};

}  // namespace lbmini

#endif  // LBMINI_FLUID_DATA_HPP_
