#ifndef LBMINI_FLUID_DATA_HPP_
#define LBMINI_FLUID_DATA_HPP_

namespace lbmini {

template <typename Scalar_>
struct FluidData {
  Scalar_ densityL;        // Left state density
  Scalar_ pressureL;       // Left state pressure
  Scalar_ densityR;        // Right state density
  Scalar_ pressureR;       // Right state pressure
  Scalar_ viscosity;       // Kinematic viscosity (nu)
  Scalar_ prandtl;         // Prandtl number
  Scalar_ gamma;           // Adiabatic heat capacity ratio (γ)
  Scalar_ constant;        // Molar gas constant
  Scalar_ specificHeatCp;  // Specific heat at constant pressure
  Scalar_ specificHeatCv;  // Specific heat at constant volume
  Scalar_ conductivity;    // Thermal conductivity
};

}  // namespace lbmini

#endif  // LBMINI_FLUID_DATA_HPP_
