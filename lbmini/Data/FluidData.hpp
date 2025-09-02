#ifndef LBMINI_DATA_FLUID_DATA_HPP_
#define LBMINI_DATA_FLUID_DATA_HPP_

namespace lbmini {

template <typename Scalar>
struct FluidData {
  Scalar densityL;        // Left state density
  Scalar pressureL;       // Left state pressure
  Scalar densityR;        // Right state density
  Scalar pressureR;       // Right state pressure
  Scalar viscosity;       // Kinematic viscosity (nu)
  Scalar prandtl;         // Prandtl number
  Scalar gamma;           // Adiabatic heat capacity ratio (γ)
  Scalar constant;        // Molar gas constant
  Scalar specificHeatCp;  // Specific heat at constant pressure
  Scalar specificHeatCv;  // Specific heat at constant volume
  Scalar conductivity;    // Thermal conductivity
};

}  // namespace lbmini

#endif  // LBMINI_DATA_FLUID_DATA_HPP_
