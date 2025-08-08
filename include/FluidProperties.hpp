#ifndef LBMINI_FLUIDPROPERTIES_HPP_
#define LBMINI_FLUIDPROPERTIES_HPP_

namespace lbmini
{
    template <typename Scalar_>
    struct FluidProperties
    {
        Scalar_ densityL; // Left state density
        Scalar_ pressureL; // Left state pressure
        Scalar_ densityR; // Right state density
        Scalar_ pressureR; // Right state pressure
        Scalar_ viscosity; // Kinematic viscosity (nu)
        Scalar_ specificHeat; // Specific heat
        Scalar_ gamma; // Adiabatic heat capacity ratio (γ)
    };
}

#endif // LBMINI_FLUIDPROPERTIES_HPP_
