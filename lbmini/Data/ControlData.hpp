#ifndef LBMINI_CONTROL_DATA_HPP_
#define LBMINI_CONTROL_DATA_HPP_

#include <Eigen/Dense>

namespace lbmini {
template<typename Scalar>
struct ControlData {
  using Index = Eigen::Index;

  Scalar tmax;     // Simulation time
  Scalar Ux;       // Shifted lattice velocity in x-direction
  Scalar Uy;       // Shifted lattice velocity in y-direction
  Scalar Uz;       // Shifted lattice velocity in z-direction
  Scalar idw;      // Inverse density weighting
  Index printStep; // Printing frequency

  Scalar U(const Index d) const {
    if (d == 0)
      return Ux;
    if (d == 1)
      return Uy;
    if (d == 2)
      return Uz;
    return 0;
  }
};
} // namespace lbmini

#endif  // LBMINI_CONTROL_DATA_HPP_
