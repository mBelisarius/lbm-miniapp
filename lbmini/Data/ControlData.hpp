#ifndef LBMINI_CONTROL_DATA_HPP_
#define LBMINI_CONTROL_DATA_HPP_

#include <Eigen/Dense>

namespace lbmini {

template <typename Scalar_>
struct ControlData {
  using Index = Eigen::Index;

  Scalar_ tmax;     // Simulation time
  Scalar_ Ux;       // Shifted lattice velocity in x-direction
  Scalar_ Uy;       // Shifted lattice velocity in y-direction
  Scalar_ Uz;       // Shifted lattice velocity in z-direction
  Scalar_ idw;      // Inverse density weighting
  Index printStep;  // Printing frequency

  Scalar_ U(const Index d) const {
    if (d == 0)
      return Ux;
    if (d == 1)
      return Uy;
    if (d == 2)
      return Uz;
    return 0;
  }
};

}  // namespace lbmini

#endif  // LBMINI_CONTROL_DATA_HPP_
