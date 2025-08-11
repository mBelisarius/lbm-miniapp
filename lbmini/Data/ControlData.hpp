#ifndef LBMINI_CONTROL_DATA_HPP_
#define LBMINI_CONTROL_DATA_HPP_

namespace lbmini {

template <typename Scalar_>
struct ControlData {
  Scalar_ tmax;  // Simulation time
};

}  // namespace lbmini

#endif  // LBMINI_CONTROL_DATA_HPP_
