#ifndef LBMINI_LBMBASE_HPP_
#define LBMINI_LBMBASE_HPP_

#include <Eigen/Dense>

namespace lbmini {

using namespace Eigen;

// Base LBM class: abstract interface for a DnQm lattice.
template <typename Scalar_, Index Dim_, Index Speeds_>
class LbmClassBase {
public:
  virtual ~LbmClassBase() = default;

  static constexpr Index Dim() { return Dim_; }

  static constexpr Index Speeds() { return Speeds_; }

  virtual Scalar_ Velocities(Index index, Index dir) const = 0;

  virtual Scalar_ Weights(Index index) const = 0;

  virtual Index Opposite(Index index) const = 0;
};
}

#endif  // LBMINI_LBMBASE_HPP_
