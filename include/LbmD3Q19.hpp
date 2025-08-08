#ifndef LBMINI_LBMD3Q19_HPP_
#define LBMINI_LBMD3Q19_HPP_

#include <Eigen/Dense>

#include "LbmBase.hpp"

namespace lbmini {

using namespace Eigen;

// D3Q19 implementation using Eigen matrices/vectors.
template <typename Scalar_>
class LbmD3Q19 : public LbmClassBase<Scalar_, 3, 19> {
public:
  using Base = LbmClassBase<Scalar_, 3, 19>;
  using Base::Dim, Base::Speeds;

  ~LbmD3Q19() override = default;

  Scalar_ Velocities(Index index, Index dir) const override {
    return kVelocities_(index, dir);
  }

  Scalar_ Weights(Index index) const override {
    return kWeights_(index);
  }

  Index Opposite(Index index) const override { return kOpposite_(index); }

private:
  const Matrix<Scalar_, Speeds(), Dim()> kVelocities_ {
      { 0, 0, 0 },
      { 1, 0, 0 },
      { -1, 0, 0 },
      { 0, 1, 0 },
      { 0, -1, 0 },
      { 0, 0, 1 },
      { 0, 0, -1 },
      { 1, 1, 0 },
      { -1, -1, 0 },
      { 1, -1, 0 },
      { -1, 1, 0 },
      { 1, 0, 1 },
      { -1, 0, -1 },
      { 1, 0, -1 },
      { -1, 0, 1 },
      { 0, 1, 1 },
      { 0, -1, -1 },
      { 0, 1, -1 },
      { 0, -1, 1 }
  };

  const Vector<Scalar_, Speeds()> kWeights_ {
    // Rest weight
    1.0 / 3.0,
    // Face-centered directions
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    // Edge-centered directions
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0
  };

  const Vector<Scalar_, Speeds()> kOpposite_ { 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17 };
};

}

#endif  // LBMINI_LBMD3Q19_HPP_
