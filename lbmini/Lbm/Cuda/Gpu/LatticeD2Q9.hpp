#ifndef LBMINI_CUDA_LATTICED2Q9_HPP_
#define LBMINI_CUDA_LATTICED2Q9_HPP_

#include "Lbm/ILattice.hpp"

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

namespace lbmini::cuda::gpu {
template<typename Scalar>
class LatticeD2Q9 : public ILattice<Scalar, 2, 9> {
public:
  using Index = long;

  using Lattice = ILattice<Scalar, 2, 9>;

  static constexpr Scalar Cs2 = Scalar{ 1.0 } / Scalar{ 3.0 };

  __host__ __device__ static constexpr Index Velocity(Index index, Index dir) {
    return kVelocity_[index][dir];
  }

  __host__ __device__ static constexpr Index Opposite(Index index) {
    return kOpposite_[index];
  }

  __host__ __device__ static constexpr Scalar Weights(Index idc, Scalar tem) {
    const Scalar wZero = Scalar{ 1.0 } - tem;
    const Scalar wNonZero = Scalar{ 0.5 } * tem;
    Scalar w = Scalar{ 1.0 };
    for (Index d = 0; d < Lattice::Dim(); ++d) {
      const Index vid = kVelocity_[idc][d];
      const Scalar isNonZero = static_cast<Scalar>(vid * vid);
      w *= wZero + isNonZero * (wNonZero - wZero);
    }
    return w;
  }

  __host__ __device__ static constexpr Scalar Cshift(Index idc, Index dir, Scalar Ushift) {
    return static_cast<Scalar>(kVelocity_[idc][dir]) + Ushift;
  }

private:
  static constexpr Index kVelocity_[Lattice::Speeds()][Lattice::Dim()] = {
    { 0, 0 },
    { 1, 0 },
    { -1, 0 },
    { 0, 1 },
    { 0, -1 },
    { 1, 1 },
    { -1, -1 },
    { 1, -1 },
    { -1, 1 },
  };

  static constexpr Index kOpposite_[Lattice::Speeds()] = {
    0,
    2,
    1,
    4,
    3,
    6,
    5,
    8,
    7
  };
};
} // namespace lbmini::cuda::gpu

#endif // LBMINI_CUDA_LATTICED2Q9_HPP_
