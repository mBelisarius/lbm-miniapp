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
/**
 * @brief D2Q9 lattice of the CUDA backend.
 *
 * Same velocity ordering and weight formula as `lbmini::plain::LatticeD2Q9`.
 * The velocity set is expressed arithmetically instead of through a static
 * table, so every member is usable from device code under both nvcc and
 * clang-CUDA (no host-only constexpr array is odr-used), and a fully unrolled
 * direction loop folds each lookup into an immediate.
 */
template<typename Scalar>
class LatticeD2Q9 : public ILattice<Scalar, 2, 9> {
public:
  using Index = long;

  using Lattice = ILattice<Scalar, 2, 9>;

  static constexpr Scalar Cs2 = Scalar{ 1.0 } / Scalar{ 3.0 };

  /// Ordering: rest, +x, -x, +y, -y, (+x,+y), (-x,-y), (+x,-y), (-x,+y).
  __host__ __device__ static constexpr Index Velocity(Index index, Index dir) {
    if (dir == 0)
      return (index == 1 || index == 5 || index == 7) ? 1 : ((index == 2 || index == 6 || index == 8) ? -1 : 0);
    return (index == 3 || index == 5 || index == 8) ? 1 : ((index == 4 || index == 6 || index == 7) ? -1 : 0);
  }

  __host__ __device__ static constexpr Index Opposite(Index index) {
    return (index == 0) ? 0 : ((index % 2 != 0) ? index + 1 : index - 1);
  }

  __host__ __device__ static constexpr Scalar Weights(Index idc, Scalar tem) {
    const Scalar wZero = Scalar{ 1.0 } - tem;
    const Scalar wNonZero = Scalar{ 0.5 } * tem;
    Scalar w = Scalar{ 1.0 };
    for (Index d = 0; d < Lattice::Dim(); ++d) {
      const Index vid = Velocity(idc, d);
      const Scalar isNonZero = static_cast<Scalar>(vid * vid);
      w *= wZero + isNonZero * (wNonZero - wZero);
    }
    return w;
  }

  __host__ __device__ static constexpr Scalar Cshift(Index idc, Index dir, Scalar Ushift) {
    return static_cast<Scalar>(Velocity(idc, dir)) + Ushift;
  }
};
} // namespace lbmini::cuda::gpu

#endif // LBMINI_CUDA_LATTICED2Q9_HPP_
