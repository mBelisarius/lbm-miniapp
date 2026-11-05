#ifndef LBMINI_ALPAKA_LATTICED2Q9_HPP_
#define LBMINI_ALPAKA_LATTICED2Q9_HPP_

#include "Lbm/ILattice.hpp"

/**
 * @def LBMINI_ALPAKA_HOST_ACC
 * @brief Host + accelerator decoration usable *without* including alpaka.
 *
 * The lattice is part of the public, alpaka-free interface of the backend
 * (`Executor.cpp` and the tests name it), so it cannot use `ALPAKA_FN_HOST_ACC`.
 * Both nvcc and clang-CUDA define `__CUDACC__` (hipcc `__HIPCC__`) in every pass
 * of a device translation unit, which is exactly when the decoration matters.
 */
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LBMINI_ALPAKA_HOST_ACC __host__ __device__
#else
#define LBMINI_ALPAKA_HOST_ACC
#endif

namespace lbmini::alpaka {
/**
 * @brief Portable D2Q9 lattice descriptor for the Alpaka LBM backend.
 *
 * Numerically identical to `lbmini::plain::LatticeD2Q9`. Every accessor is
 * `constexpr` and host + accelerator callable, and the velocity / opposite
 * tables are function-local, so a lookup with a compile-time direction (all
 * kernel loops over `idc` are fully unrolled) folds to an immediate and never
 * touches memory on any device.
 *
 * Stateless and trivially copyable: it never travels as a kernel argument,
 * the kernels only use it as a type.
 */
template<typename Scalar>
class LatticeD2Q9 : public ILattice<Scalar, 2, 9> {
public:
  using Index = int;

  using Lattice = ILattice<Scalar, 2, 9>;

  /// Lattice sound speed squared (in reduced units).
  static constexpr Scalar Cs2 = Scalar{ 1.0 } / Scalar{ 3.0 };

  /// Discrete velocity component `dir` of direction `index` (integer -1/0/+1).
  LBMINI_ALPAKA_HOST_ACC static constexpr Index Velocity(const Index index, const Index dir) {
    constexpr Index kVelocity[Lattice::Speeds()][Lattice::Dim()] = {
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
    return kVelocity[index][dir];
  }

  /// Index of the direction opposite to `index` (used for bounce-back).
  LBMINI_ALPAKA_HOST_ACC static constexpr Index Opposite(const Index index) {
    constexpr Index kOpposite[Lattice::Speeds()] = { 0, 2, 1, 4, 3, 6, 5, 8, 7 };
    return kOpposite[index];
  }

  /**
   * @brief Branchless, temperature-dependent D2Q9 weight.
   *
   * Same multiplicative fused form as the other portable backends: a
   * per-dimension linear interpolation between the "zero" and "non-zero"
   * velocity branches, i.e. a handful of FMAs and no divergence.
   */
  LBMINI_ALPAKA_HOST_ACC static constexpr Scalar Weights(const Index idc, const Scalar tem) {
    const Scalar wZero = Scalar{ 1.0 } - tem;
    const Scalar wNonZero = Scalar{ 0.5 } * tem;
    Scalar w = Scalar{ 1.0 };
    for (Index d = 0; d < Lattice::Dim(); ++d) {
      const Index vid = Velocity(idc, d);
      const Scalar isNonZero = static_cast<Scalar>(vid * vid); // 0 or 1
      w *= wZero + isNonZero * (wNonZero - wZero);
    }
    return w;
  }

  /// Shifted discrete velocity `c_i + U` cast to `Scalar`.
  LBMINI_ALPAKA_HOST_ACC static constexpr Scalar Cshift(const Index idc, const Index dir, const Scalar Ushift) {
    return static_cast<Scalar>(Velocity(idc, dir)) + Ushift;
  }
};
} // namespace lbmini::alpaka

#endif // LBMINI_ALPAKA_LATTICED2Q9_HPP_
