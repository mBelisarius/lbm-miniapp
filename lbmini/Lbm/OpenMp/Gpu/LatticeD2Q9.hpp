#ifndef LBMINI_OPENMP_GPU_LATTICED2Q9_HPP_
#define LBMINI_OPENMP_GPU_LATTICED2Q9_HPP_

#include "Lbm/ILattice.hpp"

namespace lbmini::openmp::gpu {
/**
 * @brief OpenMP-target (GPU) D2Q9 lattice descriptor.
 *
 * Numerically and structurally identical to `lbmini::plain::LatticeD2Q9` and
 * `lbmini::openmp::cpu::LatticeD2Q9`. Helpers are stateless, `static constexpr`
 * / `static`, so the class is trivially callable from inside an
 * `#pragma omp target teams distribute parallel for` region — the only GPU
 * specificity: all helpers are `static constexpr` / `static inline`, so
 * nvc++/Clang make them implicitly available inside a `target` region
 * without needing a `declare target` wrapper (which triggers an ICE on
 * nvc++ 26.1 when used around a template with `static constexpr` member
 * arrays that depend on a base-class `constexpr` function).
 */
template<typename Scalar>
class LatticeD2Q9 : public ILattice<Scalar, 2, 9> {
public:
  using Index = long;

  using Lattice = ILattice<Scalar, 2, 9>;

  /// Lattice sound speed squared (in reduced units).
  static constexpr Scalar Cs2 = Scalar{ 1.0 } / Scalar{ 3.0 };

  /// Discrete velocity component `dir` of direction `index` (integer -1/0/+1).
  static constexpr Index Velocity(Index index, Index dir) {
    return kVelocity_[index][dir];
  }

  /// Index of the direction opposite to `index` (bounce-back).
  static constexpr Index Opposite(Index index) {
    return kOpposite_[index];
  }

  /**
   * @brief Branchless, temperature-dependent D2Q9 weight.
   *
   * Expresses `Wi(T)` as a per-dimension linear interpolation between the
   * "zero" and "non-zero" branches — compiles to a few FMAs, which is
   * critical inside a device `teams distribute parallel for` loop where we
   * want every lane uniform.
   */
  static constexpr Scalar Weights(Index idc, Scalar tem) {
    const Scalar wZero = Scalar{ 1.0 } - tem;
    const Scalar wNonZero = Scalar{ 0.5 } * tem;
    Scalar w = Scalar{ 1.0 };
    for (Index d = 0; d < Lattice::Dim(); ++d) {
      const Index vid = kVelocity_[idc][d];
      const Scalar isNonZero = static_cast<Scalar>(vid * vid); // 0 or 1
      w *= wZero + isNonZero * (wNonZero - wZero);
    }
    return w;
  }

  /// Shifted discrete velocity `c_i + U` cast to `Scalar`.
  static constexpr Scalar Cshift(Index idc, Index dir, Scalar Ushift) {
    return static_cast<Scalar>(kVelocity_[idc][dir]) + Ushift;
  }

private:
  /// D2Q9 discrete velocity set, stored `[index][dir]`.
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

  /// Opposite-direction table (bounce-back boundary conditions).
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
} // namespace lbmini::openmp::gpu

#endif // LBMINI_OPENMP_GPU_LATTICED2Q9_HPP_
