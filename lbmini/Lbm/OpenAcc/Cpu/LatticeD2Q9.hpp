#ifndef LBMINI_OPENACC_CPU_LATTICED2Q9_HPP_
#define LBMINI_OPENACC_CPU_LATTICED2Q9_HPP_

#include "Lbm/ILattice.hpp"

namespace lbmini::openacc::cpu {
/**
 * @brief OpenMP-CPU D2Q9 lattice descriptor.
 *
 * Identical in semantics to `lbmini::plain::LatticeD2Q9` — a stateless
 * descriptor providing the discrete-velocity set, opposite-direction table,
 * branchless temperature-dependent weights, and shifted-velocity helper.
 *
 * All helpers are `static constexpr` / `static` and depend only on their
 * arguments, so the class is trivially callable from any OpenMP parallel /
 * SIMD region without synchronization or per-call heap traffic.
 */
template<typename Scalar>
class LatticeD2Q9 : public ILattice<Scalar, 2, 9> {
public:
  using Index = long;

  using Lattice = ILattice<Scalar, 2, 9>;

  /// Lattice sound speed squared (in reduced units).
  static constexpr Scalar Cs2 = Scalar{ 1.0 } / Scalar{ 3.0 };

  /// Discrete velocity component `dir` of direction `index` (integer -1/0/+1).
  #pragma acc routine seq
  static constexpr Index Velocity(Index index, Index dir) {
    return kVelocity_[index][dir];
  }

  /// Index of the direction opposite to `index` (used for bounce-back).
  #pragma acc routine seq
  static constexpr Index Opposite(Index index) {
    return kOpposite_[index];
  }

  /**
   * @brief Branchless, temperature-dependent D2Q9 weight.
   *
   * The multiplicative fused form expresses `Wi(T)` as a per-dimension linear
   * interpolation between the "zero" and "non-zero" branches, compiling down
   * to a few FMAs — ideal for OpenMP `simd` regions that inline this call.
   */
  #pragma acc routine seq
  static Scalar Weights(Index idc, Scalar tem) {
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

  /**
   * @brief Shifted discrete velocity `c_i + U` cast to `Scalar`.
   *
   * Canonical combination consumed by the shifted-lattice collision and
   * streaming kernels; keeps the kernels free of integer-table lookups.
   */
  #pragma acc routine seq
  static Scalar Cshift(Index idc, Index dir, Scalar Ushift) {
    return static_cast<Scalar>(kVelocity_[idc][dir]) + Ushift;
  }

private:
  /**
   * @brief D2Q9 discrete velocity set, stored `[index][dir]`.
   *
   * `static constexpr` so the table lives in read-only memory and can be
   * inlined at every call site.
   */
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

  /// Opposite-direction table (used for bounce-back boundary conditions).
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
} // namespace lbmini::openacc::cpu

#endif // LBMINI_OPENACC_CPU_LATTICED2Q9_HPP_
