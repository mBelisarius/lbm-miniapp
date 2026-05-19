#ifndef LBMINI_KOKKOS_LATTICED2Q9_HPP_
#define LBMINI_KOKKOS_LATTICED2Q9_HPP_

#include <Kokkos_Core.hpp>

#include "Lbm/ILattice.hpp"

namespace lbmini::kokkos {
/**
 * @brief Portable D2Q9 lattice descriptor for the Kokkos LBM backend.
 *
 * Numerically identical to `lbmini::plain::LatticeD2Q9`; every accessor is
 * decorated with `KOKKOS_INLINE_FUNCTION` (`__host__ __device__ inline` on
 * CUDA/HIP builds, a plain `inline` on Serial/OpenMP builds) so the very same
 * descriptor can be called from host code and from inside every
 * `Kokkos::parallel_for` body, regardless of the execution space Kokkos was
 * configured with.
 *
 * ### Kokkos design decisions
 *
 * **Function-local velocity tables.** The `plain` backend keeps `kVelocity_`
 * and `kOpposite_` as `static constexpr` class members. Taking a runtime
 * subscript of such a member from device code odr-uses a host object, which
 * `nvcc` rejects ("identifier undefined in device code") unless the table is
 * duplicated into `__constant__` memory — exactly what the CUDA backend had to
 * do. Declaring the tables `constexpr` *inside* each accessor keeps the class
 * single-source: the compiler either constant-folds the lookup or materialises
 * the table in the device's constant/local memory automatically.
 *
 * **Stateless and trivially copyable.** No data members, no virtual dispatch
 * at call sites: the descriptor can be captured by value into a
 * `KOKKOS_LAMBDA` without dragging any host pointer into the device closure.
 */
template<typename Scalar>
class LatticeD2Q9 : public ILattice<Scalar, 2, 9> {
public:
  using Index = long;

  using Lattice = ILattice<Scalar, 2, 9>;

  /// Lattice sound speed squared (in reduced units).
  static constexpr Scalar Cs2 = Scalar{ 1.0 } / Scalar{ 3.0 };

  /// Discrete velocity component `dir` of direction `index` (integer -1/0/+1).
  KOKKOS_INLINE_FUNCTION static constexpr Index Velocity(const Index index, const Index dir) {
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
  KOKKOS_INLINE_FUNCTION static constexpr Index Opposite(const Index index) {
    constexpr Index kOpposite[Lattice::Speeds()] = { 0, 2, 1, 4, 3, 6, 5, 8, 7 };
    return kOpposite[index];
  }

  /**
   * @brief Branchless, temperature-dependent D2Q9 weight.
   *
   * For D2Q9 the velocity entries are in `{-1, 0, +1}`, so `vid * vid` is `1`
   * iff the component is non-zero. The multiplicative fused form below
   * expresses `Wi(T)` as a per-dimension linear interpolation between the
   * "zero" and "non-zero" branches, avoiding warp divergence on GPU and
   * compiling down to a few FMA instructions.
   */
  KOKKOS_INLINE_FUNCTION static constexpr Scalar Weights(const Index idc, const Scalar tem) {
    const Scalar wZero = Scalar{ 1.0 } - tem;
    const Scalar wNonZero = Scalar{ 0.5 } * tem;
    Scalar w = Scalar{ 1.0 };
    for (Index d = 0; d < Lattice::Dim(); ++d) {
      const Index vid = Velocity(idc, d);
      const Scalar isNonZero = static_cast<Scalar>(vid * vid); // 0 or 1
      // select(isNonZero, wNonZero, wZero) via fma: linear interpolation
      w *= wZero + isNonZero * (wNonZero - wZero);
    }
    return w;
  }

  /**
   * @brief Shifted discrete velocity `c_i + U` cast to `Scalar`.
   *
   * Canonical combination consumed by the shifted-lattice collision and
   * streaming kernels; kept here so those kernels never touch the integer
   * velocity table directly.
   */
  KOKKOS_INLINE_FUNCTION static constexpr Scalar Cshift(const Index idc, const Index dir, const Scalar Ushift) {
    return static_cast<Scalar>(Velocity(idc, dir)) + Ushift;
  }
};
} // namespace lbmini::kokkos

#endif // LBMINI_KOKKOS_LATTICED2Q9_HPP_
