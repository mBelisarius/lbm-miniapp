#ifndef LBMINI_KOKKOS_GPU_LATTICED2Q9_HPP_
#define LBMINI_KOKKOS_GPU_LATTICED2Q9_HPP_

#include "Lbm/Kokkos/LatticeD2Q9.hpp"

namespace lbmini::kokkos::gpu {
/**
 * @brief GPU-target alias of the portable Kokkos D2Q9 lattice descriptor.
 *
 * `KOKKOS_INLINE_FUNCTION` already makes every accessor callable from device
 * code, so no GPU-specific descriptor is needed — the single-source property
 * Kokkos is benchmarked for.
 */
template<typename Scalar>
using LatticeD2Q9 = lbmini::kokkos::LatticeD2Q9<Scalar>;
} // namespace lbmini::kokkos::gpu

#endif // LBMINI_KOKKOS_GPU_LATTICED2Q9_HPP_
