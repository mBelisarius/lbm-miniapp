#ifndef LBMINI_KOKKOS_CPU_LATTICED2Q9_HPP_
#define LBMINI_KOKKOS_CPU_LATTICED2Q9_HPP_

#include "Lbm/Kokkos/LatticeD2Q9.hpp"

namespace lbmini::kokkos::cpu {
/**
 * @brief CPU-target alias of the portable Kokkos D2Q9 lattice descriptor.
 *
 * The descriptor is execution-space agnostic, so the CPU and GPU targets share
 * the very same definition; only `LbmTube` differs, and only in the execution
 * space it is instantiated with.
 */
template<typename Scalar>
using LatticeD2Q9 = lbmini::kokkos::LatticeD2Q9<Scalar>;
} // namespace lbmini::kokkos::cpu

#endif // LBMINI_KOKKOS_CPU_LATTICED2Q9_HPP_
