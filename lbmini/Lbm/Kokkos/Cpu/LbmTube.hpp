#ifndef LBMINI_KOKKOS_CPU_LBMTUBE_HPP_
#define LBMINI_KOKKOS_CPU_LBMTUBE_HPP_

#include <Kokkos_Core.hpp>

#include "Lbm/Kokkos/LbmTube.hpp"

namespace lbmini::kokkos::cpu {
/**
 * @brief CPU instantiation of the portable Kokkos LBM tube solver.
 *
 * Binds the shared implementation to `Kokkos::DefaultHostExecutionSpace`, i.e.
 * OpenMP (or Threads / Serial, depending on how Kokkos was configured). Note
 * that this is the *host* space even when Kokkos was built with a device
 * backend, so the CPU and GPU measurements can be taken from a single binary.
 */
template<typename Scalar, typename LatticeType>
using LbmTube = lbmini::kokkos::LbmTube<Scalar, LatticeType, Kokkos::DefaultHostExecutionSpace>;
} // namespace lbmini::kokkos::cpu

#endif // LBMINI_KOKKOS_CPU_LBMTUBE_HPP_
