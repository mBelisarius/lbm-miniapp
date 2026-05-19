#ifndef LBMINI_KOKKOS_GPU_LBMTUBE_HPP_
#define LBMINI_KOKKOS_GPU_LBMTUBE_HPP_

#include <Kokkos_Core.hpp>

#include "Lbm/Kokkos/LbmTube.hpp"

namespace lbmini::kokkos::gpu {
/**
 * @brief True when Kokkos was configured with an accelerator backend.
 *
 * `Kokkos::DefaultExecutionSpace` silently falls back to the host space when
 * the library is built without CUDA / HIP / SYCL / OpenMPTarget. The factory in
 * `lbmini::Executor` queries this flag to warn that a "GPU" run would in fact
 * be executed on the host.
 */
inline constexpr bool kHasDeviceBackend =
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL) || \
  defined(KOKKOS_ENABLE_OPENMPTARGET)
  true;
#else
  false;
#endif

/**
 * @brief GPU instantiation of the portable Kokkos LBM tube solver.
 *
 * Binds the shared implementation to `Kokkos::DefaultExecutionSpace` — CUDA,
 * HIP or SYCL, whichever Kokkos was configured with. Not a single line of the
 * solver differs from the CPU instantiation below the alias: the views move to
 * device memory and the `parallel_for` bodies become device kernels purely
 * through the execution-space template argument.
 */
template<typename Scalar, typename LatticeType>
using LbmTube = lbmini::kokkos::LbmTube<Scalar, LatticeType, Kokkos::DefaultExecutionSpace>;
} // namespace lbmini::kokkos::gpu

#endif // LBMINI_KOKKOS_GPU_LBMTUBE_HPP_
