#ifndef LBMINI_RAJA_GPU_LBMTUBE_HPP_
#define LBMINI_RAJA_GPU_LBMTUBE_HPP_

#include "Lbm/Raja/LbmTube.hpp"

namespace lbmini::raja::gpu {
/**
 * @brief True when RAJA was configured with an accelerator back end.
 *
 * `DeviceBackend` silently degenerates to the host policy and host memory when
 * the library is built without CUDA / HIP. The factory in `lbmini::Executor`
 * queries this flag to warn that a "GPU" run would in fact be executed on the
 * host.
 */
inline constexpr bool kHasDeviceBackend = lbmini::raja::kHasDeviceBackend;

/**
 * @brief GPU instantiation of the portable RAJA LBM tube solver.
 *
 * Binds the shared implementation to `DeviceBackend` — `RAJA::cuda_exec` or
 * `RAJA::hip_exec` over device memory, whichever RAJA was configured with. Not
 * a single line of the solver differs from the CPU instantiation below the
 * alias: the buffers move to device memory and the `RAJA::forall` bodies become
 * device kernels purely through the backend template argument.
 */
template<typename Scalar, typename LatticeType>
using LbmTube = lbmini::raja::LbmTube<Scalar, LatticeType, lbmini::raja::DeviceBackend>;
} // namespace lbmini::raja::gpu

#endif // LBMINI_RAJA_GPU_LBMTUBE_HPP_
