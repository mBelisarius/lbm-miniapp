#ifndef LBMINI_ALPAKA_GPU_LBMTUBE_HPP_
#define LBMINI_ALPAKA_GPU_LBMTUBE_HPP_

#include "Lbm/Alpaka/LbmTube.hpp"

namespace lbmini::alpaka::gpu {
/**
 * @brief GPU instantiation of the alpaka LBM tube solver.
 *
 * alpaka `api::cuda` + `exec::gpuCuda`, one thread per cell, asynchronous
 * queue. Only available when the project was configured with
 * `-DENABLE_ALPAKA_CUDA=ON` (`kHasDeviceBackend`); the definition lives in
 * `Gpu/LbmTube.cpp`.
 */
template<typename Scalar, typename LatticeType>
using LbmTube = lbmini::alpaka::LbmTube<Scalar, LatticeType, lbmini::TargetEnum::GPU>;
} // namespace lbmini::alpaka::gpu

#endif // LBMINI_ALPAKA_GPU_LBMTUBE_HPP_
