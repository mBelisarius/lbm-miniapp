#ifndef LBMINI_ALPAKA_CPU_LBMTUBE_HPP_
#define LBMINI_ALPAKA_CPU_LBMTUBE_HPP_

#include "Lbm/Alpaka/LbmTube.hpp"

namespace lbmini::alpaka::cpu {
/**
 * @brief CPU instantiation of the alpaka LBM tube solver.
 *
 * alpaka `api::host` + `exec::cpuOmpBlocks`: alpaka allocates the buffers and
 * launches, once per time step, one block per OpenMP thread of the calling
 * thread's team (so `Performance.cores` / `OMP_NUM_THREADS` apply as for the
 * OpenMP backend); the blocks balance the row segments of the shared SIMD kernel
 * (`Lbm/HostKernel.hpp`, x-fastest padded layout) dynamically. The definition
 * lives in `Cpu/LbmTube.cpp` / `Cpu/LbmTubeImpl.hpp`.
 */
template<typename Scalar, typename LatticeType>
using LbmTube = lbmini::alpaka::LbmTube<Scalar, LatticeType, lbmini::TargetEnum::CPU>;
} // namespace lbmini::alpaka::cpu

#endif // LBMINI_ALPAKA_CPU_LBMTUBE_HPP_
