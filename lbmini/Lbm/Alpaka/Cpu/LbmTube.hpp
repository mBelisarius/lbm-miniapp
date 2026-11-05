#ifndef LBMINI_ALPAKA_CPU_LBMTUBE_HPP_
#define LBMINI_ALPAKA_CPU_LBMTUBE_HPP_

#include "Lbm/Alpaka/LbmTube.hpp"

namespace lbmini::alpaka::cpu {
/**
 * @brief CPU instantiation of the alpaka LBM tube solver.
 *
 * alpaka `api::host` + `exec::cpuOmpBlocks`: every kernel is an OpenMP
 * parallel loop over contiguous blocks of cells, run on the calling thread's
 * team (so `Performance.cores` / `OMP_NUM_THREADS` apply as for the OpenMP
 * backend). The definition lives in `Cpu/LbmTube.cpp`.
 */
template<typename Scalar, typename LatticeType>
using LbmTube = lbmini::alpaka::LbmTube<Scalar, LatticeType, lbmini::TargetEnum::CPU>;
} // namespace lbmini::alpaka::cpu

#endif // LBMINI_ALPAKA_CPU_LBMTUBE_HPP_
