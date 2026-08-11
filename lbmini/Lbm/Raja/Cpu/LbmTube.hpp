#ifndef LBMINI_RAJA_CPU_LBMTUBE_HPP_
#define LBMINI_RAJA_CPU_LBMTUBE_HPP_

#include "Lbm/Raja/LbmTube.hpp"

namespace lbmini::raja::cpu {
/**
 * @brief CPU instantiation of the portable RAJA LBM tube solver.
 *
 * Binds the shared implementation to `HostBackend`, i.e. the
 * `RAJA::omp_parallel_for_exec` policy (or `RAJA::seq_exec` when RAJA was built
 * without OpenMP) over host memory. Thread count follows the ambient OpenMP
 * settings, which `lbmini::Executor` derives from `PerformanceData::cores`.
 */
template<typename Scalar, typename LatticeType>
using LbmTube = lbmini::raja::LbmTube<Scalar, LatticeType, lbmini::raja::HostBackend>;
} // namespace lbmini::raja::cpu

#endif // LBMINI_RAJA_CPU_LBMTUBE_HPP_
