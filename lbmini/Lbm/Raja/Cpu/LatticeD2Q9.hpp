#ifndef LBMINI_RAJA_CPU_LATTICED2Q9_HPP_
#define LBMINI_RAJA_CPU_LATTICED2Q9_HPP_

#include "Lbm/Raja/LatticeD2Q9.hpp"

namespace lbmini::raja::cpu {
/**
 * @brief CPU-target alias of the portable RAJA D2Q9 lattice descriptor.
 *
 * The descriptor is execution-policy agnostic, so the CPU and GPU targets share
 * the very same definition; only `LbmTube` differs, and only in the backend tag
 * it is instantiated with.
 */
template<typename Scalar>
using LatticeD2Q9 = lbmini::raja::LatticeD2Q9<Scalar>;
} // namespace lbmini::raja::cpu

#endif // LBMINI_RAJA_CPU_LATTICED2Q9_HPP_
