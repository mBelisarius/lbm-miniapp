#ifndef LBMINI_RAJA_GPU_LATTICED2Q9_HPP_
#define LBMINI_RAJA_GPU_LATTICED2Q9_HPP_

#include "Lbm/Raja/LatticeD2Q9.hpp"

namespace lbmini::raja::gpu {
/**
 * @brief GPU-target alias of the portable RAJA D2Q9 lattice descriptor.
 *
 * Identical to the CPU alias: every accessor is already `RAJA_HOST_DEVICE`, so
 * the descriptor needs no accelerator-specific variant.
 */
template<typename Scalar>
using LatticeD2Q9 = lbmini::raja::LatticeD2Q9<Scalar>;
} // namespace lbmini::raja::gpu

#endif // LBMINI_RAJA_GPU_LATTICED2Q9_HPP_
