#ifndef LBMINI_ALPAKA_GPU_LATTICED2Q9_HPP_
#define LBMINI_ALPAKA_GPU_LATTICED2Q9_HPP_

#include "Lbm/Alpaka/LatticeD2Q9.hpp"

namespace lbmini::alpaka::gpu {
/// GPU-target alias of the portable alpaka D2Q9 lattice descriptor.
template<typename Scalar>
using LatticeD2Q9 = lbmini::alpaka::LatticeD2Q9<Scalar>;
} // namespace lbmini::alpaka::gpu

#endif // LBMINI_ALPAKA_GPU_LATTICED2Q9_HPP_
