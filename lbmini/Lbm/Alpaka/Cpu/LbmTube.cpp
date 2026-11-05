/**
 * @file LbmTube.cpp (Alpaka CPU)
 * @brief Host instantiation of the alpaka LBM solver: `api::host`,
 *        `exec::cpuOmpBlocks` (OpenMP).
 *
 * Compiled as C++ in the alpaka host target (see `alpaka_finalize()` in
 * CMakeLists.txt); it is the only translation unit that generates the CPU
 * kernels. The CPU target runs the vectorised host kernel shared with the other
 * CPU backends (`Cpu/LbmTubeImpl.hpp`), not the per-cell device kernels.
 */
#include "Lbm/Alpaka/LbmTubeImpl.hpp"
#include "Lbm/Alpaka/Cpu/LbmTubeImpl.hpp"

static_assert(ALPAKA_OMP, "The alpaka CPU backend requires OpenMP (alpaka_DEP_OMP=ON).");

template class lbmini::alpaka::detail::Engine<double, lbmini::alpaka::LatticeD2Q9<double>, lbmini::TargetEnum::CPU>;
