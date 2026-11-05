/**
 * @file LbmTube.cpp (Alpaka GPU)
 * @brief CUDA instantiation of the alpaka LBM solver: `api::cuda`,
 *        `exec::gpuCuda`.
 *
 * `alpaka_finalize()` compiles this file as CUDA because its target links
 * `alpaka::cuda`; the kernels in `LbmTubeImpl.hpp` become device code without a
 * single CUDA-specific line in the solver.
 */
#include "Lbm/Alpaka/LbmTubeImpl.hpp"

#if !ALPAKA_LANG_CUDA
#error "Lbm/Alpaka/Gpu/LbmTube.cpp must be compiled as CUDA (link the target against alpaka::cuda)."
#endif

template class lbmini::alpaka::detail::Engine<double, lbmini::alpaka::LatticeD2Q9<double>, lbmini::TargetEnum::GPU>;

namespace lbmini::alpaka::gpu {
int DeviceCount() {
  try {
    return static_cast<int>(
      ::alpaka::onHost::makeDeviceSelector(::alpaka::api::cuda, ::alpaka::deviceKind::nvidiaGpu).getDeviceCount()
    );
  } catch (...) {
    // No driver / no usable device: report "none" and let the caller decide.
    return 0;
  }
}
} // namespace lbmini::alpaka::gpu
