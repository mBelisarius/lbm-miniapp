#ifndef LBMINI_RAJA_BACKENDS_HPP_
#define LBMINI_RAJA_BACKENDS_HPP_

#include <cstddef>
#include <cstring>
#include <stdexcept>

#include <RAJA/RAJA.hpp>

#if defined(RAJA_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(RAJA_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace lbmini::raja {
/**
 * @file
 * @brief Execution policies and memory traits that parameterise
 *        `lbmini::raja::LbmTube`.
 *
 * RAJA is deliberately *only* a loop-abstraction layer: unlike Kokkos, it ships
 * no data container and no allocator, so where a `Kokkos::View` decides both
 * "where does this array live" and "how is it indexed", a RAJA application must
 * answer the first question itself (raw pointers, Umpire, CHAI or, as here, a
 * two-line traits class) and keeps plain C indexing for the second.
 *
 * The two questions are therefore split into two orthogonal traits, bundled by
 * a *backend tag*:
 *  - `ExecPolicy` — the RAJA policy handed to `RAJA::forall`;
 *  - `Memory`     — allocation, host<->device transfer and synchronisation.
 *
 * `LbmTube` is templated on the tag, so the same solver source compiles to the
 * OpenMP CPU version and to the CUDA/HIP GPU version, exactly as the Kokkos
 * backend does through its execution-space template argument.
 */

/// Threads per block used by the device policies. 128 keeps the register-heavy
/// collision kernel (Newton-Raphson workspace in registers) at a good occupancy
/// while still giving the scheduler several blocks per SM.
inline constexpr int kDeviceBlockSize = 128;

#if defined(RAJA_ENABLE_OPENMP)
/// Host loop policy: one OpenMP team over the cells, matching the `openmp::cpu`
/// backend's `#pragma omp parallel for`.
using HostExecPolicy = RAJA::omp_parallel_for_exec;
#else
using HostExecPolicy = RAJA::seq_exec;
#endif

#if defined(RAJA_ENABLE_CUDA)
using DeviceExecPolicy = RAJA::cuda_exec<kDeviceBlockSize>;
#elif defined(RAJA_ENABLE_HIP)
using DeviceExecPolicy = RAJA::hip_exec<kDeviceBlockSize>;
#else
/// No accelerator policy was compiled in: the "GPU" target degenerates to the
/// host one (see `lbmini::raja::gpu::kHasDeviceBackend`).
using DeviceExecPolicy = HostExecPolicy;
#endif

/// True when RAJA was configured with an accelerator back end.
inline constexpr bool kHasDeviceBackend =
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
  true;
#else
  false;
#endif

/**
 * @brief Host-memory traits: plain `new[]` allocations, `memcpy` transfers.
 *
 * `CopyIn` / `CopyOut` are ordinary copies here; keeping them in the interface
 * (instead of letting host code touch the solver arrays directly) is what makes
 * `LbmTube` agnostic of where its buffers live.
 */
struct HostMemory {
  template<typename T>
  [[nodiscard]] static T* Allocate(const std::size_t count) {
    if (count == 0)
      return nullptr;
    // Value-initialised, so every buffer starts at zero like the `assign()` of
    // the host backends and the value-initialised `Kokkos::View`s.
    return new T[count]();
  }

  template<typename T>
  static void Release(T* ptr) noexcept {
    delete[] ptr;
  }

  template<typename T>
  static void CopyIn(T* deviceDst, const T* hostSrc, const std::size_t count) {
    std::memcpy(deviceDst, hostSrc, count * sizeof(T));
  }

  template<typename T>
  static void CopyOut(T* hostDst, const T* deviceSrc, const std::size_t count) {
    std::memcpy(hostDst, deviceSrc, count * sizeof(T));
  }

  static void Synchronize() {}
};

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
#if defined(RAJA_ENABLE_CUDA)
#define LBMINI_GPU_MALLOC       cudaMalloc
#define LBMINI_GPU_FREE         cudaFree
#define LBMINI_GPU_MEMSET       cudaMemset
#define LBMINI_GPU_MEMCPY       cudaMemcpy
#define LBMINI_GPU_H2D          cudaMemcpyHostToDevice
#define LBMINI_GPU_D2H          cudaMemcpyDeviceToHost
#define LBMINI_GPU_SYNCHRONIZE  cudaDeviceSynchronize
#else
#define LBMINI_GPU_MALLOC       hipMalloc
#define LBMINI_GPU_FREE         hipFree
#define LBMINI_GPU_MEMSET       hipMemset
#define LBMINI_GPU_MEMCPY       hipMemcpy
#define LBMINI_GPU_H2D          hipMemcpyHostToDevice
#define LBMINI_GPU_D2H          hipMemcpyDeviceToHost
#define LBMINI_GPU_SYNCHRONIZE  hipDeviceSynchronize
#endif

/**
 * @brief Device-memory traits: explicit `cudaMalloc` / `cudaMemcpy` pairs.
 *
 * This is the price RAJA charges for its portability: the loop bodies below are
 * written once, but the data motion the Kokkos backend gets for free from
 * `View` + `deep_copy` has to be spelled out here. The buffers are zeroed at
 * allocation so the solver sees the same initial state on every backend.
 */
struct DeviceMemory {
  template<typename T>
  [[nodiscard]] static T* Allocate(const std::size_t count) {
    if (count == 0)
      return nullptr;
    T* ptr = nullptr;
    if (LBMINI_GPU_MALLOC(&ptr, count * sizeof(T)) != 0 || ptr == nullptr)
      throw std::runtime_error("lbmini::raja: device allocation failed.");
    LBMINI_GPU_MEMSET(ptr, 0, count * sizeof(T));
    return ptr;
  }

  template<typename T>
  static void Release(T* ptr) noexcept {
    if (ptr)
      LBMINI_GPU_FREE(ptr);
  }

  template<typename T>
  static void CopyIn(T* deviceDst, const T* hostSrc, const std::size_t count) {
    LBMINI_GPU_MEMCPY(deviceDst, hostSrc, count * sizeof(T), LBMINI_GPU_H2D);
  }

  template<typename T>
  static void CopyOut(T* hostDst, const T* deviceSrc, const std::size_t count) {
    LBMINI_GPU_MEMCPY(hostDst, deviceSrc, count * sizeof(T), LBMINI_GPU_D2H);
  }

  static void Synchronize() {
    LBMINI_GPU_SYNCHRONIZE();
  }
};
#else
/// Without an accelerator build the device traits alias the host ones, so the
/// GPU instantiation still compiles (and runs on the CPU).
using DeviceMemory = HostMemory;
#endif

/// Backend tag bundling the host execution policy with host memory.
struct HostBackend {
  using ExecPolicy = HostExecPolicy;
  using Memory = HostMemory;
  static constexpr bool kIsDevice = false;
};

/// Backend tag bundling the accelerator execution policy with device memory.
struct DeviceBackend {
  using ExecPolicy = DeviceExecPolicy;
  using Memory = DeviceMemory;
  static constexpr bool kIsDevice = kHasDeviceBackend;
};
} // namespace lbmini::raja

#endif // LBMINI_RAJA_BACKENDS_HPP_
