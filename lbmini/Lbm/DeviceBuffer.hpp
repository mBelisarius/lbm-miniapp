#ifndef LBMINI_LBM_DEVICEBUFFER_HPP_
#define LBMINI_LBM_DEVICEBUFFER_HPP_

#include <cstddef>
#include <cstdlib>
#include <new>
#include <vector>

/**
 * @def LBMINI_UNROLL(N)
 * @brief Portable loop-unroll hint across GCC, Clang, nvc++ and NVCC.
 *
 * OpenMP 5.1's `#pragma omp unroll partial(N)` is not yet supported by
 * `nvc++` (NVIDIA HPC SDK) — it emits `invalid text in pragma`. To stay
 * vendor-neutral without falling back to manual unrolling, we wrap the
 * compiler's native unroll directive in a `_Pragma` macro:
 *   - GCC           → `#pragma GCC unroll N`
 *   - Clang / NVCC  → `#pragma unroll N`
 *   - nvc++ (NVHPC) → `#pragma unroll(N)`
 *   - other         → no-op (compiler may still unroll on its own)
 */
#define LBMINI_UNROLL_STR_(X) #X
#define LBMINI_UNROLL_STR(X)  LBMINI_UNROLL_STR_(X)
#if defined(__NVCOMPILER)
  #define LBMINI_UNROLL(N) _Pragma(LBMINI_UNROLL_STR(unroll(N)))
#elif defined(__clang__) || defined(__CUDACC__)
  #define LBMINI_UNROLL(N) _Pragma(LBMINI_UNROLL_STR(unroll N))
#elif defined(__GNUC__)
  #define LBMINI_UNROLL(N) _Pragma(LBMINI_UNROLL_STR(GCC unroll N))
#else
  #define LBMINI_UNROLL(N)
#endif

namespace lbmini {

/**
 * @brief Cache-line-aligned allocator for host-side LBM storage.
 *
 * Guarantees 64-byte alignment (cache-line / AVX-512 register width) so that
 * the flat SoA buffers produce fully aligned loads / stores in the hot
 * per-cell kernels. Implemented on top of `std::aligned_alloc`, which requires
 * the requested size to be a multiple of the alignment — we round up
 * internally.
 */
template <typename T, std::size_t Align = 64>
struct AlignedAllocator {
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal = std::true_type;

  template <class U>
  struct rebind { using other = AlignedAllocator<U, Align>; };

  AlignedAllocator() noexcept = default;
  template <class U>
  explicit AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

  [[nodiscard]] static T* allocate(std::size_t n) {
    if (n == 0) return nullptr;
    const std::size_t bytes = n * sizeof(T);
    const std::size_t rounded = ((bytes + Align - 1) / Align) * Align;
    void* p = std::aligned_alloc(Align, rounded);
    if (!p) throw std::bad_alloc();
    return static_cast<T*>(p);
  }

  static void deallocate(T* p, std::size_t) noexcept {
    std::free(p);
  }
};

template <class T, class U, std::size_t A>
inline bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return true; }
template <class T, class U, std::size_t A>
inline bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return false; }

/**
 * @brief Storage abstraction used by `LbmTube` for every large flat array.
 *
 * For the host-side backends (`plain`, `openmp::cpu`) this is simply an
 * aligned `std::vector`, giving us contiguous 64-byte-aligned memory that
 * vectorises cleanly and produces stable NUMA first-touch under OpenMP.
 *
 * GPU backends (`openmp::gpu`, future `cuda`) will provide a partial
 * specialisation of `DeviceBuffer<T>` that wraps `cudaMalloc*` /
 * `hipMalloc*` / `omp_target_alloc`, so the same `LbmTube` template
 * instantiates over host or device storage without a second rewrite. The
 * abstraction is kept intentionally small — only the operations used by
 * `LbmTube` (`assign`, `data`, `size`, `swap`, `begin/end`) are assumed.
 */
template <typename T>
using DeviceBuffer = std::vector<T, AlignedAllocator<T, 64>>;

} // namespace lbmini

#endif // LBMINI_LBM_DEVICEBUFFER_HPP_
