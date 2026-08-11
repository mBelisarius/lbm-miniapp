#ifndef LBMINI_KOKKOS_RUNTIME_HPP_
#define LBMINI_KOKKOS_RUNTIME_HPP_

#include <Kokkos_Core.hpp>

namespace lbmini::kokkos {
/**
 * @brief RAII owner of the Kokkos runtime lifetime.
 *
 * Kokkos requires `Kokkos::initialize()` to run before the first `View` is
 * allocated and `Kokkos::finalize()` to run after the last one is released.
 * The other backends of this mini-app have no such global state, so instead of
 * leaking framework-specific bootstrap code into `lbmini::Executor` the
 * runtime is owned by the solver itself: `LbmTube` declares a `Runtime` as its
 * **first** member, hence it is constructed before every `View` and destroyed
 * after all of them, which is exactly the ordering Kokkos mandates.
 *
 * The guard is idempotent — if the host application (or a previous solver
 * instance) already initialised Kokkos, this object becomes a no-op and leaves
 * finalisation to whoever owns the runtime. A `Kokkos::ScopeGuard` cannot be
 * used for that: it aborts when the runtime is already up.
 */
class Runtime {
public:
  /**
   * @brief Boots Kokkos if needed.
   * @param numThreads Host threads for the CPU execution space (0 = let Kokkos
   *        decide from `OMP_NUM_THREADS` / hardware concurrency).
   */
  explicit Runtime(const int numThreads = 0) {
    if (::Kokkos::is_initialized() || ::Kokkos::is_finalized())
      return;

    ::Kokkos::InitializationSettings settings;
    if (numThreads > 0)
      settings.set_num_threads(numThreads);
    ::Kokkos::initialize(settings);
    owned_ = true;
  }

  Runtime(const Runtime&) = delete;

  Runtime& operator=(const Runtime&) = delete;

  ~Runtime() {
    if (owned_ && ::Kokkos::is_initialized() && !::Kokkos::is_finalized())
      ::Kokkos::finalize();
  }

private:
  bool owned_ = false;
};
} // namespace lbmini::kokkos

#endif // LBMINI_KOKKOS_RUNTIME_HPP_
