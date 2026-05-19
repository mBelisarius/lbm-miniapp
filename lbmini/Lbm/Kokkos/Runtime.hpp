#ifndef LBMINI_KOKKOS_RUNTIME_HPP_
#define LBMINI_KOKKOS_RUNTIME_HPP_

#include <cstdlib>
#include <mutex>

#include <Kokkos_Core.hpp>

namespace lbmini::kokkos {
/**
 * @brief Process-lifetime owner of the Kokkos runtime.
 *
 * Kokkos requires `Kokkos::initialize()` to run before the first `View` is
 * allocated and `Kokkos::finalize()` to run after the last one is released,
 * and it cannot be re-initialised once finalised. The previous version of this
 * guard finalised as soon as the *first* solver — the one that happened to
 * boot the runtime — was destroyed, which left every `View` of any longer-lived
 * solver dangling: a textbook use-after-free, and one that only shows up when
 * several solvers exist at once (exactly what a benchmarking campaign or the
 * test binary does). See docs/memory-safety.md §5.3.
 *
 * The lifetime is therefore no longer tied to a solver at all:
 *  - initialisation happens **once per process**, under `std::call_once`, so
 *    concurrent constructions cannot race;
 *  - finalisation is deferred to process exit through `std::atexit`, i.e.
 *    after the last solver has been destroyed, and it still runs, so a leak
 *    check does not report the Kokkos internals as lost;
 *  - if the host application already owns Kokkos (a `ScopeGuard` in `main`, or
 *    another library), this class stays a complete no-op and never finalises
 *    what it did not start.
 *
 * `LbmTube` keeps it as its first member: constructing the guard before any
 * `View` remains mandatory, only the destruction side became a no-op.
 */
class Runtime {
public:
  /**
   * @brief Boots Kokkos if needed.
   * @param numThreads Host threads for the CPU execution space (0 = let Kokkos
   *        decide from `OMP_NUM_THREADS` / hardware concurrency).
   */
  explicit Runtime(const int numThreads = 0) {
    static std::once_flag once;
    std::call_once(
      once,
      [numThreads] {
        // Someone else owns the runtime (or already tore it down): stay out.
        if (::Kokkos::is_initialized() || ::Kokkos::is_finalized())
          return;

        ::Kokkos::InitializationSettings settings;
        if (numThreads > 0)
          settings.set_num_threads(numThreads);
        ::Kokkos::initialize(settings);

        std::atexit(
          [] {
            if (::Kokkos::is_initialized() && !::Kokkos::is_finalized())
              ::Kokkos::finalize();
          }
        );
      }
    );
  }

  Runtime(const Runtime&) = delete;

  Runtime& operator=(const Runtime&) = delete;

  Runtime(Runtime&&) = delete;

  Runtime& operator=(Runtime&&) = delete;

  // Deliberately trivial: destroying a solver must never finalise a runtime
  // that another solver's `View`s still depend on.
  ~Runtime() = default;
};
} // namespace lbmini::kokkos

#endif // LBMINI_KOKKOS_RUNTIME_HPP_
