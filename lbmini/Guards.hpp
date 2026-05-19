#ifndef LBMINI_GUARDS_HPP_
#define LBMINI_GUARDS_HPP_

/**
 * @file Guards.hpp
 * @brief Memory-safety guards shared by every backend.
 *
 * The mini-app is a benchmarking harness: the hot loops must stay exactly as
 * fast as they are today, so **none of these guards runs per cell or per time
 * step**. They come in three flavours, all of them paid for once:
 *
 *  - `ValidateMeshExtents()` — O(1), called from the constructor of every
 *    solver (and from the YAML reader). It rejects the inputs that turn the
 *    flat-index arithmetic (`i * ny + j`, `idc * N + cell`, `y % ny`,
 *    `i < nx / 2`) into an out-of-bounds access or a division by zero, and it
 *    rejects meshes whose cell count does not fit the `std::int32_t` streaming
 *    tables.
 *  - `VerifyIndexTable()` — O(table), called once after the streaming tables
 *    are built, i.e. once per run against thousands of steps of real work.
 *    This is what makes the unchecked gather in `stream()` safe: the indices
 *    are proven to be inside `[0, N)` *before* the first kernel launch, so the
 *    kernel itself keeps dereferencing them with no bounds check at all.
 *  - `LBMINI_ASSERT()` / `LBMINI_ASSERT_INDEX()` — compiled out entirely in
 *    `NDEBUG` builds (which is what the benchmarks use). They exist so that a
 *    Debug or sanitizer run pinpoints the violation instead of corrupting
 *    memory silently.
 *
 * See `docs/memory-safety.md` for the verification procedure these guards
 * belong to.
 */

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <Eigen/Core>

namespace lbmini {
/// Largest cell count representable by the `std::int32_t` streaming tables.
inline constexpr Eigen::Index kMaxCells = static_cast<Eigen::Index>(std::numeric_limits<std::int32_t>::max());

/**
 * @brief Rejects mesh extents that would break the index arithmetic.
 *
 * Every backend derives its buffer sizes and its flat indices from
 * `nx`, `ny` and the lattice speed count. This is the single place where the
 * assumptions behind that arithmetic are checked:
 *  - `nx >= 1`, `ny >= 1`: a zero extent means an empty allocation that the
 *    kernels still index, and `ny == 0` makes the streaming `y % ny` wrap a
 *    division by zero;
 *  - `nx * ny` must not overflow, and must stay inside `std::int32_t`, which
 *    is the type of the precomputed gather tables — past that the indices wrap
 *    negative and the gather reads a wild address;
 *  - `nx * ny * speeds * 4` (the table length) must not overflow `Index`.
 *
 * @throws std::invalid_argument on a degenerate or oversized mesh.
 */
inline void ValidateMeshExtents(
  const Eigen::Index nx,
  const Eigen::Index ny,
  const Eigen::Index speeds,
  const char* backend
) {
  using Index = Eigen::Index;

  const std::string where = std::string(backend) + ": mesh " + std::to_string(nx) + "x" + std::to_string(ny);

  if (nx < 1 || ny < 1)
    throw std::invalid_argument(where + " is degenerate; both extents must be >= 1.");

  if (nx > kMaxCells / ny)
    throw std::invalid_argument(
      where + " has more than " + std::to_string(kMaxCells) +
      " cells, which does not fit the int32 streaming tables."
    );

  const Index cells = nx * ny;
  constexpr Index kIndexMax = std::numeric_limits<Index>::max();
  if (speeds < 1 || cells > kIndexMax / (speeds * Index{ 4 }))
    throw std::invalid_argument(where + " overflows the streaming-table size for this lattice.");
}

/**
 * @brief Proves that a precomputed gather table only addresses valid cells.
 *
 * Runs once, right after the table is built, and is the reason the streaming
 * kernels may dereference `streamIdx_` without any per-element bounds check.
 * A failure here is a geometry bug, never a user error, so it aborts loudly
 * instead of clamping — a clamped index would silently produce wrong physics.
 *
 * @throws std::out_of_range if any entry falls outside `[0, cells)`.
 */
inline void VerifyIndexTable(
  const std::int32_t* table,
  const std::size_t count,
  const Eigen::Index cells,
  const char* backend
) {
  if (count == 0)
    return;

  if (table == nullptr)
    throw std::out_of_range(std::string(backend) + ": streaming table was never allocated.");

  const auto limit = static_cast<std::int32_t>(cells);
  for (std::size_t k = 0; k < count; ++k) {
    if (table[k] < 0 || table[k] >= limit)
      throw std::out_of_range(
        std::string(backend) + ": streaming index " + std::to_string(table[k]) +
        " at table position " + std::to_string(k) + " is outside [0, " + std::to_string(cells) + ")."
      );
  }
}

/**
 * @brief Same proof for a table already materialised in a host container.
 */
template<typename Container>
void VerifyIndexTable(const Container& table, const Eigen::Index cells, const char* backend) {
  VerifyIndexTable(table.data(), static_cast<std::size_t>(table.size()), cells, backend);
}
} // namespace lbmini

/**
 * @def LBMINI_ASSERT
 * @brief Debug-only invariant check. Compiled out in `NDEBUG` (benchmark)
 *        builds, so it can be used inside kernels without costing anything.
 */
#ifdef NDEBUG
#define LBMINI_ASSERT(condition, message) ((void)0)
#define LBMINI_ASSERT_INDEX(index, size) ((void)0)
#else
#define LBMINI_ASSERT(condition, message) assert((condition) && (message))
#define LBMINI_ASSERT_INDEX(index, size)                                    \
  assert((index) >= 0 && static_cast<long long>(index) <                    \
         static_cast<long long>(size) && "index out of bounds")
#endif

#endif // LBMINI_GUARDS_HPP_
