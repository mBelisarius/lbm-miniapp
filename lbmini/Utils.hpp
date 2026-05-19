#ifndef LBMINI_UTILS_HPP_
#define LBMINI_UTILS_HPP_

#include <bit>
#include <cstdint>
#include <type_traits>

namespace lbmini {

template<typename Scalar_, typename Index>
Scalar_ Kronecker(Index a, Index b) {
  return (a == b) ? 1.0 : 0.0;
};

/**
 * Host-side replacement for `std::isfinite` that survives `-ffast-math`.
 *
 * `-ffast-math` implies `-ffinite-math-only`, under which the compiler folds
 * `std::isfinite(x)` to `true`, silently turning every NaN/Inf guard (solver
 * fallbacks, test assertions) into dead code. This inspects the exponent bits
 * with integer arithmetic, which fast-math never rewrites.
 */
template<typename Scalar_>
constexpr bool IsFinite(Scalar_ x) noexcept {
  static_assert(std::is_floating_point_v<Scalar_> && (sizeof(Scalar_) == 4 || sizeof(Scalar_) == 8),
                "IsFinite supports IEEE-754 binary32/binary64 only");
  if constexpr (sizeof(Scalar_) == 4) {
    constexpr std::uint32_t kExponentMask = 0x7F800000u;
    return (std::bit_cast<std::uint32_t>(x) & kExponentMask) != kExponentMask;
  } else {
    constexpr std::uint64_t kExponentMask = 0x7FF0000000000000ull;
    return (std::bit_cast<std::uint64_t>(x) & kExponentMask) != kExponentMask;
  }
}

}

#endif // LBMINI_UTILS_HPP_
