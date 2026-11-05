#ifndef LBMINI_LBM_SIMD_HPP_
#define LBMINI_LBM_SIMD_HPP_

/**
 * @file Simd.hpp
 * @brief Portable fixed-width SIMD lanes for the CPU solvers (`double` only).
 *
 * The CPU backends evaluate the per-cell physics for `W` neighbouring cells at
 * once. Relying on the compiler's auto-vectoriser for that is not an option:
 * the fused cell kernel is a straight-line block of ~850 operations with 8
 * exponentials (~2200 and 36 before the per-axis factorisation of the closure,
 * see `HostKernel.hpp`), a Newton loop and a few data-dependent branches, and every
 * mainstream auto-vectoriser gives up on it (the previous scalar kernels reached
 * ~10 % of the FMA peak of a core). Instead the kernel is written once
 * against `Ops<double, W>`:
 *
 *  - up to the native register width the lane type is a GCC vector extension
 *    (`__attribute__((vector_size))`, understood by GCC, Clang and `nvc++`), so
 *    every operator is a real vector instruction on any ISA, with no intrinsics;
 *  - wider than a register, `Wide` stacks `W / native` independent registers and
 *    applies every operation to each of them in turn. The fused kernel is a long
 *    chain of dependent operations (the Newton iterations, the serial sums over
 *    the nine directions), so one register's worth of cells is latency-bound;
 *    interleaving two or more independent blocks in the same instruction stream
 *    is what keeps the FMA ports busy. (GCC's own lowering of over-wide generic
 *    vectors keeps them in memory and is several times slower.)
 *
 * The vector extension has no transcendental functions, and `libmvec` is only
 * reachable through `-ffast-math` auto-vectorisation, so `Exp` is implemented
 * here: round-to-nearest range reduction plus a degree-13 Taylor polynomial,
 * within a few units in the last place of `std::exp` for the whole range
 * `[-708, 709]` (the solvers clamp the argument to `[-700, 700]`).
 *
 * `Ops<double, 1>` is the scalar instantiation of the same interface: it is the
 * fallback for compilers without the extension and the oracle the unit tests
 * compare the vector widths against.
 *
 * Everything here is written to survive `-ffast-math`, which the solvers are
 * built with: the "add 1.5 * 2^52 to round" trick and the two-constant
 * Cody-Waite reduction are reassociated into something less accurate by that
 * flag. The rounding trick is therefore used only behind `LBMINI_SIMD_OPAQUE`
 * (an empty `asm` statement that makes the sum opaque to the optimiser), which is
 * ~12 % faster than the portable convert-based rounding that remains as the
 * fallback; the second ln 2 constant is applied as a multiplicative correction
 * (see `Exp`).
 */

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

#if (defined(__x86_64__) || defined(__i386__)) && defined(__SSE2__) && (defined(__GNUC__) || defined(__clang__)) && \
  !defined(__NVCOMPILER)
#include <immintrin.h>
#define LBMINI_SIMD_X86 1
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(__NVCOMPILER)
#define LBMINI_SIMD_INLINE [[gnu::always_inline]] inline
#define LBMINI_SIMD_VECTOR_EXTENSION 1
#else
#define LBMINI_SIMD_INLINE inline
#endif

#if defined(__GNUC__) && !defined(__clang__) && !defined(__NVCOMPILER)
/// GCC compiles `mask ? a : b` on vector types to the best blend of the ISA
/// (`vblendvpd`, AVX-512 mask blends) and folds `a > b ? a : b` into `vmaxpd`.
#define LBMINI_SIMD_VECTOR_TERNARY 1
#endif

/// `LBMINI_SIMD_OPAQUE(v)`: the optimiser must treat `v` as unknown from here on (an empty
/// `asm` that "modifies" it in a vector register). `-ffast-math` cannot then fold the rounding
/// trick `(y + c) - c` of `Exp`. Left undefined where no constraint letter is known.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__NVCOMPILER)
#if defined(__x86_64__) || defined(__i386__)
#define LBMINI_SIMD_OPAQUE(v) __asm__("" : "+x"(v))
#elif defined(__aarch64__)
#define LBMINI_SIMD_OPAQUE(v) __asm__("" : "+w"(v))
#endif
#endif

namespace lbmini::simd {
/// Width of the widest vector register the target offers, in bytes.
constexpr int NativeBytes() {
#if defined(LBMINI_SIMD_BYTES)
  return LBMINI_SIMD_BYTES;
#elif !defined(LBMINI_SIMD_VECTOR_EXTENSION)
  return 8;
#elif defined(__AVX512F__)
  return 64;
#elif defined(__AVX__)
  return 32;
#elif defined(__SSE2__) || defined(__ARM_NEON) || defined(__aarch64__) || defined(__ALTIVEC__)
  return 16;
#else
  return 8;
#endif
}

/// Number of `double` lanes of the widest native vector.
inline constexpr int kNativeLanes = NativeBytes() / static_cast<int>(sizeof(double));

/// Independent registers per block of the kernel (see the file comment): enough registers to
/// make a block of 16 cells (the row padding of `host::Geometry`), whatever the register width.
/// The fused step is a long chain of dependent operations (three Newton iterations, each an
/// exponential followed by two divisions), so the more independent blocks are interleaved the
/// better the FMA ports are fed: measured on AVX2, 2 -> 4 registers is +12 % per core.
#if defined(LBMINI_SIMD_UNROLL)
inline constexpr int kUnroll = LBMINI_SIMD_UNROLL;
#else
inline constexpr int kUnroll = (kNativeLanes >= 16) ? 1 : 16 / kNativeLanes;
#endif

/// Cells per block of the solvers: `kUnroll` native registers.
inline constexpr int kLanes = (kNativeLanes == 1) ? 1 : kNativeLanes * kUnroll;

template<typename T, int W>
struct Ops;

namespace detail {
/// Taylor coefficients 1/k!, k = 2..13 (k = 0, 1 are folded into the Horner chain).
inline constexpr double kExpC2 = 1.0 / 2.0;
inline constexpr double kExpC3 = 1.0 / 6.0;
inline constexpr double kExpC4 = 1.0 / 24.0;
inline constexpr double kExpC5 = 1.0 / 120.0;
inline constexpr double kExpC6 = 1.0 / 720.0;
inline constexpr double kExpC7 = 1.0 / 5040.0;
inline constexpr double kExpC8 = 1.0 / 40320.0;
inline constexpr double kExpC9 = 1.0 / 362880.0;
inline constexpr double kExpC10 = 1.0 / 3628800.0;
inline constexpr double kExpC11 = 1.0 / 39916800.0;
inline constexpr double kExpC12 = 1.0 / 479001600.0;
inline constexpr double kExpC13 = 1.0 / 6227020800.0;
/// `ln 2` split in a 32-bit head (so `n * kLn2Hi` is exact for |n| < 2^11) and a tail.
inline constexpr double kLn2Hi = 6.93147180369123816490e-01;
inline constexpr double kLn2Lo = 1.90821492927058770002e-10;
inline constexpr double kLog2E = 1.44269504088896338700e+00;
/// Offset that makes `floor(y + kExpBias)` positive for every |y| <= 1010, and
/// the half that turns the truncation into a round-to-nearest.
inline constexpr double kExpBias = 1024.5;
/// `1.5 * 2^52`: adding it rounds a double of magnitude < 2^51 to the nearest integer in its low mantissa bits.
inline constexpr double kExpMagic = 6755399441055744.0;
} // namespace detail

/// Scalar instantiation: the reference every vector width is tested against.
template<>
struct Ops<double, 1> {
  using V = double;
  using M = bool;
  static constexpr int kLanes = 1;

  LBMINI_SIMD_INLINE static V Splat(const double x) { return x; }

  LBMINI_SIMD_INLINE static V Load(const double* p) { return *p; }

  LBMINI_SIMD_INLINE static void Store(double* p, const V v) { *p = v; }

  /// Store that bypasses the cache (identical to `Store` for a scalar).
  LBMINI_SIMD_INLINE static void StoreStream(double* p, const V v) { *p = v; }

  LBMINI_SIMD_INLINE static void StreamFence() {}

  LBMINI_SIMD_INLINE static V Select(const M m, const V a, const V b) { return m ? a : b; }

  LBMINI_SIMD_INLINE static V Min(const V a, const V b) { return (a < b) ? a : b; }

  LBMINI_SIMD_INLINE static V Max(const V a, const V b) { return (a > b) ? a : b; }

  LBMINI_SIMD_INLINE static V Abs(const V a) {
    return std::bit_cast<V>(std::bit_cast<std::uint64_t>(a) & 0x7FFFFFFFFFFFFFFFull);
  }

  LBMINI_SIMD_INLINE static M And(const M a, const M b) { return a && b; }

  LBMINI_SIMD_INLINE static M Or(const M a, const M b) { return a || b; }

  LBMINI_SIMD_INLINE static M True() { return true; }

  LBMINI_SIMD_INLINE static bool Any(const M m) { return m; }

  LBMINI_SIMD_INLINE static bool All(const M m) { return m; }

  LBMINI_SIMD_INLINE static double Lane(const V v, const int /*lane*/) { return v; }

  LBMINI_SIMD_INLINE static V Exp(const V x) {
    using namespace detail;
    const double y = x * kLog2E;
    const auto k = static_cast<std::int64_t>(y + kExpBias); // n + 1024, n = round(y)
    const double n = static_cast<double>(k) - 1024.0;
    const double r = x - n * kLn2Hi; // exact: kLn2Hi has 32 significant bits
    // exp(x) = 2^n * exp(r) * exp(-n * kLn2Lo). The tail enters as a *factor*: written as
    // `r - n * kLn2Lo` it is merged with the head by -ffast-math (`x - n * (hi + lo)`),
    // which silently drops the extra 32 bits of ln 2 and costs ~n * 2e-17 relative accuracy.
    const double t = n * (-kLn2Lo);
    const double tail = 1.0 + t * (1.0 + 0.5 * t);
    double p = kExpC13;
    p = p * r + kExpC12;
    p = p * r + kExpC11;
    p = p * r + kExpC10;
    p = p * r + kExpC9;
    p = p * r + kExpC8;
    p = p * r + kExpC7;
    p = p * r + kExpC6;
    p = p * r + kExpC5;
    p = p * r + kExpC4;
    p = p * r + kExpC3;
    p = p * r + kExpC2;
    p = p * r + 1.0;
    p = p * r + 1.0;
    const double scale = std::bit_cast<double>(static_cast<std::uint64_t>(k - 1) << 52); // 2^n
    return p * tail * scale;
  }
};

#ifdef LBMINI_SIMD_VECTOR_EXTENSION
/// Vector type of `W` lanes of `T`. A dependent `T` is required: with a plain `double` GCC
/// resolves the typedef before the template is instantiated and sees a scalar.
template<typename T, int W>
struct VectorOf {
  typedef T type __attribute__((vector_size(sizeof(T) * W)));
};

namespace detail {
/// `W`-lane GCC vector, `W` a power of two >= 2 that the target holds in one register.
template<int W>
struct Native {
  static_assert(W >= 2 && (W & (W - 1)) == 0, "lane count must be a power of two");

  using V = typename VectorOf<double, W>::type;
  /// Comparison result type of the compiler (all-ones / all-zeros 64-bit lanes).
  using M = decltype(std::declval<V>() < std::declval<V>());
  using I32 = typename VectorOf<std::int32_t, W>::type;

  static constexpr int kLanes = W;

  LBMINI_SIMD_INLINE static V Splat(const double x) {
    V r = {};
    return r + x;
  }

  LBMINI_SIMD_INLINE static V Load(const double* p) {
    V v;
    std::memcpy(&v, p, sizeof(V));
    return v;
  }

  LBMINI_SIMD_INLINE static void Store(double* p, const V v) { std::memcpy(p, &v, sizeof(V)); }

  /**
   * Store that bypasses the cache hierarchy (non-temporal), so that a line that is written once and
   * only read again by the next time step does not first have to be read from memory (the
   * write-allocate of an ordinary store: +50 % traffic of a streaming kernel). `p` must be aligned
   * to the register width; pair a run of streaming stores with `StreamFence`. Falls back to `Store`
   * where the target has no such instruction.
   */
  LBMINI_SIMD_INLINE static void StoreStream(double* p, const V v) {
#ifdef LBMINI_SIMD_X86
#if defined(__AVX512F__)
    if constexpr (W == 8) {
      _mm512_stream_pd(p, reinterpret<__m512d>(v));
      return;
    }
#endif
#if defined(__AVX__)
    if constexpr (W == 4) {
      _mm256_stream_pd(p, reinterpret<__m256d>(v));
      return;
    }
#endif
    if constexpr (W == 2) {
      _mm_stream_pd(p, reinterpret<__m128d>(v));
      return;
    }
#endif
    Store(p, v);
  }

  /// Orders the streaming stores of this thread before whatever follows (a no-op without them).
  LBMINI_SIMD_INLINE static void StreamFence() {
#ifdef LBMINI_SIMD_X86
    _mm_sfence();
#endif
  }

  LBMINI_SIMD_INLINE static V Select(const M m, const V a, const V b) {
#ifdef LBMINI_SIMD_VECTOR_TERNARY
    return m ? a : b;
#else
    return reinterpret<V>((reinterpret<M>(a) & m) | (reinterpret<M>(b) & ~m));
#endif
  }

  LBMINI_SIMD_INLINE static V Min(const V a, const V b) { return Select(a < b, a, b); }

  LBMINI_SIMD_INLINE static V Max(const V a, const V b) { return Select(a > b, a, b); }

  LBMINI_SIMD_INLINE static V Abs(const V a) {
    M mask = {};
    mask += static_cast<std::int64_t>(0x7FFFFFFFFFFFFFFFll);
    return reinterpret<V>(reinterpret<M>(a) & mask);
  }

  LBMINI_SIMD_INLINE static M And(const M a, const M b) { return a & b; }

  LBMINI_SIMD_INLINE static M Or(const M a, const M b) { return a | b; }

  LBMINI_SIMD_INLINE static M True() {
    M m = {};
    return ~m;
  }

  LBMINI_SIMD_INLINE static bool Any(const M m) {
    std::int64_t acc = 0;
    for (int l = 0; l < W; ++l)
      acc |= m[l];
    return acc != 0;
  }

  LBMINI_SIMD_INLINE static bool All(const M m) {
    std::int64_t acc = -1;
    for (int l = 0; l < W; ++l)
      acc &= m[l];
    return acc != 0;
  }

  LBMINI_SIMD_INLINE static double Lane(const V v, const int lane) { return v[lane]; }

  LBMINI_SIMD_INLINE static V Exp(const V x) {
    using namespace detail;
    const V y = x * kLog2E;
#ifdef LBMINI_SIMD_OPAQUE
    // n = round(y) with the "add 1.5 * 2^52" trick: the sum holds the integer in the low
    // mantissa bits, which also is the exponent field of 2^n once shifted (no int <-> double
    // conversions, one add and one subtract). -ffast-math would fold `(y + c) - c` to `y`, so the
    // sum is made opaque to the optimiser.
    V t = y + kExpMagic;
    LBMINI_SIMD_OPAQUE(t);
    const V n = t - kExpMagic;
#else
    // Truncating conversion of the biased argument (positive, so truncation == floor).
    const I32 k = __builtin_convertvector(y + kExpBias, I32);
    const V n = __builtin_convertvector(k, V) - 1024.0;
#endif
    const V r = x - n * kLn2Hi;
    const V tt = n * (-kLn2Lo); // the tail is a factor, see the scalar version
    const V tail = (tt * (tt * 0.5 + 1.0)) + 1.0;
    V p = Splat(kExpC13);
    p = p * r + kExpC12;
    p = p * r + kExpC11;
    p = p * r + kExpC10;
    p = p * r + kExpC9;
    p = p * r + kExpC8;
    p = p * r + kExpC7;
    p = p * r + kExpC6;
    p = p * r + kExpC5;
    p = p * r + kExpC4;
    p = p * r + kExpC3;
    p = p * r + kExpC2;
    p = p * r + 1.0;
    p = p * r + 1.0;
    // 2^n written straight into the exponent field: (n + 1023) << 52.
#ifdef LBMINI_SIMD_OPAQUE
    // The bit pattern of `t` is `0x4338... + n`; the constant is a multiple of 2^52 and shifts out.
    M bits = reinterpret<M>(t) + 1023;
#else
    const I32 e = k - 1; // n + 1023
    M bits = __builtin_convertvector(e, M);
#endif
    bits = bits << 52;
    return p * tail * reinterpret<V>(bits);
  }

private:
  /// Bit-pattern reinterpretation between same-sized vector types.
  template<typename To, typename From>
  LBMINI_SIMD_INLINE static To reinterpret(const From& from) {
    To to;
    std::memcpy(&to, &from, sizeof(To));
    return to;
  }
};

/**
 * @brief `U` independent registers of `Base` treated as one wide vector.
 *
 * Every operator and function applies the `Base` operation to each register in
 * turn; nothing is shared between the registers, so an out-of-order core
 * overlaps their dependency chains.
 */
template<typename Base, int U>
struct Wide {
  struct M {
    typename Base::M m[U];
  };

  struct V {
    typename Base::V v[U];

#define LBMINI_SIMD_WIDE_BINARY(OP)                                                       \
  LBMINI_SIMD_INLINE friend V operator OP(const V& a, const V& b) {                       \
    V r;                                                                                  \
    for (int u = 0; u < U; ++u)                                                           \
      r.v[u] = a.v[u] OP b.v[u];                                                          \
    return r;                                                                             \
  }                                                                                       \
  LBMINI_SIMD_INLINE friend V operator OP(const V& a, const double s) {                   \
    V r;                                                                                  \
    for (int u = 0; u < U; ++u)                                                           \
      r.v[u] = a.v[u] OP s;                                                               \
    return r;                                                                             \
  }                                                                                       \
  LBMINI_SIMD_INLINE friend V operator OP(const double s, const V& b) {                   \
    V r;                                                                                  \
    for (int u = 0; u < U; ++u)                                                           \
      r.v[u] = s OP b.v[u];                                                               \
    return r;                                                                             \
  }                                                                                       \
  LBMINI_SIMD_INLINE friend V& operator OP##=(V& a, const V& b) {                         \
    for (int u = 0; u < U; ++u)                                                           \
      a.v[u] = a.v[u] OP b.v[u];                                                          \
    return a;                                                                             \
  }

    LBMINI_SIMD_WIDE_BINARY(+)
    LBMINI_SIMD_WIDE_BINARY(-)
    LBMINI_SIMD_WIDE_BINARY(*)
    LBMINI_SIMD_WIDE_BINARY(/)
#undef LBMINI_SIMD_WIDE_BINARY

    LBMINI_SIMD_INLINE friend V operator-(const V& a) {
      V r;
      for (int u = 0; u < U; ++u)
        r.v[u] = -a.v[u];
      return r;
    }

#define LBMINI_SIMD_WIDE_COMPARE(OP)                                                      \
  LBMINI_SIMD_INLINE friend M operator OP(const V& a, const V& b) {                       \
    M r;                                                                                  \
    for (int u = 0; u < U; ++u)                                                           \
      r.m[u] = a.v[u] OP b.v[u];                                                          \
    return r;                                                                             \
  }

    LBMINI_SIMD_WIDE_COMPARE(<)
    LBMINI_SIMD_WIDE_COMPARE(>)
    LBMINI_SIMD_WIDE_COMPARE(<=)
    LBMINI_SIMD_WIDE_COMPARE(>=)
#undef LBMINI_SIMD_WIDE_COMPARE
  };

  static constexpr int kLanes = Base::kLanes * U;

  LBMINI_SIMD_INLINE static V Splat(const double x) {
    V r;
    for (int u = 0; u < U; ++u)
      r.v[u] = Base::Splat(x);
    return r;
  }

  LBMINI_SIMD_INLINE static V Load(const double* p) {
    V r;
    for (int u = 0; u < U; ++u)
      r.v[u] = Base::Load(p + u * Base::kLanes);
    return r;
  }

  LBMINI_SIMD_INLINE static void Store(double* p, const V& v) {
    for (int u = 0; u < U; ++u)
      Base::Store(p + u * Base::kLanes, v.v[u]);
  }

  LBMINI_SIMD_INLINE static void StoreStream(double* p, const V& v) {
    for (int u = 0; u < U; ++u)
      Base::StoreStream(p + u * Base::kLanes, v.v[u]);
  }

  LBMINI_SIMD_INLINE static void StreamFence() { Base::StreamFence(); }

  LBMINI_SIMD_INLINE static V Select(const M& m, const V& a, const V& b) {
    V r;
    for (int u = 0; u < U; ++u)
      r.v[u] = Base::Select(m.m[u], a.v[u], b.v[u]);
    return r;
  }

  LBMINI_SIMD_INLINE static V Min(const V& a, const V& b) {
    V r;
    for (int u = 0; u < U; ++u)
      r.v[u] = Base::Min(a.v[u], b.v[u]);
    return r;
  }

  LBMINI_SIMD_INLINE static V Max(const V& a, const V& b) {
    V r;
    for (int u = 0; u < U; ++u)
      r.v[u] = Base::Max(a.v[u], b.v[u]);
    return r;
  }

  LBMINI_SIMD_INLINE static V Abs(const V& a) {
    V r;
    for (int u = 0; u < U; ++u)
      r.v[u] = Base::Abs(a.v[u]);
    return r;
  }

  LBMINI_SIMD_INLINE static M And(const M& a, const M& b) {
    M r;
    for (int u = 0; u < U; ++u)
      r.m[u] = Base::And(a.m[u], b.m[u]);
    return r;
  }

  LBMINI_SIMD_INLINE static M Or(const M& a, const M& b) {
    M r;
    for (int u = 0; u < U; ++u)
      r.m[u] = Base::Or(a.m[u], b.m[u]);
    return r;
  }

  LBMINI_SIMD_INLINE static M True() {
    M r;
    for (int u = 0; u < U; ++u)
      r.m[u] = Base::True();
    return r;
  }

  LBMINI_SIMD_INLINE static bool Any(const M& m) {
    bool r = false;
    for (int u = 0; u < U; ++u)
      r = r || Base::Any(m.m[u]);
    return r;
  }

  LBMINI_SIMD_INLINE static bool All(const M& m) {
    bool r = true;
    for (int u = 0; u < U; ++u)
      r = r && Base::All(m.m[u]);
    return r;
  }

  LBMINI_SIMD_INLINE static double Lane(const V& v, const int lane) {
    return Base::Lane(v.v[lane / Base::kLanes], lane % Base::kLanes);
  }

  LBMINI_SIMD_INLINE static V Exp(const V& x) {
    V r;
    for (int u = 0; u < U; ++u)
      r.v[u] = Base::Exp(x.v[u]);
    return r;
  }
};

/// Registers of `Native<kNativeLanes>` stacked to `W` lanes; a single `Native` below that.
template<int W, bool kFits = (W <= kNativeLanes)>
struct Select;

template<int W>
struct Select<W, true> {
  using type = Native<W>;
};

template<int W>
struct Select<W, false> {
  static_assert(W % kNativeLanes == 0, "lane count must be a multiple of the native width");
  using type = Wide<Native<kNativeLanes>, W / kNativeLanes>;
};
} // namespace detail

/// `W`-lane instantiation, `W` a power of two >= 2.
template<int W>
struct Ops<double, W> : detail::Select<W>::type {};
#endif // LBMINI_SIMD_VECTOR_EXTENSION
} // namespace lbmini::simd

#endif // LBMINI_LBM_SIMD_HPP_
