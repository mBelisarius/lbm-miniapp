#ifndef LBMINI_LBM_ENERGYCLOSURE_HPP_
#define LBMINI_LBM_ENERGYCLOSURE_HPP_

#include <type_traits>

/**
 * Newton-Raphson iterations of the energy closure, the same in every backend: a fixed count in place of a
 * stopping criterion keeps the cost per node independent of the flow state and the control flow uniform
 * across lanes and threads. Five is the largest number of iterations that the reference formulation
 * (Rathnayake et al.) needed to converge in its supersonic cases. It is a macro because it also feeds the
 * `unroll` pragmas, the OpenCL build options and the literal damping factors of the OpenACC lane kernel.
 * The damping of iteration `k` is `2^-k`.
 */
#ifndef LBMINI_NEWTON_ITERATIONS
#define LBMINI_NEWTON_ITERATIONS 5
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LBMINI_ENERGY_INLINE __host__ __device__ __forceinline__
#else
#define LBMINI_ENERGY_INLINE inline
#endif

namespace lbmini::energy {
inline constexpr int kNewtonIterations = LBMINI_NEWTON_ITERATIONS;
static_assert(kNewtonIterations >= 1, "at least one Newton iteration");

template<typename Value>
struct AxisMoments {
  Value z;
  Value jacobian;
  Value residual;
};

/// Body shared by the two overloads of `ScaledAxis`.
#define LBMINI_ENERGY_SCALED_AXIS_BODY                                          \
  const Value z = minus + zero + plus;                                          \
  const Value difference = plus - minus;                                        \
  return {                                                                      \
    z,                                                                          \
    (plus + minus) * z - difference * difference + tiny * (z * z),              \
    (difference + shiftMinusTarget * z) * z                                     \
  };

/**
 * Multiply the normalised residual and regularised variance of one axis by Z^2.
 * Their quotient is unchanged, but Newton no longer needs 1/Z.
 *
 * Two overloads: scalars (device-callable under nvcc/hipcc) and compiler vector types (host only).
 * A `__host__ __device__` signature that mentions a GCC vector type is an error in nvcc, and
 * `-DENABLE_KOKKOS_CUDA=ON` / `-DENABLE_RAJA_CUDA=ON` compile every translation unit with it.
 */
template<typename Value>
requires std::is_arithmetic_v<Value>
LBMINI_ENERGY_INLINE AxisMoments<Value> ScaledAxis(
  const Value minus, const Value zero, const Value plus, const Value shiftMinusTarget, const Value tiny
) {
  LBMINI_ENERGY_SCALED_AXIS_BODY
}

template<typename Value>
requires(!std::is_arithmetic_v<Value>)
inline AxisMoments<Value> ScaledAxis(
  const Value minus, const Value zero, const Value plus, const Value shiftMinusTarget, const Value tiny
) {
  LBMINI_ENERGY_SCALED_AXIS_BODY
}

#undef LBMINI_ENERGY_SCALED_AXIS_BODY
} // namespace lbmini::energy

#undef LBMINI_ENERGY_INLINE

#endif
