#include "Lbm/Cuda/Gpu/LbmTube.hpp"

#include <cstddef>
#include <cstdint>

#include "Lbm/Cuda/Gpu/LatticeD2Q9.hpp"

/**
 * @file LbmTube.cu
 * @brief Kernels of `lbmini::cuda::gpu::LbmTube` (see the class for the design).
 *
 * Code-generation rules that keep the fused step kernel free of local memory:
 *
 *  - every direction / dimension loop is fully unrolled, so each lattice
 *    velocity is an immediate and every per-cell array (`f`, `g`, `feq`, `e`,
 *    ...) is a set of registers, never a dynamically indexed stack array;
 *  - all run-time constants (`KernelParams`: fluid data, shifted velocities,
 *    the per-direction stencil) are read from the `__grid_constant__` kernel
 *    parameter with compile-time offsets, i.e. as constant-bank operands of the
 *    arithmetic instructions themselves.
 */

namespace lbmini::cuda::gpu {
namespace {
/// One thread per cell. The step kernel needs ~150 registers per thread: a
/// 256-thread block then occupies most of an SM's 64 Ki registers (8 resident
/// warps), whereas three 128-thread blocks fit (12 warps), which hides more of
/// the double-precision latency (measured ~3% faster). Forcing 4 blocks
/// (`__launch_bounds__(128, 4)`, <= 128 registers) spills and is slower.
constexpr int kBlockSize = 128;

using internal::KernelParams;

/**
 * @brief Per-cell LBM physics on register arrays.
 *
 * Reproduces the arithmetic of the plain reference operation for operation
 * and in the same order. Values the reference recomputes inside its loops
 * (`Cshift`, `Cshift * Cshift`, `Weights(idc, T)`) are evaluated once, with
 * the same expression, on the host (`KernelParams`) or once per cell.
 */
template<typename Scalar>
struct Physics {
  using Lattice = LatticeD2Q9<Scalar>;
  using Params = KernelParams<Scalar>;
  static constexpr int kQ = internal::kQ;
  static constexpr int kD = internal::kD;
  static constexpr Scalar kTiny = Scalar{ 1.0e-12 };
  static constexpr Scalar kMaxExp = Scalar{ 700.0 };
  static constexpr int kMaxIter = 3;
  /// Bound on `M` below which `Exponentials()` factorises (`2 M < 600 < 700`).
  static constexpr Scalar kFactorBound = Scalar{ 300.0 };

  __device__ static constexpr int Vx(const int idc) { return static_cast<int>(Lattice::Velocity(idc, 0)); }

  __device__ static constexpr int Vy(const int idc) { return static_cast<int>(Lattice::Velocity(idc, 1)); }

  /**
   * @brief `Lattice::Weights(idc, tem)` for all directions.
   *
   * The reference evaluates `w = 1; w *= wZero + isNonZero * (wNonZero - wZero)`
   * per dimension, with `isNonZero` in {0, 1}: the factor is exactly `wZero` for
   * a zero component and `wZero + (wNonZero - wZero)` otherwise.
   */
  __device__ __forceinline__ static void Weights(const Scalar tem, Scalar (&w)[kQ]) {
    const Scalar wZero = Scalar{ 1 } - tem;
    const Scalar wNonZero = Scalar{ 0.5 } * tem;
    const Scalar factorNonZero = wZero + (wNonZero - wZero);
    #pragma unroll
    for (int idc = 0; idc < kQ; ++idc)
      w[idc] = (Vx(idc) == 0 ? wZero : factorNonZero) * (Vy(idc) == 0 ? wZero : factorNonZero);
  }

  /// Density equilibrium, factorised product form.
  __device__ __forceinline__ static void Feq(
    const Params& p,
    const Scalar rhoc,
    const Scalar ux,
    const Scalar uy,
    const Scalar temc,
    Scalar (&feq)[kQ]
  ) {
    const Scalar uiaX = ux - p.ushiftX;
    const Scalar uiaY = uy - p.ushiftY;
    const Scalar uia2tX = uiaX * uiaX + temc;
    const Scalar uia2tY = uiaY * uiaY + temc;
    #pragma unroll
    for (int idc = 0; idc < kQ; ++idc) {
      Scalar phi = rhoc;
      #pragma unroll
      for (int d = 0; d < kD; ++d) {
        const int vi = (d == 0) ? Vx(idc) : Vy(idc);
        const Scalar uia = (d == 0) ? uiaX : uiaY;
        const Scalar uia2t = (d == 0) ? uia2tX : uia2tY;
        Scalar pf;
        if (vi == 0)
          pf = Scalar{ 1 } - uia2t;
        else if (vi == 1)
          pf = Scalar{ 0.5 } * (uia + uia2t);
        else
          pf = Scalar{ 0.5 } * (-uia + uia2t);
        phi *= pf;
      }
      feq[idc] = phi;
    }
  }

  /// rho, u, T, p from one cell's populations.
  __device__ __forceinline__ static void Macroscopic(
    const Params& p,
    const Scalar (&f)[kQ],
    const Scalar (&g)[kQ],
    Scalar& rho,
    Scalar& ux,
    Scalar& uy,
    Scalar& tem,
    Scalar& pres
  ) {
    Scalar rhoSum = Scalar{ 0 };
    Scalar nrg = Scalar{ 0 };
    Scalar momX = Scalar{ 0 };
    Scalar momY = Scalar{ 0 };
    #pragma unroll
    for (int idc = 0; idc < kQ; ++idc) {
      rhoSum += f[idc];
      nrg += g[idc];
      momX += p.cs[0][idc] * f[idc];
      momY += p.cs[1][idc] * f[idc];
    }
    const Scalar invRho = Scalar{ 1 } / rhoSum;
    ux = momX * invRho;
    uy = momY * invRho;
    const Scalar kin = Scalar{ 0.5 } * (ux * ux + uy * uy);
    tem = (Scalar{ 0.5 } * nrg * invRho - kin) * p.invCv;
    rho = rhoSum;
    pres = p.rgas * rhoSum * tem;
  }

  /// `s_i = xi . (c_i + U)` and their (clamped) maximum.
  __device__ __forceinline__ static Scalar Exponents(
    const Params& p,
    const Scalar xi0,
    const Scalar xi1,
    Scalar (&si)[kQ]
  ) {
    Scalar smax = -kMaxExp;
    #pragma unroll
    for (int idc = 0; idc < kQ; ++idc) {
      // Reference: `s = 0; s += xi0 * cx; s += xi1 * cy`. Starting from the
      // first product differs only in the sign of an all-zero sum, which
      // neither the maximum nor `exp(s - smax)` can observe.
      Scalar s = xi0 * p.cs[0][idc];
      s += xi1 * p.cs[1][idc];
      si[idc] = s;
      smax = (s > smax) ? s : smax;
    }
    return (smax < kMaxExp) ? smax : kMaxExp;
  }

  /// `exp(clamp(s_i - smax))`, the reference's clamped exponent.
  __device__ __forceinline__ static Scalar ClampedExp(const Scalar s, const Scalar smax) {
    Scalar expo = s - smax;
    expo = (expo < kMaxExp) ? expo : kMaxExp;
    expo = (expo > -kMaxExp) ? expo : -kMaxExp;
    return ::exp(expo);
  }

  /**
   * @brief `e_i = W_i * exp(clamp(s_i - smax))` for all directions.
   *
   * Double precision runs at 1/64 of the single-precision rate on consumer
   * GPUs and the kernel is bound by it, not by memory; the 9 exponentials per
   * evaluation (36 per cell and step) are its largest cost. With
   * `c_ia + U_a = v_ia + U_a`, `v_ia` in {-1, 0, 1}, the exponent factorises:
   *
   *   exp(s_i - smax) = exp(xi . U - smax) * exp(xi_0)^v_ix * exp(xi_1)^v_iy,
   *
   * three exponentials instead of nine. It equals the reference to rounding
   * error whenever neither clamp can be active, which `|s_i| <= M =
   * (|xi_0| + |xi_1|) * cbound` guarantees for `2 M < 700` (the spread
   * `s_i - smax` is at least `-2 M`, and no factor can overflow). Otherwise —
   * including any NaN warm start — the reference evaluation is used verbatim.
   */
  __device__ __forceinline__ static void Exponentials(
    const Params& p,
    const Scalar xi0,
    const Scalar xi1,
    const Scalar (&wi)[kQ],
    Scalar (&e)[kQ]
  ) {
    Scalar si[kQ];
    const Scalar smax = Exponents(p, xi0, xi1, si);
    if ((::fabs(xi0) + ::fabs(xi1)) * p.cbound < kFactorBound) {
      const Scalar base = ::exp(xi0 * p.ushiftX + xi1 * p.ushiftY - smax);
      const Scalar ex = ::exp(xi0);
      const Scalar ey = ::exp(xi1);
      const Scalar bx[3] = { base / ex, base, base * ex }; // v_x = -1, 0, 1
      const Scalar invEy = Scalar{ 1 } / ey;
      #pragma unroll
      for (int idc = 0; idc < kQ; ++idc) {
        const Scalar b = bx[Vx(idc) + 1];
        const Scalar val = (Vy(idc) == 0) ? b : b * ((Vy(idc) > 0) ? ey : invEy);
        e[idc] = wi[idc] * val;
      }
    } else {
      #pragma unroll
      for (int idc = 0; idc < kQ; ++idc)
        e[idc] = wi[idc] * ClampedExp(si[idc], smax);
    }
  }

  /**
   * @brief Equilibria + Newton-Raphson energy closure + BGK collision, in place.
   *
   * `f`/`g` enter as the pre-collision populations of the cell and leave as the
   * post-collision ones; `xi0`/`xi1` are the Newton warm start (read and
   * updated). The Newton loop runs a fixed number of iterations with clamped
   * reciprocals and a sticky `solverOk` predicate, so control flow only
   * diverges in the (rare) degenerate fallback.
   */
  __device__ __forceinline__ static void Collide(
    const Params& p,
    const Scalar rhoc,
    const Scalar ux,
    const Scalar uy,
    const Scalar temc,
    Scalar (&f)[kQ],
    Scalar (&g)[kQ],
    Scalar& xi0,
    Scalar& xi1
  ) {
    const Scalar u2 = ux * ux + uy * uy;

    Scalar feqLocal[kQ];
    Feq(p, rhoc, ux, uy, temc, feqLocal);

    Scalar wi[kQ];
    Weights(temc, wi);

    // Geq via Newton-Raphson on the normalised energy flux.
    const Scalar E = p.cv * temc + Scalar{ 0.5 } * u2;
    const Scalar targetE = Scalar{ 2 } * rhoc * E;
    const Scalar targetM0 = Scalar{ 2 } * rhoc * ux * (E + temc) / targetE;
    const Scalar targetM1 = Scalar{ 2 } * rhoc * uy * (E + temc) / targetE;

    Scalar alpha = Scalar{ 1 };
    Scalar e[kQ];
    bool solverOk = true;

    #pragma unroll 1
    for (int iter = 0; iter < kMaxIter; ++iter) {
      Exponentials(p, xi0, xi1, wi, e);

      Scalar S1x = Scalar{ 0 };
      Scalar S1y = Scalar{ 0 };
      Scalar S2xx = Scalar{ 0 };
      Scalar S2xy = Scalar{ 0 };
      Scalar S2yy = Scalar{ 0 };
      Scalar Z = Scalar{ 0 };
      #pragma unroll
      for (int idc = 0; idc < kQ; ++idc) {
        const Scalar wev = e[idc];
        Z += wev;
        S1x += p.cs[0][idc] * wev;
        S1y += p.cs[1][idc] * wev;
        S2xx += p.css[0][idc] * wev;
        S2xy += p.css[1][idc] * wev; // == S2yx: (c_x c_y) w and (c_y c_x) w round alike
        S2yy += p.css[2][idc] * wev;
      }

      const Scalar Zsafe = (Z > kTiny) ? Z : kTiny;
      solverOk = solverOk && (Z > kTiny);
      const Scalar invZ = Scalar{ 1 } / Zsafe;

      // J is symmetric bit for bit: J[1][0] = J[0][1].
      const Scalar J00 = (S2xx - S1x * S1x * invZ) * invZ + kTiny;
      const Scalar J01 = (S2xy - S1x * S1y * invZ) * invZ;
      const Scalar J11 = (S2yy - S1y * S1y * invZ) * invZ + kTiny;

      const Scalar detJ = J00 * J11 - J01 * J01;
      const Scalar absDet = ::fabs(detJ);
      const Scalar detSafe = (absDet > kTiny) ? detJ : ((detJ >= Scalar{ 0 }) ? kTiny : -kTiny);
      solverOk = solverOk && (absDet > kTiny);
      const Scalar invDet = Scalar{ 1 } / detSafe;

      const Scalar r0 = S1x * invZ - targetM0;
      const Scalar r1 = S1y * invZ - targetM1;
      xi0 -= alpha * (J11 * r0 - J01 * r1) * invDet;
      xi1 -= alpha * (-J01 * r0 + J00 * r1) * invDet;
      alpha *= Scalar{ 0.5 };
    }

    Scalar geqLocal[kQ];
    if (solverOk) {
      // Recompute e[], Z with the final xi.
      Exponentials(p, xi0, xi1, wi, geqLocal);
      Scalar Z = Scalar{ 0 };
      #pragma unroll
      for (int idc = 0; idc < kQ; ++idc)
        Z += geqLocal[idc];
      const Scalar scale = targetE / Z;
      #pragma unroll
      for (int idc = 0; idc < kQ; ++idc)
        geqLocal[idc] = scale * geqLocal[idc];
      // The reference also stores `log(scale / rho) - smax` as warm-start
      // component 0; nothing ever reads it, so it is neither computed nor stored.
    } else {
      // Fallback: normalised-Wi distribution, cold restart of the warm start.
      xi0 = Scalar{ 0 };
      xi1 = Scalar{ 0 };
      Scalar sumW = Scalar{ 0 };
      #pragma unroll
      for (int idc = 0; idc < kQ; ++idc)
        sumW += wi[idc];
      if (sumW <= kTiny) {
        const Scalar uni = targetE / Scalar(kQ);
        #pragma unroll
        for (int idc = 0; idc < kQ; ++idc)
          geqLocal[idc] = uni;
      } else {
        const Scalar sc = targetE / sumW;
        #pragma unroll
        for (int idc = 0; idc < kQ; ++idc)
          geqLocal[idc] = wi[idc] * sc;
      }
    }

    // Collision (BGK + Knudsen sensor + thermal-relaxation correction).
    const Scalar tau = p.viscosity / (rhoc * temc) + Scalar{ 0.5 };
    const Scalar omega = Scalar{ 1 } / tau;
    const Scalar diffusivity = p.conductivity / (rhoc * p.cp);
    const Scalar tauT = diffusivity / temc + Scalar{ 0.5 };
    Scalar omegaT = Scalar{ 1 } / tauT;

    Scalar eps = Scalar{ 0 };
    #pragma unroll
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar diff = f[idc] - feqLocal[idc];
      const Scalar den = (feqLocal[idc] > kTiny) ? feqLocal[idc] : kTiny;
      eps += ::fabs(diff) / den;
    }
    eps /= Scalar(kQ);
    Scalar sigma = Scalar{ 1 };
    if (eps >= Scalar{ 1 })
      sigma = omega;
    else if (eps >= Scalar{ 0.1 })
      sigma = Scalar{ 1.35 };
    else if (eps >= Scalar{ 0.01 })
      sigma = Scalar{ 1.05 };

    Scalar omegaL = omega / sigma;
    omegaL = (omegaL > Scalar{ 1 }) ? omegaL : Scalar{ 1 };
    omegaL = (omegaL < (Scalar{ 2 } - Scalar{ 1e-7 })) ? omegaL : (Scalar{ 2 } - Scalar{ 1e-7 });
    omegaT = (omegaT > Scalar{ 1 }) ? omegaT : Scalar{ 1 };
    omegaT = (omegaT < (Scalar{ 2 } - Scalar{ 1e-7 })) ? omegaT : (Scalar{ 2 } - Scalar{ 1e-7 });

    // L[a] = sum_i 2 * (u . c_i) * c_i_a * (f - feq)   (c_i unshifted)
    Scalar L0 = Scalar{ 0 };
    Scalar L1 = Scalar{ 0 };
    #pragma unroll
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar cix0 = static_cast<Scalar>(Vx(idc));
      const Scalar ciy0 = static_cast<Scalar>(Vy(idc));
      const Scalar uvi = ux * cix0 + uy * ciy0;
      const Scalar aux = Scalar{ 2 } * uvi * (f[idc] - feqLocal[idc]);
      L0 += aux * cix0;
      L1 += aux * ciy0;
    }

    const Scalar invT = Scalar{ 1 } / temc;
    #pragma unroll
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar fOld = f[idc];
      f[idc] = fOld + omegaL * (feqLocal[idc] - fOld);
      const Scalar cidotL = L0 * p.cs[0][idc] + L1 * p.cs[1][idc];
      const Scalar gDiff = wi[idc] * cidotL * invT;
      const Scalar gOld = g[idc];
      g[idc] = gOld + omegaL * (geqLocal[idc] - gOld) + (omegaL - omegaT) * gDiff;
    }
  }
};

/**
 * @brief Initial state: Sod left/right macroscopics, equilibrium seeding, the
 *        macroscopic recovery of the reference `Init()`, and the first
 *        collision with a cold Newton start — one pass, nothing transferred
 *        from the host.
 */
template<typename Scalar>
__global__ void __launch_bounds__(kBlockSize) InitKernel(
  const __grid_constant__ KernelParams<Scalar> p,
  Scalar* __restrict__ fOut,
  Scalar* __restrict__ gOut,
  Scalar* __restrict__ xiOut,
  Scalar* __restrict__ rhoOut,
  Scalar* __restrict__ pOut,
  Scalar* __restrict__ temOut,
  Scalar* __restrict__ uOut
) {
  using Phys = Physics<Scalar>;
  constexpr int kQ = Phys::kQ;

  const std::int32_t cell = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cell >= p.cells)
    return;
  const std::ptrdiff_t N = p.cells;
  const std::int32_t i = cell / p.ny;

  // Sod-like left/right state (reference `Init()` host loop).
  const bool left = i < p.nx / 2;
  const Scalar rho0 = left ? p.densityL : p.densityR;
  const Scalar p0 = left ? p.pressureL : p.pressureR;
  const Scalar tem0 = LatticeD2Q9<Scalar>::Cs2 * p0 / (rho0 * p.rgas);
  const Scalar ux0 = Scalar{ 0 };
  const Scalar uy0 = Scalar{ 0 };

  // seedEquilibria(): f = feq, g = Wi(T)-normalised geq (xi == 0).
  Scalar f[kQ];
  Scalar g[kQ];
  Phys::Feq(p, rho0, ux0, uy0, tem0, f);
  Scalar wi[kQ];
  Phys::Weights(tem0, wi);
  const Scalar u2 = ux0 * ux0 + uy0 * uy0;
  const Scalar E = p.cv * tem0 + Scalar{ 0.5 } * u2;
  const Scalar targetE = Scalar{ 2 } * rho0 * E;
  Scalar sumW = Scalar{ 0 };
  #pragma unroll
  for (int idc = 0; idc < kQ; ++idc)
    sumW += wi[idc];
  const Scalar sc = (sumW > Phys::kTiny) ? targetE / sumW : targetE / Scalar(kQ);
  #pragma unroll
  for (int idc = 0; idc < kQ; ++idc)
    g[idc] = ((sumW > Phys::kTiny) ? wi[idc] : Scalar{ 1 }) * sc;

  // computeMacroscopic(): the state every getter sees after Init().
  Scalar rho;
  Scalar ux;
  Scalar uy;
  Scalar tem;
  Scalar pres;
  Phys::Macroscopic(p, f, g, rho, ux, uy, tem, pres);
  rhoOut[cell] = rho;
  pOut[cell] = pres;
  temOut[cell] = tem;
  uOut[2 * cell + 0] = ux;
  uOut[2 * cell + 1] = uy;

  // First collision, cold Newton start.
  Scalar xi0 = Scalar{ 0 };
  Scalar xi1 = Scalar{ 0 };
  Phys::Collide(p, rho, ux, uy, tem, f, g, xi0, xi1);

  #pragma unroll
  for (int idc = 0; idc < kQ; ++idc) {
    fOut[idc * N + cell] = f[idc];
    gOut[idc * N + cell] = g[idc];
  }
  xiOut[0 * N + cell] = xi0;
  xiOut[1 * N + cell] = xi1;
}

/**
 * @brief One full time step: pull streaming + macroscopics + collision.
 *
 * @tparam kStoreMacro write `rho`, `p`, `T`, `u` (only needed on the last step
 *         of a `Run()`, the only point where a getter can observe them).
 */
template<typename Scalar, bool kStoreMacro>
__global__ void __launch_bounds__(kBlockSize) StepKernel(
  const __grid_constant__ KernelParams<Scalar> p,
  const Scalar* __restrict__ fIn,
  const Scalar* __restrict__ gIn,
  Scalar* __restrict__ fOut,
  Scalar* __restrict__ gOut,
  Scalar* __restrict__ xiIo,
  Scalar* __restrict__ rhoOut,
  Scalar* __restrict__ pOut,
  Scalar* __restrict__ temOut,
  Scalar* __restrict__ uOut
) {
  using Phys = Physics<Scalar>;
  constexpr int kQ = Phys::kQ;

  const std::int32_t cell = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cell >= p.cells)
    return;
  const std::ptrdiff_t N = p.cells;
  const std::int32_t nx = p.nx;
  const std::int32_t ny = p.ny;
  const std::int32_t i = cell / ny;
  const std::int32_t j = cell - i * ny;
  const auto& st = p.stencil;

  // Pull the post-collision populations of the previous step
  // (same gather as `lbmini::GatherSources`).
  Scalar f[kQ];
  Scalar g[kQ];
  #pragma unroll
  for (int idc = 0; idc < kQ; ++idc) {
    const Scalar* __restrict__ fp = fIn + idc * N;
    const Scalar* __restrict__ gp = gIn + idc * N;
    if (st.nearest[idc]) {
      // Degenerate-weight fallback of the reference table (uniform branch:
      // `nearest` depends on the direction only).
      const auto xN = static_cast<std::int32_t>(::round(static_cast<Scalar>(i) - st.cix[idc]));
      const auto yN = static_cast<std::int32_t>(::round(static_cast<Scalar>(j) - st.ciy[idc]));
      std::int32_t yW = yN % ny;
      yW += (yW < 0) ? ny : 0;
      const std::int32_t src = (xN < 0 || xN >= nx) ? cell : xN * ny + yW;
      f[idc] = fp[src];
      g[idc] = gp[src];
    } else {
      const std::int32_t xa = i + st.ox[idc];
      const std::int32_t rowA = ((xa < 0) ? 0 : ((xa >= nx) ? nx - 1 : xa)) * ny;
      const std::int32_t xb = xa + 1;
      const std::int32_t rowB = ((xb < 0) ? 0 : ((xb >= nx) ? nx - 1 : xb)) * ny;
      std::int32_t ya = j + st.oy0[idc];
      ya -= (ya >= ny) ? ny : 0;
      std::int32_t yb = j + st.oy1[idc];
      yb -= (yb >= ny) ? ny : 0;

      const Scalar w0 = st.w[idc][0];
      const Scalar w1 = st.w[idc][1];
      const Scalar w2 = st.w[idc][2];
      const Scalar w3 = st.w[idc][3];
      f[idc] = w0 * fp[rowA + ya] + w1 * fp[rowB + ya] + w2 * fp[rowA + yb] + w3 * fp[rowB + yb];
      g[idc] = w0 * gp[rowA + ya] + w1 * gp[rowB + ya] + w2 * gp[rowA + yb] + w3 * gp[rowB + yb];
    }
  }

  Scalar rho;
  Scalar ux;
  Scalar uy;
  Scalar tem;
  Scalar pres;
  Phys::Macroscopic(p, f, g, rho, ux, uy, tem, pres);
  if constexpr (kStoreMacro) {
    rhoOut[cell] = rho;
    pOut[cell] = pres;
    temOut[cell] = tem;
    uOut[2 * cell + 0] = ux;
    uOut[2 * cell + 1] = uy;
  }

  Scalar xi0 = xiIo[0 * N + cell];
  Scalar xi1 = xiIo[1 * N + cell];

  Phys::Collide(p, rho, ux, uy, tem, f, g, xi0, xi1);

  #pragma unroll
  for (int idc = 0; idc < kQ; ++idc) {
    fOut[idc * N + cell] = f[idc];
    gOut[idc * N + cell] = g[idc];
  }
  xiIo[0 * N + cell] = xi0;
  xiIo[1 * N + cell] = xi1;
}

unsigned int GridSize(const std::int32_t cells) {
  return static_cast<unsigned int>((cells + kBlockSize - 1) / kBlockSize);
}
} // namespace

namespace internal {
template<typename Scalar>
void LaunchInit(
  const KernelParams<Scalar>& params,
  Scalar* fOut,
  Scalar* gOut,
  const DeviceFields<Scalar>& fields,
  cudaStream_t stream
) {
  InitKernel<Scalar><<<GridSize(params.cells), kBlockSize, 0, stream>>>(
    params, fOut, gOut, fields.xi, fields.rho, fields.p, fields.tem, fields.u
  );
  CheckCuda(cudaGetLastError(), "launch InitKernel");
}

template<typename Scalar>
void LaunchStep(
  const KernelParams<Scalar>& params,
  const Scalar* fIn,
  const Scalar* gIn,
  Scalar* fOut,
  Scalar* gOut,
  const DeviceFields<Scalar>& fields,
  const bool storeMacro,
  cudaStream_t stream
) {
  const unsigned int grid = GridSize(params.cells);
  if (storeMacro)
    StepKernel<Scalar, true><<<grid, kBlockSize, 0, stream>>>(
      params, fIn, gIn, fOut, gOut, fields.xi, fields.rho, fields.p, fields.tem, fields.u
    );
  else
    StepKernel<Scalar, false><<<grid, kBlockSize, 0, stream>>>(
      params, fIn, gIn, fOut, gOut, fields.xi, fields.rho, fields.p, fields.tem, fields.u
    );
  CheckCuda(cudaGetLastError(), "launch StepKernel");
}

template void LaunchInit<double>(
  const KernelParams<double>&, double*, double*, const DeviceFields<double>&, cudaStream_t
);

template void LaunchStep<double>(
  const KernelParams<double>&, const double*, const double*, double*, double*, const DeviceFields<double>&, bool,
  cudaStream_t
);
} // namespace internal
} // namespace lbmini::cuda::gpu
