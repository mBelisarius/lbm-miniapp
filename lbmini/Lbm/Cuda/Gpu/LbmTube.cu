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
    Scalar& invRho,
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
    invRho = Scalar{ 1 } / rhoSum;
    ux = momX * invRho;
    uy = momY * invRho;
    const Scalar kin = Scalar{ 0.5 } * (ux * ux + uy * uy);
    tem = (Scalar{ 0.5 } * nrg * invRho - kin) * p.invCv;
    rho = rhoSum;
    pres = p.rgas * rhoSum * tem;
  }

  /**
   * @brief `p[c + 1] = W_c * exp(xi * c - max_c' (xi * c'))` for `c = -1, 0, 1`: one axis of the closure.
   *
   * Double precision runs at 1/64 of the single-precision rate on consumer
   * GPUs and the kernel is bound by it, not by memory, so the number of FP64
   * instructions per cell is the cost. The D2Q9 weights and the exponents of
   * the energy equilibrium are tensor products, `W_q = Wx(c_x) Wy(c_y)` and
   * `s_q = xi_0 (c_x + Ux) + xi_1 (c_y + Uy)`, hence
   * `W_q exp(s_q - smax) = Px(c_x) Py(c_y)` (the `U` shift and the common maximum
   * cancel), with a single exponential per axis, `E = exp(-|xi|)`: the exponents
   * `xi c - max` are `0`, `-|xi|` and `-2 |xi|` in the order of the sign of `xi`,
   * i.e. `1`, `E` and `E^2`. That is 2 exponentials per evaluation (8 per cell and
   * step; the reference evaluates 9 per evaluation, 36 per cell and step) and it
   * turns the nine-term sums of the Newton step into three-term ones.
   *
   * It equals the reference to rounding error while neither of its clamps
   * (`|s - smax| <= 700`) is active; beyond that every omitted term is below
   * `exp(-700)` relative to the largest one.
   */
  __device__ __forceinline__ static void AxisExp(
    const Scalar xi,
    const Scalar wn,
    const Scalar wZero,
    Scalar (&p)[3]
  ) {
    const Scalar t = -::fabs(xi);
    const Scalar e = ::exp((t > -kMaxExp) ? t : -kMaxExp);
    const Scalar e2 = e * e;
    const bool pos = xi >= Scalar{ 0 };
    p[0] = wn * (pos ? e2 : Scalar{ 1 });
    p[1] = wZero * e;
    p[2] = wn * (pos ? Scalar{ 1 } : e2);
  }

  /// Equilibrium energy populations from the Newton solution `xi`.
  __device__ __forceinline__ static void GeqFromXi(
    const Scalar xi0,
    const Scalar xi1,
    const Scalar wn,
    const Scalar wZero,
    const Scalar targetE,
    Scalar (&geq)[kQ]
  ) {
    Scalar px[3];
    Scalar py[3];
    AxisExp(xi0, wn, wZero, px);
    AxisExp(xi1, wn, wZero, py);
    const Scalar scale = targetE / ((px[0] + px[1] + px[2]) * (py[0] + py[1] + py[2]));
    Scalar sx[3];
    #pragma unroll
    for (int k = 0; k < 3; ++k)
      sx[k] = scale * px[k];
    #pragma unroll
    for (int idc = 0; idc < kQ; ++idc)
      geq[idc] = sx[Vx(idc) + 1] * py[Vy(idc) + 1];
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
    const Scalar invRho,
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
    // The per-axis factors of `wi` (`W_q = Wx(c_x) Wy(c_y)`), see `AxisExp`.
    const Scalar wZero = Scalar{ 1 } - temc;
    const Scalar wn = wZero + (Scalar{ 0.5 } * temc - wZero);

    // Geq via Newton-Raphson on the normalised energy flux.
    //
    // The closure distribution `W_q exp(xi . (c_q + U))` is a product of an x and a
    // y factor, so x and y are independent: the Jacobian of the reference is
    // diagonal (its off-diagonal term is exactly `S2xy / Z - (S1x / Z) (S1y / Z) = 0`)
    // and the iteration is two scalar Newton steps,
    // `xi_a -= alpha (m1_a - target_a) / var_a`, each from three weighted
    // exponentials of one axis (see `AxisExp`). Divisions are 8 FP64 slots each,
    // so the two axes share one reciprocal per stage.
    const Scalar E = p.cv * temc + Scalar{ 0.5 } * u2;
    const Scalar targetE = Scalar{ 2 } * rhoc * E;
    const Scalar invE = Scalar{ 1 } / E;
    const Scalar targetM0 = ux * (E + temc) * invE; // 2 rho u (E + T) / (2 rho E)
    const Scalar targetM1 = uy * (E + temc) * invE;

    Scalar alpha = Scalar{ 1 };
    bool solverOk = true;

    #pragma unroll
    for (int iter = 0; iter < kMaxIter; ++iter) {
      Scalar px[3];
      Scalar py[3];
      AxisExp(xi0, wn, wZero, px);
      AxisExp(xi1, wn, wZero, py);
      const Scalar zx = px[0] + px[1] + px[2];
      const Scalar zy = py[0] + py[1] + py[2];
      const Scalar zxy = zx * zy;
      solverOk = solverOk && (zxy > kTiny);
      const Scalar rz = Scalar{ 1 } / zxy;
      const Scalar invZx = zy * rz;
      const Scalar invZy = zx * rz;

      // Moments of the unshifted velocity; `c_a + U_a` shifts the mean only.
      const Scalar m1x = (px[2] - px[0]) * invZx;
      const Scalar m1y = (py[2] - py[0]) * invZy;
      const Scalar jxx = ((px[2] + px[0]) * invZx - m1x * m1x) + kTiny;
      const Scalar jyy = ((py[2] + py[0]) * invZy - m1y * m1y) + kTiny;
      const Scalar jxy = jxx * jyy;
      solverOk = solverOk && (::fabs(jxy) > kTiny);

      const Scalar rj = alpha * (Scalar{ 1 } / jxy);
      const Scalar r0 = (m1x + p.ushiftX) - targetM0;
      const Scalar r1 = (m1y + p.ushiftY) - targetM1;
      xi0 -= r0 * jyy * rj; // alpha r0 / jxx
      xi1 -= r1 * jxx * rj; // alpha r1 / jyy
      alpha *= Scalar{ 0.5 };
    }

    Scalar geqLocal[kQ];
    if (solverOk) {
      GeqFromXi(xi0, xi1, wn, wZero, targetE, geqLocal);
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
    // `tau` and `tauT` share `1 / (rho T)`, `omega` and `omegaT` one division.
    const Scalar invT = Scalar{ 1 } / temc;
    const Scalar tau = p.viscosity * invRho * invT + Scalar{ 0.5 };
    const Scalar tauT = p.condOverCp * invRho * invT + Scalar{ 0.5 };
    const Scalar rTau = Scalar{ 1 } / (tau * tauT);
    const Scalar omega = tauT * rTau;
    Scalar omegaT = tau * rTau;

    Scalar eps = Scalar{ 0 };
    #pragma unroll
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar diff = f[idc] - feqLocal[idc];
      const Scalar den = (feqLocal[idc] > kTiny) ? feqLocal[idc] : kTiny;
      eps += ::fabs(diff) / den;
    }
    eps *= Scalar{ 1 } / Scalar(kQ);
    // `omega / sigma` with `sigma` in {omega, 1.35, 1.05, 1}, without a division.
    Scalar omegaL = omega;
    if (eps >= Scalar{ 1 })
      omegaL = Scalar{ 1 };
    else if (eps >= Scalar{ 0.1 })
      omegaL = omega * (Scalar{ 1 } / Scalar{ 1.35 });
    else if (eps >= Scalar{ 0.01 })
      omegaL = omega * (Scalar{ 1 } / Scalar{ 1.05 });

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
  Scalar invRho;
  Scalar ux;
  Scalar uy;
  Scalar tem;
  Scalar pres;
  Phys::Macroscopic(p, f, g, rho, invRho, ux, uy, tem, pres);
  rhoOut[cell] = rho;
  pOut[cell] = pres;
  temOut[cell] = tem;
  uOut[2 * cell + 0] = ux;
  uOut[2 * cell + 1] = uy;

  // First collision, cold Newton start.
  Scalar xi0 = Scalar{ 0 };
  Scalar xi1 = Scalar{ 0 };
  Phys::Collide(p, rho, invRho, ux, uy, tem, f, g, xi0, xi1);

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
  Scalar invRho;
  Scalar ux;
  Scalar uy;
  Scalar tem;
  Scalar pres;
  Phys::Macroscopic(p, f, g, rho, invRho, ux, uy, tem, pres);
  if constexpr (kStoreMacro) {
    rhoOut[cell] = rho;
    pOut[cell] = pres;
    temOut[cell] = tem;
    uOut[2 * cell + 0] = ux;
    uOut[2 * cell + 1] = uy;
  }

  Scalar xi0 = xiIo[0 * N + cell];
  Scalar xi1 = xiIo[1 * N + cell];

  Phys::Collide(p, rho, invRho, ux, uy, tem, f, g, xi0, xi1);

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
