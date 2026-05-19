#include "Lbm/Cuda/Gpu/LbmTube.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include "Guards.hpp"
#include "Lbm/Cuda/Gpu/LatticeD2Q9.hpp"

namespace lbmini::cuda::gpu {
__constant__ int kD2Q9Cx[9] = { 0, 1, -1, 0, 0, 1, -1, 1, -1 };
__constant__ int kD2Q9Cy[9] = { 0, 0, 0, 1, -1, 1, -1, -1, 1 };

/**
 * @brief Recomputes macroscopic fields (rho, p, T, u) from f_ and g_.
 *
 * This CUDA kernel is executed with optimized grid/block dimensions for maximum warp occupancy.
 * Each thread handles a single cell and scalarizes operations via `#pragma unroll`
 * over the 9 discrete velocities (D2Q9 lattice) directly from global memory.
 */
template<typename Scalar>
__global__ void computeMacroscopicKernel(
  const Scalar* __restrict__ pF,
  const Scalar* __restrict__ pG,
  Scalar* __restrict__ pRho,
  Scalar* __restrict__ pP,
  Scalar* __restrict__ pT,
  Scalar* __restrict__ pU,
  const Scalar Ushift0,
  const Scalar Ushift1,
  const Scalar invCv,
  const Scalar Rgas,
  int nx,
  int ny,
  int N
) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell < N) {
    Scalar rho = Scalar{ 0 };
    Scalar nrg = Scalar{ 0 };
    Scalar mx = Scalar{ 0 };
    Scalar my = Scalar{ 0 };
    #pragma unroll 9
    for (int idc = 0; idc < 9; ++idc) {
      const Scalar fi = pF[idc * N + cell];
      const Scalar gi = pG[idc * N + cell];
      rho += fi;
      nrg += gi;
      mx += (static_cast<Scalar>(kD2Q9Cx[idc]) + Ushift0) * fi;
      my += (static_cast<Scalar>(kD2Q9Cy[idc]) + Ushift1) * fi;
    }

    pRho[cell] = rho;

    const Scalar invRho = Scalar{ 1 } / rho;
    const Scalar ux = mx * invRho;
    const Scalar uy = my * invRho;
    pU[0 * N + cell] = ux;
    pU[1 * N + cell] = uy;

    const Scalar kin = Scalar{ 0.5 } * (ux * ux + uy * uy);
    const Scalar Tv = (Scalar{ 0.5 } * nrg * invRho - kin) * invCv;
    pT[cell] = Tv;
    pP[cell] = Rgas * rho * Tv;
  }
}

/**
 * @brief Initializes f_ and g_ using local equilibria.
 *
 * This CUDA kernel calculates the Feq and Geq distributions from initial macroscopic
 * fields completely locally on the device to avoid data migration from the host.
 * It uses branchless logic and `#pragma unroll` for efficiency.
 */
template<typename Scalar>
__global__ void seedEquilibriaKernel(
  Scalar* __restrict__ pF,
  Scalar* __restrict__ pG,
  const Scalar* __restrict__ pRho,
  const Scalar* __restrict__ pT,
  const Scalar* __restrict__ pU,
  const Scalar Ushift0,
  const Scalar Ushift1,
  const Scalar kTiny,
  int nx,
  int ny,
  int N,
  const Scalar cv
) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell < N) {
    const Scalar rho = pRho[cell];
    const Scalar Tv = pT[cell];
    const Scalar ux = pU[0 * N + cell];
    const Scalar uy = pU[1 * N + cell];
    const Scalar u2 = ux * ux + uy * uy;

    // Feq (product form, branchless).
    const Scalar uia_x = ux - Ushift0;
    const Scalar uia_y = uy - Ushift1;
    const Scalar ux2t = uia_x * uia_x + Tv;
    const Scalar uy2t = uia_y * uia_y + Tv;
    #pragma unroll 9
    for (int idc = 0; idc < 9; ++idc) {
      const int vx = kD2Q9Cx[idc], vy = kD2Q9Cy[idc];
      const Scalar pfx = (vx == 0) ? (Scalar{ 1 } - ux2t) : (Scalar{ 0.5 } * (static_cast<Scalar>(vx) * uia_x + ux2t));
      const Scalar pfy = (vy == 0) ? (Scalar{ 1 } - uy2t) : (Scalar{ 0.5 } * (static_cast<Scalar>(vy) * uia_y + uy2t));
      pF[idc * N + cell] = rho * pfx * pfy;
    }

    // Geq: temperature-dependent weights, xi==0 at init.
    const Scalar E = cv * Tv + Scalar{ 0.5 } * u2;
    const Scalar targetE = Scalar{ 2 } * rho * E;
    const Scalar wZero = Scalar{ 1 } - Tv;
    const Scalar wNonZero = Scalar{ 0.5 } * Tv;
    Scalar sumW = Scalar{ 0 };
    #pragma unroll 9
    for (int idc = 0; idc < 9; ++idc) {
      const Scalar wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
      const Scalar wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
      sumW += wx * wy;
    }
    const Scalar sc = (sumW > kTiny) ? targetE / sumW : targetE / Scalar{ 9 };
    #pragma unroll 9
    for (int idc = 0; idc < 9; ++idc) {
      const Scalar wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
      const Scalar wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
      pG[idc * N + cell] = wx * wy * sc;
    }
  }
}

/**
 * @brief Fused BGK collision and equilibrium computation.
 *
 * This CUDA kernel minimizes memory traffic by performing BGK collision and local operations
 * primarily using registers. Thread divergence is carefully avoided by using a fixed
 * number of Newton-Raphson iterations and branchless ternary logic (`nrOk`, temperature weights).
 * The `feqLocal` array and other per-thread state are small enough to reside in registers.
 */
template<typename Scalar>
__global__ void collisionAndEquilibriaKernel(
  Scalar* __restrict__ pF,
  Scalar* __restrict__ pG,
  Scalar* __restrict__ pLastGx,
  const Scalar* __restrict__ pRho,
  const Scalar* __restrict__ pT,
  const Scalar* __restrict__ pU,
  const Scalar Ushift0,
  const Scalar Ushift1,
  const Scalar visc,
  const Scalar cond,
  const Scalar cp,
  const Scalar cv,
  const Scalar kTiny,
  const Scalar kMaxE,
  int nx,
  int ny,
  int N
) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell < N) {
    const Scalar rho = pRho[cell];
    const Scalar Tv = pT[cell];
    const Scalar ux = pU[0 * N + cell];
    const Scalar uy = pU[1 * N + cell];
    const Scalar u2 = ux * ux + uy * uy;

    // Temperature-dependent weights (branchless, compile-time selects).
    const Scalar wZero = Scalar{ 1 } - Tv;
    const Scalar wNonZero = Scalar{ 0.5 } * Tv;

    // Compute feq (product form, fully branchless)
    const Scalar uia_x = ux - Ushift0;
    const Scalar uia_y = uy - Ushift1;
    const Scalar ux2t = uia_x * uia_x + Tv;
    const Scalar uy2t = uia_y * uia_y + Tv;
    Scalar feqLocal[9];
    #pragma unroll 9
    for (int idc = 0; idc < 9; ++idc) {
      const int vx = kD2Q9Cx[idc], vy = kD2Q9Cy[idc];
      const Scalar pfx = (vx == 0) ? (Scalar{ 1 } - ux2t) : (Scalar{ 0.5 } * (static_cast<Scalar>(vx) * uia_x + ux2t));
      const Scalar pfy = (vy == 0) ? (Scalar{ 1 } - uy2t) : (Scalar{ 0.5 } * (static_cast<Scalar>(vy) * uia_y + uy2t));
      feqLocal[idc] = rho * pfx * pfy;
    }

    // Newton-Raphson for geq
    // nrOk accumulates as a Scalar (1.0 = all good, 0.0 = degenerate) to
    // keep all threads in the warp executing the same instructions (no
    // branch on bool that would cause warp divergence near shock waves).
    const Scalar E = cv * Tv + Scalar{ 0.5 } * u2;
    const Scalar targetE = Scalar{ 2 } * rho * E;

    const Scalar denom0 = (targetE > kTiny) ? targetE : kTiny;
    Scalar targetM[2];
    targetM[0] = Scalar{ 2 } * rho * ux * (E + Tv) / denom0;
    targetM[1] = Scalar{ 2 } * rho * uy * (E + Tv) / denom0;

    Scalar xi[2];
    xi[0] = pLastGx[1 * N + cell];
    xi[1] = pLastGx[2 * N + cell];

    Scalar alpha = Scalar{ 1 };
    Scalar si[9], e[9];
    Scalar Z = Scalar{ 0 };
    Scalar smax;
    Scalar nrOk = Scalar{ 1 }; // 1.0 = solver healthy, 0.0 = degenerate

    #pragma unroll 3
    for (int iter = 0; iter < 3; ++iter) {
      smax = -kMaxE;
      #pragma unroll 9
      for (int idc = 0; idc < 9; ++idc) {
        const Scalar s = xi[0] * (static_cast<Scalar>(kD2Q9Cx[idc]) + Ushift0) + xi[1] * (static_cast<Scalar>(kD2Q9Cy[idc]) + Ushift1);
        si[idc] = s;
        smax = (s > smax) ? s : smax;
      }
      smax = (smax < kMaxE) ? smax : kMaxE;

      Scalar S1[2] = { Scalar{ 0 }, Scalar{ 0 } };
      Scalar S2_00 = Scalar{ 0 }, S2_01 = Scalar{ 0 };
      Scalar S2_10 = Scalar{ 0 }, S2_11 = Scalar{ 0 };
      Z = Scalar{ 0 };
      #pragma unroll 9
      for (int idc = 0; idc < 9; ++idc) {
        Scalar expo = si[idc] - smax;
        expo = (expo < kMaxE) ? expo : kMaxE;
        expo = (expo > -kMaxE) ? expo : -kMaxE;
        const Scalar wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
        const Scalar wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
        const Scalar wev = wx * wy * std::exp(expo);
        e[idc] = wev;
        Z += wev;
        const Scalar c0 = static_cast<Scalar>(kD2Q9Cx[idc]) + Ushift0;
        const Scalar c1 = static_cast<Scalar>(kD2Q9Cy[idc]) + Ushift1;
        S1[0] += c0 * wev;
        S1[1] += c1 * wev;
        S2_00 += c0 * c0 * wev;
        S2_01 += c0 * c1 * wev;
        S2_10 += c1 * c0 * wev;
        S2_11 += c1 * c1 * wev;
      }

      const Scalar Zsafe = (Z > kTiny) ? Z : kTiny;
      // Accumulate degenerate flag as Scalar to avoid branch divergence.
      nrOk *= (Z > kTiny) ? Scalar{ 1 } : Scalar{ 0 };
      const Scalar invZ = Scalar{ 1 } / Zsafe;

      Scalar J00 = (S2_00 - S1[0] * S1[0] * invZ) * invZ + kTiny;
      Scalar J01 = (S2_01 - S1[0] * S1[1] * invZ) * invZ;
      Scalar J10 = (S2_10 - S1[1] * S1[0] * invZ) * invZ;
      Scalar J11 = (S2_11 - S1[1] * S1[1] * invZ) * invZ + kTiny;

      const Scalar detJ = J00 * J11 - J01 * J10;
      const Scalar absDet = std::fabs(detJ);
      nrOk *= (absDet > kTiny) ? Scalar{ 1 } : Scalar{ 0 };
      const Scalar detSafe = (absDet > kTiny) ? detJ : ((detJ >= Scalar{ 0 }) ? kTiny : -kTiny);
      const Scalar invDet = Scalar{ 1 } / detSafe;

      const Scalar r0 = S1[0] * invZ - targetM[0];
      const Scalar r1 = S1[1] * invZ - targetM[1];
      xi[0] -= alpha * (J11 * r0 - J01 * r1) * invDet;
      xi[1] -= alpha * (-J10 * r0 + J00 * r1) * invDet;
      alpha *= Scalar{ 0.5 };
    }

    // Compute final geqLocal (branchless, no if(solverOk))
    // nrOk == 1 → use NR result; nrOk == 0 → use uniform fallback.
    // Both branches are computed; the ternary select is warp-uniform.
    smax = -kMaxE;
    #pragma unroll 9
    for (int idc = 0; idc < 9; ++idc) {
      const Scalar s = xi[0] * (static_cast<Scalar>(kD2Q9Cx[idc]) + Ushift0) + xi[1] * (static_cast<Scalar>(kD2Q9Cy[idc]) + Ushift1);
      si[idc] = s;
      smax = (s > smax) ? s : smax;
    }
    smax = (smax < kMaxE) ? smax : kMaxE;
    Z = Scalar{ 0 };
    #pragma unroll 9
    for (int idc = 0; idc < 9; ++idc) {
      Scalar expo = si[idc] - smax;
      expo = (expo < kMaxE) ? expo : kMaxE;
      expo = (expo > -kMaxE) ? expo : -kMaxE;
      const Scalar wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
      const Scalar wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
      e[idc] = wx * wy * std::exp(expo);
      Z += e[idc];
    }

    // Uniform fallback: sumW of weights at xi==0 (all si==0, all expo==0).
    Scalar sumW = Scalar{ 0 };
    #pragma unroll 9
    for (int idc = 0; idc < 9; ++idc) {
      const Scalar wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
      const Scalar wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
      sumW += wx * wy;
    }
    const Scalar sumWsafe = (sumW > kTiny) ? sumW : Scalar{ 9 };
    const Scalar Zsafe = (Z > kTiny) ? Z : kTiny;

    // Branchless select: NR result when nrOk==1, uniform fallback when 0.
    const Scalar scaleNR = targetE / Zsafe;
    const Scalar scaleFB = targetE / sumWsafe;
    const Scalar scale = nrOk * scaleNR + (Scalar{ 1 } - nrOk) * scaleFB;

    Scalar geqLocal[9];
    #pragma unroll 9
    for (int idc = 0; idc < 9; ++idc) {
      const Scalar wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
      const Scalar wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
      // NR path: scale * e[idc]; fallback path: scale * wx*wy.
      geqLocal[idc] = scale * (nrOk * e[idc] + (Scalar{ 1 } - nrOk) * wx * wy);
    }

    // Store xi (branchless: zero in fallback path).
    pLastGx[0 * N + cell] = nrOk * (std::log(scaleNR / rho) - smax);
    pLastGx[1 * N + cell] = nrOk * xi[0];
    pLastGx[2 * N + cell] = nrOk * xi[1];

    // BGK + thermal correction
    const Scalar tau = visc / (rho * Tv) + Scalar{ 0.5 };
    const Scalar omega = Scalar{ 1 } / tau;
    const Scalar diff = cond / (rho * cp);
    const Scalar tauT = diff / Tv + Scalar{ 0.5 };
    Scalar omegaT = Scalar{ 1 } / tauT;

    // Non-equilibrium stress limiter (Tran et al. §3.3).
    Scalar eps = Scalar{ 0 };
    #pragma unroll 9
    for (int idc = 0; idc < 9; ++idc) {
      const Scalar fi = pF[idc * N + cell];
      const Scalar d = fi - feqLocal[idc];
      const Scalar den = (feqLocal[idc] > kTiny) ? feqLocal[idc] : kTiny;
      eps += std::fabs(d) / den;
    }
    eps /= Scalar{ 9 };
    const Scalar sigma = (eps >= Scalar{ 1 })
                           ? omega
                           : (eps >= Scalar{ 0.1 })
                           ? Scalar{ 1.35 }
                           : (eps >= Scalar{ 0.01 })
                           ? Scalar{ 1.05 }
                           : Scalar{ 1 };

    Scalar omegaL = omega / sigma;
    omegaL = (omegaL > Scalar{ 1 }) ? omegaL : Scalar{ 1 };
    omegaL = (omegaL < (Scalar{ 2 } - Scalar{ 1e-7 })) ? omegaL : (Scalar{ 2 } - Scalar{ 1e-7 });
    omegaT = (omegaT > Scalar{ 1 }) ? omegaT : Scalar{ 1 };
    omegaT = (omegaT < (Scalar{ 2 } - Scalar{ 1e-7 })) ? omegaT : (Scalar{ 2 } - Scalar{ 1e-7 });

    // L correction vector.
    Scalar L0 = Scalar{ 0 };
    Scalar L1 = Scalar{ 0 };
    #pragma unroll 9
    for (int idc = 0; idc < 9; ++idc) {
      const Scalar fi = pF[idc * N + cell];
      const Scalar cix0 = static_cast<Scalar>(kD2Q9Cx[idc]);
      const Scalar ciy0 = static_cast<Scalar>(kD2Q9Cy[idc]);
      const Scalar uvi = ux * cix0 + uy * ciy0;
      const Scalar aux = Scalar{ 2 } * uvi * (fi - feqLocal[idc]);
      L0 += aux * cix0;
      L1 += aux * ciy0;
    }

    const Scalar invT = Scalar{ 1 } / Tv;
    #pragma unroll 9
    for (int idc = 0; idc < 9; ++idc) {
      const int fi_idx = idc * N + cell;
      const Scalar fOld = pF[fi_idx];
      pF[fi_idx] = fOld + omegaL * (feqLocal[idc] - fOld);
      const Scalar c0 = static_cast<Scalar>(kD2Q9Cx[idc]) + Ushift0;
      const Scalar c1 = static_cast<Scalar>(kD2Q9Cy[idc]) + Ushift1;
      const Scalar cidotL = L0 * c0 + L1 * c1;
      const Scalar wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
      const Scalar wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
      const Scalar Wi = wx * wy;
      const Scalar gDiff = Wi * cidotL * invT;
      const Scalar gOld = pG[fi_idx];
      pG[fi_idx] = gOld + omegaL * (geqLocal[idc] - gOld) + (omegaL - omegaT) * gDiff;
    }
  }
}

/**
 * @brief Combined streaming and macroscopic update.
 *
 * This CUDA kernel fuses the streaming step with the macroscopic reduction to avoid an extra
 * device memory round-trip. It performs on-the-fly Inverse Distance Weighting (IDW) streaming
 * directly in registers without precomputed lookup tables to save memory bandwidth.
 * Uses fully branchless logic for fractional cell IDW evaluation (`exp(-p*log(r²+kTiny))`).
 */
template<typename Scalar, int NyTemplate>
__global__ void streamAndMacroscopicKernel(
  const Scalar* __restrict__ pF,
  const Scalar* __restrict__ pG,
  Scalar* __restrict__ pFaux,
  Scalar* __restrict__ pGaux,
  Scalar* __restrict__ pRho,
  Scalar* __restrict__ pP,
  Scalar* __restrict__ pT,
  Scalar* __restrict__ pU,
  const Scalar Ushift0,
  const Scalar Ushift1,
  const Scalar invCv,
  const Scalar Rgas,
  const Scalar kTiny,
  const Scalar idwExp,
  int nx,
  int ny,
  int N
) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell < N) {
    int ci, cj;
    if constexpr (NyTemplate == 1) {
      ci = cell;
      cj = 0;
    } else if constexpr (NyTemplate == 2) {
      ci = cell >> 1;
      cj = cell & 1;
    } else {
      ci = cell / ny;
      cj = cell % ny;
    }

    const Scalar negHalfIdw = Scalar{ -0.5 } * idwExp;

    Scalar rho = Scalar{ 0 };
    Scalar nrg = Scalar{ 0 };
    Scalar mx = Scalar{ 0 };
    Scalar my = Scalar{ 0 };

    #pragma unroll 9
    for (int idc = 0; idc < 9; ++idc) {
      const Scalar cix_s = static_cast<Scalar>(kD2Q9Cx[idc]) + Ushift0;
      const Scalar ciy_s = static_cast<Scalar>(kD2Q9Cy[idc]) + Ushift1;

      // Source position (fractional cell coordinates).
      const Scalar xSrc = static_cast<Scalar>(ci) - cix_s;
      const Scalar ySrc = static_cast<Scalar>(cj) - ciy_s;

      // Floor corners.
      const int x0 = static_cast<int>(std::floor(xSrc));
      const int y0 = static_cast<int>(std::floor(ySrc));
      const int x1 = x0 + 1;
      const int y1 = y0 + 1;

      // Clamp x (wall BC), wrap y (periodic BC).
      const int cx0i = (x0 < 0) ? 0 : (x0 >= nx ? nx - 1 : x0);
      const int cx1i = (x1 < 0) ? 0 : (x1 >= nx ? nx - 1 : x1);
      int wy0 = y0 % ny;
      if (wy0 < 0)
        wy0 += ny;
      int wy1 = y1 % ny;
      if (wy1 < 0)
        wy1 += ny;

      const int c00 = cx0i * ny + wy0;
      const int c10 = cx1i * ny + wy0;
      const int c01 = cx0i * ny + wy1;
      const int c11 = cx1i * ny + wy1;
      const int off = idc * N;

      Scalar fi, gi;

      // Diagonal shift: 2-D IDW via exp(-p*log(r²+kTiny)).
      // Branchless — no snap-to-node cascade. When any corner coincides
      // with the source point (r²→0), log(kTiny) is finite and large in
      // magnitude, so exp(·) → +∞ and after wsum normalisation the result
      // converges correctly to the node value.
      const Scalar dx0 = xSrc - static_cast<Scalar>(x0);
      const Scalar dy0 = ySrc - static_cast<Scalar>(y0);
      const Scalar dx1 = xSrc - static_cast<Scalar>(x1);
      const Scalar dy1 = ySrc - static_cast<Scalar>(y1);

      const Scalar w00 = std::exp(negHalfIdw * std::log(dx0 * dx0 + dy0 * dy0 + kTiny));
      const Scalar w10 = std::exp(negHalfIdw * std::log(dx1 * dx1 + dy0 * dy0 + kTiny));
      const Scalar w01 = std::exp(negHalfIdw * std::log(dx0 * dx0 + dy1 * dy1 + kTiny));
      const Scalar w11 = std::exp(negHalfIdw * std::log(dx1 * dx1 + dy1 * dy1 + kTiny));
      const Scalar wsum = w00 + w10 + w01 + w11;
      const Scalar invSum = Scalar{ 1 } / wsum;
      fi = (w00 * pF[off + c00] + w10 * pF[off + c10] + w01 * pF[off + c01] + w11 * pF[off + c11]) * invSum;
      gi = (w00 * pG[off + c00] + w10 * pG[off + c10] + w01 * pG[off + c01] + w11 * pG[off + c11]) * invSum;

      pFaux[off + cell] = fi;
      pGaux[off + cell] = gi;
      rho += fi;
      nrg += gi;
      mx += cix_s * fi;
      my += ciy_s * fi;
    }

    const Scalar invRho = Scalar{ 1 } / rho;
    const Scalar ux = mx * invRho;
    const Scalar uy = my * invRho;
    pU[0 * N + cell] = ux;
    pU[1 * N + cell] = uy;
    const Scalar kin = Scalar{ 0.5 } * (ux * ux + uy * uy);
    const Scalar Tv = (Scalar{ 0.5 } * nrg * invRho - kin) * invCv;
    pRho[cell] = rho;
    pT[cell] = Tv;
    pP[cell] = Rgas * rho * Tv;
  }
}


namespace internal {

template<typename Scalar>
void launchComputeMacroscopic(
  const Scalar* pF,
  const Scalar* pG,
  Scalar* pRho,
  Scalar* pP,
  Scalar* pT,
  Scalar* pU,
  const Scalar Ushift0,
  const Scalar Ushift1,
  const Scalar invCv,
  const Scalar Rgas,
  const int nx,
  const int ny,
  const int N
) {
  int threads = 64;
  int blocks = (N + threads - 1) / threads;
  computeMacroscopicKernel<Scalar><<<blocks, threads>>>(pF, pG, pRho, pP, pT, pU, Ushift0, Ushift1, invCv, Rgas, nx, ny, N);
  CheckCuda(cudaGetLastError(), "launch computeMacroscopicKernel");
}

template<typename Scalar>
void launchSeedEquilibria(
  Scalar* pF,
  Scalar* pG,
  const Scalar* pRho,
  const Scalar* pT,
  const Scalar* pU,
  const Scalar Ushift0,
  const Scalar Ushift1,
  const Scalar kTiny,
  const int nx,
  const int ny,
  const int N,
  const Scalar cv
) {
  int threads = 64;
  int blocks = (N + threads - 1) / threads;
  seedEquilibriaKernel<Scalar><<<blocks, threads>>>(pF, pG, pRho, pT, pU, Ushift0, Ushift1, kTiny, nx, ny, N, cv);
  CheckCuda(cudaGetLastError(), "launch seedEquilibriaKernel");
}

template<typename Scalar>
void launchCollisionAndEquilibria(
  Scalar* pF,
  Scalar* pG,
  Scalar* pLastGx,
  const Scalar* pRho,
  const Scalar* pT,
  const Scalar* pU,
  const Scalar Ushift0,
  const Scalar Ushift1,
  const Scalar visc,
  const Scalar cond,
  const Scalar cp,
  const Scalar cv,
  const Scalar kTiny,
  const Scalar kMaxE,
  const int nx,
  const int ny,
  const int N
) {
  int threads = 64;
  int blocks = (N + threads - 1) / threads;
  collisionAndEquilibriaKernel<Scalar><<<blocks, threads>>>(pF, pG, pLastGx, pRho, pT, pU, Ushift0, Ushift1, visc, cond, cp, cv, kTiny, kMaxE, nx, ny, N);
  CheckCuda(cudaGetLastError(), "launch collisionAndEquilibriaKernel");
}

template<typename Scalar>
void launchStreamAndMacroscopic(
  const Scalar* pF,
  const Scalar* pG,
  Scalar* pFaux,
  Scalar* pGaux,
  Scalar* pRho,
  Scalar* pP,
  Scalar* pT,
  Scalar* pU,
  const Scalar Ushift0,
  const Scalar Ushift1,
  const Scalar invCv,
  const Scalar Rgas,
  const Scalar kTiny,
  const Scalar idwExp,
  const int nx,
  const int ny,
  const int N
) {
  int threads = 64;
  int blocks = (N + threads - 1) / threads;
  if (ny == 1) {
    streamAndMacroscopicKernel<Scalar, 1><<<blocks, threads>>>(pF, pG, pFaux, pGaux, pRho, pP, pT, pU, Ushift0, Ushift1, invCv, Rgas, kTiny, idwExp, nx, ny, N);
  } else if (ny == 2) {
    streamAndMacroscopicKernel<Scalar, 2><<<blocks, threads>>>(pF, pG, pFaux, pGaux, pRho, pP, pT, pU, Ushift0, Ushift1, invCv, Rgas, kTiny, idwExp, nx, ny, N);
  } else {
    streamAndMacroscopicKernel<Scalar, 0><<<blocks, threads>>>(pF, pG, pFaux, pGaux, pRho, pP, pT, pU, Ushift0, Ushift1, invCv, Rgas, kTiny, idwExp, nx, ny, N);
  }
  CheckCuda(cudaGetLastError(), "launch streamAndMacroscopicKernel");
}

template void launchComputeMacroscopic<double>(
  const double*, const double*, double*, double*, double*, double*,
  double, double, double, double, int, int, int
);

template void launchSeedEquilibria<double>(
  double*, double*, const double*, const double*, const double*,
  double, double, double, int, int, int, double
);

template void launchCollisionAndEquilibria<double>(
  double*, double*, double*, const double*, const double*, const double*,
  double, double, double, double, double, double, double, double, int, int, int
);

template void launchStreamAndMacroscopic<double>(
  const double*, const double*, double*, double*, double*, double*,
  double*, double*, double, double, double, double, double, double, int, int, int
);

} // namespace internal
} // namespace lbmini::cuda::gpu
