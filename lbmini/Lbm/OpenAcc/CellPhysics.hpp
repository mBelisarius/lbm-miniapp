#ifndef LBMINI_OPENACC_CELLPHYSICS_HPP_
#define LBMINI_OPENACC_CELLPHYSICS_HPP_

/**
 * @file CellPhysics.hpp
 * @brief Per-cell D2Q9 physics shared by the NVIDIA HPC SDK directive backends
 *        (`openacc::cpu`, `openacc::gpu`, `openmp::gpu`).
 *
 * Plain inline C++ without any directive: nvc++ compiles a device version of
 * every function whose definition is visible where an OpenACC compute region
 * or an OpenMP `target` region calls it, and inlines it there, so all per-cell
 * arrays stay in registers. The functions reproduce the arithmetic of the
 * plain reference solver (and of `lbmini::alpaka`) operation for operation.
 *
 * The D2Q9 velocity set is spelled out as `constexpr` conditional expressions
 * instead of a lookup table: every loop over the directions is fully unrolled,
 * each `Cx(idc)` / `Cy(idc)` folds to an immediate, and no static array has to
 * be made visible to (or loaded on) the device. nvc++ 26.1 also cannot place
 * the `static constexpr` member tables of the lattice templates on the device.
 */

#include <cmath>
#include <cstdint>

#include "Data.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/Stencil.hpp"

namespace lbmini::directive {
inline constexpr int kQ = 9; ///< D2Q9 speeds
inline constexpr int kD = 2; ///< spatial dimensions

/// `c_ix` of direction `idc` of the D2Q9 set (same order as the lattice tables).
constexpr int Cx(const int idc) {
  return (idc == 1 || idc == 5 || idc == 7) ? 1 : ((idc == 2 || idc == 6 || idc == 8) ? -1 : 0);
}

/// `c_iy` of direction `idc` of the D2Q9 set (same order as the lattice tables).
constexpr int Cy(const int idc) {
  return (idc == 3 || idc == 5 || idc == 8) ? 1 : ((idc == 4 || idc == 6 || idc == 7) ? -1 : 0);
}

/**
 * @brief Everything a kernel needs besides the field pointers.
 *
 * Plain old data, built once on the host. The GPU backends copy it once into
 * device memory and hand the kernels a pointer: every thread then reads the
 * same addresses (broadcast loads served by L1), and no launch carries the
 * ~0.6 kB block as an argument or triggers an implicit transfer.
 */
template<typename Scalar>
struct CellParams {
  std::int32_t nx;
  std::int32_t ny;
  std::int32_t cells;

  Scalar ushiftX;
  Scalar ushiftY;
  Scalar cs2;
  Scalar cv;
  Scalar cp;
  Scalar invCv;
  Scalar rgas;
  Scalar viscosity;
  Scalar conductivity;

  Scalar densityL;
  Scalar pressureL;
  Scalar densityR;
  Scalar pressureR;

  DirectionStencil<Scalar, kQ> stencil;
};

/// Builds the kernel parameters (incl. the per-direction streaming stencil).
template<typename Scalar, typename LatticeType>
CellParams<Scalar> MakeCellParams(
  const FluidData<Scalar>& fluid,
  const MeshData<Scalar, 2>& mesh,
  const ControlData<Scalar>& control
) {
  static_assert(LatticeType::Speeds() == kQ && LatticeType::Dim() == kD, "D2Q9 only");
  CellParams<Scalar> p{};
  p.nx = static_cast<std::int32_t>(mesh.size[0]);
  p.ny = static_cast<std::int32_t>(mesh.size[1]);
  p.cells = static_cast<std::int32_t>(mesh.size[0] * mesh.size[1]);
  p.ushiftX = control.U(0);
  p.ushiftY = control.U(1);
  p.cs2 = LatticeType::Cs2;
  p.cv = fluid.specificHeatCv;
  p.cp = fluid.specificHeatCp;
  p.invCv = Scalar{ 1 } / fluid.specificHeatCv;
  p.rgas = fluid.constant;
  p.viscosity = fluid.viscosity;
  p.conductivity = fluid.conductivity;
  p.densityL = fluid.densityL;
  p.pressureL = fluid.pressureL;
  p.densityR = fluid.densityR;
  p.pressureR = fluid.pressureR;
  p.stencil = BuildDirectionStencil<LatticeType>(p.ny, control.U(0), control.U(1), control.idw);
  return p;
}

/**
 * @brief Register-level physics of one cell.
 *
 * `Weight` keeps the reference formula `w *= wZero + s * (wNonZero - wZero)`
 * with `s = c_ia^2` resolved at compile time.
 */
template<typename Scalar>
struct Cell {
  static constexpr Scalar kTiny = Scalar{ 1.0e-12 };
  static constexpr Scalar kMaxExp = Scalar{ 700.0 };
  static constexpr int kMaxIter = 3;

  /// Temperature-dependent D2Q9 weight `W_i(T)`.
  static inline Scalar Weight(const int idc, const Scalar tem) {
    const Scalar wZero = Scalar{ 1 } - tem;
    const Scalar wNonZero = Scalar{ 0.5 } * tem;
    const Scalar wx = (Cx(idc) != 0) ? wZero + (wNonZero - wZero) : wZero;
    const Scalar wy = (Cy(idc) != 0) ? wZero + (wNonZero - wZero) : wZero;
    return (Scalar{ 1 } * wx) * wy;
  }

  /// Density equilibrium, factorised product form.
  static inline void Feq(
    const Scalar rhoc,
    const Scalar ux,
    const Scalar uy,
    const Scalar temc,
    const CellParams<Scalar>& p,
    Scalar (&feq)[kQ]
  ) {
    const Scalar uiaX = ux - p.ushiftX;
    const Scalar uiaY = uy - p.ushiftY;
    const Scalar uia2tX = uiaX * uiaX + temc;
    const Scalar uia2tY = uiaY * uiaY + temc;
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      const int vx = Cx(idc);
      const int vy = Cy(idc);
      const Scalar pfx = (vx == 0) ? Scalar{ 1 } - uia2tX
        : ((vx == 1) ? Scalar{ 0.5 } * (uiaX + uia2tX) : Scalar{ 0.5 } * (-uiaX + uia2tX));
      const Scalar pfy = (vy == 0) ? Scalar{ 1 } - uia2tY
        : ((vy == 1) ? Scalar{ 0.5 } * (uiaY + uia2tY) : Scalar{ 0.5 } * (-uiaY + uia2tY));
      feq[idc] = (rhoc * pfx) * pfy;
    }
  }

  /// rho, u, T, p from one cell's populations.
  static inline void Macroscopic(
    const Scalar (&f)[kQ],
    const Scalar (&g)[kQ],
    const CellParams<Scalar>& p,
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
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      rhoSum += f[idc];
      nrg += g[idc];
      momX += (static_cast<Scalar>(Cx(idc)) + p.ushiftX) * f[idc];
      momY += (static_cast<Scalar>(Cy(idc)) + p.ushiftY) * f[idc];
    }
    const Scalar invRho = Scalar{ 1 } / rhoSum;
    ux = momX * invRho;
    uy = momY * invRho;
    const Scalar kin = Scalar{ 0.5 } * (ux * ux + uy * uy);
    tem = (Scalar{ 0.5 } * nrg * invRho - kin) * p.invCv;
    rho = rhoSum;
    pres = p.rgas * rhoSum * tem;
  }

  /**
   * @brief Equilibria + Newton-Raphson energy closure + BGK collision, in place.
   *
   * `f`/`g` enter as the pre-collision populations of the cell and leave as the
   * post-collision ones; `xi` is the Newton warm start (read and updated). The
   * Newton loop runs a fixed number of iterations with clamped reciprocals and
   * a sticky `solverOk` predicate, so control flow stays uniform across a warp
   * (or a SIMD lane group on the CPU).
   */
  static inline void Collide(
    const CellParams<Scalar>& p,
    const Scalar rhoc,
    const Scalar ux,
    const Scalar uy,
    const Scalar temc,
    Scalar (&f)[kQ],
    Scalar (&g)[kQ],
    Scalar (&xi)[kD]
  ) {
    const Scalar ushiftX = p.ushiftX;
    const Scalar ushiftY = p.ushiftY;
    const Scalar u2 = ux * ux + uy * uy;

    Scalar feqLocal[kQ];
    Feq(rhoc, ux, uy, temc, p, feqLocal);

    // Geq via Newton-Raphson on the normalised energy flux.
    const Scalar E = p.cv * temc + Scalar{ 0.5 } * u2;
    const Scalar targetE = Scalar{ 2 } * rhoc * E;
    const Scalar targetMx = Scalar{ 2 } * rhoc * ux * (E + temc) / targetE;
    const Scalar targetMy = Scalar{ 2 } * rhoc * uy * (E + temc) / targetE;

    Scalar alpha = Scalar{ 1 };
    Scalar si[kQ];
    Scalar e[kQ];
    Scalar Z = Scalar{ 0 };
    Scalar smax = Scalar{ 0 };
    bool solverOk = true;

    LBMINI_UNROLL(3)
    for (int iter = 0; iter < kMaxIter; ++iter) {
      smax = -kMaxExp;
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc) {
        Scalar s = Scalar{ 0 };
        s += xi[0] * (static_cast<Scalar>(Cx(idc)) + ushiftX);
        s += xi[1] * (static_cast<Scalar>(Cy(idc)) + ushiftY);
        si[idc] = s;
        smax = (s > smax) ? s : smax;
      }
      smax = (smax < kMaxExp) ? smax : kMaxExp;

      Scalar S1x = Scalar{ 0 };
      Scalar S1y = Scalar{ 0 };
      Scalar S2xx = Scalar{ 0 };
      Scalar S2xy = Scalar{ 0 };
      Scalar S2yx = Scalar{ 0 };
      Scalar S2yy = Scalar{ 0 };
      Z = Scalar{ 0 };
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc) {
        Scalar expo = si[idc] - smax;
        expo = (expo < kMaxExp) ? expo : kMaxExp;
        expo = (expo > -kMaxExp) ? expo : -kMaxExp;
        const Scalar wev = Weight(idc, temc) * std::exp(expo);
        e[idc] = wev;
        Z += wev;
        const Scalar cx = static_cast<Scalar>(Cx(idc)) + ushiftX;
        const Scalar cy = static_cast<Scalar>(Cy(idc)) + ushiftY;
        S1x += cx * wev;
        S2xx += cx * cx * wev;
        S2xy += cx * cy * wev;
        S1y += cy * wev;
        S2yx += cy * cx * wev;
        S2yy += cy * cy * wev;
      }

      const Scalar Zsafe = (Z > kTiny) ? Z : kTiny;
      solverOk = solverOk && (Z > kTiny);
      const Scalar invZ = Scalar{ 1 } / Zsafe;

      Scalar J00 = (S2xx - S1x * S1x * invZ) * invZ;
      const Scalar J01 = (S2xy - S1x * S1y * invZ) * invZ;
      const Scalar J10 = (S2yx - S1y * S1x * invZ) * invZ;
      Scalar J11 = (S2yy - S1y * S1y * invZ) * invZ;
      J00 += kTiny;
      J11 += kTiny;

      const Scalar detJ = J00 * J11 - J01 * J10;
      const Scalar absDet = std::fabs(detJ);
      const Scalar detSafe = (absDet > kTiny) ? detJ : ((detJ >= Scalar{ 0 }) ? kTiny : -kTiny);
      solverOk = solverOk && (absDet > kTiny);
      const Scalar invDet = Scalar{ 1 } / detSafe;

      const Scalar r0 = S1x * invZ - targetMx;
      const Scalar r1 = S1y * invZ - targetMy;
      xi[0] -= alpha * (J11 * r0 - J01 * r1) * invDet;
      xi[1] -= alpha * (-J10 * r0 + J00 * r1) * invDet;
      alpha *= Scalar{ 0.5 };
    }

    Scalar geqLocal[kQ];
    if (solverOk) {
      // Recompute e[], Z with the final xi.
      smax = -kMaxExp;
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc) {
        Scalar s = Scalar{ 0 };
        s += xi[0] * (static_cast<Scalar>(Cx(idc)) + ushiftX);
        s += xi[1] * (static_cast<Scalar>(Cy(idc)) + ushiftY);
        si[idc] = s;
        smax = (s > smax) ? s : smax;
      }
      smax = (smax < kMaxExp) ? smax : kMaxExp;
      Z = Scalar{ 0 };
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc) {
        Scalar expo = si[idc] - smax;
        expo = (expo < kMaxExp) ? expo : kMaxExp;
        expo = (expo > -kMaxExp) ? expo : -kMaxExp;
        e[idc] = Weight(idc, temc) * std::exp(expo);
        Z += e[idc];
      }
      const Scalar scale = targetE / Z;
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc)
        geqLocal[idc] = scale * e[idc];
      // The reference also stores `log(scale / rho) - smax` as warm-start
      // component 0; nothing ever reads it, so it is neither computed nor stored.
    } else {
      // Fallback: normalised-Wi distribution, cold restart of the warm start.
      xi[0] = Scalar{ 0 };
      xi[1] = Scalar{ 0 };
      Scalar sumW = Scalar{ 0 };
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc)
        sumW += Weight(idc, temc);
      if (sumW <= kTiny) {
        const Scalar uni = targetE / Scalar(kQ);
        LBMINI_UNROLL(9)
        for (int idc = 0; idc < kQ; ++idc)
          geqLocal[idc] = uni;
      } else {
        const Scalar sc = targetE / sumW;
        LBMINI_UNROLL(9)
        for (int idc = 0; idc < kQ; ++idc)
          geqLocal[idc] = Weight(idc, temc) * sc;
      }
    }

    // Collision (BGK + Knudsen sensor + thermal-relaxation correction).
    const Scalar tau = p.viscosity / (rhoc * temc) + Scalar{ 0.5 };
    const Scalar omega = Scalar{ 1 } / tau;
    const Scalar diffusivity = p.conductivity / (rhoc * p.cp);
    const Scalar tauT = diffusivity / temc + Scalar{ 0.5 };
    Scalar omegaT = Scalar{ 1 } / tauT;

    Scalar eps = Scalar{ 0 };
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar diff = f[idc] - feqLocal[idc];
      const Scalar den = (feqLocal[idc] > kTiny) ? feqLocal[idc] : kTiny;
      eps += std::fabs(diff) / den;
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
    Scalar Lx = Scalar{ 0 };
    Scalar Ly = Scalar{ 0 };
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar cix0 = static_cast<Scalar>(Cx(idc));
      const Scalar ciy0 = static_cast<Scalar>(Cy(idc));
      const Scalar uvi = ux * cix0 + uy * ciy0;
      const Scalar aux = Scalar{ 2 } * uvi * (f[idc] - feqLocal[idc]);
      Lx += aux * cix0;
      Ly += aux * ciy0;
    }

    const Scalar invT = Scalar{ 1 } / temc;
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar fOld = f[idc];
      f[idc] = fOld + omegaL * (feqLocal[idc] - fOld);
      const Scalar cidotL = Lx * (static_cast<Scalar>(Cx(idc)) + ushiftX)
        + Ly * (static_cast<Scalar>(Cy(idc)) + ushiftY);
      const Scalar Wi = Weight(idc, temc);
      const Scalar gDiff = Wi * cidotL * invT;
      const Scalar gOld = g[idc];
      g[idc] = gOld + omegaL * (geqLocal[idc] - gOld) + (omegaL - omegaT) * gDiff;
    }
  }

  /**
   * @brief Initial state of one cell (`cell = i * ny + j`).
   *
   * Sod left/right macroscopics, equilibrium seeding (`f = feq`, `g` = the
   * `W_i(T)`-normalised `geq` of a cold warm start), the macroscopic recovery
   * of the reference `Init()` (returned through `rho .. pres`) and the first
   * collision; `f`, `g`, `xi` leave as the post-collision state.
   */
  static inline void Init(
    const CellParams<Scalar>& p,
    const std::int32_t i,
    Scalar (&f)[kQ],
    Scalar (&g)[kQ],
    Scalar (&xi)[kD],
    Scalar& rho,
    Scalar& ux,
    Scalar& uy,
    Scalar& tem,
    Scalar& pres
  ) {
    const bool left = i < p.nx / 2;
    const Scalar rho0 = left ? p.densityL : p.densityR;
    const Scalar p0 = left ? p.pressureL : p.pressureR;
    const Scalar tem0 = p.cs2 * p0 / (rho0 * p.rgas);
    const Scalar ux0 = Scalar{ 0 };
    const Scalar uy0 = Scalar{ 0 };

    Feq(rho0, ux0, uy0, tem0, p, f);
    const Scalar u2 = ux0 * ux0 + uy0 * uy0;
    const Scalar E = p.cv * tem0 + Scalar{ 0.5 } * u2;
    const Scalar targetE = Scalar{ 2 } * rho0 * E;
    Scalar sumW = Scalar{ 0 };
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc)
      sumW += Weight(idc, tem0);
    const Scalar sc = (sumW > kTiny) ? targetE / sumW : targetE / Scalar(kQ);
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar wi = (sumW > kTiny) ? Weight(idc, tem0) : Scalar{ 1 };
      g[idc] = wi * sc;
    }

    Macroscopic(f, g, p, rho, ux, uy, tem, pres);

    xi[0] = Scalar{ 0 };
    xi[1] = Scalar{ 0 };
    Collide(p, rho, ux, uy, tem, f, g, xi);
  }

  /**
   * @brief Pull streaming of cell `(i, j)`: the 9 post-collision populations of
   *        the previous step, interpolated from their 4 IDW source cells.
   *
   * Source columns are clamped to `[0, nx)`, rows wrap periodically; the
   * `nearest` fallback (uniform branch, it depends on the direction only)
   * copies the nearest node, or the cell itself outside the tube — exactly
   * `lbmini::GatherSources`.
   */
  static inline void Gather(
    const CellParams<Scalar>& p,
    const Scalar* __restrict__ fIn,
    const Scalar* __restrict__ gIn,
    const std::int32_t i,
    const std::int32_t j,
    Scalar (&f)[kQ],
    Scalar (&g)[kQ]
  ) {
    const std::int32_t nx = p.nx;
    const std::int32_t ny = p.ny;
    const std::int32_t N = p.cells;
    const std::int32_t cell = i * ny + j;
    const DirectionStencil<Scalar, kQ>& st = p.stencil;
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar* __restrict__ fp = fIn + static_cast<std::int64_t>(idc) * N;
      const Scalar* __restrict__ gp = gIn + static_cast<std::int64_t>(idc) * N;
      if (st.nearest[idc]) {
        const auto xN = static_cast<std::int32_t>(std::round(static_cast<Scalar>(i) - st.cix[idc]));
        const auto yN = static_cast<std::int32_t>(std::round(static_cast<Scalar>(j) - st.ciy[idc]));
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
  }
};
} // namespace lbmini::directive

#endif // LBMINI_OPENACC_CELLPHYSICS_HPP_
