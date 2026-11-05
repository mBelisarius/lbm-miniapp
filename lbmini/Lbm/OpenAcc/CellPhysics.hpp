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
 * arrays stay in registers. The functions are the per-cell physics of
 * `lbmini::cuda::gpu` (`Physics` in `Cuda/Gpu/LbmTube.cu`), including its
 * factorised energy closure (`AxisExp`): equal to the arithmetic of the plain
 * reference solver to rounding error, not bit for bit.
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
  Scalar invCv;
  Scalar rgas;
  Scalar viscosity;
  Scalar condOverCp; ///< `conductivity / cp`: the thermal diffusivity times `rho`

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
  p.invCv = Scalar{ 1 } / fluid.specificHeatCv;
  p.rgas = fluid.constant;
  p.viscosity = fluid.viscosity;
  p.condOverCp = fluid.conductivity / fluid.specificHeatCp;
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
 * The D2Q9 weights `W_q = Wx(c_x) Wy(c_y)` are tensor products, see `AxisExp`.
 */
template<typename Scalar>
struct Cell {
  static constexpr Scalar kTiny = Scalar{ 1.0e-12 };
  static constexpr Scalar kMaxExp = Scalar{ 700.0 };
  static constexpr int kMaxIter = 3;

  /**
   * @brief Temperature-dependent D2Q9 weights `W_q(T)` of all directions.
   *
   * The reference evaluates `w = 1; w *= wZero + isNonZero * (wNonZero - wZero)`
   * per dimension, with `isNonZero` in {0, 1}: the factor is exactly `wZero` for
   * a zero component and `wn = wZero + (wNonZero - wZero)` otherwise.
   */
  static inline void Weights(const Scalar wZero, const Scalar wn, Scalar (&w)[kQ]) {
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc)
      w[idc] = (Cx(idc) != 0 ? wn : wZero) * (Cy(idc) != 0 ? wn : wZero);
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

  /// rho, 1/rho, u, T, p from one cell's populations.
  static inline void Macroscopic(
    const Scalar (&f)[kQ],
    const Scalar (&g)[kQ],
    const CellParams<Scalar>& p,
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
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      rhoSum += f[idc];
      nrg += g[idc];
      momX += (static_cast<Scalar>(Cx(idc)) + p.ushiftX) * f[idc];
      momY += (static_cast<Scalar>(Cy(idc)) + p.ushiftY) * f[idc];
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
   * Double precision is the bottleneck on consumer GPUs (1/64 of the single
   * precision rate), so the number of FP64 instructions per cell is the cost.
   * The D2Q9 weights and the exponents of the energy equilibrium are tensor
   * products, `W_q = Wx(c_x) Wy(c_y)` and `s_q = xi_0 (c_x + Ux) + xi_1 (c_y + Uy)`,
   * hence `W_q exp(s_q - smax) = Px(c_x) Py(c_y)` (the `U` shift and the common
   * maximum cancel), with a single exponential per axis, `E = exp(-|xi|)`: the
   * exponents `xi c - max` are `0`, `-|xi|` and `-2 |xi|` in the order of the
   * sign of `xi`, i.e. `1`, `E` and `E^2`. That is 2 exponentials per evaluation
   * (8 per cell and step; the per-direction form of the reference evaluates 9 per
   * evaluation, 36 per cell and step) and it turns the nine-term sums of the
   * Newton step into three-term ones.
   *
   * It equals the reference to rounding error while neither of its clamps
   * (`|s - smax| <= 700`) is active; beyond that every omitted term is below
   * `exp(-700)` relative to the largest one.
   */
  static inline void AxisExp(const Scalar xi, const Scalar wn, const Scalar wZero, Scalar (&p)[3]) {
    const Scalar t = -std::fabs(xi);
    const Scalar e = std::exp((t > -kMaxExp) ? t : -kMaxExp);
    const Scalar e2 = e * e;
    const bool pos = xi >= Scalar{ 0 };
    p[0] = wn * (pos ? e2 : Scalar{ 1 });
    p[1] = wZero * e;
    p[2] = wn * (pos ? Scalar{ 1 } : e2);
  }

  /// Equilibrium energy populations from the Newton solution `xi`.
  static inline void GeqFromXi(
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
    LBMINI_UNROLL(3)
    for (int k = 0; k < 3; ++k)
      sx[k] = scale * px[k];
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc)
      geq[idc] = sx[Cx(idc) + 1] * py[Cy(idc) + 1];
  }

  /**
   * @brief Equilibria + Newton-Raphson energy closure + BGK collision, in place.
   *
   * `f`/`g` enter as the pre-collision populations of the cell and leave as the
   * post-collision ones; `xi` is the Newton warm start (read and updated).
   * The Newton loop runs a fixed number of iterations with clamped reciprocals
   * and a sticky `solverOk` predicate, so control flow stays uniform across a
   * warp except in the (rare) degenerate fallback.
   */
  static inline void Collide(
    const CellParams<Scalar>& p,
    const Scalar rhoc,
    const Scalar invRho,
    const Scalar ux,
    const Scalar uy,
    const Scalar temc,
    Scalar (&f)[kQ],
    Scalar (&g)[kQ],
    Scalar (&xi)[kD]
  ) {
    const Scalar u2 = ux * ux + uy * uy;

    Scalar feqLocal[kQ];
    Feq(rhoc, ux, uy, temc, p, feqLocal);

    // The per-axis factors of the weights (`W_q = Wx(c_x) Wy(c_y)`), see `AxisExp`.
    const Scalar wZero = Scalar{ 1 } - temc;
    const Scalar wn = wZero + (Scalar{ 0.5 } * temc - wZero);
    Scalar wi[kQ];
    Weights(wZero, wn, wi);

    // Geq via Newton-Raphson on the normalised energy flux.
    //
    // The closure distribution `W_q exp(xi . (c_q + U))` is a product of an x and a
    // y factor, so x and y are independent: the Jacobian of the reference is
    // diagonal (its off-diagonal term is exactly `S2xy / Z - (S1x / Z) (S1y / Z) = 0`)
    // and the iteration is two scalar Newton steps,
    // `xi_a -= alpha (m1_a - target_a) / var_a`, each from three weighted
    // exponentials of one axis (see `AxisExp`). Divisions are expensive on
    // a GPU, so the two axes share one reciprocal per stage.
    const Scalar E = p.cv * temc + Scalar{ 0.5 } * u2;
    const Scalar targetE = Scalar{ 2 } * rhoc * E;
    const Scalar invE = Scalar{ 1 } / E;
    const Scalar targetMx = ux * (E + temc) * invE; // 2 rho u (E + T) / (2 rho E)
    const Scalar targetMy = uy * (E + temc) * invE;

    Scalar alpha = Scalar{ 1 };
    bool solverOk = true;

    LBMINI_UNROLL(3)
    for (int iter = 0; iter < kMaxIter; ++iter) {
      Scalar px[3];
      Scalar py[3];
      AxisExp(xi[0], wn, wZero, px);
      AxisExp(xi[1], wn, wZero, py);
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
      solverOk = solverOk && (std::fabs(jxy) > kTiny);

      const Scalar rj = alpha * (Scalar{ 1 } / jxy);
      const Scalar r0 = (m1x + p.ushiftX) - targetMx;
      const Scalar r1 = (m1y + p.ushiftY) - targetMy;
      xi[0] -= r0 * jyy * rj; // alpha r0 / jxx
      xi[1] -= r1 * jxx * rj; // alpha r1 / jyy
      alpha *= Scalar{ 0.5 };
    }

    Scalar geqLocal[kQ];
    if (solverOk) {
      GeqFromXi(xi[0], xi[1], wn, wZero, targetE, geqLocal);
      // The reference also stores `log(scale / rho) - smax` as warm-start
      // component 0; nothing ever reads it, so it is neither computed nor stored.
    } else {
      // Fallback: normalised-Wi distribution, cold restart of the warm start.
      xi[0] = Scalar{ 0 };
      xi[1] = Scalar{ 0 };
      Scalar sumW = Scalar{ 0 };
      LBMINI_UNROLL(9)
      for (int idc = 0; idc < kQ; ++idc)
        sumW += wi[idc];
      if (sumW <= kTiny) {
        const Scalar uni = targetE / Scalar(kQ);
        LBMINI_UNROLL(9)
        for (int idc = 0; idc < kQ; ++idc)
          geqLocal[idc] = uni;
      } else {
        const Scalar sc = targetE / sumW;
        LBMINI_UNROLL(9)
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
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar diff = f[idc] - feqLocal[idc];
      const Scalar den = (feqLocal[idc] > kTiny) ? feqLocal[idc] : kTiny;
      eps += std::fabs(diff) / den;
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

    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar fOld = f[idc];
      f[idc] = fOld + omegaL * (feqLocal[idc] - fOld);
      const Scalar cidotL = Lx * (static_cast<Scalar>(Cx(idc)) + p.ushiftX)
        + Ly * (static_cast<Scalar>(Cy(idc)) + p.ushiftY);
      const Scalar gDiff = wi[idc] * cidotL * invT;
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
    const Scalar wZero = Scalar{ 1 } - tem0;
    const Scalar wn = wZero + (Scalar{ 0.5 } * tem0 - wZero);
    Scalar w0[kQ];
    Weights(wZero, wn, w0);
    Scalar sumW = Scalar{ 0 };
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc)
      sumW += w0[idc];
    const Scalar sc = (sumW > kTiny) ? targetE / sumW : targetE / Scalar(kQ);
    LBMINI_UNROLL(9)
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar wi = (sumW > kTiny) ? w0[idc] : Scalar{ 1 };
      g[idc] = wi * sc;
    }

    Scalar invRho;
    Macroscopic(f, g, p, rho, invRho, ux, uy, tem, pres);

    xi[0] = Scalar{ 0 };
    xi[1] = Scalar{ 0 };
    Collide(p, rho, invRho, ux, uy, tem, f, g, xi);
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
