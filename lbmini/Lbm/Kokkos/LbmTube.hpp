#ifndef LBMINI_KOKKOS_LBMTUBE_HPP_
#define LBMINI_KOKKOS_LBMTUBE_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include <Kokkos_Core.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/Kokkos/Runtime.hpp"
#include "Lbm/Stencil.hpp"

/// Unroll a short, fixed-trip loop (directions / dimensions). Fully unrolled
/// loops turn every lattice lookup into an immediate and keep the per-cell
/// arrays in registers instead of GPU local memory.
/// `LBMINI_KOKKOS_NO_UNROLL` keeps a loop rolled (the Newton iterations: one copy of the
/// body instead of three, which is what the instruction cache of a GPU wants).
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__clang__)
#define LBMINI_KOKKOS_UNROLL _Pragma("unroll")
#define LBMINI_KOKKOS_NO_UNROLL _Pragma("unroll 1")
#elif defined(__GNUC__)
#define LBMINI_KOKKOS_UNROLL _Pragma("GCC unroll 16")
#define LBMINI_KOKKOS_NO_UNROLL _Pragma("GCC unroll 1")
#else
#define LBMINI_KOKKOS_UNROLL
#define LBMINI_KOKKOS_NO_UNROLL
#endif

namespace lbmini::kokkos {

/// Plain Old Data bundle of simulation parameters and the shared gather stencil,
/// trivially copyable to device constant/parameter memory.
template<typename Scalar, int Q>
struct Params {
  std::int32_t nx;
  std::int32_t ny;
  std::int32_t cells;

  Scalar ushiftX;
  Scalar ushiftY;

  Scalar cv;
  Scalar invCv;
  Scalar rgas;
  Scalar viscosity;
  Scalar condOverCp; ///< `conductivity / cp`: `kappa / (rho cp) = condOverCp / rho`

  Scalar densityL;
  Scalar pressureL;
  Scalar densityR;
  Scalar pressureR;

  DirectionStencil<Scalar, Q> stencil;
};

/**
 * @brief Per-cell LBM physics, shared by Kokkos kernels.
 *
 * Every function operates on register arrays of one cell. The arithmetic is that of
 * the reference solver except for the energy closure, which is evaluated in the
 * factorised form of `AxisExp`; it agrees with the reference to rounding error.
 */
template<typename Scalar, typename LatticeType>
struct Physics {
  static constexpr int kQ = static_cast<int>(LatticeType::Speeds());
  static constexpr int kD = static_cast<int>(LatticeType::Dim());
  static constexpr Scalar kTiny = Scalar{ 1.0e-12 };
  static constexpr Scalar kMaxExp = Scalar{ 700.0 };
  static constexpr int kMaxIter = 3;

  static_assert(kD == 2 && kQ == 9, "the Kokkos backend implements the D2Q9 closure");

  KOKKOS_INLINE_FUNCTION static constexpr int Vx(const int idc) {
    return static_cast<int>(LatticeType::Velocity(idc, 0));
  }

  KOKKOS_INLINE_FUNCTION static constexpr int Vy(const int idc) {
    return static_cast<int>(LatticeType::Velocity(idc, 1));
  }

  /// Density equilibrium, factorised product form.
  KOKKOS_INLINE_FUNCTION static void Feq(
    const Scalar rhoc,
    const Scalar ux,
    const Scalar uy,
    const Scalar temc,
    const Scalar ushiftX,
    const Scalar ushiftY,
    Scalar (&feq)[kQ]
  ) {
    LBMINI_KOKKOS_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      Scalar phi = rhoc;
      LBMINI_KOKKOS_UNROLL
      for (int d = 0; d < kD; ++d) {
        const auto vi = LatticeType::Velocity(idc, d);
        const Scalar uia = (d == 0 ? ux : uy) - (d == 0 ? ushiftX : ushiftY);
        const Scalar uia2t = uia * uia + temc;
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

  /// rho, 1 / rho, u, T, p from one cell's populations.
  KOKKOS_INLINE_FUNCTION static void Macroscopic(
    const Scalar (&f)[kQ],
    const Scalar (&g)[kQ],
    const Params<Scalar, kQ>& p,
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
    LBMINI_KOKKOS_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      rhoSum += f[idc];
      nrg += g[idc];
      momX += LatticeType::Cshift(idc, 0, p.ushiftX) * f[idc];
      momY += LatticeType::Cshift(idc, 1, p.ushiftY) * f[idc];
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
  KOKKOS_INLINE_FUNCTION static void AxisExp(
    const Scalar xi,
    const Scalar wn,
    const Scalar wZero,
    Scalar (&p)[3]
  ) {
    const Scalar t = -Kokkos::abs(xi);
    const Scalar e = Kokkos::exp((t > -kMaxExp) ? t : -kMaxExp);
    const Scalar e2 = e * e;
    const bool pos = xi >= Scalar{ 0 };
    p[0] = wn * (pos ? e2 : Scalar{ 1 });
    p[1] = wZero * e;
    p[2] = wn * (pos ? Scalar{ 1 } : e2);
  }

  /// Equilibrium energy populations from the Newton solution `xi`.
  KOKKOS_INLINE_FUNCTION static void GeqFromXi(
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
    LBMINI_KOKKOS_UNROLL
    for (int k = 0; k < 3; ++k)
      sx[k] = scale * px[k];
    LBMINI_KOKKOS_UNROLL
    for (int idc = 0; idc < kQ; ++idc)
      geq[idc] = sx[Vx(idc) + 1] * py[Vy(idc) + 1];
  }

  /**
   * @brief Equilibria + Newton-Raphson energy closure + BGK collision, in place.
   *
   * `f`/`g` enter as the pre-collision populations of the cell and leave as the
   * post-collision ones; `xi` is the Newton warm start (read and updated). The
   * Newton loop runs a fixed number of iterations with clamped reciprocals and
   * a sticky `solverOk` predicate: control flow stays uniform across a warp.
   */
  KOKKOS_INLINE_FUNCTION static void Collide(
    const Params<Scalar, kQ>& p,
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
    Feq(rhoc, ux, uy, temc, p.ushiftX, p.ushiftY, feqLocal);

    // The per-axis factors of the weights (`W_q = Wx(c_x) Wy(c_y)`, see `AxisExp`); the
    // factor is exactly the one `LatticeType::Weights` evaluates for each dimension.
    const Scalar wZero = Scalar{ 1 } - temc;
    const Scalar wn = wZero + (Scalar{ 0.5 } * temc - wZero);
    Scalar wi[kQ];
    LBMINI_KOKKOS_UNROLL
    for (int idc = 0; idc < kQ; ++idc)
      wi[idc] = (Vx(idc) == 0 ? wZero : wn) * (Vy(idc) == 0 ? wZero : wn);

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

    LBMINI_KOKKOS_NO_UNROLL
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
      solverOk = solverOk && (Kokkos::abs(jxy) > kTiny);

      const Scalar rj = alpha * (Scalar{ 1 } / jxy);
      const Scalar r0 = (m1x + p.ushiftX) - targetM0;
      const Scalar r1 = (m1y + p.ushiftY) - targetM1;
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
      LBMINI_KOKKOS_UNROLL
      for (int idc = 0; idc < kQ; ++idc)
        sumW += wi[idc];
      if (sumW <= kTiny) {
        const Scalar uni = targetE / Scalar(kQ);
        LBMINI_KOKKOS_UNROLL
        for (int idc = 0; idc < kQ; ++idc)
          geqLocal[idc] = uni;
      } else {
        const Scalar sc = targetE / sumW;
        LBMINI_KOKKOS_UNROLL
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
    LBMINI_KOKKOS_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar diff = f[idc] - feqLocal[idc];
      const Scalar den = (feqLocal[idc] > kTiny) ? feqLocal[idc] : kTiny;
      eps += Kokkos::abs(diff) / den;
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
    Scalar L[kD] = { Scalar{ 0 }, Scalar{ 0 } };
    LBMINI_KOKKOS_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar cix0 = static_cast<Scalar>(Vx(idc));
      const Scalar ciy0 = static_cast<Scalar>(Vy(idc));
      const Scalar uvi = ux * cix0 + uy * ciy0;
      const Scalar aux = Scalar{ 2 } * uvi * (f[idc] - feqLocal[idc]);
      L[0] += aux * cix0;
      L[1] += aux * ciy0;
    }

    LBMINI_KOKKOS_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar fOld = f[idc];
      f[idc] = fOld + omegaL * (feqLocal[idc] - fOld);
      const Scalar cidotL = L[0] * LatticeType::Cshift(idc, 0, p.ushiftX)
        + L[1] * LatticeType::Cshift(idc, 1, p.ushiftY);
      const Scalar gDiff = wi[idc] * cidotL * invT;
      const Scalar gOld = g[idc];
      g[idc] = gOld + omegaL * (geqLocal[idc] - gOld) + (omegaL - omegaT) * gDiff;
    }
  }
};

/**
 * @brief Functor for initial state: Sod left/right macroscopics, equilibrium seeding,
 *        macroscopic recovery of the reference Init(), and the first collision.
 */
template<typename Scalar, typename LatticeType, typename MemorySpace>
struct InitFunctor {
  using Index = Eigen::Index;
  using Phys = Physics<Scalar, LatticeType>;
  static constexpr int kQ = Phys::kQ;
  static constexpr int kD = Phys::kD;

  Params<Scalar, kQ> p;
  Kokkos::View<Scalar*, MemorySpace> fOut;
  Kokkos::View<Scalar*, MemorySpace> gOut;
  Kokkos::View<Scalar*, MemorySpace> xiOut;
  Kokkos::View<Scalar*, MemorySpace> rhoOut;
  Kokkos::View<Scalar*, MemorySpace> pOut;
  Kokkos::View<Scalar*, MemorySpace> temOut;
  Kokkos::View<Scalar*, MemorySpace> uOut;

  KOKKOS_INLINE_FUNCTION void operator()(const Index cell) const {
    const std::int32_t i = static_cast<std::int32_t>(cell / p.ny);
    const Index N = p.cells;

    // Sod-like left/right state (reference Init() host loop).
    const bool left = (i < p.nx / 2);
    const Scalar rho0 = left ? p.densityL : p.densityR;
    const Scalar p0 = left ? p.pressureL : p.pressureR;
    const Scalar tem0 = LatticeType::Cs2 * p0 / (rho0 * p.rgas);
    const Scalar ux0 = Scalar{ 0 };
    const Scalar uy0 = Scalar{ 0 };

    // seedEquilibria(): f = feq, g = Wi(T)-normalised geq (xi == 0).
    Scalar f[kQ];
    Scalar g[kQ];
    Phys::Feq(rho0, ux0, uy0, tem0, p.ushiftX, p.ushiftY, f);
    const Scalar u2 = ux0 * ux0 + uy0 * uy0;
    const Scalar E = p.cv * tem0 + Scalar{ 0.5 } * u2;
    const Scalar targetE = Scalar{ 2 } * rho0 * E;
    Scalar sumW = Scalar{ 0 };
    LBMINI_KOKKOS_UNROLL
    for (int idc = 0; idc < kQ; ++idc)
      sumW += LatticeType::Weights(idc, tem0);
    const Scalar sc = (sumW > Phys::kTiny) ? targetE / sumW : targetE / Scalar(kQ);
    LBMINI_KOKKOS_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar wi = (sumW > Phys::kTiny) ? LatticeType::Weights(idc, tem0) : Scalar{ 1 };
      g[idc] = wi * sc;
    }

    // computeMacroscopic(): the state every getter sees after Init().
    Scalar rho;
    Scalar invRho;
    Scalar ux;
    Scalar uy;
    Scalar tem;
    Scalar pres;
    Phys::Macroscopic(f, g, p, rho, invRho, ux, uy, tem, pres);
    rhoOut(cell) = rho;
    pOut(cell) = pres;
    temOut(cell) = tem;
    uOut(0 * N + cell) = ux;
    uOut(1 * N + cell) = uy;

    // First collision, cold Newton start.
    Scalar xi[kD] = { Scalar{ 0 }, Scalar{ 0 } };
    Phys::Collide(p, rho, invRho, ux, uy, tem, f, g, xi);

    LBMINI_KOKKOS_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      fOut(idc * N + cell) = f[idc];
      gOut(idc * N + cell) = g[idc];
    }
    LBMINI_KOKKOS_UNROLL
    for (int d = 0; d < kD; ++d)
      xiOut(d * N + cell) = xi[d];
  }
};

/**
 * @brief Functor for one full time step: pull streaming + macroscopics + collision.
 *
 * @tparam kStoreMacro write rho, p, T, u to global memory (only needed on the final step
 *         of a Run() chunk).
 */
template<typename Scalar, typename LatticeType, typename MemorySpace, bool kStoreMacro>
struct StepFunctor {
  using Index = Eigen::Index;
  using Phys = Physics<Scalar, LatticeType>;
  static constexpr int kQ = Phys::kQ;
  static constexpr int kD = Phys::kD;

  Params<Scalar, kQ> p;
  Kokkos::View<Scalar*, MemorySpace> fIn;
  Kokkos::View<Scalar*, MemorySpace> gIn;
  Kokkos::View<Scalar*, MemorySpace> fOut;
  Kokkos::View<Scalar*, MemorySpace> gOut;
  Kokkos::View<Scalar*, MemorySpace> xiIo;
  Kokkos::View<Scalar*, MemorySpace> rhoOut;
  Kokkos::View<Scalar*, MemorySpace> pOut;
  Kokkos::View<Scalar*, MemorySpace> temOut;
  Kokkos::View<Scalar*, MemorySpace> uOut;

  KOKKOS_INLINE_FUNCTION void operator()(const Index cell) const {
    const Index N = p.cells;
    const std::int32_t nx = p.nx;
    const std::int32_t ny = p.ny;
    const auto& st = p.stencil;

    const std::int32_t i = static_cast<std::int32_t>(cell / ny);
    const std::int32_t j = static_cast<std::int32_t>(cell - i * ny);

    // Pull the post-collision populations of the previous step.
    Scalar f[kQ];
    Scalar g[kQ];
    LBMINI_KOKKOS_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      const Index plane = idc * N;
      if (st.nearest[idc]) {
        const auto xN = static_cast<std::int32_t>(
          Kokkos::round(static_cast<Scalar>(i) - st.cix[idc]));
        const auto yN = static_cast<std::int32_t>(
          Kokkos::round(static_cast<Scalar>(j) - st.ciy[idc]));
        std::int32_t yW = yN % ny;
        yW += (yW < 0) ? ny : 0;
        const std::int32_t src = (xN < 0 || xN >= nx) ? static_cast<std::int32_t>(cell) : xN * ny + yW;
        f[idc] = fIn(plane + src);
        g[idc] = gIn(plane + src);
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
        f[idc] = w0 * fIn(plane + rowA + ya) + w1 * fIn(plane + rowB + ya)
               + w2 * fIn(plane + rowA + yb) + w3 * fIn(plane + rowB + yb);
        g[idc] = w0 * gIn(plane + rowA + ya) + w1 * gIn(plane + rowB + ya)
               + w2 * gIn(plane + rowA + yb) + w3 * gIn(plane + rowB + yb);
      }
    }

    Scalar rho;
    Scalar invRho;
    Scalar ux;
    Scalar uy;
    Scalar tem;
    Scalar pres;
    Phys::Macroscopic(f, g, p, rho, invRho, ux, uy, tem, pres);
    if constexpr (kStoreMacro) {
      rhoOut(cell) = rho;
      pOut(cell) = pres;
      temOut(cell) = tem;
      uOut(0 * N + cell) = ux;
      uOut(1 * N + cell) = uy;
    }

    Scalar xi[kD];
    LBMINI_KOKKOS_UNROLL
    for (int d = 0; d < kD; ++d)
      xi[d] = xiIo(d * N + cell);

    Phys::Collide(p, rho, invRho, ux, uy, tem, f, g, xi);

    LBMINI_KOKKOS_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      fOut(idc * N + cell) = f[idc];
      gOut(idc * N + cell) = g[idc];
    }
    LBMINI_KOKKOS_UNROLL
    for (int d = 0; d < kD; ++d)
      xiIo(d * N + cell) = xi[d];
  }
};

/**
 * @brief Modernised Kokkos variant of the compressible LBM tube solver.
 *
 * Implements single-pass pull-stream-and-collide execution matching the Alpaka backend:
 *  - DirectionStencil eliminates 432 B/cell/step streaming table lookups.
 *  - Ping-pong double-buffered distribution views (fA/fB, gA/gB) eliminate intermediate
 *    buffers (faux/gaux).
 *  - 2-component warm start xi (2 * N) reduces memory traffic.
 *  - Lazy macroscopic writeback (kStoreMacro) skips intermediate global writes.
 *  - Single-source execution across Host and Device execution spaces.
 */
template<typename Scalar, typename LatticeType, typename ExecSpace = Kokkos::DefaultExecutionSpace>
class LbmTube : public ILbmTube<Scalar, LatticeType> {
public:
  using Index = Eigen::Index;

  template<typename Type, Index NumIndices>
  using Tensor = Eigen::Tensor<Type, NumIndices, Eigen::RowMajor>;

  static constexpr Index kDim_ = LatticeType::Dim();
  static constexpr Index kQ_ = LatticeType::Speeds();

  using ExecutionSpace = ExecSpace;
  using MemorySpace = typename ExecSpace::memory_space;
  using ScalarView = Kokkos::View<Scalar*, MemorySpace>;
  using ScalarHostView = typename ScalarView::host_mirror_type;
  // sm_89 (RTX 3500 Ada): <128,2> compiles the step kernel to 160 registers without spills,
  // 3 resident blocks per SM; <128,4> caps it at 128 registers and spills (56 B stack, 17 local
  // loads/11 stores) and measures slower. The kernel is FP64-compute-bound (~690 FP64
  // instructions per cell), so prefer no spills.
  using RangePolicy = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<Index>, Kokkos::LaunchBounds<128, 2>>;

  LbmTube(
    const FluidData<Scalar>& fluid,
    const MeshData<Scalar, kDim_>& mesh,
    const ControlData<Scalar>& control,
    const PerformanceData& performance
  );

  LbmTube(const LbmTube&) = delete;
  LbmTube& operator=(const LbmTube&) = delete;
  LbmTube(LbmTube&&) = delete;
  LbmTube& operator=(LbmTube&&) = delete;

  [[nodiscard]] Tensor<Scalar, kDim_> P() const override;
  [[nodiscard]] Tensor<Scalar, kDim_> Rho() const override;
  [[nodiscard]] Tensor<Scalar, kDim_> T() const override;
  [[nodiscard]] Tensor<Scalar, kDim_ + 1> U() const override;

  void Init() override;
  void Step(bool save) override;
  void Run(Index steps, bool save) override;

  [[nodiscard]] Index cellIndex(const Index i, const Index j) const { return i * ny_ + j; }
  [[nodiscard]] Index distIndex(const Index idc, const Index cell) const { return idc * N_ + cell; }
  [[nodiscard]] Index uIndex(const Index d, const Index cell) const { return d * N_ + cell; }

  template<bool kStoreMacro>
  void step();

private:
  Runtime runtime_;

  const FluidData<Scalar> kFluid_;
  const MeshData<Scalar, kDim_> kMesh_;
  const ControlData<Scalar> kControl_;
  const PerformanceData kPerformance_;

  Index nx_;
  Index ny_;
  Index N_;        // nx_ * ny_
  Index uSize_;    // N_ * Dim
  Index distSize_; // N_ * Q

  Params<Scalar, kQ_> params_;

  // Device-resident flat SoA storage.
  ScalarView rho_; // size N_
  ScalarView p_;   // size N_
  ScalarView tem_; // size N_
  ScalarView u_;   // size uSize_, d-major
  ScalarView fA_;  // size distSize_, idc-major (ping-pong current)
  ScalarView fB_;  // size distSize_, idc-major (ping-pong next)
  ScalarView gA_;  // size distSize_, idc-major (ping-pong current)
  ScalarView gB_;  // size distSize_, idc-major (ping-pong next)
  ScalarView xi_;  // size N_ * kDim_, d-major (Newton warm-start)

  // Persistent host mirrors used by the copy-out getters.
  mutable ScalarHostView rhoHost_;
  mutable ScalarHostView pHost_;
  mutable ScalarHostView temHost_;
  mutable ScalarHostView uHost_;
};

template<typename Scalar, typename LatticeType, typename ExecSpace>
LbmTube<Scalar, LatticeType, ExecSpace>::LbmTube(
  const FluidData<Scalar>& fluid,
  const MeshData<Scalar, kDim_>& mesh,
  const ControlData<Scalar>& control,
  const PerformanceData& performance
)
  : runtime_(static_cast<int>(performance.cores)),
    kFluid_(fluid), kMesh_(mesh), kControl_(control), kPerformance_(performance) {
  ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "kokkos::LbmTube");

  nx_ = mesh.size[0];
  ny_ = mesh.size[1];
  N_ = nx_ * ny_;
  uSize_ = N_ * kDim_;
  distSize_ = N_ * kQ_;

  params_.nx = static_cast<std::int32_t>(nx_);
  params_.ny = static_cast<std::int32_t>(ny_);
  params_.cells = static_cast<std::int32_t>(N_);
  params_.ushiftX = control.U(0);
  params_.ushiftY = control.U(1);
  params_.cv = fluid.specificHeatCv;
  params_.invCv = Scalar{ 1 } / fluid.specificHeatCv;
  params_.rgas = fluid.constant;
  params_.viscosity = fluid.viscosity;
  params_.condOverCp = fluid.conductivity / fluid.specificHeatCp;
  params_.densityL = fluid.densityL;
  params_.pressureL = fluid.pressureL;
  params_.densityR = fluid.densityR;
  params_.pressureR = fluid.pressureR;
  params_.stencil = BuildDirectionStencil<LatticeType>(ny_, control.U(0), control.U(1), control.idw);

  rho_ = ScalarView("lbmini::rho", N_);
  p_ = ScalarView("lbmini::p", N_);
  tem_ = ScalarView("lbmini::tem", N_);
  u_ = ScalarView("lbmini::u", uSize_);

  fA_ = ScalarView("lbmini::fA", distSize_);
  fB_ = ScalarView("lbmini::fB", distSize_);
  gA_ = ScalarView("lbmini::gA", distSize_);
  gB_ = ScalarView("lbmini::gB", distSize_);

  xi_ = ScalarView("lbmini::xi", N_ * kDim_);

  rhoHost_ = Kokkos::create_mirror_view(rho_);
  pHost_ = Kokkos::create_mirror_view(p_);
  temHost_ = Kokkos::create_mirror_view(tem_);
  uHost_ = Kokkos::create_mirror_view(u_);
}

template<typename Scalar, typename LatticeType, typename ExecSpace>
auto LbmTube<Scalar, LatticeType, ExecSpace>::P() const -> Tensor<Scalar, kDim_> {
  Kokkos::deep_copy(pHost_, p_);
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(pHost_.data(), pHost_.data() + N_, out.data());
  return out;
}

template<typename Scalar, typename LatticeType, typename ExecSpace>
auto LbmTube<Scalar, LatticeType, ExecSpace>::Rho() const -> Tensor<Scalar, kDim_> {
  Kokkos::deep_copy(rhoHost_, rho_);
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(rhoHost_.data(), rhoHost_.data() + N_, out.data());
  return out;
}

template<typename Scalar, typename LatticeType, typename ExecSpace>
auto LbmTube<Scalar, LatticeType, ExecSpace>::T() const -> Tensor<Scalar, kDim_> {
  Kokkos::deep_copy(temHost_, tem_);
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(temHost_.data(), temHost_.data() + N_, out.data());
  return out;
}

template<typename Scalar, typename LatticeType, typename ExecSpace>
auto LbmTube<Scalar, LatticeType, ExecSpace>::U() const -> Tensor<Scalar, kDim_ + 1> {
  Kokkos::deep_copy(uHost_, u_);
  Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
  Scalar* dst = out.data();
  for (Index cell = 0; cell < N_; ++cell)
    for (Index d = 0; d < kDim_; ++d) {
      LBMINI_ASSERT_INDEX(uIndex(d, cell), uSize_);
      dst[cell * kDim_ + d] = uHost_(d * N_ + cell);
    }
  return out;
}

template<typename Scalar, typename LatticeType, typename ExecSpace>
void LbmTube<Scalar, LatticeType, ExecSpace>::Init() {
  InitFunctor<Scalar, LatticeType, MemorySpace> fn{
    params_, fA_, gA_, xi_, rho_, p_, tem_, u_
  };
  Kokkos::parallel_for("lbmini::kokkos::InitKernel", RangePolicy(0, N_), fn);
  Kokkos::fence();
}

template<typename Scalar, typename LatticeType, typename ExecSpace>
template<bool kStoreMacro>
void LbmTube<Scalar, LatticeType, ExecSpace>::step() {
  StepFunctor<Scalar, LatticeType, MemorySpace, kStoreMacro> fn{
    params_, fA_, gA_, fB_, gB_, xi_, rho_, p_, tem_, u_
  };
  Kokkos::parallel_for("lbmini::kokkos::StepKernel", RangePolicy(0, N_), fn);
}

template<typename Scalar, typename LatticeType, typename ExecSpace>
void LbmTube<Scalar, LatticeType, ExecSpace>::Step(const bool save) {
  Run(1, save);
}

template<typename Scalar, typename LatticeType, typename ExecSpace>
void LbmTube<Scalar, LatticeType, ExecSpace>::Run(const Index steps, bool /*save*/) {
  for (Index t = 0; t < steps; ++t) {
    if (t + 1 == steps) {
      step<true>();
    } else {
      step<false>();
    }
    std::swap(fA_, fB_);
    std::swap(gA_, gB_);
  }
  Kokkos::fence();
}

} // namespace lbmini::kokkos

#endif // LBMINI_KOKKOS_LBMTUBE_HPP_
