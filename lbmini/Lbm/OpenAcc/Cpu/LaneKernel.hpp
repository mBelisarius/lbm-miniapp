#ifndef LBMINI_OPENACC_CPU_LANEKERNEL_HPP_
#define LBMINI_OPENACC_CPU_LANEKERNEL_HPP_

/**
 * @file LaneKernel.hpp
 * @brief The fused D2Q9 step of the OpenACC CPU solver, written for `acc loop vector`.
 *
 * `nvc++` vectorises neither the monolithic per-cell kernel of the GPU backends
 * (the cell function is too large to inline into the compute region, and a call
 * in the loop body blocks the vectoriser) nor GCC vector extensions (they are not
 * supported in compute regions at all). What it does vectorise is a *lane loop*:
 * an `acc loop vector` over `kVL` neighbouring cells whose body is straight-line
 * code. This file is therefore the lane-loop formulation of the same arithmetic as
 * `host::Kernel` (see `Lbm/HostKernel.hpp`) - same layout (x-fastest, padded rows),
 * same factorised energy closure (see `LBMINI_ACC_AXIS`: two exponentials per
 * evaluation instead of nine, a diagonal Jacobian, three-term sums), the exponential
 * evaluated by an inline polynomial instead of a library call:
 *
 *  - one block is `kVL` consecutive cells of a row; the per-block state lives in
 *    a gang-private `Block` of `[direction][lane]` arrays, so every phase is a
 *    separate vector loop over unit-stride lane arrays. The nine directions of a
 *    phase are unrolled by the `LBMINI_ACC_Q` macro: a loop nested in the lane loop
 *    is the one `nvc++` vectorises instead of the lane loop. Loops that store to
 *    memory are the other way round (direction loop outside, one unit-stride store
 *    per lane loop), because `nvc++` reports a dependence between the nine stores
 *    to `plane`-strided addresses and refuses to vectorise them;
 *  - the phases are `acc routine vector` functions (pull, macroscopics + equilibria,
 *    the three Newton iterations, energy equilibrium, collision, stores), called in
 *    order for every block from the gang loop of the solver;
 *  - the Newton and energy-equilibrium phases are chains of dependent operations (a
 *    polynomial and eleven squarings per exponential, divisions) that are much longer
 *    than the reorder window of the core, so one 4-lane vector alone leaves the FP
 *    units idle: their loop body holds `kJam` independent copies, advanced iteration by
 *    iteration (`LBMINI_ACC_JAM_COPIES`), which overlaps the latencies (about 2x on
 *    the Newton phase, see `kVL` / `kJam`);
 *  - branches become selects. With `nvc++` 26.3 a lane loop stays scalar, without any
 *    diagnostic, when a select is nested three deep or when the operands of a select are
 *    computed inside its branches; the selects below are therefore at most two deep and
 *    take operands that are already computed. The solver-failure fallback is only
 *    evaluated (for every lane, then selected) in a block where some lane failed, a
 *    uniform branch.
 *
 * Without `-Mvect=simd:256` the compiler emits 128-bit vectors even for an AVX2 target,
 * and `exp` needs the polynomial below because the library's vector `exp` is a
 * per-lane call (see the CMake flags of the NVHPC build).
 */

#include <cmath>
#include <cstdint>

#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/HostKernel.hpp"

/// `LBMINI_ACC_Q(M)` expands `M(q)` for every D2Q9 direction with a literal `q`.
#define LBMINI_ACC_Q(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8)

/// `LBMINI_ACC_AX(c, a0, a1, a2)` picks the per-axis factor of the velocity component `c`: `a0` (-1), `a1` (0), `a2` (+1).
#define LBMINI_ACC_AX(c, a0, a1, a2) ((c) == 0 ? (a1) : ((c) == 1 ? (a2) : (a0)))

/**
 * `exp(x)` for `|x| <= 700`, as straight-line arithmetic that `nvc++` vectorises:
 * `exp(x) = exp(x / 2^11)^(2^11)`, a degree-13 Taylor polynomial of the reduced
 * argument (|r| <= 0.35, truncation 4e-18) and eleven squarings. The squarings
 * amplify the rounding error to ~2e-13 relative, far below the 1e-9 the backends
 * are compared at and invisible in the physics; the exponent-field tricks of
 * `Lbm/Simd.hpp` (bit casts, integer conversions) are not vectorised by `nvc++`.
 */
#define LBMINI_ACC_EXP(res, xv)                                                                         \
  {                                                                                                     \
    const double r_ = (xv) * (1.0 / 2048.0);                                                            \
    double p_ = 1.0 / 6227020800.0;                                                                     \
    p_ = p_ * r_ + 1.0 / 479001600.0;                                                                   \
    p_ = p_ * r_ + 1.0 / 39916800.0;                                                                    \
    p_ = p_ * r_ + 1.0 / 3628800.0;                                                                     \
    p_ = p_ * r_ + 1.0 / 362880.0;                                                                      \
    p_ = p_ * r_ + 1.0 / 40320.0;                                                                       \
    p_ = p_ * r_ + 1.0 / 5040.0;                                                                        \
    p_ = p_ * r_ + 1.0 / 720.0;                                                                         \
    p_ = p_ * r_ + 1.0 / 120.0;                                                                         \
    p_ = p_ * r_ + 1.0 / 24.0;                                                                          \
    p_ = p_ * r_ + 1.0 / 6.0;                                                                           \
    p_ = p_ * r_ + 0.5;                                                                                 \
    p_ = p_ * r_ + 1.0;                                                                                 \
    p_ = p_ * r_ + 1.0;                                                                                 \
    p_ = p_ * p_;                                                                                       \
    p_ = p_ * p_;                                                                                       \
    p_ = p_ * p_;                                                                                       \
    p_ = p_ * p_;                                                                                       \
    p_ = p_ * p_;                                                                                       \
    p_ = p_ * p_;                                                                                       \
    p_ = p_ * p_;                                                                                       \
    p_ = p_ * p_;                                                                                       \
    p_ = p_ * p_;                                                                                       \
    p_ = p_ * p_;                                                                                       \
    p_ = p_ * p_;                                                                                       \
    (res) = p_;                                                                                         \
  }

/**
 * `a0, a1, a2 = W_c exp(xi c - max_c' (xi c'))` for `c = -1, 0, 1`: one axis of the energy closure
 * (`host::Kernel::AxisExp`). The D2Q9 weights and the closure exponents are tensor products of an
 * x and a y factor, so one exponential per axis, `exp(-|xi|)`, gives the three values
 * `1, E, E^2` (in the order of the sign of `xi`) that the nine directions need. `wn` is the
 * weight of a non-zero velocity component, `wZero` that of a zero one. A NaN `xi` clamps to `-kMaxExp`.
 */
#define LBMINI_ACC_AXIS(xi_, wn_, wz_, a0_, a1_, a2_)                                                   \
  {                                                                                                     \
    const double xv_ = (xi_);                                                                           \
    const double t_ = -((xv_ < 0.0) ? -xv_ : xv_);                                                      \
    const double te_ = (t_ > -kMaxExp) ? t_ : -kMaxExp;                                                 \
    double e_;                                                                                          \
    LBMINI_ACC_EXP(e_, te_)                                                                             \
    const double e2_ = e_ * e_;                                                                         \
    const bool pos_ = xv_ >= 0.0;                                                                       \
    (a0_) = (wn_) * (pos_ ? e2_ : 1.0);                                                                 \
    (a1_) = (wz_) * e_;                                                                                 \
    (a2_) = (wn_) * (pos_ ? 1.0 : e2_);                                                                 \
  }

/// Cells per block, and the number of independent copies of the lane-loop body of the latency-bound phases (see `LBMINI_ACC_JAM_COPIES`).
#ifndef LBMINI_ACC_VL
#define LBMINI_ACC_VL 16
#endif
#ifndef LBMINI_ACC_JAM
#define LBMINI_ACC_JAM 4
#endif

namespace lbmini::openacc::cpu::lane {
inline constexpr int kQ = 9;
inline constexpr int kD = 2;
inline constexpr int kVL = LBMINI_ACC_VL; ///< cells per block (one `Block`, one vector loop per phase)
inline constexpr int kJam = LBMINI_ACC_JAM; ///< independent copies of the loop body in `Newton` and `Equilibrium`
inline constexpr int kStride = kVL / kJam;
static_assert(kVL % kJam == 0 && kStride % 4 == 0, "a block splits into whole 256-bit vectors per interleaved set");
inline constexpr double kTiny = 1.0e-12;
inline constexpr double kMaxExp = 700.0;
inline constexpr double kOmegaMax = 2.0 - 1.0e-7;

/**
 * `LBMINI_ACC_JAM_COPIES(M)` expands `M(c)` for the literals `c = 0 .. kJam - 1`; `LBMINI_ACC_LANE(c)` is
 * the lane of copy `c` in the loop `for l < kStride` (`l`, `l + kStride`, ...). A loop body written
 * for several copies holds `kJam` independent copies of a dependency chain in straight-line code, so the
 * out-of-order core overlaps the latency of the exponential and of the divisions of one copy with the
 * other copies (a single chain is latency-bound: it is longer than the reorder window of the core).
 */
#define LBMINI_ACC_LANE(c) (l + (c) * kStride)
#if LBMINI_ACC_JAM == 1
#define LBMINI_ACC_JAM_COPIES(M) M(0)
#elif LBMINI_ACC_JAM == 2
#define LBMINI_ACC_JAM_COPIES(M) M(0) M(1)
#elif LBMINI_ACC_JAM == 4
#define LBMINI_ACC_JAM_COPIES(M) M(0) M(1) M(2) M(3)
#else
#error "LBMINI_ACC_JAM must be 1, 2 or 4"
#endif

using Index = host::Index;
using Params = host::Params<double>;

/// `c_x` of direction `q` of the D2Q9 set (same order as the lattice tables).
constexpr int Cx(const int q) {
  return (q == 1 || q == 5 || q == 7) ? 1 : ((q == 2 || q == 6 || q == 8) ? -1 : 0);
}

/// `c_y` of direction `q` of the D2Q9 set.
constexpr int Cy(const int q) {
  return (q == 3 || q == 5 || q == 8) ? 1 : ((q == 4 || q == 6 || q == 7) ? -1 : 0);
}

/// Per-block scratch, private to the gang (one per thread), `[direction][lane]`.
struct Block {
  double f[kQ][kVL];
  double g[kQ][kVL];
  double feq[kQ][kVL];
  double geq[kQ][kVL];
  double xi[kD][kVL];
  double rho[kVL];
  double invRho[kVL];
  double ux[kVL];
  double uy[kVL];
  double tem[kVL];
  double pres[kVL];
  double targetE[kVL];
  double targetMx[kVL];
  double targetMy[kVL];
  double eps[kVL];
  double Lx[kVL];
  double Ly[kVL];
  int ok[kVL];
};

/// State of one row: where each direction gathers from, and the tube bounds.
struct Row {
  Index j;
  Index pitch;
  Index plane;
  Index ny;
  Index xLo;
  Index xHi;
  Index cellRow;
  Index src[kQ][2]; ///< `q * plane + wrappedRow * pitch` of the two y corners
};

#pragma acc routine seq
inline void BeginRow(const Params& p, const host::Geometry& geo, const Index xLo, const Index xHi, const Index j, Row& r) {
  r.j = j;
  r.pitch = geo.pitch;
  r.plane = geo.plane;
  r.ny = geo.ny;
  r.xLo = xLo;
  r.xHi = xHi;
  r.cellRow = j * geo.pitch;
  for (int q = 0; q < kQ; ++q) {
    Index y0 = j + p.stencil.oy0[q];
    y0 -= (y0 >= geo.ny) ? geo.ny : 0;
    Index y1 = j + p.stencil.oy1[q];
    y1 -= (y1 >= geo.ny) ? geo.ny : 0;
    r.src[q][0] = static_cast<Index>(q) * geo.plane + y0 * geo.pitch;
    r.src[q][1] = static_cast<Index>(q) * geo.plane + y1 * geo.pitch;
  }
}

/// Pulls the post-collision populations of the cells `[i0, i0 + kVL)`: 4-corner IDW gather.
#pragma acc routine vector
inline void Pull(
  const Params& p,
  const double* __restrict__ fIn,
  const double* __restrict__ gIn,
  const Row& r,
  const Index i0,
  const bool interior,
  Block& B
) {
  const auto& st = p.stencil;
  if (p.anyNearest) {
    // Degenerate weights: the generic gather of `lbmini::GatherSources`, lane by lane.
    #pragma acc loop seq
    for (int l = 0; l < kVL; ++l) {
      const Index i = i0 + l;
      for (int q = 0; q < kQ; ++q) {
        if (st.nearest[q] != 0) {
          const auto xN = static_cast<Index>(std::round(static_cast<double>(i) - st.cix[q]));
          const auto yN = static_cast<Index>(std::round(static_cast<double>(r.j) - st.ciy[q]));
          Index yW = yN % r.ny;
          yW += (yW < 0) ? r.ny : 0;
          const Index src = (xN < r.xLo || xN > r.xHi) ? r.cellRow + i : yW * r.pitch + xN;
          B.f[q][l] = fIn[static_cast<Index>(q) * r.plane + src];
          B.g[q][l] = gIn[static_cast<Index>(q) * r.plane + src];
        } else {
          Index xa = i + st.ox[q];
          Index xb = xa + 1;
          xa = (xa < r.xLo) ? r.xLo : ((xa > r.xHi) ? r.xHi : xa);
          xb = (xb < r.xLo) ? r.xLo : ((xb > r.xHi) ? r.xHi : xb);
          const double* fa = fIn + r.src[q][0];
          const double* fb = fIn + r.src[q][1];
          const double* ga = gIn + r.src[q][0];
          const double* gb = gIn + r.src[q][1];
          B.f[q][l] = st.w[q][0] * fa[xa] + st.w[q][1] * fa[xb] + st.w[q][2] * fb[xa] + st.w[q][3] * fb[xb];
          B.g[q][l] = st.w[q][0] * ga[xa] + st.w[q][1] * ga[xb] + st.w[q][2] * gb[xa] + st.w[q][3] * gb[xb];
        }
      }
    }
  } else if (interior) {
    #pragma acc loop seq
    for (int q = 0; q < kQ; ++q) {
      const Index x0 = i0 + st.ox[q];
      const double* fa = fIn + r.src[q][0] + x0;
      const double* fb = fIn + r.src[q][1] + x0;
      const double* ga = gIn + r.src[q][0] + x0;
      const double* gb = gIn + r.src[q][1] + x0;
      const double w0 = st.w[q][0];
      const double w1 = st.w[q][1];
      const double w2 = st.w[q][2];
      const double w3 = st.w[q][3];
      #pragma acc loop vector independent
      for (int l = 0; l < kVL; ++l) {
        B.f[q][l] = w0 * fa[l] + w1 * fa[l + 1] + w2 * fb[l] + w3 * fb[l + 1];
        B.g[q][l] = w0 * ga[l] + w1 * ga[l + 1] + w2 * gb[l] + w3 * gb[l + 1];
      }
    }
  } else {
    #pragma acc loop seq
    for (int q = 0; q < kQ; ++q) {
      const Index x0 = i0 + st.ox[q];
      const double* fa = fIn + r.src[q][0];
      const double* fb = fIn + r.src[q][1];
      const double* ga = gIn + r.src[q][0];
      const double* gb = gIn + r.src[q][1];
      const double w0 = st.w[q][0];
      const double w1 = st.w[q][1];
      const double w2 = st.w[q][2];
      const double w3 = st.w[q][3];
      #pragma acc loop vector independent
      for (int l = 0; l < kVL; ++l) {
        Index xa = x0 + l;
        Index xb = xa + 1;
        xa = (xa < r.xLo) ? r.xLo : ((xa > r.xHi) ? r.xHi : xa);
        xb = (xb < r.xLo) ? r.xLo : ((xb > r.xHi) ? r.xHi : xb);
        B.f[q][l] = w0 * fa[xa] + w1 * fa[xb] + w2 * fb[xa] + w3 * fb[xb];
        B.g[q][l] = w0 * ga[xa] + w1 * ga[xb] + w2 * gb[xa] + w3 * gb[xb];
      }
    }
  }
}

/// Sod seed of the cells `[i0, i0 + kVL)` (global column `gx0 + lane`): `f = feq`, `g` = normalised energy equilibrium.
#pragma acc routine vector
inline void Seed(const Params& p, const Index nxGlobal, const Index gx0, Block& B) {
  #pragma acc loop vector independent
  for (int l = 0; l < kVL; ++l) {
    Index gi = gx0 + l;
    gi = (gi < nxGlobal - 1) ? gi : nxGlobal - 1;
    const bool left = gi < nxGlobal / 2;
    const double rho0 = left ? p.densityL : p.densityR;
    const double p0 = left ? p.pressureL : p.pressureR;
    const double tem0 = p.cs2 * p0 / (rho0 * p.rgas);
    const double uax = 0.0 - p.ushiftX;
    const double uay = 0.0 - p.ushiftY;
    const double tx = uax * uax + tem0;
    const double ty = uay * uay + tem0;
    const double pxm = 0.5 * ((-uax) + tx);
    const double px0 = 1.0 - tx;
    const double pxp = 0.5 * (uax + tx);
    const double pym = 0.5 * ((-uay) + ty);
    const double py0 = 1.0 - ty;
    const double pyp = 0.5 * (uay + ty);
    const double wZero = 1.0 - tem0;
    const double wNonZero = 0.5 * tem0;
    const double wn = wZero + (wNonZero - wZero);
    double sumW = 0.0;
#define LBMINI_ACC_SEED(q)                                                                                    \
  {                                                                                                           \
    B.f[q][l] = (rho0 * (Cx(q) == 0 ? px0 : (Cx(q) == 1 ? pxp : pxm))) * (Cy(q) == 0 ? py0 : (Cy(q) == 1 ? pyp : pym)); \
    sumW += (Cx(q) != 0 ? wn : wZero) * (Cy(q) != 0 ? wn : wZero);                                           \
  }
    LBMINI_ACC_Q(LBMINI_ACC_SEED)
#undef LBMINI_ACC_SEED
    const double E = p.cv * tem0 + 0.5 * 0.0;
    const double targetE = 2.0 * rho0 * E;
    const bool useW = sumW > kTiny;
    const double sc = useW ? targetE / sumW : targetE / 9.0;
#define LBMINI_ACC_SEEDG(q) B.g[q][l] = (useW ? (Cx(q) != 0 ? wn : wZero) * (Cy(q) != 0 ? wn : wZero) : 1.0) * sc;
    LBMINI_ACC_Q(LBMINI_ACC_SEEDG)
#undef LBMINI_ACC_SEEDG
  }
}

/**
 * Macroscopic recovery, equilibria and Newton targets of a block; the warm start
 * `xi` is loaded from memory (a step) or starts cold at zero (`Init`).
 */
#pragma acc routine vector
inline void Prepare(const Params& p, const double* __restrict__ xiIo, const Index cell, const Index plane, const bool coldStart, Block& B) {
  const double ushiftX = p.ushiftX;
  const double ushiftY = p.ushiftY;
  const double cv = p.cv;
  const double invCv = p.invCv;
  const double rgas = p.rgas;
  #pragma acc loop vector independent
  for (int l = 0; l < kVL; ++l) {
    double rhoSum = 0.0;
    double nrg = 0.0;
    double momX = 0.0;
    double momY = 0.0;
#define LBMINI_ACC_MACRO(q)                                                                                   \
  {                                                                                                           \
    rhoSum += B.f[q][l];                                                                                      \
    nrg += B.g[q][l];                                                                                         \
    momX += (static_cast<double>(Cx(q)) + ushiftX) * B.f[q][l];                                               \
    momY += (static_cast<double>(Cy(q)) + ushiftY) * B.f[q][l];                                               \
  }
    LBMINI_ACC_Q(LBMINI_ACC_MACRO)
#undef LBMINI_ACC_MACRO
    const double invRho = 1.0 / rhoSum;
    const double ux = momX * invRho;
    const double uy = momY * invRho;
    const double kin = 0.5 * (ux * ux + uy * uy);
    const double tem = (0.5 * nrg * invRho - kin) * invCv;
    B.rho[l] = rhoSum;
    B.invRho[l] = invRho;
    B.ux[l] = ux;
    B.uy[l] = uy;
    B.tem[l] = tem;
    B.pres[l] = rgas * rhoSum * tem;

    const double u2 = ux * ux + uy * uy;
    const double uax = ux - ushiftX;
    const double uay = uy - ushiftY;
    const double tx = uax * uax + tem;
    const double ty = uay * uay + tem;
    const double pxm = 0.5 * ((-uax) + tx);
    const double px0 = 1.0 - tx;
    const double pxp = 0.5 * (uax + tx);
    const double pym = 0.5 * ((-uay) + ty);
    const double py0 = 1.0 - ty;
    const double pyp = 0.5 * (uay + ty);
#define LBMINI_ACC_PREP(q)                                                                                    \
  B.feq[q][l] = (rhoSum * LBMINI_ACC_AX(Cx(q), pxm, px0, pxp)) * LBMINI_ACC_AX(Cy(q), pym, py0, pyp);
    LBMINI_ACC_Q(LBMINI_ACC_PREP)
#undef LBMINI_ACC_PREP
    const double E = cv * tem + 0.5 * u2;
    const double invE = 1.0 / E;
    B.targetE[l] = 2.0 * rhoSum * E;
    B.targetMx[l] = ux * (E + tem) * invE;
    B.targetMy[l] = uy * (E + tem) * invE;
    B.xi[0][l] = coldStart ? 0.0 : xiIo[cell + l];
    B.xi[1][l] = coldStart ? 0.0 : xiIo[plane + cell + l];
  }
}

/// Writes rho, p, T, u of the block (one unit-stride store per vector loop).
#pragma acc routine vector
inline void StoreMacro(
  const Index cell,
  const Index plane,
  double* __restrict__ rho,
  double* __restrict__ pres,
  double* __restrict__ tem,
  double* __restrict__ u,
  const Block& B
) {
  #pragma acc loop vector independent
  for (int l = 0; l < kVL; ++l)
    rho[cell + l] = B.rho[l];
  #pragma acc loop vector independent
  for (int l = 0; l < kVL; ++l)
    pres[cell + l] = B.pres[l];
  #pragma acc loop vector independent
  for (int l = 0; l < kVL; ++l)
    tem[cell + l] = B.tem[l];
  #pragma acc loop vector independent
  for (int l = 0; l < kVL; ++l)
    u[cell + l] = B.ux[l];
  #pragma acc loop vector independent
  for (int l = 0; l < kVL; ++l)
    u[plane + cell + l] = B.uy[l];
}

/**
 * One Newton-Raphson iteration of the energy closure with damping `alpha_` (a literal) for the copy `c_`: two
 * scalar steps, one per axis (the Jacobian of the factorised closure is diagonal, see
 * `host::Kernel::Collide`), from three weighted exponentials per axis. Works on the copy's variables
 * `xi0_c_`, `xi1_c_`, `ok_c_` (the solver-health predicate), `wn_c_`, `wz_c_`, `tmx_c_`, `tmy_c_` of `Newton`.
 */
#define LBMINI_ACC_NEWTON(alpha_, c_)                                                                         \
  {                                                                                                           \
    double px0_, px1_, px2_, py0_, py1_, py2_;                                                                \
    LBMINI_ACC_AXIS(xi0_##c_, wn_##c_, wz_##c_, px0_, px1_, px2_)                                             \
    LBMINI_ACC_AXIS(xi1_##c_, wn_##c_, wz_##c_, py0_, py1_, py2_)                                             \
    const double zx_ = px0_ + px1_ + px2_;                                                                    \
    const double zy_ = py0_ + py1_ + py2_;                                                                    \
    const double zxy_ = zx_ * zy_;                                                                            \
    ok_##c_ &= (zxy_ > kTiny) ? 1 : 0;                                                                        \
    const double rz_ = 1.0 / zxy_;                                                                            \
    const double invZx_ = zy_ * rz_;                                                                          \
    const double invZy_ = zx_ * rz_;                                                                          \
    const double m1x_ = (px2_ - px0_) * invZx_;                                                               \
    const double m1y_ = (py2_ - py0_) * invZy_;                                                               \
    const double jxx_ = ((px2_ + px0_) * invZx_ - m1x_ * m1x_) + kTiny;                                       \
    const double jyy_ = ((py2_ + py0_) * invZy_ - m1y_ * m1y_) + kTiny;                                       \
    const double jxy_ = jxx_ * jyy_;                                                                          \
    ok_##c_ &= (((jxy_ < 0.0) ? -jxy_ : jxy_) > kTiny) ? 1 : 0;                                               \
    const double rj_ = (alpha_) * (1.0 / jxy_);                                                               \
    const double r0_ = (m1x_ + ushiftX) - tmx_##c_;                                                           \
    const double r1_ = (m1y_ + ushiftY) - tmy_##c_;                                                           \
    xi0_##c_ -= r0_ * jyy_ * rj_;                                                                             \
    xi1_##c_ -= r1_ * jxx_ * rj_;                                                                             \
  }

#define LBMINI_ACC_NEWTON_LOAD(c_)                                                                            \
  double xi0_##c_ = B.xi[0][LBMINI_ACC_LANE(c_)];                                                             \
  double xi1_##c_ = B.xi[1][LBMINI_ACC_LANE(c_)];                                                             \
  const double tmx_##c_ = B.targetMx[LBMINI_ACC_LANE(c_)];                                                    \
  const double tmy_##c_ = B.targetMy[LBMINI_ACC_LANE(c_)];                                                    \
  const double wz_##c_ = 1.0 - B.tem[LBMINI_ACC_LANE(c_)];                                                    \
  const double wn_##c_ = wz_##c_ + (0.5 * B.tem[LBMINI_ACC_LANE(c_)] - wz_##c_);                              \
  int ok_##c_ = 1;

#define LBMINI_ACC_NEWTON_1(c_) LBMINI_ACC_NEWTON(1.0, c_)
#define LBMINI_ACC_NEWTON_2(c_) LBMINI_ACC_NEWTON(0.5, c_)
#define LBMINI_ACC_NEWTON_3(c_) LBMINI_ACC_NEWTON(0.25, c_)

#define LBMINI_ACC_NEWTON_STORE(c_)                                                                           \
  B.xi[0][LBMINI_ACC_LANE(c_)] = xi0_##c_;                                                                    \
  B.xi[1][LBMINI_ACC_LANE(c_)] = xi1_##c_;                                                                    \
  B.ok[LBMINI_ACC_LANE(c_)] = ok_##c_;

/**
 * The three damped Newton iterations (`alpha = 1, 1/2, 1/4`) of every lane of the block, from the warm
 * start `B.xi`; clears `B.ok` where the solve failed. The copies of the lane loop body advance iteration
 * by iteration, so that the chains of their exponentials and divisions sit next to each other.
 */
#pragma acc routine vector
inline void Newton(const Params& p, Block& B) {
  const double ushiftX = p.ushiftX;
  const double ushiftY = p.ushiftY;
  #pragma acc loop vector independent
  for (int l = 0; l < kStride; ++l) {
    LBMINI_ACC_JAM_COPIES(LBMINI_ACC_NEWTON_LOAD)
    LBMINI_ACC_JAM_COPIES(LBMINI_ACC_NEWTON_1)
    LBMINI_ACC_JAM_COPIES(LBMINI_ACC_NEWTON_2)
    LBMINI_ACC_JAM_COPIES(LBMINI_ACC_NEWTON_3)
    LBMINI_ACC_JAM_COPIES(LBMINI_ACC_NEWTON_STORE)
  }
}
#undef LBMINI_ACC_NEWTON_STORE
#undef LBMINI_ACC_NEWTON_3
#undef LBMINI_ACC_NEWTON_2
#undef LBMINI_ACC_NEWTON_1
#undef LBMINI_ACC_NEWTON_LOAD
#undef LBMINI_ACC_NEWTON


/// `LBMINI_ACC_Q2(M, a)` expands `M(q, a)` for every D2Q9 direction with a literal `q`.
#define LBMINI_ACC_Q2(M, a) M(0, a) M(1, a) M(2, a) M(3, a) M(4, a) M(5, a) M(6, a) M(7, a) M(8, a)

#define LBMINI_ACC_GEQ_Q(q, c_) B.geq[q][LBMINI_ACC_LANE(c_)] = LBMINI_ACC_AX(Cx(q), sx0, sx1, sx2) * LBMINI_ACC_AX(Cy(q), py0, py1, py2);

/// Energy equilibrium of the copy `c_` from the solved `xi`.
#define LBMINI_ACC_EQ_LANE(c_)                                                                                \
  {                                                                                                           \
    const double tem = B.tem[LBMINI_ACC_LANE(c_)];                                                            \
    const double wZero = 1.0 - tem;                                                                           \
    const double wn = wZero + (0.5 * tem - wZero);                                                            \
    double px0, px1, px2, py0, py1, py2;                                                                      \
    LBMINI_ACC_AXIS(B.xi[0][LBMINI_ACC_LANE(c_)], wn, wZero, px0, px1, px2)                                   \
    LBMINI_ACC_AXIS(B.xi[1][LBMINI_ACC_LANE(c_)], wn, wZero, py0, py1, py2)                                   \
    const double scale = B.targetE[LBMINI_ACC_LANE(c_)] / ((px0 + px1 + px2) * (py0 + py1 + py2));            \
    const double sx0 = scale * px0;                                                                           \
    const double sx1 = scale * px1;                                                                           \
    const double sx2 = scale * px2;                                                                           \
    LBMINI_ACC_Q2(LBMINI_ACC_GEQ_Q, c_)                                                                       \
  }

/**
 * Energy equilibrium from the Newton solution. A block whose lanes all solved takes the plain vector
 * loop; a block with a failed lane (rare: degenerate states) restarts those lanes from `xi = 0` with
 * the normalised-weight fallback, computing both variants for every lane and selecting.
 */
#pragma acc routine vector
inline void Equilibrium(Block& B) {
  int failed = 0;
  #pragma acc loop seq
  for (int l = 0; l < kVL; ++l)
    failed |= 1 - B.ok[l];
  if (failed == 0) {
    #pragma acc loop vector independent
    for (int l = 0; l < kStride; ++l) {
      LBMINI_ACC_JAM_COPIES(LBMINI_ACC_EQ_LANE)
    }
  } else {
    #pragma acc loop vector independent
    for (int l = 0; l < kVL; ++l) {
      const bool ok = B.ok[l] != 0;
      const double xi0 = ok ? B.xi[0][l] : 0.0;
      const double xi1 = ok ? B.xi[1][l] : 0.0;
      B.xi[0][l] = xi0;
      B.xi[1][l] = xi1;
      const double tem = B.tem[l];
      const double wZero = 1.0 - tem;
      const double wn = wZero + (0.5 * tem - wZero);
      double px0, px1, px2, py0, py1, py2;
      LBMINI_ACC_AXIS(xi0, wn, wZero, px0, px1, px2)
      LBMINI_ACC_AXIS(xi1, wn, wZero, py0, py1, py2)
      const double targetE = B.targetE[l];
      const double scale = targetE / ((px0 + px1 + px2) * (py0 + py1 + py2));
      const double sx0 = scale * px0;
      const double sx1 = scale * px1;
      const double sx2 = scale * px2;
      double sumW = 0.0;
#define LBMINI_ACC_SUMW(q) sumW += (Cx(q) != 0 ? wn : wZero) * (Cy(q) != 0 ? wn : wZero);
      LBMINI_ACC_Q(LBMINI_ACC_SUMW)
#undef LBMINI_ACC_SUMW
      const double sc = targetE / sumW;
      const double uni = targetE / 9.0;
      const bool degenerate = sumW <= kTiny;
#define LBMINI_ACC_GEQ(q)                                                                                     \
  {                                                                                                           \
    const double gOk_ = LBMINI_ACC_AX(Cx(q), sx0, sx1, sx2) * LBMINI_ACC_AX(Cy(q), py0, py1, py2);            \
    const double gFb_ = (Cx(q) != 0 ? wn : wZero) * (Cy(q) != 0 ? wn : wZero) * sc;                           \
    const double gDeg_ = degenerate ? uni : gFb_;                                                             \
    B.geq[q][l] = ok ? gOk_ : gDeg_;                                                                          \
  }
      LBMINI_ACC_Q(LBMINI_ACC_GEQ)
#undef LBMINI_ACC_GEQ
    }
  }
}
#undef LBMINI_ACC_EQ_LANE
#undef LBMINI_ACC_GEQ_Q
#undef LBMINI_ACC_Q2

/// BGK + Knudsen-sensor collision of a block; leaves the post-collision populations in `B.f`, `B.g`.
#pragma acc routine vector
inline void Collide(const Params& p, Block& B) {
  const double ushiftX = p.ushiftX;
  const double ushiftY = p.ushiftY;
  const double viscosity = p.viscosity;
  const double condOverCp = p.conductivity / p.cp;
  // Knudsen sensor `eps` and the moments `L[a] = sum_i 2 (u . c_i) c_i_a (f - feq)` (c_i unshifted).
  #pragma acc loop vector independent
  for (int l = 0; l < kVL; ++l) {
    const double ux = B.ux[l];
    const double uy = B.uy[l];
    double eps = 0.0;
    double Lx = 0.0;
    double Ly = 0.0;
#define LBMINI_ACC_EPS(q)                                                                                     \
  {                                                                                                           \
    const double diff_ = B.f[q][l] - B.feq[q][l];                                                             \
    const double den_ = (B.feq[q][l] > kTiny) ? B.feq[q][l] : kTiny;                                          \
    eps += ((diff_ < 0.0) ? -diff_ : diff_) / den_;                                                           \
    const double cix_ = static_cast<double>(Cx(q));                                                           \
    const double ciy_ = static_cast<double>(Cy(q));                                                           \
    const double uvi_ = ux * cix_ + uy * ciy_;                                                                \
    const double aux_ = 2.0 * uvi_ * diff_;                                                                   \
    Lx += aux_ * cix_;                                                                                        \
    Ly += aux_ * ciy_;                                                                                        \
  }
    LBMINI_ACC_Q(LBMINI_ACC_EPS)
#undef LBMINI_ACC_EPS
    B.eps[l] = eps * (1.0 / 9.0);
    B.Lx[l] = Lx;
    B.Ly[l] = Ly;
  }
  #pragma acc loop vector independent
  for (int l = 0; l < kVL; ++l) {
    const double invRho = B.invRho[l];
    const double tem = B.tem[l];
    const double eps = B.eps[l];
    const double Lx = B.Lx[l];
    const double Ly = B.Ly[l];
    // `tau` and `tauT` share `1 / (rho T)`, `omega` and `omegaT` one division.
    const double invT = 1.0 / tem;
    const double tau = viscosity * invRho * invT + 0.5;
    const double tauT = condOverCp * invRho * invT + 0.5;
    const double rTau = 1.0 / (tau * tauT);
    const double omega = tauT * rTau;
    double omegaT = tau * rTau;
    // `omega / sigma` with `sigma` in {omega, 1.35, 1.05, 1}, without a division; two selects deep at most,
    // which is what `nvc++` still if-converts in a vector loop.
    const double mult = (eps >= 0.1) ? (1.0 / 1.35) : ((eps >= 0.01) ? (1.0 / 1.05) : 1.0);
    const double omegaS = omega * mult;
    double omegaL = (eps >= 1.0) ? 1.0 : omegaS;
    omegaL = (omegaL > 1.0) ? omegaL : 1.0;
    omegaL = (omegaL < kOmegaMax) ? omegaL : kOmegaMax;
    omegaT = (omegaT > 1.0) ? omegaT : 1.0;
    omegaT = (omegaT < kOmegaMax) ? omegaT : kOmegaMax;
    const double dOmega = omegaL - omegaT;
    const double wZero = 1.0 - tem;
    const double wn = wZero + (0.5 * tem - wZero);
#define LBMINI_ACC_OUT(q)                                                                                     \
  {                                                                                                           \
    const double fOld_ = B.f[q][l];                                                                           \
    const double cidotL_ = Lx * (static_cast<double>(Cx(q)) + ushiftX) + Ly * (static_cast<double>(Cy(q)) + ushiftY); \
    const double gDiff_ = ((Cx(q) != 0 ? wn : wZero) * (Cy(q) != 0 ? wn : wZero)) * cidotL_ * invT;         \
    const double gOld_ = B.g[q][l];                                                                           \
    B.f[q][l] = fOld_ + omegaL * (B.feq[q][l] - fOld_);                                                       \
    B.g[q][l] = gOld_ + omegaL * (B.geq[q][l] - gOld_) + dOmega * gDiff_;                                     \
  }
    LBMINI_ACC_Q(LBMINI_ACC_OUT)
#undef LBMINI_ACC_OUT
  }
}

/// Writes the post-collision state of a block: the populations and the warm start.
#pragma acc routine vector
inline void StoreState(
  const Index cell,
  const Index plane,
  double* __restrict__ fOut,
  double* __restrict__ gOut,
  double* __restrict__ xiIo,
  const Block& B
) {
  #pragma acc loop seq
  for (int q = 0; q < kQ; ++q) {
    double* fo = fOut + static_cast<Index>(q) * plane + cell;
    double* go = gOut + static_cast<Index>(q) * plane + cell;
    #pragma acc loop vector independent
    for (int l = 0; l < kVL; ++l)
      fo[l] = B.f[q][l];
    #pragma acc loop vector independent
    for (int l = 0; l < kVL; ++l)
      go[l] = B.g[q][l];
  }
  #pragma acc loop vector independent
  for (int l = 0; l < kVL; ++l)
    xiIo[cell + l] = B.xi[0][l];
  #pragma acc loop vector independent
  for (int l = 0; l < kVL; ++l)
    xiIo[plane + cell + l] = B.xi[1][l];
}
} // namespace lbmini::openacc::cpu::lane

#endif // LBMINI_OPENACC_CPU_LANEKERNEL_HPP_
