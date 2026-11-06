/**
 * @file kernels.cl
 * @brief Fused D2Q9 compressible-LBM kernels of the two OpenCL backends.
 *
 * The source is not self-contained: the host (`Lbm/OpenCl/Program.hpp`,
 * `ProgramDefines()`) prepends a block of `#define`s that bakes the mesh
 * extents, the fluid constants and the per-direction streaming stencil into
 * the program as compile-time constants. Every table below is indexed only by
 * fully unrolled loop counters, so each entry folds into an immediate operand
 * and every branch that depends on the direction alone is resolved at JIT
 * time.
 *
 * State between steps is the POST-collision populations `f`, `g`
 * (direction-major SoA `f[idc * N + cell]`, `cell = i * ny + j`) in two
 * ping-pong buffers per field, plus the Newton warm start `xi[d * N + cell]`.
 * One time step is one kernel: pull streaming from the four IDW source cells,
 * macroscopic sums, equilibria, Newton-Raphson energy closure and BGK/Knudsen
 * collision, all in registers. The arithmetic is that of
 * the reference of the test suite (`test/ReferenceTube.hpp`), except for the factorised energy closure (see
 * `AxisExp`: 2 exponentials per evaluation, two scalar Newton steps, shared
 * divisions), which agrees with it to rounding error; the program is never
 * built with relaxed-math options.
 *
 * Optional host macros:
 *  - `LBM_2D`: 2-D NDRange, dimension 0 = `j` (fastest), dimension 1 = `i`;
 *    otherwise a 1-D NDRange over the flat cell index.
 *  - `LBM_KERNEL_ATTR`: attribute list prepended to every kernel (e.g. a
 *    `reqd_work_group_size`).
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef LBM_KERNEL_ATTR
#define LBM_KERNEL_ATTR
#endif

typedef double real;

#define LBM_TINY 1.0e-12
#define LBM_MAX_EXP 700.0
#define LBM_OMEGA_MAX (2.0 - 1.0e-7)

__constant int kCx[LBM_Q] = LBM_CX;
__constant int kCy[LBM_Q] = LBM_CY;
__constant int kStOx[LBM_Q] = LBM_ST_OX;
__constant int kStOy0[LBM_Q] = LBM_ST_OY0;
__constant int kStOy1[LBM_Q] = LBM_ST_OY1;
__constant int kStNearest[LBM_Q] = LBM_ST_NEAREST;
__constant real kStW[LBM_Q * 4] = LBM_ST_W;
__constant real kStCix[LBM_Q] = LBM_ST_CIX;
__constant real kStCiy[LBM_Q] = LBM_ST_CIY;

/// Shifted lattice velocity `c_i + U` (reference `Cshift`), a constant per direction.
#define CSX(q) ((real)kCx[q] + LBM_USHIFT_X)
#define CSY(q) ((real)kCy[q] + LBM_USHIFT_Y)

/**
 * Temperature-dependent lattice weight of direction `q` (reference `Weights`):
 * the product over both axes of `wZero = 1 - T` for a zero velocity component
 * and `wZero + (0.5 T - wZero)` for a non-zero one. `wz`/`wn` are those two
 * factors, computed once per cell.
 */
inline real Weight(const int q, const real wz, const real wn) {
  return (kCx[q] != 0 ? wn : wz) * (kCy[q] != 0 ? wn : wz);
}

inline void WeightFactors(const real tem, real* wz, real* wn) {
  const real wZero = 1.0 - tem;
  const real wNonZero = 0.5 * tem;
  *wz = wZero;
  *wn = wZero + (wNonZero - wZero);
}

/// Density equilibrium, factorised product form.
inline void Feq(const real rho, const real ux, const real uy, const real tem, real feq[LBM_Q]) {
  const real uax = ux - LBM_USHIFT_X;
  const real uay = uy - LBM_USHIFT_Y;
  const real tx = uax * uax + tem;
  const real ty = uay * uay + tem;
  const real px0 = 1.0 - tx;
  const real pxp = 0.5 * (uax + tx);
  const real pxm = 0.5 * (-uax + tx);
  const real py0 = 1.0 - ty;
  const real pyp = 0.5 * (uay + ty);
  const real pym = 0.5 * (-uay + ty);
#pragma unroll
  for (int q = 0; q < LBM_Q; ++q) {
    const real pfx = kCx[q] == 0 ? px0 : (kCx[q] == 1 ? pxp : pxm);
    const real pfy = kCy[q] == 0 ? py0 : (kCy[q] == 1 ? pyp : pym);
    feq[q] = rho * pfx * pfy;
  }
}

/**
 * `1 / x` to about one unit in the last place, from a single-precision seed.
 *
 * A double-precision division costs 8 FP64 instructions on an NVIDIA GPU, and the FP64 pipe (1/64 of the
 * FP32 rate on consumer parts) is the bottleneck. The `1 / float(x)` seed runs on the FP32 pipe and has a
 * relative error `e <= 6e-8`; `r (1 + e + e^2)` then has an error of `e^3 ~ 2e-22`, i.e. rounding only, in
 * three FP64 FMAs. Not correctly rounded (within ~1 ulp). `x` must lie in the single-precision range:
 * every caller divides by a positive state variable, or by a quantity whose range `solverOk` checks.
 */
inline real Recip(const real x) {
  const real r0 = (real)(1.0f / (float)x);
  const real e = 1.0 - x * r0;
  return r0 + r0 * (e + e * e);
}

/// rho, 1 / rho, u, T, p of one cell.
inline void Macroscopic(
  const real f[LBM_Q],
  const real g[LBM_Q],
  real* rho,
  real* invRho,
  real* ux,
  real* uy,
  real* tem,
  real* pres
) {
  real rhoSum = 0.0;
  real nrg = 0.0;
  real momX = 0.0;
  real momY = 0.0;
#pragma unroll
  for (int q = 0; q < LBM_Q; ++q) {
    rhoSum += f[q];
    nrg += g[q];
    momX += CSX(q) * f[q];
    momY += CSY(q) * f[q];
  }
  const real rinv = Recip(rhoSum);
  const real vx = momX * rinv;
  const real vy = momY * rinv;
  const real kin = 0.5 * (vx * vx + vy * vy);
  const real t = (0.5 * nrg * rinv - kin) * LBM_INV_CV;
  *rho = rhoSum;
  *invRho = rinv;
  *ux = vx;
  *uy = vy;
  *tem = t;
  *pres = LBM_RGAS * rhoSum * t;
}

/**
 * `p[c + 1] = W_c * exp(xi * c - max_c' (xi * c'))` for `c = -1, 0, 1`: one axis
 * of the energy closure.
 *
 * The D2Q9 weights and the exponents of the energy equilibrium are tensor
 * products, `W_q = Wx(c_x) Wy(c_y)` and `s_q = xi_0 (c_x + Ux) + xi_1 (c_y + Uy)`,
 * so `W_q exp(s_q - smax)` of the reference is `Px(c_x) Py(c_y)` with one factor
 * per axis (the `U` shift and the common maximum cancel in the product). An axis
 * needs a single exponential, `E = exp(-|xi|)`: the three exponents `xi c - max`
 * are `0`, `-|xi|` and `-2 |xi|` in the order of the sign of `xi`, i.e. `1`, `E`
 * and `E^2`. That is 2 exponentials per evaluation (8 per cell and step) instead
 * of the 9 of the per-direction form (36 per cell and step), and the nine-term
 * sums of the Newton step become three-term ones. Double precision is the
 * bottleneck of a GPU, so the number of FP64 instructions is the cost.
 *
 * It equals the reference to rounding error while neither of its clamps
 * (`|s - smax| <= 700`) is active; beyond that every omitted term is below
 * `exp(-700)` relative to the largest one. A NaN `xi` takes the clamp.
 */
inline void AxisExp(const real xi, const real wn, const real wz, real p[3]) {
  const real t = -fabs(xi);
  const real e = exp((t > -LBM_MAX_EXP) ? t : -LBM_MAX_EXP);
  const real e2 = e * e;
  const bool pos = xi >= 0.0;
  p[0] = wn * (pos ? e2 : 1.0);
  p[1] = wz * e;
  p[2] = wn * (pos ? 1.0 : e2);
}

/// The factors of the product form of the density equilibrium: `feq(q) = a[c_x + 1] * b[c_y + 1]`,
/// `a = rho * px`, `b = py` (`c = -1, 0, +1`); see `Feq`.
inline void FeqFactors(const real rho, const real ux, const real uy, const real tem, real a[3], real b[3]) {
  const real uax = ux - LBM_USHIFT_X;
  const real uay = uy - LBM_USHIFT_Y;
  const real tx = uax * uax + tem;
  const real ty = uay * uay + tem;
  a[0] = rho * (0.5 * (-uax + tx));
  a[1] = rho * (1.0 - tx);
  a[2] = rho * (0.5 * (uax + tx));
  b[0] = 0.5 * (-uay + ty);
  b[1] = 1.0 - ty;
  b[2] = 0.5 * (uay + ty);
}

/// Energy equilibrium from the Newton solution `xi` as the factors of the product form:
/// `geq(q) = sx[c_x + 1] * gy[c_y + 1]`.
inline void GeqFactors(const real xi[2], const real wn, const real wz, const real targetE, real sx[3], real gy[3]) {
  real px[3];
  AxisExp(xi[0], wn, wz, px);
  AxisExp(xi[1], wn, wz, gy);
  const real scale = targetE * Recip((px[0] + px[1] + px[2]) * (gy[0] + gy[1] + gy[2]));
#pragma unroll
  for (int k = 0; k < 3; ++k)
    sx[k] = scale * px[k];
}

/**
 * Equilibria + Newton-Raphson energy closure + BGK collision, in place.
 *
 * `f`/`g` enter pre-collision and leave post-collision; `xi` is the Newton
 * warm start (read and updated). Fixed iteration count, clamped reciprocals
 * and a sticky `solverOk` predicate keep the control flow uniform.
 */
inline void Collide(
  const real rhoc,
  const real invRho,
  const real ux,
  const real uy,
  const real temc,
  real f[LBM_Q],
  real g[LBM_Q],
  real xi[2]
) {
  const real u2 = ux * ux + uy * uy;
  real wz;
  real wn;
  WeightFactors(temc, &wz, &wn);

  // The density equilibrium in product form, `feq(q) = a[c_x + 1] * b[c_y + 1]`: only six values
  // are kept and the products are formed where they are used.
  real a[3];
  real b[3];
  FeqFactors(rhoc, ux, uy, temc, a, b);

  // Geq via Newton-Raphson on the normalised energy flux.
  //
  // The closure distribution `W_q exp(xi . (c_q + U))` is a product of an x and a
  // y factor, so x and y are independent: the Jacobian of the reference is
  // diagonal (its off-diagonal term is exactly `S2xy / Z - (S1x / Z) (S1y / Z) = 0`)
  // and the iteration is two scalar Newton steps,
  // `xi_a -= alpha (m1_a - target_a) / var_a`, each from three weighted
  // exponentials of one axis (see `AxisExp`). A division costs about 8 FP64
  // instructions, so the two axes share one reciprocal per stage.
  const real E = LBM_CV * temc + 0.5 * u2;
  const real targetE = 2.0 * rhoc * E;
  const real invE = Recip(E);
  const real targetMx = ux * (E + temc) * invE; // 2 rho u (E + T) / (2 rho E)
  const real targetMy = uy * (E + temc) * invE;

  real alpha = 1.0;
  bool solverOk = true;

  // Unrolling copies the loop body once per iteration for no speed.
#pragma nounroll
  for (int iter = 0; iter < LBM_MAX_ITER; ++iter) {
    real px[3];
    real py[3];
    AxisExp(xi[0], wn, wz, px);
    AxisExp(xi[1], wn, wz, py);
    const real zx = px[0] + px[1] + px[2];
    const real zy = py[0] + py[1] + py[2];
    const real zxy = zx * zy;
    solverOk = solverOk && (zxy > LBM_TINY);
    // Scale residual and regularised variance by Z^2: their quotient
    // no longer needs the normalisation reciprocal.
    const real dx = px[2] - px[0];
    const real dy = py[2] - py[0];
    const real jxx = (px[2] + px[0]) * zx - dx * dx + LBM_TINY * (zx * zx);
    const real jyy = (py[2] + py[0]) * zy - dy * dy + LBM_TINY * (zy * zy);
    const real jxy = jxx * jyy;
    solverOk = solverOk && (fabs(jxy) > LBM_TINY * (zxy * zxy));

    const real rj = alpha * Recip(jxy);
    const real r0 = (dx + (LBM_USHIFT_X - targetMx) * zx) * zx;
    const real r1 = (dy + (LBM_USHIFT_Y - targetMy) * zy) * zy;
    xi[0] -= r0 * jyy * rj; // alpha r0 / jxx
    xi[1] -= r1 * jxx * rj; // alpha r1 / jyy
    alpha *= 0.5;
  }

  // Energy equilibrium `geq(q) = sx[c_x + 1] * gy[c_y + 1]`.
  real sx[3];
  real gy[3];
  if (solverOk) {
    GeqFactors(xi, wn, wz, targetE, sx, gy);
    // The reference also stores `log(scale / rho) - smax` as warm-start
    // component 0; nothing ever reads it, so it is neither computed nor stored.
  } else {
    // Fallback: normalised-Wi distribution (`W_q = wf(c_x) wf(c_y)`, summing to `(2 wn + wz)^2`),
    // cold restart of the warm start.
    xi[0] = 0.0;
    xi[1] = 0.0;
    const real sumAxis = wn + wz + wn;
    const real sumW = sumAxis * sumAxis;
    const real wfb[3] = { wn, wz, wn };
    if (sumW <= LBM_TINY) {
      const real uni = targetE / (real)LBM_Q;
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        sx[k] = uni;
        gy[k] = 1.0;
      }
    } else {
      const real sc = targetE / sumW;
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        sx[k] = wfb[k] * sc;
        gy[k] = wfb[k];
      }
    }
  }

  // Collision (BGK + Knudsen sensor + thermal-relaxation correction).
  // `tau` and `tauT` share `1 / (rho T)`, `omega` and `omegaT` one division.
  const real invT = Recip(temc);
  const real tau = LBM_VISCOSITY * invRho * invT + 0.5;
  const real tauT = LBM_COND_OVER_CP * invRho * invT + 0.5;
  const real rTau = Recip(tau * tauT);
  const real omega = tauT * rTau;
  real omegaT = tau * rTau;

  // Non-equilibrium part `d = f - feq`: the sensor, the thermal correction and the BGK update all use it.
  real d[LBM_Q];
#pragma unroll
  for (int q = 0; q < LBM_Q; ++q)
    d[q] = f[q] - a[kCx[q] + 1] * b[kCy[q] + 1];

  // Knudsen sensor `eps = mean_q |d_q| / max(feq_q, tiny)`.
  //
  // While every factor of the product form is well above the clamp (the whole physical range;
  // `a_min * b_min > tiny` is then guaranteed) no `max` is active and
  // `|d_q| / feq_q = |d_q| (1 / a_i)(1 / b_j)`: the six reciprocals come from a single one
  // (Montgomery's trick) and the sum factorises by rows, instead of nine divisions.
  // Otherwise (negative or vanishing populations) the per-direction form of the reference is used.
  const real kFloorFactor = 1.0e-5;
  real epsSum = 0.0;
  const real aMin = (a[0] < a[1] ? a[0] : a[1]) < a[2] ? (a[0] < a[1] ? a[0] : a[1]) : a[2];
  const real bMin = (b[0] < b[1] ? b[0] : b[1]) < b[2] ? (b[0] < b[1] ? b[0] : b[1]) : b[2];
  if (aMin > kFloorFactor && bMin > kFloorFactor) {
    const real pa = (a[0] * a[1]) * a[2];
    const real pb = (b[0] * b[1]) * b[2];
    const real rab = Recip(pa * pb);
    const real rpa = pb * rab; // 1 / pa
    const real rpb = pa * rab; // 1 / pb
    const real ra[3] = { (a[1] * a[2]) * rpa, (a[0] * a[2]) * rpa, (a[0] * a[1]) * rpa };
    const real rb[3] = { (b[1] * b[2]) * rpb, (b[0] * b[2]) * rpb, (b[0] * b[1]) * rpb };
    real row[3] = { 0.0, 0.0, 0.0 };
#pragma unroll
    for (int q = 0; q < LBM_Q; ++q)
      row[kCy[q] + 1] += ra[kCx[q] + 1] * fabs(d[q]);
    epsSum = (rb[0] * row[0] + rb[1] * row[1]) + rb[2] * row[2];
  } else {
#pragma unroll
    for (int q = 0; q < LBM_Q; ++q) {
      const real feq = a[kCx[q] + 1] * b[kCy[q] + 1];
      epsSum += fabs(d[q]) / ((feq > LBM_TINY) ? feq : LBM_TINY);
    }
  }
  const real eps = epsSum * (1.0 / (real)LBM_Q);
  // `omega / sigma` with `sigma` in {omega, 1.35, 1.05, 1}, without a division.
  real omegaL = omega;
  if (eps >= 1.0)
    omegaL = 1.0;
  else if (eps >= 0.1)
    omegaL = omega * (1.0 / 1.35);
  else if (eps >= 0.01)
    omegaL = omega * (1.0 / 1.05);

  omegaL = (omegaL > 1.0) ? omegaL : 1.0;
  omegaL = (omegaL < LBM_OMEGA_MAX) ? omegaL : LBM_OMEGA_MAX;
  omegaT = (omegaT > 1.0) ? omegaT : 1.0;
  omegaT = (omegaT < LBM_OMEGA_MAX) ? omegaT : LBM_OMEGA_MAX;

  // L[a] = sum_i 2 (u . c_i) c_i_a d_i with the unshifted c_i, i.e. from the second moments of d:
  // Lx = 2 (ux Mxx + uy Mxy), Ly = 2 (ux Mxy + uy Myy).
  real mxx = 0.0;
  real myy = 0.0;
  real mxy = 0.0;
#pragma unroll
  for (int q = 0; q < LBM_Q; ++q) {
    if (kCx[q] != 0)
      mxx += d[q];
    if (kCy[q] != 0)
      myy += d[q];
    if (kCx[q] * kCy[q] > 0)
      mxy += d[q];
    else if (kCx[q] * kCy[q] < 0)
      mxy -= d[q];
  }
  // The thermal correction `(omegaL - omegaT) w_q (L . (c_q + U)) / T`, with `(omegaL - omegaT) / T` folded into L.
  const real scaleL = 2.0 * ((omegaL - omegaT) * invT);
  const real lx = scaleL * (ux * mxx + uy * mxy);
  const real ly = scaleL * (ux * mxy + uy * myy);
  real tx[3];
  real ty[3];
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    tx[k] = lx * ((real)(k - 1) + LBM_USHIFT_X);
    ty[k] = ly * ((real)(k - 1) + LBM_USHIFT_Y);
  }
  // Weights: `W_q = wf(c_x) wf(c_y)`, `wf = wn` for a moving and `wz` for a rest component.
  const real wf[3] = { wn, wz, wn };

#pragma unroll
  for (int q = 0; q < LBM_Q; ++q) {
    const int ix = kCx[q] + 1;
    const int iy = kCy[q] + 1;
    f[q] -= omegaL * d[q];
    const real gOld = g[q];
    g[q] = gOld + omegaL * (sx[ix] * gy[iy] - gOld) + (wf[ix] * wf[iy]) * (tx[ix] + ty[iy]);
  }
}

/// Work-item -> cell; `false` for the padding work-items of a rounded-up range.
inline bool CellIndex(int* i, int* j) {
#ifdef LBM_2D
  *j = (int)get_global_id(0);
  *i = (int)get_global_id(1);
  return *j < LBM_NY && *i < LBM_NX;
#else
  const int cell = (int)get_global_id(0);
  *i = cell / LBM_NY;
  *j = cell - *i * LBM_NY;
  return cell < LBM_N;
#endif
}

/**
 * Initial state: Sod left/right macroscopics, equilibrium seeding (`f = feq`,
 * `g` = Wi(T)-normalised energy equilibrium), the macroscopic recovery of the
 * reference `Init()` and the first collision with a cold warm start. Writing
 * every buffer from the device also first-touches the pages on a CPU device.
 */
LBM_KERNEL_ATTR
__kernel void initKernel(
  __global real* restrict fOut,
  __global real* restrict gOut,
  __global real* restrict xiOut,
  __global real* restrict rhoOut,
  __global real* restrict pOut,
  __global real* restrict temOut,
  __global real* restrict uOut
) {
  int i;
  int j;
  if (!CellIndex(&i, &j))
    return;
  const int cell = i * LBM_NY + j;

  const bool left = i < LBM_NX / 2;
  const real rho0 = left ? LBM_DENSITY_L : LBM_DENSITY_R;
  const real p0 = left ? LBM_PRESSURE_L : LBM_PRESSURE_R;
  const real tem0 = LBM_CS2 * p0 / (rho0 * LBM_RGAS);
  const real ux0 = 0.0;
  const real uy0 = 0.0;

  real f[LBM_Q];
  real g[LBM_Q];
  Feq(rho0, ux0, uy0, tem0, f);
  real wz;
  real wn;
  WeightFactors(tem0, &wz, &wn);
  const real u2 = ux0 * ux0 + uy0 * uy0;
  const real E = LBM_CV * tem0 + 0.5 * u2;
  const real targetE = 2.0 * rho0 * E;
  real sumW = 0.0;
#pragma unroll
  for (int q = 0; q < LBM_Q; ++q)
    sumW += Weight(q, wz, wn);
  const real sc = (sumW > LBM_TINY) ? targetE / sumW : targetE / (real)LBM_Q;
#pragma unroll
  for (int q = 0; q < LBM_Q; ++q)
    g[q] = ((sumW > LBM_TINY) ? Weight(q, wz, wn) : 1.0) * sc;

  real rho;
  real invRho;
  real ux;
  real uy;
  real tem;
  real pres;
  Macroscopic(f, g, &rho, &invRho, &ux, &uy, &tem, &pres);
  rhoOut[cell] = rho;
  pOut[cell] = pres;
  temOut[cell] = tem;
  uOut[cell] = ux;
  uOut[LBM_N + cell] = uy;

  real xi[2] = { 0.0, 0.0 };
  Collide(rho, invRho, ux, uy, tem, f, g, xi);

#pragma unroll
  for (int q = 0; q < LBM_Q; ++q) {
    fOut[q * LBM_N + cell] = f[q];
    gOut[q * LBM_N + cell] = g[q];
  }
  xiOut[cell] = xi[0];
  xiOut[LBM_N + cell] = xi[1];
}

/**
 * One time step of one cell. `store` (a literal at both call sites) selects
 * whether the macroscopic fields are written: only the last step of a `Run()`
 * can be observed by a getter.
 */
inline void StepCell(
  __global const real* restrict fIn,
  __global const real* restrict gIn,
  __global real* restrict fOut,
  __global real* restrict gOut,
  __global real* restrict xiIo,
  __global real* restrict rhoOut,
  __global real* restrict pOut,
  __global real* restrict temOut,
  __global real* restrict uOut,
  const bool store
) {
  int i;
  int j;
  if (!CellIndex(&i, &j))
    return;
  const int cell = i * LBM_NY + j;

  // Pull the post-collision populations of the previous step: columns clamped
  // to [0, nx), rows wrapped, nearest-node fallback self-copying outside the
  // tube (`lbmini::GatherSources`). Every condition on the stencil is a
  // compile-time constant per unrolled direction.
  real f[LBM_Q];
  real g[LBM_Q];
#pragma unroll
  for (int q = 0; q < LBM_Q; ++q) {
    __global const real* restrict fp = fIn + q * LBM_N;
    __global const real* restrict gp = gIn + q * LBM_N;
    if (kStNearest[q]) {
      const int xN = (int)round((real)i - kStCix[q]);
      const int yN = (int)round((real)j - kStCiy[q]);
      int yW = yN % LBM_NY;
      yW += (yW < 0) ? LBM_NY : 0;
      const int src = (xN < 0 || xN >= LBM_NX) ? cell : xN * LBM_NY + yW;
      f[q] = fp[src];
      g[q] = gp[src];
    } else {
      const int xa = i + kStOx[q];
      const int rowA = ((xa < 0) ? 0 : ((xa >= LBM_NX) ? LBM_NX - 1 : xa)) * LBM_NY;
      const int xb = xa + 1;
      const int rowB = ((xb < 0) ? 0 : ((xb >= LBM_NX) ? LBM_NX - 1 : xb)) * LBM_NY;
      int ya = j + kStOy0[q];
      ya -= (ya >= LBM_NY) ? LBM_NY : 0;
      int yb = j + kStOy1[q];
      yb -= (yb >= LBM_NY) ? LBM_NY : 0;

      // Zero-weight corners (all but one for a snapped direction) are dropped
      // at compile time; the reference adds `0 * f`, which differs only for
      // non-finite populations, where the run is already lost.
      real fs = 0.0;
      real gs = 0.0;
      bool first = true;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const real w = kStW[q * 4 + k];
        if (w != 0.0) {
          const int src = ((k & 1) ? rowB : rowA) + ((k & 2) ? yb : ya);
          fs = first ? w * fp[src] : fs + w * fp[src];
          gs = first ? w * gp[src] : gs + w * gp[src];
          first = false;
        }
      }
      f[q] = fs;
      g[q] = gs;
    }
  }

  real rho;
  real invRho;
  real ux;
  real uy;
  real tem;
  real pres;
  Macroscopic(f, g, &rho, &invRho, &ux, &uy, &tem, &pres);
  if (store) {
    rhoOut[cell] = rho;
    pOut[cell] = pres;
    temOut[cell] = tem;
    uOut[cell] = ux;
    uOut[LBM_N + cell] = uy;
  }

  real xi[2];
  xi[0] = xiIo[cell];
  xi[1] = xiIo[LBM_N + cell];

  Collide(rho, invRho, ux, uy, tem, f, g, xi);

#pragma unroll
  for (int q = 0; q < LBM_Q; ++q) {
    fOut[q * LBM_N + cell] = f[q];
    gOut[q * LBM_N + cell] = g[q];
  }
  xiIo[cell] = xi[0];
  xiIo[LBM_N + cell] = xi[1];
}

/// Time step without macroscopic output (every step of a `Run()` but the last).
LBM_KERNEL_ATTR
__kernel void stepKernel(
  __global const real* restrict fIn,
  __global const real* restrict gIn,
  __global real* restrict fOut,
  __global real* restrict gOut,
  __global real* restrict xiIo,
  __global real* restrict rhoOut,
  __global real* restrict pOut,
  __global real* restrict temOut,
  __global real* restrict uOut
) {
  StepCell(fIn, gIn, fOut, gOut, xiIo, rhoOut, pOut, temOut, uOut, false);
}

/// Time step that also stores rho, p, T, u (last step of a `Run()`).
LBM_KERNEL_ATTR
__kernel void stepStoreKernel(
  __global const real* restrict fIn,
  __global const real* restrict gIn,
  __global real* restrict fOut,
  __global real* restrict gOut,
  __global real* restrict xiIo,
  __global real* restrict rhoOut,
  __global real* restrict pOut,
  __global real* restrict temOut,
  __global real* restrict uOut
) {
  StepCell(fIn, gIn, fOut, gOut, xiIo, rhoOut, pOut, temOut, uOut, true);
}
