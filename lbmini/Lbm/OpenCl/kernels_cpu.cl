/**
 * @file kernels_cpu.cl
 * @brief Vectorised fused D2Q9 compressible-LBM kernels of `opencl::cpu::LbmTube`.
 *
 * OpenCL C image of `Lbm/HostKernel.hpp` (`host::Kernel`): same layout, same
 * stencil logic, same arithmetic per cell, including the factorised energy
 * closure (2 exponentials per Newton evaluation, see `AxisExp`). The unit of work
 * is not a cell but a *block* of `LBM_VW` neighbouring cells of one row, held in
 * explicit OpenCL vector types, so the per-cell physics (about 850 flops and 8
 * exponentials) runs in SIMD lanes. The compiler of an OpenCL CPU runtime cannot
 * get there on its own: across work-items the fused scalar kernel has
 * data-dependent control flow, divisions and gathers that defeat every
 * work-group vectoriser (~4 % of the arithmetic peak of a core).
 *
 * The source is not self-contained: the host (`Lbm/OpenCl/Cpu/Engine.hpp`)
 * prepends the `#define`s of `Lbm/OpenCl/Program.hpp` (fluid constants, lattice
 * shift, per-direction stencil as `__constant` tables, folded to immediates by
 * the unrolled loops) and the geometry of the CPU path:
 *
 *  - `LBM_VW`      cells per block (a power of two, 2 ... 16), the vector length;
 *  - `LBM_PITCH`   padded row length (a multiple of 16 >= `LBM_VW`), `LBM_PLANE` the stride
 *                  between populations; `cell = j * LBM_PITCH + i`, `f[q * LBM_PLANE + cell]`;
 *  - `LBM_BLOCKS`  blocks per row, `LBM_CHUNK` blocks per row segment, `LBM_SEGMENTS`
 *                  segments in all (`chunk * ny` of them);
 *  - `LBM_INTERIOR_LO`, `LBM_INTERIOR_HI`  first/last block start whose gather window
 *                  stays inside the tube, `LBM_ANY_NEAREST` 1 if a direction needs the
 *                  nearest-node fallback (then every block takes the lane-by-lane path);
 *  - `LBM_STREAM`  1 to write the post-collision populations with non-temporal stores
 *                  (`host::UseStreamingStores`: a working set much larger than the last-level
 *                  cache, where an ordinary store first reads the line: 464 instead of 320 B per
 *                  cell and step). The host enables it only when the buffers are aligned to
 *                  one vector (`CL_DEVICE_MEM_BASE_ADDR_ALIGN`), which `cell`, `LBM_PLANE`
 *                  and `LBM_PITCH` preserve.
 *
 * The kernels are launched with one work-group of one work-item per worker thread
 * of the device, and the work-items deal the row segments out to themselves
 * through an atomic queue (see `stepKernel`). Segment `s` is row `s % ny` of column
 * chunk `s / ny`: consecutive segments are consecutive rows of the same chunk, so
 * the five rows a segment gathers from are still in cache when the next row needs them.
 *
 * The program is never built with relaxed-math options: divisions, NaN
 * propagation and the lane selects keep their IEEE semantics, which the equivalence
 * test against `lbmini::plain::LbmTube` (relative 1e-9) depends on. The kernels
 * only use fused multiply-adds where the compiler contracts `a * b + c`.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define LBM_CAT_(a, b) a##b
#define LBM_CAT(a, b) LBM_CAT_(a, b)

typedef LBM_CAT(double, LBM_VW) real;
/// Result type of a comparison of two `real`: all-ones / all-zeros 64-bit lanes.
typedef LBM_CAT(long, LBM_VW) mask;

#define VLOAD LBM_CAT(vload, LBM_VW)
#define VSTORE LBM_CAT(vstore, LBM_VW)
#define AS_REAL LBM_CAT(as_double, LBM_VW)
#define AS_MASK LBM_CAT(as_long, LBM_VW)

/// Store of a population block: non-temporal when `LBM_STREAM` (the pointer is aligned to a vector).
#if LBM_STREAM
#define STORE_POP(v, ptr) __builtin_nontemporal_store((v), (__global real*)(ptr))
#else
#define STORE_POP(v, ptr) VSTORE((v), 0, (ptr))
#endif

#define LBM_INLINE inline __attribute__((always_inline))

#define LBM_TINY 1.0e-12
#define LBM_MAX_EXP 700.0
#define LBM_OMEGA_MAX (2.0 - 1.0e-7)

__constant int kCx[LBM_Q] = LBM_CX;
__constant int kCy[LBM_Q] = LBM_CY;
__constant int kStOx[LBM_Q] = LBM_ST_OX;
__constant int kStOy0[LBM_Q] = LBM_ST_OY0;
__constant int kStOy1[LBM_Q] = LBM_ST_OY1;
__constant int kStNearest[LBM_Q] = LBM_ST_NEAREST;
__constant double kStW[LBM_Q * 4] = LBM_ST_W;
__constant double kStCix[LBM_Q] = LBM_ST_CIX;
__constant double kStCiy[LBM_Q] = LBM_ST_CIY;

/// Shifted lattice velocity `c_i + U`, a constant per direction.
#define CSX(q) ((double)kCx[q] + LBM_USHIFT_X)
#define CSY(q) ((double)kCy[q] + LBM_USHIFT_Y)

#define R(x) ((real)(x))

LBM_INLINE real Sel(const mask m, const real a, const real b) {
  return select(b, a, m);
}

LBM_INLINE real VMax(const real a, const real b) {
  return select(b, a, a > b);
}

/**
 * Vector `exp`, within 2 ulp of `std::exp` over the solver's range `[-700, 700]`
 * (the arguments are clamped to it): `x = n ln2 + r`, `|r| <= ln2 / 2`, Taylor
 * polynomial of degree 13 for `exp(r)`, and `2^n` written straight into the
 * exponent field. The builtin `exp` (a similar polynomial) measured ~9 % slower
 * in the fused kernel.
 *
 * `n = round(x log2 e)` comes from the "add 1.5 * 2^52" trick: the rounded
 * integer then sits in the low mantissa bits of `t`, which also provides the
 * exponent field of `2^n` after a bias and a shift (`n + 1023 < 4096`, so only
 * the low 12 bits matter). `ln2` is split in a 32-bit head, so `n * hi` is exact.
 */
LBM_INLINE real VExp(const real x) {
  const real t = x * 1.44269504088896338700e+00 + 6755399441055744.0;
  const real n = t - 6755399441055744.0;
  real r = x - n * 6.93147180369123816490e-01;
  r = r - n * 1.90821492927058770002e-10;
  real p = R(1.0 / 6227020800.0);
  p = p * r + 1.0 / 479001600.0;
  p = p * r + 1.0 / 39916800.0;
  p = p * r + 1.0 / 3628800.0;
  p = p * r + 1.0 / 362880.0;
  p = p * r + 1.0 / 40320.0;
  p = p * r + 1.0 / 5040.0;
  p = p * r + 1.0 / 720.0;
  p = p * r + 1.0 / 120.0;
  p = p * r + 1.0 / 24.0;
  p = p * r + 1.0 / 6.0;
  p = p * r + 1.0 / 2.0;
  p = p * r + 1.0;
  p = p * r + 1.0;
  const mask bits = (AS_MASK(t) + 1023) << 52;
  return p * AS_REAL(bits);
}

/// `Weight(q) = Wx * Wy`, `W = wZero` for a zero velocity component, `wNonZero` otherwise.
LBM_INLINE void Weights(const real temc, real w[LBM_Q]) {
  const real wZero = 1.0 - temc;
  const real wNonZero = 0.5 * temc;
  const real wn = wZero + (wNonZero - wZero);
#pragma unroll
  for (int q = 0; q < LBM_Q; ++q)
    w[q] = (kCx[q] != 0 ? wn : wZero) * (kCy[q] != 0 ? wn : wZero);
}

/// The factors of the product form of the density equilibrium: `feq(q) = a[c_x + 1] * b[c_y + 1]`,
/// `a = rho * px`, `b = py` (`c = -1, 0, +1`).
LBM_INLINE void FeqFactors(const real rhoc, const real ux, const real uy, const real temc, real a[3], real b[3]) {
  const real uax = ux - LBM_USHIFT_X;
  const real uay = uy - LBM_USHIFT_Y;
  const real tx = uax * uax + temc;
  const real ty = uay * uay + temc;
  a[0] = rhoc * (0.5 * ((-uax) + tx));
  a[1] = rhoc * (1.0 - tx);
  a[2] = rhoc * (0.5 * (uax + tx));
  b[0] = 0.5 * ((-uay) + ty);
  b[1] = 1.0 - ty;
  b[2] = 0.5 * (uay + ty);
}

LBM_INLINE void Feq(const real rhoc, const real ux, const real uy, const real temc, real feq[LBM_Q]) {
  real a[3];
  real b[3];
  FeqFactors(rhoc, ux, uy, temc, a, b);
#pragma unroll
  for (int q = 0; q < LBM_Q; ++q)
    feq[q] = a[kCx[q] + 1] * b[kCy[q] + 1];
}

LBM_INLINE void Macroscopic(
  const real f[LBM_Q],
  const real g[LBM_Q],
  real* rho,
  real* invRho,
  real* ux,
  real* uy,
  real* tem,
  real* pres
) {
  real rhoSum = R(0.0);
  real nrg = R(0.0);
  real momX = R(0.0);
  real momY = R(0.0);
#pragma unroll
  for (int q = 0; q < LBM_Q; ++q) {
    rhoSum += f[q];
    nrg += g[q];
    momX += CSX(q) * f[q];
    momY += CSY(q) * f[q];
  }
  const real rinv = 1.0 / rhoSum;
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
 * sums of the Newton step become three-term ones.
 *
 * It equals the reference to rounding error while neither of its clamps
 * (`|s - smax| <= 700`) is active; beyond that every omitted term is below
 * `exp(-700)` relative to the largest one. A NaN `xi` takes the clamp.
 */
LBM_INLINE void AxisExp(const real xi, const real wn, const real wz, real p[3]) {
  const real e = VExp(VMax(-fabs(xi), R(-LBM_MAX_EXP)));
  const real e2 = e * e;
  const mask pos = (xi >= 0.0);
  p[0] = wn * Sel(pos, e2, R(1.0));
  p[1] = wz * e;
  p[2] = wn * Sel(pos, R(1.0), e2);
}

/// Energy equilibrium from the Newton solution `xi` as the factors of the product form:
/// `geq(q) = sx[c_x + 1] * gy[c_y + 1]`.
LBM_INLINE void GeqFactors(const real xi[2], const real wn, const real wz, const real targetE, real sx[3], real gy[3]) {
  real px[3];
  AxisExp(xi[0], wn, wz, px);
  AxisExp(xi[1], wn, wz, gy);
  const real scale = targetE / ((px[0] + px[1] + px[2]) * (gy[0] + gy[1] + gy[2]));
#pragma unroll
  for (int k = 0; k < 3; ++k)
    sx[k] = scale * px[k];
}

/// The reference's fallback (normalised weights, uniform when they vanish) in the same factored form:
/// `W_q = wf(c_x) wf(c_y)`, `wf = wn` for a moving and `wz` for a rest component, summing to `(2 wn + wz)^2`.
LBM_INLINE void GeqFallbackFactors(const real wn, const real wz, const real targetE, real sx[3], real gy[3]) {
  const real wf[3] = { wn, wz, wn };
  const real sumAxis = wn + wz + wn;
  const real sumW = sumAxis * sumAxis;
  const real uni = targetE / R((double)LBM_Q);
  const real sc = targetE / sumW;
  const mask degenerate = (sumW <= LBM_TINY);
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    sx[k] = Sel(degenerate, uni, wf[k] * sc);
    gy[k] = Sel(degenerate, R(1.0), wf[k]);
  }
}

/**
 * Equilibria, Newton-Raphson energy closure and BGK/Knudsen collision, in place.
 *
 * `f`/`g` enter pre-collision and leave post-collision; `xi` is the Newton warm
 * start (read and updated). A fixed number of Newton iterations with clamped
 * reciprocals and a sticky per-lane `ok` predicate, as in the reference; lanes
 * that are not `ok` restart from `xi = 0` with the fallback energy equilibrium.
 */
LBM_INLINE void Collide(
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
  // The per-axis factors of the weights (`W_q = Wx(c_x) Wy(c_y)`), see `AxisExp`.
  const real wz = 1.0 - temc;
  const real wn = wz + (0.5 * temc - wz);

  // The density equilibrium in product form, `feq(q) = a[c_x + 1] * b[c_y + 1]`: only six vectors
  // are kept and the products are formed where they are used.
  real a[3];
  real b[3];
  FeqFactors(rhoc, ux, uy, temc, a, b);

  // Geq via Newton-Raphson on the normalised energy flux.
  //
  // The closure distribution `W_q exp(xi . (c_q + U))` is a product of an x and a
  // y factor, so x and y are independent: the Jacobian of the reference is
  // diagonal (its off-diagonal term cancels exactly, `S2xy / Z - (S1x / Z) (S1y / Z) = 0`)
  // and the two moments of the iteration come from three weighted
  // exponentials per axis. Each Newton step is then two scalar steps,
  // `xi_a -= alpha (m1_a - target_a) / var_a`, with 2 exponentials per
  // iteration instead of 9, and without the 9-term sums (see `AxisExp`).
  const real E = LBM_CV * temc + 0.5 * u2;
  const real targetE = 2.0 * rhoc * E;
  // `2 rho u (E + T) / (2 rho E)`: one reciprocal for both components.
  const real invE = 1.0 / E;
  const real targetMx = ux * (E + temc) * invE;
  const real targetMy = uy * (E + temc) * invE;

  real alpha = R(1.0);
  mask ok = (mask)(-1);
#pragma nounroll
  for (int iter = 0; iter < LBM_MAX_ITER; ++iter) {
    real px[3];
    real py[3];
    AxisExp(xi[0], wn, wz, px);
    AxisExp(xi[1], wn, wz, py);
    const real zx = px[0] + px[1] + px[2];
    const real zy = py[0] + py[1] + py[2];
    const real zxy = zx * zy;
    ok = ok & (zxy > LBM_TINY);
    // Scale residual and regularised variance by Z^2, cancelling 1/Z.
    const real dx = px[2] - px[0];
    const real dy = py[2] - py[0];
    const real jxx = (px[2] + px[0]) * zx - dx * dx + LBM_TINY * (zx * zx);
    const real jyy = (py[2] + py[0]) * zy - dy * dy + LBM_TINY * (zy * zy);
    const real jxy = jxx * jyy;
    ok = ok & (fabs(jxy) > LBM_TINY * (zxy * zxy));

    const real rj = alpha * (1.0 / jxy);
    const real r0 = (dx + (LBM_USHIFT_X - targetMx) * zx) * zx;
    const real r1 = (dy + (LBM_USHIFT_Y - targetMy) * zy) * zy;
    xi[0] -= r0 * jyy * rj;
    xi[1] -= r1 * jxx * rj;
    alpha *= 0.5;
  }

  // Energy equilibrium `geq(q) = sx[c_x + 1] * gy[c_y + 1]`.
  // All lanes ok: only the exponential closure runs; none: only the fallback; mixed: both, selected per lane.
  real sx[3];
  real gy[3];
  if (any(ok)) {
    GeqFactors(xi, wn, wz, targetE, sx, gy);
  } else {
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      sx[k] = R(0.0);
      gy[k] = R(0.0);
    }
  }
  if (!all(ok)) {
    real sxFb[3];
    real gyFb[3];
    GeqFallbackFactors(wn, wz, targetE, sxFb, gyFb);
    xi[0] = Sel(ok, xi[0], R(0.0));
    xi[1] = Sel(ok, xi[1], R(0.0));
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      sx[k] = Sel(ok, sx[k], sxFb[k]);
      gy[k] = Sel(ok, gy[k], gyFb[k]);
    }
  }

  // Collision (BGK + Knudsen sensor + thermal-relaxation correction).
  //
  // `tau = mu / (rho T) + 1/2` and `tauT = kappa / (rho cp T) + 1/2` share `1 / (rho T)`;
  // `omega = 1 / tau` and `omegaT = 1 / tauT` share one division.
  const real invT = 1.0 / temc;
  const real tau = LBM_VISCOSITY * invRho * invT + 0.5;
  const real tauT = LBM_COND_OVER_CP * invRho * invT + 0.5;
  const real rTau = 1.0 / (tau * tauT);
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
  // `|d_q| / feq_q = |d_q| (1 / a_i)(1 / b_j)`: the six reciprocals come from a single division
  // (Montgomery's trick) and the sum factorises by rows, instead of nine divisions.
  // Otherwise (negative or vanishing populations) the per-direction form of the reference is used.
  const real floorFactor = R(1.0e-5);
  const real aMin = fmin(fmin(a[0], a[1]), a[2]);
  const real bMin = fmin(fmin(b[0], b[1]), b[2]);
  real epsSum = R(0.0);
  if (all((aMin > floorFactor) & (bMin > floorFactor))) {
    const real pa = (a[0] * a[1]) * a[2];
    const real pb = (b[0] * b[1]) * b[2];
    const real rab = 1.0 / (pa * pb);
    const real rpa = pb * rab; // 1 / pa
    const real rpb = pa * rab; // 1 / pb
    const real ra[3] = { (a[1] * a[2]) * rpa, (a[0] * a[2]) * rpa, (a[0] * a[1]) * rpa };
    const real rb[3] = { (b[1] * b[2]) * rpb, (b[0] * b[2]) * rpb, (b[0] * b[1]) * rpb };
    real row[3] = { R(0.0), R(0.0), R(0.0) };
#pragma unroll
    for (int q = 0; q < LBM_Q; ++q)
      row[kCy[q] + 1] += ra[kCx[q] + 1] * fabs(d[q]);
    epsSum = (rb[0] * row[0] + rb[1] * row[1]) + rb[2] * row[2];
  } else {
#pragma unroll
    for (int q = 0; q < LBM_Q; ++q) {
      const real feq = a[kCx[q] + 1] * b[kCy[q] + 1];
      epsSum += fabs(d[q]) / Sel(feq > LBM_TINY, feq, R(LBM_TINY));
    }
  }
  const real eps = epsSum * (1.0 / (double)LBM_Q);
  // `omega / sigma` with `sigma` in {omega, 1.35, 1.05, 1}, without a division.
  real omegaL = Sel(
    eps >= 1.0,
    R(1.0),
    Sel(eps >= 0.1, omega * (1.0 / 1.35), Sel(eps >= 0.01, omega * (1.0 / 1.05), omega))
  );

  omegaL = Sel(omegaL > 1.0, omegaL, R(1.0));
  omegaL = Sel(omegaL < LBM_OMEGA_MAX, omegaL, R(LBM_OMEGA_MAX));
  omegaT = Sel(omegaT > 1.0, omegaT, R(1.0));
  omegaT = Sel(omegaT < LBM_OMEGA_MAX, omegaT, R(LBM_OMEGA_MAX));

  // L[a] = sum_i 2 (u . c_i) c_i_a d_i with the unshifted c_i, i.e. from the second moments of d:
  // Lx = 2 (ux Mxx + uy Mxy), Ly = 2 (ux Mxy + uy Myy).
  real mxx = R(0.0);
  real myy = R(0.0);
  real mxy = R(0.0);
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
  const real dOmega = omegaL - omegaT;
  // The thermal correction `dOmega w_q (L . (c_q + U)) / T`, with `dOmega / T` folded into L.
  const real scaleL = 2.0 * (dOmega * invT);
  const real lx = scaleL * (ux * mxx + uy * mxy);
  const real ly = scaleL * (ux * mxy + uy * myy);
  real tx[3];
  real ty[3];
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    tx[k] = lx * ((double)(k - 1) + LBM_USHIFT_X);
    ty[k] = ly * ((double)(k - 1) + LBM_USHIFT_Y);
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

/// `LBM_VW` source columns `[x, x + LBM_VW)` of one row, each clamped into `[0, nx)`.
/// Only the first and last block of a row come here: a rolled loop keeps the code small.
LBM_INLINE real LoadClamped(__global const double* row, const long x) {
  double tmp[LBM_VW];
#pragma nounroll
  for (int l = 0; l < LBM_VW; ++l) {
    long xs = x + l;
    xs = (xs < 0) ? 0 : ((xs > LBM_NX - 1) ? LBM_NX - 1 : xs);
    tmp[l] = row[xs];
  }
  return VLOAD(0, tmp);
}

/// Nearest-node fallback of the gather (`lbmini::GatherSources`), lane by lane.
LBM_INLINE void GatherNearest(
  __global const double* fIn,
  __global const double* gIn,
  const long j,
  const long i0,
  const int q,
  real* f,
  real* g
) {
  double tf[LBM_VW];
  double tg[LBM_VW];
#pragma nounroll
  for (int l = 0; l < LBM_VW; ++l) {
    const long i = i0 + l;
    const long xN = (long)round((double)i - kStCix[q]);
    const long yN = (long)round((double)j - kStCiy[q]);
    long yW = yN % LBM_NY;
    yW += (yW < 0) ? LBM_NY : 0;
    const long src = (xN < 0 || xN > LBM_NX - 1) ? j * LBM_PITCH + i : yW * LBM_PITCH + xN;
    tf[l] = fIn[(long)q * LBM_PLANE + src];
    tg[l] = gIn[(long)q * LBM_PLANE + src];
  }
  *f = VLOAD(0, tf);
  *g = VLOAD(0, tg);
}

/**
 * Pulls the post-collision populations of the cells `[i0, i0 + LBM_VW)` of row `j`:
 * four inverse-distance corners per direction, two wrapped source rows, columns
 * clamped into the tube. `interior` (a literal at both call sites) says that
 * the whole gather window of the block is inside the tube, which turns the lane
 * clamp into plain unaligned vector loads. Zero-weight corners (all but one for
 * a snapped direction) are dropped at compile time, so is the nearest-node
 * branch of a regular direction.
 */
LBM_INLINE void Pull(
  __global const double* fIn,
  __global const double* gIn,
  const long j,
  const long i0,
  const int interior,
  real f[LBM_Q],
  real g[LBM_Q]
) {
#pragma unroll
  for (int q = 0; q < LBM_Q; ++q) {
    if (kStNearest[q]) {
      GatherNearest(fIn, gIn, j, i0, q, &f[q], &g[q]);
    } else {
      long ya = j + kStOy0[q];
      ya -= (ya >= LBM_NY) ? LBM_NY : 0;
      long yb = j + kStOy1[q];
      yb -= (yb >= LBM_NY) ? LBM_NY : 0;
      const long rowA = (long)q * LBM_PLANE + ya * LBM_PITCH;
      const long rowB = (long)q * LBM_PLANE + yb * LBM_PITCH;
      const long x0 = i0 + kStOx[q];

      real fs = R(0.0);
      real gs = R(0.0);
      int first = 1;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const double w = kStW[q * 4 + k];
        if (w != 0.0) {
          const long row = (k & 2) ? rowB : rowA;
          const long x = x0 + (k & 1);
          const real fv = interior ? VLOAD(0, fIn + row + x) : LoadClamped(fIn + row, x);
          const real gv = interior ? VLOAD(0, gIn + row + x) : LoadClamped(gIn + row, x);
          fs = first ? w * fv : fs + w * fv;
          gs = first ? w * gv : gs + w * gv;
          first = 0;
        }
      }
      f[q] = fs;
      g[q] = gs;
    }
  }
}

/**
 * Initial state of one block: Sod left/right macroscopics, equilibrium seeding
 * (`f = feq`, `g` = Wi(T)-normalised energy equilibrium), the macroscopic recovery
 * of the reference `Init()` and the first collision with a cold warm start.
 * Pad columns take the right state. Writing every buffer from the device also
 * first-touches the pages on the cores that will update them.
 */
LBM_INLINE void InitBlock(
  __global double* restrict fOut,
  __global double* restrict gOut,
  __global double* restrict xiOut,
  __global double* restrict rhoOut,
  __global double* restrict pOut,
  __global double* restrict temOut,
  __global double* restrict uOut,
  const long j,
  const long i0
) {
  double rl[LBM_VW];
  double pl[LBM_VW];
#pragma unroll
  for (int l = 0; l < LBM_VW; ++l) {
    const long gi = (i0 + l < LBM_NX - 1) ? i0 + l : LBM_NX - 1;
    const int left = gi < LBM_NX / 2;
    rl[l] = left ? LBM_DENSITY_L : LBM_DENSITY_R;
    pl[l] = left ? LBM_PRESSURE_L : LBM_PRESSURE_R;
  }
  const real rho0 = VLOAD(0, rl);
  const real p0 = VLOAD(0, pl);
  const real tem0 = LBM_CS2 * p0 / (rho0 * LBM_RGAS);
  const real zero = R(0.0);

  real f[LBM_Q];
  real g[LBM_Q];
  Feq(rho0, zero, zero, tem0, f);
  real w[LBM_Q];
  Weights(tem0, w);
  const real E = LBM_CV * tem0 + 0.5 * zero;
  const real targetE = 2.0 * rho0 * E;
  real sumW = zero;
#pragma unroll
  for (int q = 0; q < LBM_Q; ++q)
    sumW += w[q];
  const mask useW = (sumW > LBM_TINY);
  const real sc = Sel(useW, targetE / sumW, targetE / R((double)LBM_Q));
#pragma unroll
  for (int q = 0; q < LBM_Q; ++q)
    g[q] = Sel(useW, w[q], R(1.0)) * sc;

  real rho;
  real invRho;
  real ux;
  real uy;
  real tem;
  real pres;
  Macroscopic(f, g, &rho, &invRho, &ux, &uy, &tem, &pres);
  const long cell = j * LBM_PITCH + i0;
  VSTORE(rho, 0, rhoOut + cell);
  VSTORE(pres, 0, pOut + cell);
  VSTORE(tem, 0, temOut + cell);
  VSTORE(ux, 0, uOut + cell);
  VSTORE(uy, 0, uOut + LBM_PLANE + cell);

  real xi[2] = { zero, zero };
  Collide(rho, invRho, ux, uy, tem, f, g, xi);

#pragma unroll
  for (int q = 0; q < LBM_Q; ++q) {
    VSTORE(f[q], 0, fOut + (long)q * LBM_PLANE + cell);
    VSTORE(g[q], 0, gOut + (long)q * LBM_PLANE + cell);
  }
  VSTORE(xi[0], 0, xiOut + cell);
  VSTORE(xi[1], 0, xiOut + LBM_PLANE + cell);
}

/**
 * One time step of the block `[i0, i0 + LBM_VW)` of row `j`. `store` selects
 * whether the macroscopic fields are written: only the last step of a `Run()`
 * can be observed by a getter.
 */
LBM_INLINE void StepBlock(
  __global const double* restrict fIn,
  __global const double* restrict gIn,
  __global double* restrict fOut,
  __global double* restrict gOut,
  __global double* restrict xiIo,
  __global double* restrict rhoOut,
  __global double* restrict pOut,
  __global double* restrict temOut,
  __global double* restrict uOut,
  const long j,
  const long i0,
  const int store
) {
  real f[LBM_Q];
  real g[LBM_Q];
  if (!LBM_ANY_NEAREST && i0 >= LBM_INTERIOR_LO && i0 <= LBM_INTERIOR_HI)
    Pull(fIn, gIn, j, i0, 1, f, g);
  else
    Pull(fIn, gIn, j, i0, 0, f, g);

  real rho;
  real invRho;
  real ux;
  real uy;
  real tem;
  real pres;
  Macroscopic(f, g, &rho, &invRho, &ux, &uy, &tem, &pres);
  const long cell = j * LBM_PITCH + i0;
  if (store) {
    VSTORE(rho, 0, rhoOut + cell);
    VSTORE(pres, 0, pOut + cell);
    VSTORE(tem, 0, temOut + cell);
    VSTORE(ux, 0, uOut + cell);
    VSTORE(uy, 0, uOut + LBM_PLANE + cell);
  }

  real xi[2] = { VLOAD(0, xiIo + cell), VLOAD(0, xiIo + LBM_PLANE + cell) };
  Collide(rho, invRho, ux, uy, tem, f, g, xi);

#pragma unroll
  for (int q = 0; q < LBM_Q; ++q) {
    STORE_POP(f[q], fOut + (long)q * LBM_PLANE + cell);
    STORE_POP(g[q], gOut + (long)q * LBM_PLANE + cell);
  }
  VSTORE(xi[0], 0, xiIo + cell);
  VSTORE(xi[1], 0, xiIo + LBM_PLANE + cell);
}

/**
 * Row segment `seg` = `LBM_CHUNK` consecutive blocks `[b0, b1)` of row `j`:
 * `chunk = seg / ny`, `j = seg % ny`, so consecutive segments are consecutive rows
 * of the same column chunk. The tail chunk of a row is shorter.
 */
LBM_INLINE void Segment(const int seg, long* j, long* b0, long* b1) {
  const long chunk = seg / LBM_NY;
  *j = seg - chunk * LBM_NY;
  *b0 = chunk * LBM_CHUNK;
  *b1 = (*b0 + LBM_CHUNK < LBM_BLOCKS) ? *b0 + LBM_CHUNK : LBM_BLOCKS;
}

/**
 * Initial state of the whole mesh. Like the time step, the kernel is launched with one
 * work-group of one work-item per worker thread of the device, and the work-items
 * deal the row segments out to themselves through the atomic queue `counter`
 * (see `stepKernel`); `nextCounter` is zeroed for the launch that follows.
 */
__kernel void initKernel(
  __global double* restrict fOut,
  __global double* restrict gOut,
  __global double* restrict xiOut,
  __global double* restrict rhoOut,
  __global double* restrict pOut,
  __global double* restrict temOut,
  __global double* restrict uOut,
  volatile __global int* counter,
  __global int* nextCounter
) {
  if (get_group_id(0) == 0)
    *nextCounter = 0;
  for (int seg = atomic_inc(counter); seg < LBM_SEGMENTS; seg = atomic_inc(counter)) {
    long j;
    long b0;
    long b1;
    Segment(seg, &j, &b0, &b1);
    for (long b = b0; b < b1; ++b)
      InitBlock(fOut, gOut, xiOut, rhoOut, pOut, temOut, uOut, j, b * LBM_VW);
  }
}

/**
 * One time step of the whole mesh, with dynamic self-scheduling.
 *
 * The host launches one work-group of one work-item per worker thread of the device,
 * and each work-item loops: it takes the next row segment from the atomic queue
 * `counter` (an `atomic_inc` per ~256 cells) until none is left. The runtime's own
 * work-group scheduler deals out chunks of `remaining / threads` work-groups, which
 * is a static schedule in disguise for a launch of a few hundred short work-groups
 * and leaves the other threads idle whenever one core is slower (an E-core, an SMT
 * sibling, a time-sliced vCPU); dealing out single segments balances to ~1 segment.
 *
 * Two queues alternate between launches: `counter` is drained by this launch,
 * `nextCounter` (drained by the next launch, which only starts after this one has
 * finished: the queue is in-order) is zeroed here, so no host command sits between
 * two steps. `store` selects whether the macroscopic fields are written.
 */
__kernel void stepKernel(
  __global const double* restrict fIn,
  __global const double* restrict gIn,
  __global double* restrict fOut,
  __global double* restrict gOut,
  __global double* restrict xiIo,
  __global double* restrict rhoOut,
  __global double* restrict pOut,
  __global double* restrict temOut,
  __global double* restrict uOut,
  volatile __global int* counter,
  __global int* nextCounter,
  const int store
) {
  if (get_group_id(0) == 0)
    *nextCounter = 0;
  for (int seg = atomic_inc(counter); seg < LBM_SEGMENTS; seg = atomic_inc(counter)) {
    long j;
    long b0;
    long b1;
    Segment(seg, &j, &b0, &b1);
    for (long b = b0; b < b1; ++b)
      StepBlock(fIn, gIn, fOut, gOut, xiIo, rhoOut, pOut, temOut, uOut, j, b * LBM_VW, store);
  }
#if LBM_STREAM
  // Non-temporal stores are weakly ordered: make them visible before the work-item ends (the next
  // launch reads them).
#if defined(__x86_64__)
  __builtin_ia32_sfence();
#else
  mem_fence(CLK_GLOBAL_MEM_FENCE);
#endif
#endif
}
