#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

__constant int kD2Q9Cx[9] = { 0, 1, -1, 0, 0, 1, -1, 1, -1 };
__constant int kD2Q9Cy[9] = { 0, 0, 0, 1, -1, 1, -1, -1, 1 };

inline double polyT(const double x, const double Tem, const double cv) {
    const double a0 = 2.0 * cv * cv * Tem * Tem;
    const double a1 = cv * Tem;
    return a0 + a1 * x + 0.5 * x * x;
}

inline double dpolyT(const double x, const double Tem, const double cv) {
    return cv * Tem + x;
}

inline double idwValue(const double frac, const double C) {
    const double eps = 1.0e-12;
    if (frac < eps) return 1.0;
    return 1.0 / pow(frac, C);
}

__kernel void computeMacroscopicKernel(
  __global const double* restrict pF,
  __global const double* restrict pG,
  __global double* restrict pRho,
  __global double* restrict pP,
  __global double* restrict pT,
  __global double* restrict pU,
  const double Ushift0,
  const double Ushift1,
  const double invCv,
  const double Rgas,
  int nx,
  int ny,
  int N
) {
  int cell = get_global_id(0);
  if (cell < N) {
    double rho = (double)(0);
    double nrg = (double)(0);
    double mx = (double)(0);
    double my = (double)(0);
    
    for (int idc = 0; idc < 9; ++idc) {
      const double fi = pF[idc * N + cell];
      const double gi = pG[idc * N + cell];
      rho += fi;
      nrg += gi;
      mx += ((double)(kD2Q9Cx[idc]) + Ushift0) * fi;
      my += ((double)(kD2Q9Cy[idc]) + Ushift1) * fi;
    }

    pRho[cell] = rho;

    const double invRho = (double)(1) / rho;
    const double ux = mx * invRho;
    const double uy = my * invRho;
    pU[0 * N + cell] = ux;
    pU[1 * N + cell] = uy;

    const double kin = (double)(0.5) * (ux * ux + uy * uy);
    const double Tv = ((double)(0.5) * nrg * invRho - kin) * invCv;
    pT[cell] = Tv;
    pP[cell] = Rgas * rho * Tv;
  }
}

__kernel void seedEquilibriaKernel(
  __global double* restrict pF,
  __global double* restrict pG,
  __global const double* restrict pRho,
  __global const double* restrict pT,
  __global const double* restrict pU,
  const double Ushift0,
  const double Ushift1,
  const double kTiny,
  int nx,
  int ny,
  int N,
  const double cv
) {
  int cell = get_global_id(0);
  if (cell < N) {
    const double rho = pRho[cell];
    const double Tv = pT[cell];
    const double ux = pU[0 * N + cell];
    const double uy = pU[1 * N + cell];
    const double u2 = ux * ux + uy * uy;

    // Feq (product form, branchless).
    const double uia_x = ux - Ushift0;
    const double uia_y = uy - Ushift1;
    const double ux2t = uia_x * uia_x + Tv;
    const double uy2t = uia_y * uia_y + Tv;
    
    for (int idc = 0; idc < 9; ++idc) {
      const int vx = kD2Q9Cx[idc], vy = kD2Q9Cy[idc];
      const double pfx = (vx == 0) ? ((double)(1) - ux2t) : ((double)(0.5) * ((double)(vx) * uia_x + ux2t));
      const double pfy = (vy == 0) ? ((double)(1) - uy2t) : ((double)(0.5) * ((double)(vy) * uia_y + uy2t));
      pF[idc * N + cell] = rho * pfx * pfy;
    }

    // Geq: temperature-dependent weights, xi==0 at init.
    const double E = cv * Tv + (double)(0.5) * u2;
    const double targetE = (double)(2) * rho * E;
    const double wZero = (double)(1) - Tv;
    const double wNonZero = (double)(0.5) * Tv;
    double sumW = (double)(0);
    
    for (int idc = 0; idc < 9; ++idc) {
      const double wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
      const double wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
      sumW += wx * wy;
    }
    const double sc = (sumW > kTiny) ? targetE / sumW : targetE / (double)(9);
    
    for (int idc = 0; idc < 9; ++idc) {
      const double wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
      const double wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
      pG[idc * N + cell] = wx * wy * sc;
    }
  }
}

__kernel void collisionAndEquilibriaKernel(
  __global double* restrict pF,
  __global double* restrict pG,
  __global double* restrict pLastGx,
  __global const double* restrict pRho,
  __global const double* restrict pT,
  __global const double* restrict pU,
  const double Ushift0,
  const double Ushift1,
  const double visc,
  const double cond,
  const double cp,
  const double cv,
  const double kTiny,
  const double kMaxE,
  int nx,
  int ny,
  int N
) {
  int cell = get_global_id(0);
  if (cell < N) {
    const double rho = pRho[cell];
    const double Tv = pT[cell];
    const double ux = pU[0 * N + cell];
    const double uy = pU[1 * N + cell];
    const double u2 = ux * ux + uy * uy;

    // Temperature-dependent weights (branchless, compile-time selects).
    const double wZero = (double)(1) - Tv;
    const double wNonZero = (double)(0.5) * Tv;

    // Compute feq (product form, fully branchless)
    const double uia_x = ux - Ushift0;
    const double uia_y = uy - Ushift1;
    const double ux2t = uia_x * uia_x + Tv;
    const double uy2t = uia_y * uia_y + Tv;
    double feqLocal[9];
    
    for (int idc = 0; idc < 9; ++idc) {
      const int vx = kD2Q9Cx[idc], vy = kD2Q9Cy[idc];
      const double pfx = (vx == 0) ? ((double)(1) - ux2t) : ((double)(0.5) * ((double)(vx) * uia_x + ux2t));
      const double pfy = (vy == 0) ? ((double)(1) - uy2t) : ((double)(0.5) * ((double)(vy) * uia_y + uy2t));
      feqLocal[idc] = rho * pfx * pfy;
    }

    // Newton-Raphson for geq
    // nrOk accumulates as a double (1.0 = all good, 0.0 = degenerate) to
    // keep all threads in the warp executing the same instructions (no
    // branch on bool that would cause warp divergence near shock waves).
    const double E = cv * Tv + (double)(0.5) * u2;
    const double targetE = (double)(2) * rho * E;

    const double denom0 = (targetE > kTiny) ? targetE : kTiny;
    double targetM[2];
    targetM[0] = (double)(2) * rho * ux * (E + Tv) / denom0;
    targetM[1] = (double)(2) * rho * uy * (E + Tv) / denom0;

    double xi[2];
    xi[0] = pLastGx[1 * N + cell];
    xi[1] = pLastGx[2 * N + cell];

    double alpha = (double)(1);
    double si[9], e[9];
    double Z = (double)(0);
    double smax;
    double nrOk = (double)(1); // 1.0 = solver healthy, 0.0 = degenerate

    
    for (int iter = 0; iter < 3; ++iter) {
      smax = -kMaxE;
      
      for (int idc = 0; idc < 9; ++idc) {
        const double s = xi[0] * ((double)(kD2Q9Cx[idc]) + Ushift0) + xi[1] * ((double)(kD2Q9Cy[idc]) + Ushift1);
        si[idc] = s;
        smax = (s > smax) ? s : smax;
      }
      smax = (smax < kMaxE) ? smax : kMaxE;

      double S1[2] = { (double)(0), (double)(0) };
      double S2_00 = (double)(0), S2_01 = (double)(0);
      double S2_10 = (double)(0), S2_11 = (double)(0);
      Z = (double)(0);
      
      for (int idc = 0; idc < 9; ++idc) {
        double expo = si[idc] - smax;
        expo = (expo < kMaxE) ? expo : kMaxE;
        expo = (expo > -kMaxE) ? expo : -kMaxE;
        const double wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
        const double wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
        const double wev = wx * wy * exp(expo);
        e[idc] = wev;
        Z += wev;
        const double c0 = (double)(kD2Q9Cx[idc]) + Ushift0;
        const double c1 = (double)(kD2Q9Cy[idc]) + Ushift1;
        S1[0] += c0 * wev;
        S1[1] += c1 * wev;
        S2_00 += c0 * c0 * wev;
        S2_01 += c0 * c1 * wev;
        S2_10 += c1 * c0 * wev;
        S2_11 += c1 * c1 * wev;
      }

      const double Zsafe = (Z > kTiny) ? Z : kTiny;
      // Accumulate degenerate flag as double to avoid branch divergence.
      nrOk *= (Z > kTiny) ? (double)(1) : (double)(0);
      const double invZ = (double)(1) / Zsafe;

      double J00 = (S2_00 - S1[0] * S1[0] * invZ) * invZ + kTiny;
      double J01 = (S2_01 - S1[0] * S1[1] * invZ) * invZ;
      double J10 = (S2_10 - S1[1] * S1[0] * invZ) * invZ;
      double J11 = (S2_11 - S1[1] * S1[1] * invZ) * invZ + kTiny;

      const double detJ = J00 * J11 - J01 * J10;
      const double absDet = fabs(detJ);
      nrOk *= (absDet > kTiny) ? (double)(1) : (double)(0);
      const double detSafe = (absDet > kTiny) ? detJ : ((detJ >= (double)(0)) ? kTiny : -kTiny);
      const double invDet = (double)(1) / detSafe;

      const double r0 = S1[0] * invZ - targetM[0];
      const double r1 = S1[1] * invZ - targetM[1];
      xi[0] -= alpha * (J11 * r0 - J01 * r1) * invDet;
      xi[1] -= alpha * (-J10 * r0 + J00 * r1) * invDet;
      alpha *= (double)(0.5);
    }

    // Compute final geqLocal (branchless, no if(solverOk))
    // nrOk == 1 → use NR result; nrOk == 0 → use uniform fallback.
    // Both branches are computed; the ternary select is warp-uniform.
    smax = -kMaxE;
    
    for (int idc = 0; idc < 9; ++idc) {
      const double s = xi[0] * ((double)(kD2Q9Cx[idc]) + Ushift0) + xi[1] * ((double)(kD2Q9Cy[idc]) + Ushift1);
      si[idc] = s;
      smax = (s > smax) ? s : smax;
    }
    smax = (smax < kMaxE) ? smax : kMaxE;
    Z = (double)(0);
    
    for (int idc = 0; idc < 9; ++idc) {
      double expo = si[idc] - smax;
      expo = (expo < kMaxE) ? expo : kMaxE;
      expo = (expo > -kMaxE) ? expo : -kMaxE;
      const double wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
      const double wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
      e[idc] = wx * wy * exp(expo);
      Z += e[idc];
    }

    // Uniform fallback: sumW of weights at xi==0 (all si==0, all expo==0).
    double sumW = (double)(0);
    
    for (int idc = 0; idc < 9; ++idc) {
      const double wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
      const double wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
      sumW += wx * wy;
    }
    const double sumWsafe = (sumW > kTiny) ? sumW : (double)(9);
    const double Zsafe = (Z > kTiny) ? Z : kTiny;

    // Branchless select: NR result when nrOk==1, uniform fallback when 0.
    const double scaleNR = targetE / Zsafe;
    const double scaleFB = targetE / sumWsafe;
    const double scale = nrOk * scaleNR + ((double)(1) - nrOk) * scaleFB;

    double geqLocal[9];
    
    for (int idc = 0; idc < 9; ++idc) {
      const double wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
      const double wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
      // NR path: scale * e[idc]; fallback path: scale * wx*wy.
      geqLocal[idc] = scale * (nrOk * e[idc] + ((double)(1) - nrOk) * wx * wy);
    }

    // Store xi (branchless: zero in fallback path).
    pLastGx[0 * N + cell] = nrOk * (log(scaleNR / rho) - smax);
    pLastGx[1 * N + cell] = nrOk * xi[0];
    pLastGx[2 * N + cell] = nrOk * xi[1];

    // BGK + thermal correction
    const double tau = visc / (rho * Tv) + (double)(0.5);
    const double omega = (double)(1) / tau;
    const double diff = cond / (rho * cp);
    const double tauT = diff / Tv + (double)(0.5);
    double omegaT = (double)(1) / tauT;

    // Non-equilibrium stress limiter (Tran et al. §3.3).
    double eps = (double)(0);
    
    for (int idc = 0; idc < 9; ++idc) {
      const double fi = pF[idc * N + cell];
      const double d = fi - feqLocal[idc];
      const double den = (feqLocal[idc] > kTiny) ? feqLocal[idc] : kTiny;
      eps += fabs(d) / den;
    }
    eps /= (double)(9);
    const double sigma = (eps >= (double)(1))
                           ? omega
                           : (eps >= (double)(0.1))
                           ? (double)(1.35)
                           : (eps >= (double)(0.01))
                           ? (double)(1.05)
                           : (double)(1);

    double omegaL = omega / sigma;
    omegaL = (omegaL > (double)(1)) ? omegaL : (double)(1);
    omegaL = (omegaL < ((double)(2) - (double)(1e-7))) ? omegaL : ((double)(2) - (double)(1e-7));
    omegaT = (omegaT > (double)(1)) ? omegaT : (double)(1);
    omegaT = (omegaT < ((double)(2) - (double)(1e-7))) ? omegaT : ((double)(2) - (double)(1e-7));

    // L correction vector.
    double L0 = (double)(0);
    double L1 = (double)(0);
    
    for (int idc = 0; idc < 9; ++idc) {
      const double fi = pF[idc * N + cell];
      const double cix0 = (double)(kD2Q9Cx[idc]);
      const double ciy0 = (double)(kD2Q9Cy[idc]);
      const double uvi = ux * cix0 + uy * ciy0;
      const double aux = (double)(2) * uvi * (fi - feqLocal[idc]);
      L0 += aux * cix0;
      L1 += aux * ciy0;
    }

    const double invT = (double)(1) / Tv;
    
    for (int idc = 0; idc < 9; ++idc) {
      const int fi_idx = idc * N + cell;
      const double fOld = pF[fi_idx];
      pF[fi_idx] = fOld + omegaL * (feqLocal[idc] - fOld);
      const double c0 = (double)(kD2Q9Cx[idc]) + Ushift0;
      const double c1 = (double)(kD2Q9Cy[idc]) + Ushift1;
      const double cidotL = L0 * c0 + L1 * c1;
      const double wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
      const double wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
      const double Wi = wx * wy;
      const double gDiff = Wi * cidotL * invT;
      const double gOld = pG[fi_idx];
      pG[fi_idx] = gOld + omegaL * (geqLocal[idc] - gOld) + (omegaL - omegaT) * gDiff;
    }
  }
}

__kernel void streamAndMacroscopicKernel(
  __global const double* restrict pF,
  __global const double* restrict pG,
  __global double* restrict pFaux,
  __global double* restrict pGaux,
  __global double* restrict pRho,
  __global double* restrict pP,
  __global double* restrict pT,
  __global double* restrict pU,
  const double Ushift0,
  const double Ushift1,
  const double invCv,
  const double Rgas,
  const double kTiny,
  const double idwExp,
  int nx,
  int ny,
  int N
) {
  int cell = get_global_id(0);
  if (cell < N) {
    int ci, cj;
    if (ny == 1) { ci = cell; cj = 0; } else if (ny == 2) { ci = cell >> 1; cj = cell & 1; } else {
      ci = cell / ny;
      cj = cell % ny;
    }

    const double negHalfIdw = ((double)(-0.5 )) * idwExp;

    double rho = ((double)(0 ));
    double nrg = ((double)(0 ));
    double mx = ((double)(0 ));
    double my = ((double)(0 ));

    
    for (int idc = 0; idc < 9; ++idc) {
      const double cix_s = ((double)(kD2Q9Cx[idc])) + Ushift0;
      const double ciy_s = ((double)(kD2Q9Cy[idc])) + Ushift1;

      // Source position (fractional cell coordinates).
      const double xSrc = ((double)(ci)) - cix_s;
      const double ySrc = ((double)(cj)) - ciy_s;

      // Floor corners.
      const int x0 = ((int)(floor(xSrc)));
      const int y0 = ((int)(floor(ySrc)));
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

      double fi, gi;

      // Diagonal shift: 2-D IDW via exp(-p*log(r²+kTiny)).
      // Branchless — no snap-to-node cascade. When any corner coincides
      // with the source point (r²→0), log(kTiny) is finite and large in
      // magnitude, so exp(·) → +∞ and after wsum normalisation the result
      // converges correctly to the node value.
      const double dx0 = xSrc - ((double)(x0));
      const double dy0 = ySrc - ((double)(y0));
      const double dx1 = xSrc - ((double)(x1));
      const double dy1 = ySrc - ((double)(y1));

      const double w00 = exp(negHalfIdw * log(dx0 * dx0 + dy0 * dy0 + kTiny));
      const double w10 = exp(negHalfIdw * log(dx1 * dx1 + dy0 * dy0 + kTiny));
      const double w01 = exp(negHalfIdw * log(dx0 * dx0 + dy1 * dy1 + kTiny));
      const double w11 = exp(negHalfIdw * log(dx1 * dx1 + dy1 * dy1 + kTiny));
      const double wsum = w00 + w10 + w01 + w11;
      const double invSum = ((double)(1 )) / wsum;
      fi = (w00 * pF[off + c00] + w10 * pF[off + c10] + w01 * pF[off + c01] + w11 * pF[off + c11]) * invSum;
      gi = (w00 * pG[off + c00] + w10 * pG[off + c10] + w01 * pG[off + c01] + w11 * pG[off + c11]) * invSum;

      pFaux[off + cell] = fi;
      pGaux[off + cell] = gi;
      rho += fi;
      nrg += gi;
      mx += cix_s * fi;
      my += ciy_s * fi;
    }

    const double invRho = ((double)(1 )) / rho;
    const double ux = mx * invRho;
    const double uy = my * invRho;
    pU[0 * N + cell] = ux;
    pU[1 * N + cell] = uy;
    const double kin = ((double)(0.5 )) * (ux * ux + uy * uy);
    const double Tv = (((double)(0.5 )) * nrg * invRho - kin) * invCv;
    pRho[cell] = rho;
    pT[cell] = Tv;
    pP[cell] = Rgas * rho * Tv;
  }
}

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

inline void global_barrier(volatile __global int* sync_counter, volatile __global int* sync_flag, int num_workgroups) {
    int local_id = get_local_id(0);
    if (local_id == 0) {
        int arrived = atomic_add(sync_counter, 1);
        if (arrived == num_workgroups - 1) {
            atomic_xchg(sync_counter, 0);
            atomic_add(sync_flag, 1);
        } else {
            int current_flag = atomic_add(sync_flag, 0);
            while (atomic_add(sync_flag, 0) == current_flag) {
                mem_fence(CLK_GLOBAL_MEM_FENCE);
            }
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void timeLoopKernel(
  __global double* pF0,
  __global double* pG0,
  __global double* pF1,
  __global double* pG1,
  __global double* restrict pLastGx,
  __global double* restrict pRho,
  __global double* restrict pT,
  __global double* restrict pU,
  __global double* restrict pP,
  const double Ushift0,
  const double Ushift1,
  const double visc,
  const double cond,
  const double cp,
  const double cv,
  const double kTiny,
  const double kMaxE,
  const double invCv,
  const double Rgas,
  int nx,
  int ny,
  int N,
  int steps,
  volatile __global int* sync_counter,
  volatile __global int* sync_flag,
  int num_workgroups
) {
  int global_id = get_global_id(0);
  int global_size = get_global_size(0);
  
  for (int step = 0; step < steps; ++step) {
      __global double* fCur = (step % 2 == 0) ? pF0 : pF1;
      __global double* gCur = (step % 2 == 0) ? pG0 : pG1;
      __global double* fAlt = (step % 2 == 0) ? pF1 : pF0;
      __global double* gAlt = (step % 2 == 0) ? pG1 : pG0;

      for (int cell = global_id; cell < N; cell += global_size) {
          const double rho = pRho[cell];
          const double Tv = pT[cell];
          const double ux = pU[0 * N + cell];
          const double uy = pU[1 * N + cell];
          const double u2 = ux * ux + uy * uy;
          const double wZero = 1.0 - Tv;
          const double wNonZero = 0.5 * Tv;
          const double uia_x = ux - Ushift0;
          const double uia_y = uy - Ushift1;
          const double ux2t = uia_x * uia_x + Tv;
          const double uy2t = uia_y * uia_y + Tv;
          double feqLocal[9];
          for (int idc = 0; idc < 9; ++idc) {
              const int vx = kD2Q9Cx[idc], vy = kD2Q9Cy[idc];
              const double pfx = (vx == 0) ? (1.0 - ux2t) : (0.5 * (vx * uia_x + ux2t));
              const double pfy = (vy == 0) ? (1.0 - uy2t) : (0.5 * (vy * uia_y + uy2t));
              feqLocal[idc] = rho * pfx * pfy;
          }
          const double E = cv * Tv + 0.5 * u2;
          const double targetE = 2.0 * rho * E;
          const double denom0 = (targetE > kTiny) ? targetE : kTiny;
          double targetM[2] = { 2.0 * rho * ux * (E + Tv) / denom0, 2.0 * rho * uy * (E + Tv) / denom0 };
          double xi[2] = { pLastGx[1 * N + cell], pLastGx[2 * N + cell] };
          double si[9], e_w[9];
          double Z = 0.0, smax;
          double nrOk = 1.0;
          for (int iter = 0; iter < 3; ++iter) {
              smax = -kMaxE;
              for (int idc = 0; idc < 9; ++idc) {
                  const double s = xi[0] * (kD2Q9Cx[idc] + Ushift0) + xi[1] * (kD2Q9Cy[idc] + Ushift1);
                  si[idc] = s;
                  smax = (s > smax) ? s : smax;
              }
              smax = (smax < kMaxE) ? smax : kMaxE;
              double S1[2] = { 0.0, 0.0 };
              double S2_00 = 0.0, S2_01 = 0.0, S2_10 = 0.0, S2_11 = 0.0;
              Z = 0.0;
              for (int idc = 0; idc < 9; ++idc) {
                  double expo = si[idc] - smax;
                  expo = clamp(expo, -kMaxE, kMaxE);
                  const double wx = (kD2Q9Cx[idc] == 0) ? wZero : wNonZero;
                  const double wy = (kD2Q9Cy[idc] == 0) ? wZero : wNonZero;
                  const double wev = wx * wy * exp(expo);
                  e_w[idc] = wev;
                  Z += wev;
                  const double c0 = kD2Q9Cx[idc] + Ushift0;
                  const double c1 = kD2Q9Cy[idc] + Ushift1;
                  S1[0] += c0 * wev;  S1[1] += c1 * wev;
                  S2_00 += c0 * c0 * wev; S2_01 += c0 * c1 * wev;
                  S2_10 += c1 * c0 * wev; S2_11 += c1 * c1 * wev;
              }
              const double Zsafe = (Z > kTiny) ? Z : kTiny;
              nrOk *= (Z > kTiny) ? 1.0 : 0.0;
              const double invZ = 1.0 / Zsafe;
              double J00 = (S2_00 - S1[0] * S1[0] * invZ) * invZ + kTiny;
              double J01 = (S2_01 - S1[0] * S1[1] * invZ) * invZ;
              double J10 = (S2_10 - S1[1] * S1[0] * invZ) * invZ;
              double J11 = (S2_11 - S1[1] * S1[1] * invZ) * invZ + kTiny;
              double invDet = 1.0 / (J00 * J11 - J01 * J10);
              double invJ00 = J11 * invDet;
              double invJ01 = -J01 * invDet;
              double invJ10 = -J10 * invDet;
              double invJ11 = J00 * invDet;
              double d0 = S1[0] * invZ - targetM[0];
              double d1 = S1[1] * invZ - targetM[1];
              xi[0] -= (invJ00 * d0 + invJ01 * d1) * nrOk;
              xi[1] -= (invJ10 * d0 + invJ11 * d1) * nrOk;
          }
          pLastGx[1 * N + cell] = xi[0] * nrOk;
          pLastGx[2 * N + cell] = xi[1] * nrOk;
          const double Zsafe = (Z > kTiny) ? Z : kTiny;
          const double sc = (targetE > kTiny) ? (targetE / Zsafe) : 0.0;
          for (int idc = 0; idc < 9; ++idc) {
              double geq = e_w[idc] * sc;
              double f_i = fCur[idc * N + cell];
              double g_i = gCur[idc * N + cell];
              fCur[idc * N + cell] = f_i - visc * (f_i - feqLocal[idc]);
              gCur[idc * N + cell] = g_i - cond * (g_i - geq);
          }
      }

      global_barrier(sync_counter, sync_flag, num_workgroups);

      for (int cell = global_id; cell < N; cell += global_size) {
          int cx = cell % nx;
          int cy = cell / nx;
          double rho_new = 0.0, nrg_new = 0.0;
          double mx_new = 0.0, my_new = 0.0;
          for (int idc = 0; idc < 9; ++idc) {
              int vx = kD2Q9Cx[idc], vy = kD2Q9Cy[idc];
              int nx_src = (cx - vx + nx) % nx;
              int ny_src = (cy - vy + ny) % ny;
              int src_cell = ny_src * nx + nx_src;
              double fi = fCur[idc * N + src_cell];
              double gi = gCur[idc * N + src_cell];
              fAlt[idc * N + cell] = fi;
              gAlt[idc * N + cell] = gi;
              rho_new += fi;
              nrg_new += gi;
              mx_new += (vx + Ushift0) * fi;
              my_new += (vy + Ushift1) * fi;
          }
          pRho[cell] = rho_new;
          double invRho = 1.0 / rho_new;
          double ux = mx_new * invRho;
          double uy = my_new * invRho;
          pU[0 * N + cell] = ux;
          pU[1 * N + cell] = uy;
          double kin = 0.5 * (ux * ux + uy * uy);
          double Tv = (0.5 * nrg_new * invRho - kin) * invCv;
          pT[cell] = Tv;
          pP[cell] = Rgas * rho_new * Tv;
      }
      
      global_barrier(sync_counter, sync_flag, num_workgroups);
  }
}
