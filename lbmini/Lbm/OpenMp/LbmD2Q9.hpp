#ifndef LBMINI_OPENMP_LBMD2Q9_HPP_
#define LBMINI_OPENMP_LBMD2Q9_HPP_

#include <cmath>
#include <cfloat>

#include "Data/ControlData.hpp"
#include "Data/FluidData.hpp"
#include "Lbm/LbmBase.hpp"

namespace lbmini::openmp {
template<typename Scalar>
class LbmD2Q9 {
public:
  using Index = int;

  // using Base = LbmClassBase<Scalar, 2, 9>;
  // using Base::Dim, Base::Speeds;
  static constexpr Index Dim() { return 2; }

  static constexpr Index Speeds() { return 9; }

  static constexpr Index Velocity(Index index, Index dir) { return kVelocity_[index][dir]; }

  static constexpr Index Opposite(Index index) { return kOpposite_[index]; }

  static Scalar Weights(Index idc, Scalar tem);

  static void Init(
    const Scalar u0[Dim()],
    Scalar* rho0,
    Scalar* p0,
    Scalar* rho,
    Scalar* p,
    Scalar* tem,
    Scalar* u,
    Scalar* f,
    Scalar* feq,
    Scalar* g,
    Scalar* geq,
    Scalar* lastGx,
    const FluidData<Scalar>& pFluid,
    const ControlData<Scalar>& pControl
  );

  static void ComputeMacroscopic(
    Scalar* rho,
    Scalar* p,
    Scalar* tem,
    Scalar* u,
    Scalar* f,
    Scalar* feq,
    Scalar* g,
    Scalar* geq,
    Scalar* lastGx,
    const FluidData<Scalar>& pFluid,
    const ControlData<Scalar>& pControl
  );

  static void Collision(
    Scalar* rho,
    Scalar* p,
    Scalar* tem,
    Scalar* u,
    Scalar* f,
    Scalar* feq,
    Scalar* g,
    Scalar* geq,
    Scalar* lastGx,
    const FluidData<Scalar>& pFluid,
    const ControlData<Scalar>& pControl
  );

protected:
  static Scalar computePhiAxis(Index idc, Index idd, Scalar tem, const Scalar* u, const ControlData<Scalar>& pControl);

  static void computeFeq(
    Scalar* rho,
    Scalar* p,
    Scalar* tem,
    Scalar* u,
    Scalar* f,
    Scalar* feq,
    Scalar* g,
    Scalar* geq,
    Scalar* lastGx,
    const FluidData<Scalar>& pFluid,
    const ControlData<Scalar>& pControl
  );

  static void computeGeq(
    Scalar* rho,
    Scalar* p,
    Scalar* tem,
    Scalar* u,
    Scalar* f,
    Scalar* feq,
    Scalar* g,
    Scalar* geq,
    Scalar* lastGx,
    const FluidData<Scalar>& pFluid,
    const ControlData<Scalar>& pControl
  );

private:
  struct Workspace {
    // from ComputeMacroscopic
    Scalar mom[Dim()];

    // from Collision
    Scalar P[Dim()][Dim()];
    Scalar Peq[Dim()][Dim()];
    Scalar L[Dim()];
    Scalar ci_coll[Dim()];

    // from computeGeq
    Scalar Wi[Speeds()];
    Scalar cshift[Speeds()][Dim()];
    Scalar targetQ[Dim()];
    Scalar m_target[Dim()];
    Scalar xi[Dim()];
    Scalar si[Speeds()];
    Scalar e[Speeds()];
    Scalar S1n[Dim()];
    Scalar S2n[Dim()][Dim()];
    Scalar r[Dim()];
    Scalar J[Dim()][Dim()];
    Scalar delta[Dim()];
    Scalar bestXi[Dim()];
    Scalar xicand[Dim()];
    Scalar S1n_c[Dim()];
    Scalar mcand[Dim()];
    Scalar S1final[Dim()];
  };

  // Constants
  inline static const Scalar kTiny_ = Scalar(1.0e-12);
  inline static const Scalar kMaxExp_ = Scalar(700.0);
  inline static const Scalar kCs2_ = Scalar(1.0) / Scalar(3.0);

  inline static const Index kVelocity_[Speeds()][Dim()] = {
    { 0, 0 },
    { 1, 0 },
    { -1, 0 },
    { 0, 1 },
    { 0, -1 },
    { 1, 1 },
    { -1, -1 },
    { 1, -1 },
    { -1, 1 },
  };

  inline static const Index kOpposite_[Speeds()] = {
    0,
    2,
    1,
    4,
    3,
    6,
    5,
    8,
    7,
  };
};

#pragma omp declare target

template<typename Scalar>
auto LbmD2Q9<Scalar>::Weights(const Index idc, const Scalar tem) -> Scalar {
  const Scalar weightZero = Scalar(1.0) - tem;
  const Scalar weightNonZero = Scalar(0.5) * tem;
  Scalar weight = Scalar(1.0);

  for (Index idd = 0; idd < Dim(); ++idd) {
    Index vid = Velocity(idc, idd);
    if (vid == Index(0))
      weight *= weightZero;
    else
      weight *= weightNonZero;
  }

  return weight;
}

template<typename Scalar>
auto LbmD2Q9<Scalar>::Init(
  const Scalar u0[Dim()],
  Scalar* rho0,
  Scalar* p0,
  Scalar* rho,
  Scalar* p,
  Scalar* tem,
  Scalar* u,
  Scalar* f,
  Scalar* feq,
  Scalar* g,
  Scalar* geq,
  Scalar* lastGx,
  const FluidData<Scalar>& pFluid,
  const ControlData<Scalar>& pControl
) -> void {
  // Initialize velocity
  for (Index idd = 0; idd < Dim(); ++idd)
    u[idd] = u0[idd];

  // Initialize density and pressure
  *rho = *rho0;
  *p = *p0;

  // Initialize temperature
  *tem = kCs2_ * (*p) / ((*rho) * pFluid.constant);

  // Initialize f to feq
  computeFeq(rho, p, tem, u, f, feq, g, geq, lastGx, pFluid, pControl);

  // Invalidate lastGx at initialization to avoid accidental reuse from previous sim
  for (Index i = 0; i < Dim() + 1; ++i)
    lastGx[i] = Scalar(0.0);

  // Initialize g to geq
  computeGeq(rho, p, tem, u, f, feq, g, geq, lastGx, pFluid, pControl);

#pragma omp simd
  for (Index idc = 0; idc < Speeds(); ++idc) {
    f[idc] = feq[idc];
    g[idc] = geq[idc];
  }

  // Recompute macroscopic so derived fields are consistent after init
  ComputeMacroscopic(rho, p, tem, u, f, feq, g, geq, lastGx, pFluid, pControl);
}

template<typename Scalar>
auto LbmD2Q9<Scalar>::ComputeMacroscopic(
  Scalar* rho,
  Scalar* p,
  Scalar* tem,
  Scalar* u,
  Scalar* f,
  Scalar* feq,
  Scalar* g,
  Scalar* geq,
  Scalar* lastGx,
  const FluidData<Scalar>& pFluid,
  const ControlData<Scalar>& pControl
) -> void {
  Workspace ws_;

  // Compute rho (Eq. 7) and momentum (mom, Eq. 8)
  *rho = Scalar(0.0);
  // Compute total fluid energy (Eq. 9): S = sum_i g_i = 2 * rho * E
  Scalar nrg = Scalar(0.0);
  // Compute moments
  for (Index i = 0; i < Dim(); ++i) {
    ws_.mom[i] = Scalar(0.0);
  }

#pragma omp simd
  for (Index idc = 0; idc < Speeds(); ++idc) {
    *rho += f[idc];
    nrg += g[idc];
    for (Index idd = 0; idd < Dim(); ++idd) {
      Scalar ci = static_cast<Scalar>(Velocity(idc, idd)) + pControl.U(idd);
      ws_.mom[idd] += ci * f[idc];
    }
  }

  Scalar nrgKinect = Scalar(0.0);
  for (Index idd = 0; idd < Dim(); ++idd) {
    u[idd] = ws_.mom[idd] / (*rho);
    nrgKinect += u[idd] * u[idd];
  }

  nrg /= Scalar(2.0) * (*rho);
  nrgKinect /= Scalar(2.0);

  // Compute temperature (Eq. 10)
  *tem = (nrg - nrgKinect) / pFluid.specificHeatCv;

  // Compute pressure (ideal gas law)
  *p = pFluid.constant * (*rho) * (*tem);
}

template<typename Scalar>
auto LbmD2Q9<Scalar>::Collision(
  Scalar* rho,
  Scalar* p,
  Scalar* tem,
  Scalar* u,
  Scalar* f,
  Scalar* feq,
  Scalar* g,
  Scalar* geq,
  Scalar* lastGx,
  const FluidData<Scalar>& pFluid,
  const ControlData<Scalar>& pControl
) -> void {
  Workspace ws_;

  Scalar u2 = Scalar(0.0);
  for (Index idd = 0; idd < Dim(); ++idd)
    u2 += u[idd] * u[idd];

  // Compute distributions
  computeFeq(rho, p, tem, u, f, feq, g, geq, lastGx, pFluid, pControl);
  computeGeq(rho, p, tem, u, f, feq, g, geq, lastGx, pFluid, pControl);

  // Relaxation factor (Eq. 3)
  Scalar tau = pFluid.viscosity / ((*rho) * (*tem)) + Scalar(0.5);
  Scalar omega = Scalar(1.0) / tau;

  // Thermal relaxation factor
  const Scalar diffusivity = pFluid.conductivity / ((*rho) * pFluid.specificHeatCp);
  const Scalar tauThermal = diffusivity / (*tem) + Scalar(0.5);
  Scalar omegaThermal = Scalar(1.0) / tauThermal;

  // Knudsen sensor epsilon (Eq. 19)
  Scalar eps = Scalar(0.0);
#pragma omp simd
  for (Index idc = 0; idc < Speeds(); ++idc) {
    Scalar diff = f[idc] - feq[idc];
    eps += fabs(diff) / (((feq[idc]) > (kTiny_)) ? (feq[idc]) : (kTiny_));
  }
  eps /= Scalar(Speeds());

  // sigma(ε) from (Eq. 20)
  Scalar sigma = Scalar(1.0);
  if (eps >= Scalar(1.0))
    sigma = omega;
  else if (eps >= Scalar(1.0e-1))
    sigma = Scalar(1.35);
  else if (eps >= Scalar(1.0e-2))
    sigma = Scalar(1.05);

  Scalar omegaLoc = omega / sigma;
  Scalar omegaThermalLoc = omegaThermal;

  // Clamp omegas to safe bounds
  // TODO: Review the omega clamp
  omegaLoc = (omegaLoc > Scalar(1.0)) ? omegaLoc : Scalar(1.0);
  omegaLoc = (omegaLoc < (Scalar(2.0) - Scalar(1.0e-7))) ? omegaLoc : (Scalar(2.0) - Scalar(1.0e-7));
  omegaThermalLoc = (omegaThermalLoc > Scalar(1.0)) ? omegaThermalLoc : Scalar(1.0);
  omegaThermalLoc = (omegaThermalLoc < (Scalar(2.0) - Scalar(1.0e-7))) ? omegaThermalLoc : (Scalar(2.0) - Scalar(1.0e-7));

  // Compute pressure tensor (Eq. 13a): P_ab = sum_i ci_a ci_b f_i
  // Compute equilibrium pressure tensor (Eq. 11): P_ab_eq = rho u_a u_b + rho T delta_ab
  // Compute L = u_b (P_ab - P_ab_eq)
  for (Index i = 0; i < Dim(); ++i) {
    for (Index j = 0; j < Dim(); ++j) {
      ws_.P[i][j] = Scalar(0.0);
      ws_.Peq[i][j] = Scalar(0.0);
    }
    ws_.L[i] = Scalar(0.0);
  }

  for (Index a = 0; a < Dim(); ++a) {
    for (Index b = 0; b < Dim(); ++b) {
      Scalar p_ab = 0, peq_ab = 0;
#pragma omp simd
      for (Index idc = 0; idc < Speeds(); ++idc) {
        Scalar cia = static_cast<Scalar>(Velocity(idc, a));
        Scalar cib = static_cast<Scalar>(Velocity(idc, b));
        p_ab += (cia * cib) * f[idc];
        peq_ab += (cia * cib) * feq[idc];
      }
      ws_.P[a][b] = p_ab;
      ws_.Peq[a][b] = peq_ab;

      ws_.L[a] += Scalar(2.0) * u[b] * (p_ab - peq_ab);
    }
  }

  // Collisions
#pragma omp simd
  for (Index idc = 0; idc < Speeds(); ++idc) {
    // f distribution
    f[idc] += omegaLoc * (feq[idc] - f[idc]);

    // Compute (g* - geq) per (Eq. 13): delta = Wi * ( ci · ( (P - Peq) * u ) ) / T
    Scalar Wi = Weights(idc, *tem);

    // ci for projection uses shifted velocities (moments)
    ws_.ci_coll[0] = static_cast<Scalar>(Velocity(idc, 0)) + pControl.U(0);
    ws_.ci_coll[1] = static_cast<Scalar>(Velocity(idc, 1)) + pControl.U(1);

    const Scalar cidotL = ws_.ci_coll[0] * ws_.L[0] + ws_.ci_coll[1] * ws_.L[1];
    const Scalar gDiff = Wi * (cidotL / (*tem));

    // Energy distribution
    g[idc] += omegaLoc * (geq[idc] - g[idc]) + (omegaLoc - omegaThermalLoc) * gDiff;
  }
}

template<typename Scalar>
auto LbmD2Q9<Scalar>::computePhiAxis(
  const Index idc,
  const Index idd,
  const Scalar tem,
  const Scalar* u,
  const ControlData<Scalar>& pControl
) -> Scalar {
  const Index vi = Velocity(idc, idd);
  const Scalar uia = u[idd] - pControl.U(idd);
  const Scalar uia2 = uia * uia;

  if (vi == Index(0))
    return Scalar(1.0) - (uia2 + tem);
  if (vi == Index(1))
    return Scalar(0.5) * (uia + uia2 + tem);
  if (vi == Index(-1))
    return Scalar(0.5) * (-uia + uia2 + tem);

  return Scalar(0.0);
}

template<typename Scalar>
auto LbmD2Q9<Scalar>::computeFeq(
  Scalar* rho,
  Scalar* p,
  Scalar* tem,
  Scalar* u,
  Scalar* f,
  Scalar* feq,
  Scalar* g,
  Scalar* geq,
  Scalar* lastGx,
  const FluidData<Scalar>& pFluid,
  const ControlData<Scalar>& pControl
) -> void {
#pragma omp simd
  for (Index idc = 0; idc < Speeds(); ++idc) {
    feq[idc] = *rho;
    for (Index idd = 0; idd < Dim(); ++idd)
      feq[idc] *= computePhiAxis(idc, idd, *tem, u, pControl);
  }
}

template<typename Scalar>
auto LbmD2Q9<Scalar>::computeGeq(
  Scalar* rho,
  Scalar* p,
  Scalar* tem,
  Scalar* u,
  Scalar* f,
  Scalar* feq,
  Scalar* g,
  Scalar* geq,
  Scalar* lastGx,
  const FluidData<Scalar>& pFluid,
  const ControlData<Scalar>& pControl
) -> void {
  Workspace ws_;

  // Macroscopic quantities
  Scalar u2 = Scalar(0.0);
  for (Index idd = 0; idd < Dim(); ++idd)
    u2 += u[idd] * u[idd];
  Scalar E = pFluid.specificHeatCv * (*tem) + Scalar(0.5) * u2;

  // Precompute Wi (axis product) and shifted velocities cshift = c + U
#pragma omp simd
  for (Index idc = 0; idc < Speeds(); ++idc) {
    ws_.Wi[idc] = Weights(idc, *tem);
    for (Index d = 0; d < Dim(); ++d)
      ws_.cshift[idc][d] = static_cast<Scalar>(Velocity(idc, d)) + pControl.U(d);
  }

  // Targets of the root-finding method
  const Scalar targetE = Scalar(2.0) * (*rho) * E; // sum_i geq_i
  for (Index idd = 0; idd < Dim(); ++idd)
    ws_.targetQ[idd] = Scalar(2.0) * (*rho) * u[idd] * (E + (*tem));

  // Handle degenerate trivial case
  if (targetE <= Scalar(0.0)) {
    // fallback: simple equal distribution
    const Scalar total = Scalar(2.0) * (*rho) * E;
#pragma omp simd
    for (Index idc = 0; idc < Speeds(); ++idc)
      geq[idc] = total / Scalar(Speeds());

    for (Index iddP1 = 0; iddP1 < Dim() + 1; ++iddP1)
      lastGx[iddP1] = Scalar(0.0);

    return;
  }

  // Desired mean (flux per unit energy)
  for (Index i = 0; i < Dim(); ++i) {
    ws_.m_target[i] = ws_.targetQ[i] / targetE;
  }

  // Solve for xi such that m(xi) = m_target
  // Initial guess for xi: use cached warm start if available, else zero
  ws_.xi[0] = lastGx[1];
  ws_.xi[1] = lastGx[2];

  // Newton parameters
  const Index maxIter = 100;
  const Scalar tolRel = kTiny_; // relative tolerance on mean residual
  bool xiConverged = false;

  // Scaling used in relative residual
  const Scalar m_target_norm = sqrt(ws_.m_target[0] * ws_.m_target[0] + ws_.m_target[1] * ws_.m_target[1]);
  Scalar mScale = ((Scalar(1.0)) > (m_target_norm) ? (Scalar(1.0)) : (m_target_norm));

  // Newton iterations
  for (Index iter = 0; iter < maxIter; ++iter) {
    // Compute s_i = xi · c_i and smax for numeric stability
    Scalar smax = -HUGE_VAL;
#pragma omp simd
    for (Index idc = 0; idc < Speeds(); ++idc) {
      Scalar dot = Scalar(0);
      for (Index d = 0; d < Dim(); ++d)
        dot += ws_.xi[d] * ws_.cshift[idc][d];
      ws_.si[idc] = dot;
      if (ws_.si[idc] > smax)
        smax = ws_.si[idc];
    }

    if (smax > kMaxExp_)
      smax = kMaxExp_;
    if (smax < -kMaxExp_)
      smax = -kMaxExp_;

    // Build weighted exponentials e_i = Wi * exp(si - smax)
    for (Index i = 0; i < Speeds(); ++i) { ws_.e[i] = Scalar(0); }
    Scalar Z = Scalar(0);                                         // sum e_i
    for (Index i = 0; i < Dim(); ++i) { ws_.S1n[i] = Scalar(0); } // sum ci * e_i
    for (int i = 0; i < Dim(); ++i) {
      for (int j = 0; j < Dim(); ++j) {
        ws_.S2n[i][j] = Scalar(0);
      }
    } // sum ci ci^T * e_i

    bool bad = false;
    for (Index idc = 0; idc < Speeds(); ++idc) {
      Scalar expo = ws_.si[idc] - smax;
      if (expo > kMaxExp_)
        expo = kMaxExp_;
      if (expo < -kMaxExp_)
        expo = -kMaxExp_;
      Scalar ev = exp(expo);
      if (!isfinite(ev)) {
        bad = true;
        break;
      }
      ws_.e[idc] = ws_.Wi[idc] * ev;
      Z += ws_.e[idc];
      for (Index a = 0; a < Dim(); ++a) {
        ws_.S1n[a] += ws_.cshift[idc][a] * ws_.e[idc];
        for (Index b = 0; b < Dim(); ++b)
          ws_.S2n[a][b] += ws_.cshift[idc][a] * ws_.cshift[idc][b] * ws_.e[idc];
      }
    }
    if (bad || Z <= kTiny_)
      break;

    // m(xi) = S1n / Z
    ws_.mcand[0] = ws_.S1n[0] / Z;
    ws_.mcand[1] = ws_.S1n[1] / Z;

    // Residual r = mxi - m_target
    ws_.r[0] = ws_.mcand[0] - ws_.m_target[0];
    ws_.r[1] = ws_.mcand[1] - ws_.m_target[1];
    const Scalar rn = sqrt(ws_.r[0] * ws_.r[0] + ws_.r[1] * ws_.r[1]);
    if (rn / mScale < tolRel) {
      xiConverged = true;
      break;
    }

    // Jacobian J = Cov = S2n / Z - mxi * mxi^T  (2x2 SPD)
    for (int i = 0; i < Dim(); ++i) {
      for (int j = 0; j < Dim(); ++j) {
        ws_.J[i][j] = (ws_.S2n[i][j] / Z) - (ws_.mcand[i] * ws_.mcand[j]);
      }
    }

    // If J is nearly singular, add tiny LM regularizer
    Scalar lm = Scalar(0.0);
    Scalar J_00 = ws_.J[0][0], J_01 = ws_.J[0][1], J_11 = ws_.J[1][1];
    Scalar detJ = J_00 * J_11 - J_01 * J_01;

    if (J_00 <= 0 || detJ <= (sizeof(Scalar) == sizeof(float) ? FLT_EPSILON : DBL_EPSILON)) {
      // Add small diagonal (Levenberg-Marquardt) and retry
      lm = ((Scalar(1.0e-12)) > (Scalar(1.0e-6) * (J_00 + J_11)) ? (Scalar(1.0e-12)) : (Scalar(1.0e-6) * (J_00 + J_11)));
      J_00 += lm;
      J_11 += lm;
      detJ = J_00 * J_11 - J_01 * J_01;
      if (fabs(detJ) < (sizeof(Scalar) == sizeof(float) ? FLT_EPSILON : DBL_EPSILON)) {
        // Cannot solve, break to fallback
        break;
      }
    }

    // Newton step: delta = -J^{-1} r
    const Scalar invDetJ = Scalar(1.0) / detJ;
    ws_.delta[0] = invDetJ * (J_11 * (-ws_.r[0]) - J_01 * (-ws_.r[1]));
    ws_.delta[1] = invDetJ * (-J_01 * (-ws_.r[0]) + J_00 * (-ws_.r[1]));
    if (!isfinite(ws_.delta[0]) || !isfinite(ws_.delta[1]))
      break;

    // Damped line-search on xi: try full step then backtrack if residual norm not improved
    Scalar alpha = Scalar(1.0);
    const Scalar alphaMin = Scalar(1.0e-12);
    Scalar bestRn = rn;
    ws_.bestXi[0] = ws_.xi[0];
    ws_.bestXi[1] = ws_.xi[1];
    bool accepted = false;
    for (int ls = 0; ls < 20; ++ls) {
      ws_.xicand[0] = ws_.xi[0] + alpha * ws_.delta[0];
      ws_.xicand[1] = ws_.xi[1] + alpha * ws_.delta[1];

      // Quick evaluation of residual at xicand (repeat same routine)
      // Compute s_i, smax, e_i, Z, S1n
      Scalar candSmax = -HUGE_VAL;
#pragma omp simd
      for (Index idc = 0; idc < Speeds(); ++idc) {
        Scalar dot = Scalar(0);
        for (Index d = 0; d < Dim(); ++d)
          dot += ws_.xicand[d] * ws_.cshift[idc][d];
        if (dot > candSmax)
          candSmax = dot;
      }
      if (candSmax > kMaxExp_)
        candSmax = kMaxExp_;
      if (candSmax < -kMaxExp_)
        candSmax = -kMaxExp_;

      Scalar Zc = Scalar(0);
      for (Index i = 0; i < Dim(); ++i) { ws_.S1n_c[i] = Scalar(0); }
      bool badc = false;
      for (Index idc = 0; idc < Speeds(); ++idc) {
        Scalar dot = Scalar(0);
        for (Index d = 0; d < Dim(); ++d)
          dot += ws_.xicand[d] * ws_.cshift[idc][d];
        Scalar expo = dot - candSmax;
        if (expo > kMaxExp_)
          expo = kMaxExp_;
        if (expo < -kMaxExp_)
          expo = -kMaxExp_;
        Scalar ev = exp(expo);
        if (!isfinite(ev)) {
          badc = true;
          break;
        }
        Scalar ei = ws_.Wi[idc] * ev;
        Zc += ei;
        for (Index a = 0; a < Dim(); ++a)
          ws_.S1n_c[a] += ws_.cshift[idc][a] * ei;
      }
      if (badc || Zc <= kTiny_) {
        alpha *= Scalar(0.5);
        if (alpha < alphaMin)
          break;
        continue;
      }

      ws_.mcand[0] = ws_.S1n_c[0] / Zc;
      ws_.mcand[1] = ws_.S1n_c[1] / Zc;
      const Scalar r0_cand = ws_.mcand[0] - ws_.m_target[0];
      const Scalar r1_cand = ws_.mcand[1] - ws_.m_target[1];
      const Scalar rn_cand = sqrt(r0_cand * r0_cand + r1_cand * r1_cand);

      if (rn_cand < bestRn) {
        bestRn = rn_cand;
        ws_.bestXi[0] = ws_.xicand[0];
        ws_.bestXi[1] = ws_.xicand[1];
        accepted = true;
        break;
      }
      alpha *= Scalar(0.5);
      if (alpha < alphaMin)
        break;
    }

    // Fallback to tiny step
    if (!accepted) {
      ws_.xi[0] += Scalar(1e-6) * ws_.delta[0];
      ws_.xi[1] += Scalar(1e-6) * ws_.delta[1];
    } else {
      ws_.xi[0] = ws_.bestXi[0];
      ws_.xi[1] = ws_.bestXi[1];
    }

    // Check small delta
    const Scalar abs_d0 = fabs(ws_.delta[0]);
    const Scalar abs_d1 = fabs(ws_.delta[1]);
    if ((((abs_d0) > (abs_d1)) ? (abs_d0) : (abs_d1)) < kTiny_) {
      xiConverged = true;
      break;
    }
  }

  // If xi converged: compute chi and geq, else fallback
  bool success = false;
  if (xiConverged) {
    // Final compute of e_i and Z using xi (with smax factoring)
    Scalar smax = -HUGE_VAL;
#pragma omp simd
    for (Index idc = 0; idc < Speeds(); ++idc) {
      Scalar dot = Scalar(0);
      for (Index d = 0; d < Dim(); ++d)
        dot += ws_.xi[d] * ws_.cshift[idc][d];
      ws_.si[idc] = dot;
      if (ws_.si[idc] > smax)
        smax = ws_.si[idc];
    }

    if (smax > kMaxExp_)
      smax = kMaxExp_;
    if (smax < -kMaxExp_)
      smax = -kMaxExp_;

    for (Index i = 0; i < Speeds(); ++i) { ws_.e[i] = Scalar(0); }
    Scalar Z = Scalar(0);
#pragma omp simd
    for (Index idc = 0; idc < Speeds(); ++idc) {
      Scalar expo = ws_.si[idc] - smax;

      if (expo > kMaxExp_)
        expo = kMaxExp_;
      if (expo < -kMaxExp_)
        expo = -kMaxExp_;

      Scalar ev = exp(expo);
      if (!isfinite(ev)) {
        Z = Scalar(0);
        break;
      }

      ws_.e[idc] = ws_.Wi[idc] * ev;
      Z += ws_.e[idc];
    }

    if (Z > kTiny_) {
      // Scale factor so that sum geq = targetE: geq_i = (targetE / Z) * e_i
      const Scalar scaleFactor = targetE / Z;
#pragma omp simd
      for (Index idc = 0; idc < Speeds(); ++idc)
        geq[idc] = scaleFactor * ws_.e[idc];

      // Final sanity test: recompute S1 and check it matches targetQ (within FP)
      for (Index i = 0; i < Dim(); ++i) { ws_.S1final[i] = Scalar(0); }
      Scalar S0final = Scalar(0);
#pragma omp simd
      for (Index idc = 0; idc < Speeds(); ++idc) {
        S0final += geq[idc];
        for (Index d = 0; d < Dim(); ++d)
          ws_.S1final[d] += ws_.cshift[idc][d] * geq[idc];
      }

      // Small numerical tolerance check
      const Scalar s1_final_0_minus_target = ws_.S1final[0] - ws_.targetQ[0];
      const Scalar s1_final_1_minus_target = ws_.S1final[1] - ws_.targetQ[1];
      const Scalar s1_minus_target_norm =
        sqrt(s1_final_0_minus_target * s1_final_0_minus_target + s1_final_1_minus_target * s1_final_1_minus_target);
      const Scalar targetQ_norm = sqrt(ws_.targetQ[0] * ws_.targetQ[0] + ws_.targetQ[1] * ws_.targetQ[1]);
      const Scalar tol_check = Scalar(1e-10) * (((Scalar(1.0)) > (fabs(targetE))) ? (Scalar(1.0)) : (fabs(targetE)));
      if (fabs(S0final - targetE) < tol_check && s1_minus_target_norm / (((Scalar(1.0)) > (targetQ_norm)) ? (Scalar(1.0)) : (targetQ_norm)) < Scalar(1e-9)) {
        // Success; store multipliers for warm start (chi recovered from scaleFactor and smax if wanted)

        // Compute chi from scaleFactor relation: scaleFactor = rho * exp(chi + smax)
        // => chi = log(scaleFactor / rho) - smax
        const Scalar chi_val = log(scaleFactor / (*rho)) - smax;
        lastGx[0] = chi_val;
        lastGx[1] = ws_.xi[0];
        lastGx[2] = ws_.xi[1];
        success = true;
      } else {
        // Numerical mismatch (rare): fall through to fallback below
      }
    }
  }

  // Fallback: conservative normalized-Wi distribution (if solver failed)
  if (!success) {
    for (Index iddP1 = 0; iddP1 < Dim() + 1; ++iddP1)
      lastGx[iddP1] = Scalar(0.0);

    const Scalar total = Scalar(2.0) * (*rho) * E;
    Scalar sumW = Scalar(0);
#pragma omp simd
    for (Index idc = 0; idc < Speeds(); ++idc)
      sumW += ws_.Wi[idc];

    if (sumW <= kTiny_) {
      const Scalar uni = total / Scalar(Speeds());
#pragma omp simd
      for (Index idc = 0; idc < Speeds(); ++idc)
        geq[idc] = uni;
    } else {
      const Scalar uni = total / sumW;
#pragma omp simd
      for (Index idc = 0; idc < Speeds(); ++idc)
        geq[idc] = ws_.Wi[idc] * uni;
    }
  }
}

#pragma omp end declare target
} // namespace lbmini::openmp

#endif // LBMINI_OPENMP_LBMD2Q9_HPP_
