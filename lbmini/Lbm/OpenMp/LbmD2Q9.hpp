#ifndef LBMINI_OPENMP_LBMD2Q9_HPP_
#define LBMINI_OPENMP_LBMD2Q9_HPP_

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

#include "Data/ControlData.hpp"
#include "Data/FluidData.hpp"
#include "Lbm/LbmBase.hpp"

namespace lbmini::openmp {
template<typename Scalar_>
class LbmD2Q9 : public LbmClassBase<Scalar_, 2, 9> {
public:
  using Index = Eigen::Index;

  template<typename Type, Index Size>
  using Array = Eigen::Array<Type, Size, 1>;

  template<typename Type, Index Size>
  using Vector = Eigen::Vector<Type, Size>;

  template<typename Type, Index Rows, Index Cols>
  using Matrix = Eigen::Matrix<Type, Rows, Cols>;

  template<typename Type, Index NumIndices>
  using Tensor = Eigen::Tensor<Type, NumIndices>;

  using Base = LbmClassBase<Scalar_, 2, 9>;
  using Base::Dim, Base::Speeds;

  LbmD2Q9() = default;

  LbmD2Q9(const FluidData<Scalar_>* pFluid, const ControlData<Scalar_>* pControl);

  LbmD2Q9(const LbmD2Q9&) = default;

  LbmD2Q9(LbmD2Q9&&) = default;

  ~LbmD2Q9() override = default;

  LbmD2Q9& operator=(const LbmD2Q9&) = default;

  LbmD2Q9& operator=(LbmD2Q9&&) = default;

  Scalar_& U(const Index id) { return u_(id); }
  const Scalar_& U(const Index id) const { return u_(id); }

  Scalar_& P() { return p_; }
  const Scalar_& P() const { return p_; }

  Scalar_& Rho() { return rho_; }
  const Scalar_& Rho() const { return rho_; }

  Scalar_& Tem() { return tem_; }
  const Scalar_& Tem() const { return tem_; }

  Scalar_& F(const Index ic) { return f_(ic); }
  const Scalar_& F(const Index ic) const { return f_(ic); }

  Scalar_& G(const Index ic) { return g_(ic); }
  const Scalar_& G(const Index ic) const { return g_(ic); }

  Scalar_& Feq(const Index ic) { return feq_(ic); }
  const Scalar_& Feq(const Index ic) const { return feq_(ic); }

  Scalar_& Geq(const Index ic) { return geq_(ic); }
  const Scalar_& Geq(const Index ic) const { return geq_(ic); }

  Index Velocities(Index index, Index dir) const override { return kVelocities_(index, dir); }

  Scalar_ Weights(Index index) const override {
    Scalar_ weight;
    if (index == Index(0))
      weight = Scalar_(1.0) - tem_;
    else
      weight = Scalar_(0.5) * tem_;

    return weight;
  }

  Index Opposite(Index index) const override { return kOpposite_(index); }

  void Init(const Vector<Scalar_, Dim()>& u0, Scalar_ rho0, Scalar_ p0);

  void ComputeMacroscopic();

  void ComputeFeq();

  void ComputeGeq();

  void Collision();

private:
  // Constants
  inline static const Scalar_ kTiny_ = Scalar_(1.0e-12);
  inline static const Scalar_ kMaxExp_ = Scalar_(700.0);
  inline static const Scalar_ kCs2_ = Scalar_(1.0) / Scalar_(3.0);

  inline static const Matrix<Index, Speeds(), Dim()> kVelocities_{
    { 0, 0 }, { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }, { 1, 1 }, { -1, -1 }, { 1, -1 }, { -1, 1 },
  };

  inline static const Vector<Index, Speeds()> kOpposite_{
    0, 2, 1, 4, 3, 6, 5, 8, 7,
  };

  // Simulation data
  const FluidData<Scalar_>* pFluid_;
  const ControlData<Scalar_>* pControl_;

  // Macroscopic
  Vector<Scalar_, Dim()> u_;
  Scalar_ rho_;
  Scalar_ p_;
  Scalar_ tem_;

  // Distributions
  Vector<Scalar_, Speeds()> f_;
  Vector<Scalar_, Speeds()> feq_;
  Vector<Scalar_, Speeds()> g_;
  Vector<Scalar_, Speeds()> geq_;

  // Newton multipliers cache
  bool hasLastGx() const { return lastGxValid_; }
  Vector<Scalar_, 3> lastGx() const { return lastGx_; }
  void setLastGx(const Vector<Scalar_, 3>& x) {
    lastGx_ = x;
    lastGxValid_ = true;
  }

  Vector<Scalar_, 3> lastGx_;
  bool lastGxValid_;
};

template<typename Scalar_>
LbmD2Q9<Scalar_>::LbmD2Q9(const FluidData<Scalar_>* pFluid, const ControlData<Scalar_>* pControl) {
  // Data
  pFluid_ = pFluid;
  pControl_ = pControl;

  // Macroscopic
  u_.setZero();
  rho_ = Scalar_(0.0);
  p_ = Scalar_(0.0);
  tem_ = Scalar_(0.0);

  // Distributions
  f_.setZero();
  feq_.setZero();
  g_.setZero();
  geq_.setZero();

  // Newton multipliers cache
  lastGx_.setZero();
  lastGxValid_ = false;
}

template<typename Scalar_>
auto LbmD2Q9<Scalar_>::Init(const Vector<Scalar_, Dim()>& u0, Scalar_ rho0, Scalar_ p0) -> void {
  // Initialize velocity
  for (Index idd = 0; idd < Dim(); ++idd)
    u_(idd) = u0(idd);

  // Initialize density and pressure
  rho_ = rho0;
  p_ = p0;

  // Initialize temperature
  tem_ = kCs2_ * p_ / (rho_ * pFluid_->constant);

  // Initialize f to feq
  ComputeFeq();
  for (Index idc = 0; idc < Speeds(); ++idc)
    f_(idc) = feq_(idc);

  // Initialize g to geq
  // Invalidate lastX_ at initialization to avoid accidental reuse from previous sim
  lastGxValid_ = false;
  ComputeGeq();
  for (Index idc = 0; idc < Speeds(); ++idc)
    g_(idc) = geq_(idc);

  // Recompute macroscopic so derived fields are consistent after init
  ComputeMacroscopic();
}

template<typename Scalar_>
auto LbmD2Q9<Scalar_>::ComputeMacroscopic() -> void {
  // Compute rho (Eq. 7) and momentum (mom, Eq. 8)
  rho_ = Scalar_(0.0);
  // Compute total fluid energy (Eq. 9): S = sum_i g_i = 2 * rho * E
  Scalar_ nrg = Scalar_(0.0);
  // Compute moments
  Vector<Scalar_, Dim()> mom = Vector<Scalar_, Dim()>::Zero();

  for (Index idc = 0; idc < Speeds(); ++idc) {
    rho_ += f_(idc);
    nrg += g_(idc);
    for (Index idd = 0; idd < Dim(); ++idd) {
      Scalar_ ci = static_cast<Scalar_>(Velocities(idc, idd)) + pControl_->U(idd);
      mom(idd) += ci * f_(idc);
    }
  }

  Scalar_ nrgKinect = Scalar_(0.0);
  for (Index idd = 0; idd < Dim(); ++idd) {
    u_(idd) = mom(idd) / rho_;
    nrgKinect += u_(idd) * u_(idd);
  }

  nrg /= Scalar_(2.0) * rho_;
  nrgKinect /= Scalar_(2.0);

  // Compute temperature (Eq. 10)
  tem_ = (nrg - nrgKinect) / pFluid_->specificHeatCv;

  // Compute pressure (ideal gas law)
  p_ = pFluid_->constant * rho_ * tem_;
}

template<typename Scalar_>
auto LbmD2Q9<Scalar_>::ComputeFeq() -> void {
  auto phiAxis = [&](Index vi, Scalar_ uia, Scalar_ tem) -> Scalar_ {
    const Scalar_ uia2 = uia * uia;
    if (vi == Index(0))
      return Scalar_(1.0) - (uia2 + tem);
    if (vi == Index(1))
      return Scalar_(0.5) * (uia + uia2 + tem);
    if (vi == Index(-1))
      return Scalar_(0.5) * (-uia + uia2 + tem);
    return Scalar_(0.0);
  };

  for (Index idc = 0; idc < Speeds(); ++idc) {
    feq_(idc) = rho_;
    for (Index idd = 0; idd < Dim(); ++idd) {
      Scalar_ ue = u_(idd) - pControl_->U(idd);
      feq_(idc) *= phiAxis(Velocities(idc, idd), ue, tem_);
    }
  }
}

template<typename Scalar_>
auto LbmD2Q9<Scalar_>::ComputeGeq() -> void {
  // Macroscopic quantities
  Scalar_ u2 = Scalar_(0.0);
  for (Index d = 0; d < Dim(); ++d)
    u2 += u_(d) * u_(d);
  Scalar_ E = pFluid_->specificHeatCv * tem_ + Scalar_(0.5) * u2;

  // Precompute Wi (axis product) and shifted velocities cshift = c + U
  Vector<Scalar_, Speeds()> Wi;
  Wi.setZero();
  std::vector<Vector<Scalar_, Dim()>> cshift(static_cast<size_t>(Speeds()));
  for (Index idc = 0; idc < Speeds(); ++idc) {
    const Index vix = Velocities(idc, 0), viy = Velocities(idc, 1);
    const Scalar_ Wix = Weights(vix);
    const Scalar_ Wiy = Weights(viy);
    Wi(idc) = Wix * Wiy;
    for (Index d = 0; d < Dim(); ++d)
      cshift[static_cast<size_t>(idc)](d) = static_cast<Scalar_>(Velocities(idc, d)) + pControl_->U(d);
  }

  // Targets of the root-finding method
  const Scalar_ targetE = Scalar_(2.0) * rho_ * E; // sum_i geq_i
  Vector<Scalar_, Dim()> targetQ;
  for (Index d = 0; d < Dim(); ++d)
    targetQ(d) = Scalar_(2.0) * rho_ * u_(d) * (E + tem_);

  // Handle degenerate trivial case
  if (targetE <= Scalar_(0)) {
    // fallback: simple equal distribution
    const Scalar_ total = Scalar_(2.0) * rho_ * E;
    for (Index idc = 0; idc < Speeds(); ++idc)
      geq_(idc) = total / Scalar_(Speeds());
    lastGxValid_ = false;
    return;
  }

  // Desired mean (flux per unit energy)
  const Vector<Scalar_, Dim()> m_target = targetQ / targetE; // 2-vector

  // Solve for xi such that m(xi) = m_target
  // Initial guess for xi: use cached warm start if available, else zero
  Vector<Scalar_, Dim()> xi = Vector<Scalar_, Dim()>::Zero();
  if (lastGxValid_) {
    xi(0) = lastGx_(1);
    xi(1) = lastGx_(2);
  }

  // Newton parameters
  const Index maxIter = 100;
  const Scalar_ tolRel = Scalar_(1e-12); // relative tolerance on mean residual
  const Scalar_ tiny = kTiny_;
  bool xiConverged = false;

  // Scaling used in relative residual
  Scalar_ mScale = std::max(Scalar_(1.0), m_target.norm());

  // Newton iterations
  for (Index iter = 0; iter < maxIter; ++iter) {
    // Compute s_i = xi · c_i and smax for numeric stability
    Vector<Scalar_, Speeds()> si;
    Scalar_ smax = -std::numeric_limits<Scalar_>::infinity();
    for (Index idc = 0; idc < Speeds(); ++idc) {
      Scalar_ dot = Scalar_(0);
      for (Index d = 0; d < Dim(); ++d)
        dot += xi(d) * cshift[static_cast<size_t>(idc)](d);
      si(idc) = dot;
      if (si(idc) > smax)
        smax = si(idc);
    }

    if (smax > kMaxExp_)
      smax = kMaxExp_;
    if (smax < -kMaxExp_)
      smax = -kMaxExp_;

    // Build weighted exponentials e_i = Wi * exp(si - smax)
    Vector<Scalar_, Speeds()> e;
    e.setZero();
    Scalar_ Z = Scalar_(0);                                                    // sum e_i
    Vector<Scalar_, Dim()> S1n = Vector<Scalar_, Dim()>::Zero();               // sum ci * e_i
    Matrix<Scalar_, Dim(), Dim()> S2n = Matrix<Scalar_, Dim(), Dim()>::Zero(); // sum ci ci^T * e_i

    bool bad = false;
    for (Index idc = 0; idc < Speeds(); ++idc) {
      Scalar_ expo = si(idc) - smax;
      if (expo > kMaxExp_)
        expo = kMaxExp_;
      if (expo < -kMaxExp_)
        expo = -kMaxExp_;
      Scalar_ ev = std::exp(expo);
      if (!std::isfinite(ev)) {
        bad = true;
        break;
      }
      e(idc) = Wi(idc) * ev;
      Z += e(idc);
      for (Index a = 0; a < Dim(); ++a) {
        S1n(a) += cshift[static_cast<size_t>(idc)](a) * e(idc);
        for (Index b = 0; b < Dim(); ++b)
          S2n(a, b) += cshift[static_cast<size_t>(idc)](a) * cshift[static_cast<size_t>(idc)](b) * e(idc);
      }
    }
    if (bad || Z <= tiny)
      break;

    // m(xi) = S1n / Z
    Vector<Scalar_, Dim()> mxi = S1n / Z;

    // Residual r = mxi - m_target
    Vector<Scalar_, Dim()> r = mxi - m_target;
    const Scalar_ rn = r.norm();
    if (rn / mScale < tolRel) {
      xiConverged = true;
      break;
    }

    // Jacobian J = Cov = S2n / Z - mxi * mxi^T  (2x2 SPD)
    Matrix<Scalar_, Dim(), Dim()> J = (S2n / Z) - (mxi * mxi.transpose());

    // If J is nearly singular, add tiny LM regularizer
    Scalar_ lm = Scalar_(0.0);
    Eigen::LDLT<Matrix<Scalar_, Dim(), Dim()>> ldlt;
    ldlt.compute(J);
    if (ldlt.info() != Eigen::Success || std::abs(ldlt.rcond()) < Scalar_(1e-16)) {
      // Add small diagonal (Levenberg-Marquardt) and retry
      lm = std::max(Scalar_(1.0e-12), Scalar_(1.0e-6) * J.trace());
      Matrix<Scalar_, Dim(), Dim()> Jreg = J;
      Jreg(0, 0) += lm;
      Jreg(1, 1) += lm;
      ldlt.compute(Jreg);
      if (ldlt.info() != Eigen::Success) {
        // Cannot solve, break to fallback
        break;
      }
    }

    // Newton step: delta = -J^{-1} r
    Matrix<Scalar_, Dim(), 1> delta = ldlt.solve(-r);
    if (!delta.allFinite())
      break;

    // Damped line-search on xi: try full step then backtrack if residual norm not improved
    Scalar_ alpha = Scalar_(1.0);
    const Scalar_ alphaMin = Scalar_(1.0e-12);
    Scalar_ bestRn = rn;
    Vector<Scalar_, Dim()> bestXi = xi;
    bool accepted = false;
    for (int ls = 0; ls < 20; ++ls) {
      Vector<Scalar_, Dim()> xicand = xi + alpha * delta;

      // Quick evaluation of residual at xicand (repeat same routine)
      // Compute s_i, smax, e_i, Z, S1n
      Scalar_ candSmax = -std::numeric_limits<Scalar_>::infinity();
      for (Index idc = 0; idc < Speeds(); ++idc) {
        Scalar_ dot = Scalar_(0);
        for (Index d = 0; d < Dim(); ++d)
          dot += xicand(d) * cshift[static_cast<size_t>(idc)](d);
        if (dot > candSmax)
          candSmax = dot;
      }
      if (candSmax > kMaxExp_)
        candSmax = kMaxExp_;
      if (candSmax < -kMaxExp_)
        candSmax = -kMaxExp_;

      Scalar_ Zc = Scalar_(0);
      Vector<Scalar_, Dim()> S1n_c = Vector<Scalar_, Dim()>::Zero();
      bool badc = false;
      for (Index idc = 0; idc < Speeds(); ++idc) {
        Scalar_ dot = Scalar_(0);
        for (Index d = 0; d < Dim(); ++d)
          dot += xicand(d) * cshift[static_cast<size_t>(idc)](d);
        Scalar_ expo = dot - candSmax;
        if (expo > kMaxExp_)
          expo = kMaxExp_;
        if (expo < -kMaxExp_)
          expo = -kMaxExp_;
        Scalar_ ev = std::exp(expo);
        if (!std::isfinite(ev)) {
          badc = true;
          break;
        }
        Scalar_ ei = Wi(idc) * ev;
        Zc += ei;
        for (Index a = 0; a < Dim(); ++a)
          S1n_c(a) += cshift[static_cast<size_t>(idc)](a) * ei;
      }
      if (badc || Zc <= tiny) {
        alpha *= Scalar_(0.5);
        if (alpha < alphaMin)
          break;
        continue;
      }

      Vector<Scalar_, Dim()> mcand = S1n_c / Zc;
      const Scalar_ rn_cand = (mcand - m_target).norm();

      if (rn_cand < bestRn) {
        bestRn = rn_cand;
        bestXi = xicand;
        accepted = true;
        break;
      }
      alpha *= Scalar_(0.5);
      if (alpha < alphaMin)
        break;
    }

    // Fallback to tiny step
    if (!accepted) {
      xi += Scalar_(1e-6) * delta;
    } else {
      xi = bestXi;
    }

    // Check small delta
    if (delta.cwiseAbs().maxCoeff() < tiny) {
      xiConverged = true;
      break;
    }
  }

  // If xi converged: compute chi and geq, else fallback
  bool success = false;
  if (xiConverged) {
    // Final compute of e_i and Z using xi (with smax factoring)
    Vector<Scalar_, Speeds()> si;
    si.setZero();
    Scalar_ smax = -std::numeric_limits<Scalar_>::infinity();
    for (Index idc = 0; idc < Speeds(); ++idc) {
      Scalar_ dot = Scalar_(0);
      for (Index d = 0; d < Dim(); ++d)
        dot += xi(d) * cshift[static_cast<size_t>(idc)](d);
      si(idc) = dot;
      if (si(idc) > smax)
        smax = si(idc);
    }

    if (smax > kMaxExp_)
      smax = kMaxExp_;
    if (smax < -kMaxExp_)
      smax = -kMaxExp_;

    Vector<Scalar_, Speeds()> e;
    e.setZero();
    Scalar_ Z = Scalar_(0);
    for (Index idc = 0; idc < Speeds(); ++idc) {
      Scalar_ expo = si(idc) - smax;

      if (expo > kMaxExp_)
        expo = kMaxExp_;
      if (expo < -kMaxExp_)
        expo = -kMaxExp_;

      Scalar_ ev = std::exp(expo);
      if (!std::isfinite(ev)) {
        Z = Scalar_(0);
        break;
      }

      e(idc) = Wi(idc) * ev;
      Z += e(idc);
    }

    if (Z > tiny) {
      // Scale factor so that sum geq = targetE: geq_i = (targetE / Z) * e_i
      const Scalar_ scaleFactor = targetE / Z;
      for (Index idc = 0; idc < Speeds(); ++idc) {
        geq_(idc) = scaleFactor * e(idc);
      }

      // Final sanity test: recompute S1 and check it matches targetQ (within FP)
      Vector<Scalar_, Dim()> S1final = Vector<Scalar_, Dim()>::Zero();
      Scalar_ S0final = Scalar_(0);
      for (Index idc = 0; idc < Speeds(); ++idc) {
        S0final += geq_(idc);
        for (Index d = 0; d < Dim(); ++d)
          S1final(d) += cshift[static_cast<size_t>(idc)](d) * geq_(idc);
      }

      // Small numerical tolerance check
      const Scalar_ tol_check = Scalar_(1e-10) * std::max(Scalar_(1.0), std::abs(targetE));
      if (std::abs(S0final - targetE) < tol_check && (S1final - targetQ).norm() / std::max(Scalar_(1.0), targetQ.norm()) < Scalar_(1e-9)) {
        // Success; store multipliers for warm start (chi recovered from scaleFactor and smax if wanted)
        Vector<Scalar_, 3> Xnew;

        // Compute chi from scaleFactor relation: scaleFactor = rho * exp(chi + smax)
        // => chi = log(scaleFactor / rho) - smax
        const Scalar_ chi_val = std::log(scaleFactor / rho_) - smax;
        Xnew(0) = chi_val;
        Xnew(1) = xi(0);
        Xnew(2) = xi(1);
        lastGx_ = Xnew;
        lastGxValid_ = true;
        success = true;
      } else {
        // Numerical mismatch (rare): fall through to fallback below
      }
    }
  }

  // Fallback: conservative normalized-Wi distribution (if solver failed)
  if (!success) {
    lastGxValid_ = false;
    const Scalar_ total = Scalar_(2.0) * rho_ * E;
    Scalar_ sumW = Scalar_(0);
    for (Index idc = 0; idc < Speeds(); ++idc)
      sumW += Wi(idc);

    if (sumW <= tiny) {
      const Scalar_ uni = total / Scalar_(Speeds());
      for (Index idc = 0; idc < Speeds(); ++idc)
        geq_(idc) = uni;
    } else {
      for (Index idc = 0; idc < Speeds(); ++idc)
        geq_(idc) = (Wi(idc) / sumW) * total;
    }
  }
}

template<typename Scalar_>
auto LbmD2Q9<Scalar_>::Collision() -> void {
  Scalar_ u2 = Scalar_(0.0);
  for (Index idd = 0; idd < Dim(); ++idd)
    u2 += u_(idd) * u_(idd);

  // Compute distributions
  ComputeFeq();
  ComputeGeq();

  // Relaxation factor (Eq. 3)
  Scalar_ tau = pFluid_->viscosity / (rho_ * tem_) + Scalar_(0.5);
  Scalar_ omega = Scalar_(1.0) / tau;

  // Thermal relaxation factor
  const Scalar_ diffusivity = pFluid_->conductivity / (rho_ * pFluid_->specificHeatCp);
  const Scalar_ tauThermal = diffusivity / tem_ + Scalar_(0.5);
  Scalar_ omegaThermal = Scalar_(1.0) / tauThermal;

  // Knudsen sensor epsilon (Eq. 19)
  Scalar_ eps = Scalar_(0.0);
  for (Index idc = 0; idc < Speeds(); ++idc) {
    Scalar_ diff = f_(idc) - feq_(idc);
    eps += std::abs(diff) / std::max(feq_(idc), kTiny_);
  }
  eps /= Scalar_(Speeds());

  // sigma(ε) from (Eq. 20)
  Scalar_ sigma = Scalar_(1.0);
  if (eps >= Scalar_(1.0))
    sigma = omega;
  else if (eps >= Scalar_(1.0e-1))
    sigma = Scalar_(1.35);
  else if (eps >= Scalar_(1.0e-2))
    sigma = Scalar_(1.05);

  Scalar_ omegaLoc = omega / sigma;
  Scalar_ omegaThermalLoc = omegaThermal;

  // Clamp omegas to safe bounds
  // TODO: Review the omega clamp
  omegaLoc = std::min(std::max(omegaLoc, Scalar_(1.0)), Scalar_(2.0) - Scalar_(1.0e-7));
  omegaThermalLoc = std::min(std::max(omegaThermalLoc, Scalar_(1.0)), Scalar_(2.0) - Scalar_(1.0e-7));

  // Compute pressure tensor (Eq. 13a): P_ab = sum_i ci_a ci_b f_i
  Matrix<Scalar_, Dim(), Dim()> P;
  P.setZero();
  // Compute equilibrium pressure tensor (Eq. 11): P_ab_eq = rho u_a u_b + rho T delta_ab
  Matrix<Scalar_, Dim(), Dim()> Peq;
  Peq.setZero();
  // Compute L = u_b (P_ab - P_ab_eq)
  Vector<Scalar_, Dim()> L;
  L.setZero();
  for (Index a = 0; a < Dim(); ++a) {
    for (Index b = 0; b < Dim(); ++b) {
      for (Index idc = 0; idc < Speeds(); ++idc) {
        Scalar_ cia = static_cast<Scalar_>(Velocities(idc, a));
        Scalar_ cib = static_cast<Scalar_>(Velocities(idc, b));
        P(a, b) += (cia * cib) * f_(idc);
        Peq(a, b) += (cia * cib) * feq_(idc);
      }

      L(a) += Scalar_(2.0) * u_(b) * (P(a, b) - Peq(a, b));
    }
  }

  // Collisions
  for (Index idc = 0; idc < Speeds(); ++idc) {
    // f distribution
    f_(idc) += omegaLoc * (feq_(idc) - f_(idc));

    // Compute (g* - geq) per (Eq. 13): delta = Wi * ( ci · ( (P - Peq) * u ) ) / T
    Index vix = Velocities(idc, 0);
    Index viy = Velocities(idc, 1);
    Scalar_ Wix = Weights(vix);
    Scalar_ Wiy = Weights(viy);
    Scalar_ Wi = Wix * Wiy;

    // ci for projection uses shifted velocities (moments)
    Matrix<Scalar_, Dim(), 1> ci;
    ci(0) = static_cast<Scalar_>(Velocities(idc, 0)) + pControl_->U(0);
    ci(1) = static_cast<Scalar_>(Velocities(idc, 1)) + pControl_->U(1);

    Scalar_ cidotL = ci(0) * L(0) + ci(1) * L(1);
    Scalar_ gDiff = Wi * (cidotL / tem_);

    // Energy distribution
    g_(idc) += omegaLoc * (geq_(idc) - g_(idc)) + (omegaLoc - omegaThermalLoc) * gDiff;
  }
}
} // namespace lbmini::openmp

#endif // LBMINI_OPENMP_LBMD2Q9_HPP_
