#ifndef LBMINI_LBMTUBE_HPP_
#define LBMINI_LBMTUBE_HPP_

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "Data.hpp"

namespace lbmini {
template<typename Scalar_, typename LbmClassType_>
class LbmTube {
public:
  using Index = Eigen::Index;

  template<typename Type, Index Size>
  using Vector = Eigen::Vector<Type, Size>;

  template<typename Type, Index Rows, Index Cols>
  using Matrix = Eigen::Matrix<Type, Rows, Cols>;

  template<typename Type, Index NumIndices>
  using Tensor = Eigen::Tensor<Type, NumIndices>;

  LbmTube(
    const LbmClassType_& lbmClass,
    const Vector<Index, LbmClassType_::Dim()>& sizes,
    const FluidData<Scalar_>& fluidProps
  );

  auto P() const { return p_; }

  auto Rho() const { return rho_; }

  auto T() const { return tem_; }

  auto U() const { return u_; }

  void Init();

  void ComputeMacroscopic(Scalar_ dt);

  Scalar_ ComputeEps(const Vector<Index, LbmClassType_::Dim()>& idx) const;

  void ComputePsi();

  void Collision(Scalar_ dt);

  // HRR collision: now uses per-cell τ (with JST) and A(1,FD) computed with half-central/half-upwind derivatives
  void CollisionHRR(Scalar_ dt);

  // Energy collision (simple BGK fallback). Uses per-cell τ from p and eps_
  void CollisionEnergy(Scalar_ dt);

  void RegularizeMomentum();

  void RegularizeEnergy();

  void Streaming();

  void StreamingEnergy();

  void Step(Scalar_ dt);

protected:
  // Compute feq with second-order temperature(θ) correction (A(0)_αβ)
  Scalar_ computeFeq(Index iu, const Vector<Index, LbmClassType_::Dim()>& idx) const;

  Scalar_ diffCentral(const std::function<Scalar_(Index, Index, Index)>& field, const Vector<Index, LbmClassType_::Dim()>& idx, Index ax) const;

  void iterateDim(const std::function<void(const Vector<Index, LbmClassType_::Dim()>&)>& func);

private:
  const LbmClassType_& kLbmClass_;
  const Vector<Index, LbmClassType_::Dim()> kSizes_;
  const FluidData<Scalar_> kFluidProps_;

  // Macroscopic fields
  Tensor<Scalar_, LbmClassType_::Dim()> rho_; // Density
  Tensor<Scalar_, LbmClassType_::Dim()> p_;   // Pressure
  Tensor<Scalar_, LbmClassType_::Dim()> tem_; // Temperature
  Tensor<Scalar_, LbmClassType_::Dim()> s_;   // Entropy
  Tensor<Scalar_, LbmClassType_::Dim()> eps_; // JST sensor

  // Velocities
  Tensor<Scalar_, LbmClassType_::Dim() + 1> u_;

  // Distributions
  Tensor<Scalar_, LbmClassType_::Dim() + 1> f_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> fAux_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> g_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> gAux_;

  // psi correction with same layout as f
  Tensor<Scalar_, LbmClassType_::Dim() + 1> psi_;
};

template<typename Scalar_, typename LbmClassType_>
LbmTube<Scalar_, LbmClassType_>::LbmTube(
  const LbmClassType_& lbmClass,
  const Vector<Index, LbmClassType_::Dim()>& sizes,
  const FluidData<Scalar_>& fluidProps
)
  : kLbmClass_(lbmClass), kSizes_(sizes), kFluidProps_(fluidProps) {
  // Allocate memory for Tensor<Scalar_, (Nx, Ny, Nz)>
  {
    Eigen::array<Index, LbmClassType_::Dim()> dims3;
    dims3[0] = kSizes_[0];
    dims3[1] = kSizes_[1];
    dims3[2] = kSizes_[2];
    rho_ = Tensor<Scalar_, LbmClassType_::Dim()>(dims3);
    p_ = Tensor<Scalar_, LbmClassType_::Dim()>(dims3);
    tem_ = Tensor<Scalar_, LbmClassType_::Dim()>(dims3);
    s_ = Tensor<Scalar_, LbmClassType_::Dim()>(dims3);
    eps_ = Tensor<Scalar_, LbmClassType_::Dim()>(dims3);
  }

  // Allocate memory for Tensor<Scalar_, (Nx, Ny, Nz, dims)>
  {
    Eigen::array<Index, LbmClassType_::Dim() + 1> dims4;
    dims4[0] = kSizes_[0];
    dims4[1] = kSizes_[1];
    dims4[2] = kSizes_[2];
    dims4[3] = kLbmClass_.Dim();
    u_ = Tensor<Scalar_, LbmClassType_::Dim() + 1>(dims4);
  }

  // Allocate memory for Tensor<Scalar_, (Nx, Ny, Nz, speeds)>
  {
    Eigen::array<Index, LbmClassType_::Dim() + 1> dimsF;
    dimsF[0] = kSizes_[0];
    dimsF[1] = kSizes_[1];
    dimsF[2] = kSizes_[2];
    dimsF[3] = kLbmClass_.Speeds();
    f_ = Tensor<Scalar_, LbmClassType_::Dim() + 1>(dimsF);
    fAux_ = Tensor<Scalar_, LbmClassType_::Dim() + 1>(dimsF);
    g_ = Tensor<Scalar_, LbmClassType_::Dim() + 1>(dimsF);
    gAux_ = Tensor<Scalar_, LbmClassType_::Dim() + 1>(dimsF);
    psi_ = Tensor<Scalar_, LbmClassType_::Dim() + 1>(dimsF);
  }

  // Initialize memory to zero (Tensor default ctor leaves uninitialized)
  iterateDim(
    [&](const Vector<Index, LbmClassType_::Dim()>& idx) {
      const Index ix = idx[0], iy = idx[1], iz = idx[2];
      rho_(ix, iy, iz) = 0.0;
      p_(ix, iy, iz) = 0.0;
      tem_(ix, iy, iz) = 0.0;
      s_(ix, iy, iz) = 0.0;
      eps_(ix, iy, iz) = 0.0;

      for (Index d = 0; d < kLbmClass_.Dim(); ++d) {
        u_(ix, iy, iz, d) = 0.0;
      }

      for (Index iu = 0; iu < kLbmClass_.Speeds(); ++iu) {
        f_(ix, iy, iz, iu) = 0.0;
        fAux_(ix, iy, iz, iu) = 0.0;
        g_(ix, iy, iz, iu) = 0.0;
        gAux_(ix, iy, iz, iu) = 0.0;
        psi_(ix, iy, iz, iu) = 0.0;
      }
    }
  );
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::Init() {
  auto initializer = [&](const Vector<Index, LbmClassType_::Dim()>& idx) {
    // Assign values by reference
    const Index ix = idx[0], iy = idx[1], iz = idx[2];
    Scalar_& pRef = p_(ix, iy, iz);
    Scalar_& rhoRef = rho_(ix, iy, iz);
    Scalar_& temRef = tem_(ix, iy, iz);
    Scalar_ energy; // Initialize g to internal energy (E_internal = p/(gamma-1)), no kinetic

    // Apply shock tube condition along x-axis
    if (ix < kSizes_[0] / 2) {
      pRef = kFluidProps_.pressureL;
      rhoRef = kFluidProps_.densityL;
    } else {
      rhoRef = kFluidProps_.densityR;
      pRef = kFluidProps_.pressureR;
    }

    temRef = pRef / (rhoRef * kFluidProps_.constant);
    energy = pRef / (kFluidProps_.gamma - 1.0);

    // Zero initial velocity
    u_(ix, iy, iz, 0) = 0.0;
    u_(ix, iy, iz, 1) = 0.0;
    u_(ix, iy, iz, 2) = 0.0;

    // Initialize f to equilibrium using feq with theta correction
    for (Index iu = 0; iu < kLbmClass_.Speeds(); ++iu) {
      Scalar_ feq = computeFeq(iu, idx);
      f_(ix, iy, iz, iu) = feq;
      fAux_(ix, iy, iz, iu) = feq;
      g_(ix, iy, iz, iu) = kLbmClass_.Weights(iu) * energy;
      gAux_(ix, iy, iz, iu) = kLbmClass_.Weights(iu) * energy;
      psi_(ix, iy, iz, iu) = 0.0;
    }

    // Zero eps
    eps_(ix, iy, iz) = 0.0;
  };

  iterateDim(initializer);
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::ComputeMacroscopic(Scalar_ dt) {
  const Scalar_ cs2 = 1.0 / 3.0;
  const Scalar_ dt2 = dt * dt;

  auto macroscopicTranform = [&](const Vector<Index, LbmClassType_::Dim()>& idx) {
      const Index ix = idx[0], iy = idx[1], iz = idx[2];

      Scalar_ rho = 0.0;
      Vector<Scalar_, 3> mom = Vector<Scalar_, 3>::Zero();

      for (Index iu = 0; iu < kLbmClass_.Speeds(); ++iu) {
        Scalar_ fi = f_(ix, iy, iz, iu);
        rho += fi;
        for (Index d = 0; d < 3; ++d) {
          mom[d] += fi * kLbmClass_.Velocities(iu, d);
          mom[d] += dt2 * kLbmClass_.Velocities(iu, d) * psi_(ix, iy, iz, iu);
        }
      }
      rho_(ix, iy, iz) = rho;

      if (rho > 1.0e-14) {
        for (Index d = 0; d < 3; ++d)
          u_(ix, iy, iz, d) = mom[d] / rho;
      } else {
        for (Index d = 0; d < 3; ++d)
          u_(ix, iy, iz, d) = 0.0;
      }

      // Energy from g
      Scalar_ energy = 0.0;
      for (Index i = 0; i < kLbmClass_.Speeds(); ++i)
        energy += g_(ix, iy, iz, i);

      // Pressure: p = (gamma - 1) * (E - 0.5 * rho * u^2)
      Scalar_ u2 = u_(ix, iy, iz, 0) * u_(ix, iy, iz, 0) + u_(ix, iy, iz, 1) * u_(ix, iy, iz, 1) + u_(ix, iy, iz, 2) * u_(ix, iy, iz, 2);
      Scalar_ pressure = (kFluidProps_.gamma - 1.0) * (energy - 0.5 * rho * u2);
      p_(ix, iy, iz) = pressure;

      // Entropy (for diagnostics)
      s_(ix, iy, iz) = std::log(pressure / std::pow(rho, kFluidProps_.gamma));
  };

  iterateDim(macroscopicTranform);
}

template<typename Scalar_, typename LbmClassType_>
Scalar_ LbmTube<Scalar_, LbmClassType_>::ComputeEps(const Vector<Index, LbmClassType_::Dim()>& idx) const {
  const Scalar_ kappa = 1.0; // JST parameter

  auto pAt = [&](Index ix, Index iy, Index iz) -> Scalar_ {
    // X is clamped, Y/Z are periodic
    ix = std::clamp(ix, Index(0), kSizes_[0] - 1);
    iy = (iy + kSizes_[1]) % kSizes_[1];
    iz = (iz + kSizes_[2]) % kSizes_[2];
    return p_(ix, iy, iz);
  };

  const Index ix = idx[0], iy = idx[1], iz = idx[2];
  Scalar_ p0 = pAt(ix, iy, iz);

  // Axis offsets for (-1, +1) neighbors
  const Index offsets[6][3] = {
    { -1, 0, 0 },
    { +1, 0, 0 },
    // X-axis neighbors
    { 0, -1, 0 },
    { 0, +1, 0 },
    // Y-axis neighbors
    { 0, 0, -1 },
    { 0, 0, +1 } // Z-axis neighbors
  };

  Scalar_ epsMax = 0.0;

  for (Index axis = 0; axis < 3; ++axis) {
    // Get axis-specific neighbor indices
    const Index* neg = offsets[axis * 2 + 0];
    const Index* pos = offsets[axis * 2 + 1];

    Scalar_ pm1 = pAt(ix + neg[0], iy + neg[1], iz + neg[2]);
    Scalar_ pp1 = pAt(ix + pos[0], iy + pos[1], iz + pos[2]);

    Scalar_ denom = pm1 + 2.0 * p0 + pp1;
    if (std::abs(denom) < 1.0e-14)
      denom = 1.0e-14;

    Scalar_ eps_axis = kappa * std::abs(pm1 - 2.0 * p0 + pp1) / denom;
    epsMax = std::max(epsMax, eps_axis);
  }

  return epsMax;
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::ComputePsi() {
  // Compute psi correction
  // Uses the half-central/half-upwind stencil (Eq. A.18) if the JST sensor is triggered.
  const Scalar_ cs2 = 1.0 / 3.0;
  Scalar_ dx = 1.0 / static_cast<Scalar_>(kSizes_[0]);

  auto derivUpwindBiased = [&](
    const std::function<Scalar_(Index, Index, Index)>& field,
    const Vector<Index, LbmClassType_::Dim()>& idx,
    Index axis,
    Scalar_ u_axis,
    bool useUpwind
  ) -> Scalar_ {
    if (!useUpwind)
      return diffCentral(field, idx, axis);

    // upwind-biased half-central/half-upwind (A.18)
    const Index ix = idx[0], iy = idx[1], iz = idx[2];
    if (axis == 0) {
      if (u_axis > 0) {
        if (ix >= 2 && ix + 1 < kSizes_[0])
          return (field(ix - 2, iy, iz) - 5.0 * field(ix - 1, iy, iz) + 3.0 * field(ix, iy, iz) + field(ix + 1, iy, iz)) / (4.0 * dx);
        else
          return diffCentral(field, idx, axis);
      } else {
        if (ix + 2 < kSizes_[0] && ix >= 1)
          return (-field(ix + 2, iy, iz) + 5.0 * field(ix + 1, iy, iz) - 3.0 * field(ix, iy, iz) - field(ix - 1, iy, iz)) / (4.0 * dx);
        else
          return diffCentral(field, idx, axis);
      }
    } else if (axis == 1) {
      auto idxOffset = [&](Index offset) -> Index {
        return static_cast<Index>((iy + offset + kSizes_[1]) % kSizes_[1]);
      };
      if (u_axis > 0) {
        return (field(ix, idxOffset(-2), iz) - 5.0 * field(ix, idxOffset(-1), iz) + 3.0 * field(ix, idxOffset(0), iz) + field(ix, idxOffset(1), iz)) / (4.0 * dx);
      } else {
        return (-field(ix, idxOffset(2), iz) + 5.0 * field(ix, idxOffset(1), iz) - 3.0 * field(ix, idxOffset(0), iz) - field(ix, idxOffset(-1), iz)) / (4.0 * dx);
      }
    } else {
      auto idxOffset = [&](Index offset) -> Index {
        return static_cast<Index>((iz + offset + kSizes_[2]) % kSizes_[2]);
      };
      if (u_axis > 0) {
        return (field(ix, iy, idxOffset(-2)) - 5.0 * field(ix, iy, idxOffset(-1)) + 3.0 * field(ix, iy, idxOffset(0)) + field(ix, iy, idxOffset(1))) / (4.0 * dx);
      } else {
        return (-field(ix, iy, idxOffset(2)) + 5.0 * field(ix, iy, idxOffset(1)) - 3.0 * field(ix, iy, idxOffset(0)) - field(ix, iy, idxOffset(-1))) / (4.0 * dx);
      }
    }
  };

  // iterate and build psi and eps_
  auto psiOp = [&](const Vector<Index, LbmClassType_::Dim()>& idx) {
    const Index ix = idx[0], iy = idx[1], iz = idx[2];

    Scalar_ eps = ComputeEps(idx);
    eps_(ix, iy, iz) = eps;

    bool use_upwind = (eps > 1.0e-14);

    Scalar_ ux = u_(ix, iy, iz, 0);
    Scalar_ uy = u_(ix, iy, iz, 1);
    Scalar_ uz = u_(ix, iy, iz, 2);
    Scalar_ rhoCell = rho_(ix, iy, iz);
    Scalar_ pCell = p_(ix, iy, iz);

    // define fields used in Eq. (10)
    auto phi_x = [&](Index i, Index j, Index k) -> Scalar_ {
      Scalar_ rr = rho_(i, j, k);
      Scalar_ uxx = u_(i, j, k, 0);
      Scalar_ the = p_(i, j, k) / (rr * cs2);
      return rr * uxx * (1.0 - (p_(i, j, k) / (rr * cs2)) - uxx * uxx);
    };
    auto phi_x_cross = [&](Index i, Index j, Index k) -> Scalar_ {
      Scalar_ rr = rho_(i, j, k);
      return rr * u_(i, j, k, 0) * u_(i, j, k, 1) * u_(i, j, k, 2);
    };
    auto phi_y = [&](Index i, Index j, Index k) -> Scalar_ {
      Scalar_ rr = rho_(i, j, k);
      Scalar_ uyy = u_(i, j, k, 1);
      return rr * uyy * (1.0 - (p_(i, j, k) / (rr * cs2)) - uyy * uyy);
    };
    auto phi_y_cross = [&](Index i, Index j, Index k) -> Scalar_ {
      Scalar_ rr = rho_(i, j, k);
      return rr * u_(i, j, k, 0) * u_(i, j, k, 1) * u_(i, j, k, 2);
    };
    auto phi_z = [&](Index i, Index j, Index k) -> Scalar_ {
      Scalar_ rr = rho_(i, j, k);
      Scalar_ uzz = u_(i, j, k, 2);
      return rr * uzz * (1.0 - (p_(i, j, k) / (rr * cs2)) - uzz * uzz);
    };
    auto phi_z_cross = [&](Index i, Index j, Index k) -> Scalar_ {
      Scalar_ rr = rho_(i, j, k);
      return rr * u_(i, j, k, 0) * u_(i, j, k, 1) * u_(i, j, k, 2);
    };

    // derivatives (maybe upwind-biased)
    Scalar_ dPhiX_dx = derivUpwindBiased(phi_x, idx, 0, ux, use_upwind);
    Scalar_ dCrossX_dx = derivUpwindBiased(phi_x_cross, idx, 0, ux, use_upwind);
    Scalar_ dPhiY_dy = derivUpwindBiased(phi_y, idx, 1, uy, use_upwind);
    Scalar_ dCrossY_dy = derivUpwindBiased(phi_y_cross, idx, 1, uy, use_upwind);
    Scalar_ dPhiZ_dz = derivUpwindBiased(phi_z, idx, 2, uz, use_upwind);
    Scalar_ dCrossZ_dz = derivUpwindBiased(phi_z_cross, idx, 2, uz, use_upwind);

    for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
      Scalar_ H2_xx = kLbmClass_.Velocities(i, 0) * kLbmClass_.Velocities(i, 0) - cs2;
      Scalar_ H2_yy = kLbmClass_.Velocities(i, 1) * kLbmClass_.Velocities(i, 1) - cs2;
      Scalar_ H2_zz = kLbmClass_.Velocities(i, 2) * kLbmClass_.Velocities(i, 2) - cs2;
      Scalar_ H2_yz = kLbmClass_.Velocities(i, 1) * kLbmClass_.Velocities(i, 2);
      Scalar_ H2_xz = kLbmClass_.Velocities(i, 0) * kLbmClass_.Velocities(i, 2);
      Scalar_ H2_xy = kLbmClass_.Velocities(i, 0) * kLbmClass_.Velocities(i, 1);

      Scalar_ term = 0.0;
      term += H2_xx * dPhiX_dx - H2_yz * dCrossX_dx;
      term += H2_yy * dPhiY_dy - H2_xz * dCrossY_dy;
      term += H2_zz * dPhiZ_dz - H2_xy * dCrossZ_dz;

      psi_(ix, iy, iz, i) = kLbmClass_.Weights(i) * term / (2.0 * cs2 * cs2);
    }
  };

  iterateDim(psiOp);
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::Collision(Scalar_ dt) {
  const Scalar_ cs2 = 1.0 / 3.0;

  auto collisionOp = [&](const Vector<Index, LbmClassType_::Dim()>& idx) {
    const Index ix = idx[0], iy = idx[1], iz = idx[2];

    // τe = μ / p + max_eps * dt, τ = τe/dt + 0.5
    Scalar_ tau_e = kFluidProps_.viscosity / p_(ix, iy, iz) + eps_(ix, iy, iz) * dt;
    Scalar_ tau = tau_e / dt + 0.5;
    Scalar_ omega = 1.0 / tau;

    // BGK collision: relax toward equilibrium
    for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
      Scalar_ feq = computeFeq(i, idx);
      Scalar_& fRef = f_(ix, iy, iz, i);
      fRef -= omega * (fRef - feq);
    }
  };

  iterateDim(collisionOp);
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::CollisionHRR(Scalar_ dt) {
  // Compute psi from current macroscopic (will have been computed by caller)
  ComputePsi();

  const Scalar_ cs2 = 1.0 / 3.0;
  const Scalar_ cs4 = cs2 * cs2;
  const Scalar_ cs6 = cs4 * cs2;

  // HRR weight
  const Scalar_ sigma = 0.7;

  // d/dx helper for velocity derivatives with upwind choice
  auto dU = [&](
    Index ix,
    Index iy,
    Index iz,
    int ax,
    int beta,
    bool use_upwind
  ) -> Scalar_ {
    // build accessor for component beta
    auto field = [&](Index X, Index Y, Index Z) -> Scalar_ {
      return u_(X, Y, Z, beta);
    };
    // estimate local velocity in axis direction (for stencil orientation)
    Scalar_ u_axis = u_(ix, iy, iz, ax);
    // call same deriv_upwind_biased used in ComputePsi (copy small version)
    Scalar_ dx = 1.0 / static_cast<Scalar_>(kSizes_[0]);
    auto central = [&](Index i, Index j, Index k, int ax2) -> Scalar_ {
      if (ax2 == 0) {
        Index ip = std::min(i + 1, kSizes_[0] - 1);
        Index im = std::max<Index>(i - 1, 0);
        return (field(ip, j, k) - field(im, j, k)) / (2.0 * dx);
      } else if (ax2 == 1) {
        Index yp = (j + 1) % kSizes_[1];
        Index ym = (j + kSizes_[1] - 1) % kSizes_[1];
        return (field(i, yp, k) - field(i, ym, k)) / (2.0 * dx);
      } else {
        Index zp = (k + 1) % kSizes_[2];
        Index zm = (k + kSizes_[2] - 1) % kSizes_[2];
        return (field(i, j, zp) - field(i, j, zm)) / (2.0 * dx);
      }
    };

    if (!use_upwind)
      return central(ix, iy, iz, ax);

    if (ax == 0) {
      if (u_axis > 0) {
        if (ix >= 2 && ix + 1 < kSizes_[0]) {
          return (field(ix - 2, iy, iz) - 5.0 * field(ix - 1, iy, iz) + 3.0 * field(ix, iy, iz) + field(ix + 1, iy, iz)) / (4.0 * dx);
        } else
          return central(ix, iy, iz, ax);
      } else {
        if (ix + 2 < kSizes_[0] && ix >= 1) {
          return (-field(ix + 2, iy, iz) + 5.0 * field(ix + 1, iy, iz) - 3.0 * field(ix, iy, iz) - field(ix - 1, iy, iz)) / (4.0 * dx);
        } else
          return central(ix, iy, iz, ax);
      }
    } else if (ax == 1) {
      auto id = [&](int off) -> Index {
        return static_cast<Index>((iy + off + kSizes_[1]) % kSizes_[1]);
      };
      if (u_axis > 0) {
        return (field(ix, id(-2), iz) - 5.0 * field(ix, id(-1), iz) + 3.0 * field(ix, id(0), iz) + field(ix, id(1), iz)) / (4.0 * dx);
      } else {
        return (-field(ix, id(2), iz) + 5.0 * field(ix, id(1), iz) - 3.0 * field(ix, id(0), iz) - field(ix, id(-1), iz)) / (4.0 * dx);
      }
    } else {
      auto id = [&](int off) -> Index {
        return static_cast<Index>((iz + off + kSizes_[2]) % kSizes_[2]);
      };
      if (u_axis > 0) {
        return (field(ix, iy, id(-2)) - 5.0 * field(ix, iy, id(-1)) + 3.0 * field(ix, iy, id(0)) + field(ix, iy, id(1))) / (4.0 * dx);
      } else {
        return (-field(ix, iy, id(2)) + 5.0 * field(ix, iy, id(1)) - 3.0 * field(ix, iy, id(0)) - field(ix, iy, id(-1))) / (4.0 * dx);
      }
    }
  };

  auto collisionHRROperator = [&](const Vector<Index, LbmClassType_::Dim()>& idx) {
    const Index ix = idx[0], iy = idx[1], iz = idx[2];

    Scalar_ rho = rho_(ix, iy, iz);
    Scalar_ ux = u_(ix, iy, iz, 0);
    Scalar_ uy = u_(ix, iy, iz, 1);
    Scalar_ uz = u_(ix, iy, iz, 2);
    Scalar_ p_local = p_(ix, iy, iz);
    Scalar_ eps_local = eps_(ix, iy, iz);

    // local tau using Eq (23): τe = μ/p + max_eps*dt ; τ = τe/dt + 0.5
    Scalar_ tau_e = kFluidProps_.viscosity / p_local + eps_local * dt;
    Scalar_ tau = tau_e / dt + 0.5;
    if (tau <= 0.5)
      tau = 0.500001;
    Scalar_ omega = 1.0 / tau;
    Scalar_ relax_factor = 1.0 - 1.0 / tau; // (1 - 1/τ) per paper
    // note: for small tau < 1 this can be negative; JST artificial-vis
    // and sensible μ/p keep τ typically > 0.5

    // compute feq and fneq (with psi half correction)
    std::vector<Scalar_> feq_vec(kLbmClass_.Speeds());
    std::vector<Scalar_> fneq_vec(kLbmClass_.Speeds());
    for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
      feq_vec[i] = computeFeq(i, rho, idx);
      Scalar_ fcur = f_(ix, iy, iz, i);
      fneq_vec[i] = fcur - feq_vec[i] + 0.5 * dt * psi_(ix, iy, iz, i);
    }

    // kinetic A1 from fneq
    Matrix<Scalar_, 3, 3> A1 = Matrix<Scalar_, 3, 3>::Zero();
    for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
      for (Index a = 0; a < 3; ++a)
        for (Index b = 0; b < 3; ++b)
          A1(a, b) += kLbmClass_.Velocities(i, a) *
            kLbmClass_.Velocities(i, b) * fneq_vec[i];
    }

    // compute Chapman-Enskog FD estimate A1_FD using derivatives of u
    // (using half-central/half-upwind when needed)
    bool useUpwind = (eps_local > 1.0e-12);
    Matrix<Scalar_, 3, 3> A1_FD = Matrix<Scalar_, 3, 3>::Zero();
    // divergence
    Scalar_ divu = dU(ix, iy, iz, 0, 0, useUpwind) + dU(ix, iy, iz, 1, 1, useUpwind) + dU(ix, iy, iz, 2, 2, useUpwind);
    for (Index a = 0; a < 3; ++a) {
      for (Index b = 0; b < 3; ++b) {
        Scalar_ delta = (a == b) ? 1.0 : 0.0;
        Scalar_ term = dU(ix, iy, iz, a, b, useUpwind) + dU(ix, iy, iz, b, a, useUpwind) - (2.0 / 3.0) * delta * divu;
        A1_FD(a, b) = -tau * p_local * term;
      }
    }

    // hybrid HRR blend
    Matrix<Scalar_, 3, 3> A1_HRR = sigma * A1 + (1.0 - sigma) * A1_FD;

    // reconstruct R(f_neq) via second and approximated third order
    for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
      Scalar_ sec_sum = 0.0;
      for (Index a = 0; a < 3; ++a) {
        for (Index b = 0; b < 3; ++b) {
          Scalar_ delta = (a == b) ? 1.0 : 0.0;
          Scalar_ H2 = kLbmClass_.Velocities(i, a) * kLbmClass_.Velocities(i, b) - cs2 * delta;
          sec_sum += H2 * A1_HRR(a, b);
        }
      }
      Scalar_ second_piece = kLbmClass_.Weights(i) * sec_sum / (2.0 * cs4);

      // approximate third-order using A1_abc ≈ u_a A1_bc + u_b A1_ca +
      // u_c A1_ab
      Scalar_ third_sum = 0.0;
      for (Index a = 0; a < 3; ++a) {
        for (Index b = 0; b < 3; ++b) {
          for (Index c = 0; c < 3; ++c) {
            Scalar_ delta_term = kLbmClass_.Velocities(i, a) * ((b == c) ? 1.0 : 0.0) + kLbmClass_.Velocities(i, b) * ((c == a) ? 1.0 : 0.0) + kLbmClass_.Velocities(i, c) * ((a == b) ? 1.0 : 0.0);
            Scalar_ H3 = kLbmClass_.Velocities(i, a) * kLbmClass_.Velocities(i, b) * kLbmClass_.Velocities(i, c) - cs2 * delta_term;
            Scalar_ A1abc = ux * A1_HRR(b, c) + uy * A1_HRR(c, a) + uz * A1_HRR(a, b);
            third_sum += H3 * A1abc;
          }
        }
      }
      Scalar_ third_piece = kLbmClass_.Weights(i) * third_sum / (6.0 * cs6);

      Scalar_ R_i = second_piece + third_piece;
      Scalar_ newfi = feq_vec[i] + relax_factor * R_i + dt * dt * psi_(ix, iy, iz, i);
      fAux_(ix, iy, iz, i) = newfi;
    }
  };

  iterateDim(collisionHRROperator);

  // After forming fAux_ with post-collision values, swap into f_
  std::swap(f_, fAux_);
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::CollisionEnergy(Scalar_ dt) {
  const Scalar_ cs2 = 1.0 / 3.0;

  auto collE = [&](const Vector<Index, LbmClassType_::Dim()>& idx) {
    const Index ix = idx[0], iy = idx[1], iz = idx[2];
    // energy from g
    Scalar_ E = 0.0;
    for (Index i = 0; i < kLbmClass_.Speeds(); ++i)
      E += g_(ix, iy, iz, i);

    Scalar_ p_local = p_(ix, iy, iz);
    Scalar_ eps_local = eps_(ix, iy, iz);
    Scalar_ tau_e = kFluidProps_.viscosity / p_local + eps_local * dt;
    Scalar_ tau = tau_e / dt + 0.5;
    if (tau <= 0.5)
      tau = 0.500001;
    Scalar_ omega = 1.0 / tau;

    for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
      Scalar_ geq = kLbmClass_.Weights(i) * E;
      Scalar_& gcur = g_(ix, iy, iz, i);
      gcur -= omega * (gcur - geq);
    }
  };

  iterateDim(collE);
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::RegularizeMomentum() {
  const Scalar_ cs2 = 1.0 / 3.0;
  const Scalar_ cs4 = cs2 * cs2;

  auto regularizationOperator = [&](const Vector<Index, LbmClassType_::Dim()>& idx) {
    const Index ix = idx[0], iy = idx[1], iz = idx[2];
    Scalar_ rho = rho_(ix, iy, iz);
    Scalar_ ux = u_(ix, iy, iz, 0);
    Scalar_ uy = u_(ix, iy, iz, 1);
    Scalar_ uz = u_(ix, iy, iz, 2);

    Vector<Scalar_, LbmClassType_::Speeds()> fneq; // compile-time size allowed if C++ supports
    for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
      Scalar_ cu = 3.0 * (kLbmClass_.Velocities(i, 0) * ux + kLbmClass_.Velocities(i, 1) * uy + kLbmClass_.Velocities(i, 2) * uz);
      Scalar_ feq = kLbmClass_.Weights(i) * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy + uz * uz));
      fneq(i) = f_(ix, iy, iz, i) - feq;
    }

    Matrix<Scalar_, 3, 3> M = Matrix<Scalar_, 3, 3>::Zero();
    for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
      for (Index a = 0; a < 3; ++a) {
        for (Index b = 0; b < 3; ++b)
          M(a, b) += fneq(i) * kLbmClass_.Velocities(i, a) * kLbmClass_.Velocities(i, b);
      }
    }

    for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
      Scalar_ sum = 0.0;
      for (Index a = 0; a < 3; ++a) {
        for (Index b = 0; b < 3; ++b) {
          Scalar_ delta = (a == b) ? 1.0 : 0.0;
          sum += (kLbmClass_.Velocities(i, a) * kLbmClass_.Velocities(i, b) - cs2 * delta) * M(a, b);
        }
      }
      Scalar_ fneq_reg = kLbmClass_.Weights(i) * sum / (2.0 * cs4);

      Scalar_ cu = 3.0 * (kLbmClass_.Velocities(i, 0) * ux + kLbmClass_.Velocities(i, 1) * uy + kLbmClass_.Velocities(i, 2) * uz);
      Scalar_ feq = kLbmClass_.Weights(i) * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy + uz * uz));
      f_(ix, iy, iz, i) = feq + fneq_reg;
    }
  };

  iterateDim(regularizationOperator);
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::RegularizeEnergy() {
  // placeholder: use same regularization as momentum but on g
  const Scalar_ cs2 = 1.0 / 3.0;
  const Scalar_ cs4 = cs2 * cs2;

  auto regularizationOperator = [&](const Vector<Index, LbmClassType_::Dim()>& idx) {
    const Index ix = idx[0], iy = idx[1], iz = idx[2];
    Vector<Scalar_, LbmClassType_::Speeds()> gneq;
    for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
      Scalar_ geq = kLbmClass_.Weights(i) * g_(ix, iy, iz, i);
      gneq(i) = g_(ix, iy, iz, i) - geq;
    }

    Matrix<Scalar_, 3, 3> M = Matrix<Scalar_, 3, 3>::Zero();
    for (Index i = 0; i < kLbmClass_.Speeds(); ++i)
      for (Index a = 0; a < 3; ++a)
        for (Index b = 0; b < 3; ++b)
          M(a, b) += gneq(i) * kLbmClass_.Velocities(i, a) * kLbmClass_.Velocities(i, b);

    for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
      Scalar_ sum = 0;
      for (Index a = 0; a < 3; ++a) {
        for (Index b = 0; b < 3; ++b) {
          Scalar_ delta = (a == b) ? 1.0 : 0.0;
          sum += (kLbmClass_.Velocities(i, a) * kLbmClass_.Velocities(i, b) - cs2 * delta) * M(a, b);
        }
      }

      Scalar_ gneq_reg = kLbmClass_.Weights(i) * sum / (2.0 * cs4);
      Scalar_ geq = kLbmClass_.Weights(i) * g_(ix, iy, iz, i);
      g_(ix, iy, iz, i) = geq + gneq_reg;
    }
  };

  iterateDim(regularizationOperator);
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::Streaming() {
  auto streamingOp = [&](const Vector<Index, LbmClassType_::Dim()>& idx) {
    const Index ix = idx[0], iy = idx[1], iz = idx[2];
    for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
      Index xSrc = ix - kLbmClass_.Velocities(i, 0);
      Index ySrc = iy - kLbmClass_.Velocities(i, 1);
      Index zSrc = iz - kLbmClass_.Velocities(i, 2);

      // x bounce-back (no-slip) boundary
      if (xSrc < 0 || xSrc >= kSizes_[0]) {
        fAux_(ix, iy, iz, i) = f_(ix, iy, iz, kLbmClass_.Opposite(i));
        continue;
      }

      // y periodic boundary
      if (ySrc < 0)
        ySrc += kSizes_[1];
      else if (ySrc >= kSizes_[1])
        ySrc -= kSizes_[1];

      // z periodic
      if (zSrc < 0)
        zSrc += kSizes_[2];
      else if (zSrc >= kSizes_[2])
        zSrc -= kSizes_[2];

      // Stream
      fAux_(ix, iy, iz, i) = f_(xSrc, ySrc, zSrc, i);
    }
  };

  iterateDim(streamingOp);
  std::swap(f_, fAux_);
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::StreamingEnergy() {
  auto streamingOperatorE = [&](const Vector<Index, LbmClassType_::Dim()>& indices) {
    const Index ix = indices[0], iy = indices[1], iz = indices[2];
    for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
      // bounce back in x
      if ((ix == 0 && kLbmClass_.Velocities(i, 0) < 0) ||
        (ix == kSizes_[0] - 1 && kLbmClass_.Velocities(i, 0) > 0)) {
        gAux_(ix, iy, iz, i) = g_(ix, iy, iz, kLbmClass_.Opposite(i));
        continue;
      }

      Index xSrc = static_cast<Index>(ix - kLbmClass_.Velocities(i, 0));
      if (xSrc < 0 || xSrc >= kSizes_[0]) {
        gAux_(ix, iy, iz, i) = g_(ix, iy, iz, kLbmClass_.Opposite(i));
        continue;
      }

      Index ySrc = (iy - static_cast<Index>(kLbmClass_.Velocities(i, 1)) + kSizes_[1]) % kSizes_[1];
      Index zSrc = (iz - static_cast<Index>(kLbmClass_.Velocities(i, 2)) + kSizes_[2]) % kSizes_[2];

      gAux_(ix, iy, iz, i) = g_(xSrc, ySrc, zSrc, i);
    }
  };

  iterateDim(streamingOperatorE);
  std::swap(g_, gAux_);
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::Step(Scalar_ dt) {
  Collision(dt);
  //CollisionHRR(dt);
  Streaming();

  ComputeMacroscopic(dt);

  //ComputePsi();

  //RegularizeMomentum();

  //CollisionEnergy(dt);
  //StreamingEnergy();

  //RegularizeEnergy();

  //ComputeMacroscopic(dt);
}

template<typename Scalar_, typename LbmClassType_>
Scalar_ LbmTube<Scalar_, LbmClassType_>::computeFeq(Index iu, const Vector<Index, LbmClassType_::Dim()>& idx) const {
  const Scalar_ cs2 = 1.0 / 3.0;
  const Scalar_ cs4 = cs2 * cs2;
  const Scalar_ cs6 = cs4 * cs2;

  auto delta = [&](Index a, Index b) {
    return (a == b) ? 1.0 : 0.0;
  };

  Scalar_ ux = u_(idx[0], idx[1], idx[2], 0);
  Scalar_ uy = u_(idx[0], idx[1], idx[2], 1);
  Scalar_ uz = u_(idx[0], idx[1], idx[2], 2);
  Scalar_ rho = rho_(idx[0], idx[1], idx[2]);

  // TODO: Fix theta
  // Scalar_ theta = p_(idx[0], idx[1], idx[2]) / (rho * cs2);
  // Scalar_ theta = kFluidProps_.constant * tem_(idx[0], idx[1], idx[2]) / cs2;
  Scalar_ theta = 1.0;

  // Linear term
  Scalar_ ci_dot_u = kLbmClass_.Velocities(iu, 0) * ux + kLbmClass_.Velocities(iu, 1) * uy + kLbmClass_.Velocities(iu, 2) * uz;
  Scalar_ linear = ci_dot_u / cs2;

  // A0_αβ = ρ uα uβ + ρ cs2 (θ - 1) δαβ
  Eigen::Tensor<Scalar_, 2> A0_ab = Eigen::Tensor<Scalar_, 2>(3, 3);
  for (Index a = 0; a < 3; ++a) {
    for (Index b = 0; b < 3; ++b) {
      Scalar_ ua = (a == 0) ? ux : ((a == 1) ? uy : uz);
      Scalar_ ub = (b == 0) ? ux : ((b == 1) ? uy : uz);
      A0_ab(a, b) = ua * ub + cs2 * (theta - 1.0) * delta(a, b);
    }
  }

  // Second-order Hermite H2_iαβ = ciα ciβ - cs2 δαβ
  Scalar_ second = 0.0;
  for (Index a = 0; a < 3; ++a)
    for (Index b = 0; b < 3; ++b) {
      Scalar_ H2 = kLbmClass_.Velocities(iu, a) * kLbmClass_.Velocities(iu, b) - cs2;
      second += H2 * A0_ab(a, b);
    }
  second /= 2.0 * cs4;

  // A0_αβγ = ρ uα uβ uγ + ρ cs2 (θ - 1) δαβγ
  Eigen::Tensor<Scalar_, 3> A0_abc = Eigen::Tensor<Scalar_, 3>(3, 3, 3);
  for (Index a = 0; a < 3; ++a) {
    for (Index b = 0; b < 3; ++b) {
      for (Index c = 0; c < 3; ++c) {
        Scalar_ ua = (a == 0) ? ux : ((a == 1) ? uy : uz);
        Scalar_ ub = (b == 0) ? ux : ((b == 1) ? uy : uz);
        Scalar_ uc = (c == 0) ? ux : ((c == 1) ? uy : uz);
        Scalar_ ud = ua * delta(b, c) + ub * delta(a, c) + uc * delta(a, b);
        A0_abc(a, b, c) = ua * ub * uc + cs2 * (theta - 1.0) * ud;
      }
    }
  }

  // Third-order Hermite H3_iαβγ = ciα ciβ ciγ - cs2 δαβγ
  Scalar_ third = 0.0;
  for (Index a = 0; a < 3; ++a) {
    for (Index b = 0; b < 3; ++b) {
      for (Index c = 0; c < 3; ++c) {
        Scalar_ cd = kLbmClass_.Velocities(iu, a) * delta(b, c) + kLbmClass_.Velocities(iu, b) * delta(a, c) + kLbmClass_.Velocities(iu, c) * delta(a, b);
        Scalar_ H3 = kLbmClass_.Velocities(iu, a) * kLbmClass_.Velocities(iu, b) * kLbmClass_.Velocities(iu, c) - cs2 * cd;
        third += H3 * A0_abc(a, b, c);
      }
    }
  }
  third /= 6.0 * cs6;

  Scalar_ feq = kLbmClass_.Weights(iu) * rho * (1.0 + linear + second + third);

  return feq;
}

template<typename Scalar_, typename LbmClassType_>
Scalar_ LbmTube<Scalar_, LbmClassType_>::diffCentral(
  const std::function<Scalar_(Index, Index, Index)>& field,
  const Vector<Index, LbmClassType_::Dim()>& idx,
  Index ax
) const {
  // Half-central differentiation
  const Index ix = idx[0], iy = idx[1], iz = idx[2];
  Scalar_ dx = 1.0 / static_cast<Scalar_>(kSizes_[0]);

  if (ax == 0) {
    Index ip = std::min(ix + 1, kSizes_[0] - 1);
    Index im = std::max<Index>(ix - 1, 0);
    return (field(ip, iy, iz) - field(im, iy, iz)) / (2.0 * dx);
  }

  if (ax == 1) {
    Index yp = (iy + 1) % kSizes_[1];
    Index ym = (iy + kSizes_[1] - 1) % kSizes_[1];
    return (field(ix, yp, iz) - field(ix, ym, iz)) / (2.0 * dx);
  }

  if (ax == 2) {
    Index zp = (iz + 1) % kSizes_[2];
    Index zm = (iz + kSizes_[2] - 1) % kSizes_[2];
    return (field(ix, iy, zp) - field(ix, iy, zm)) / (2.0 * dx);
  }

  throw std::runtime_error("Invalid axis");
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::iterateDim(const std::function<void(const Vector<Index, LbmClassType_::Dim()>&)>& func) {
  Vector<Index, LbmClassType_::Dim()> indices = Vector<Index, LbmClassType_::Dim()>::Zero();

  while (true) {
    func(indices);

    Index dim = LbmClassType_::Dim() - 1;
    ++indices[dim];

    while (dim >= 0 && indices[dim] >= kSizes_[dim]) {
      indices[dim] = 0;
      --dim;
      if (dim >= 0)
        ++indices[dim];
    }
    if (dim < 0)
      break;
  }
}

} // namespace lbmini

#endif  // LBMINI_LBMTUBE_HPP_
