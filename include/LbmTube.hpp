#ifndef LBMINI_LBMTUBE_HPP_
#define LBMINI_LBMTUBE_HPP_

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "FluidProperties.hpp"

namespace lbmini {

using namespace Eigen;

// LBM simulation class for compressible shock tube (Sod problem)
// This class now holds two sets of distribution functions:
//   f_ for mass/momentum
//   g_ for energy
template <typename Scalar_, typename LbmClassType_>
class LbmTube {
public:
  LbmTube(const LbmClassType_& lbmClass, const Vector<Index, LbmClassType_::Dim()>& sizes, const FluidProperties<Scalar_>& fluidProps)
  : kLbmClass_(lbmClass), kSizes_(sizes), kFluidProps_(fluidProps) {
    rho_ = Tensor<Scalar_, LbmClassType_::Dim()>(std::array<Index, LbmClassType_::Dim()> { kSizes_[0], kSizes_[1], kSizes_[2] });
    p_ = Tensor<Scalar_, LbmClassType_::Dim()>(std::array<Index, LbmClassType_::Dim()> { kSizes_[0], kSizes_[1], kSizes_[2] });
    s_ = Tensor<Scalar_, LbmClassType_::Dim()>(std::array<Index, LbmClassType_::Dim()> { kSizes_[0], kSizes_[1], kSizes_[2] });
    u_ = Tensor<Scalar_, LbmClassType_::Dim() + 1>(std::array<Index, LbmClassType_::Dim() + 1> { kSizes_[0], kSizes_[1], kSizes_[2], kLbmClass_.Dim() });
    f_ = Tensor<Scalar_, LbmClassType_::Dim() + 1>(std::array<Index, LbmClassType_::Dim() + 1> { kSizes_[0], kSizes_[1], kSizes_[2], kLbmClass_.Speeds() });
    fAux_ = Tensor<Scalar_, LbmClassType_::Dim() + 1>(std::array<Index, LbmClassType_::Dim() + 1> { kSizes_[0], kSizes_[1], kSizes_[2], kLbmClass_.Speeds() });
    g_ = Tensor<Scalar_, LbmClassType_::Dim() + 1>(std::array<Index, LbmClassType_::Dim() + 1> { kSizes_[0], kSizes_[1], kSizes_[2], kLbmClass_.Speeds() });
    gAux_ = Tensor<Scalar_, LbmClassType_::Dim() + 1>(std::array<Index, LbmClassType_::Dim() + 1> { kSizes_[0], kSizes_[1], kSizes_[2], kLbmClass_.Speeds() });
  }

  auto Rho() { return rho_; }

  void Init() {
    auto initializer = [&](const Vector<Index, LbmClassType_::Dim()>& idx) {
      // Assign values by reference
      Scalar_& rhoRef = rho_(idx[0], idx[1], idx[2]);
      Scalar_& uxRef = u_(idx[0], idx[1], idx[2], 0);
      Scalar_& uyRef = u_(idx[0], idx[1], idx[2], 1);
      Scalar_& uzRef = u_(idx[0], idx[1], idx[2], 2);

      // Apply shock tube condition along x-axis
      if (idx[0] < kSizes_[0] / 2) {
        rhoRef = kFluidProps_.densityL;
      } else {
        rhoRef = kFluidProps_.densityR;
      }

      // Zero initial velocity
      uxRef = 0.0;
      uyRef = 0.0;
      uzRef = 0.0;

      // Initialize the momentum distribution f_ to its equilibrium
      for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
        // TODO: generalize for DnQm
        Scalar_ cu = 3.0 * (kLbmClass_.Velocities(i, 0) * uxRef + kLbmClass_.Velocities(i, 1) * uyRef + kLbmClass_.Velocities(i, 2) * uzRef);
        Scalar_ feq = kLbmClass_.Weights(i) * rhoRef * (1.0 + cu + 0.5 * cu * cu - 1.5 * (uxRef * uxRef + uyRef * uyRef + uzRef * uzRef));
        f_(idx[0], idx[1], idx[2], i) = feq;
      }

      // Initialize the energy distribution g_
      // For energy, use the equation: E = p / (γ - 1) [since u=0], where p is the initial pressure
      Scalar_ energy;
      if (idx[0] < kSizes_[0] / 2) {
        energy = kFluidProps_.pressureL / (kFluidProps_.gamma - 1.0);
      } else {
        energy = kFluidProps_.pressureR / (kFluidProps_.gamma - 1.0);
      }
      for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
        Scalar_ geq = kLbmClass_.Weights(i) * energy;
        g_(idx[0], idx[1], idx[2], i) = geq;
      }
    };

    iterateDim(initializer);
  }

  void Collision(Scalar_ dt) {
    // Compute relaxation time tau from viscosity.
    // For standard LBM, c_s^2 = 1/3
    Scalar_ cs2 = 1.0 / 3.0;
    Scalar_ dx = 1.0 / static_cast<Scalar_>(kSizes_[0]);
    Scalar_ visc_lat = kFluidProps_.viscosity * dt / (dx * dx);
    Scalar_ tau = 0.5 + visc_lat / cs2;
    Scalar_ omega = 1.0 / tau;

    // Safety checks
    if (tau <= static_cast<Scalar_>(0.5)) {
      throw std::runtime_error("Computed tau <= 0.5 (unstable). Reduce viscosity or dt/dx^2.");
    }

    auto collisionOperator = [&](const Vector<Index, LbmClassType_::Dim()>& idx) {
      // Assign values by reference
      Scalar_& rhoRef = rho_(idx[0], idx[1], idx[2]);
      Scalar_& uxRef = u_(idx[0], idx[1], idx[2], 0);
      Scalar_& uyRef = u_(idx[0], idx[1], idx[2], 1);
      Scalar_& uzRef = u_(idx[0], idx[1], idx[2], 2);

      for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
        // TODO: generalize for DnQm
        Scalar_ cu = 3.0 * (kLbmClass_.Velocities(i, 0) * uxRef + kLbmClass_.Velocities(i, 1) * uyRef + kLbmClass_.Velocities(i, 2) * uzRef);
        Scalar_ feq = kLbmClass_.Weights(i) * rhoRef * (1.0 + cu + 0.5 * cu * cu - 1.5 * (uxRef * uxRef + uyRef * uyRef + uzRef * uzRef));

        // BGK collision: relax toward equilibrium
        Scalar_& fRef = f_(idx[0], idx[1], idx[2], i);
        fRef -= omega * (fRef - feq);
      }
    };

    iterateDim(collisionOperator);
  }

  // void CollisionHRR(Scalar_ dt) {
  //   // Hybrid Recursive Regularized collision with forcing term.
  //   // Implements: f_i(x+ciδt, t+δt) = f_eq,i + (1-1/τ)*R(f_neq,i) + dt² * psi_i.
  //   // Compute dt² * psi_i correction (assumes dt_ is defined).
  //   ComputePsi(); // Update member psi_ with computed correction term.
  //
  //   constexpr Scalar_ cs2 = 1.0 / 3.0;
  //   constexpr Scalar_ cs4 = cs2 * cs2;
  //   constexpr Scalar_ cs6 = cs4 * cs2;
  //   constexpr Scalar_ sigma = 0.7; // TODO: HRR weighting parameter sigma (set based on flow regime, e.g., 0.7 for viscous supersonic).
  //
  //   auto collisionHRROperator = [&](const Vector<Index, LbmClassType_::Dim()>& indices) {
  //     Scalar_ rho = rho_(indices[0], indices[1], indices[2]);
  //     Scalar_ ux = u_(indices[0], indices[1], indices[2], 0);
  //     Scalar_ uy = u_(indices[0], indices[1], indices[2], 1);
  //     Scalar_ uz = u_(indices[0], indices[1], indices[2], 2);
  //
  //     // Compute local pressure p and then temperature parameter θ = p/(ρ c_s²)
  //     Scalar_ p = p_(indices[0], indices[1], indices[2]);
  //     Scalar_ theta = (rho > 1e-12) ? p / (rho * cs2) : 1.0; // Ideal gas law
  //
  //     // Compute second-order moments
  //     Tensor<Scalar_, 2> momentEq2(3, 3);
  //     momentEq2.setZero();
  //     for (Index i1 = 0; i1 < 3; ++i1) {
  //       for (Index i2 = 0; i2 < 3; ++i2) {
  //         Scalar_ delta = (i1 == i2) ? 1.0 : 0.0;
  //         momentEq2(i1, i2) = rho * u_(indices[0], indices[1], indices[2], i1) * u_(indices[0], indices[1], indices[2], i2) + rho * cs2 * (theta - 1.0) * delta;
  //       }
  //     }
  //
  //     // Compute third-order moments
  //     Tensor<Scalar_, 3> momentEq3(3, 3, 3);
  //     momentEq3.setZero();
  //     for (Index ix = 0; ix < 3; ++ix) {
  //       for (Index iy = 0; iy < 3; ++iy) {
  //         for (Index iz = 0; iz < 3; ++iz) {
  //           Scalar_ delta = ux * ((iy == iz) ? 1.0 : 0.0) + uy * ((ix == iz) ? 1.0 : 0.0) + uz * ((ix == iy) ? 1.0 : 0.0);
  //           momentEq3(ix, iy, iz) = rho * u_(indices[0], indices[1], indices[2], ix) * u_(indices[0], indices[1], indices[2], iy) * u_(indices[0], indices[1], indices[2], iz) + rho * cs2 * (theta - 1.0) * delta;
  //         }
  //       }
  //     }
  //
  //     // Compute improved equilibrium distribution f_eq using a third-order Grad-Hermite expansion
  //     Vector<Scalar_, LbmClassType_::Speeds()> feq, fneq;
  //     for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
  //       Scalar_ cu = kLbmClass_.Velocities(i, 0) * ux + kLbmClass_.Velocities(i, 1) * uy + kLbmClass_.Velocities(i, 2) * uz;
  //
  //       // Compute second-order Hermite polynomial
  //       Scalar_ hermit2 = 0.0;
  //       for (Index alpha = 0; alpha < 3; ++alpha) {
  //         for (Index beta = 0; beta < 3; ++beta) {
  //           Scalar_ delta = (alpha == beta) ? 1.0 : 0.0;
  //           hermit2 += kLbmClass_.Velocities(i, alpha) * kLbmClass_.Velocities(i, beta); // TODO: analyze - cs2 * delta;
  //         }
  //       }
  //       hermit2 -= cs2;
  //
  //       // Compute third-order Hermite polynomial
  //       Tensor<Scalar_, 3> hermit3(3, 3, 3);
  //       hermit3.setZero();
  //       for (Index ix = 0; ix < 3; ++ix) {
  //         for (Index iy = 0; iy < 3; ++iy) {
  //           for (Index iz = 0; iz < 3; ++iz) {
  //             Scalar_ delta = kLbmClass_.Velocities(i, ix) * ((iy == iz) ? 1.0 : 0.0) + kLbmClass_.Velocities(i, iy) * ((ix == iz) ? 1.0 : 0.0) + kLbmClass_.Velocities(i, iz) * ((ix == iy) ? 1.0 : 0.0);
  //             hermit3(ix, iy, iz) = kLbmClass_.Velocities(i, ix) * kLbmClass_.Velocities(i, iy) * kLbmClass_.Velocities(i, iz) - cs2 * delta;
  //           }
  //         }
  //       }
  //
  //       // Assemble second-order term
  //       Scalar_ term2 = 0.0;
  //       term2 += hermit2 * momentEq2 / (2.0 * cs4);
  //
  //       // Assemble third-order term
  //       Scalar_ term3 = 0.0;
  //       term3 += 3.0 * (hermit3(0, 0, 1) + hermit3(1, 2, 2)) * (momentEq3(0, 0, 1) + momentEq3(1, 2, 2));
  //       term3 += (hermit3(0, 0, 1) - hermit3(1, 2, 2)) * (momentEq3(0, 0, 1) - momentEq3(1, 2, 2));
  //       term3 += 3.0 * (hermit3(0, 2, 2) + hermit3(0, 1, 1)) * (momentEq3(0, 2, 2) + momentEq3(0, 1, 1));
  //       term3 += (hermit3(0, 2, 2) - hermit3(0, 1, 1)) * (momentEq3(0, 2, 2) - momentEq3(0, 1, 1));
  //       term3 += 3.0 * (hermit3(1, 1, 2) + hermit3(0, 0, 2)) * (momentEq3(1, 1, 2) + momentEq3(0, 0, 2));
  //       term3 += (hermit3(1, 1, 2) - hermit3(0, 0, 2)) * (momentEq3(1, 1, 2) - momentEq3(0, 0, 2));
  //       term3 /= (6.0 * cs6);
  //
  //       feq(i) = kLbmClass_.Weights(i) * (rho + rho * cu / cs2 + term2 + term3);
  //       fneq(i) = f_(indices[0], indices[1], indices[2], i) - feq(i) + 0.5 * dt * psi_(indices[0], indices[1], indices[2], i);
  //
  //       // Compute second-order non-equilibrium moments
  //       Tensor<Scalar_, 2> momentNEq2(3, 3);
  //       momentNEq2.setZero();
  //       for (Index i1 = 0; i1 < 3; ++i1) {
  //         for (Index i2 = 0; i2 < 3; ++i2) {
  //           momentNEq2(i1, i2) += kLbmClass_.Velocities(i, i1) * kLbmClass_.Velocities(i, i2) * fneq(i);
  //         }
  //       }
  //       momentNEq2 *= sigma; // Apply hybrid weighting.
  //
  //       // Compute third-order non-equilibrium moments
  //       Tensor<Scalar_, 3> momentNEq3(3, 3, 3);
  //       momentNEq3.setZero();
  //       for (Index ix = 0; ix < 3; ++ix) {
  //         for (Index iy = 0; iy < 3; ++iy) {
  //           for (Index iz = 0; iz < 3; ++iz) {
  //             momentNEq3(ix, iy, iz) += ux * momentNEq2(iy, iz);
  //             momentNEq3(ix, iy, iz) += uy * momentNEq2(iy, ix);
  //             momentNEq3(ix, iy, iz) += uz * momentNEq2(ix, iy);
  //           }
  //         }
  //       }
  //
  //     // Reconstruct the regularized non-equilibrium R(f_neq)
  //       Scalar_ reg_fneq = 0.0;
  //       for (Index ix = 0; ix < 3; ++ix) {
  //         for (Index iy = 0; iy < 3; ++iy) {
  //           Scalar_ delta = (ix == iy) ? 1.0 : 0.0;
  //           reg_fneq += (kLbmClass_.Velocities(i, ix) * kLbmClass_.Velocities(i, iy) - cs2 * delta) * A1(ix, iy);
  //         }
  //       }
  //       reg_fneq *= kLbmClass_.Weights(i) / (2.0 * cs4);
  //
  //       // Update distribution: include relaxation factor (1 - 1/τ), where ω = 1/τ.
  //       // Using our computed ω (omega), we have:
  //       f_(indices[0], indices[1], indices[2], i) = feq(i) + (1.0 - omega) * reg_fneq + dt_ * dt_ * psi_(indices[0], indices[1], indices[2], i);
  //     }
  //   };
  //
  //   iterateDim(collisionHRROperator);
  // }

  void Streaming() {
    // For each lattice node and each velocity direction, stream the distribution
    auto streamingOperator = [&](const Vector<Index, LbmClassType_::Dim()>& idx) {
      for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
        // TODO: generalize for DnQm
        Index xSrc = idx[0] - static_cast<Index>(kLbmClass_.Velocities(i, 0));
        Index ySrc = idx[1] - static_cast<Index>(kLbmClass_.Velocities(i, 1));
        Index zSrc = idx[2] - static_cast<Index>(kLbmClass_.Velocities(i, 2));

        // x no-slip condition (bounce-back)
        if (xSrc < 0 || xSrc >= kSizes_[0]) {
          fAux_(idx[0], idx[1], idx[2], i) = f_(idx[0], idx[1], idx[2], kLbmClass_.Opposite(i));
          continue;
        }

        // y periodic boundary
        if (ySrc < 0) {
          ySrc += kSizes_[1];
        } else if (ySrc >= kSizes_[1]) {
          ySrc -= kSizes_[1];
        }

        // z periodic boundary
        if (zSrc < 0) {
          zSrc += kSizes_[2];
        } else if (zSrc >= kSizes_[2]) {
          zSrc -= kSizes_[2];
        }

        // Stream
        fAux_(idx[0], idx[1], idx[2], i) = f_(xSrc, ySrc, zSrc, i);
      }
    };

    iterateDim(streamingOperator);
    std::swap(f_, fAux_);
  }

  void StreamingEnergy() {
    auto streamingOperatorE = [&](const Vector<Index, LbmClassType_::Dim()>& indices) {
      for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
        // TODO: generalize for DnQm
        // Todo: fix boundary conditions

        // x no-slip condition (bounce-back)
        if ((indices[0] == 0 && kLbmClass_.Velocities(i, 0) < 0) || (indices[0] == kSizes_[0] - 1 && kLbmClass_.Velocities(i, 0) > 0)) {
          gAux_(indices[0], indices[1], indices[2], i) = g_(indices[0], indices[1], indices[2], kLbmClass_.Opposite(i));
          continue;
        }

        // If the computed xSrc is out-of-bound (shouldn't occur if boundaries are properly handled), then perform bounce-back
        Index xSrc = indices[0] - static_cast<Index>(kLbmClass_.Velocities(i, 0));
        if (xSrc < 0 || xSrc >= kSizes_[0]) {
          gAux_(indices[0], indices[1], indices[2], i) = g_(indices[0], indices[1], indices[2], kLbmClass_.Opposite(i));
          continue;
        }

        // y and z periodic boundaries
        Index ySrc = (indices[1] - static_cast<Index>(kLbmClass_.Velocities(i, 1)) + kSizes_[1]) % kSizes_[1];
        Index zSrc = (indices[2] - static_cast<Index>(kLbmClass_.Velocities(i, 2)) + kSizes_[2]) % kSizes_[2];
        gAux_(indices[0], indices[1], indices[2], i) = g_(xSrc, ySrc, zSrc, i);
      }
    };

    iterateDim(streamingOperatorE);
    std::swap(g_, gAux_);
  }

  void RegularizeMomentum() {
    // Speed of sound squared in lattice units: c_s^2 = 1/3.
    const Scalar_ cs2 = 1.0 / 3.0;
    const Scalar_ cs4 = cs2 * cs2;

    auto regularizationOperator = [&](const Vector<Index, LbmClassType_::Dim()>& indices) {
      Scalar_ rho = rho_(indices[0], indices[1], indices[2]);
      Scalar_ ux = u_(indices[0], indices[1], indices[2], 0);
      Scalar_ uy = u_(indices[0], indices[1], indices[2], 1);
      Scalar_ uz = u_(indices[0], indices[1], indices[2], 2);

      // Compute equilibrium distribution and non-equilibrium parts
      Vector<Scalar_, LbmClassType_::Speeds()> fneq;
      for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
        Scalar_ cu = 3.0 * (kLbmClass_.Velocities(i, 0) * ux + kLbmClass_.Velocities(i, 1) * uy + kLbmClass_.Velocities(i, 2) * uz);
        Scalar_ feq = kLbmClass_.Weights(i) * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy + uz * uz));
        fneq(i) = f_(indices[0], indices[1], indices[2], i) - feq;
      }

      // Compute the non-equilibrium moment tensor M_{alpha,beta}
      Matrix<Scalar_, 3, 3> M = Matrix<Scalar_, 3, 3>::Zero();
      for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
        for (Index alpha = 0; alpha < 3; ++alpha) {
          for (Index beta = 0; beta < 3; ++beta) {
            M(alpha, beta) += fneq(i) * kLbmClass_.Velocities(i, alpha) * kLbmClass_.Velocities(i, beta);
          }
        }
      }

      // Reconstruct the regularized non-equilibrium distribution
      for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
        Scalar_ sum = 0.0;
        for (Index alpha = 0; alpha < 3; ++alpha) {
          for (Index beta = 0; beta < 3; ++beta) {
            Scalar_ delta = (alpha == beta) ? 1.0 : 0.0; // delta_{alpha,beta} is 1 if alpha==beta, else 0.
            sum += (kLbmClass_.Velocities(i, alpha) * kLbmClass_.Velocities(i, beta) - cs2 * delta) * M(alpha, beta);
          }
        }

        // The regularized non-equilibrium part.
        Scalar_ fneq_reg = kLbmClass_.Weights(i) * sum / (2.0 * cs4);

        // Recompute equilibrium distribution.
        Scalar_ cu = 3.0 * (kLbmClass_.Velocities(i, 0) * ux + kLbmClass_.Velocities(i, 1) * uy + kLbmClass_.Velocities(i, 2) * uz);
        Scalar_ feq = kLbmClass_.Weights(i) * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy + uz * uz));

        // Update the distribution with the equilibrium plus regularized non-equilibrium part.
        f_(indices[0], indices[1], indices[2], i) = feq + fneq_reg;
      }
    };

    iterateDim(regularizationOperator);
  }

  void RegularizeEnergy() {
    // Speed of sound squared in lattice units: c_s^2 = 1/3.
    const Scalar_ cs2 = 1.0 / 3.0;
    const Scalar_ cs4 = cs2 * cs2;

    auto regularizationOperator = [&](const Vector<Index, LbmClassType_::Dim()>& indices) {
      Scalar_ rho = rho_(indices[0], indices[1], indices[2]);
      Scalar_ ux = u_(indices[0], indices[1], indices[2], 0);
      Scalar_ uy = u_(indices[0], indices[1], indices[2], 1);
      Scalar_ uz = u_(indices[0], indices[1], indices[2], 2);

      // Compute equilibrium distribution and non-equilibrium parts
      Vector<Scalar_, LbmClassType_::Speeds()> fneq;
      for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
        Scalar_ cu = 3.0 * (kLbmClass_.Velocities(i, 0) * ux + kLbmClass_.Velocities(i, 1) * uy + kLbmClass_.Velocities(i, 2) * uz);
        Scalar_ feq = kLbmClass_.Weights(i) * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy + uz * uz));
        fneq(i) = f_(indices[0], indices[1], indices[2], i) - feq;
      }

      // Compute the non-equilibrium moment tensor M_{alpha,beta}
      Matrix<Scalar_, 3, 3> M = Matrix<Scalar_, 3, 3>::Zero();
      for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
        for (Index alpha = 0; alpha < 3; ++alpha) {
          for (Index beta = 0; beta < 3; ++beta) {
            M(alpha, beta) += fneq(i) * kLbmClass_.Velocities(i, alpha) * kLbmClass_.Velocities(i, beta);
          }
        }
      }

      // Reconstruct the regularized non-equilibrium distribution
      for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
        Scalar_ sum = 0.0;
        for (Index alpha = 0; alpha < 3; ++alpha) {
          for (Index beta = 0; beta < 3; ++beta) {
            Scalar_ delta = (alpha == beta) ? 1.0 : 0.0; // delta_{alpha,beta} is 1 if alpha==beta, else 0.
            sum += (kLbmClass_.Velocities(i, alpha) * kLbmClass_.Velocities(i, beta) - cs2 * delta) * M(alpha, beta);
          }
        }

        // The regularized non-equilibrium part.
        Scalar_ fneq_reg = kLbmClass_.Weights(i) * sum / (2.0 * cs4);

        // Recompute equilibrium distribution.
        Scalar_ cu = 3.0 * (kLbmClass_.Velocities(i, 0) * ux + kLbmClass_.Velocities(i, 1) * uy + kLbmClass_.Velocities(i, 2) * uz);
        Scalar_ feq = kLbmClass_.Weights(i) * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy + uz * uz));

        // Update the distribution with the equilibrium plus regularized non-equilibrium part.
        f_(indices[0], indices[1], indices[2], i) = feq + fneq_reg;
      }
    };

    iterateDim(regularizationOperator);
  }

  void ComputeMacroscopic() {
    auto macroscopicTranform = [&](const Vector<Index, LbmClassType_::Dim()>& idx) {
      // Assign values by reference
      Scalar_& uxRef = u_(idx[0], idx[1], idx[2], 0);
      Scalar_& uyRef = u_(idx[0], idx[1], idx[2], 1);
      Scalar_& uzRef = u_(idx[0], idx[1], idx[2], 2);

      Scalar_ rho = 0.0;
      Scalar_ ux = 0.0;
      Scalar_ uy = 0.0;
      Scalar_ uz = 0.0;

      for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
        // TODO: generalize for DnQm
        Scalar_ fi = f_(idx[0], idx[1], idx[2], i);
        rho += fi;
        ux += fi * kLbmClass_.Velocities(i, 0);
        uy += fi * kLbmClass_.Velocities(i, 1);
        uz += fi * kLbmClass_.Velocities(i, 2);
      }

      rho_(idx[0], idx[1], idx[2]) = rho;

      // Avoid division by zero
      if (rho > 1.0e-11) {
        uxRef = ux / rho;
        uyRef = uy / rho;
        uzRef = uz / rho;
      } else {
        uxRef = 0.0;
        uyRef = 0.0;
        uzRef = 0.0;
      }

      // Compute energy from g_
      Scalar_ energy = 0.0;
      for (Index i = 0; i < kLbmClass_.Speeds(); ++i) {
        energy += g_(idx[0], idx[1], idx[2], i);
      }

      // Compute pressure using ideal gas law: p = (γ - 1)*(E - 0.5*ρ*u^2)
      Scalar_ u2 = uxRef * uxRef + uyRef * uyRef + uzRef * uzRef;
      Scalar_ pressure = (kFluidProps_.gamma - 1.0) * (energy - 0.5 * rho * u2);
      Scalar_ entropy = std::log(pressure / std::pow(rho, kFluidProps_.gamma));
      p_(idx[0], idx[1], idx[2]) = pressure;
      s_(idx[0], idx[1], idx[2]) = entropy;
    };

    iterateDim(macroscopicTranform);
  }

protected:
  void iterateDim(const std::function<void(const Vector<Index, LbmClassType_::Dim()>&)>& func) {
    Vector<Index, LbmClassType_::Dim()> indices = Vector<Index, LbmClassType_::Dim()>::Zero();

    while (true) {
      // Iteration process
      func(indices);

      // Start at the last dimension
      Index dim = LbmClassType_::Dim() - 1;
      ++indices[dim];

      // Carry over if the index exceeds the limit for that dimension
      while (dim >= 0 && indices[dim] >= kSizes_[dim]) {
        indices[dim] = 0;
        --dim;
        if (dim >= 0) ++indices[dim];
      }

      // If we've carried past the first dimension, then we're done
      if (dim < 0) break;
    }
  }

private:
  const LbmClassType_& kLbmClass_;
  const Vector<Index, LbmClassType_::Dim()> kSizes_;
  const FluidProperties<Scalar_> kFluidProps_;

  Tensor<Scalar_, LbmClassType_::Dim()> rho_;
  Tensor<Scalar_, LbmClassType_::Dim()> p_;
  Tensor<Scalar_, LbmClassType_::Dim()> s_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> u_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> f_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> fAux_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> g_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> gAux_;
};

}

#endif  // LBMINI_LBMTUBE_HPP_
