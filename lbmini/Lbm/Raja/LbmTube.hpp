#ifndef LBMINI_RAJA_LBMTUBE_HPP_
#define LBMINI_RAJA_LBMTUBE_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include <RAJA/RAJA.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/Raja/Backends.hpp"
#include "Lbm/Stencil.hpp"

/// Unroll a short, fixed-trip loop (directions / dimensions).
/// `LBMINI_RAJA_NO_UNROLL` keeps a loop rolled (the Newton iterations: one copy of the
/// body instead of three, which is what the instruction cache of a GPU wants).
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__clang__)
#define LBMINI_RAJA_UNROLL _Pragma("unroll")
#define LBMINI_RAJA_NO_UNROLL _Pragma("unroll 1")
#elif defined(__GNUC__)
#define LBMINI_RAJA_UNROLL _Pragma("GCC unroll 16")
#define LBMINI_RAJA_NO_UNROLL _Pragma("GCC unroll 1")
#else
#define LBMINI_RAJA_UNROLL
#define LBMINI_RAJA_NO_UNROLL
#endif

namespace lbmini::raja {

/**
 * @brief Plain-data parameters captured by value in RAJA device/host lambdas.
 */
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
 * @brief Stationary physics routines (equilibria, macroscopics, collision).
 *
 * Decorated with `RAJA_HOST_DEVICE inline` so they execute cleanly across host
 * and accelerator devices with zero warp divergence and fully unrolled loops.
 * The arithmetic is that of the reference solver except for the energy closure,
 * which is evaluated in the factorised form of `AxisExp`; it agrees with the
 * reference to rounding error.
 */
template<typename Scalar, typename LatticeType>
struct Physics {
  static constexpr int kQ = static_cast<int>(LatticeType::Speeds());
  static constexpr int kD = static_cast<int>(LatticeType::Dim());
  static constexpr Scalar kTiny = Scalar{ 1.0e-12 };
  static constexpr Scalar kMaxExp = Scalar{ 700.0 };
  static constexpr int kMaxIter = 3;

  static_assert(kD == 2 && kQ == 9, "the RAJA backend implements the D2Q9 closure");

  RAJA_HOST_DEVICE inline static constexpr int Vx(const int idc) {
    return static_cast<int>(LatticeType::Velocity(idc, 0));
  }

  RAJA_HOST_DEVICE inline static constexpr int Vy(const int idc) {
    return static_cast<int>(LatticeType::Velocity(idc, 1));
  }

  RAJA_HOST_DEVICE inline static void Feq(
    const Scalar rhoc,
    const Scalar ux,
    const Scalar uy,
    const Scalar temc,
    const Scalar ushiftX,
    const Scalar ushiftY,
    Scalar (&feq)[kQ]
  ) {
    LBMINI_RAJA_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      Scalar phi = rhoc;
      LBMINI_RAJA_UNROLL
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

  RAJA_HOST_DEVICE inline static void Macroscopic(
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
    LBMINI_RAJA_UNROLL
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
  RAJA_HOST_DEVICE inline static void AxisExp(
    const Scalar xi,
    const Scalar wn,
    const Scalar wZero,
    Scalar (&p)[3]
  ) {
    const Scalar t = -std::fabs(xi);
    const Scalar e = std::exp((t > -kMaxExp) ? t : -kMaxExp);
    const Scalar e2 = e * e;
    const bool pos = xi >= Scalar{ 0 };
    p[0] = wn * (pos ? e2 : Scalar{ 1 });
    p[1] = wZero * e;
    p[2] = wn * (pos ? Scalar{ 1 } : e2);
  }

  /// Equilibrium energy populations from the Newton solution `xi`.
  RAJA_HOST_DEVICE inline static void GeqFromXi(
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
    LBMINI_RAJA_UNROLL
    for (int k = 0; k < 3; ++k)
      sx[k] = scale * px[k];
    LBMINI_RAJA_UNROLL
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
  RAJA_HOST_DEVICE inline static void Collide(
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
    LBMINI_RAJA_UNROLL
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

    LBMINI_RAJA_NO_UNROLL
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
      LBMINI_RAJA_UNROLL
      for (int idc = 0; idc < kQ; ++idc)
        sumW += wi[idc];
      if (sumW <= kTiny) {
        const Scalar uni = targetE / Scalar(kQ);
        LBMINI_RAJA_UNROLL
        for (int idc = 0; idc < kQ; ++idc)
          geqLocal[idc] = uni;
      } else {
        const Scalar sc = targetE / sumW;
        LBMINI_RAJA_UNROLL
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
    LBMINI_RAJA_UNROLL
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
    Scalar L[kD] = { Scalar{ 0 }, Scalar{ 0 } };
    LBMINI_RAJA_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar cix0 = static_cast<Scalar>(Vx(idc));
      const Scalar ciy0 = static_cast<Scalar>(Vy(idc));
      const Scalar uvi = ux * cix0 + uy * ciy0;
      const Scalar aux = Scalar{ 2 } * uvi * (f[idc] - feqLocal[idc]);
      L[0] += aux * cix0;
      L[1] += aux * ciy0;
    }

    LBMINI_RAJA_UNROLL
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
 * @brief Modernised RAJA variant of the compressible LBM tube solver.
 *
 * Implements single-pass pull-stream-and-collide execution matching the Alpaka backend:
 *  - DirectionStencil eliminates 432 B/cell/step streaming table lookups.
 *  - Ping-pong raw device pointer swaps (fA/fB, gA/gB) eliminate intermediate
 *    buffers (faux/gaux).
 *  - Compact 2-component warm start xi (2 * N) reduces memory traffic.
 *  - Lazy macroscopic writeback (kStoreMacro) skips intermediate global writes.
 *  - Single-source execution across HostBackend and DeviceBackend.
 */
template<typename Scalar, typename LatticeType, typename Backend = DeviceBackend>
class LbmTube : public ILbmTube<Scalar, LatticeType> {
public:
  using Index = Eigen::Index;

  template<typename Type, Index NumIndices>
  using Tensor = Eigen::Tensor<Type, NumIndices, Eigen::RowMajor>;

  static constexpr Index kDim_ = LatticeType::Dim();
  static constexpr Index kQ_ = LatticeType::Speeds();

  using ExecPolicy = typename Backend::ExecPolicy;
  using Memory = typename Backend::Memory;
  using RangeSegment = RAJA::TypedRangeSegment<Index>;

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

  ~LbmTube() override;

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
  const FluidData<Scalar> kFluid_;
  const MeshData<Scalar, kDim_> kMesh_;
  const ControlData<Scalar> kControl_;
  const PerformanceData kPerformance_;

  Index nx_;
  Index ny_;
  Index N_;        // nx_ * ny_
  Index uSize_;    // N_ * Dim
  Index distSize_; // N_ * Q

  Params<Scalar, kQ_> params_{};

  // Flat SoA storage owned by Backend::Memory.
  Scalar* rho_ = nullptr;    // size N_
  Scalar* p_ = nullptr;      // size N_
  Scalar* tem_ = nullptr;    // size N_
  Scalar* u_ = nullptr;      // size uSize_, d-major
  Scalar* fA_ = nullptr;     // size distSize_, idc-major (ping-pong)
  Scalar* fB_ = nullptr;     // size distSize_, idc-major (ping-pong)
  Scalar* gA_ = nullptr;     // size distSize_, idc-major (ping-pong)
  Scalar* gB_ = nullptr;     // size distSize_, idc-major (ping-pong)
  Scalar* xi_ = nullptr;     // size N_ * kDim_, d-major (Newton warm-start)

  Scalar* fCur_ = nullptr;
  Scalar* fNext_ = nullptr;
  Scalar* gCur_ = nullptr;
  Scalar* gNext_ = nullptr;

  mutable lbmini::DeviceBuffer<Scalar> rhoHost_;
  mutable lbmini::DeviceBuffer<Scalar> pHost_;
  mutable lbmini::DeviceBuffer<Scalar> temHost_;
  mutable lbmini::DeviceBuffer<Scalar> uHost_;
};

template<typename Scalar, typename LatticeType, typename Backend>
LbmTube<Scalar, LatticeType, Backend>::LbmTube(
  const FluidData<Scalar>& fluid,
  const MeshData<Scalar, kDim_>& mesh,
  const ControlData<Scalar>& control,
  const PerformanceData& performance
) :
  kFluid_(fluid),
  kMesh_(mesh),
  kControl_(control),
  kPerformance_(performance) {
  ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "raja::LbmTube");

  nx_ = mesh.size[0];
  ny_ = mesh.size[1];
  N_ = nx_ * ny_;
  uSize_ = N_ * kDim_;
  distSize_ = N_ * kQ_;

  rho_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(N_));
  p_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(N_));
  tem_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(N_));
  u_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(uSize_));

  fA_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(distSize_));
  fB_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(distSize_));
  gA_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(distSize_));
  gB_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(distSize_));

  xi_ = Memory::template Allocate<Scalar>(static_cast<std::size_t>(N_ * kDim_));

  fCur_ = fA_;
  fNext_ = fB_;
  gCur_ = gA_;
  gNext_ = gB_;

  rhoHost_.assign(N_, Scalar{ 0 });
  pHost_.assign(N_, Scalar{ 0 });
  temHost_.assign(N_, Scalar{ 0 });
  uHost_.assign(uSize_, Scalar{ 0 });

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
}

template<typename Scalar, typename LatticeType, typename Backend>
LbmTube<Scalar, LatticeType, Backend>::~LbmTube() {
  Memory::Release(xi_);
  Memory::Release(gB_);
  Memory::Release(gA_);
  Memory::Release(fB_);
  Memory::Release(fA_);
  Memory::Release(u_);
  Memory::Release(tem_);
  Memory::Release(p_);
  Memory::Release(rho_);
}

template<typename Scalar, typename LatticeType, typename Backend>
auto LbmTube<Scalar, LatticeType, Backend>::P() const -> Tensor<Scalar, kDim_> {
  Memory::CopyOut(pHost_.data(), p_, static_cast<std::size_t>(N_));
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(pHost_.begin(), pHost_.end(), out.data());
  return out;
}

template<typename Scalar, typename LatticeType, typename Backend>
auto LbmTube<Scalar, LatticeType, Backend>::Rho() const -> Tensor<Scalar, kDim_> {
  Memory::CopyOut(rhoHost_.data(), rho_, static_cast<std::size_t>(N_));
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(rhoHost_.begin(), rhoHost_.end(), out.data());
  return out;
}

template<typename Scalar, typename LatticeType, typename Backend>
auto LbmTube<Scalar, LatticeType, Backend>::T() const -> Tensor<Scalar, kDim_> {
  Memory::CopyOut(temHost_.data(), tem_, static_cast<std::size_t>(N_));
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(temHost_.begin(), temHost_.end(), out.data());
  return out;
}

template<typename Scalar, typename LatticeType, typename Backend>
auto LbmTube<Scalar, LatticeType, Backend>::U() const -> Tensor<Scalar, kDim_ + 1> {
  Memory::CopyOut(uHost_.data(), u_, static_cast<std::size_t>(uSize_));
  Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
  Scalar* dst = out.data();
  for (Index cell = 0; cell < N_; ++cell)
    for (Index d = 0; d < kDim_; ++d)
      dst[cell * kDim_ + d] = uHost_[d * N_ + cell];
  return out;
}

template<typename Scalar, typename LatticeType, typename Backend>
void LbmTube<Scalar, LatticeType, Backend>::Init() {
  fCur_ = fA_;
  fNext_ = fB_;
  gCur_ = gA_;
  gNext_ = gB_;

  const Params<Scalar, kQ_> p = params_;
  Scalar* __restrict__ fOut = fCur_;
  Scalar* __restrict__ gOut = gCur_;
  Scalar* __restrict__ xiOut = xi_;
  Scalar* __restrict__ rhoOut = rho_;
  Scalar* __restrict__ pOut = p_;
  Scalar* __restrict__ temOut = tem_;
  Scalar* __restrict__ uOut = u_;

  using Phys = Physics<Scalar, LatticeType>;
  constexpr int kQ = Phys::kQ;
  constexpr int kD = Phys::kD;

  RAJA::forall<ExecPolicy>(
    RangeSegment(0, N_),
    [=] RAJA_HOST_DEVICE(const Index cell) {
      const Index N = p.cells;
      const std::int32_t i = static_cast<std::int32_t>(cell / p.ny);

      Scalar rho0;
      Scalar pres0;
      if (i < p.nx / 2) {
        rho0 = p.densityL;
        pres0 = p.pressureL;
      } else {
        rho0 = p.densityR;
        pres0 = p.pressureR;
      }
      const Scalar tem0 = LatticeType::Cs2 * pres0 / (rho0 * p.rgas);
      const Scalar ux0 = Scalar{ 0 };
      const Scalar uy0 = Scalar{ 0 };

      Scalar f[kQ];
      Scalar g[kQ];
      Phys::Feq(rho0, ux0, uy0, tem0, p.ushiftX, p.ushiftY, f);
      const Scalar u2 = ux0 * ux0 + uy0 * uy0;
      const Scalar E = p.cv * tem0 + Scalar{ 0.5 } * u2;
      const Scalar targetE = Scalar{ 2 } * rho0 * E;
      Scalar sumW = Scalar{ 0 };
      LBMINI_RAJA_UNROLL
      for (int idc = 0; idc < kQ; ++idc)
        sumW += LatticeType::Weights(idc, tem0);
      const Scalar sc = (sumW > Phys::kTiny) ? targetE / sumW : targetE / Scalar(kQ);
      LBMINI_RAJA_UNROLL
      for (int idc = 0; idc < kQ; ++idc) {
        const Scalar wi = (sumW > Phys::kTiny) ? LatticeType::Weights(idc, tem0) : Scalar{ 1 };
        g[idc] = wi * sc;
      }

      Scalar rho;
      Scalar invRho;
      Scalar ux;
      Scalar uy;
      Scalar tem;
      Scalar pres;
      Phys::Macroscopic(f, g, p, rho, invRho, ux, uy, tem, pres);
      rhoOut[cell] = rho;
      pOut[cell] = pres;
      temOut[cell] = tem;
      uOut[0 * N + cell] = ux;
      uOut[1 * N + cell] = uy;

      Scalar xi[kD] = { Scalar{ 0 }, Scalar{ 0 } };
      Phys::Collide(p, rho, invRho, ux, uy, tem, f, g, xi);

      LBMINI_RAJA_UNROLL
      for (int idc = 0; idc < kQ; ++idc) {
        fOut[idc * N + cell] = f[idc];
        gOut[idc * N + cell] = g[idc];
      }
      LBMINI_RAJA_UNROLL
      for (int d = 0; d < kD; ++d)
        xiOut[d * N + cell] = xi[d];
    }
  );
  Memory::Synchronize();
}

template<typename Scalar, typename LatticeType, typename Backend>
template<bool kStoreMacro>
void LbmTube<Scalar, LatticeType, Backend>::step() {
  const Params<Scalar, kQ_> p = params_;
  const Scalar* __restrict__ fIn = fCur_;
  const Scalar* __restrict__ gIn = gCur_;
  Scalar* __restrict__ fOut = fNext_;
  Scalar* __restrict__ gOut = gNext_;
  Scalar* __restrict__ xiIo = xi_;
  Scalar* __restrict__ rhoOut = rho_;
  Scalar* __restrict__ pOut = p_;
  Scalar* __restrict__ temOut = tem_;
  Scalar* __restrict__ uOut = u_;

  using Phys = Physics<Scalar, LatticeType>;
  constexpr int kQ = Phys::kQ;
  constexpr int kD = Phys::kD;

  RAJA::forall<ExecPolicy>(
    RangeSegment(0, N_),
    [=] RAJA_HOST_DEVICE(const Index cell) {
      const Index N = p.cells;
      const std::int32_t nx = p.nx;
      const std::int32_t ny = p.ny;
      const auto& st = p.stencil;

      const std::int32_t i = static_cast<std::int32_t>(cell / ny);
      const std::int32_t j = static_cast<std::int32_t>(cell - i * ny);

      Scalar f[kQ];
      Scalar g[kQ];
      LBMINI_RAJA_UNROLL
      for (int idc = 0; idc < kQ; ++idc) {
        const Index plane = idc * N;
        if (st.nearest[idc]) {
          const auto xN = static_cast<std::int32_t>(
            std::round(static_cast<Scalar>(i) - st.cix[idc]));
          const auto yN = static_cast<std::int32_t>(
            std::round(static_cast<Scalar>(j) - st.ciy[idc]));
          std::int32_t yW = yN % ny;
          yW += (yW < 0) ? ny : 0;
          const std::int32_t src = (xN < 0 || xN >= nx) ? static_cast<std::int32_t>(cell) : xN * ny + yW;
          f[idc] = fIn[plane + src];
          g[idc] = gIn[plane + src];
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
          f[idc] = w0 * fIn[plane + rowA + ya] + w1 * fIn[plane + rowB + ya]
                 + w2 * fIn[plane + rowA + yb] + w3 * fIn[plane + rowB + yb];
          g[idc] = w0 * gIn[plane + rowA + ya] + w1 * gIn[plane + rowB + ya]
                 + w2 * gIn[plane + rowA + yb] + w3 * gIn[plane + rowB + yb];
        }
      }

      Scalar rho;
      Scalar invRho;
      Scalar ux;
      Scalar uy;
      Scalar tem;
      Scalar pres;
      Phys::Macroscopic(f, g, p, rho, invRho, ux, uy, tem, pres);
      if (kStoreMacro) {
        rhoOut[cell] = rho;
        pOut[cell] = pres;
        temOut[cell] = tem;
        uOut[0 * N + cell] = ux;
        uOut[1 * N + cell] = uy;
      }

      Scalar xi[kD];
      LBMINI_RAJA_UNROLL
      for (int d = 0; d < kD; ++d)
        xi[d] = xiIo[d * N + cell];

      Phys::Collide(p, rho, invRho, ux, uy, tem, f, g, xi);

      LBMINI_RAJA_UNROLL
      for (int idc = 0; idc < kQ; ++idc) {
        fOut[idc * N + cell] = f[idc];
        gOut[idc * N + cell] = g[idc];
      }
      LBMINI_RAJA_UNROLL
      for (int d = 0; d < kD; ++d)
        xiIo[d * N + cell] = xi[d];
    }
  );
}

template<typename Scalar, typename LatticeType, typename Backend>
void LbmTube<Scalar, LatticeType, Backend>::Step(const bool save) {
  Run(1, save);
}

template<typename Scalar, typename LatticeType, typename Backend>
void LbmTube<Scalar, LatticeType, Backend>::Run(const Index steps, bool /*save*/) {
  for (Index t = 0; t < steps; ++t) {
    if (t + 1 == steps)
      step<true>();
    else
      step<false>();
    std::swap(fCur_, fNext_);
    std::swap(gCur_, gNext_);
  }
  Memory::Synchronize();
}

} // namespace lbmini::raja

#endif // LBMINI_RAJA_LBMTUBE_HPP_
