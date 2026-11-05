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

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__clang__)
#define LBMINI_RAJA_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define LBMINI_RAJA_UNROLL _Pragma("GCC unroll 16")
#else
#define LBMINI_RAJA_UNROLL
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
  Scalar cp;
  Scalar invCv;
  Scalar rgas;
  Scalar viscosity;
  Scalar conductivity;

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
 */
template<typename Scalar, typename LatticeType>
struct Physics {
  static constexpr int kQ = static_cast<int>(LatticeType::Speeds());
  static constexpr int kD = static_cast<int>(LatticeType::Dim());
  static constexpr Scalar kTiny = Scalar{ 1.0e-12 };
  static constexpr Scalar kMaxExp = Scalar{ 700.0 };
  static constexpr int kMaxIter = 3;

  static_assert(kD == 2, "the RAJA backend implements the 2-D closure");

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
    const Scalar invRho = Scalar{ 1 } / rhoSum;
    ux = momX * invRho;
    uy = momY * invRho;
    const Scalar kin = Scalar{ 0.5 } * (ux * ux + uy * uy);
    tem = (Scalar{ 0.5 } * nrg * invRho - kin) * p.invCv;
    rho = rhoSum;
    pres = p.rgas * rhoSum * tem;
  }

  RAJA_HOST_DEVICE inline static void Collide(
    const Params<Scalar, kQ>& p,
    const Scalar rhoc,
    const Scalar ux,
    const Scalar uy,
    const Scalar temc,
    Scalar (&f)[kQ],
    Scalar (&g)[kQ],
    Scalar (&xi)[kD]
  ) {
    const Scalar ushiftX = p.ushiftX;
    const Scalar ushiftY = p.ushiftY;
    const Scalar u2 = ux * ux + uy * uy;

    Scalar feqLocal[kQ];
    Feq(rhoc, ux, uy, temc, ushiftX, ushiftY, feqLocal);

    // Geq via Newton-Raphson on the normalised energy flux.
    const Scalar E = p.cv * temc + Scalar{ 0.5 } * u2;
    const Scalar targetE = Scalar{ 2 } * rhoc * E;
    Scalar targetM[kD];
    LBMINI_RAJA_UNROLL
    for (int d = 0; d < kD; ++d)
      targetM[d] = Scalar{ 2 } * rhoc * (d == 0 ? ux : uy) * (E + temc) / targetE;

    Scalar alpha = Scalar{ 1 };
    Scalar si[kQ];
    Scalar Z = Scalar{ 0 };
    Scalar smax = Scalar{ 0 };
    bool solverOk = true;

    for (int iter = 0; iter < kMaxIter; ++iter) {
      smax = -kMaxExp;
      LBMINI_RAJA_UNROLL
      for (int idc = 0; idc < kQ; ++idc) {
        Scalar s = Scalar{ 0 };
        LBMINI_RAJA_UNROLL
        for (int d = 0; d < kD; ++d)
          s += xi[d] * LatticeType::Cshift(idc, d, (d == 0 ? ushiftX : ushiftY));
        si[idc] = s;
        smax = (s > smax) ? s : smax;
      }
      smax = (smax < kMaxExp) ? smax : kMaxExp;

      Scalar S1[kD] = { Scalar{ 0 }, Scalar{ 0 } };
      Scalar S2[kD][kD] = { { Scalar{ 0 }, Scalar{ 0 } }, { Scalar{ 0 }, Scalar{ 0 } } };
      Z = Scalar{ 0 };
      LBMINI_RAJA_UNROLL
      for (int idc = 0; idc < kQ; ++idc) {
        Scalar expo = si[idc] - smax;
        expo = (expo < kMaxExp) ? expo : kMaxExp;
        expo = (expo > -kMaxExp) ? expo : -kMaxExp;
        const Scalar wev = LatticeType::Weights(idc, temc) * std::exp(expo);
        Z += wev;
        LBMINI_RAJA_UNROLL
        for (int a = 0; a < kD; ++a) {
          const Scalar cia = LatticeType::Cshift(idc, a, (a == 0 ? ushiftX : ushiftY));
          S1[a] += cia * wev;
          LBMINI_RAJA_UNROLL
          for (int b = 0; b < kD; ++b)
            S2[a][b] += cia * LatticeType::Cshift(idc, b, (b == 0 ? ushiftX : ushiftY)) * wev;
        }
      }

      const Scalar Zsafe = (Z > kTiny) ? Z : kTiny;
      solverOk = solverOk && (Z > kTiny);
      const Scalar invZ = Scalar{ 1 } / Zsafe;

      Scalar J[kD][kD];
      LBMINI_RAJA_UNROLL
      for (int a = 0; a < kD; ++a)
        LBMINI_RAJA_UNROLL
        for (int b = 0; b < kD; ++b)
          J[a][b] = (S2[a][b] - S1[a] * S1[b] * invZ) * invZ;
      J[0][0] += kTiny;
      J[1][1] += kTiny;

      const Scalar detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
      const Scalar absDet = std::fabs(detJ);
      const Scalar detSafe = (absDet > kTiny) ? detJ : ((detJ >= Scalar{ 0 }) ? kTiny : -kTiny);
      solverOk = solverOk && (absDet > kTiny);
      const Scalar invDet = Scalar{ 1 } / detSafe;

      const Scalar r0 = S1[0] * invZ - targetM[0];
      const Scalar r1 = S1[1] * invZ - targetM[1];
      xi[0] -= alpha * (J[1][1] * r0 - J[0][1] * r1) * invDet;
      xi[1] -= alpha * (-J[1][0] * r0 + J[0][0] * r1) * invDet;
      alpha *= Scalar{ 0.5 };
    }

    Scalar geqLocal[kQ];
    if (solverOk) {
      smax = -kMaxExp;
      LBMINI_RAJA_UNROLL
      for (int idc = 0; idc < kQ; ++idc) {
        Scalar s = Scalar{ 0 };
        LBMINI_RAJA_UNROLL
        for (int d = 0; d < kD; ++d)
          s += xi[d] * LatticeType::Cshift(idc, d, (d == 0 ? ushiftX : ushiftY));
        si[idc] = s;
        smax = (s > smax) ? s : smax;
      }
      smax = (smax < kMaxExp) ? smax : kMaxExp;
      Z = Scalar{ 0 };
      LBMINI_RAJA_UNROLL
      for (int idc = 0; idc < kQ; ++idc) {
        Scalar expo = si[idc] - smax;
        expo = (expo < kMaxExp) ? expo : kMaxExp;
        expo = (expo > -kMaxExp) ? expo : -kMaxExp;
        const Scalar wev = LatticeType::Weights(idc, temc) * std::exp(expo);
        Z += wev;
        geqLocal[idc] = wev;
      }
      const Scalar scale = targetE / Z;
      LBMINI_RAJA_UNROLL
      for (int idc = 0; idc < kQ; ++idc)
        geqLocal[idc] *= scale;
    } else {
      xi[0] = Scalar{ 0 };
      xi[1] = Scalar{ 0 };
      Scalar sumW = Scalar{ 0 };
      LBMINI_RAJA_UNROLL
      for (int idc = 0; idc < kQ; ++idc)
        sumW += LatticeType::Weights(idc, temc);
      if (sumW <= kTiny) {
        const Scalar uni = targetE / Scalar(kQ);
        LBMINI_RAJA_UNROLL
        for (int idc = 0; idc < kQ; ++idc)
          geqLocal[idc] = uni;
      } else {
        const Scalar sc = targetE / sumW;
        LBMINI_RAJA_UNROLL
        for (int idc = 0; idc < kQ; ++idc)
          geqLocal[idc] = LatticeType::Weights(idc, temc) * sc;
      }
    }

    // Collision (BGK + Knudsen sensor + thermal-relaxation correction).
    const Scalar tau = p.viscosity / (rhoc * temc) + Scalar{ 0.5 };
    const Scalar omega = Scalar{ 1 } / tau;
    const Scalar diffusivity = p.conductivity / (rhoc * p.cp);
    const Scalar tauT = diffusivity / temc + Scalar{ 0.5 };
    Scalar omegaT = Scalar{ 1 } / tauT;

    Scalar eps = Scalar{ 0 };
    LBMINI_RAJA_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar diff = f[idc] - feqLocal[idc];
      const Scalar den = (feqLocal[idc] > kTiny) ? feqLocal[idc] : kTiny;
      eps += std::fabs(diff) / den;
    }
    eps /= Scalar(kQ);
    Scalar sigma = Scalar{ 1 };
    if (eps >= Scalar{ 1 })
      sigma = omega;
    else if (eps >= Scalar{ 0.1 })
      sigma = Scalar{ 1.35 };
    else if (eps >= Scalar{ 0.01 })
      sigma = Scalar{ 1.05 };

    Scalar omegaL = omega / sigma;
    omegaL = (omegaL > Scalar{ 1 }) ? omegaL : Scalar{ 1 };
    omegaL = (omegaL < (Scalar{ 2 } - Scalar{ 1e-7 })) ? omegaL : (Scalar{ 2 } - Scalar{ 1e-7 });
    omegaT = (omegaT > Scalar{ 1 }) ? omegaT : Scalar{ 1 };
    omegaT = (omegaT < (Scalar{ 2 } - Scalar{ 1e-7 })) ? omegaT : (Scalar{ 2 } - Scalar{ 1e-7 });

    // L[a] = sum_i 2 * (u . c_i) * c_i_a * (f - feq)   (c_i unshifted)
    Scalar L[kD] = { Scalar{ 0 }, Scalar{ 0 } };
    LBMINI_RAJA_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar cix0 = static_cast<Scalar>(LatticeType::Velocity(idc, 0));
      const Scalar ciy0 = static_cast<Scalar>(LatticeType::Velocity(idc, 1));
      const Scalar uvi = ux * cix0 + uy * ciy0;
      const Scalar aux = Scalar{ 2 } * uvi * (f[idc] - feqLocal[idc]);
      L[0] += aux * cix0;
      L[1] += aux * ciy0;
    }

    const Scalar invT = Scalar{ 1 } / temc;
    LBMINI_RAJA_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar fOld = f[idc];
      f[idc] = fOld + omegaL * (feqLocal[idc] - fOld);
      const Scalar cidotL = L[0] * LatticeType::Cshift(idc, 0, ushiftX)
        + L[1] * LatticeType::Cshift(idc, 1, ushiftY);
      const Scalar Wi = LatticeType::Weights(idc, temc);
      const Scalar gDiff = Wi * cidotL * invT;
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
  params_.cp = fluid.specificHeatCp;
  params_.invCv = Scalar{ 1 } / fluid.specificHeatCv;
  params_.rgas = fluid.constant;
  params_.viscosity = fluid.viscosity;
  params_.conductivity = fluid.conductivity;
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
      Scalar ux;
      Scalar uy;
      Scalar tem;
      Scalar pres;
      Phys::Macroscopic(f, g, p, rho, ux, uy, tem, pres);
      rhoOut[cell] = rho;
      pOut[cell] = pres;
      temOut[cell] = tem;
      uOut[0 * N + cell] = ux;
      uOut[1 * N + cell] = uy;

      Scalar xi[kD] = { Scalar{ 0 }, Scalar{ 0 } };
      Phys::Collide(p, rho, ux, uy, tem, f, g, xi);

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
      Scalar ux;
      Scalar uy;
      Scalar tem;
      Scalar pres;
      Phys::Macroscopic(f, g, p, rho, ux, uy, tem, pres);
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

      Phys::Collide(p, rho, ux, uy, tem, f, g, xi);

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
