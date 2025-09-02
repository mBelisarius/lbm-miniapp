#ifndef LBMINI_OPENMPGPU_LBMTUBE_HPP_
#define LBMINI_OPENMPGPU_LBMTUBE_HPP_

#include <Eigen/Dense>
#include <omp.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cstdio>
#include <cmath>

#include "Data.hpp"

namespace lbmini::openmp::gpu {
// Forward declaration
template<typename>
class LbmD2Q9;

template<typename Scalar_, typename LbmClassType_>
class LbmTube {
public:
  using Index = Eigen::Index;

  template<typename Type, Index Size>
  using Array = Eigen::Array<Type, Size, 1>;

  template<typename Type, Index Size>
  using Vector = Eigen::Vector<Type, Size>;

  template<typename Type, Index Size>
  using VectorMap = Eigen::Map<Vector<Type, Size>>;

  template<typename Type, Index NumIndices>
  using Tensor = Eigen::Tensor<Type, NumIndices, Eigen::RowMajor>;

  template<typename Type, Index NumIndices>
  using TensorMap = Eigen::Map<Tensor<Type, NumIndices>>;

  LbmTube(
    const FluidData<Scalar_>& fluid,
    const MeshData<Scalar_, LbmClassType_::Dim()>& mesh,
    const ControlData<Scalar_>& control,
    const PerformanceData& performance
  );

  Tensor<Scalar_, LbmClassType_::Dim()> P() const { return p_; }

  Tensor<Scalar_, LbmClassType_::Dim()> Rho() const { return rho_; }

  Tensor<Scalar_, LbmClassType_::Dim()> T() const { return tem_; }

  Tensor<Scalar_, LbmClassType_::Dim() + 1> U() const { return u_; }

  void Init();

  void Step();

  void Run(Index steps);

private:
  const Scalar_ kTiny_ = Scalar_(1.0e-12);
  const FluidData<Scalar_> kFluid_;
  const MeshData<Scalar_, LbmClassType_::Dim()> kMesh_;
  const ControlData<Scalar_> kControl_;
  const PerformanceData kPerformance_;

  Index macSize_;
  Index uSize_;
  Index distSize_;

  // SoA data storage
  Tensor<Scalar_, LbmClassType_::Dim()> rho_;
  Tensor<Scalar_, LbmClassType_::Dim()> p_;
  Tensor<Scalar_, LbmClassType_::Dim()> tem_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> u_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> f_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> feq_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> faux_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> g_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> geq_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> gaux_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> lastGx_;
};

template<typename Scalar_, typename LbmClassType_>
LbmTube<Scalar_, LbmClassType_>::LbmTube(
  const FluidData<Scalar_>& fluid,
  const MeshData<Scalar_, LbmClassType_::Dim()>& mesh,
  const ControlData<Scalar_>& control,
  const PerformanceData& performance
)
  : kFluid_(fluid), kMesh_(mesh), kControl_(control), kPerformance_(performance) {
  if (kPerformance_.cores > 0)
    omp_set_num_threads(kPerformance_.cores);

  macSize_ = mesh.size[0] * mesh.size[1];
  uSize_ = macSize_ * LbmClassType_::Dim();
  distSize_ = macSize_ * LbmClassType_::Speeds();

  Eigen::array<Index, LbmClassType_::Dim()> dims;
  for (Index i = 0; i < LbmClassType_::Dim(); ++i)
    dims[i] = mesh.size[i];
  rho_.resize(dims);
  p_.resize(dims);
  tem_.resize(dims);

  Eigen::array<Index, LbmClassType_::Dim() + 1> u_dims;
  for (Index i = 0; i < LbmClassType_::Dim(); ++i)
    u_dims[i] = mesh.size[i];
  u_dims[LbmClassType_::Dim()] = LbmClassType_::Dim();
  u_.resize(u_dims);
  lastGx_.resize(u_dims);

  Eigen::array<Index, LbmClassType_::Dim() + 1> dist_dims;
  for (Index i = 0; i < LbmClassType_::Dim(); ++i)
    dist_dims[i] = mesh.size[i];
  dist_dims[LbmClassType_::Dim()] = LbmClassType_::Speeds();
  f_.resize(dist_dims);
  feq_.resize(dist_dims);
  faux_.resize(dist_dims);
  g_.resize(dist_dims);
  geq_.resize(dist_dims);
  gaux_.resize(dist_dims);
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::Init() {
  auto* pRho = rho_.data();
  auto* pP = p_.data();
  auto* pTem = tem_.data();
  auto* pU = u_.data();
  auto* pF = f_.data();
  auto* pFeq = feq_.data();
  auto* pG = g_.data();
  auto* pGeq = geq_.data();
  auto* pLastGx = lastGx_.data();

  #pragma omp target data map(to: kFluid_, kMesh_, kControl_) \
                          map(tofrom: pRho[0:macSize_], pP[0:macSize_], pTem[0:macSize_], pU[0:uSize_], pLastGx[0:uSize_], pF[0:distSize_], pG[0:distSize_]) \
                          map(alloc: pFeq[0:distSize_], pFaux[0:distSize_], pGeq[0:distSize_], pGaux[0:distSize_])
  {
    #pragma omp target teams distribute parallel for collapse(2)
    for (Index i = 0; i < kMesh_.size[0]; ++i) {
      for (Index j = 0; j < kMesh_.size[1]; ++j) {
        Scalar_ rho0, p0;
        if (i < kMesh_.size[0] / 2) {
          rho0 = kFluid_.densityL;
          p0 = kFluid_.pressureL;
        } else {
          rho0 = kFluid_.densityR;
          p0 = kFluid_.pressureR;
        }

        Vector<Scalar_, LbmClassType_::Dim()> u0;
        u0.setZero();

        const Index mac_idx = i * kMesh_.size[1] + j;
        const Index u_idx = mac_idx * LbmClassType_::Dim();
        const Index dist_idx = mac_idx * LbmClassType_::Speeds();

        LbmClassType_::Init(
          u0.data(),
          &rho0,
          &p0,
          &pRho[mac_idx],
          &pP[mac_idx],
          &pTem[mac_idx],
          &pU[u_idx],
          &pF[dist_idx],
          &pFeq[dist_idx],
          &pG[dist_idx],
          &pGeq[dist_idx],
          &pLastGx[u_idx],
          kFluid_,
          kControl_
        );
      }
    }
  }
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::Step() {
  Run(1);
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::Run(Index steps) {
  auto* pRho = rho_.data();
  auto* pP = p_.data();
  auto* pTem = tem_.data();
  auto* pU = u_.data();
  auto* pF = f_.data();
  auto* pFeq = feq_.data();
  auto* pFaux = faux_.data();
  auto* pG = g_.data();
  auto* pGeq = geq_.data();
  auto* pGaux = gaux_.data();
  auto* pLastGx = lastGx_.data();

  #pragma omp target data map(to: kFluid_, kMesh_, kControl_) \
                          map(tofrom: pRho[0:macSize_], pP[0:macSize_], pTem[0:macSize_], pU[0:uSize_], pLastGx[0:uSize_], pF[0:distSize_], pG[0:distSize_]) \
                          map(alloc: pFeq[0:distSize_], pFaux[0:distSize_], pGeq[0:distSize_], pGaux[0:distSize_])
  {
    for (Index t = 0; t < steps; ++t) {
      // Collision
      #pragma omp target teams distribute parallel for collapse(2)
      for (Index i = 0; i < kMesh_.size[0]; ++i) {
        for (Index j = 0; j < kMesh_.size[1]; ++j) {
          const Index macIdx = i * kMesh_.size[1] + j;
          const Index uIdx = macIdx * LbmClassType_::Dim();
          const Index distIdx = macIdx * LbmClassType_::Speeds();
          LbmClassType_::Collision(
            &pRho[macIdx],
            &pP[macIdx],
            &pTem[macIdx],
            &pU[uIdx],
            &pF[distIdx],
            &pFeq[distIdx],
            &pG[distIdx],
            &pGeq[distIdx],
            &pLastGx[uIdx],
            kFluid_,
            kControl_
          );
        }
      }

      // Streaming
      #pragma omp target teams distribute parallel for collapse(2)
      for (Index i = 0; i < kMesh_.size[0]; ++i) {
        for (Index j = 0; j < kMesh_.size[1]; ++j) {
          auto sampleNode = [&](Index sx, Index sy, Index idc) -> Index {
            Index sxClamped = sx;
            if (sxClamped < 0)
              sxClamped = 0;
            else if (sxClamped >= kMesh_.size[0])
              sxClamped = kMesh_.size[0] - 1;
            Index syWrapped = sy;
            if (syWrapped < 0)
              syWrapped += kMesh_.size[1];
            else if (syWrapped >= kMesh_.size[1])
              syWrapped -= kMesh_.size[1];
            const Index inIdx = (sxClamped * kMesh_.size[1] + syWrapped) * LbmClassType_::Speeds() + idc;
            return inIdx;
          };
          #pragma omp simd
          for (Index idc = 0; idc < LbmClassType_::Speeds(); ++idc) {
            const Scalar_ cix = static_cast<Scalar_>(LbmClassType_::Velocity(idc, 0)) + kControl_.U(0);
            const Scalar_ ciy = static_cast<Scalar_>(LbmClassType_::Velocity(idc, 1)) + kControl_.U(1);
            const Scalar_ xSrcF = static_cast<Scalar_>(i) - cix;
            const Scalar_ ySrcF = static_cast<Scalar_>(j) - ciy;
            const Index outIdx = (i * kMesh_.size[1] + j) * LbmClassType_::Speeds() + idc;

            if (fabs(ciy) < kTiny_) {
              const Index x0 = static_cast<Index>(floor(xSrcF));
              const Index x1 = x0 + 1;
              const Scalar_ xFrac = xSrcF - static_cast<Scalar_>(x0);

              if (abs(xFrac) < kTiny_) {
                Index xClamp = x0;
                if (xClamp < 0)
                  xClamp = 0;
                if (xClamp >= kMesh_.size[0])
                  xClamp = kMesh_.size[0] - 1;
                Index yWrap = static_cast<Index>(floor(ySrcF));
                if (yWrap < 0)
                  yWrap += kMesh_.size[1];
                if (yWrap >= kMesh_.size[1])
                  yWrap -= kMesh_.size[1];

                const Index inIdx = (xClamp * kMesh_.size[1] + yWrap) * LbmClassType_::Speeds() + idc;
                pFaux[outIdx] = pF[inIdx];
                pGaux[outIdx] = pG[inIdx];
                continue;
              }

              Scalar_ f0, g0, f1, g1;
              {
                Index xClamp = x0;
                if (xClamp < 0)
                  xClamp = 0;
                if (xClamp >= kMesh_.size[0])
                  xClamp = kMesh_.size[0] - 1;
                Index yWrap = static_cast<Index>(floor(ySrcF));
                if (yWrap < 0)
                  yWrap += kMesh_.size[1];
                if (yWrap >= kMesh_.size[1])
                  yWrap -= kMesh_.size[1];
                const Index in_idx = (xClamp * kMesh_.size[1] + yWrap) * LbmClassType_::Speeds() + idc;
                f0 = pF[in_idx];
                g0 = pG[in_idx];
              }
              {
                Index xClamp = x1;
                if (xClamp < 0)
                  xClamp = 0;
                if (xClamp >= kMesh_.size[0])
                  xClamp = kMesh_.size[0] - 1;
                Index yWrap = static_cast<Index>(floor(ySrcF));
                if (yWrap < 0)
                  yWrap += kMesh_.size[1];
                if (yWrap >= kMesh_.size[1])
                  yWrap -= kMesh_.size[1];
                const Index in_idx = (xClamp * kMesh_.size[1] + yWrap) * LbmClassType_::Speeds() + idc;
                f1 = pF[in_idx];
                g1 = pG[in_idx];
              }
              pFaux[outIdx] = (Scalar_(1.0) - xFrac) * f0 + xFrac * f1;
              pGaux[outIdx] = (Scalar_(1.0) - xFrac) * g0 + xFrac * g1;
              continue;
            }

            const Index x0 = static_cast<Index>(floor(xSrcF));
            const Index y0 = static_cast<Index>(floor(ySrcF));
            const Index x1 = x0 + 1;
            const Index y1 = y0 + 1;

            const Index inIdx00 = sampleNode(x0, y0, idc);
            const Index inIdx10 = sampleNode(x1, y0, idc);
            const Index inIdx01 = sampleNode(x0, y1, idc);
            const Index inIdx11 = sampleNode(x1, y1, idc);

            const Scalar_ dx00 = xSrcF - Scalar_(x0);
            const Scalar_ dy00 = ySrcF - Scalar_(y0);
            const Scalar_ d00 = sqrt(dx00 * dx00 + dy00 * dy00);
            if (d00 < kTiny_) {
              pFaux[outIdx] = pF[inIdx00];
              pGaux[outIdx] = pG[inIdx00];
              continue;
            }

            const Scalar_ dx10 = xSrcF - Scalar_(x1);
            const Scalar_ dy10 = ySrcF - Scalar_(y0);
            const Scalar_ d10 = sqrt(dx10 * dx10 + dy10 * dy10);
            if (d10 < kTiny_) {
              pFaux[outIdx] = pF[inIdx10];
              pGaux[outIdx] = pG[inIdx10];
              continue;
            }

            const Scalar_ dx01 = xSrcF - Scalar_(x0);
            const Scalar_ dy01 = ySrcF - Scalar_(y1);
            const Scalar_ d01 = sqrt(dx01 * dx01 + dy01 * dy01);
            if (d01 < kTiny_) {
              pFaux[outIdx] = pF[inIdx01];
              pGaux[outIdx] = pG[inIdx01];
              continue;
            }

            const Scalar_ dx11 = xSrcF - Scalar_(x1);
            const Scalar_ dy11 = ySrcF - Scalar_(y1);
            const Scalar_ d11 = sqrt(dx11 * dx11 + dy11 * dy11);
            if (d11 < kTiny_) {
              pFaux[outIdx] = pF[inIdx11];
              pGaux[outIdx] = pG[inIdx11];
              continue;
            }

            const Scalar_ w00 = Scalar_(1.0) / pow(((d00) > (kTiny_) ? (d00) : (kTiny_)), kControl_.idw);
            const Scalar_ w10 = Scalar_(1.0) / pow(((d10) > (kTiny_) ? (d10) : (kTiny_)), kControl_.idw);
            const Scalar_ w01 = Scalar_(1.0) / pow(((d01) > (kTiny_) ? (d01) : (kTiny_)), kControl_.idw);
            const Scalar_ w11 = Scalar_(1.0) / pow(((d11) > (kTiny_) ? (d11) : (kTiny_)), kControl_.idw);
            const Scalar_ wsum = w00 + w10 + w01 + w11;

            if (!std::isfinite(wsum) || wsum < kTiny_) {
              Index xNearest = static_cast<Index>(round(xSrcF));
              Index yNearest = static_cast<Index>(round(ySrcF));
              if (xNearest < 0 || xNearest >= kMesh_.size[0]) {
                const Index in_idx = (i * kMesh_.size[1] + j) * LbmClassType_::Speeds() + LbmClassType_::Opposite(idc);
                pFaux[outIdx] = pF[in_idx];
                pGaux[outIdx] = pG[in_idx];
              } else {
                if (yNearest < 0)
                  yNearest += kMesh_.size[1];
                if (yNearest >= kMesh_.size[1])
                  yNearest -= kMesh_.size[1];
                const Index in_idx = (xNearest * kMesh_.size[1] + yNearest) * LbmClassType_::Speeds() + idc;
                pFaux[outIdx] = pF[in_idx];
                pGaux[outIdx] = pG[in_idx];
              }
              continue;
            }
            pFaux[outIdx] = (w00 * pF[inIdx00] + w10 * pF[inIdx10] + w01 * pF[inIdx01] + w11 * pF[inIdx11]) / wsum;
            pGaux[outIdx] = (w00 * pG[inIdx00] + w10 * pG[inIdx10] + w01 * pG[inIdx01] + w11 * pG[inIdx11]) / wsum;
          }
        }
      }

      // Fused Swap and Update macroscopic fields
      #pragma omp target teams distribute parallel for collapse(2)
      for (Index i = 0; i < kMesh_.size[0]; ++i) {
        for (Index j = 0; j < kMesh_.size[1]; ++j) {
          const Index macIdx = i * kMesh_.size[1] + j;
          const Index uIdx = macIdx * LbmClassType_::Dim();
          const Index distIdx = macIdx * LbmClassType_::Speeds();

          #pragma omp simd
          for (Index idc = 0; idc < LbmClassType_::Speeds(); ++idc) {
            const Index idx = distIdx + idc;
            pF[idx] = pFaux[idx];
            pG[idx] = pGaux[idx];
          }

          LbmClassType_::ComputeMacroscopic(
            &pRho[macIdx],
            &pP[macIdx],
            &pTem[macIdx],
            &pU[uIdx],
            &pF[distIdx],
            &pFeq[distIdx],
            &pG[distIdx],
            &pGeq[distIdx],
            &pLastGx[uIdx],
            kFluid_,
            kControl_
          );
        }
      }
    }
  }
}
} // namespace lbmini::openmp::gpu

#endif // LBMINI_OPENMPGPU_LBMTUBE_HPP_
