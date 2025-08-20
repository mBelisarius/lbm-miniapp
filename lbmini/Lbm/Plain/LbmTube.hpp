#ifndef LBMINI_LBMTUBE_HPP_
#define LBMINI_LBMTUBE_HPP_

#include <Eigen/Dense>
#include <array>
#include <functional>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"

namespace lbmini {
template<typename Scalar_, typename LbmClassType_>
class LbmTube {
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

  LbmTube(
    const FluidData<Scalar_>& fluid,
    const MeshData<Scalar_, LbmClassType_::Dim()>& mesh,
    const ControlData<Scalar_>& control
  );

  Tensor<Scalar_, LbmClassType_::Dim()> P() const {
    Tensor<Scalar_, LbmClassType_::Dim()> p(kSizes_[0], kSizes_[1]);
    for (Index idx = 0; idx < kSizes_[0]; ++idx)
      for (Index idy = 0; idy < kSizes_[1]; ++idy)
        p(idx, idy) = lattices_(idx, idy).P();

    return p;
  }

  Tensor<Scalar_, LbmClassType_::Dim()> Rho() const {
    Tensor<Scalar_, LbmClassType_::Dim()> rho(kSizes_[0], kSizes_[1]);
    for (Index idx = 0; idx < kSizes_[0]; ++idx)
      for (Index idy = 0; idy < kSizes_[1]; ++idy)
        rho(idx, idy) = lattices_(idx, idy).Rho();

    return rho;
  }

  Tensor<Scalar_, LbmClassType_::Dim()> T() const {
    Tensor<Scalar_, LbmClassType_::Dim()> tem(kSizes_[0], kSizes_[1]);
    for (Index idx = 0; idx < kSizes_[0]; ++idx)
      for (Index idy = 0; idy < kSizes_[1]; ++idy)
        tem(idx, idy) = lattices_(idx, idy).Tem();

    return tem;
  }

  Tensor<Scalar_, LbmClassType_::Dim() + 1> U() const {
    Tensor<Scalar_, LbmClassType_::Dim() + 1> u(kSizes_[0], kSizes_[1], LbmClassType_::Dim());
    for (Index idx = 0; idx < kSizes_[0]; ++idx)
      for (Index idy = 0; idy < kSizes_[1]; ++idy)
        for (Index idd = 0; idd < LbmClassType_::Dim(); ++idd)
          u(idx, idy, idd) = lattices_(idx, idy).U(idd);

    return u;
  }

  void Init();

  void ComputeMacroscopic();

  void Collision();

  void Streaming();

  void Step();

protected:
  void iterateDim(const std::function<void(const Array<Index, LbmClassType_::Dim()>&)>& func);

  void swapDistributions();

private:
  // Constants
  const Scalar_ kTiny_ = Scalar_(10.e-11);
  const Scalar_ kCs2_ = Scalar_(1.0) / Scalar_(3.0);

  // Members
  const FluidData<Scalar_> kFluid_;
  const ControlData<Scalar_> kControl_;
  const Array<Index, LbmClassType_::Dim()> kSizes_;
  Tensor<LbmClassType_, LbmClassType_::Dim()> lattices_;

  // Distributions
  Tensor<Scalar_, LbmClassType_::Dim() + 1> fAux_;
  Tensor<Scalar_, LbmClassType_::Dim() + 1> gAux_;
};

template<typename Scalar_, typename LbmClassType_>
LbmTube<Scalar_, LbmClassType_>::LbmTube(
  const FluidData<Scalar_>& fluid,
  const MeshData<Scalar_, LbmClassType_::Dim()>& mesh,
  const ControlData<Scalar_>& control
)
  : kFluid_(fluid), kControl_(control), kSizes_(mesh.size) {
  std::array<Index, LbmClassType_::Dim()> dims;
  for (Index i = 0; i < LbmClassType_::Dim(); ++i)
    dims[i] = kSizes_[i];
  lattices_.resize(dims);

  std::array<Index, LbmClassType_::Dim() + 1> dimsP1;
  for (Index i = 0; i < LbmClassType_::Dim(); ++i)
    dimsP1[i] = kSizes_[i];
  dimsP1[LbmClassType_::Dim()] = LbmClassType_::Speeds();
  fAux_.resize(dimsP1);
  gAux_.resize(dimsP1);

  iterateDim(
    [&](const Array<Index, LbmClassType_::Dim()>& idx) {
      lattices_(idx) = LbmClassType_(&kFluid_, &kControl_);
    }
  );
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::Init() {
  auto initOp = [&](const Array<Index, LbmClassType_::Dim()>& idx) {
    Vector<Scalar_, LbmClassType_::Dim()> u0;
    Scalar_ rho0;
    Scalar_ p0;

    // Apply shock tube condition along x-axis
    if (idx[0] < kSizes_[0] / 2) {
      rho0 = kFluid_.densityL;
      p0 = kFluid_.pressureL;
    } else {
      rho0 = kFluid_.densityR;
      p0 = kFluid_.pressureR;
    }

    // Zero initial velocity
    for (Index idd = 0; idd < LbmClassType_::Dim(); ++idd)
      u0(idd) = Scalar_(0.0);

    // Initialize lattice
    lattices_(idx[0], idx[1]).Init(u0, rho0, p0);
  };

  iterateDim(initOp);
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::ComputeMacroscopic() {
  iterateDim(
    [&](const Array<Index, LbmClassType_::Dim()>& idx) {
      lattices_(idx[0], idx[1]).ComputeMacroscopic();
    }
  );
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::Collision() {
  iterateDim(
    [&](const Array<Index, LbmClassType_::Dim()>& idx) {
      lattices_(idx[0], idx[1]).Collision();
    }
  );
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::Streaming() {
  const Index nx = kSizes_[0];
  const Index ny = kSizes_[1];
  const Scalar_ tiny = kTiny_;

  Scalar_ alpha = kControl_.idw;

  auto streamingOp = [&](const Array<Index, LbmClassType_::Dim()>& idx) {
    auto& latticeIdx = lattices_(idx);

    for (Index idc = 0; idc < LbmClassType_::Speeds(); ++idc) {
      // Continuous source position: s = x - (c + U)
      const Scalar_ cix = static_cast<Scalar_>(latticeIdx.Velocities(idc, 0)) + kControl_.U(0);
      const Scalar_ ciy = static_cast<Scalar_>(latticeIdx.Velocities(idc, 1)) + kControl_.U(1);

      const Scalar_ xSrcF = static_cast<Scalar_>(idx[0]) - cix;
      const Scalar_ ySrcF = static_cast<Scalar_>(idx[1]) - ciy;

      // Fast 1D special-case: if shift is (nearly) only in x (|ci_y + U_y| < tiny),
      // use (Rathnayake Eq. 4.13) two-node interpolation (exact & cheaper)
      if (std::abs(ciy) < tiny) {
        // Nearest integer neighbors in x
        const Index x0 = static_cast<Index>(std::floor(xSrcF));
        const Index x1 = x0 + 1;
        const Scalar_ xFrac = xSrcF - static_cast<Scalar_>(x0);

        // If exactly on node, sample directly (clamp x to domain)
        if (std::abs(xFrac) < tiny) {
          Index xClamp = x0;
          if (xClamp < 0) xClamp = 0;
          if (xClamp >= nx) xClamp = nx - 1;
          Index yWrap = static_cast<Index>(std::floor(ySrcF));
          if (yWrap < 0) yWrap += ny;
          if (yWrap >= ny) yWrap -= ny;
          auto& srcLat = lattices_(xClamp, yWrap);
          fAux_(idx[0], idx[1], idc) = srcLat.F(idc);
          gAux_(idx[0], idx[1], idc) = srcLat.G(idc);
          continue;
        }

        // Sample left and right neighbors (permissive clamp at x)
        std::pair<Scalar_, Scalar_> s0, s1;

        // Sample at x0 (clamp to [0, nx-1] if out-of-bounds)
        {
          Index xClamp = x0;
          if (xClamp < 0) xClamp = 0;
          if (xClamp >= nx) xClamp = nx - 1;
          Index yWrap = static_cast<Index>(std::floor(ySrcF));
          if (yWrap < 0) yWrap += ny;
          if (yWrap >= ny) yWrap -= ny;
          auto& src0 = lattices_(xClamp, yWrap);
          s0 = { src0.F(idc), src0.G(idc) };
        }

        // Sample at x1 (clamp to [0, nx-1] if out-of-bounds)
        {
          Index xClamp = x1;
          if (xClamp < 0) xClamp = 0;
          if (xClamp >= nx) xClamp = nx - 1;
          Index yWrap = static_cast<Index>(std::floor(ySrcF));
          if (yWrap < 0) yWrap += ny;
          if (yWrap >= ny) yWrap -= ny;
          auto& src1 = lattices_(xClamp, yWrap);
          s1 = { src1.F(idc), src1.G(idc) };
        }

        // Linear interpolation (Rathnayake Eq. 4.13)
        const Scalar_ fval = (Scalar_(1.0) - xFrac) * s0.first + xFrac * s1.first;
        const Scalar_ gval = (Scalar_(1.0) - xFrac) * s0.second + xFrac * s1.second;

        fAux_(idx[0], idx[1], idc) = fval;
        gAux_(idx[0], idx[1], idc) = gval;
        continue;
      }

      // Full 2D IDW interpolation
      // Integer neighbours around source (floor/ceil)
      const Index x0 = static_cast<Index>(std::floor(xSrcF));
      const Index y0 = static_cast<Index>(std::floor(ySrcF));
      const Index x1 = x0 + 1;
      const Index y1 = y0 + 1;

      // helper to sample a lattice node (returns pair<f,g>)
      auto sampleNode = [&](Index sx, Index sy) -> std::pair<Scalar_, Scalar_> {
        // clamp x to domain (permissive / nearest interior)
        Index sxClamped = sx;
        if (sxClamped < 0) sxClamped = 0;
        else if (sxClamped >= nx) sxClamped = nx - 1;

        // y boundary: periodic wrap
        Index syWrapped = sy;
        if (syWrapped < 0)
          syWrapped += ny;
        else if (syWrapped >= ny)
          syWrapped -= ny;

        auto& srcLat = lattices_(sxClamped, syWrapped);
        return { srcLat.F(idc), srcLat.G(idc) };
      };

      // gather the four neighbor samples (may include bounce-back samples)
      const auto s00 = sampleNode(x0, y0);
      const auto s10 = sampleNode(x1, y0);
      const auto s01 = sampleNode(x0, y1);
      const auto s11 = sampleNode(x1, y1);

      // compute Euclidean distances to neighbors
      const Scalar_ dx00 = xSrcF - Scalar_(x0);
      const Scalar_ dy00 = ySrcF - Scalar_(y0);
      const Scalar_ d00 = std::sqrt(dx00 * dx00 + dy00 * dy00);

      const Scalar_ dx10 = xSrcF - Scalar_(x1);
      const Scalar_ dy10 = ySrcF - Scalar_(y0);
      const Scalar_ d10 = std::sqrt(dx10 * dx10 + dy10 * dy10);

      const Scalar_ dx01 = xSrcF - Scalar_(x0);
      const Scalar_ dy01 = ySrcF - Scalar_(y1);
      const Scalar_ d01 = std::sqrt(dx01 * dx01 + dy01 * dy01);

      const Scalar_ dx11 = xSrcF - Scalar_(x1);
      const Scalar_ dy11 = ySrcF - Scalar_(y1);
      const Scalar_ d11 = std::sqrt(dx11 * dx11 + dy11 * dy11);

      // Direct hit checks (distance ~0) -> return that node's value to avoid div-by-zero
      if (d00 < tiny) {
        fAux_(idx[0], idx[1], idc) = s00.first;
        gAux_(idx[0], idx[1], idc) = s00.second;
        continue;
      }
      if (d10 < tiny) {
        fAux_(idx[0], idx[1], idc) = s10.first;
        gAux_(idx[0], idx[1], idc) = s10.second;
        continue;
      }
      if (d01 < tiny) {
        fAux_(idx[0], idx[1], idc) = s01.first;
        gAux_(idx[0], idx[1], idc) = s01.second;
        continue;
      }
      if (d11 < tiny) {
        fAux_(idx[0], idx[1], idc) = s11.first;
        gAux_(idx[0], idx[1], idc) = s11.second;
        continue;
      }

      // compute IDW weights with exponent alpha
      const Scalar_ w00 = Scalar_(1.0) / std::pow(std::max(d00, tiny), alpha);
      const Scalar_ w10 = Scalar_(1.0) / std::pow(std::max(d10, tiny), alpha);
      const Scalar_ w01 = Scalar_(1.0) / std::pow(std::max(d01, tiny), alpha);
      const Scalar_ w11 = Scalar_(1.0) / std::pow(std::max(d11, tiny), alpha);

      const Scalar_ wsum = w00 + w10 + w01 + w11;

      // If weights are degenerate, fall back to nearest-neighbor/bounce-back
      if (!std::isfinite(wsum) || wsum < tiny) {
        Index xNearest = static_cast<Index>(std::round(xSrcF));
        Index yNearest = static_cast<Index>(std::round(ySrcF));
        if (xNearest < 0 || xNearest >= nx) {
          fAux_(idx[0], idx[1], idc) = latticeIdx.F(latticeIdx.Opposite(idc));
          gAux_(idx[0], idx[1], idc) = latticeIdx.G(latticeIdx.Opposite(idc));
        } else {
          if (yNearest < 0)
            yNearest += ny;
          if (yNearest >= ny)
            yNearest -= ny;
          auto& srcLat = lattices_(xNearest, yNearest);
          fAux_(idx[0], idx[1], idc) = srcLat.F(idc);
          gAux_(idx[0], idx[1], idc) = srcLat.G(idc);
        }
        continue;
      }

      // Weighted average
      const Scalar_ fval = (w00 * s00.first + w10 * s10.first + w01 * s01.first + w11 * s11.first) / wsum;
      const Scalar_ gval = (w00 * s00.second + w10 * s10.second + w01 * s01.second + w11 * s11.second) / wsum;

      fAux_(idx[0], idx[1], idc) = fval;
      gAux_(idx[0], idx[1], idc) = gval;
    }
  };

  iterateDim(streamingOp);
  swapDistributions();
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::Step() {
  Collision();
  Streaming();
  ComputeMacroscopic();
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::iterateDim(const std::function<void(const Array<Index, LbmClassType_::Dim()>&)>& func) {
  Array<Index, LbmClassType_::Dim()> indices = Array<Index, LbmClassType_::Dim()>::Zero();

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

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::swapDistributions() {
  iterateDim(
    [&](const Array<Index, LbmClassType_::Dim()>& idx) {
      auto& lattice = lattices_(idx);
      for (Index idc = 0; idc < LbmClassType_::Speeds(); ++idc) {
        lattice.F(idc) = fAux_(idx[0], idx[1], idc);
        lattice.G(idc) = gAux_(idx[0], idx[1], idc);
      }
    }
  );
}
} // namespace lbmini

#endif  // LBMINI_LBMTUBE_HPP_
