#ifndef LBMINI_PLAIN_LBMTUBE_HPP_
#define LBMINI_PLAIN_LBMTUBE_HPP_

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"

namespace lbmini::plain {
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

protected:
  template<typename Func>
  void iterateDim(Func func);

  void streamingOp(const Array<Index, LbmClassType_::Dim()>& idx);

  void swapDistributionsOp(const Array<Index, LbmClassType_::Dim()>& idx);

private:
  // Constants
  const Scalar_ kTiny_ = Scalar_(1.0e-12);
  const Scalar_ kCs2_ = Scalar_(1.0) / Scalar_(3.0);

  // Members
  const FluidData<Scalar_> kFluid_;
  const ControlData<Scalar_> kControl_;
  const Array<Index, LbmClassType_::Dim()> kSizes_;
  Tensor<LbmClassType_, LbmClassType_::Dim()> lattices_;

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
  : kFluid_(fluid), kControl_(control), kSizes_(mesh.size) {
  Eigen::array<Index, LbmClassType_::Dim()> dims;
  for (Index i = 0; i < LbmClassType_::Dim(); ++i)
    dims[i] = kSizes_[i];
  lattices_.resize(dims);
  rho_.resize(dims);
  p_.resize(dims);
  tem_.resize(dims);

  Eigen::array<Index, LbmClassType_::Dim() + 1> u_dims;
  for (Index i = 0; i < LbmClassType_::Dim(); ++i)
    u_dims[i] = kSizes_[i];
  u_dims[LbmClassType_::Dim()] = LbmClassType_::Dim();
  u_.resize(u_dims);

  Eigen::array<Index, LbmClassType_::Dim() + 1> dist_dims;
  for (Index i = 0; i < LbmClassType_::Dim(); ++i)
    dist_dims[i] = kSizes_[i];
  dist_dims[LbmClassType_::Dim()] = LbmClassType_::Speeds();
  f_.resize(dist_dims);
  feq_.resize(dist_dims);
  faux_.resize(dist_dims);
  g_.resize(dist_dims);
  geq_.resize(dist_dims);
  gaux_.resize(dist_dims);

  Eigen::array<Index, LbmClassType_::Dim() + 1> lastGx_dims;
  for (Index i = 0; i < LbmClassType_::Dim(); ++i)
    lastGx_dims[i] = kSizes_[i];
  lastGx_dims[LbmClassType_::Dim()] = 3;
  lastGx_.resize(lastGx_dims);

  iterateDim(
    [&](const Array<Index, LbmClassType_::Dim()>& idx) {
      auto i = idx[0];
      auto j = idx[1];
      lattices_(i, j) = LbmClassType_(
        &kFluid_,
        &kControl_,
        &rho_(i, j),
        &p_(i, j),
        &tem_(i, j),
        &u_(i, j, 0),
        &f_(i, j, 0),
        &feq_(i, j, 0),
        &g_(i, j, 0),
        &geq_(i, j, 0),
        &lastGx_(i, j, 0)
      );
    }
  );
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::Init() {
  auto initOp = [&](const Array<Index, LbmClassType_::Dim()>& idx) {
    // Apply shock tube condition along x-axis
    if (idx[0] < kSizes_[0] / 2) {
      rho_(idx[0], idx[1]) = kFluid_.densityL;
      p_(idx[0], idx[1]) = kFluid_.pressureL;
    } else {
      rho_(idx[0], idx[1]) = kFluid_.densityR;
      p_(idx[0], idx[1]) = kFluid_.pressureR;
    }

    // Zero initial velocity
    Vector<Scalar_, LbmClassType_::Dim()> u0;
    u0.setZero();

    // Initialize lattice
    lattices_(idx[0], idx[1]).Init(u0, &rho_(idx[0], idx[1]), &p_(idx[0], idx[1]));
  };

  iterateDim(initOp);
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::Step() {
  // Collision
  iterateDim(
    [&](const Array<Index, LbmClassType_::Dim()>& idx) {
      lattices_(idx[0], idx[1]).Collision();
    }
  );

  // Streaming
  iterateDim(
    [&](const Array<Index, LbmClassType_::Dim()>& idx) {
      streamingOp(idx);
    }
  );

  // Swap distributions and update macroscopic fields
  iterateDim(
    [&](const Array<Index, LbmClassType_::Dim()>& idx) {
      swapDistributionsOp(idx);
      lattices_(idx[0], idx[1]).ComputeMacroscopic();
    }
  );
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::Run(Index steps) {
  for (Index step = 0; step < steps; ++step)
    Step();
}

template<typename Scalar_, typename LbmClassType_>
template<typename Func>
void LbmTube<Scalar_, LbmClassType_>::iterateDim(Func func) {
  Eigen::Array<Index, LbmClassType_::Dim(), 1> indices = Eigen::Array<Index, LbmClassType_::Dim(), 1>::Zero();

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
void LbmTube<Scalar_, LbmClassType_>::streamingOp(const Array<Index, LbmClassType_::Dim()>& idx) {
  auto& latticeIdx = lattices_(idx[0], idx[1]);

  for (Index idc = 0; idc < LbmClassType_::Speeds(); ++idc) {
    // Continuous source position: s = x - (c + U)
    const Scalar_ cix = static_cast<Scalar_>(latticeIdx.Velocity(idc, 0)) + kControl_.U(0);
    const Scalar_ ciy = static_cast<Scalar_>(latticeIdx.Velocity(idc, 1)) + kControl_.U(1);

    const Scalar_ xSrcF = static_cast<Scalar_>(idx[0]) - cix;
    const Scalar_ ySrcF = static_cast<Scalar_>(idx[1]) - ciy;

    // Fast 1D special-case: if shift is (nearly) only in x (|ci_y + U_y| < tiny),
    // use (Rathnayake Eq. 4.13) two-node interpolation (exact & cheaper)
    if (std::abs(ciy) < kTiny_) {
      // Nearest integer neighbors in x
      const Index x0 = static_cast<Index>(std::floor(xSrcF));
      const Index x1 = x0 + 1;
      const Scalar_ xFrac = xSrcF - static_cast<Scalar_>(x0);

      // If exactly on node, sample directly (clamp x to domain)
      if (std::abs(xFrac) < kTiny_) {
        Index xClamp = x0;
        if (xClamp < 0)
          xClamp = 0;
        if (xClamp >= kSizes_[0])
          xClamp = kSizes_[0] - 1;

        Index yWrap = static_cast<Index>(std::floor(ySrcF));
        if (yWrap < 0)
          yWrap += kSizes_[1];
        if (yWrap >= kSizes_[1])
          yWrap -= kSizes_[1];

        faux_(idx[0], idx[1], idc) = f_(xClamp, yWrap, idc);
        gaux_(idx[0], idx[1], idc) = g_(xClamp, yWrap, idc);
        continue;
      }

      // Sample left and right neighbors (permissive clamp at x)
      std::pair<Scalar_, Scalar_> s0, s1;

      // Sample at x0 (clamp to [0, nx-1] if out-of-bounds)
      {
        Index xClamp = x0;
        if (xClamp < 0)
          xClamp = 0;
        if (xClamp >= kSizes_[0])
          xClamp = kSizes_[0] - 1;
        Index yWrap = static_cast<Index>(std::floor(ySrcF));
        if (yWrap < 0)
          yWrap += kSizes_[1];
        if (yWrap >= kSizes_[1])
          yWrap -= kSizes_[1];

        s0 = { f_(xClamp, yWrap, idc), g_(xClamp, yWrap, idc) };
      }

      // Sample at x1 (clamp to [0, nx-1] if out-of-bounds)
      {
        Index xClamp = x1;
        if (xClamp < 0)
          xClamp = 0;
        if (xClamp >= kSizes_[0])
          xClamp = kSizes_[0] - 1;
        Index yWrap = static_cast<Index>(std::floor(ySrcF));
        if (yWrap < 0)
          yWrap += kSizes_[1];
        if (yWrap >= kSizes_[1])
          yWrap -= kSizes_[1];

        s1 = { f_(xClamp, yWrap, idc), g_(xClamp, yWrap, idc) };
      }

      // Linear interpolation (Rathnayake Eq. 4.13)
      const Scalar_ fval = (Scalar_(1.0) - xFrac) * s0.first + xFrac * s1.first;
      const Scalar_ gval = (Scalar_(1.0) - xFrac) * s0.second + xFrac * s1.second;

      faux_(idx[0], idx[1], idc) = fval;
      gaux_(idx[0], idx[1], idc) = gval;
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
      if (sxClamped < 0)
        sxClamped = 0;
      else if (sxClamped >= kSizes_[0])
        sxClamped = kSizes_[0] - 1;

      // y boundary: periodic wrap
      Index syWrapped = sy;
      if (syWrapped < 0)
        syWrapped += kSizes_[1];
      else if (syWrapped >= kSizes_[1])
        syWrapped -= kSizes_[1];

      return { f_(sxClamped, syWrapped, idc), g_(sxClamped, syWrapped, idc) };
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
    if (d00 < kTiny_) {
      faux_(idx[0], idx[1], idc) = s00.first;
      gaux_(idx[0], idx[1], idc) = s00.second;
      continue;
    }
    if (d10 < kTiny_) {
      faux_(idx[0], idx[1], idc) = s10.first;
      gaux_(idx[0], idx[1], idc) = s10.second;
      continue;
    }
    if (d01 < kTiny_) {
      faux_(idx[0], idx[1], idc) = s01.first;
      gaux_(idx[0], idx[1], idc) = s01.second;
      continue;
    }
    if (d11 < kTiny_) {
      faux_(idx[0], idx[1], idc) = s11.first;
      gaux_(idx[0], idx[1], idc) = s11.second;
      continue;
    }

    // compute IDW weights with exponent alpha
    const Scalar_ w00 = Scalar_(1.0) / std::pow(std::max(d00, kTiny_), kControl_.idw);
    const Scalar_ w10 = Scalar_(1.0) / std::pow(std::max(d10, kTiny_), kControl_.idw);
    const Scalar_ w01 = Scalar_(1.0) / std::pow(std::max(d01, kTiny_), kControl_.idw);
    const Scalar_ w11 = Scalar_(1.0) / std::pow(std::max(d11, kTiny_), kControl_.idw);

    const Scalar_ wsum = w00 + w10 + w01 + w11;

    // If weights are degenerate, fall back to nearest-neighbor/bounce-back
    if (!std::isfinite(wsum) || wsum < kTiny_) {
      Index xNearest = static_cast<Index>(std::round(xSrcF));
      Index yNearest = static_cast<Index>(std::round(ySrcF));
      if (xNearest < 0 || xNearest >= kSizes_[0]) {
        faux_(idx[0], idx[1], idc) = f_(idx[0], idx[1], latticeIdx.Opposite(idc));
        gaux_(idx[0], idx[1], idc) = g_(idx[0], idx[1], latticeIdx.Opposite(idc));
      } else {
        if (yNearest < 0)
          yNearest += kSizes_[1];
        if (yNearest >= kSizes_[1])
          yNearest -= kSizes_[1];

        faux_(idx[0], idx[1], idc) = f_(xNearest, yNearest, idc);
        gaux_(idx[0], idx[1], idc) = g_(xNearest, yNearest, idc);
      }
      continue;
    }

    // Weighted average
    const Scalar_ fval = (w00 * s00.first + w10 * s10.first + w01 * s01.first + w11 * s11.first) / wsum;
    const Scalar_ gval = (w00 * s00.second + w10 * s10.second + w01 * s01.second + w11 * s11.second) / wsum;

    faux_(idx[0], idx[1], idc) = fval;
    gaux_(idx[0], idx[1], idc) = gval;
  }
}

template<typename Scalar_, typename LbmClassType_>
void LbmTube<Scalar_, LbmClassType_>::swapDistributionsOp(const Array<Index, LbmClassType_::Dim()>& idx) {
  for (Index idc = 0; idc < LbmClassType_::Speeds(); ++idc) {
    f_(idx[0], idx[1], idc) = faux_(idx[0], idx[1], idc);
    g_(idx[0], idx[1], idc) = gaux_(idx[0], idx[1], idc);
  }
}
} // namespace lbmini::plain

#endif  // LBMINI_PLAIN_LBMTUBE_HPP_
