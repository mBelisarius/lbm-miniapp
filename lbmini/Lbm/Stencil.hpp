#ifndef LBMINI_LBM_STENCIL_HPP_
#define LBMINI_LBM_STENCIL_HPP_

#include <cmath>
#include <cstdint>

#include "Utils.hpp"

namespace lbmini {
/**
 * @brief Per-direction shifted-lattice gather, identical for every cell.
 *
 * With a uniform lattice shift `U`, the streaming source of direction `idc`
 * for cell `(i, j)` is the point `(i, j) - (c_i + U)`. Because
 * `floor(i - cx) = i + floor(-cx)` for integer `i`, the integer corner offsets,
 * the corner distances — hence the inverse-distance weights, the "snap to a
 * node" decision and the degenerate fallback — do not depend on the cell. The
 * former per-`(cell, idc)` tables (`4 x int32 + 4 x Scalar` = 48 B per entry,
 * 432 B per cell and step of pure streaming metadata) therefore collapse into
 * this `Q`-entry structure, and the streaming kernel only has to clamp the
 * source column to the tube and wrap the source row periodically.
 *
 * Corner order matches the reference tables: `(x0, y0)`, `(x1, y0)`,
 * `(x0, y1)`, `(x1, y1)` with `x1 = x0 + 1`, `y1 = y0 + 1`.
 *
 * Plain old data made of fixed-size `int32` / `Scalar` arrays only (no `bool`,
 * no pointer), so it can be copied verbatim into constant / kernel-parameter
 * memory of any device, including an OpenCL `struct` of the same declaration.
 */
template<typename Scalar, int Q>
struct DirectionStencil {
  std::int32_t ox[Q];      ///< `x0 - i` (unclamped column offset of corners 0 and 2)
  std::int32_t oy0[Q];     ///< `(y0 - j) mod ny`, in `[0, ny)`
  std::int32_t oy1[Q];     ///< `(y1 - j) mod ny`, in `[0, ny)`
  std::int32_t nearest[Q]; ///< 1: degenerate IDW, nearest node (self-copy outside the tube)
  Scalar w[Q][4];          ///< normalised IDW weights (one-hot when snapped to a node)
  Scalar cix[Q];           ///< `c_ix + Ux`, used by the nearest-node fallback only
  Scalar ciy[Q];           ///< `c_iy + Uy`, used by the nearest-node fallback only
};

/**
 * @brief Evaluates the per-direction gather with the reference table formulas.
 *
 * Same arithmetic as the per-cell `buildStreamingTables()` of the reference
 * solver, evaluated once at `(0, 0)`: corner distances `d_k`, snap to the
 * first corner closer than `tiny`, otherwise the normalised weights
 * `d_k^-idw`, otherwise (non-finite or vanishing sum) the nearest node. The
 * weights agree with the per-cell table bit for bit whenever the shift is a
 * dyadic rational (e.g. the default 0.25) and to rounding error otherwise.
 *
 * @param ny   periodic extent (rows wrap modulo `ny`)
 * @param ux   lattice shift `U_x`
 * @param uy   lattice shift `U_y`
 * @param idw  inverse-distance exponent
 */
template<typename LatticeType, typename Scalar>
DirectionStencil<Scalar, static_cast<int>(LatticeType::Speeds())> BuildDirectionStencil(
  const std::int64_t ny,
  const Scalar ux,
  const Scalar uy,
  const Scalar idw
) {
  constexpr int kQ = static_cast<int>(LatticeType::Speeds());
  constexpr Scalar kTiny = Scalar{ 1.0e-12 };
  auto wrap = [ny](const std::int64_t y) -> std::int32_t {
    std::int64_t w = y % ny;
    if (w < 0)
      w += ny;
    return static_cast<std::int32_t>(w);
  };

  DirectionStencil<Scalar, kQ> st{};
  for (int idc = 0; idc < kQ; ++idc) {
    const Scalar cix = static_cast<Scalar>(LatticeType::Velocity(idc, 0)) + ux;
    const Scalar ciy = static_cast<Scalar>(LatticeType::Velocity(idc, 1)) + uy;
    const Scalar xSrc = -cix;
    const Scalar ySrc = -ciy;
    const auto x0 = static_cast<std::int64_t>(std::floor(xSrc));
    const auto y0 = static_cast<std::int64_t>(std::floor(ySrc));
    const std::int64_t x1 = x0 + 1;
    const std::int64_t y1 = y0 + 1;

    st.ox[idc] = static_cast<std::int32_t>(x0);
    st.oy0[idc] = wrap(y0);
    st.oy1[idc] = wrap(y1);
    st.cix[idc] = cix;
    st.ciy[idc] = ciy;
    st.nearest[idc] = 0;

    const Scalar dx00 = xSrc - static_cast<Scalar>(x0);
    const Scalar dy00 = ySrc - static_cast<Scalar>(y0);
    const Scalar dx10 = xSrc - static_cast<Scalar>(x1);
    const Scalar dy01 = ySrc - static_cast<Scalar>(y1);
    const Scalar d[4] = {
      std::sqrt(dx00 * dx00 + dy00 * dy00),
      std::sqrt(dx10 * dx10 + dy00 * dy00),
      std::sqrt(dx00 * dx00 + dy01 * dy01),
      std::sqrt(dx10 * dx10 + dy01 * dy01),
    };

    // Snap to the first corner (reference order) that coincides with the source.
    int snapped = -1;
    for (int k = 0; k < 4 && snapped < 0; ++k)
      if (d[k] < kTiny)
        snapped = k;
    if (snapped >= 0) {
      for (int k = 0; k < 4; ++k)
        st.w[idc][k] = (k == snapped) ? Scalar{ 1 } : Scalar{ 0 };
      continue;
    }

    Scalar w[4];
    for (int k = 0; k < 4; ++k)
      w[k] = Scalar{ 1 } / std::pow(d[k], idw);
    const Scalar wsum = w[0] + w[1] + w[2] + w[3];
    if (!IsFinite(wsum) || wsum < kTiny) {
      st.nearest[idc] = 1;
      for (int k = 0; k < 4; ++k)
        st.w[idc][k] = Scalar{ 0 };
      continue;
    }
    const Scalar invSum = Scalar{ 1 } / wsum;
    for (int k = 0; k < 4; ++k)
      st.w[idc][k] = w[k] * invSum;
  }
  return st;
}

/**
 * @brief Host-side reference of the gather every fused kernel performs.
 *
 * Resolves direction `idc` of cell `(i, j)` of an `nx x ny` tube into four
 * flat source indices `x * ny + y` and their weights: columns clamped to
 * `[0, nx)`, rows wrapped into `[0, ny)`, the nearest-node fallback
 * self-copying outside the tube. The kernels inline the same logic; this
 * function exists so the tests can prove, for every cell, that it addresses
 * only cells of the domain and reproduces the reference streaming table.
 */
template<typename Scalar, int Q>
void GatherSources(
  const DirectionStencil<Scalar, Q>& st,
  const int idc,
  const std::int64_t i,
  const std::int64_t j,
  const std::int64_t nx,
  const std::int64_t ny,
  std::int64_t (&src)[4],
  Scalar (&w)[4]
) {
  if (st.nearest[idc] != 0) {
    const auto xN = static_cast<std::int64_t>(std::round(static_cast<Scalar>(i) - st.cix[idc]));
    const auto yN = static_cast<std::int64_t>(std::round(static_cast<Scalar>(j) - st.ciy[idc]));
    std::int64_t yW = yN % ny;
    yW += (yW < 0) ? ny : 0;
    const std::int64_t cell = (xN < 0 || xN >= nx) ? i * ny + j : xN * ny + yW;
    for (int k = 0; k < 4; ++k) {
      src[k] = cell;
      w[k] = (k == 0) ? Scalar{ 1 } : Scalar{ 0 };
    }
    return;
  }
  auto clampX = [nx](const std::int64_t x) { return (x < 0) ? std::int64_t{ 0 } : ((x >= nx) ? nx - 1 : x); };
  const std::int64_t xa = i + st.ox[idc];
  const std::int64_t rowA = clampX(xa) * ny;
  const std::int64_t rowB = clampX(xa + 1) * ny;
  std::int64_t ya = j + st.oy0[idc];
  ya -= (ya >= ny) ? ny : 0;
  std::int64_t yb = j + st.oy1[idc];
  yb -= (yb >= ny) ? ny : 0;
  src[0] = rowA + ya;
  src[1] = rowB + ya;
  src[2] = rowA + yb;
  src[3] = rowB + yb;
  for (int k = 0; k < 4; ++k)
    w[k] = st.w[idc][k];
}
} // namespace lbmini

#endif // LBMINI_LBM_STENCIL_HPP_
