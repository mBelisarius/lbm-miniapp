#ifndef LBMINI_PLAIN_LBMTUBE_HPP_
#define LBMINI_PLAIN_LBMTUBE_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/OpenAcc/CellPhysics.hpp"

namespace lbmini::plain {
/**
 * @brief Serial, scalar variant of the compressible LBM tube solver: the
 *        single-core baseline of every other backend.
 *
 * It runs the same algorithm as all the other backends and differs from them
 * only in how the work is executed: one thread, one cell at a time, no
 * vector intrinsics, no directives. The per-cell physics (gather stencil,
 * macroscopic reduction, equilibria, factorised Newton-Raphson energy closure
 * and BGK/Knudsen collision) is `lbmini::directive::Cell` of
 * `Lbm/OpenAcc/CellPhysics.hpp`, the one used by the directive backends and
 * the transcription of the arithmetic of `lbmini::cuda::gpu`; the streaming
 * stencil is the per-direction `lbmini::DirectionStencil`. Hence there are no
 * per-`(cell, direction)` index tables, and the two ping-pong buffers of each
 * population are the only large state.
 *
 * The independent specification of the model, which keeps the general
 * per-direction closure and the precomputed streaming tables, lives in
 * `test/ReferenceTube.hpp`; `test/backend_equivalence_test.cpp` checks this
 * class against it.
 *
 * ### State and data layout
 *
 * The state after `n` steps is the set of *post*-collision populations,
 * structure-of-arrays, `cell = i * ny + j`:
 *
 * | buffer      | size    | layout              | role                                |
 * | :---------- | :------ | :------------------ | :---------------------------------- |
 * | `fA/fB`     | `Q * N` | `f[idc * N + cell]` | post-collision density (ping-pong)  |
 * | `gA/gB`     | `Q * N` | `g[idc * N + cell]` | post-collision energy (ping-pong)   |
 * | `xi`        | `2 * N` | `xi[d * N + cell]`  | Newton-Raphson warm start           |
 * | `rho/p/tem` | `N`     | `x[cell]`           | macroscopic output                  |
 * | `u`         | `2 * N` | `u[d * N + cell]`   | macroscopic output                  |
 *
 * That is 43 values, 344 B, per cell in double precision.
 *
 * ### Time step
 *
 * `Init()` seeds the Sod left/right equilibria, recovers the macroscopic
 * fields (what the getters see after `Init()`) and runs the first collision.
 * Each step is then one fused pass over the cells: pull-stream the 9
 * directions from their 4 inverse-distance source cells, reduce them to the
 * macroscopic fields, evaluate the equilibria, the energy closure and the
 * collision, and write the post-collision populations to the other ping-pong
 * buffer (a pointer swap, never a copy). The macroscopic fields are stored by
 * the last step of a `Run()` only, the only point where a getter can see them.
 */
template<typename Scalar, typename LatticeType>
class LbmTube : public ILbmTube<Scalar, LatticeType> {
public:
  using Index = Eigen::Index;

  template<typename Type, Index NumIndices>
  using Tensor = Eigen::Tensor<Type, NumIndices, Eigen::RowMajor>;

  static constexpr Index kDim_ = LatticeType::Dim();
  static constexpr Index kQ_ = LatticeType::Speeds();

  LbmTube(
    const FluidData<Scalar>& fluid,
    const MeshData<Scalar, kDim_>& mesh,
    const ControlData<Scalar>& control,
    const PerformanceData& performance
  );

  // The solver caches raw pointers into its own buffers (`fCur_`/`fNext_`,
  // `gCur_`/`gNext_`); a copy would leave them dangling into the source's
  // buffers. Solvers are heavy, unique objects held behind `ILbmTubeNd*`, so
  // the operations are deleted rather than implemented, uniformly across
  // every backend. See docs/memory-safety.md §5.2.
  LbmTube(const LbmTube&) = delete;

  LbmTube& operator=(const LbmTube&) = delete;

  LbmTube(LbmTube&&) = delete;

  LbmTube& operator=(LbmTube&&) = delete;

  /**
   * @name Macroscopic field getters
   * @brief Copy-out accessors for pressure / density / temperature / velocity.
   *
   * They return the fields of the last `Init()` or of the last step of the
   * last `Run()`.
   * @{
   */
  [[nodiscard]] Tensor<Scalar, kDim_> P() const override;

  [[nodiscard]] Tensor<Scalar, kDim_> Rho() const override;

  [[nodiscard]] Tensor<Scalar, kDim_> T() const override;

  [[nodiscard]] Tensor<Scalar, kDim_ + 1> U() const override;

  /** @} */

  void Init() override;

  void Step(bool save) override;

  void Run(Index steps, bool save) override;

private:
  using Params = directive::CellParams<Scalar>;
  using Cell = directive::Cell<Scalar>;
  static constexpr int kQ = directive::kQ;
  static constexpr int kD = directive::kD;

  /// One fused stream + macroscopic + collision step over all cells.
  template<bool kStoreMacro>
  void step();

  const FluidData<Scalar> kFluid_;
  const MeshData<Scalar, kDim_> kMesh_;
  const ControlData<Scalar> kControl_;
  const PerformanceData kPerformance_;

  Index nx_;
  Index ny_;
  Index N_;

  Params params_;

  lbmini::DeviceBuffer<Scalar> fA_;
  lbmini::DeviceBuffer<Scalar> fB_;
  lbmini::DeviceBuffer<Scalar> gA_;
  lbmini::DeviceBuffer<Scalar> gB_;
  lbmini::DeviceBuffer<Scalar> xi_;
  lbmini::DeviceBuffer<Scalar> rho_;
  lbmini::DeviceBuffer<Scalar> p_;
  lbmini::DeviceBuffer<Scalar> tem_;
  lbmini::DeviceBuffer<Scalar> u_;

  // Ping-pong roles of the distribution buffers (swapped, never copied).
  Scalar* fCur_ = nullptr;
  Scalar* fNext_ = nullptr;
  Scalar* gCur_ = nullptr;
  Scalar* gNext_ = nullptr;
};

template<typename Scalar, typename LatticeType>
LbmTube<Scalar, LatticeType>::LbmTube(
  const FluidData<Scalar>& fluid,
  const MeshData<Scalar, kDim_>& mesh,
  const ControlData<Scalar>& control,
  const PerformanceData& performance
)
  : kFluid_(fluid), kMesh_(mesh), kControl_(control), kPerformance_(performance) {
  // Everything below derives from these two extents, so they are checked
  // before the first allocation: a zero extent yields an empty buffer the
  // loops still index, and a cell count beyond int32 wraps the 32-bit cell
  // indices of the gather.
  ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "plain::LbmTube");

  nx_ = mesh.size[0];
  ny_ = mesh.size[1];
  N_ = nx_ * ny_;
  params_ = directive::MakeCellParams<Scalar, LatticeType>(fluid, mesh, control);

  fA_.assign(kQ * N_, Scalar{ 0 });
  fB_.assign(kQ * N_, Scalar{ 0 });
  gA_.assign(kQ * N_, Scalar{ 0 });
  gB_.assign(kQ * N_, Scalar{ 0 });
  xi_.assign(kD * N_, Scalar{ 0 });
  rho_.assign(N_, Scalar{ 0 });
  p_.assign(N_, Scalar{ 0 });
  tem_.assign(N_, Scalar{ 0 });
  u_.assign(kD * N_, Scalar{ 0 });

  fCur_ = fA_.data();
  fNext_ = fB_.data();
  gCur_ = gA_.data();
  gNext_ = gB_.data();
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::P() const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(p_.begin(), p_.end(), out.data());
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::Rho() const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(rho_.begin(), rho_.end(), out.data());
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::T() const -> Tensor<Scalar, kDim_> {
  Tensor<Scalar, kDim_> out(nx_, ny_);
  std::copy(tem_.begin(), tem_.end(), out.data());
  return out;
}

template<typename Scalar, typename LatticeType>
auto LbmTube<Scalar, LatticeType>::U() const -> Tensor<Scalar, kDim_ + 1> {
  // SoA d-major -> RowMajor [i][j][d] Eigen tensor.
  Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
  Scalar* dst = out.data();
  for (Index cell = 0; cell < N_; ++cell)
    for (Index d = 0; d < kD; ++d)
      dst[cell * kD + d] = u_[d * N_ + cell];
  return out;
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Init() {
  fCur_ = fA_.data();
  fNext_ = fB_.data();
  gCur_ = gA_.data();
  gNext_ = gB_.data();

  const Params& prm = params_;
  Scalar* __restrict__ fOut = fCur_;
  Scalar* __restrict__ gOut = gCur_;
  Scalar* __restrict__ xiOut = xi_.data();
  Scalar* __restrict__ rhoOut = rho_.data();
  Scalar* __restrict__ pOut = p_.data();
  Scalar* __restrict__ temOut = tem_.data();
  Scalar* __restrict__ uOut = u_.data();
  const auto nx = static_cast<std::int32_t>(nx_);
  const auto ny = static_cast<std::int32_t>(ny_);
  const std::int64_t N = N_;

  for (std::int32_t ci = 0; ci < nx; ++ci) {
    for (std::int32_t cj = 0; cj < ny; ++cj) {
      const std::int64_t cell = static_cast<std::int64_t>(ci) * ny + cj;
      Scalar f[kQ];
      Scalar g[kQ];
      Scalar xi[kD];
      Scalar rho;
      Scalar ux;
      Scalar uy;
      Scalar tem;
      Scalar pres;
      Cell::Init(prm, ci, f, g, xi, rho, ux, uy, tem, pres);
      rhoOut[cell] = rho;
      pOut[cell] = pres;
      temOut[cell] = tem;
      uOut[0 * N + cell] = ux;
      uOut[1 * N + cell] = uy;
      for (int idc = 0; idc < kQ; ++idc) {
        fOut[idc * N + cell] = f[idc];
        gOut[idc * N + cell] = g[idc];
      }
      xiOut[0 * N + cell] = xi[0];
      xiOut[1 * N + cell] = xi[1];
    }
  }
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Step(const bool save) {
  Run(1, save);
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Run(const Index steps, bool /*save*/) {
  for (Index t = 0; t < steps; ++t) {
    if (t + 1 == steps)
      step<true>();
    else
      step<false>();
    std::swap(fCur_, fNext_);
    std::swap(gCur_, gNext_);
  }
}

template<typename Scalar, typename LatticeType>
template<bool kStoreMacro>
void LbmTube<Scalar, LatticeType>::step() {
  const Params& prm = params_;
  const Scalar* __restrict__ fIn = fCur_;
  const Scalar* __restrict__ gIn = gCur_;
  Scalar* __restrict__ fOut = fNext_;
  Scalar* __restrict__ gOut = gNext_;
  Scalar* __restrict__ xiIo = xi_.data();
  Scalar* __restrict__ rhoOut = rho_.data();
  Scalar* __restrict__ pOut = p_.data();
  Scalar* __restrict__ temOut = tem_.data();
  Scalar* __restrict__ uOut = u_.data();
  const auto nx = static_cast<std::int32_t>(nx_);
  const auto ny = static_cast<std::int32_t>(ny_);
  const std::int64_t N = N_;

  for (std::int32_t ci = 0; ci < nx; ++ci) {
    for (std::int32_t cj = 0; cj < ny; ++cj) {
      const std::int64_t cell = static_cast<std::int64_t>(ci) * ny + cj;

      Scalar f[kQ];
      Scalar g[kQ];
      Cell::Gather(prm, fIn, gIn, ci, cj, f, g);

      Scalar rho;
      Scalar invRho;
      Scalar ux;
      Scalar uy;
      Scalar tem;
      Scalar pres;
      Cell::Macroscopic(f, g, prm, rho, invRho, ux, uy, tem, pres);
      if constexpr (kStoreMacro) {
        rhoOut[cell] = rho;
        pOut[cell] = pres;
        temOut[cell] = tem;
        uOut[0 * N + cell] = ux;
        uOut[1 * N + cell] = uy;
      }

      Scalar xi[kD] = { xiIo[0 * N + cell], xiIo[1 * N + cell] };
      Cell::Collide(prm, rho, invRho, ux, uy, tem, f, g, xi);

      for (int idc = 0; idc < kQ; ++idc) {
        fOut[idc * N + cell] = f[idc];
        gOut[idc * N + cell] = g[idc];
      }
      xiIo[0 * N + cell] = xi[0];
      xiIo[1 * N + cell] = xi[1];
    }
  }
}
} // namespace lbmini::plain

#endif // LBMINI_PLAIN_LBMTUBE_HPP_
