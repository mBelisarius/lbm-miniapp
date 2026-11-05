#ifndef LBMINI_CUDA_LBMTUBE_HPP_
#define LBMINI_CUDA_LBMTUBE_HPP_

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Guards.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/Stencil.hpp"
#include "Lbm/Cuda/Gpu/LatticeD2Q9.hpp"

namespace lbmini::cuda::gpu {

using Index = Eigen::Index;

namespace internal {
inline void CheckCuda(const cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

/// Directions and dimensions of the only lattice the kernels implement (D2Q9).
inline constexpr int kQ = 9;
inline constexpr int kD = 2;

/**
 * @brief Everything the kernels need besides the buffers.
 *
 * Trivially copyable and passed *by value* as a `__grid_constant__` kernel
 * argument: it lives in the kernel-parameter constant bank of each launch, so
 * every field read with a compile-time offset (all of them once the direction
 * loops are unrolled) is an immediate constant-bank operand — no load, no
 * register, no per-thread stack copy. Unlike a module-wide `__constant__`
 * symbol it is private to the launch, so concurrent solvers with different
 * meshes or shifts cannot overwrite each other's parameters.
 *
 * The shifted velocities and their pairwise products are the exact values the
 * reference evaluates in its inner loops (`Cshift(idc, a) = c_ia + U_a`,
 * `Cshift(idc, a) * Cshift(idc, b)`), computed once on the host.
 */
template<typename Scalar>
struct KernelParams {
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

  Scalar cs[kD][kQ];  ///< `c_ia + U_a`
  Scalar condOverCp;  ///< `conductivity / cp`: `kappa / (rho cp) = condOverCp / rho`

  DirectionStencil<Scalar, kQ> stencil;
};

/// Device pointers of the per-cell state that is not ping-ponged.
template<typename Scalar>
struct DeviceFields {
  Scalar* xi;  ///< Newton warm start, `xi[d * N + cell]`
  Scalar* rho; ///< density, `rho[cell]`
  Scalar* p;   ///< pressure, `p[cell]`
  Scalar* tem; ///< temperature, `tem[cell]`
  Scalar* u;   ///< velocity, interleaved `u[cell * D + d]` (the getter layout)
};

/// Initial condition + first collision into `fOut`/`gOut` (asynchronous).
template<typename Scalar>
void LaunchInit(
  const KernelParams<Scalar>& params,
  Scalar* fOut,
  Scalar* gOut,
  const DeviceFields<Scalar>& fields,
  cudaStream_t stream
);

/// One fused stream + macroscopic + collision step (asynchronous).
template<typename Scalar>
void LaunchStep(
  const KernelParams<Scalar>& params,
  const Scalar* fIn,
  const Scalar* gIn,
  Scalar* fOut,
  Scalar* gOut,
  const DeviceFields<Scalar>& fields,
  bool storeMacro,
  cudaStream_t stream
);

/// Owning `cudaMalloc` array; released exactly once, never copied or moved.
template<typename Scalar>
class DeviceArray {
public:
  DeviceArray(const Index count, const char* what) {
    CheckCuda(cudaMalloc(&ptr_, static_cast<std::size_t>(count) * sizeof(Scalar)), what);
  }

  DeviceArray(const DeviceArray&) = delete;

  DeviceArray& operator=(const DeviceArray&) = delete;

  DeviceArray(DeviceArray&&) = delete;

  DeviceArray& operator=(DeviceArray&&) = delete;

  ~DeviceArray() { cudaFree(ptr_); }

  [[nodiscard]] Scalar* get() const { return ptr_; }

private:
  Scalar* ptr_ = nullptr;
};

/// Owning non-blocking CUDA stream (does not synchronise with the legacy default stream).
class Stream {
public:
  Stream() { CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate"); }

  Stream(const Stream&) = delete;

  Stream& operator=(const Stream&) = delete;

  Stream(Stream&&) = delete;

  Stream& operator=(Stream&&) = delete;

  ~Stream() {
    // Nothing may still be reading or writing a buffer when it is released.
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }

  [[nodiscard]] cudaStream_t get() const { return stream_; }

private:
  cudaStream_t stream_ = nullptr;
};
} // namespace internal

/**
 * @brief CUDA variant of the compressible LBM tube solver.
 *
 * Numerically equivalent to `lbmini::plain::LbmTube` (same operations in the
 * same order, up to FMA contraction and the device math library), with the
 * fused single-kernel design of the alpaka backend written in native CUDA.
 *
 * ### Data layout and memory traffic
 *
 * | buffer      | size    | layout              | role                               |
 * | :---------- | :------ | :------------------ | :--------------------------------- |
 * | `fA/fB`     | `Q * N` | `f[idc * N + cell]` | post-collision density (ping-pong) |
 * | `gA/gB`     | `Q * N` | `g[idc * N + cell]` | post-collision energy (ping-pong)  |
 * | `xi`        | `D * N` | `xi[d * N + cell]`  | Newton-Raphson warm start          |
 * | `rho/p/tem` | `N`     | `x[cell]`           | macroscopic output                 |
 * | `u`         | `D * N` | `u[cell * D + d]`   | macroscopic output                 |
 *
 * `cell = i * ny + j`: a direction plane is contiguous, so consecutive threads
 * of a warp touch consecutive addresses for every load and store. The state
 * after `n` steps is kept as *post*-collision populations: `Init()` runs the
 * first collision itself and each step's collision prepares the next one. The
 * observable macroscopic fields are identical to the reference
 * collide-then-stream sequence. Only the two warm-start components the
 * collision reads are kept (the reference's component 0 is never read).
 *
 * ### One kernel per step
 *
 * Each thread owns one cell and *pulls* the 9 post-collision populations of
 * the previous step from the four inverse-distance-weighted source cells of
 * each direction, reduces them to the macroscopic fields, evaluates the
 * equilibria, the Newton-Raphson energy closure and the BGK / Knudsen
 * collision in registers, and writes the new populations to the other
 * ping-pong buffer (pointers are swapped, never copied). With a uniform
 * lattice shift the gather is the same for every cell up to a clamp in x and a
 * wrap in y, so the kernel reads the per-direction `lbmini::DirectionStencil`
 * (`Lbm/Stencil.hpp`) from the parameter bank instead of per-cell tables;
 * indices are in range by construction. Per cell and step that is
 * 2 x 9 x 8 B of streamed reads (neighbour reuse is served by L1/L2),
 * 2 x 9 x 8 B of writes and 2 x 2 x 8 B of warm start, ~320 B.
 *
 * ### Host side
 *
 * `Run(steps)` enqueues `steps` kernels on the solver's own stream, the last
 * one also storing the macroscopic fields (the only point a getter can observe
 * them), and synchronises once at the end so a caller's wall clock covers the
 * device work. The getters copy device to host on the same stream.
 */
template<typename Scalar, typename LatticeType>
class LbmTube : public ILbmTube<Scalar, LatticeType> {
public:
  using Index = Eigen::Index;

  template<typename Type, Index NumIndices>
  using Tensor = Eigen::Tensor<Type, NumIndices, Eigen::RowMajor>;

  static constexpr Index kDim_ = LatticeType::Dim();
  static constexpr Index kQ_ = LatticeType::Speeds();

  static_assert(kDim_ == internal::kD && kQ_ == internal::kQ, "the CUDA kernels implement the D2Q9 lattice");

  LbmTube(
    const FluidData<Scalar>& fluid,
    const MeshData<Scalar, kDim_>& mesh,
    const ControlData<Scalar>& control,
    const PerformanceData& performance
  );

  // The solver owns `cudaMalloc` arrays and a stream, and caches the ping-pong
  // roles `fCur_`/`fNext_`/`gCur_`/`gNext_` as raw pointers into them; a copy
  // or move would leave those dangling. Solvers are heavy, unique objects held
  // behind `ILbmTubeNd*`, so the operations are deleted rather than
  // implemented, uniformly across every backend. See docs/memory-safety.md §5.2.
  LbmTube(const LbmTube&) = delete;

  LbmTube& operator=(const LbmTube&) = delete;

  LbmTube(LbmTube&&) = delete;

  LbmTube& operator=(LbmTube&&) = delete;

  ~LbmTube() override = default;

  [[nodiscard]] Tensor<Scalar, kDim_> P() const override { return copyScalarField(p_.get(), "pressure"); }

  [[nodiscard]] Tensor<Scalar, kDim_> Rho() const override { return copyScalarField(rho_.get(), "density"); }

  [[nodiscard]] Tensor<Scalar, kDim_> T() const override { return copyScalarField(tem_.get(), "temperature"); }

  [[nodiscard]] Tensor<Scalar, kDim_ + 1> U() const override {
    // The device field is already interleaved `[cell][d]`, i.e. RowMajor
    // `[i][j][d]`: one straight copy, no host-side transposition.
    Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
    copyToHost(out.data(), u_.get(), N_ * kDim_, "cudaMemcpy velocity to host");
    return out;
  }

  void Init() override;

  void Step(bool save) override;

  void Run(Index steps, bool save) override;

private:
  [[nodiscard]] Tensor<Scalar, kDim_> copyScalarField(const Scalar* src, const char* what) const {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    copyToHost(out.data(), src, N_, (std::string("cudaMemcpy ") + what + " to host").c_str());
    return out;
  }

  void copyToHost(Scalar* dst, const Scalar* src, const Index count, const char* what) const {
    internal::CheckCuda(
      cudaMemcpyAsync(dst, src, static_cast<std::size_t>(count) * sizeof(Scalar), cudaMemcpyDeviceToHost, stream_.get()),
      what
    );
    internal::CheckCuda(cudaStreamSynchronize(stream_.get()), what);
  }

  [[nodiscard]] internal::DeviceFields<Scalar> fields() const {
    return { xi_.get(), rho_.get(), p_.get(), tem_.get(), u_.get() };
  }

  static internal::KernelParams<Scalar> makeParams(
    const FluidData<Scalar>& fluid,
    const MeshData<Scalar, kDim_>& mesh,
    const ControlData<Scalar>& control
  );

  Index nx_;
  Index ny_;
  Index N_;
  internal::KernelParams<Scalar> params_;

  internal::DeviceArray<Scalar> fA_;
  internal::DeviceArray<Scalar> fB_;
  internal::DeviceArray<Scalar> gA_;
  internal::DeviceArray<Scalar> gB_;
  internal::DeviceArray<Scalar> xi_;
  internal::DeviceArray<Scalar> rho_;
  internal::DeviceArray<Scalar> p_;
  internal::DeviceArray<Scalar> tem_;
  internal::DeviceArray<Scalar> u_;

  // Declared after the arrays, hence destroyed before them: its destructor
  // drains every kernel and copy that still uses them before they are freed.
  internal::Stream stream_;

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
  const PerformanceData& /*performance*/
)
  : nx_((ValidateMeshExtents(mesh.size[0], mesh.size[1], kQ_, "CUDA GPU"), mesh.size[0])),
    ny_(mesh.size[1]),
    N_(nx_ * ny_),
    params_(makeParams(fluid, mesh, control)),
    fA_(N_ * kQ_, "cudaMalloc f"),
    fB_(N_ * kQ_, "cudaMalloc f (ping-pong)"),
    gA_(N_ * kQ_, "cudaMalloc g"),
    gB_(N_ * kQ_, "cudaMalloc g (ping-pong)"),
    xi_(N_ * kDim_, "cudaMalloc xi"),
    rho_(N_, "cudaMalloc rho"),
    p_(N_, "cudaMalloc pressure"),
    tem_(N_, "cudaMalloc temperature"),
    u_(N_ * kDim_, "cudaMalloc velocity"),
    stream_(),
    fCur_(fA_.get()),
    fNext_(fB_.get()),
    gCur_(gA_.get()),
    gNext_(gB_.get()) {}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Init() {
  // Every buffer is written by the kernel, so a re-Init() simply restarts
  // from the initial condition; nothing is transferred from the host.
  fCur_ = fA_.get();
  fNext_ = fB_.get();
  gCur_ = gA_.get();
  gNext_ = gB_.get();
  internal::LaunchInit<Scalar>(params_, fCur_, gCur_, fields(), stream_.get());
  internal::CheckCuda(cudaStreamSynchronize(stream_.get()), "Init");
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Step(const bool save) {
  Run(1, save);
}

template<typename Scalar, typename LatticeType>
void LbmTube<Scalar, LatticeType>::Run(const Index steps, bool /*save*/) {
  if (steps <= 0)
    return;
  for (Index t = 0; t < steps; ++t) {
    internal::LaunchStep<Scalar>(params_, fCur_, gCur_, fNext_, gNext_, fields(), t + 1 == steps, stream_.get());
    std::swap(fCur_, fNext_);
    std::swap(gCur_, gNext_);
  }
  // Asynchronous launches: wait once so the caller's wall clock covers the
  // device work and any fault surfaces here rather than in a later getter.
  internal::CheckCuda(cudaStreamSynchronize(stream_.get()), "Run");
}

template<typename Scalar, typename LatticeType>
internal::KernelParams<Scalar> LbmTube<Scalar, LatticeType>::makeParams(
  const FluidData<Scalar>& fluid,
  const MeshData<Scalar, kDim_>& mesh,
  const ControlData<Scalar>& control
) {
  internal::KernelParams<Scalar> p{};
  p.nx = static_cast<std::int32_t>(mesh.size[0]);
  p.ny = static_cast<std::int32_t>(mesh.size[1]);
  p.cells = static_cast<std::int32_t>(mesh.size[0] * mesh.size[1]);
  p.ushiftX = control.U(0);
  p.ushiftY = control.U(1);
  p.cv = fluid.specificHeatCv;
  p.cp = fluid.specificHeatCp;
  p.invCv = Scalar{ 1 } / fluid.specificHeatCv;
  p.rgas = fluid.constant;
  p.viscosity = fluid.viscosity;
  p.conductivity = fluid.conductivity;
  p.densityL = fluid.densityL;
  p.pressureL = fluid.pressureL;
  p.densityR = fluid.densityR;
  p.pressureR = fluid.pressureR;
  p.condOverCp = fluid.conductivity / fluid.specificHeatCp;
  for (Index idc = 0; idc < kQ_; ++idc) {
    p.cs[0][idc] = LatticeType::Cshift(idc, 0, control.U(0));
    p.cs[1][idc] = LatticeType::Cshift(idc, 1, control.U(1));
  }
  p.stencil = BuildDirectionStencil<LatticeType>(p.ny, control.U(0), control.U(1), control.idw);
  return p;
}

} // namespace lbmini::cuda::gpu

#endif // LBMINI_CUDA_LBMTUBE_HPP_
