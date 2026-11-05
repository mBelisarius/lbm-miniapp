#ifndef LBMINI_ALPAKA_CPU_LBMTUBEIMPL_HPP_
#define LBMINI_ALPAKA_CPU_LBMTUBEIMPL_HPP_

/**
 * @file Cpu/LbmTubeImpl.hpp
 * @brief Host implementation of `lbmini::alpaka::detail::Engine` for `TargetEnum::CPU`.
 *
 * Only included by `Cpu/LbmTube.cpp`, after `Lbm/Alpaka/LbmTubeImpl.hpp`, whose
 * `Engine::State` derives from the `HostState` defined here. It lives in its own
 * header because it pulls in the GCC vector-extension kernel of
 * `Lbm/HostKernel.hpp`, which the CUDA flavour of the engine (compiled from the
 * same `LbmTubeImpl.hpp`) must never see.
 */

#include "Lbm/Alpaka/LbmTubeImpl.hpp"

#if ALPAKA_OMP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <omp.h>

#include "Lbm/HostKernel.hpp"
#include "Lbm/Simd.hpp"

namespace lbmini::alpaka::detail {
/**
 * @brief Initial state of the row segments owned by this block (first touch).
 *
 * The segments are split into one contiguous range per block, so every page of
 * the distributions is first written by the thread that will mostly update it.
 */
template<typename Scalar, typename LatticeType>
struct HostInitKernel {
  using Kernel = host::Kernel<Scalar, LatticeType, simd::kLanes>;

  ALPAKA_FN_ACC void operator()(
    ::alpaka::onAcc::concepts::Acc auto const& acc,
    const host::Params<Scalar>& params,
    const host::Geometry& geo,
    const host::Partition& part,
    const host::Fields<Scalar>& fields
  ) const {
    const auto [block] = acc.getIdxWithin(::alpaka::onAcc::origin::grid, ::alpaka::onAcc::unit::blocks);
    const auto [blocks] = acc.getExtentsOf(::alpaka::onAcc::origin::grid, ::alpaka::onAcc::unit::blocks);
    const host::Index segments = part.Segments();
    const host::Index first = segments * static_cast<host::Index>(block) / static_cast<host::Index>(blocks);
    const host::Index last = segments * (static_cast<host::Index>(block) + 1) / static_cast<host::Index>(blocks);
    for (host::Index s = first; s < last; ++s) {
      const host::Index i0 = part.FirstBlock(s) * simd::kLanes;
      Kernel::InitRow(params, geo, fields, geo.nx, i0, part.Row(s), i0, part.BlockCount(s));
    }
  }
};

/**
 * @brief One time step: the block pulls row segments from a shared counter.
 *
 * alpaka's `exec::cpuOmpBlocks` hands the blocks to the OpenMP team with a
 * static `omp for`. The launch has exactly one block per thread, and the block
 * does not derive its work from its index: it takes the next unprocessed segment
 * from `next` until none is left, which is a dynamic schedule with chunk 1.
 *
 * @tparam kStoreMacro write `rho`, `p`, `T`, `u` (last step of a `Run()`).
 */
template<typename Scalar, typename LatticeType, bool kStoreMacro>
struct HostStepKernel {
  using Kernel = host::Kernel<Scalar, LatticeType, simd::kLanes>;

  ALPAKA_FN_ACC void operator()(
    ::alpaka::onAcc::concepts::Acc auto const& /*acc*/,
    const host::Params<Scalar>& params,
    const host::Geometry& geo,
    const host::Partition& part,
    const host::Fields<Scalar>& fields,
    std::atomic<host::Index>* const next
  ) const {
    const host::Index segments = part.Segments();
    const host::Index xHi = geo.nx - 1;
    // Relaxed is enough: the counter only hands out work; the launch itself
    // orders the data between steps.
    for (host::Index s = next->fetch_add(1, std::memory_order_relaxed); s < segments;
         s = next->fetch_add(1, std::memory_order_relaxed))
      Kernel::template StepRow<kStoreMacro>(
        params, geo, fields, 0, xHi, part.Row(s), part.FirstBlock(s) * simd::kLanes, part.BlockCount(s)
      );
  }
};

/**
 * @brief alpaka state of one CPU solver, driving the shared SIMD host kernel.
 *
 * Numerically equivalent to `lbmini::plain::LbmTube` (see `Lbm/HostKernel.hpp`;
 * only `exp` differs in the last bit). Unlike the device target, which runs one
 * fused kernel per *cell*, a CPU core is fed whole vector blocks: the physics is
 * the single `host::Kernel` that every host backend shares, and alpaka provides
 * the buffers, the queue and the parallel launch.
 *
 * ### State and data layout
 *
 * Structure of arrays, **x-fastest**, `cell = j * pitch + i` (see `host::Geometry`):
 *
 * | buffer        | size      | layout                  | role                               |
 * | :------------ | :-------- | :---------------------- | :--------------------------------- |
 * | `fA/fB`       | `Q * pl`  | `f[idc * plane + cell]` | post-collision density (ping-pong) |
 * | `gA/gB`       | `Q * pl`  | `g[idc * plane + cell]` | post-collision energy (ping-pong)  |
 * | `xi`          | `D * pl`  | `xi[d * plane + cell]`  | Newton-Raphson warm start          |
 * | `rho/p/tem`   | `plane`   | `x[cell]`               | macroscopic output                 |
 * | `u`           | `D * pl`  | `u[d * plane + cell]`   | macroscopic output                 |
 *
 * The buffers are alpaka host buffers (`onHost::alloc`). alpaka aligns them to
 * the SIMD register width only, so each one is over-allocated by a cache line
 * and used from its first 64-byte boundary. Nothing is zero-filled except the
 * macroscopic fields: the distributions and the warm start are first written by
 * `Init()` from the thread that owns the segment (NUMA first touch).
 *
 * The solver state is the set of *post*-collision populations, as for every host
 * backend: `Init()` runs the first collision, every step pulls the previous
 * post-collision state, collides and writes the other ping-pong buffer (the
 * pointers are swapped, never copied); `rho`, `p`, `T`, `u` are written by the
 * last step of a `Run()` only.
 *
 * ### Launch and scheduling
 *
 * One kernel launch per time step on a blocking queue. The launch spec is
 * `ThreadSpec{ blocks = omp_get_max_threads(), threads = 1 }`: exactly one
 * block per OpenMP thread. The blocks share a `std::atomic` segment counter
 * (reset by the host between launches, which are synchronous), so the work is
 * balanced dynamically in segments of ~256 cells even though alpaka itself
 * distributes the blocks statically. A static schedule leaves a slow core (a
 * hybrid CPU's E-core, an SMT sibling, a time-sliced vCPU) holding back the
 * whole team at the end of every step.
 *
 * The thread count and the `OMP_PLACES` / `OMP_PROC_BIND` affinity are
 * process-global settings owned by `lbmini::Executor`; the launch only consumes
 * them. A `Run()` must not be issued from inside an OpenMP parallel region
 * (alpaka would then execute the kernel collectively with the enclosing team).
 */
template<typename Scalar, typename LatticeType>
class HostState {
public:
  using Index = Eigen::Index;

  HostState(
    const FluidData<Scalar>& fluid,
    const MeshData<Scalar, LatticeType::Dim()>& mesh,
    const ControlData<Scalar>& control
  )
    : geo_(host::MakeGeometry(mesh.size[0], mesh.size[1])),
      params_(host::MakeParams<Scalar, LatticeType>(fluid, mesh, control)),
      part_(host::MakePartition(geo_.pitch, geo_.ny, simd::kLanes, omp_get_max_threads())),
      device_(MakeDevice()),
      queue_(device_.makeQueue(Traits::kQueueKind)),
      fA_(alloc(kQ * geo_.plane)),
      fB_(alloc(kQ * geo_.plane)),
      gA_(alloc(kQ * geo_.plane)),
      gB_(alloc(kQ * geo_.plane)),
      xi_(alloc(kD * geo_.plane)),
      rho_(alloc(geo_.plane)),
      p_(alloc(geo_.plane)),
      tem_(alloc(geo_.plane)),
      u_(alloc(kD * geo_.plane)),
      fCur_(aligned(fA_)),
      fNext_(aligned(fB_)),
      gCur_(aligned(gA_)),
      gNext_(aligned(gB_)),
      xiData_(aligned(xi_)),
      rhoData_(aligned(rho_)),
      pData_(aligned(p_)),
      temData_(aligned(tem_)),
      uData_(aligned(u_)) {
    // The macroscopic fields are what the getters see before the first Init().
    for (Buffer* field : { &rho_, &p_, &tem_, &u_ })
      ::alpaka::onHost::memset(queue_, *field, 0);
    ::alpaka::onHost::wait(queue_);
  }

  HostState(const HostState&) = delete;

  HostState& operator=(const HostState&) = delete;

  HostState(HostState&&) = delete;

  HostState& operator=(HostState&&) = delete;

  ~HostState() {
    // Nothing may still be reading or writing a buffer when it is released.
    ::alpaka::onHost::wait(queue_);
  }

  void Init() {
    // The buffers are only ever written by kernels, so a re-Init() simply
    // restarts from the initial condition, in the current ping-pong buffer.
    host::Fields<Scalar> fields;
    fields.fOut = fCur_;
    fields.gOut = gCur_;
    fields.xi = xiData_;
    fields.rho = rhoData_;
    fields.p = pData_;
    fields.tem = temData_;
    fields.u = uData_;
    queue_.enqueue(
      launchSpec(),
      ::alpaka::KernelBundle{ HostInitKernel<Scalar, LatticeType>{}, params_, geo_, part_, fields }
    );
    ::alpaka::onHost::wait(queue_);
  }

  void Run(const Index steps) {
    for (Index t = 0; t < steps; ++t) {
      if (t + 1 == steps)
        enqueueStep<true>();
      else
        enqueueStep<false>();
      std::swap(fCur_, fNext_);
      std::swap(gCur_, gNext_);
    }
    ::alpaka::onHost::wait(queue_);
  }

  void CopyRho(Scalar* dst) const { host::ExportField(rhoData_, geo_, host::Index{ 0 }, dst); }

  void CopyP(Scalar* dst) const { host::ExportField(pData_, geo_, host::Index{ 0 }, dst); }

  void CopyT(Scalar* dst) const { host::ExportField(temData_, geo_, host::Index{ 0 }, dst); }

  /// Copy the velocity into `dst` as `[cell][d]` (RowMajor `[i][j][d]`).
  void CopyU(Scalar* dst) const { host::ExportVector(uData_, uData_ + geo_.plane, geo_, host::Index{ 0 }, dst); }

private:
  using Traits = TargetTraits<TargetEnum::CPU>;
  using Extent = ::alpaka::Vec<Index, 1>;
  using Device = decltype(::alpaka::onHost::makeDeviceSelector(Traits::kApi, Traits::kDeviceKind).makeDevice(0));
  using Queue = decltype(std::declval<Device&>().makeQueue(Traits::kQueueKind));
  using Buffer = decltype(::alpaka::onHost::alloc<Scalar>(std::declval<Device&>(), Extent{}));

  static constexpr int kQ = host::kQ;
  static constexpr int kD = host::kD;
  static constexpr std::uintptr_t kAlignment = 64;
  static constexpr Index kAlignmentSlack = kAlignment / sizeof(Scalar);

  template<bool kStoreMacro>
  void enqueueStep() {
    // Every launch is synchronous (blocking queue), so no block still reads the counter.
    next_.store(0, std::memory_order_relaxed);
    host::Fields<Scalar> fields;
    fields.fIn = fCur_;
    fields.gIn = gCur_;
    fields.fOut = fNext_;
    fields.gOut = gNext_;
    fields.xi = xiData_;
    fields.rho = rhoData_;
    fields.p = pData_;
    fields.tem = temData_;
    fields.u = uData_;
    queue_.enqueue(
      launchSpec(),
      ::alpaka::KernelBundle{
        HostStepKernel<Scalar, LatticeType, kStoreMacro>{}, params_, geo_, part_, fields, &next_
      }
    );
  }

  /// One block per OpenMP thread of the calling thread's team.
  static auto launchSpec() {
    const auto blocks = static_cast<std::uint32_t>(std::max(1, omp_get_max_threads()));
    return ::alpaka::onHost::ThreadSpec{ ::alpaka::Vec{ blocks }, ::alpaka::Vec{ std::uint32_t{ 1 } }, Traits::kExecutor };
  }

  static Device MakeDevice() {
    auto selector = ::alpaka::onHost::makeDeviceSelector(Traits::kApi, Traits::kDeviceKind);
    if (!selector.isAvailable())
      throw std::runtime_error(std::string(Traits::kName) + ": no alpaka device available for this target.");
    return selector.makeDevice(0);
  }

  /// `count` scalars plus the slack needed to start on a cache line.
  Buffer alloc(const host::Index count) {
    return ::alpaka::onHost::alloc<Scalar>(device_, Extent{ static_cast<Index>(count) + kAlignmentSlack });
  }

  static Scalar* aligned(Buffer& buffer) {
    const auto address = reinterpret_cast<std::uintptr_t>(::alpaka::onHost::data(buffer));
    return reinterpret_cast<Scalar*>((address + kAlignment - 1) / kAlignment * kAlignment);
  }

  host::Geometry geo_;
  host::Params<Scalar> params_;
  host::Partition part_;

  // Declaration order is construction order: the device and its queue outlive
  // every buffer allocated from them.
  Device device_;
  Queue queue_;

  Buffer fA_;
  Buffer fB_;
  Buffer gA_;
  Buffer gB_;
  Buffer xi_;
  Buffer rho_;
  Buffer p_;
  Buffer tem_;
  Buffer u_;

  // Cache-line aligned views into the buffers above.
  // Ping-pong roles: `fCur_`/`gCur_` hold the current post-collision state.
  Scalar* fCur_;
  Scalar* fNext_;
  Scalar* gCur_;
  Scalar* gNext_;
  Scalar* xiData_;
  Scalar* rhoData_;
  Scalar* pData_;
  Scalar* temData_;
  Scalar* uData_;

  alignas(64) std::atomic<host::Index> next_{ 0 };
};
} // namespace lbmini::alpaka::detail

#endif // ALPAKA_OMP

#endif // LBMINI_ALPAKA_CPU_LBMTUBEIMPL_HPP_
