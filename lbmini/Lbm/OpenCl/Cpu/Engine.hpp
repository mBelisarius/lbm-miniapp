#ifndef LBMINI_OPENCL_CPU_ENGINE_HPP_
#define LBMINI_OPENCL_CPU_ENGINE_HPP_

/**
 * @file Engine.hpp
 * @brief Device state and time loop of `opencl::cpu::LbmTube` (vectorised CPU path).
 *
 * The GPU variant keeps the per-cell kernels of `Lbm/OpenCl/kernels.cl` and the
 * engine of `Lbm/OpenCl/Engine.hpp`. On a CPU device that formulation reaches
 * ~4 % of the arithmetic peak, because no work-group vectoriser can turn the
 * fused scalar kernel (data-dependent control flow, divisions, `exp`, wrapped
 * gathers) into SIMD lanes. This engine drives `kernels_cpu.cl` instead: the
 * OpenCL C image of the shared host kernel (`Lbm/HostKernel.hpp`), in which a
 * work-item owns a *block* of `VW` neighbouring cells of one row, in explicit
 * `doubleVW` vectors.
 *
 * ### Design
 *
 * - **Layout of the host kernel**, sized and exported with the same helpers
 *   (`host::MakeGeometry`, `host::ExportField`): x-fastest, `cell = j * pitch + i`
 *   with the pitch padded to a multiple of 16 columns, `x[plane_index * plane + cell]`
 *   with a staggered plane stride. `f`, `g` live in two ping-pong buffers each
 *   and the roles alternate by binding the other pre-built kernel object;
 *   nothing is copied. The pad columns are ordinary cells.
 * - **Vector length** `VW = 2 x CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE` (8 lanes
 *   on AVX2): two independent registers per work-item, because the fused step is
 *   a long chain of dependent operations and one register's worth of cells is
 *   latency-bound.
 * - **Launch shape: one work-group of one work-item per worker thread**
 *   (`CL_DEVICE_MAX_COMPUTE_UNITS`, which pocl sets from `POCL_MAX_PTHREAD_COUNT`).
 *   The cells are cut into row segments (`host::Partition`: ~256 cells of one
 *   row, at least eight per worker, consecutive segments being consecutive rows
 *   of the same column chunk so that the gather window of five rows stays in
 *   cache), and the work-items take them from an atomic queue in the kernel.
 *   The scheduler of pocl's pthread device deals out `remaining / threads`
 *   work-groups at a time, which for a launch of a few hundred short
 *   work-groups is a static schedule; a self-scheduling loop balances to one
 *   segment even when some cores are slower than others.
 * - **No host synchronisation inside a `Run()`**: all steps are enqueued back to
 *   back on an in-order queue and followed by one `finish()` (one more per 4096
 *   steps, to bound the commands in flight). The two queue counters alternate,
 *   each launch zeroing the one the next launch drains. The macroscopic fields
 *   are written only by the last step of a `Run()` (a kernel argument).
 * - **JIT specialisation** per solver instance (`Program.hpp`) plus the
 *   geometry, and a one-segment launch of every kernel in the constructor:
 *   pocl generates the native work-group function lazily, on the first launch.
 *   Doing it here keeps that one-time cost out of the first `Run()`.
 * - **Getters** map the macroscopic buffers (device memory is host memory on a
 *   CPU device) and transpose them with `host::ExportField`.
 */

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "Lbm/HostKernel.hpp"
#include "Lbm/OpenCl/Program.hpp"

namespace lbmini::opencl::detail {
template<typename Scalar, typename LatticeType>
class CpuEngine {
public:
  using Index = Eigen::Index;

  static constexpr Index kDim_ = LatticeType::Dim();
  static constexpr Index kQ_ = LatticeType::Speeds();

  static_assert(std::is_same_v<Scalar, double>, "the OpenCL kernels are written for double precision");
  static_assert(kDim_ == 2 && kQ_ == 9, "the OpenCL kernels implement the D2Q9 lattice");

  CpuEngine(
    const FluidData<Scalar>& fluid,
    const MeshData<Scalar, kDim_>& mesh,
    const ControlData<Scalar>& control,
    const std::string& missingDevice,
    const std::string& who
  )
    : who_(who), nx_(mesh.size[0]), ny_(mesh.size[1]) {
    device_ = SelectDevice(CL_DEVICE_TYPE_CPU, who_, missingDevice);
    context_ = cl::Context(device_);
    queue_ = cl::CommandQueue(context_, device_);

    lanes_ = Lanes(device_);
    geo_ = host::MakeGeometry(nx_, ny_);
    workers_ = std::max<cl_uint>(1, device_.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
    part_ = host::MakePartition(geo_.pitch, geo_.ny, lanes_, static_cast<int>(workers_));

    program_ = BuildProgram(
      context_,
      device_,
      GeometryDefines(host::MakeParams<Scalar, LatticeType>(fluid, mesh, control))
        + ProgramDefines<LatticeType>(fluid, mesh, control, who_)
        + LoadKernelSource(who_, "kernels_cpu.cl"),
      // -w: every vector-typed function of the (fully inlined) kernels draws a -Wpsabi note
      "-w",
      who_
    );

    const auto dist = static_cast<std::size_t>(kQ_ * geo_.plane) * sizeof(Scalar);
    const auto field = static_cast<std::size_t>(geo_.plane) * sizeof(Scalar);
    for (int s = 0; s < 2; ++s) {
      f_[s] = MakeBuffer(dist, "clCreateBuffer(f)");
      g_[s] = MakeBuffer(dist, "clCreateBuffer(g)");
      counter_[s] = MakeBuffer(sizeof(cl_int), "clCreateBuffer(counter)");
    }
    xi_ = MakeBuffer(kDim_ * field, "clCreateBuffer(xi)");
    rho_ = MakeBuffer(field, "clCreateBuffer(rho)");
    p_ = MakeBuffer(field, "clCreateBuffer(p)");
    tem_ = MakeBuffer(field, "clCreateBuffer(T)");
    u_ = MakeBuffer(kDim_ * field, "clCreateBuffer(u)");

    // A launch drains `counter_[own]` and zeroes the other queue: the initial state is
    // "parity 1", so that the first step (parity 0) finds its queue empty.
    initKernel_ = MakeKernel("initKernel");
    SetArgs(initKernel_, { &f_[0], &g_[0], &xi_, &rho_, &p_, &tem_, &u_, &counter_[1], &counter_[0] });
    // [parity][store]: parity 0 reads set 0 and writes set 1.
    for (int parity = 0; parity < 2; ++parity) {
      for (int store = 0; store < 2; ++store) {
        cl::Kernel& k = stepKernel_[parity][store] = MakeKernel("stepKernel");
        SetArgs(
          k,
          { &f_[parity], &g_[parity], &f_[1 - parity], &g_[1 - parity], &xi_, &rho_, &p_, &tem_, &u_,
            &counter_[parity], &counter_[1 - parity] }
        );
        Check(k.setArg(11, static_cast<cl_int>(store)), "clSetKernelArg", who_);
      }
    }

    // Defined (zero) contents for the first-launch warm-up and for getters called before `Init()`.
    for (const cl::Buffer* buffer : { &f_[0], &f_[1], &g_[0], &g_[1], &xi_, &rho_, &p_, &tem_, &u_, &counter_[0], &counter_[1] })
      Zero(*buffer);
    WarmUp();
  }

  CpuEngine(const CpuEngine&) = delete;

  CpuEngine& operator=(const CpuEngine&) = delete;

  CpuEngine(CpuEngine&&) = delete;

  CpuEngine& operator=(CpuEngine&&) = delete;

  ~CpuEngine() {
    // Nothing may still be reading or writing a buffer when it is released.
    if (queue_())
      queue_.finish();
  }

  /// Initial condition + first collision; restarts the ping-pong at set 0.
  void Init() {
    SetQueue(1, 0);
    Launch(initKernel_, "initKernel");
    Check(queue_.finish(), "clFinish(init)", who_);
    parity_ = 0;
  }

  /// `steps` fused steps, one host synchronisation at the end.
  void Run(const Index steps) {
    if (steps <= 0)
      return;
    for (Index s = 0; s < steps; ++s) {
      const int store = (s + 1 == steps) ? 1 : 0;
      Launch(stepKernel_[parity_][store], "stepKernel");
      parity_ = 1 - parity_;
      // The launches are far cheaper than the steps: a long `Run()` would otherwise keep
      // every one of them queued in the runtime. One drain per window costs nothing.
      if ((s + 1) % kMaxQueued == 0)
        Check(queue_.finish(), "clFinish(window)", who_);
    }
    Check(queue_.finish(), "clFinish(run)", who_);
  }

  void CopyRho(Scalar* dst) const { CopyField(rho_, dst); }

  void CopyP(Scalar* dst) const { CopyField(p_, dst); }

  void CopyT(Scalar* dst) const { CopyField(tem_, dst); }

  /// Velocity in the AoS order `(i, j, d)` of the getter `Tensor`.
  void CopyU(Scalar* dst) const {
    const Scalar* src = Map(u_, kDim_ * geo_.plane);
    host::ExportVector(src, src + geo_.plane, geo_, host::Index{ 0 }, dst);
    Unmap(u_, src);
  }

private:
  /// Steps enqueued back to back before the queue is drained once.
  static constexpr Index kMaxQueued = 4096;

  /// Cells per work-item: two registers of the device's preferred double vector (2 ... 16 lanes).
  static int Lanes(const cl::Device& device) {
    const auto preferred = static_cast<unsigned>(std::max<cl_uint>(1, device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>()));
    return static_cast<int>(std::clamp(2U * std::bit_floor(preferred), 2U, 16U));
  }

  /// `#define`s of `kernels_cpu.cl` that `ProgramDefines()` does not provide.
  std::string GeometryDefines(const host::Params<Scalar>& params) const {
    // A block is interior if the gather window of every direction stays inside the tube.
    const host::Index interiorLo = -params.minOx;
    const host::Index interiorHi = (nx_ - 1) - (lanes_ - 1) - params.maxOx1;
    auto define = [](const char* name, const host::Index value) {
      const std::string text = std::to_string(value);
      return "#define " + std::string(name) + " " + (value < 0 ? "(" + text + ")" : text) + "\n";
    };
    return define("LBM_VW", lanes_) + define("LBM_PITCH", geo_.pitch) + define("LBM_PLANE", geo_.plane)
      + define("LBM_BLOCKS", part_.blocksPerRow) + define("LBM_CHUNK", part_.chunkBlocks)
      + define("LBM_SEGMENTS", part_.Segments()) + define("LBM_INTERIOR_LO", interiorLo)
      + define("LBM_INTERIOR_HI", interiorHi) + define("LBM_ANY_NEAREST", params.anyNearest ? 1 : 0);
  }

  cl::Buffer MakeBuffer(const std::size_t bytes, const char* what) {
    cl_int err = CL_SUCCESS;
    cl::Buffer buffer(context_, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    Check(err, what, who_);
    return buffer;
  }

  cl::Kernel MakeKernel(const char* name) {
    cl_int err = CL_SUCCESS;
    cl::Kernel kernel(program_, name, &err);
    Check(err, "clCreateKernel", who_);
    return kernel;
  }

  void SetArgs(cl::Kernel& kernel, std::initializer_list<const cl::Buffer*> buffers) {
    cl_uint arg = 0;
    for (const cl::Buffer* buffer : buffers)
      Check(kernel.setArg(arg++, *buffer), "clSetKernelArg", who_);
  }

  /// One work-group of one work-item per worker thread; they share the segments (see the kernels).
  void Launch(const cl::Kernel& kernel, const char* what) {
    Check(queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(workers_), cl::NDRange(1)), what, who_);
  }

  void Zero(const cl::Buffer& buffer) {
    std::size_t bytes = 0;
    Check(buffer.getInfo(CL_MEM_SIZE, &bytes), "clGetMemObjectInfo", who_);
    Check(queue_.enqueueFillBuffer(buffer, std::uint8_t{ 0 }, 0, bytes), "clEnqueueFillBuffer", who_);
  }

  /// Sets the number of segments already taken from queue `which`.
  void SetQueue(const int which, const cl_int taken) {
    Check(queue_.enqueueFillBuffer(counter_[which], taken, 0, sizeof(cl_int)), "clEnqueueFillBuffer", who_);
  }

  /// Compiles the work-group functions with a launch of every kernel that does one segment of work.
  void WarmUp() {
    const auto last = static_cast<cl_int>(part_.Segments() - 1);
    SetQueue(1, last);
    Launch(initKernel_, "initKernel");
    SetQueue(0, last);
    Launch(stepKernel_[0][1], "stepKernel");
    SetQueue(0, 0);
    SetQueue(1, 0);
    Check(queue_.finish(), "clFinish(warm-up)", who_);
  }

  const Scalar* Map(const cl::Buffer& buffer, const Index count) const {
    cl_int err = CL_SUCCESS;
    void* ptr = queue_.enqueueMapBuffer(
      buffer, CL_TRUE, CL_MAP_READ, 0, static_cast<std::size_t>(count) * sizeof(Scalar), nullptr, nullptr, &err
    );
    Check(err, "clEnqueueMapBuffer", who_);
    return static_cast<const Scalar*>(ptr);
  }

  void Unmap(const cl::Buffer& buffer, const Scalar* ptr) const {
    Check(queue_.enqueueUnmapMemObject(buffer, const_cast<Scalar*>(ptr)), "clEnqueueUnmapMemObject", who_);
    Check(queue_.finish(), "clFinish(unmap)", who_);
  }

  void CopyField(const cl::Buffer& buffer, Scalar* dst) const {
    const Scalar* src = Map(buffer, geo_.plane);
    host::ExportField(src, geo_, host::Index{ 0 }, dst);
    Unmap(buffer, src);
  }

  const std::string who_;
  const Index nx_;
  const Index ny_;

  host::Geometry geo_;
  host::Partition part_;
  int lanes_ = 0;
  cl_uint workers_ = 1;

  cl::Device device_;
  cl::Context context_;
  cl::CommandQueue queue_;
  cl::Program program_;

  cl::Buffer f_[2];
  cl::Buffer g_[2];
  cl::Buffer counter_[2];
  cl::Buffer xi_;
  cl::Buffer rho_;
  cl::Buffer p_;
  cl::Buffer tem_;
  cl::Buffer u_;

  cl::Kernel initKernel_;
  cl::Kernel stepKernel_[2][2];

  int parity_ = 0;
};
} // namespace lbmini::opencl::detail

#endif // LBMINI_OPENCL_CPU_ENGINE_HPP_
