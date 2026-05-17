#ifndef LBMINI_OPENCL_GPU_LBMTUBE_HPP_
#define LBMINI_OPENCL_GPU_LBMTUBE_HPP_

#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <iostream>
#include <fstream>
#include <CL/cl2.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Data.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/OpenCl/Gpu/LatticeD2Q9.hpp"


#ifndef CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR
#define CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR (1 << 0)
#endif

typedef void* cl_command_buffer_khr_ptr;

typedef cl_command_buffer_khr_ptr (CL_API_CALL *pfn_clCreateCommandBufferKHR)(cl_uint, const cl_command_queue*, const cl_properties*, cl_int*);

typedef cl_int (CL_API_CALL *pfn_clCommandNDRangeKernelKHR)(
  cl_command_buffer_khr_ptr,
  cl_command_queue,
  const cl_properties*,
  cl_kernel,
  cl_uint,
  const size_t*,
  const size_t*,
  const size_t*,
  cl_uint,
  const void*,
  void*,
  void*
);

typedef cl_int (CL_API_CALL *pfn_clFinalizeCommandBufferKHR)(cl_command_buffer_khr_ptr);

typedef cl_int (CL_API_CALL *pfn_clEnqueueCommandBufferKHR)(cl_uint, cl_command_queue*, cl_command_buffer_khr_ptr, cl_uint, const cl_event*, cl_event*);

typedef cl_int (CL_API_CALL *pfn_clReleaseCommandBufferKHR)(cl_command_buffer_khr_ptr);

#include "Lbm/CountCores.hpp"

namespace lbmini::opencl::gpu {

// POCL (Portable Computing Language) uses a library constructor to read environment
// variables like POCL_MAX_PTHREAD_COUNT upon library load.
// Standard setenv() calls in main() execute too late. This constructor runs beforehand.
#ifdef __linux__
__attribute__((constructor))
static void InitPoclEnvironment() {
  std::string perfCoresStr = std::to_string(lbmini::CountPerformanceCores());
  setenv("POCL_MAX_PTHREAD_COUNT", perfCoresStr.c_str(), 0);
  // Explicitly restrict process affinity to P-cores instead of relying on POCL_AFFINITY
  lbmini::SetProcessToPerformanceCores();
}
#endif

/**
 * @brief OpenCL variant of the compressible LBM tube solver for GPUs.
 *
 * Numerically identical to `lbmini::plain::LbmTube`. GPU/CPU-specific choices
 * are handled via custom OpenCL kernels generated from the CUDA implementation.
 *
 * ### OpenCL design decisions
 *
 * **Runtime Compilation (JIT).** We do not use C++ string inline compilation
 * to avoid hard-to-read code. Instead, kernels are stored in a separate `.cl`
 * file and read at runtime, or pre-compiled into SPIR-V format if required by
 * the target architecture before execution, keeping the C++ code clean.
 *
 * **Memory Layout.** The device buffers utilize a Structure-of-Arrays (SoA) layout
 * (`pF[idc * N + cell]`) for optimal coalesced memory access across OpenCL work-items,
 * mapping efficiently onto both SIMD lanes on the CPU and warp/wavefronts on GPUs.
 * The field getters transpose this SoA memory back into the Array-of-Structures (AoS)
 * format expected by Eigen `Tensor`s.
 *
 * **Branchless Warp Execution.** The streaming and macroscopic steps heavily use
 * branchless IDW interpolation and fixed loops to avoid work-group divergence,
 * identical to the CUDA strategy.
 *
 * ### Implementation difficulties
 *
 * **SoA vs AoS transpose bugs.** Initial implementation caused velocity fields (`ux`)
 * to be shifted because of a mismatch in array indexing when copying from the flat
 * OpenCL buffer back to the `Eigen::Tensor`.
 *
 * **Buffer allocations.** Subtleties in allocating appropriately sized memory for
 * intermediate arrays like `lastGxDev_` caused out-of-bounds segfaults until aligned
 * with the correct `N_ * (kDim_ + 1)` size.
 *
 * **Thread Utilization and Dispatch Overhead.** The OpenCL CPU/GPU backends can suffer from
 * host dispatch overhead when queueing multiple kernels per time step, leading to poor thread
 * utilization on the CPU and latency on the GPU. Unlike OpenMP's persistent thread teams,
 * the OpenCL runtime experiences significant overhead waiting for the command queue.
 * Migrating the time loop into the kernel or using OpenCL Command-Buffers/Event-Graphs
 * is required for optimal performance.
 */
template<typename Scalar, typename LatticeType>
class LbmTube : public ILbmTube<Scalar, LatticeType> {
public:
  using Index = Eigen::Index;
  template<typename Type, Index NumIndices>
  using Tensor = Eigen::Tensor<Type, NumIndices, Eigen::RowMajor>;

  static constexpr Index kDim_ = LatticeType::Dim();
  static constexpr Index kQ_ = LatticeType::Speeds();

  LbmTube(const FluidData<Scalar>& fluid, const MeshData<Scalar, kDim_>& mesh, const ControlData<Scalar>& control, const PerformanceData& performance)
    : kFluid_(fluid), kMesh_(mesh), kControl_(control), kPerformance_(performance) {
    nx_ = kMesh_.nx();
    ny_ = kMesh_.ny();
    N_ = nx_ * ny_;
    uSize_ = kDim_ * N_;
    distSize_ = kQ_ * N_;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl_device_type deviceType = (kPerformance_.target == lbmini::TargetEnum::GPU) ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

    cl::Platform plat;
    std::vector<cl::Device> devices;
    for (auto& p : platforms) {
      p.getDevices(deviceType, &devices);
      if (!devices.empty()) {
        plat = p;
        device_ = devices.front();
        break;
      }
    }
    if (devices.empty()) {
      platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
      device_ = devices.front();
    }

    context_ = cl::Context(device_);
    queue_ = cl::CommandQueue(context_, device_);

    std::ifstream kFile("lbmini/Lbm/OpenCl/Gpu/kernels.cl");
    if (!kFile.is_open()) {
      std::cerr << "Failed to open kernels.cl" << std::endl;
      exit(1);
    }
    std::string kernelSource((std::istreambuf_iterator<char>(kFile)), std::istreambuf_iterator<char>());

    cl::Program::Sources sources;
    sources.push_back({ kernelSource.c_str(), kernelSource.length() });
    program_ = cl::Program(context_, sources);
    if (program_.build({ device_ }) != CL_SUCCESS) {
      std::cerr << "OpenCL Build Error: " << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_) << std::endl;
      exit(1);
    }

    macroscopicKernel_ = cl::Kernel(program_, "computeMacroscopicKernel");
    seedKernel_ = cl::Kernel(program_, "seedEquilibriaKernel");
    collisionKernel_ = cl::Kernel(program_, "collisionAndEquilibriaKernel");
    streamKernel_ = cl::Kernel(program_, "streamAndMacroscopicKernel");

    rhoDev_ = cl::Buffer(context_, CL_MEM_READ_WRITE, N_ * sizeof(Scalar));
    pDev_ = cl::Buffer(context_, CL_MEM_READ_WRITE, N_ * sizeof(Scalar));
    temDev_ = cl::Buffer(context_, CL_MEM_READ_WRITE, N_ * sizeof(Scalar));
    uDev_ = cl::Buffer(context_, CL_MEM_READ_WRITE, uSize_ * sizeof(Scalar));

    fCur_ = cl::Buffer(context_, CL_MEM_READ_WRITE, distSize_ * sizeof(Scalar));
    fAlt_ = cl::Buffer(context_, CL_MEM_READ_WRITE, distSize_ * sizeof(Scalar));
    gCur_ = cl::Buffer(context_, CL_MEM_READ_WRITE, distSize_ * sizeof(Scalar));
    gAlt_ = cl::Buffer(context_, CL_MEM_READ_WRITE, distSize_ * sizeof(Scalar));
    lastGxDev_ = cl::Buffer(context_, CL_MEM_READ_WRITE, N_ * (kDim_ + 1) * sizeof(Scalar));
  }

  ~LbmTube() override {
    if (use_cb_ && pfn_release_) {
      if (cbEven_)
        pfn_release_(cbEven_);
      if (cbOdd_)
        pfn_release_(cbOdd_);
    }
  }

  Tensor<Scalar, kDim_> P() const override {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    queue_.enqueueReadBuffer(pDev_, CL_TRUE, 0, N_ * sizeof(Scalar), out.data());
    return out;
  }

  Tensor<Scalar, kDim_> Rho() const override {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    queue_.enqueueReadBuffer(rhoDev_, CL_TRUE, 0, N_ * sizeof(Scalar), out.data());
    return out;
  }

  Tensor<Scalar, kDim_> T() const override {
    Tensor<Scalar, kDim_> out(nx_, ny_);
    queue_.enqueueReadBuffer(temDev_, CL_TRUE, 0, N_ * sizeof(Scalar), out.data());
    return out;
  }

  Tensor<Scalar, kDim_ + 1> U() const override {
    Tensor<Scalar, kDim_ + 1> out(nx_, ny_, kDim_);
    std::vector<Scalar> uHost(uSize_, 0.0);
    queue_.enqueueReadBuffer(uDev_, CL_TRUE, 0, uSize_ * sizeof(Scalar), uHost.data());
    for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        int cell = i * ny_ + j;
        for (int d = 0; d < kDim_; ++d) {
          out(i, j, d) = uHost[d * N_ + cell];
        }
      }
    }
    return out;
  }

  void Init() override {
    const Scalar Cs2 = 1.0 / 3.0;
    const Scalar kRgas_ = kFluid_.constant;
    const Scalar kCv_ = 1.0 / (kFluid_.gamma - 1.0);

    std::vector<Scalar> rho(N_, kFluid_.densityL);
    std::vector<Scalar> tem(N_, Cs2 * kFluid_.pressureL / (kRgas_ * kFluid_.densityL));
    std::vector<Scalar> u(uSize_, 0.0);
    std::vector<Scalar> lastGx(N_ * (kDim_ + 1), 0.0);

    for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        int idx = i * ny_ + j;
        if (i < nx_ / 2) {
          rho[idx] = kFluid_.densityL;
          tem[idx] = Cs2 * kFluid_.pressureL / (kRgas_ * kFluid_.densityL);
        } else {
          rho[idx] = kFluid_.densityR;
          tem[idx] = Cs2 * kFluid_.pressureR / (kRgas_ * kFluid_.densityR);
        }
      }
    }

    queue_.enqueueWriteBuffer(rhoDev_, CL_TRUE, 0, N_ * sizeof(Scalar), rho.data());
    queue_.enqueueWriteBuffer(temDev_, CL_TRUE, 0, N_ * sizeof(Scalar), tem.data());
    queue_.enqueueWriteBuffer(uDev_, CL_TRUE, 0, uSize_ * sizeof(Scalar), u.data());
    queue_.enqueueWriteBuffer(lastGxDev_, CL_TRUE, 0, N_ * (kDim_ + 1) * sizeof(Scalar), lastGx.data());


    seedEquilibria();
    computeMacroscopic();

    cl_platform_id plat_id = plat();
    pfn_create_ = (pfn_clCreateCommandBufferKHR)clGetExtensionFunctionAddressForPlatform(plat_id, "clCreateCommandBufferKHR");
    pfn_ndrange_ = (pfn_clCommandNDRangeKernelKHR)clGetExtensionFunctionAddressForPlatform(plat_id, "clCommandNDRangeKernelKHR");
    pfn_finalize_ = (pfn_clFinalizeCommandBufferKHR)clGetExtensionFunctionAddressForPlatform(plat_id, "clFinalizeCommandBufferKHR");
    pfn_enqueue_ = (pfn_clEnqueueCommandBufferKHR)clGetExtensionFunctionAddressForPlatform(plat_id, "clEnqueueCommandBufferKHR");
    pfn_release_ = (pfn_clReleaseCommandBufferKHR)clGetExtensionFunctionAddressForPlatform(plat_id, "clReleaseCommandBufferKHR");

    if (pfn_create_ && pfn_ndrange_ && pfn_finalize_ && pfn_enqueue_ && pfn_release_) {
      use_cb_ = true;

      cl_command_queue q = queue_();
      cl_properties props[] = { 0x1293, CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR, 0 }; // 0x1293 is CL_COMMAND_BUFFER_FLAGS_KHR

      cl_int err;
      cbEven_ = pfn_create_(1, &q, props, &err);
      if (err != CL_SUCCESS)
        std::cerr << "CB Create Even err: " << err << std::endl;

      cbOdd_ = pfn_create_(1, &q, props, &err);
      if (err != CL_SUCCESS)
        std::cerr << "CB Create Odd err: " << err << std::endl;

      size_t global_work_size[1];
      if constexpr (std::is_same_v<LatticeType, lbmini::opencl::gpu::LatticeD2Q9<Scalar>>) {
        global_work_size[0] = N_;
      } else {
        global_work_size[0] = (N_ + 127) / 128 * 128;
      }
      size_t local_work_size[1] = { 128 };
      const size_t* lws = nullptr;
      if constexpr (!std::is_same_v<LatticeType, lbmini::opencl::gpu::LatticeD2Q9<Scalar>>) {
        lws = local_work_size;
      }

      // Even step recording
      collisionAndEquilibria(fCur_, gCur_, cbEven_, q, global_work_size, lws);
      streamAndMacroscopic(fCur_, gCur_, fAlt_, gAlt_, cbEven_, q, global_work_size, lws);
      err = pfn_finalize_(cbEven_);
      if (err != CL_SUCCESS)
        std::cerr << "CB Finalize Even err: " << err << std::endl;

      // Odd step recording
      collisionAndEquilibria(fAlt_, gAlt_, cbOdd_, q, global_work_size, lws);
      streamAndMacroscopic(fAlt_, gAlt_, fCur_, gCur_, cbOdd_, q, global_work_size, lws);
      err = pfn_finalize_(cbOdd_);
      if (err != CL_SUCCESS)
        std::cerr << "CB Finalize Odd err: " << err << std::endl;
    }
  }

  cl_platform_id plat() const {
    cl_device_id did = device_();
    cl_platform_id pid;
    clGetDeviceInfo(did, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &pid, nullptr);
    return pid;
  }


  void Step(bool save) override {
    if (use_cb_) {
      cl_command_buffer_khr_ptr cb = isEvenStep_ ? cbEven_ : cbOdd_;
      cl_int err = pfn_enqueue_(0, nullptr, cb, 0, nullptr, nullptr);
      if (err != CL_SUCCESS)
        std::cerr << "CB Enqueue err: " << err << std::endl;
      isEvenStep_ = !isEvenStep_;
    } else {
      if (isEvenStep_) {
        collisionAndEquilibria(fCur_, gCur_, nullptr, nullptr, nullptr, nullptr);
        streamAndMacroscopic(fCur_, gCur_, fAlt_, gAlt_, nullptr, nullptr, nullptr, nullptr);
      } else {
        collisionAndEquilibria(fAlt_, gAlt_, nullptr, nullptr, nullptr, nullptr);
        streamAndMacroscopic(fAlt_, gAlt_, fCur_, gCur_, nullptr, nullptr, nullptr, nullptr);
      }
      isEvenStep_ = !isEvenStep_;
    }
  }

  void Run(Index steps, bool save) override {
    for (Index s = 0; s < steps; ++s) {
      Step(save);
    }
  }

protected:
  void computeMacroscopic() {
    int arg = 0;
    macroscopicKernel_.setArg(arg++, fCur_);
    macroscopicKernel_.setArg(arg++, gCur_);
    macroscopicKernel_.setArg(arg++, rhoDev_);
    macroscopicKernel_.setArg(arg++, pDev_);
    macroscopicKernel_.setArg(arg++, temDev_);
    macroscopicKernel_.setArg(arg++, uDev_);
    macroscopicKernel_.setArg(arg++, (double)kControl_.Ux);
    macroscopicKernel_.setArg(arg++, (double)kControl_.Uy);
    macroscopicKernel_.setArg(arg++, (double)(1.0 / kFluid_.specificHeatCv));
    macroscopicKernel_.setArg(arg++, (double)kFluid_.constant);
    macroscopicKernel_.setArg(arg++, (int)nx_);
    macroscopicKernel_.setArg(arg++, (int)ny_);
    macroscopicKernel_.setArg(arg++, (int)N_);

    cl_int err = queue_.enqueueNDRangeKernel(macroscopicKernel_, cl::NullRange, cl::NDRange((N_ + 127) / 128 * 128), cl::NDRange(128));
    if (err != CL_SUCCESS)
      std::cerr << "Macro err: " << err << std::endl;
  }

  void seedEquilibria() {
    int arg = 0;
    seedKernel_.setArg(arg++, fCur_);
    seedKernel_.setArg(arg++, gCur_);
    seedKernel_.setArg(arg++, rhoDev_);
    seedKernel_.setArg(arg++, temDev_);
    seedKernel_.setArg(arg++, uDev_);
    seedKernel_.setArg(arg++, (double)kControl_.Ux);
    seedKernel_.setArg(arg++, (double)kControl_.Uy);
    seedKernel_.setArg(arg++, (double)1e-12);
    seedKernel_.setArg(arg++, (int)nx_);
    seedKernel_.setArg(arg++, (int)ny_);
    seedKernel_.setArg(arg++, (int)N_);
    seedKernel_.setArg(arg++, (double)kFluid_.specificHeatCv);

    cl_int err = queue_.enqueueNDRangeKernel(seedKernel_, cl::NullRange, cl::NDRange((N_ + 127) / 128 * 128), cl::NDRange(128));
    if (err != CL_SUCCESS)
      std::cerr << "Seed err: " << err << std::endl;
  }

  void collisionAndEquilibria(const cl::Buffer& fIn, const cl::Buffer& gIn, cl_command_buffer_khr_ptr cb, cl_command_queue q, const size_t* gws, const size_t* lws) {
    int arg = 0;
    collisionKernel_.setArg(arg++, fIn);
    collisionKernel_.setArg(arg++, gIn);
    collisionKernel_.setArg(arg++, lastGxDev_);
    collisionKernel_.setArg(arg++, rhoDev_);
    collisionKernel_.setArg(arg++, temDev_);
    collisionKernel_.setArg(arg++, uDev_);
    collisionKernel_.setArg(arg++, (double)kControl_.Ux);
    collisionKernel_.setArg(arg++, (double)kControl_.Uy);
    collisionKernel_.setArg(arg++, (double)kFluid_.viscosity);
    collisionKernel_.setArg(arg++, (double)kFluid_.conductivity);
    collisionKernel_.setArg(arg++, (double)kFluid_.specificHeatCp);
    collisionKernel_.setArg(arg++, (double)kFluid_.specificHeatCv);
    collisionKernel_.setArg(arg++, (double)1e-12);
    collisionKernel_.setArg(arg++, (double)50.0);
    collisionKernel_.setArg(arg++, (int)nx_);
    collisionKernel_.setArg(arg++, (int)ny_);
    collisionKernel_.setArg(arg++, (int)N_);


    if (cb) {
      cl_int err = pfn_ndrange_(cb, q, nullptr, collisionKernel_(), 1, nullptr, gws, lws, 0, nullptr, nullptr, nullptr);
      if (err != CL_SUCCESS)
        std::cerr << "Coll CB err: " << err << std::endl;
    } else {
      cl_int err = queue_.enqueueNDRangeKernel(collisionKernel_, cl::NullRange, cl::NDRange((N_ + 127) / 128 * 128), cl::NDRange(128));
      if (err != CL_SUCCESS)
        std::cerr << "Coll err: " << err << std::endl;
    }
  }

  void streamAndMacroscopic(
    const cl::Buffer& fIn,
    const cl::Buffer& gIn,
    const cl::Buffer& fOut,
    const cl::Buffer& gOut,
    cl_command_buffer_khr_ptr cb,
    cl_command_queue q,
    const size_t* gws,
    const size_t* lws
  ) {
    int arg = 0;
    streamKernel_.setArg(arg++, fIn);
    streamKernel_.setArg(arg++, gIn);
    streamKernel_.setArg(arg++, fOut);
    streamKernel_.setArg(arg++, gOut);
    streamKernel_.setArg(arg++, rhoDev_);
    streamKernel_.setArg(arg++, pDev_);
    streamKernel_.setArg(arg++, temDev_);
    streamKernel_.setArg(arg++, uDev_);
    streamKernel_.setArg(arg++, (double)kControl_.Ux);
    streamKernel_.setArg(arg++, (double)kControl_.Uy);
    streamKernel_.setArg(arg++, (double)(1.0 / kFluid_.specificHeatCv));
    streamKernel_.setArg(arg++, (double)kFluid_.constant);
    streamKernel_.setArg(arg++, (double)1e-12);
    streamKernel_.setArg(arg++, (double)kControl_.idw);
    streamKernel_.setArg(arg++, (int)nx_);
    streamKernel_.setArg(arg++, (int)ny_);
    streamKernel_.setArg(arg++, (int)N_);


    if (cb) {
      cl_int err = pfn_ndrange_(cb, q, nullptr, streamKernel_(), 1, nullptr, gws, lws, 0, nullptr, nullptr, nullptr);
      if (err != CL_SUCCESS)
        std::cerr << "Stream CB err: " << err << std::endl;
    } else {
      cl_int err = queue_.enqueueNDRangeKernel(streamKernel_, cl::NullRange, cl::NDRange((N_ + 127) / 128 * 128), cl::NDRange(128));
      if (err != CL_SUCCESS)
        std::cerr << "Stream err: " << err << std::endl;
    }
  }

private:
  const FluidData<Scalar> kFluid_;
  const MeshData<Scalar, kDim_> kMesh_;
  const ControlData<Scalar> kControl_;
  const PerformanceData kPerformance_;

  Index nx_;
  Index ny_;
  Index N_;
  Index uSize_;
  Index distSize_;

  cl::Context context_;
  cl::CommandQueue queue_;
  cl::Device device_;
  cl::Program program_;

  cl::Kernel macroscopicKernel_;
  cl::Kernel seedKernel_;
  cl::Kernel collisionKernel_;
  cl::Kernel streamKernel_;

  cl::Buffer rhoDev_;
  cl::Buffer pDev_;
  cl::Buffer temDev_;
  cl::Buffer uDev_;
  cl::Buffer fCur_;
  cl::Buffer fAlt_;
  cl::Buffer gCur_;
  cl::Buffer gAlt_;
  cl::Buffer lastGxDev_;

  cl_command_buffer_khr_ptr cbEven_ = nullptr;
  cl_command_buffer_khr_ptr cbOdd_ = nullptr;
  pfn_clCreateCommandBufferKHR pfn_create_ = nullptr;
  pfn_clCommandNDRangeKernelKHR pfn_ndrange_ = nullptr;
  pfn_clFinalizeCommandBufferKHR pfn_finalize_ = nullptr;
  pfn_clEnqueueCommandBufferKHR pfn_enqueue_ = nullptr;
  pfn_clReleaseCommandBufferKHR pfn_release_ = nullptr;
  bool use_cb_ = false;
  bool isEvenStep_ = true;
};
} // namespace lbmini::opencl::gpu

#endif
