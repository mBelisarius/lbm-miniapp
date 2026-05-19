/**
 * @file backend_presence_test.cpp
 * @brief Comprehensive backend presence and capability discovery test suite.
 *
 * Verifies that each acceleration backend and execution target:
 *  1. Is discoverable and present in the build/runtime environment when enabled;
 *  2. Initializes hardware and runtime contexts safely without memory corruption;
 *  3. Successfully instantiates and executes solver steps without errors or leaks;
 *  4. Gracefully skips if physical device hardware is unavailable in the runner;
 *  5. Rejects invalid backend/target combinations with descriptive exceptions.
 */

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "MemorySafetyHarness.hpp"
#include "Executor.hpp"
#include "Guards.hpp"
#include "Lbm/ILbmTube.hpp"
#include "Lbm/Plain/LatticeD2Q9.hpp"
#include "Lbm/Plain/LbmTube.hpp"

#ifdef LBMINI_TEST_OPENMP_CPU
#include <omp.h>
#include "Lbm/OpenMp/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenMp/Cpu/LbmTube.hpp"
#endif

#if defined(LBMINI_TEST_OPENMP_GPU) && defined(_OPENMP) && (defined(__NVCOMPILER) || defined(__NVHPC__))
#include <omp.h>
#include "Lbm/OpenMp/Gpu/LatticeD2Q9.hpp"
#include "Lbm/OpenMp/Gpu/LbmTube.hpp"
#endif

#if defined(LBMINI_TEST_OPENACC_CPU) && defined(_OPENACC)
#include <openacc.h>
#include "Lbm/OpenAcc/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenAcc/Cpu/LbmTube.hpp"
#endif

#if defined(LBMINI_TEST_OPENACC_GPU) && defined(_OPENACC)
#include <openacc.h>
#include "Lbm/OpenAcc/Gpu/LatticeD2Q9.hpp"
#include "Lbm/OpenAcc/Gpu/LbmTube.hpp"
#endif

#if defined(LBMINI_TEST_OPENCL_CPU) || defined(LBMINI_TEST_OPENCL_GPU)
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl2.hpp>
#endif

#ifdef LBMINI_TEST_OPENCL_CPU
#include "Lbm/OpenCl/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenCl/Cpu/LbmTube.hpp"
#endif

#ifdef LBMINI_TEST_OPENCL_GPU
#include "Lbm/OpenCl/Gpu/LatticeD2Q9.hpp"
#include "Lbm/OpenCl/Gpu/LbmTube.hpp"
#endif

#ifdef LBMINI_TEST_KOKKOS
#include <Kokkos_Core.hpp>
#include "Lbm/Kokkos/Cpu/LatticeD2Q9.hpp"
#include "Lbm/Kokkos/Cpu/LbmTube.hpp"
#if defined(LBMINI_TEST_DEVICE_BACKENDS) && defined(USE_CUDA)
#include "Lbm/Kokkos/Gpu/LatticeD2Q9.hpp"
#include "Lbm/Kokkos/Gpu/LbmTube.hpp"
#endif
#endif

#ifdef LBMINI_TEST_RAJA
#include "Lbm/Raja/Cpu/LatticeD2Q9.hpp"
#include "Lbm/Raja/Cpu/LbmTube.hpp"
#if defined(LBMINI_TEST_DEVICE_BACKENDS)
#include "Lbm/Raja/Gpu/LatticeD2Q9.hpp"
#include "Lbm/Raja/Gpu/LbmTube.hpp"
#endif
#endif

#if (defined(USE_CUDA) || defined(LBMINI_TEST_CUDA))
#include <cuda_runtime.h>
#if defined(LBMINI_TEST_DEVICE_BACKENDS)
#include "Lbm/Cuda/Gpu/LatticeD2Q9.hpp"
#include "Lbm/Cuda/Gpu/LbmTube.hpp"
#endif
#endif

#ifdef LBMINI_TEST_MPI
#include <mpi.h>
#include "Lbm/Mpi/Cpu/LatticeD2Q9.hpp"
#include "Lbm/Mpi/Cpu/LbmTube.hpp"
#endif

namespace {
using lbmini_test::Index;
using lbmini_test::MeshShape;
using lbmini_test::Scalar;

constexpr MeshShape kPresenceShape{ 32, 4 };

// Helper to verify solver execution and readback sanity
template<typename TubeT>
void VerifySolverExecution(TubeT& tube, const MeshShape& shape, const std::string& backendName) {
  tube.Init();
  tube.Run(2, false);

  const auto rho = tube.Rho();
  const auto p = tube.P();
  const auto tem = tube.T();
  const auto u = tube.U();

  ASSERT_EQ(rho.dimension(0), shape.nx) << backendName << " rho dimension(0)";
  ASSERT_EQ(rho.dimension(1), shape.ny) << backendName << " rho dimension(1)";
  ASSERT_EQ(p.dimension(0), shape.nx) << backendName << " p dimension(0)";
  ASSERT_EQ(p.dimension(1), shape.ny) << backendName << " p dimension(1)";
  ASSERT_EQ(tem.dimension(0), shape.nx) << backendName << " T dimension(0)";
  ASSERT_EQ(tem.dimension(1), shape.ny) << backendName << " T dimension(1)";
  ASSERT_EQ(u.dimension(0), shape.nx) << backendName << " u dimension(0)";
  ASSERT_EQ(u.dimension(1), shape.ny) << backendName << " u dimension(1)";
  ASSERT_EQ(u.dimension(2), Index{ 2 }) << backendName << " u dimension(2)";

  for (Index i = 0; i < shape.nx; ++i) {
    for (Index j = 0; j < shape.ny; ++j) {
      EXPECT_TRUE(lbmini::IsFinite(rho(i, j)) && rho(i, j) > Scalar{ 0 })
        << backendName << " invalid rho at (" << i << ", " << j << "): " << rho(i, j);
      EXPECT_TRUE(lbmini::IsFinite(p(i, j)) && p(i, j) > Scalar{ 0 })
        << backendName << " invalid p at (" << i << ", " << j << "): " << p(i, j);
      EXPECT_TRUE(lbmini::IsFinite(tem(i, j)) && tem(i, j) > Scalar{ 0 })
        << backendName << " invalid T at (" << i << ", " << j << "): " << tem(i, j);
      EXPECT_TRUE(lbmini::IsFinite(u(i, j, 0)))
        << backendName << " non-finite ux at (" << i << ", " << j << "): " << u(i, j, 0);
      EXPECT_TRUE(lbmini::IsFinite(u(i, j, 1)))
        << backendName << " non-finite uy at (" << i << ", " << j << "): " << u(i, j, 1);
    }
  }
}

TEST(BackendPresenceTest, PlainBackendIsPresentAndExecutes) {
  using Tube = lbmini::plain::LbmTube<Scalar, lbmini::plain::LatticeD2Q9<Scalar>>;
  Tube solver(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(kPresenceShape.nx, kPresenceShape.ny),
    lbmini_test::MakeControl(),
    lbmini_test::MakePerformance(lbmini::BackendEnum::Plain, lbmini::TargetEnum::CPU)
  );

  VerifySolverExecution(solver, kPresenceShape, "Plain CPU");
}

TEST(BackendPresenceTest, PlainBackendRejectsGpuTarget) {
  lbmini::Executor<Scalar>::Reset();
  EXPECT_THROW({
    auto* exec = lbmini::Executor<Scalar>::GetExecutor(
      lbmini_test::MakeFluid(),
      lbmini_test::MakeMesh(kPresenceShape.nx, kPresenceShape.ny),
      lbmini_test::MakeControl(),
      lbmini_test::MakePerformance(lbmini::BackendEnum::Plain, lbmini::TargetEnum::GPU)
    );
    (void)exec;
  }, std::runtime_error);
  lbmini::Executor<Scalar>::Reset();
}

#ifdef LBMINI_TEST_OPENMP_CPU
TEST(BackendPresenceTest, OpenMpCpuBackendIsPresentAndExecutes) {
  int numThreads = omp_get_max_threads();
  EXPECT_GT(numThreads, 0);

  using Tube = lbmini::openmp::cpu::LbmTube<Scalar, lbmini::openmp::cpu::LatticeD2Q9<Scalar>>;
  Tube solver(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(kPresenceShape.nx, kPresenceShape.ny),
    lbmini_test::MakeControl(),
    lbmini_test::MakePerformance(lbmini::BackendEnum::OpenMP, lbmini::TargetEnum::CPU)
  );

  VerifySolverExecution(solver, kPresenceShape, "OpenMP CPU");
}
#endif

#if defined(LBMINI_TEST_OPENMP_GPU) && defined(_OPENMP) && (defined(__NVCOMPILER) || defined(__NVHPC__))
TEST(BackendPresenceTest, OpenMpGpuBackendPresence) {
  int numDevices = omp_get_num_devices();
  if (numDevices <= 0) {
    GTEST_SKIP() << "No OpenMP target offloading GPU devices detected.";
  }

  using Tube = lbmini::openmp::gpu::LbmTube<Scalar, lbmini::openmp::gpu::LatticeD2Q9<Scalar>>;
  Tube solver(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(kPresenceShape.nx, kPresenceShape.ny),
    lbmini_test::MakeControl(),
    lbmini_test::MakePerformance(lbmini::BackendEnum::OpenMP, lbmini::TargetEnum::GPU)
  );

  VerifySolverExecution(solver, kPresenceShape, "OpenMP GPU");
}
#endif

#if defined(LBMINI_TEST_OPENACC_CPU) && defined(_OPENACC)
TEST(BackendPresenceTest, OpenAccCpuBackendIsPresentAndExecutes) {
  using Tube = lbmini::openacc::cpu::LbmTube<Scalar, lbmini::openacc::cpu::LatticeD2Q9<Scalar>>;
  Tube solver(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(kPresenceShape.nx, kPresenceShape.ny),
    lbmini_test::MakeControl(),
    lbmini_test::MakePerformance(lbmini::BackendEnum::OpenACC, lbmini::TargetEnum::CPU)
  );

  VerifySolverExecution(solver, kPresenceShape, "OpenACC CPU");
}
#endif

#if defined(LBMINI_TEST_OPENACC_GPU) && defined(_OPENACC)
TEST(BackendPresenceTest, OpenAccGpuBackendPresence) {
  int numDevices = acc_get_num_devices(acc_device_default);
  if (numDevices <= 0) {
    GTEST_SKIP() << "No OpenACC GPU devices detected.";
  }

  using Tube = lbmini::openacc::gpu::LbmTube<Scalar, lbmini::openacc::gpu::LatticeD2Q9<Scalar>>;
  Tube solver(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(kPresenceShape.nx, kPresenceShape.ny),
    lbmini_test::MakeControl(),
    lbmini_test::MakePerformance(lbmini::BackendEnum::OpenACC, lbmini::TargetEnum::GPU)
  );

  VerifySolverExecution(solver, kPresenceShape, "OpenACC GPU");
}
#endif

#ifdef LBMINI_TEST_OPENCL_CPU
TEST(BackendPresenceTest, OpenClCpuBackendPresenceAndExecution) {
  std::vector<cl::Platform> platforms;
  cl_int err = cl::Platform::get(&platforms);
  if (err != CL_SUCCESS || platforms.empty()) {
    GTEST_SKIP() << "No OpenCL platforms found on system.";
  }

  bool hasCpuDevice = false;
  for (auto& p : platforms) {
    std::vector<cl::Device> devices;
    p.getDevices(CL_DEVICE_TYPE_CPU, &devices);
    if (!devices.empty()) {
      hasCpuDevice = true;
      break;
    }
  }

  // No cross-type fallback: `opencl::cpu::LbmTube` now only ever requests a
  // CL_DEVICE_TYPE_CPU device and throws when none exists, so accepting any
  // other device here would turn a legitimate skip into a constructor throw.
  if (!hasCpuDevice) {
    GTEST_SKIP() << "No OpenCL CPU device available.";
  }

  using Tube = lbmini::opencl::cpu::LbmTube<Scalar, lbmini::opencl::cpu::LatticeD2Q9<Scalar>>;
  Tube solver(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(kPresenceShape.nx, kPresenceShape.ny),
    lbmini_test::MakeControl(),
    lbmini_test::MakePerformance(lbmini::BackendEnum::OpenCL, lbmini::TargetEnum::CPU)
  );

  VerifySolverExecution(solver, kPresenceShape, "OpenCL CPU");
}
#endif

#ifdef LBMINI_TEST_OPENCL_GPU
TEST(BackendPresenceTest, OpenClGpuBackendPresenceAndExecution) {
  std::vector<cl::Platform> platforms;
  cl_int err = cl::Platform::get(&platforms);
  if (err != CL_SUCCESS || platforms.empty()) {
    GTEST_SKIP() << "No OpenCL platforms found on system.";
  }

  bool hasGpuDevice = false;
  for (auto& p : platforms) {
    std::vector<cl::Device> devices;
    p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (!devices.empty()) {
      hasGpuDevice = true;
      break;
    }
  }

  if (!hasGpuDevice) {
    GTEST_SKIP() << "No OpenCL GPU device detected.";
  }

  using Tube = lbmini::opencl::gpu::LbmTube<Scalar, lbmini::opencl::gpu::LatticeD2Q9<Scalar>>;
  Tube solver(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(kPresenceShape.nx, kPresenceShape.ny),
    lbmini_test::MakeControl(),
    lbmini_test::MakePerformance(lbmini::BackendEnum::OpenCL, lbmini::TargetEnum::GPU)
  );

  VerifySolverExecution(solver, kPresenceShape, "OpenCL GPU");
}
#endif

#if (defined(USE_CUDA) || defined(LBMINI_TEST_CUDA)) && defined(LBMINI_TEST_DEVICE_BACKENDS)
TEST(BackendPresenceTest, CudaGpuBackendPresenceAndExecution) {
  int deviceCount = 0;
  cudaError_t cudaErr = cudaGetDeviceCount(&deviceCount);
  if (cudaErr != cudaSuccess || deviceCount <= 0) {
    GTEST_SKIP() << "No CUDA device detected or CUDA driver not initialized.";
  }

  using Tube = lbmini::cuda::gpu::LbmTube<Scalar, lbmini::cuda::gpu::LatticeD2Q9<Scalar>>;
  Tube solver(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(kPresenceShape.nx, kPresenceShape.ny),
    lbmini_test::MakeControl(),
    lbmini_test::MakePerformance(lbmini::BackendEnum::CUDA, lbmini::TargetEnum::GPU)
  );

  VerifySolverExecution(solver, kPresenceShape, "CUDA GPU");
}
#endif

#ifdef LBMINI_TEST_KOKKOS
TEST(BackendPresenceTest, KokkosCpuBackendIsPresentAndExecutes) {
  // A Kokkos built with the CUDA backend cannot initialise when no device
  // answers, and every Kokkos solver needs the runtime up: this is a missing
  // device, reported as a skip, not a failure of the CPU backend itself.
  if (!lbmini_test::KokkosEnvironment::IsUsable()) {
    GTEST_SKIP() << "Kokkos runtime unavailable (see the Kokkos::initialize() report above).";
  }

  using Tube = lbmini::kokkos::cpu::LbmTube<Scalar, lbmini::kokkos::cpu::LatticeD2Q9<Scalar>>;
  Tube solver(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(kPresenceShape.nx, kPresenceShape.ny),
    lbmini_test::MakeControl(),
    lbmini_test::MakePerformance(lbmini::BackendEnum::Kokkos, lbmini::TargetEnum::CPU)
  );

  VerifySolverExecution(solver, kPresenceShape, "Kokkos CPU");
}

#if defined(LBMINI_TEST_DEVICE_BACKENDS) && defined(USE_CUDA)
TEST(BackendPresenceTest, KokkosGpuBackendPresenceAndExecution) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
    GTEST_SKIP() << "No CUDA device detected for Kokkos GPU.";
  }

  using Tube = lbmini::kokkos::gpu::LbmTube<Scalar, lbmini::kokkos::gpu::LatticeD2Q9<Scalar>>;
  Tube solver(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(kPresenceShape.nx, kPresenceShape.ny),
    lbmini_test::MakeControl(),
    lbmini_test::MakePerformance(lbmini::BackendEnum::Kokkos, lbmini::TargetEnum::GPU)
  );

  VerifySolverExecution(solver, kPresenceShape, "Kokkos GPU");
}
#endif
#endif

#ifdef LBMINI_TEST_RAJA
TEST(BackendPresenceTest, RajaCpuBackendIsPresentAndExecutes) {
  using Tube = lbmini::raja::cpu::LbmTube<Scalar, lbmini::raja::cpu::LatticeD2Q9<Scalar>>;
  Tube solver(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(kPresenceShape.nx, kPresenceShape.ny),
    lbmini_test::MakeControl(),
    lbmini_test::MakePerformance(lbmini::BackendEnum::RAJA, lbmini::TargetEnum::CPU)
  );

  VerifySolverExecution(solver, kPresenceShape, "RAJA CPU");
}

#if defined(LBMINI_TEST_DEVICE_BACKENDS)
TEST(BackendPresenceTest, RajaGpuBackendPresenceAndExecution) {
  #ifdef USE_CUDA
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
    GTEST_SKIP() << "No CUDA device detected for RAJA GPU.";
  }
  #endif

  using Tube = lbmini::raja::gpu::LbmTube<Scalar, lbmini::raja::gpu::LatticeD2Q9<Scalar>>;
  Tube solver(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(kPresenceShape.nx, kPresenceShape.ny),
    lbmini_test::MakeControl(),
    lbmini_test::MakePerformance(lbmini::BackendEnum::RAJA, lbmini::TargetEnum::GPU)
  );

  VerifySolverExecution(solver, kPresenceShape, "RAJA GPU");
}
#endif
#endif

#ifdef LBMINI_TEST_MPI
TEST(BackendPresenceTest, MpiBackendRejectsGpuTarget) {
  EXPECT_THROW({
    lbmini::Executor<Scalar>::Reset();
    auto* exec = lbmini::Executor<Scalar>::GetExecutor(
      lbmini_test::MakeFluid(),
      lbmini_test::MakeMesh(kPresenceShape.nx, kPresenceShape.ny),
      lbmini_test::MakeControl(),
      lbmini_test::MakePerformance(lbmini::BackendEnum::MPI, lbmini::TargetEnum::GPU)
    );
    (void)exec;
  }, std::runtime_error);
  lbmini::Executor<Scalar>::Reset();
}
#endif

TEST(BackendPresenceTest, ExecutorFactoryLifecycleAndIsolation) {
  lbmini::Executor<Scalar>::Reset();

  auto* exec = lbmini::Executor<Scalar>::GetExecutor(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(kPresenceShape.nx, kPresenceShape.ny),
    lbmini_test::MakeControl(),
    lbmini_test::MakePerformance(lbmini::BackendEnum::Plain, lbmini::TargetEnum::CPU)
  );
  ASSERT_NE(exec, nullptr);
  ASSERT_NE(exec->GetLbmTube(), nullptr);

  auto* tube = exec->GetLbmTube();
  tube->Init();
  tube->Run(1, false);

  const auto rho = tube->Rho();
  EXPECT_EQ(rho.dimension(0), kPresenceShape.nx);
  EXPECT_EQ(rho.dimension(1), kPresenceShape.ny);

  // Reset clears singleton cleanly
  lbmini::Executor<Scalar>::Reset();
}

} // namespace
