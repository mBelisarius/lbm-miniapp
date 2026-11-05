#ifndef LBMINI_TEST_BACKENDTRAITS_HPP_
#define LBMINI_TEST_BACKENDTRAITS_HPP_

/**
 * @file BackendTraits.hpp
 * @brief The list of solver backends compiled into the test binary.
 *
 * One traits type per backend/target pair (`Tube`, `Performance()`, `Name()`,
 * `IsAvailable()`), and `BackendTypes`, the gtest type list every
 * backend-generic suite is instantiated over. A backend joins every such suite
 * by being added here once.
 */

#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "MemorySafetyHarness.hpp"
#include "Lbm/Plain/LatticeD2Q9.hpp"
#include "Lbm/Plain/LbmTube.hpp"

#ifdef LBMINI_TEST_OPENMP_CPU
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

#if (defined(USE_CUDA) || defined(LBMINI_TEST_CUDA)) && defined(LBMINI_TEST_DEVICE_BACKENDS)
#include <cuda_runtime.h>
#include "Lbm/Cuda/Gpu/LatticeD2Q9.hpp"
#include "Lbm/Cuda/Gpu/LbmTube.hpp"
#endif

#ifdef LBMINI_TEST_ALPAKA
#include "Lbm/Alpaka/Cpu/LatticeD2Q9.hpp"
#include "Lbm/Alpaka/Cpu/LbmTube.hpp"
#if defined(LBMINI_TEST_DEVICE_BACKENDS) && defined(USE_ALPAKA_GPU)
#include "Lbm/Alpaka/Gpu/LatticeD2Q9.hpp"
#include "Lbm/Alpaka/Gpu/LbmTube.hpp"
#endif
#endif

namespace lbmini_test {
/**
 * @brief Description of one backend under test.
 *
 * A backend joins the suite by adding its traits type to `BackendTypes`; no
 * test body needs to know which backend it is driving.
 */
struct PlainBackend {
  using Lattice = lbmini::plain::LatticeD2Q9<Scalar>;
  using Tube = lbmini::plain::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::Plain, lbmini::TargetEnum::CPU);
  }
  static const char* Name() { return "plain"; }
  static bool IsAvailable() { return true; }
};

#ifdef LBMINI_TEST_OPENMP_CPU
struct OpenMpCpuBackend {
  using Lattice = lbmini::openmp::cpu::LatticeD2Q9<Scalar>;
  using Tube = lbmini::openmp::cpu::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::OpenMP, lbmini::TargetEnum::CPU);
  }
  static const char* Name() { return "openmp_cpu"; }
  static bool IsAvailable() { return true; }
};
#endif

#if defined(LBMINI_TEST_OPENMP_GPU) && defined(_OPENMP) && (defined(__NVCOMPILER) || defined(__NVHPC__)) && defined(LBMINI_TEST_DEVICE_BACKENDS)
struct OpenMpGpuBackend {
  using Lattice = lbmini::openmp::gpu::LatticeD2Q9<Scalar>;
  using Tube = lbmini::openmp::gpu::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::OpenMP, lbmini::TargetEnum::GPU);
  }
  static const char* Name() { return "openmp_gpu"; }
  static bool IsAvailable() {
    return omp_get_num_devices() > 0;
  }
};
#endif

#if defined(LBMINI_TEST_OPENACC_CPU) && defined(_OPENACC)
struct OpenAccCpuBackend {
  using Lattice = lbmini::openacc::cpu::LatticeD2Q9<Scalar>;
  using Tube = lbmini::openacc::cpu::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::OpenACC, lbmini::TargetEnum::CPU);
  }
  static const char* Name() { return "openacc_cpu"; }
  static bool IsAvailable() { return true; }
};
#endif

#if defined(LBMINI_TEST_OPENACC_GPU) && defined(_OPENACC) && defined(LBMINI_TEST_DEVICE_BACKENDS)
struct OpenAccGpuBackend {
  using Lattice = lbmini::openacc::gpu::LatticeD2Q9<Scalar>;
  using Tube = lbmini::openacc::gpu::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::OpenACC, lbmini::TargetEnum::GPU);
  }
  static const char* Name() { return "openacc_gpu"; }
  static bool IsAvailable() {
    return acc_get_num_devices(acc_device_default) > 0;
  }
};
#endif

#ifdef LBMINI_TEST_OPENCL_CPU
struct OpenClCpuBackend {
  using Lattice = lbmini::opencl::cpu::LatticeD2Q9<Scalar>;
  using Tube = lbmini::opencl::cpu::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::OpenCL, lbmini::TargetEnum::CPU);
  }
  static const char* Name() { return "opencl_cpu"; }
  static bool IsAvailable() {
    std::vector<cl::Platform> platforms;
    cl_int err = cl::Platform::get(&platforms);
    if (err != CL_SUCCESS || platforms.empty()) return false;
    // Only a real CL_DEVICE_TYPE_CPU device counts: `opencl::cpu::LbmTube` no
    // longer falls back to any other device type and throws instead, so
    // reporting "available" here would abort the suite rather than skip it.
    for (auto& p : platforms) {
      std::vector<cl::Device> devices;
      p.getDevices(CL_DEVICE_TYPE_CPU, &devices);
      if (!devices.empty()) return true;
    }
    return false;
  }
};
#endif

#if defined(LBMINI_TEST_OPENCL_GPU) && defined(LBMINI_TEST_DEVICE_BACKENDS)
struct OpenClGpuBackend {
  using Lattice = lbmini::opencl::gpu::LatticeD2Q9<Scalar>;
  using Tube = lbmini::opencl::gpu::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::OpenCL, lbmini::TargetEnum::GPU);
  }
  static const char* Name() { return "opencl_gpu"; }
  static bool IsAvailable() {
    std::vector<cl::Platform> platforms;
    cl_int err = cl::Platform::get(&platforms);
    if (err != CL_SUCCESS || platforms.empty()) return false;
    for (auto& p : platforms) {
      std::vector<cl::Device> devices;
      p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
      if (!devices.empty()) return true;
    }
    return false;
  }
};
#endif

#ifdef LBMINI_TEST_KOKKOS
struct KokkosCpuBackend {
  using Lattice = lbmini::kokkos::cpu::LatticeD2Q9<Scalar>;
  using Tube = lbmini::kokkos::cpu::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::Kokkos, lbmini::TargetEnum::CPU);
  }
  static const char* Name() { return "kokkos_cpu"; }
  // Not `true`: a Kokkos built with the CUDA backend cannot initialise at all
  // when no device answers, so "Kokkos is compiled in" does not imply "a
  // Kokkos solver can be constructed". See lbmini_test::KokkosEnvironment.
  static bool IsAvailable() { return lbmini_test::KokkosEnvironment::IsUsable(); }
};

#if defined(LBMINI_TEST_DEVICE_BACKENDS) && defined(USE_CUDA)
struct KokkosGpuBackend {
  using Lattice = lbmini::kokkos::gpu::LatticeD2Q9<Scalar>;
  using Tube = lbmini::kokkos::gpu::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::Kokkos, lbmini::TargetEnum::GPU);
  }
  static const char* Name() { return "kokkos_gpu"; }
  static bool IsAvailable() {
    if (!lbmini_test::KokkosEnvironment::IsUsable())
      return false;
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess && count > 0);
  }
};
#endif
#endif

#ifdef LBMINI_TEST_RAJA
struct RajaCpuBackend {
  using Lattice = lbmini::raja::cpu::LatticeD2Q9<Scalar>;
  using Tube = lbmini::raja::cpu::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::RAJA, lbmini::TargetEnum::CPU);
  }
  static const char* Name() { return "raja_cpu"; }
  static bool IsAvailable() { return true; }
};

#if defined(LBMINI_TEST_DEVICE_BACKENDS)
struct RajaGpuBackend {
  using Lattice = lbmini::raja::gpu::LatticeD2Q9<Scalar>;
  using Tube = lbmini::raja::gpu::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::RAJA, lbmini::TargetEnum::GPU);
  }
  static const char* Name() { return "raja_gpu"; }
  static bool IsAvailable() {
    #ifdef USE_CUDA
    int count = 0;
    return lbmini::raja::gpu::kHasDeviceBackend && (cudaGetDeviceCount(&count) == cudaSuccess && count > 0);
    #else
    return lbmini::raja::gpu::kHasDeviceBackend;
    #endif
  }
};
#endif
#endif

#if (defined(USE_CUDA) || defined(LBMINI_TEST_CUDA)) && defined(LBMINI_TEST_DEVICE_BACKENDS)
struct CudaGpuBackend {
  using Lattice = lbmini::cuda::gpu::LatticeD2Q9<Scalar>;
  using Tube = lbmini::cuda::gpu::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::CUDA, lbmini::TargetEnum::GPU);
  }
  static const char* Name() { return "cuda_gpu"; }
  static bool IsAvailable() {
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess && count > 0);
  }
};
#endif

#ifdef LBMINI_TEST_ALPAKA
struct AlpakaCpuBackend {
  using Lattice = lbmini::alpaka::cpu::LatticeD2Q9<Scalar>;
  using Tube = lbmini::alpaka::cpu::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::Alpaka, lbmini::TargetEnum::CPU);
  }
  static const char* Name() { return "alpaka_cpu"; }
  static bool IsAvailable() { return true; }
};

#if defined(LBMINI_TEST_DEVICE_BACKENDS) && defined(USE_ALPAKA_GPU)
struct AlpakaGpuBackend {
  using Lattice = lbmini::alpaka::gpu::LatticeD2Q9<Scalar>;
  using Tube = lbmini::alpaka::gpu::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::Alpaka, lbmini::TargetEnum::GPU);
  }
  static const char* Name() { return "alpaka_gpu"; }
  static bool IsAvailable() { return lbmini::alpaka::gpu::DeviceCount() > 0; }
};
#endif
#endif

using BackendTypes = ::testing::Types<
  PlainBackend
#ifdef LBMINI_TEST_OPENMP_CPU
  , OpenMpCpuBackend
#endif
#if defined(LBMINI_TEST_OPENMP_GPU) && defined(_OPENMP) && (defined(__NVCOMPILER) || defined(__NVHPC__)) && defined(LBMINI_TEST_DEVICE_BACKENDS)
  , OpenMpGpuBackend
#endif
#if defined(LBMINI_TEST_OPENACC_CPU) && defined(_OPENACC)
  , OpenAccCpuBackend
#endif
#if defined(LBMINI_TEST_OPENACC_GPU) && defined(_OPENACC) && defined(LBMINI_TEST_DEVICE_BACKENDS)
  , OpenAccGpuBackend
#endif
#ifdef LBMINI_TEST_OPENCL_CPU
  , OpenClCpuBackend
#endif
#if defined(LBMINI_TEST_OPENCL_GPU) && defined(LBMINI_TEST_DEVICE_BACKENDS)
  , OpenClGpuBackend
#endif
#ifdef LBMINI_TEST_KOKKOS
  , KokkosCpuBackend
#if defined(LBMINI_TEST_DEVICE_BACKENDS) && defined(USE_CUDA)
  , KokkosGpuBackend
#endif
#endif
#ifdef LBMINI_TEST_RAJA
  , RajaCpuBackend
#if defined(LBMINI_TEST_DEVICE_BACKENDS)
  , RajaGpuBackend
#endif
#endif
#if (defined(USE_CUDA) || defined(LBMINI_TEST_CUDA)) && defined(LBMINI_TEST_DEVICE_BACKENDS)
  , CudaGpuBackend
#endif
#ifdef LBMINI_TEST_ALPAKA
  , AlpakaCpuBackend
#if defined(LBMINI_TEST_DEVICE_BACKENDS) && defined(USE_ALPAKA_GPU)
  , AlpakaGpuBackend
#endif
#endif
>;

/// `BackendTypes` without its first entry, the reference solver: the list of
/// backends that are *compared against* it.
template<typename List>
struct DropReference;

template<typename First, typename... Rest>
struct DropReference<::testing::Types<First, Rest...>> {
  static_assert(std::is_same_v<First, PlainBackend>, "BackendTypes must start with the reference backend");
  using type = ::testing::Types<Rest...>;
};

using OptimisedBackendTypes = DropReference<BackendTypes>::type;

/// Puts the backend name into the gtest test name, so a failure reads
/// `BackendMemorySafety/kokkos_cpu.EveryMeshShapeStaysInBounds` instead of an
/// opaque type index.
class BackendNames {
public:
  template<typename BackendT>
  static std::string GetName(int) { return BackendT::Name(); }
};
} // namespace lbmini_test

#endif // LBMINI_TEST_BACKENDTRAITS_HPP_
