/**
 * @file memory_safety_test.cpp
 * @brief Backend-agnostic memory-safety suite: run this (under sanitizers)
 *        before trusting any benchmark number.
 *
 * The tests here are not about physics — the cross-backend agreement tests in
 * `kokkos_lbm_test.cpp` / `raja_lbm_test.cpp` cover that. These check the
 * properties whose violation corrupts memory:
 *
 *  - every mesh shape (including the degenerate ones the flat-index arithmetic
 *    was never designed for) allocates, streams and reads back in bounds;
 *  - the macroscopic getters hand out *copies*, so a field can outlive the
 *    solver without dangling;
 *  - solvers can be created, destroyed, nested and interleaved in one process
 *    without one instance tearing down global state another one still uses
 *    (Kokkos `finalize`, OpenCL contexts, OpenMP device allocations, ...);
 *  - a solver is destructible through the interface pointer `Executor` stores.
 *
 * On their own these tests only prove "it did not crash". Their value comes
 * from being executed under AddressSanitizer + UndefinedBehaviorSanitizer,
 * Valgrind, or `compute-sanitizer`, which turn every out-of-bounds byte,
 * use-after-free and double-free into a hard failure. See
 * `docs/memory-safety.md` for the full procedure.
 */

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "MemorySafetyHarness.hpp"

// Only for `Executor<Scalar>::ILbmTubeT`: the suite pins down the exact
// pointer type the production code owns every backend through.
#include "Executor.hpp"
#include "Guards.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/ILattice.hpp"
#include "Lbm/ILbmTube.hpp"
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

namespace {
using lbmini_test::Index;
using lbmini_test::MeshShape;
using lbmini_test::Scalar;

/// Shape used by the value-comparing tests (the validated benchmark shape).
constexpr MeshShape kRefShape{ 64, 4 };

/// Relative tolerance for "the same computation must give the same result".
constexpr Scalar kTol = 1.0e-9;

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
>;

#ifdef LBMINI_TEST_KOKKOS
const ::testing::Environment* kKokkosEnvironment =
  ::testing::AddGlobalTestEnvironment(new lbmini_test::KokkosEnvironment);
#endif

/// Builds an initialised solver for `shape` on the heap (never on the stack:
/// the distributions are large and a heap object lets the sanitizers track its
/// exact extent and its lifetime).
template<typename BackendT>
std::unique_ptr<typename BackendT::Tube> MakeSolver(const MeshShape& shape) {
  auto tube = std::make_unique<typename BackendT::Tube>(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(shape.nx, shape.ny),
    lbmini_test::MakeControl(),
    BackendT::Performance()
  );
  tube->Init();
  return tube;
}

TEST(AlignedAllocatorSafety, StorageIsCacheLineAligned) {
  lbmini::DeviceBuffer<double> storage(17, 0.0);
  ASSERT_NE(storage.data(), nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(storage.data()) % 64U, 0U);
}

TEST(AlignedAllocatorSafety, ByteSizeOverflowIsRejected) {
  lbmini::AlignedAllocator<double, 64> allocator;
  const std::size_t overflowingCount = std::numeric_limits<std::size_t>::max() / sizeof(double) + 1U;
  EXPECT_THROW((void)allocator.allocate(overflowingCount), std::bad_array_new_length);
}

/// Puts the backend name into the gtest test name, so a failure reads
/// `BackendMemorySafety/kokkos_cpu.EveryMeshShapeStaysInBounds` instead of an
/// opaque type index.
class BackendNames {
public:
  template<typename BackendT>
  static std::string GetName(int) { return BackendT::Name(); }
};

template<typename BackendT>
class BackendMemorySafety : public ::testing::Test {
protected:
  void SetUp() override {
    if (!BackendT::IsAvailable()) {
      GTEST_SKIP() << BackendT::Name() << " backend is not available in current runtime environment.";
    }
  }
};

TYPED_TEST_SUITE(BackendMemorySafety, BackendTypes, BackendNames);

/**
 * @brief Every mesh shape must allocate, step and read back within bounds.
 *
 * This is the core out-of-bounds probe. The solvers derive `N = nx * ny` and
 * then index `i * ny + j`, wrap `y % ny` and split the initial state at
 * `nx / 2`; single-row, single-column and odd meshes are precisely the inputs
 * that break such arithmetic. Under ASan a one-element overshoot in a
 * streaming table or in a getter copy fails here.
 */
TYPED_TEST(BackendMemorySafety, EveryMeshShapeStaysInBounds) {
  for (const auto& shape : lbmini_test::MeshShapes()) {
    const std::string where = std::string(TypeParam::Name()) + " mesh " + shape.Name();
    SCOPED_TRACE(where);

    auto tube = MakeSolver<TypeParam>(shape);

    // Reading right after Init() catches tables/buffers sized from a stale or
    // wrongly rounded extent before any time stepping hides the evidence.
    (void)lbmini_test::ReadAllFields(*tube, shape, where + " after Init");
    lbmini_test::ExpectPhysical(*tube, shape, where + " after Init");

    tube->Run(lbmini_test::kSweepSteps, false);

    (void)lbmini_test::ReadAllFields(*tube, shape, where + " after Run");
    lbmini_test::ExpectPhysical(*tube, shape, where + " after Run");
  }
}

/// A longer run must not walk off the end of the ping-pong buffers: the
/// backends swap `f`/`g` with their auxiliary partner every step, so an odd
/// number of steps leaves the "other" buffer active.
TYPED_TEST(BackendMemorySafety, OddAndEvenStepCountsAreBothSafe) {
  for (const Index steps : { Index{ 1 }, Index{ 2 }, lbmini_test::kStabilitySteps + 1 }) {
    const std::string where =
      std::string(TypeParam::Name()) + " steps " + std::to_string(steps);
    SCOPED_TRACE(where);

    auto tube = MakeSolver<TypeParam>(kRefShape);
    tube->Run(steps, false);

    (void)lbmini_test::ReadAllFields(*tube, kRefShape, where);
    lbmini_test::ExpectPhysical(*tube, kRefShape, where);
  }
}

/// `Run(0)` must be a no-op, not a zero-length loop that still swaps buffers or
/// reads a table it never filled.
TYPED_TEST(BackendMemorySafety, ZeroStepRunIsANoOp) {
  auto tube = MakeSolver<TypeParam>(kRefShape);

  const auto before = tube->Rho();
  tube->Run(0, false);
  const auto after = tube->Rho();

  ASSERT_EQ(before.dimension(0), after.dimension(0));
  ASSERT_EQ(before.dimension(1), after.dimension(1));
  for (Index i = 0; i < kRefShape.nx; ++i)
    for (Index j = 0; j < kRefShape.ny; ++j)
      EXPECT_DOUBLE_EQ(after(i, j), before(i, j)) << "rho at (" << i << ", " << j << ")";
}

/**
 * @brief The getters must return owning copies, never a view of live storage.
 *
 * If a backend ever returns a tensor mapped onto its internal (or device)
 * buffer, writing to the returned object would corrupt the solver state and —
 * worse — the tensor would dangle the moment the solver dies. Both halves are
 * checked: mutating the copy leaves the solver untouched, and the copy stays
 * valid after the solver is destroyed.
 */
TYPED_TEST(BackendMemorySafety, GettersReturnIndependentCopies) {
  auto tube = MakeSolver<TypeParam>(kRefShape);
  tube->Run(lbmini_test::kSweepSteps, false);

  auto rho = tube->Rho();
  auto u = tube->U();
  const Scalar rhoBefore = rho(0, 0);
  const Scalar uBefore = u(0, 0, 0);

  rho(0, 0) = rhoBefore + Scalar{ 1234.5 };
  u(0, 0, 0) = uBefore + Scalar{ 1234.5 };

  const auto rhoAgain = tube->Rho();
  const auto uAgain = tube->U();
  EXPECT_DOUBLE_EQ(rhoAgain(0, 0), rhoBefore) << "Rho() exposed internal storage";
  EXPECT_DOUBLE_EQ(uAgain(0, 0, 0), uBefore) << "U() exposed internal storage";

  // Snapshot, then kill the solver: the tensors must still own their data.
  const auto snapshot = tube->Rho();
  tube.reset();

  Scalar checksum = Scalar{ 0 };
  for (Index i = 0; i < kRefShape.nx; ++i)
    for (Index j = 0; j < kRefShape.ny; ++j)
      checksum += snapshot(i, j);
  EXPECT_TRUE(lbmini::IsFinite(checksum)) << "field did not survive solver destruction";
  EXPECT_DOUBLE_EQ(snapshot(0, 0), rhoBefore);
}

/// Create and destroy the solver repeatedly in one process: each cycle must
/// release everything it took (LeakSanitizer / Valgrind assert that) and must
/// not free anything twice.
TYPED_TEST(BackendMemorySafety, RepeatedConstructionAndDestructionIsClean) {
  for (int cycle = 0; cycle < 2; ++cycle) {
    SCOPED_TRACE(std::string(TypeParam::Name()) + " cycle " + std::to_string(cycle));
    auto tube = MakeSolver<TypeParam>(kRefShape);
    tube->Run(lbmini_test::kSweepSteps, false);
    (void)lbmini_test::ReadAllFields(*tube, kRefShape, "repeated lifecycle");
  }
}

/**
 * @brief One solver must survive the destruction of another.
 *
 * This is the guard against framework-level teardown: a backend that owns a
 * global runtime (Kokkos `initialize`/`finalize`, an OpenCL context, a CUDA
 * context, an OpenMP target device) must not shut it down while another
 * instance still holds allocations. If it does, the outer solver's next `Run()`
 * or getter touches memory that no longer belongs to the process.
 */
TYPED_TEST(BackendMemorySafety, SolverSurvivesDestructionOfAnotherInstance) {
  auto reference = MakeSolver<TypeParam>(kRefShape);
  reference->Run(lbmini_test::kSweepSteps, false);
  const auto expected = reference->Rho();

  auto outer = MakeSolver<TypeParam>(kRefShape);
  {
    auto inner = MakeSolver<TypeParam>(kRefShape);
    inner->Run(lbmini_test::kSweepSteps, false);
    (void)lbmini_test::ReadAllFields(*inner, kRefShape, "inner solver");
  } // inner dies here: the outer one must be untouched.

  outer->Run(lbmini_test::kSweepSteps, false);
  const auto actual = outer->Rho();

  for (Index i = 0; i < kRefShape.nx; ++i) {
    for (Index j = 0; j < kRefShape.ny; ++j) {
      ASSERT_TRUE(lbmini::IsFinite(actual(i, j))) << "rho at (" << i << ", " << j << ")";
      EXPECT_NEAR(actual(i, j), expected(i, j), kTol * std::max(Scalar{ 1 }, std::fabs(expected(i, j))))
        << "rho at (" << i << ", " << j << ")";
    }
  }
}

/**
 * @brief Two live solvers must not share mutable state.
 *
 * Interleaving the steps of two instances has to give exactly what each one
 * would produce alone. A shared static scratch buffer, a shared device
 * allocation or a shared streaming table shows up here as a mismatch — and is a
 * data race (and therefore a corruption) as soon as the two run concurrently.
 */
TYPED_TEST(BackendMemorySafety, ConcurrentInstancesDoNotShareState) {
  auto alone = MakeSolver<TypeParam>(kRefShape);
  for (Index step = 0; step < lbmini_test::kSweepSteps; ++step)
    alone->Step(false);
  const auto expected = alone->Rho();

  auto first = MakeSolver<TypeParam>(kRefShape);
  auto second = MakeSolver<TypeParam>(kRefShape);
  for (Index step = 0; step < lbmini_test::kSweepSteps; ++step) {
    first->Step(false);
    second->Step(false);
  }

  const auto firstRho = first->Rho();
  const auto secondRho = second->Rho();
  for (Index i = 0; i < kRefShape.nx; ++i) {
    for (Index j = 0; j < kRefShape.ny; ++j) {
      const Scalar tol = kTol * std::max(Scalar{ 1 }, std::fabs(expected(i, j)));
      EXPECT_NEAR(firstRho(i, j), expected(i, j), tol) << "first at (" << i << ", " << j << ")";
      EXPECT_NEAR(secondRho(i, j), expected(i, j), tol) << "second at (" << i << ", " << j << ")";
    }
  }
}

/**
 * @brief Multiple threads creating, running, and destroying independent solvers
 *        concurrently must not cause data races or memory corruption.
 */
TYPED_TEST(BackendMemorySafety, MultiThreadedConcurrentExecutionIsSafe) {
  constexpr int kNumThreads = 4;
  std::vector<std::thread> workers;
  workers.reserve(kNumThreads);
  std::atomic<bool> failed{ false };

  for (int t = 0; t < kNumThreads; ++t) {
    workers.emplace_back([&failed]() {
      try {
        for (int cycle = 0; cycle < 3; ++cycle) {
          auto tube = MakeSolver<TypeParam>(kRefShape);
          tube->Run(lbmini_test::kSweepSteps, false);
          Scalar checksum = lbmini_test::ReadAllFields(*tube, kRefShape, "multi-threaded solver");
          if (!lbmini::IsFinite(checksum)) {
            failed.store(true);
          }
        }
      } catch (...) {
        failed.store(true);
      }
    });
  }

  for (auto& w : workers) {
    if (w.joinable())
      w.join();
  }

  EXPECT_FALSE(failed.load()) << TypeParam::Name() << ": multi-threaded concurrent execution failed";
}

/**
 * @brief The solver must be destructible through the polymorphic interface.
 *
 * `Executor` owns every backend as `std::unique_ptr<ILbmTube<...>>`, so the
 * destructor call goes through the base pointer. Without a virtual destructor
 * that leaks every buffer the derived class owns — silently, and only in the
 * production path, never in a test that holds the concrete type.
 */
TYPED_TEST(BackendMemorySafety, DestructionThroughInterfaceIsComplete) {
  using Tube = typename TypeParam::Tube;
  using Interface = lbmini::ILbmTube<Scalar, typename TypeParam::Lattice>;

  static_assert(std::is_base_of<Interface, Tube>::value,
                "every backend must implement the ILbmTube interface");
  static_assert(std::has_virtual_destructor<Interface>::value,
                "ILbmTube must have a virtual destructor");

  std::unique_ptr<Interface> tube = std::make_unique<Tube>(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(kRefShape.nx, kRefShape.ny),
    lbmini_test::MakeControl(),
    TypeParam::Performance()
  );
  tube->Init();
  tube->Run(lbmini_test::kSweepSteps, false);
  (void)lbmini_test::ReadAllFields(*tube, kRefShape, "interface pointer");
  tube.reset(); // must free the derived buffers, not just the base subobject
}

/**
 * @brief The pointer type `Executor` stores must be a *real* base class.
 *
 * `Executor` holds every backend as `std::unique_ptr<ILbmTubeNd<Scalar, 2>>`.
 * That used to be a `reinterpret_cast` to `ILbmTube<Scalar,
 * ILattice<Scalar,2,9>>`, an instantiation which is *not* a base class of any
 * concrete solver (`ILbmTube<Scalar, plain::LatticeD2Q9<double>>` is an
 * unrelated type): every virtual call and the final `delete` went through a
 * pointer of the wrong type — undefined behaviour that survived only because
 * the instantiations happened to share a layout.
 *
 * The virtual API now lives in the lattice-independent `ILbmTubeNd<Scalar,
 * Dim>`, so the conversion is an ordinary implicit upcast. This test keeps it
 * that way: if a backend ever stops deriving from that base, the code no
 * longer compiles here instead of corrupting memory in production.
 * See `docs/memory-safety.md` §7.
 */
TYPED_TEST(BackendMemorySafety, ExecutorInterfaceIsARealBaseClass) {
  using Tube = typename TypeParam::Tube;
  using ExecutorInterface = typename lbmini::Executor<Scalar>::ILbmTubeT;

  static_assert(std::is_base_of<ExecutorInterface, Tube>::value,
                "the type Executor stores must be a genuine base class of every backend "
                "- no reinterpret_cast is allowed to bridge the gap");
  static_assert(std::is_convertible<Tube*, ExecutorInterface*>::value,
                "the upcast Executor performs must be implicit");
  static_assert(std::has_virtual_destructor<ExecutorInterface>::value,
                "deleting a solver through the executor's pointer must run the derived destructor");

  // Owning through the base pointer, exactly as `Executor` does: allocate,
  // drive, read and destroy. Under ASan/Valgrind this is where a wrong `this`
  // or a partial destruction shows up.
  std::unique_ptr<ExecutorInterface> erased = std::make_unique<Tube>(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(kRefShape.nx, kRefShape.ny),
    lbmini_test::MakeControl(),
    TypeParam::Performance()
  );

  auto* concrete = static_cast<Tube*>(erased.get());
  EXPECT_EQ(static_cast<const void*>(erased.get()), static_cast<const void*>(concrete))
    << TypeParam::Name() << ": unexpected base-subobject offset";

  erased->Init();
  erased->Run(lbmini_test::kSweepSteps, false);
  (void)lbmini_test::ReadAllFields(*erased, kRefShape, "executor interface pointer");
  erased.reset();
}

/**
 * @brief A degenerate or oversized mesh must be rejected, never indexed.
 *
 * `lbmini::ValidateMeshExtents` is the single O(1) guard every solver runs
 * before its first allocation. The shapes below are the ones that turn
 * `i * ny + j`, `y % ny` and the int32 streaming tables into an out-of-bounds
 * access or a division by zero, so the contract is: throw, loudly, at
 * construction — see `docs/memory-safety.md` §7.
 */
TYPED_TEST(BackendMemorySafety, DegenerateMeshIsRejectedAtConstruction) {
  const std::vector<MeshShape> rejected = {
    { 0, 4 },   // empty allocation, kernels still loop
    { 4, 0 },   // ... and `y % ny` divides by zero
    { -8, 4 },  // negative extent from a hand-edited config
    { 4, -8 },
    { 0, 0 },
  };

  for (const auto& shape : rejected) {
    EXPECT_THROW(
      { auto rejectedSolver = MakeSolver<TypeParam>(shape); },
      std::invalid_argument
    ) << TypeParam::Name() << ": mesh " << shape.Name() << " must be rejected, not indexed";
  }
}

/**
 * @brief A mesh whose cell count leaves `std::int32_t` must be rejected too.
 *
 * The gather tables are `std::int32_t`, so `nx * ny` beyond 2^31 wraps to a
 * negative index and the unchecked gather in `stream()` reads a wild address.
 * The rejection has to happen before the first allocation, which is also why
 * these shapes are cheap to test: nothing is ever allocated for them.
 */
TYPED_TEST(BackendMemorySafety, OversizedMeshIsRejectedBeforeAnyAllocation) {
  constexpr Index kInt32Max = static_cast<Index>(std::numeric_limits<std::int32_t>::max());
  const std::vector<MeshShape> rejected = {
    { Index{ 1 } << 20, Index{ 1 } << 12 },  // 2^32 cells: wraps to 0
    { kInt32Max, Index{ 2 } },               // 2 * (2^31 - 1) cells
    { kInt32Max, kInt32Max },                // overflows the product itself
  };

  for (const auto& shape : rejected) {
    EXPECT_THROW(
      { auto rejectedSolver = MakeSolver<TypeParam>(shape); },
      std::invalid_argument
    ) << TypeParam::Name() << ": mesh " << shape.Name()
      << " does not fit the int32 gather tables and must be rejected";
  }
}

/**
 * @brief Solvers must not be copyable or movable.
 *
 * Every backend owns raw storage — aligned host buffers, `omp_target_alloc` /
 * `cudaMalloc` device pointers, `cl::Buffer` handles, RAJA allocations — and
 * frees it in its destructor. An implicitly generated copy or move would
 * duplicate those handles and free each of them twice, while leaving the source
 * object pointing at released memory: a double free plus a use-after-free, from
 * a single accidental assignment or a solver stored in a `std::vector`.
 * A compile-time check is the only reliable proof, because the offending code
 * would be a caller that does not exist yet. See `docs/memory-safety.md` §7.
 */
TYPED_TEST(BackendMemorySafety, SolverOwnershipIsNotCopyableOrMovable) {
  using Tube = typename TypeParam::Tube;

  static_assert(!std::is_copy_constructible<Tube>::value,
                "a solver that owns raw buffers must delete its copy constructor");
  static_assert(!std::is_copy_assignable<Tube>::value,
                "a solver that owns raw buffers must delete its copy assignment");
  static_assert(!std::is_move_constructible<Tube>::value,
                "a solver that owns raw buffers must delete its move constructor "
                "(or implement it); the default one duplicates every handle");
  static_assert(!std::is_move_assignable<Tube>::value,
                "a solver that owns raw buffers must delete its move assignment");

  // Nothing to execute: the proof is the four assertions above. The body keeps
  // the test visible in `--gtest_list_tests`, so the pre-benchmark checklist
  // can confirm the check actually ran for every backend in the type list.
  SUCCEED() << TypeParam::Name() << ": ownership is exclusive";
}
} // namespace
