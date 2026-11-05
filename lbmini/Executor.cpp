#include "Executor.hpp"

#include <cstdlib>
#include <iostream>

#include "Guards.hpp"
#ifdef USE_OPENCL
#include "Lbm/OpenCl/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenCl/Cpu/LbmTube.hpp"
#include "Lbm/OpenCl/Gpu/LatticeD2Q9.hpp"
#include "Lbm/OpenCl/Gpu/LbmTube.hpp"
#endif
#include "Lbm/OpenMp/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenMp/Cpu/LbmTube.hpp"
#if defined(_OPENMP) && (defined(__NVCOMPILER) || defined(__NVHPC__))
#include "Lbm/OpenMp/Gpu/LatticeD2Q9.hpp"
#include "Lbm/OpenMp/Gpu/LbmTube.hpp"
#endif

#ifdef _OPENACC
#include "Lbm/OpenAcc/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenAcc/Cpu/LbmTube.hpp"
#include "Lbm/OpenAcc/Gpu/LatticeD2Q9.hpp"
#include "Lbm/OpenAcc/Gpu/LbmTube.hpp"
#endif

#include "Lbm/Plain/LatticeD2Q9.hpp"
#include "Lbm/Plain/LbmTube.hpp"
#include "Lbm/CountCores.hpp"

#ifdef USE_MPI
#include <mpi.h>
#include <unistd.h>
#endif

#ifdef USE_MPI
#include "Lbm/Mpi/Cpu/LatticeD2Q9.hpp"
#include "Lbm/Mpi/Cpu/LbmTube.hpp"
#endif

#ifdef USE_CUDA
#include "Lbm/Cuda/Gpu/LatticeD2Q9.hpp"
#include "Lbm/Cuda/Gpu/LbmTube.hpp"
#endif

#ifdef USE_KOKKOS
#include "Lbm/Kokkos/Cpu/LatticeD2Q9.hpp"
#include "Lbm/Kokkos/Cpu/LbmTube.hpp"
#include "Lbm/Kokkos/Gpu/LatticeD2Q9.hpp"
#include "Lbm/Kokkos/Gpu/LbmTube.hpp"
#endif

#ifdef USE_RAJA
#include "Lbm/Raja/Cpu/LatticeD2Q9.hpp"
#include "Lbm/Raja/Cpu/LbmTube.hpp"
#include "Lbm/Raja/Gpu/LatticeD2Q9.hpp"
#include "Lbm/Raja/Gpu/LbmTube.hpp"
#endif

#ifdef USE_ALPAKA
#include "Lbm/Alpaka/Cpu/LatticeD2Q9.hpp"
#include "Lbm/Alpaka/Cpu/LbmTube.hpp"
#include "Lbm/Alpaka/Gpu/LatticeD2Q9.hpp"
#include "Lbm/Alpaka/Gpu/LbmTube.hpp"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _OPENACC
#include <openacc.h>
#endif

namespace lbmini {
namespace {
void SetProcessEnvironment(const char* name, const std::string& value, const bool overwrite) {
  if (!overwrite && std::getenv(name) != nullptr)
    return;

  #ifdef _WIN32
  const int result = _putenv_s(name, value.c_str());
  #else
  const int result = setenv(name, value.c_str(), overwrite ? 1 : 0);
  #endif
  if (result != 0)
    throw std::runtime_error(std::string("Failed to set process environment variable ") + name + ".");
}
} // namespace

template<typename Scalar>
std::unique_ptr<Executor<Scalar>> Executor<Scalar>::singleton_ = nullptr;

template<typename Scalar>
Executor<Scalar>* Executor<Scalar>::GetExecutor(
  const FluidData<Scalar>& fluid,
  const MeshData<Scalar, 2>& mesh,
  const ControlData<Scalar>& control,
  const PerformanceData& performance,
  int* argc,
  char*** argv
) {
  if (!singleton_) {
    singleton_.reset(
      new Executor(
        fluid,
        mesh,
        control,
        performance,
        argc,
        argv
      )
    );
  }

  return singleton_.get();
}

template<typename Scalar>
void Executor<Scalar>::Reset() {
  singleton_.reset();
}

template<typename Scalar>
typename Executor<Scalar>::ILbmTubeT* Executor<Scalar>::GetLbmTube() const {
  return lbmTube_.get();
}

template<typename Scalar>
Executor<Scalar>::Executor(
  const FluidData<Scalar>& fluid,
  const MeshData<Scalar, 2>& mesh,
  const ControlData<Scalar>& control,
  const PerformanceData& performance,
  int* argc,
  char*** argv
)
  : fluid_(fluid), mesh_(mesh), control_(control), performance_(performance), mpiInitializedByUs_(false) {
  #ifdef USE_MPI
  bool is_under_mpi = std::getenv("OMPI_COMM_WORLD_SIZE") != nullptr ||
    std::getenv("PMI_RANK") != nullptr ||
    std::getenv("PMIX_RANK") != nullptr;
  if (is_under_mpi && argc && argv) {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
      MPI_Init(argc, argv);
      mpiInitializedByUs_ = true;
    }
  }
  #endif

  // Rejected here, before a single buffer is allocated: every backend derives
  // its buffer sizes and its flat indices from these extents, so a degenerate
  // or oversized mesh has to fail at the same place whichever one is selected.
  ValidateMeshExtents(mesh.size[0], mesh.size[1], ILatticeT::Speeds(), "Executor");

  // Hardware concurrency setup.
  // Environment variables and the OpenMP thread count are process-global, so
  // they belong to the single owner of the process. Writing them from a solver
  // constructor would silently reconfigure instances that already exist and
  // would race with any other thread reading the environment.
  const Index cores = (performance.cores > 0) ? performance.cores : static_cast<Index>(lbmini::CountPerformanceCores());
  const std::string coresStr = std::to_string(cores);
  #if defined(_OPENMP)
  omp_set_num_threads(static_cast<int>(cores));
  #endif
  SetProcessEnvironment("ACC_NUM_CORES", coresStr, true);
  SetProcessEnvironment("POCL_MAX_PTHREAD_COUNT", coresStr, true);
  // Thread affinity: set once, and never overridden if the user (or a job
  // scheduler) already chose a placement policy.
  SetProcessEnvironment("OMP_PLACES", "cores", false);
  SetProcessEnvironment("OMP_PROC_BIND", "close", false);

  switch (performance.backend) {
    case lbmini::BackendEnum::Plain: {
      if (performance.target != lbmini::TargetEnum::CPU) {
        throw std::runtime_error("Plain backend only supports CPU target.");
      }
      using LbmLattice = lbmini::plain::LatticeD2Q9<Scalar>;
      using LbmTube = lbmini::plain::LbmTube<Scalar, LbmLattice>;
      lbmTube_.reset(new LbmTube(fluid, mesh, control, performance));
      break;
    }
    case lbmini::BackendEnum::OpenMP: {
      #ifdef _OPENMP
      switch (performance.target) {
        case lbmini::TargetEnum::CPU: {
          omp_set_default_device(0);
          using LbmLattice = lbmini::openmp::cpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::openmp::cpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(new LbmTube(fluid, mesh, control, performance));
          break;
        }
        case lbmini::TargetEnum::GPU: {
          #if defined(_OPENMP) && (defined(__NVCOMPILER) || defined(__NVHPC__))
          if (omp_get_num_devices() < 1) {
            std::cerr << "No OpenMP target devices available." << std::endl;
            lbmTube_.reset(nullptr);
            return;
          }
          omp_set_default_device(0);
          using LbmLattice = lbmini::openmp::gpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::openmp::gpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(new LbmTube(fluid, mesh, control, performance));
          break;
          #else
          throw std::runtime_error("OpenMP GPU target is only supported with NVIDIA HPC SDK (nvc++).");
          #endif
        }
        default:
          throw std::runtime_error("Invalid target for OpenMP backend.");
      }
      #else
      throw std::runtime_error("OpenMP backend not enabled.");
      #endif
      break;
    }
    case lbmini::BackendEnum::OpenACC: {
      #ifdef _OPENACC
      switch (performance.target) {
        case lbmini::TargetEnum::CPU: {
          acc_set_device_type(acc_device_host);
          using LbmLattice = lbmini::openacc::cpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::openacc::cpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(new LbmTube(fluid, mesh, control, performance));
          break;
        }
        case lbmini::TargetEnum::GPU: {
          acc_set_device_type(acc_device_default);
          using LbmLattice = lbmini::openacc::gpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::openacc::gpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(new LbmTube(fluid, mesh, control, performance));
          break;
        }
        default:
          throw std::runtime_error("Invalid target for OpenACC backend.");
      }
      #else
      throw std::runtime_error("OpenACC backend not enabled.");
      #endif
      break;
    }
    case lbmini::BackendEnum::OpenCL: {
      #ifdef USE_OPENCL
      switch (performance.target) {
        case lbmini::TargetEnum::CPU: {
          using LbmLattice = lbmini::opencl::cpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::opencl::cpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(new LbmTube(fluid, mesh, control, performance));
          break;
        }
        case lbmini::TargetEnum::GPU: {
          using LbmLattice = lbmini::opencl::gpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::opencl::gpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(new LbmTube(fluid, mesh, control, performance));
          break;
        }
        default:
          throw std::runtime_error("Invalid target for OpenCL backend.");
      }
      #else
      throw std::runtime_error("OpenCL backend not enabled.");
      #endif
      break;
    }
    case lbmini::BackendEnum::MPI: {
      #ifdef USE_MPI
      if (performance.target != lbmini::TargetEnum::CPU) {
        throw std::runtime_error("MPI backend only supports CPU target.");
      }
      bool is_under_mpi = std::getenv("OMPI_COMM_WORLD_SIZE") != nullptr ||
        std::getenv("PMI_RANK") != nullptr ||
        std::getenv("PMIX_RANK") != nullptr;
      if (!is_under_mpi && argc && argv) {
        int num_procs = performance.cores > 0 ? performance.cores : lbmini::CountPerformanceCores();
        std::string num_procs_str = std::to_string(num_procs);
        std::vector<char*> args = {
          const_cast<char*>("mpirun"),
          const_cast<char*>("--use-hwthread-cpus"),
          const_cast<char*>("--oversubscribe"),
          const_cast<char*>("-np"),
          const_cast<char*>(num_procs_str.data())
        };
        for (int i = 0; i < *argc; ++i)
          args.push_back((*argv)[i]);
        args.push_back(nullptr);
        execvp("mpirun", args.data());
        throw std::runtime_error("Failed to automatically launch mpirun.");
      }
      using LbmLattice = lbmini::mpi::cpu::LatticeD2Q9<Scalar>;
      using LbmTube = lbmini::mpi::cpu::LbmTube<Scalar, LbmLattice>;
      lbmTube_.reset(new LbmTube(fluid, mesh, control, performance));
      #else
      throw std::runtime_error("MPI backend not enabled.");
      #endif
      break;
    }
    case lbmini::BackendEnum::CUDA: {
      #ifdef USE_CUDA
      if (performance.target != lbmini::TargetEnum::GPU) {
        throw std::runtime_error("CUDA backend only supports GPU target.");
      }
      using LbmLattice = lbmini::cuda::gpu::LatticeD2Q9<Scalar>;
      using LbmTube = lbmini::cuda::gpu::LbmTube<Scalar, LbmLattice>;
      lbmTube_.reset(new LbmTube(fluid, mesh, control, performance));
      #else
      throw std::runtime_error("CUDA backend not enabled.");
      #endif
      break;
    }
    case lbmini::BackendEnum::Kokkos: {
      #ifdef USE_KOKKOS
      switch (performance.target) {
        case lbmini::TargetEnum::CPU: {
          using LbmLattice = lbmini::kokkos::cpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::kokkos::cpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(new LbmTube(fluid, mesh, control, performance));
          break;
        }
        case lbmini::TargetEnum::GPU: {
          // Without a device backend the Kokkos GPU execution space *is* the
          // host one: the run would complete and be reported as a GPU run while
          // never touching a device. Refusing it keeps the measurement honest.
          if (!lbmini::kokkos::gpu::kHasDeviceBackend) {
            throw std::runtime_error(
              "Kokkos GPU target is unavailable: Kokkos was built without a device "
              "backend (CUDA/HIP/SYCL). Rebuild with -DENABLE_KOKKOS_CUDA=ON or "
              "select the CPU target."
            );
          }
          using LbmLattice = lbmini::kokkos::gpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::kokkos::gpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(new LbmTube(fluid, mesh, control, performance));
          break;
        }
        default:
          throw std::runtime_error("Invalid target for Kokkos backend.");
      }
      #else
      throw std::runtime_error("Kokkos backend not enabled.");
      #endif
      break;
    }
    case lbmini::BackendEnum::RAJA: {
      #ifdef USE_RAJA
      switch (performance.target) {
        case lbmini::TargetEnum::CPU: {
          using LbmLattice = lbmini::raja::cpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::raja::cpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(new LbmTube(fluid, mesh, control, performance));
          break;
        }
        case lbmini::TargetEnum::GPU: {
          // Same reasoning as for Kokkos above: without a device backend the
          // RAJA GPU policy degrades to a host policy, so a "GPU" run would
          // silently execute on the CPU.
          if (!lbmini::raja::gpu::kHasDeviceBackend) {
            throw std::runtime_error(
              "RAJA GPU target is unavailable: RAJA was built without a device "
              "backend (CUDA/HIP). Rebuild with -DENABLE_RAJA_CUDA=ON or select "
              "the CPU target."
            );
          }
          using LbmLattice = lbmini::raja::gpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::raja::gpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(new LbmTube(fluid, mesh, control, performance));
          break;
        }
        default:
          throw std::runtime_error("Invalid target for RAJA backend.");
      }
      #else
      throw std::runtime_error("RAJA backend not enabled.");
      #endif
      break;
    }
    case lbmini::BackendEnum::Alpaka: {
      #ifdef USE_ALPAKA
      switch (performance.target) {
        case lbmini::TargetEnum::CPU: {
          using LbmLattice = lbmini::alpaka::cpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::alpaka::cpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(new LbmTube(fluid, mesh, control, performance));
          break;
        }
        case lbmini::TargetEnum::GPU: {
          // The device flavour is a separate CUDA library; without it there is
          // no accelerator code at all, so a "GPU" run is refused rather than
          // silently measured on the host.
          #ifdef USE_ALPAKA_GPU
          using LbmLattice = lbmini::alpaka::gpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::alpaka::gpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(new LbmTube(fluid, mesh, control, performance));
          #else
          throw std::runtime_error(
            "Alpaka GPU target is unavailable: the Alpaka backend was built without its "
            "CUDA flavour. Rebuild with -DENABLE_ALPAKA_CUDA=ON or select the CPU target."
          );
          #endif
          break;
        }
        default:
          throw std::runtime_error("Invalid target for Alpaka backend.");
      }
      #else
      throw std::runtime_error("Alpaka backend not enabled.");
      #endif
      break;
    }
    default: {
      std::cerr << "Invalid backend selected. Backend enum: " << static_cast<int>(performance.backend) << std::endl;
      lbmTube_.reset(nullptr);
      break;
    }
  }
}

template<typename Scalar>
Executor<Scalar>::~Executor() {
  asyncWriter_.reset();
  lbmTube_.reset();
  #ifdef USE_MPI
  if (mpiInitializedByUs_) {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
  }
  #endif
}

template<typename Scalar>
void Executor<Scalar>::OutputData(
  Index step,
  const std::string& outDir,
  double elapsed_seconds,
  int rank
) {
  // The getters of the distributed backend gather the global fields with
  // collective calls, so every rank has to take part; only rank 0 writes.
  auto rho = lbmTube_->Rho();
  auto p = lbmTube_->P();
  auto u = lbmTube_->U();

  if (rank != 0)
    return;

  if (!asyncWriter_) {
    asyncWriter_ = std::make_unique<AsyncWriter<Scalar>>(fluid_, mesh_);
  }

  asyncWriter_->Enqueue(OutputSnapshot<Scalar>{
    step,
    outDir,
    elapsed_seconds,
    std::move(rho),
    std::move(p),
    std::move(u)
  });
}

template<typename Scalar>
void Executor<Scalar>::SynchronizeRanks() const {
  #ifdef USE_MPI
  if (performance_.backend != lbmini::BackendEnum::MPI)
    return;
  int initialized = 0;
  int finalized = 0;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  if (initialized && !finalized)
    MPI_Barrier(MPI_COMM_WORLD);
  #endif
}

template<typename Scalar>
void Executor<Scalar>::Run(const std::string& outpath) {
  // The constructor leaves `lbmTube_` empty when the requested target is not
  // available (no OpenMP device, unknown backend). Dereferencing it here would
  // be a null deref in the middle of a benchmark campaign.
  if (!lbmTube_)
    throw std::runtime_error("No solver was created for the requested backend/target combination.");

  lbmTube_->Init();

  int rank = lbmTube_->Rank();

  OutputData(0, outpath, 0.0, rank);

  // Calculate time step and steps
  constexpr Scalar cs2 = static_cast<Scalar>(1.0) / static_cast<Scalar>(3.0);
  const Scalar dx = mesh_.lx() / static_cast<Scalar>(mesh_.nx());
  const Scalar uPhysRef = std::sqrt(fluid_.gamma * fluid_.pressureL / fluid_.densityL);
  const Scalar uLuRef = std::sqrt(fluid_.gamma * cs2);
  const Scalar dt = dx * (uLuRef / uPhysRef);
  const Index kSteps = static_cast<Index>(control_.tmax / dt);

  // A `tmax` shorter than a single lattice time step used to be accepted in
  // silence: the loop below then ran zero iterations, the step-0 snapshot was
  // written, and the process printed "Simulation completed" without a single
  // kernel launch, halo exchange or streaming step. That is a benchmark of
  // nothing - and, worse, a memory check of nothing: whatever tool wraps such
  // a run reports success on a solver that never executed. `dt` is derived
  // (`dx * uLuRef / uPhysRef`), so this is not a value a user can validate by
  // reading the configuration file, which is why it is checked here.
  if (kSteps < 1)
    throw std::runtime_error(
      "Control.tmax (" + std::to_string(control_.tmax) +
      ") is shorter than one lattice time step (dt = " + std::to_string(dt) +
      "), so this run would advance zero steps and measure nothing. Increase "
      "tmax, or refine the mesh - dt scales with lx / nx."
    );

  // Only the time marching is timed. Gathering the fields, copying them back
  // from the device and writing the snapshots happen between the timed
  // intervals, and the asynchronous writer is drained before each interval so
  // that it competes neither for cores nor for memory bandwidth. Each snapshot
  // carries the time-marching time accumulated up to its step.
  using Clock = std::chrono::steady_clock;
  const auto startRunTime = Clock::now();
  double computeSeconds = 0.0;

  const Index stepsPerOutput = std::max(kSteps / control_.printStep, Index(1));
  for (Index i = 0; i < kSteps;) {
    const Index stepsPerChunk = std::min(stepsPerOutput, kSteps - i);

    if (asyncWriter_) {
      asyncWriter_->WaitAll();
    }
    SynchronizeRanks();
    const auto chunkStart = Clock::now();
    lbmTube_->Run(stepsPerChunk, true);
    SynchronizeRanks();
    const std::chrono::duration<double> chunkSeconds = Clock::now() - chunkStart;
    computeSeconds += chunkSeconds.count();
    i += stepsPerChunk;

    if (rank == 0) {
      std::cout << "Time step: " << i << std::endl;
    }
    OutputData(i, outpath, computeSeconds, rank);
  }

  if (asyncWriter_) {
    asyncWriter_->WaitAll();
  }

  const std::chrono::duration<double> elapsed_time = Clock::now() - startRunTime;
  if (rank == 0) {
    std::cout << "Simulation completed. Time-marching time: " << computeSeconds
              << " s. Total run time: " << elapsed_time.count() << " s" << std::endl;
  }
}

template class Executor<double>;
} // namespace lbmini
