#include "Executor.hpp"

#include <cstdlib>
#include <iostream>

#include "Lbm/OpenCl/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenCl/Cpu/LbmTube.hpp"
#include "Lbm/OpenCl/Gpu/LatticeD2Q9.hpp"
#include "Lbm/OpenCl/Gpu/LbmTube.hpp"
#include "Lbm/OpenMp/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenMp/Cpu/LbmTube.hpp"
#include "Lbm/OpenMp/Gpu/LatticeD2Q9.hpp"
#include "Lbm/OpenMp/Gpu/LbmTube.hpp"

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

#ifdef __CUDACC__
#include "Lbm/Cuda/Gpu/LatticeD2Q9.hpp"
#include "Lbm/Cuda/Gpu/LbmTube.hpp"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _OPENACC
#include <openacc.h>
#endif

namespace lbmini {
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

  // Hardware concurrency setup
  if (performance.cores > 0) {
    #if defined(_OPENMP)
    omp_set_num_threads(performance.cores);
    #endif
    std::string coresStr = std::to_string(performance.cores);
    setenv("ACC_NUM_CORES", coresStr.c_str(), 1);
    setenv("POCL_MAX_PTHREAD_COUNT", coresStr.c_str(), 1);
  } else {
    std::string perfCoresStr = std::to_string(lbmini::CountPerformanceCores());
    setenv("ACC_NUM_CORES", perfCoresStr.c_str(), 1);
    setenv("POCL_MAX_PTHREAD_COUNT", perfCoresStr.c_str(), 1);
  }

  switch (performance.backend) {
    case lbmini::BackendEnum::Plain: {
      if (performance.target != lbmini::TargetEnum::CPU) {
        throw std::runtime_error("Plain backend only supports CPU target.");
      }
      using LbmLattice = lbmini::plain::LatticeD2Q9<Scalar>;
      using LbmTube = lbmini::plain::LbmTube<Scalar, LbmLattice>;
      lbmTube_.reset(reinterpret_cast<ILbmTubeT*>(new LbmTube(fluid, mesh, control, performance)));
      break;
    }
    case lbmini::BackendEnum::OpenMP: {
      #ifdef _OPENMP
      switch (performance.target) {
        case lbmini::TargetEnum::CPU: {
          omp_set_default_device(0);
          using LbmLattice = lbmini::openmp::cpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::openmp::cpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(reinterpret_cast<ILbmTubeT*>(new LbmTube(fluid, mesh, control, performance)));
          break;
        }
        case lbmini::TargetEnum::GPU: {
          if (omp_get_num_devices() < 1) {
            std::cerr << "No OpenMP target devices available." << std::endl;
            lbmTube_.reset(nullptr);
            return;
          }
          omp_set_default_device(0);
          using LbmLattice = lbmini::openmp::gpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::openmp::gpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(reinterpret_cast<ILbmTubeT*>(new LbmTube(fluid, mesh, control, performance)));
          break;
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
          lbmTube_.reset(reinterpret_cast<ILbmTubeT*>(new LbmTube(fluid, mesh, control, performance)));
          break;
        }
        case lbmini::TargetEnum::GPU: {
          acc_set_device_type(acc_device_default);
          using LbmLattice = lbmini::openacc::gpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::openacc::gpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(reinterpret_cast<ILbmTubeT*>(new LbmTube(fluid, mesh, control, performance)));
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
      switch (performance.target) {
        case lbmini::TargetEnum::CPU: {
          using LbmLattice = lbmini::opencl::cpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::opencl::cpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(reinterpret_cast<ILbmTubeT*>(new LbmTube(fluid, mesh, control, performance)));
          break;
        }
        case lbmini::TargetEnum::GPU: {
          using LbmLattice = lbmini::opencl::gpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::opencl::gpu::LbmTube<Scalar, LbmLattice>;
          lbmTube_.reset(reinterpret_cast<ILbmTubeT*>(new LbmTube(fluid, mesh, control, performance)));
          break;
        }
        default:
          throw std::runtime_error("Invalid target for OpenCL backend.");
      }
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
      lbmTube_.reset(reinterpret_cast<ILbmTubeT*>(new LbmTube(fluid, mesh, control, performance)));
      #else
      throw std::runtime_error("MPI backend not enabled.");
      #endif
      break;
    }
    case lbmini::BackendEnum::CUDA: {
      #ifdef __CUDACC__
      if (performance.target != lbmini::TargetEnum::GPU) {
        throw std::runtime_error("CUDA backend only supports GPU target.");
      }
      using LbmLattice = lbmini::cuda::gpu::LatticeD2Q9<Scalar>;
      using LbmTube = lbmini::cuda::gpu::LbmTube<Scalar, LbmLattice>;
      lbmTube_.reset(reinterpret_cast<ILbmTubeT*>(new LbmTube(fluid, mesh, control, performance)));
      #else
      throw std::runtime_error("CUDA backend not enabled (compile with nvcc).");
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
  const auto rho = lbmTube_->Rho();
  const auto p = lbmTube_->P();
  const auto u = lbmTube_->U();

  if (rank != 0)
    return;

  std::ofstream file(std::filesystem::path(outDir) / ("data_" + std::to_string(step) + ".csv"));
  file << "runtime,x,y,ux,uy,density,pressure\n";

  const Scalar dx = (mesh_.nx() > 1) ? mesh_.lx() / static_cast<Scalar>(mesh_.nx() - 1) : mesh_.lx();
  const Scalar dy = (mesh_.ny() > 1) ? mesh_.ly() / static_cast<Scalar>(mesh_.ny() - 1) : mesh_.ly();

  for (Index i = 0; i < mesh_.nx(); ++i) {
    const Scalar x_coord = 0.5 * dx + dx * i;
    for (Index j = 0; j < mesh_.ny(); ++j) {
      const Scalar y_coord = 0.5 * dy + dy * j;

      file << elapsed_seconds << ","
        << x_coord << ","
        << y_coord << ","
        << u(i, j, 0) << ","
        << u(i, j, 1) << ","
        << rho(i, j) << ","
        << p(i, j) << "\n";
    }
  }

  file.close();
}

template<typename Scalar>
void Executor<Scalar>::Run(const std::string& outpath) {
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

  const auto startRunTime = std::chrono::high_resolution_clock::now();

  const Index stepsPerOutput = std::max(kSteps / control_.printStep, Index(1));
  for (Index i = 0; i < kSteps;) {
    const Index stepsPerChunk = std::min(stepsPerOutput, kSteps - i);
    lbmTube_->Run(stepsPerChunk, true);
    i += stepsPerChunk;

    const auto current_time = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds = current_time - startRunTime;

    if (rank == 0) {
      std::cout << "Time step: " << i << std::endl;
    }
    OutputData(i, outpath, elapsed_seconds.count(), rank);
  }

  const auto end_time = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> elapsed_time = end_time - startRunTime;
  if (rank == 0) {
    std::cout << "Simulation completed. Total run time: " << elapsed_time.count() << " s" << std::endl;
  }
}

template class Executor<double>;
} // namespace lbmini
