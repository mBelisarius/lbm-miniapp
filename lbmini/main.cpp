#undef _GLIBCXX_ASSERTIONS
#define NDEBUG

#if defined(__NVCOMPILER)
namespace std {
void __glibcxx_assert_fail(const char*, int, const char*, const char*) {}
}
#pragma acc routine(std::__glibcxx_assert_fail) seq
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <unistd.h>
#include <vector>
#include <chrono>
#include <cmath>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Lbm/CountCores.hpp"
#include "Lbm/ILattice.hpp"
#include "Lbm/Cuda/Gpu/LatticeD2Q9.hpp"
#include "Lbm/Cuda/Gpu/LbmTube.hpp"
#include "Lbm/OpenAcc/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenAcc/Cpu/LbmTube.hpp"
#include "Lbm/OpenAcc/Gpu/LatticeD2Q9.hpp"
#include "Lbm/OpenAcc/Gpu/LbmTube.hpp"
#include "Lbm/OpenCl/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenCl/Cpu/LbmTube.hpp"
#include "Lbm/OpenCl/Gpu/LatticeD2Q9.hpp"
#include "Lbm/OpenCl/Gpu/LbmTube.hpp"
#include "Lbm/OpenMp/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenMp/Cpu/LbmTube.hpp"
#include "Lbm/OpenMp/Gpu/LatticeD2Q9.hpp"
#include "Lbm/OpenMp/Gpu/LbmTube.hpp"
#include "Lbm/Plain/LatticeD2Q9.hpp"
#include "Lbm/Plain/LbmTube.hpp"

#ifdef USE_MPI
#include <mpi.h>
#include "Lbm/Mpi/Cpu/LatticeD2Q9.hpp"
#include "Lbm/Mpi/Cpu/LbmTube.hpp"
#endif

// POCL (Portable Computing Language) uses a library constructor to read environment
// variables like POCL_MAX_PTHREAD_COUNT upon library load.
// Standard setenv() calls in main() execute too late. This constructor runs beforehand.
__attribute__((constructor))
void InitPoclEnvironment() {
    std::string perfCoresStr = std::to_string(lbmini::CountPerformanceCores());
    setenv("POCL_MAX_PTHREAD_COUNT", perfCoresStr.c_str(), 0);
    // Explicitly restrict process affinity to P-cores instead of relying on POCL_AFFINITY
    lbmini::SetProcessToPerformanceCores();
}

using namespace Eigen;

template<typename Scalar, typename LbmTubeType>
void OutputData(
  const LbmTubeType& lbmTube,
  const lbmini::MeshData<Scalar, 2>& mesh,
  Index step,
  const std::string& outDir,
  const double elapsed_seconds,
  const int rank
) {
  const auto rho = lbmTube.Rho();
  const auto p = lbmTube.P();
  const auto u = lbmTube.U();

  if (rank != 0) return;

  std::ofstream file(std::filesystem::path(outDir) / ("data_" + std::to_string(step) + ".csv"));
  file << "runtime,x,y,ux,uy,density,pressure\n";

  const Scalar dx = (mesh.nx() > 1) ? mesh.lx() / static_cast<Scalar>(mesh.nx() - 1) : mesh.lx();
  const Scalar dy = (mesh.ny() > 1) ? mesh.ly() / static_cast<Scalar>(mesh.ny() - 1) : mesh.ly();

  for (Index i = 0; i < mesh.nx(); ++i) {
    const Scalar x_coord = 0.5 * dx + dx * i;
    for (Index j = 0; j < mesh.ny(); ++j) {
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

template<typename Scalar, typename LbmTubeType>
void run(
  const lbmini::FluidData<Scalar>& fluid,
  const lbmini::MeshData<Scalar, 2>& mesh,
  const lbmini::ControlData<Scalar>& control,
  const lbmini::PerformanceData& performance,
  const std::string& outpath,
  const Index kSteps,
  const std::chrono::high_resolution_clock::time_point& startRunTime
) {
  // LBM simulation
  LbmTubeType lbmTube(fluid, mesh, control, performance);
  lbmTube.Init();
  
  int rank = 0;
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  OutputData(lbmTube, mesh, 0, outpath, 0.0, rank);

  const Index stepsPerOutput = std::max(kSteps / control.printStep, Index(1));
  for (Index i = 0; i < kSteps;) {
    const Index stepsPerChunk = std::min(stepsPerOutput, kSteps - i);
    lbmTube.Run(stepsPerChunk, true);
    i += stepsPerChunk;

    const auto current_time = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds = current_time - startRunTime;
    
    if (rank == 0) {
      std::cout << "Time step: " << i << std::endl;
    }
    OutputData(lbmTube, mesh, i, outpath, elapsed_seconds.count(), rank);
  }
}

int main(int argc, char* argv[]) {
  using Scalar = double;

  std::string configPath = "config.yaml";
  std::string outputPathBase = "out";
  for (int i = 1; i < argc; ++i) {
    if (std::string arg = argv[i]; arg == "--config") {
      if (i + 1 < argc) {
        configPath = argv[++i];
      }
    } else if (arg == "--outpath") {
      if (i + 1 < argc) {
        outputPathBase = argv[++i];
      }
    }
  }

  // Read simulation settings from YAML file
  auto [fluid, mesh, control, performance] = lbmini::ReadYaml<Scalar, 2>(configPath);

  bool use_mpi = (performance.backend == lbmini::BackendEnum::MPI);

#ifdef USE_MPI
  if (use_mpi) {
    bool is_under_mpi = std::getenv("OMPI_COMM_WORLD_SIZE") != nullptr ||
                        std::getenv("PMI_RANK") != nullptr ||
                        std::getenv("PMIX_RANK") != nullptr;
    
    if (!is_under_mpi) {
      int num_procs = performance.cores > 0 ? performance.cores : lbmini::CountPerformanceCores();
      std::string num_procs_str = std::to_string(num_procs);
      
      std::vector<char*> args;
      args.push_back(const_cast<char*>("mpirun"));
      args.push_back(const_cast<char*>("--use-hwthread-cpus"));
      args.push_back(const_cast<char*>("--oversubscribe"));
      args.push_back(const_cast<char*>("-np"));
      args.push_back(const_cast<char*>(num_procs_str.data()));
      for (int i = 0; i < argc; ++i) {
        args.push_back(argv[i]);
      }
      args.push_back(nullptr);
      
      execvp("mpirun", args.data());
      
      std::cerr << "Failed to automatically launch mpirun. Please run with mpirun manually." << std::endl;
      return 1;
    }
  }
#endif

  int rank = 0;
#ifdef USE_MPI
  if (use_mpi) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
#endif

  // Create unique output directory
  std::string outputPath = outputPathBase;
  
  if (rank == 0) {
    Index counter = 1;
    while (std::filesystem::exists(outputPath) && !std::filesystem::is_empty(outputPath))
      outputPath = outputPathBase + std::to_string(counter++);

    std::filesystem::create_directory(outputPath);
    std::cout << "Outputting to directory: " << outputPath << std::endl;

    // Copy config file
    std::filesystem::copy(configPath, std::filesystem::path(outputPath) / "config.yaml");
  }
  
#ifdef USE_MPI
  if (use_mpi) {
    // Broadcast the chosen output directory path to all ranks, just in case
    int pathLen = outputPath.size();
    MPI_Bcast(&pathLen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) outputPath.resize(pathLen);
    MPI_Bcast(outputPath.data(), pathLen, MPI_CHAR, 0, MPI_COMM_WORLD);
  }
#endif

  // Calculate the physical time step
  const Scalar cs2 = Scalar(1.0) / Scalar(3.0);
  const Scalar dx = mesh.lx() / static_cast<Scalar>(mesh.nx());
  const Scalar uPhysRef = std::sqrt(fluid.gamma * fluid.pressureL / fluid.densityL);
  const Scalar uLuRef = std::sqrt(fluid.gamma * cs2);
  const Scalar dt = dx * (uLuRef / uPhysRef);
  const Index kSteps = static_cast<Index>(control.tmax / dt);

  const auto start_time = std::chrono::high_resolution_clock::now();

  // LBM simulation
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
  }

  switch (performance.target) {
    case lbmini::TargetEnum::CPU: {
      switch (performance.backend) {
        case lbmini::BackendEnum::Plain: {
          std::cout << "Using plain backend." << std::endl;
          using LbmLattice = lbmini::plain::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::plain::LbmTube<Scalar, LbmLattice>;
          run<Scalar, LbmTube>(
            fluid,
            mesh,
            control,
            performance,
            outputPath,
            kSteps,
            start_time
          );
          break;
        }
        case lbmini::BackendEnum::OpenMP: {
          std::cout << "Using OpenMP backend on CPU." << std::endl;
          omp_set_default_device(0);
          using LbmLattice = lbmini::openmp::cpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::openmp::cpu::LbmTube<Scalar, LbmLattice>;
          run<Scalar, LbmTube>(
            fluid,
            mesh,
            control,
            performance,
            outputPath,
            kSteps,
            start_time
          );
          break;
        }
        case lbmini::BackendEnum::OpenACC: {
          std::cout << "Using OpenACC backend on CPU." << std::endl;
#if defined(_OPENACC)
          acc_set_device_type(acc_device_host);
#endif
          using LbmLattice = lbmini::openacc::cpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::openacc::cpu::LbmTube<Scalar, LbmLattice>;
          run<Scalar, LbmTube>(
            fluid,
            mesh,
            control,
            performance,
            outputPath,
            kSteps,
            start_time
          );
          break;
        }
        case lbmini::BackendEnum::CUDA: {
          throw std::runtime_error("Backend not available for CPU.");
        }
        case lbmini::BackendEnum::OpenCL: {
          std::cout << "Using OpenCL backend on CPU." << std::endl;
          using LbmLattice = lbmini::opencl::cpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::opencl::cpu::LbmTube<Scalar, LbmLattice>;
          run<Scalar, LbmTube>(
            fluid,
            mesh,
            control,
            performance,
            outputPath,
            kSteps,
            start_time
          );
          break;
        }
        case lbmini::BackendEnum::Kokkos:
        case lbmini::BackendEnum::RAJA: {
          throw std::runtime_error("Backend not yet implemented for CPU.");
        }
#ifdef USE_MPI
        case lbmini::BackendEnum::MPI: {
          if (rank == 0) std::cout << "Using MPI backend on CPU." << std::endl;
          using LbmLattice = lbmini::mpi::cpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::mpi::cpu::LbmTube<Scalar, LbmLattice>;
          run<Scalar, LbmTube>(
            fluid,
            mesh,
            control,
            performance,
            outputPath,
            kSteps,
            start_time
          );
          break;
        }
#else
        case lbmini::BackendEnum::MPI: {
          throw std::runtime_error("MPI Backend not compiled for CPU.");
        }
#endif
        default: {
          std::cerr << "Invalid backend selected for CPU." << std::endl;
          return 1;
        }
      };
      break;
    }

    case lbmini::TargetEnum::GPU: {
      switch (performance.backend) {
        case lbmini::BackendEnum::Plain: {
          throw std::runtime_error("Backend not available for GPU.");
        }
        case lbmini::BackendEnum::OpenMP: {
          std::cout << "Using OpenMP backend on GPU." << std::endl;
          if (omp_get_num_devices() < 1) {
            std::cerr << "No OpenMP target devices available." << std::endl;
            return 1;
          }
          // Use the first available GPU; do not force device 1 (may not exist).
          omp_set_default_device(0);
          std::cout << "  Target device: " << omp_get_default_device()
            << " / " << omp_get_num_devices() << std::endl;
          using LbmLattice = lbmini::openmp::gpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::openmp::gpu::LbmTube<Scalar, LbmLattice>;
          run<Scalar, LbmTube>(
            fluid,
            mesh,
            control,
            performance,
            outputPath,
            kSteps,
            start_time
          );
          break;
        }
        case lbmini::BackendEnum::OpenACC: {
          std::cout << "Using OpenACC backend on GPU." << std::endl;
#if defined(_OPENACC)
#if 0
          acc_set_device_type(acc_device_nvidia);
#else
          acc_set_device_type(acc_device_default);
#endif
#endif
          using LbmLattice = lbmini::openacc::gpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::openacc::gpu::LbmTube<Scalar, LbmLattice>;
          run<Scalar, LbmTube>(
            fluid,
            mesh,
            control,
            performance,
            outputPath,
            kSteps,
            start_time
          );
          break;
        }
        case lbmini::BackendEnum::CUDA: {
          std::cout << "Using CUDA backend on GPU." << std::endl;
          using LbmLattice = lbmini::cuda::gpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::cuda::gpu::LbmTube<Scalar, LbmLattice>;
          run<Scalar, LbmTube>(
            fluid,
            mesh,
            control,
            performance,
            outputPath,
            kSteps,
            start_time
          );
          break;
        }
        case lbmini::BackendEnum::OpenCL: {
          std::cout << "Using OpenCL backend on GPU." << std::endl;
          using LbmLattice = lbmini::opencl::gpu::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::opencl::gpu::LbmTube<Scalar, LbmLattice>;
          run<Scalar, LbmTube>(
            fluid,
            mesh,
            control,
            performance,
            outputPath,
            kSteps,
            start_time
          );
          break;
        }
        case lbmini::BackendEnum::Kokkos:
        case lbmini::BackendEnum::RAJA:
        case lbmini::BackendEnum::MPI: {
          throw std::runtime_error("Backend not yet implemented for GPU.");
        }
        default: {
          std::cerr << "Invalid backend selected for GPU." << std::endl;
          return 1;
        }
      };
      break;
    }

    default: {
      std::cerr << "Invalid target selected." << std::endl;
      return 1;
    }
  };

  const auto end_time = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> elapsed_time = end_time - start_time;
  if (rank == 0) {
    std::cout << "Simulation completed. Total run time: " << elapsed_time.count() << " s" << std::endl;
  }

#ifdef USE_MPI
  if (use_mpi) {
    MPI_Finalize();
  }
#endif

  return 0;
}
