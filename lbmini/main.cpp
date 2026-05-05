#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <chrono>
#include <omp.h>

#include "Data.hpp"
#include "Lbm/ILattice.hpp"
#include "Lbm/OpenMp/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenMp/Cpu/LbmTube.hpp"
#include "Lbm/OpenMp/Gpu/LatticeD2Q9.hpp"
#include "Lbm/OpenMp/Gpu/LbmTube.hpp"
#include "Lbm/Cuda/LatticeD2Q9.hpp"
#include "Lbm/Cuda/LbmTube.hpp"
#include "Lbm/Plain/LatticeD2Q9.hpp"
#include "Lbm/Plain/LbmTube.hpp"

using namespace Eigen;

template<typename Scalar, typename LbmTubeType>
void OutputData(
  const LbmTubeType& lbmTube,
  const lbmini::MeshData<Scalar, 2>& mesh,
  Index step,
  const std::string& outDir,
  const double elapsed_seconds
) {
  std::ofstream file(std::filesystem::path(outDir) / ("data_" + std::to_string(step) + ".csv"));
  file << "runtime,x,y,ux,uy,density,pressure\n";

  const Scalar dx = mesh.lx() / static_cast<Scalar>(mesh.nx() - 1);
  const Scalar dy = mesh.ly() / static_cast<Scalar>(mesh.ny() - 1); // Added dy calculation

  const auto& rho = lbmTube.Rho();
  const auto& p = lbmTube.P();
  const auto& u = lbmTube.U();

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
  OutputData(lbmTube, mesh, 0, outpath, 0.0);

  const Index stepsPerOutput = std::max(kSteps / control.printStep, Index(1));
  for (Index i = 0; i < kSteps;) {
    const Index stepsPerChunk = std::min(stepsPerOutput, kSteps - i);
    lbmTube.Run(stepsPerChunk, true);
    i += stepsPerChunk;

    std::cout << "Time step: " << i << std::endl;
    const auto current_time = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds = current_time - startRunTime;
    OutputData(lbmTube, mesh, i, outpath, elapsed_seconds.count());
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

  // Create unique output directory
  std::string outputPath = outputPathBase;
  Index counter = 1;
  while (std::filesystem::exists(outputPath) && !std::filesystem::is_empty(outputPath))
    outputPath = outputPathBase + std::to_string(counter++);

  std::filesystem::create_directory(outputPath);
  std::cout << "Outputting to directory: " << outputPath << std::endl;

  // Copy config file
  std::filesystem::copy(configPath, std::filesystem::path(outputPath) / "config.yaml");

  // Read simulation settings from YAML file
  auto [fluid, mesh, control, performance] = lbmini::ReadYaml<Scalar, 2>(configPath);

  // Calculate the physical time step
  const Scalar cs2 = Scalar(1.0) / Scalar(3.0);
  const Scalar dx = mesh.lx() / static_cast<Scalar>(mesh.nx());
  const Scalar uPhysRef = std::sqrt(fluid.gamma * fluid.pressureL / fluid.densityL);
  const Scalar uLuRef = std::sqrt(fluid.gamma * cs2);
  const Scalar dt = dx * (uLuRef / uPhysRef);
  const Index kSteps = static_cast<Index>(control.tmax / dt);

  const auto start_time = std::chrono::high_resolution_clock::now();

  // LBM simulation
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
        default: {
          std::cerr << "Invalid backend selected for CPU." << std::endl;
          return 1;
        }
      };
      break;
    }

    case lbmini::TargetEnum::GPU: {
      switch (performance.backend) {
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
        case lbmini::BackendEnum::CUDA: {
          std::cout << "Using CUDA backend on GPU." << std::endl;
          using LbmLattice = lbmini::cuda::LatticeD2Q9<Scalar>;
          using LbmTube = lbmini::cuda::LbmTube<Scalar, LbmLattice>;
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
  std::cout << "Simulation completed. Total run time: " << elapsed_time.count() << " s" << std::endl;

  return 0;
}
