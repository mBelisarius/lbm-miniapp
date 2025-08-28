#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <chrono>

#include "Data.hpp"
#include "Lbm/OpenMp/LbmD2Q9.hpp"
#include "Lbm/OpenMp/LbmTube.hpp"
#include "Lbm/Plain/LbmD2Q9.hpp"
#include "Lbm/Plain/LbmTube.hpp"

using namespace Eigen;

// output data along x slice (y=z=middle)
template<typename Scalar_, typename LbmTubeType>
void OutputData(
  const LbmTubeType& lbmTube,
  const lbmini::MeshData<Scalar_, 2>& mesh,
  Index step,
  const std::string& outDir,
  const double elapsed_seconds
) {
  std::ofstream file(std::filesystem::path(outDir) / ("data_" + std::to_string(step) + ".csv"));
  file << "runtime,x,ux,uy,density,pressure\n";

  const Scalar_ dx = mesh.lx() / static_cast<Scalar_>(mesh.nx() - 1);
  const auto& rho = lbmTube.Rho();
  const auto& p = lbmTube.P();
  const auto& u = lbmTube.U();

  const Index y_slice = mesh.ny() / 2;

  for (Index i = 0; i < mesh.nx(); ++i) {
    file << elapsed_seconds << ","
      << 0.5 * dx + dx * i << "," << u(i, y_slice, 0) << "," << u(i, y_slice, 1) << ","
      << rho(i, y_slice) << "," << p(i, y_slice)
      << "\n";
  }

  file.close();
}

template<typename Scalar_, typename LbmClassType, typename LbmTubeType>
void run(
  const lbmini::FluidData<Scalar_>& fluid,
  const lbmini::MeshData<Scalar_, LbmClassType::Dim()>& mesh,
  const lbmini::ControlData<Scalar_>& control,
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
  for (Index i = 0; i < kSteps; ) {
    const Index stepsPerChunk = std::min(stepsPerOutput, kSteps - i);
    lbmTube.Run(stepsPerChunk);
    i += stepsPerChunk;

    std::cout << "Time step: " << i << std::endl;
    const auto current_time = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds = current_time - startRunTime;
    OutputData(lbmTube, mesh, i, outpath, elapsed_seconds.count());
  }
}

int main(int argc, char* argv[]) {
  using Scalar = double;

  std::string config_path = "config.yaml";
  std::string output_path_base = "out";
  for (int i = 1; i < argc; ++i) {
    if (std::string arg = argv[i]; arg == "--config") {
      if (i + 1 < argc) {
        config_path = argv[++i];
      }
    } else if (arg == "--outpath") {
      if (i + 1 < argc) {
        output_path_base = argv[++i];
      }
    }
  }

  // Create unique output directory
  std::string output_path = output_path_base;
  Index counter = 1;
  while (std::filesystem::exists(output_path) && !std::filesystem::is_empty(output_path))
    output_path = output_path_base + std::to_string(counter++);

  std::filesystem::create_directory(output_path);
  std::cout << "Outputting to directory: " << output_path << std::endl;

  // Copy config file
  std::filesystem::copy(config_path, std::filesystem::path(output_path) / "config.yaml");

  // Read simulation settings from YAML file
  auto [fluid, mesh, control, performance] = lbmini::ReadYaml<Scalar, 2>(config_path);

  // Calculate the physical time step
  const Scalar cs2 = Scalar(1.0) / Scalar(3.0);
  const Scalar dx = mesh.lx() / static_cast<Scalar>(mesh.nx());
  const Scalar uPhysRef = std::sqrt(fluid.gamma * fluid.pressureL / fluid.densityL);
  const Scalar uLuRef = std::sqrt(fluid.gamma * cs2);
  const Scalar dt = dx * (uLuRef / uPhysRef);
  const Index kSteps = static_cast<Index>(control.tmax / dt);

  const auto start_time = std::chrono::high_resolution_clock::now();

  // LBM simulation
  if (performance.backend == 0) {
    std::cout << "Using plain backend." << std::endl;
    using LbmLattice = lbmini::plain::LbmD2Q9<Scalar>;
    using LbmTube = lbmini::plain::LbmTube<Scalar, LbmLattice>;
    run<Scalar, LbmLattice, LbmTube>(
      fluid,
      mesh,
      control,
      performance,
      output_path,
      kSteps,
      start_time
    );
  } else if (performance.backend == 1) {
    std::cout << "Using OpenMP backend." << std::endl;
    using LbmLattice = lbmini::openmp::LbmD2Q9<Scalar>;
    using LbmTube = lbmini::openmp::LbmTube<Scalar, LbmLattice>;
    run<Scalar, LbmLattice, LbmTube>(
      fluid,
      mesh,
      control,
      performance,
      output_path,
      kSteps,
      start_time
    );
  } else {
    std::cerr << "Invalid backend selected" << std::endl;
    return 1;
  }

  const auto end_time = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> elapsed_time = end_time - start_time;
  std::cout << "Simulation completed. Total run time: " << elapsed_time.count() << " s" << std::endl;

  return 0;
}
