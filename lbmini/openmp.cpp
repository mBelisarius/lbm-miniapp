#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "Data.hpp"
#include "Lbm/OpenMp/LbmD2Q9.hpp"
#include "Lbm/OpenMp/LbmTube.hpp"

using namespace Eigen;

// output data along x slice (y=z=middle)
template <typename Scalar_, typename LbmClassType>
void output_data(const lbmini::openmp::LbmTube<Scalar_, LbmClassType>& lbmTube,
                 const lbmini::MeshData<Scalar_, 2>& mesh,
                 Index step,
                 const std::string& outDir) {
  std::ofstream file(std::filesystem::path(outDir) / ("data_" + std::to_string(step) + ".csv"));
  file << "x,ux,uy,density,pressure\n";

  const Scalar_ dx = mesh.lx() / static_cast<Scalar_>(mesh.nx() - 1);
  const auto& rho = lbmTube.Rho();
  const auto& p = lbmTube.P();
  const auto& u = lbmTube.U();

  const Index y_slice = mesh.ny() / 2;

  for (Index i = 0; i < mesh.nx(); ++i) {
    file << 0.5 * dx + dx * i << ","
         << u(i, y_slice, 0) << "," << u(i, y_slice, 1) << ","
         << rho(i, y_slice) << "," << p(i, y_slice) << "\n";
  }
  file.close();
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

  // LBM simulation
  lbmini::openmp::LbmTube<Scalar, lbmini::openmp::LbmD2Q9<Scalar>> lbmTube(fluid, mesh, control, performance);
  lbmTube.Init();
  output_data(lbmTube, mesh, 0, output_path);

  for (Index i = 1; i <= kSteps; ++i) {
    lbmTube.Step();
    if ((i % std::max(kSteps / control.printStep, Index(1)) == 0) || i == kSteps) {
      std::cout << "Time step: " << i << std::endl;
      output_data(lbmTube, mesh, i, output_path);
    }
  }

  std::cout << "Simulation completed." << std::endl;
  return 0;
}
