#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "Data/DataReader.hpp"
#include "lbmini-plain/LbmD3Q19.hpp"
#include "lbmini-plain/LbmTube.hpp"

using namespace Eigen;

// output data along x slice (y=z=middle)
template <typename Scalar_, typename LbmClassType>
void output_data(const lbmini::LbmTube<Scalar_, LbmClassType>& lbmTube,
                 const Vector<Index, LbmClassType::Dim()>& sizes,
                 Index step) {
  static std::string outDirStr;
  if (outDirStr.empty()) {
    const std::string outDirBase = "out";
    outDirStr = outDirBase;
    Index counter = 1;
    while (std::filesystem::exists(outDirStr) && !std::filesystem::is_empty(outDirStr))
      outDirStr = outDirBase + std::to_string(counter++);

    std::filesystem::create_directory(outDirStr);
    std::cout << "Outputting to directory: " << outDirStr << std::endl;
  }

  std::ofstream file(std::filesystem::path(outDirStr) / ("data_" + std::to_string(step) + ".csv"));
  file << "x,ux,uy,uz,pressure,density\n";

  const auto& rho = lbmTube.Rho();
  const auto& p = lbmTube.P();
  const auto& u = lbmTube.U();

  const Index y_slice = sizes[1] / 2;
  const Index z_slice = sizes[2] / 2;

  for (Index x = 0; x < sizes[0]; ++x) {
    file << x / static_cast<Scalar_>(sizes[0]) << ","
         << u(x, y_slice, z_slice, 0) << "," << u(x, y_slice, z_slice, 1) << ","
         << u(x, y_slice, z_slice, 2) << "," << p(x, y_slice, z_slice) << ","
         << rho(x, y_slice, z_slice) << "\n";
  }
  file.close();
}

int main() {
  using Scalar = double;

  // Read simulation settings from YAML file
  auto [fluidProps, meshData, controlData] =
    lbmini::ReadYaml<Scalar, 3>("../lbmini/system.yaml");

  const Scalar cs2 = 1.0 / 3.0;
  const Scalar dx = 1.0 / static_cast<Scalar>(meshData.nx());
  const Scalar uPhysRefL = std::sqrt(fluidProps.gamma * fluidProps.pressureL / fluidProps.densityL);
  const Scalar uPhysRefR = std::sqrt(fluidProps.gamma * fluidProps.pressureR / fluidProps.densityR);
  const Scalar uPhysRef = uPhysRefR;
  const Scalar uLuRef = std::sqrt(cs2);
  const Scalar dt = uLuRef * (dx / uPhysRef);
  const Index kSteps = static_cast<Index>(controlData.tmax / dt);

  std::cout << "dt: " << dt << " | steps: " << kSteps << std::endl;

  static lbmini::LbmD3Q19<Scalar> lbmClass;
  lbmini::LbmTube lbmTube(lbmClass, meshData.size, fluidProps);

  lbmTube.Init();
  output_data(lbmTube, meshData.size, 0);

  for (Index i = 1; i <= kSteps; ++i) {
    lbmTube.Step(dt);
    if ((i % std::max(kSteps / 1000, Index(1)) == 0) || i == kSteps) {
      std::cout << "Time step: " << i << std::endl;
      output_data(lbmTube, meshData.size, i);
    }
  }

  std::cout << "Simulation completed." << std::endl;
  return 0;
}
