#define SCALAR double

#include <blas/common.h>

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "FluidProperties.hpp"
#include "LbmD3Q19.hpp"
#include "LbmTube.hpp"

using namespace Eigen;

//------------------------------------------------------------------------------
// Output density field (for a slice at y = sizes[1]/2 and z = sizes[2]/2)
template <typename Scalar_, typename LbmClassType>
void output_density(const Tensor<Scalar_, LbmClassType::Dim()>& rho, const Vector<Index, LbmClassType::Dim()>& sizes, Index step) {
  std::ofstream file("./out/density_" + std::to_string(step) + ".dat");
  for (Index x = 0; x < sizes[0]; ++x) {
    file << x << " " << rho(x, sizes[1] / 2, sizes[2] / 2) << "\n";
  }
  file.close();
}

int main() {
  // Domain parameters
  constexpr Index kNx = 1000;
  constexpr Index kNy = 1;
  constexpr Index kNz = 1;
  constexpr Scalar kDt = 1.0e-4;
  constexpr Index kSteps = 0.2 / kDt;
  const auto sizes = Vector<Index, 3> { kNx, kNy, kNz };

  // Fluid properties for the shock tube:
  lbmini::FluidProperties<double> fluidProps { };
  fluidProps.densityL = 1.0;
  fluidProps.pressureL = 1.0;
  fluidProps.densityR = 0.125;
  fluidProps.pressureR = 0.1;
  fluidProps.viscosity = 0.01;
  fluidProps.specificHeat = 1.0;
  fluidProps.gamma = 1.4;

  // LBM method
  static lbmini::LbmD3Q19<double> lbmClass;
  lbmini::LbmTube lbmTube(lbmClass, sizes, fluidProps);

  lbmTube.Init();
  output_density<double, lbmini::LbmD3Q19<double>>(lbmTube.Rho(), sizes, 0);

  for (Index i = 1; i < kSteps + 1; ++i) {
    lbmTube.Collision(kDt);
    // lbmTube.RegularizeMomentum();
    // lbmTube.CollisionEnergy();
    // lbmTube.RegularizeEnergy();
    lbmTube.Streaming();
    // lbmTube.StreamingEnergy();
    lbmTube.ComputeMacroscopic();
    if (i % 10 == 0 && i > 0) {
      std::cout << "Time step: " << i << std::endl;
      output_density<double, lbmini::LbmD3Q19<double>>(lbmTube.Rho(), sizes, i);
    }
  }

  std::cout << "Simulation completed." << std::endl;
  return 0;
}
