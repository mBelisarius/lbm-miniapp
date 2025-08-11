#ifndef LBMINI_DATA_READER_HPP_
#define LBMINI_DATA_READER_HPP_

#include <yaml-cpp/yaml.h>

#include <tuple>

#include "Data/ControlData.hpp"
#include "Data/FluidData.hpp"
#include "Data/MeshData.hpp"

namespace lbmini {

template <typename Scalar_, Eigen::Index dim_>
std::tuple<FluidData<Scalar_>, MeshData<Scalar_, dim_>, ControlData<Scalar_>>
ReadYaml(const std::string& filename) {
  YAML::Node config = YAML::LoadFile(filename);

  FluidData<Scalar_> fluid;
  fluid.densityL = config["Fluid"]["densityL"].as<Scalar_>();
  fluid.pressureL = config["Fluid"]["pressureL"].as<Scalar_>();
  fluid.temperatureL = config["Fluid"]["temperatureL"].as<Scalar_>();
  fluid.densityR = config["Fluid"]["densityR"].as<Scalar_>();
  fluid.pressureR = config["Fluid"]["pressureR"].as<Scalar_>();
  fluid.temperatureR = config["Fluid"]["temperatureR"].as<Scalar_>();
  fluid.viscosity = config["Fluid"]["viscosity"].as<Scalar_>();
  fluid.specificHeat = config["Fluid"]["specificHeat"].as<Scalar_>();
  fluid.gamma = config["Fluid"]["gamma"].as<Scalar_>();
  fluid.constant = config["Fluid"]["constant"].as<Scalar_>();

  MeshData<Scalar_, dim_> mesh;
  mesh.lenght[0] = config["Mesh"]["lx"].as<Scalar_>();
  mesh.lenght[1] = config["Mesh"]["ly"].as<Scalar_>();
  mesh.lenght[2] = config["Mesh"]["lz"].as<Scalar_>();
  mesh.size[0] = config["Mesh"]["nx"].as<Eigen::Index>();
  mesh.size[1] = config["Mesh"]["ny"].as<Eigen::Index>();
  mesh.size[2] = config["Mesh"]["nz"].as<Eigen::Index>();

  ControlData<Scalar_> control;
  control.tmax = config["Control"]["tmax"].as<Scalar_>();

  return {fluid, mesh, control};
}

}  // namespace lbmini

#endif  // LBMINI_DATA_READER_HPP_
