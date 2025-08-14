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
  using Index = Eigen::Index;

  YAML::Node config = YAML::LoadFile(filename);

  FluidData<Scalar_> fluid;
  fluid.densityL = config["Fluid"]["densityL"].as<Scalar_>();
  fluid.pressureL = config["Fluid"]["pressureL"].as<Scalar_>();
  fluid.densityR = config["Fluid"]["densityR"].as<Scalar_>();
  fluid.pressureR = config["Fluid"]["pressureR"].as<Scalar_>();
  fluid.viscosity = config["Fluid"]["viscosity"].as<Scalar_>();
  fluid.prandtl = config["Fluid"]["prandtl"].as<Scalar_>();
  fluid.gamma = config["Fluid"]["gamma"].as<Scalar_>();
  fluid.specificHeatCv = Scalar_(1.0) / (fluid.gamma - Scalar_(1.0));
  fluid.specificHeatCp = fluid.specificHeatCv + Scalar_(1.0);
  fluid.constant = fluid.specificHeatCp - fluid.specificHeatCv;
  fluid.conductivity = fluid.specificHeatCp * fluid.viscosity / fluid.prandtl;

  MeshData<Scalar_, dim_> mesh;
  mesh.lenght[0] = config["Mesh"]["lx"].as<Scalar_>();
  mesh.size[0] = config["Mesh"]["nx"].as<Index>();

  ControlData<Scalar_> control;
  control.tmax = config["Control"]["tmax"].as<Scalar_>();
  control.Ux = config["Control"]["Ux"].as<Scalar_>();
  control.idw = config["Control"]["idw"].as<Scalar_>();
  control.printStep = config["Control"]["printStep"].as<Index>();

  if (dim_ >= 2) {
    mesh.lenght[1] = config["Mesh"]["ly"].as<Scalar_>();
    mesh.size[1] = config["Mesh"]["ny"].as<Index>();

    control.Uy = config["Control"]["Uy"].as<Scalar_>();
  }

  if (dim_ >= 3) {
    mesh.lenght[2] = config["Mesh"]["lz"].as<Scalar_>();
    mesh.size[2] = config["Mesh"]["nz"].as<Index>();

    control.Uz = config["Control"]["Uz"].as<Scalar_>();
  }

  return {fluid, mesh, control};
}

}  // namespace lbmini

#endif  // LBMINI_DATA_READER_HPP_
