#ifndef LBMINI_DATA_DATA_READER_HPP_
#define LBMINI_DATA_DATA_READER_HPP_

#include <yaml-cpp/yaml.h>

#include <tuple>

#include "Data/ControlData.hpp"
#include "Data/DataEnums.hpp"
#include "Data/FluidData.hpp"
#include "Data/MeshData.hpp"
#include "Data/PerformanceData.hpp"

namespace lbmini {
template<typename Scalar, Eigen::Index Dim>
std::tuple<FluidData<Scalar>, MeshData<Scalar, Dim>, ControlData<Scalar>, PerformanceData>
ReadYaml(const std::string& filename) {
  using Index = Eigen::Index;

  YAML::Node config = YAML::LoadFile(filename);

  FluidData<Scalar> fluid{};
  fluid.densityL = config["Fluid"]["densityL"].as<Scalar>();
  fluid.pressureL = config["Fluid"]["pressureL"].as<Scalar>();
  fluid.densityR = config["Fluid"]["densityR"].as<Scalar>();
  fluid.pressureR = config["Fluid"]["pressureR"].as<Scalar>();
  fluid.viscosity = config["Fluid"]["viscosity"].as<Scalar>();
  fluid.prandtl = config["Fluid"]["prandtl"].as<Scalar>();
  fluid.gamma = config["Fluid"]["gamma"].as<Scalar>();
  fluid.specificHeatCv = Scalar(1.0) / (fluid.gamma - Scalar(1.0));
  fluid.specificHeatCp = fluid.specificHeatCv + Scalar(1.0);
  fluid.constant = fluid.specificHeatCp - fluid.specificHeatCv;
  fluid.conductivity = fluid.specificHeatCp * fluid.viscosity / fluid.prandtl;

  MeshData<Scalar, Dim> mesh{};
  mesh.lenght[0] = config["Mesh"]["lx"].as<Scalar>();
  mesh.size[0] = config["Mesh"]["nx"].as<Index>();

  ControlData<Scalar> control{};
  control.tmax = config["Control"]["tmax"].as<Scalar>();
  control.Ux = config["Control"]["Ux"].as<Scalar>();
  control.idw = config["Control"]["idw"].as<Scalar>();
  control.printStep = config["Control"]["printStep"].as<Index>();

  PerformanceData performance{};
  performance.target = config["Performance"]["target"].as<TargetEnum>();
  performance.backend = config["Performance"]["backend"].as<BackendEnum>();
  performance.cores = config["Performance"]["cores"].as<Index>();

  if (Dim >= 2) {
    mesh.lenght[1] = config["Mesh"]["ly"].as<Scalar>();
    mesh.size[1] = config["Mesh"]["ny"].as<Index>();

    control.Uy = config["Control"]["Uy"].as<Scalar>();
  }

  if (Dim >= 3) {
    mesh.lenght[2] = config["Mesh"]["lz"].as<Scalar>();
    mesh.size[2] = config["Mesh"]["nz"].as<Index>();

    control.Uz = config["Control"]["Uz"].as<Scalar>();
  }

  return { fluid, mesh, control, performance };
}
} // namespace lbmini

#endif  // LBMINI_DATA_DATA_READER_HPP_
