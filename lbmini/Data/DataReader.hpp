#ifndef LBMINI_DATA_DATA_READER_HPP_
#define LBMINI_DATA_DATA_READER_HPP_

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <tuple>

#include "Guards.hpp"
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

  if (filename.empty()) {
    throw std::invalid_argument("Configuration file path is empty.");
  }
  if (!std::filesystem::exists(filename)) {
    throw std::invalid_argument("Configuration file does not exist: " + filename);
  }
  if (std::filesystem::is_directory(filename)) {
    throw std::invalid_argument("Configuration path is a directory, not a file: " + filename);
  }

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
  control.idw = sqrt(fluid.gamma);
  control.printStep = config["Control"]["printStep"].as<Index>();

  PerformanceData performance{};
  performance.target = config["Performance"]["target"].as<TargetEnum>();
  performance.backend = config["Performance"]["backend"].as<BackendEnum>();
  performance.cores = config["Performance"]["cores"].as<Index>();

  if constexpr (Dim >= 2) {
    mesh.lenght[1] = config["Mesh"]["ly"].as<Scalar>();
    mesh.size[1] = config["Mesh"]["ny"].as<Index>();

    control.Uy = config["Control"]["Uy"].as<Scalar>();
  }

  if constexpr (Dim >= 3) {
    mesh.lenght[2] = config["Mesh"]["lz"].as<Scalar>();
    mesh.size[2] = config["Mesh"]["nz"].as<Index>();

    control.Uz = config["Control"]["Uz"].as<Scalar>();
  }

  // Fail here, loudly, rather than let a hand-edited `config.yaml` propagate a
  // zero, negative or absurd extent into `N = nx * ny`, into the allocation
  // sizes and into the index arithmetic of whichever backend is selected.
  // `kSpeeds` is the D2Q9 stencil shared by every backend of this mini-app;
  // it only takes part in the streaming-table overflow check.
  // See docs/memory-safety.md §5.5.
  constexpr Index kSpeeds = 9;
  Index ny = Index{ 1 };
  if constexpr (Dim >= 2)
    ny = mesh.size[1];
  ValidateMeshExtents(mesh.size[0], ny, kSpeeds, filename.c_str());

  if constexpr (Dim >= 3) {
    if (mesh.size[2] < 1)
      throw std::invalid_argument(filename + ": Mesh.nz must be >= 1.");
  }

  for (Index d = 0; d < Dim; ++d) {
    if (!(mesh.lenght[d] > Scalar{ 0 }))
      throw std::invalid_argument(filename + ": every mesh length must be strictly positive.");
  }

  if (!(control.tmax > Scalar{ 0 }))
    throw std::invalid_argument(filename + ": Control.tmax must be strictly positive.");

  if (control.printStep < 1)
    throw std::invalid_argument(filename + ": Control.printStep must be >= 1.");

  if (performance.cores < 0)
    throw std::invalid_argument(filename + ": Performance.cores must be >= 0 (0 = autodetect).");

  return { fluid, mesh, control, performance };
}
} // namespace lbmini

#endif  // LBMINI_DATA_DATA_READER_HPP_
