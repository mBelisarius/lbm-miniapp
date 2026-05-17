#undef _GLIBCXX_ASSERTIONS
#define NDEBUG

#if defined(__NVCOMPILER)
namespace std {
void __glibcxx_assert_fail(const char*, int, const char*, const char*) {}
}
#pragma acc routine(std::__glibcxx_assert_fail) seq
#endif

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Data.hpp"
#include "Executor.hpp"

using namespace Eigen;

int main(int argc, char* argv[]) {
  using Scalar = double;

  std::string configPath = "config.yaml";
  std::string outputPathBase = "out";

  // Parse CLI arguments
  for (int i = 1; i < argc; ++i) {
    if (std::string arg = argv[i]; arg == "--config") {
      if (i + 1 < argc) configPath = argv[++i];
    } else if (arg == "--outpath") {
      if (i + 1 < argc) outputPathBase = argv[++i];
    }
  }

  // Read simulation settings
  auto [fluid, mesh, control, performance] = lbmini::ReadYaml<Scalar, 2>(configPath);

  // LBM via Executor
  auto executor = lbmini::Executor<Scalar>::GetExecutor(fluid, mesh, control, performance, &argc, &argv);
  auto lbmTube = executor->GetLbmTube();
  if (!lbmTube) {
    throw std::runtime_error("Failed to initialize LBM backend.");
  }

  int rank = lbmTube->Rank();
  if (performance.backend != lbmini::BackendEnum::MPI && rank > 0) {
    return 0;
  }

  // Setup output directory
  std::string outputPath = outputPathBase;
  if (rank == 0) {
    Index counter = 1;
    while (std::filesystem::exists(outputPath) && !std::filesystem::is_empty(outputPath))
      outputPath = outputPathBase + std::to_string(counter++);
    std::filesystem::create_directory(outputPath);
    std::cout << "Outputting to directory: " << outputPath << std::endl;
    std::filesystem::copy(configPath, std::filesystem::path(outputPath) / "config.yaml");
  }

  lbmTube->BroadcastOutputPath(outputPath);

  executor->Run(outputPath);

  return 0;
}
