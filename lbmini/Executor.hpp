#ifndef LBMINI_EXECUTOR_HPP_
#define LBMINI_EXECUTOR_HPP_

#include <memory>
#include <string>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "Data.hpp"
#include "Lbm/ILattice.hpp"
#include "Lbm/ILbmTube.hpp"

namespace lbmini {
template<typename Scalar>
class Executor {
public:
  using Index = Eigen::Index;
  using ILatticeT = lbmini::ILattice<Scalar, 2, 9>;
  using ILbmTubeT = lbmini::ILbmTube<Scalar, ILatticeT>;

  Executor(const Executor& other) = delete;

  Executor& operator=(const Executor&) = delete;

  ~Executor();

  static Executor* GetExecutor(
    const FluidData<Scalar>& fluid,
    const MeshData<Scalar, 2>& mesh,
    const ControlData<Scalar>& control,
    const PerformanceData& performance,
    int* argc = nullptr,
    char*** argv = nullptr
  );

  void Run(const std::string& outpath);

  ILbmTubeT* GetLbmTube() const;

private:
  void OutputData(
    Index step,
    const std::string& outDir,
    double elapsed_seconds,
    int rank
  );

  Executor(
    const FluidData<Scalar>& fluid,
    const MeshData<Scalar, 2>& mesh,
    const ControlData<Scalar>& control,
    const PerformanceData& performance,
    int* argc = nullptr,
    char*** argv = nullptr
  );

  FluidData<Scalar> fluid_;
  MeshData<Scalar, 2> mesh_;
  ControlData<Scalar> control_;
  PerformanceData performance_;

  bool mpiInitializedByUs_ = false;

  static std::unique_ptr<Executor> singleton_;
  std::unique_ptr<ILbmTubeT> lbmTube_;
};
} // namespace lbmini

#endif // LBMINI_EXECUTOR_HPP_
