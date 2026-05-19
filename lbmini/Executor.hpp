#ifndef LBMINI_EXECUTOR_HPP_
#define LBMINI_EXECUTOR_HPP_

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include "AsyncWriter.hpp"
#include "Data.hpp"
#include "Lbm/ILattice.hpp"
#include "Lbm/ILbmTube.hpp"

namespace lbmini {
template<typename Scalar>
class Executor {
public:
  using Index = Eigen::Index;
  using ILatticeT = lbmini::ILattice<Scalar, 2, 9>;

  // The lattice-independent base: a genuine base class of every 2-D solver, so
  // the executor owns them through a plain implicit upcast. Storing them as
  // `ILbmTube<Scalar, ILatticeT>` used to require a `reinterpret_cast` between
  // unrelated types (undefined behaviour on every virtual call and on
  // `delete`) — see docs/memory-safety.md §5.1.
  using ILbmTubeT = lbmini::ILbmTubeNd<Scalar, 2>;

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

  static void Reset();

  void Run(const std::string& outpath);

  ILbmTubeT* GetLbmTube() const;

private:
  void OutputData(
    Index step,
    const std::string& outDir,
    double elapsed_seconds,
    int rank
  );

  // Barrier across the ranks of the distributed backend (no-op otherwise), so
  // that a timed interval starts and ends together on every rank.
  void SynchronizeRanks() const;

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
  std::unique_ptr<AsyncWriter<Scalar>> asyncWriter_;
};
} // namespace lbmini

#endif // LBMINI_EXECUTOR_HPP_
