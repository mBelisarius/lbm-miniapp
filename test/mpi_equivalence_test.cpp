#include <algorithm>
#include <cmath>
#include <iostream>

#include <mpi.h>

#include "Data.hpp"
#include "Lbm/Mpi/Cpu/LatticeD2Q9.hpp"
#include "Lbm/Mpi/Cpu/LbmTube.hpp"
#include "Lbm/Plain/LatticeD2Q9.hpp"
#include "Lbm/Plain/LbmTube.hpp"
#include "Utils.hpp"

namespace {
using Scalar = double;
using MpiTube = lbmini::mpi::cpu::LbmTube<Scalar, lbmini::mpi::cpu::LatticeD2Q9<Scalar>>;
using PlainTube = lbmini::plain::LbmTube<Scalar, lbmini::plain::LatticeD2Q9<Scalar>>;

template<typename Tensor>
bool compare(const Tensor& actual, const Tensor& expected, const char* field, int rank, int steps) {
  for (Eigen::Index k = 0; k < expected.size(); ++k) {
    const Scalar a = actual.data()[k], e = expected.data()[k];
    if (!(std::abs(a - e) <= 1.0e-9 * std::max(Scalar{ 1 }, std::abs(e)))) {
      std::cerr << "rank " << rank << " step " << steps << ' ' << field << '[' << k
                << "]: MPI=" << a << " plain=" << e << '\n';
      return false;
    }
  }
  return true;
}

bool physical(const PlainTube& plain) {
  const auto t = plain.T();
  const auto rho = plain.Rho();
  for (Eigen::Index k = 0; k < t.size(); ++k)
    if (!lbmini::IsFinite(t.data()[k]) || !(t.data()[k] > 0) || !lbmini::IsFinite(rho.data()[k]))
      return false;
  return true;
}

bool sameState(const MpiTube& mpi, const PlainTube& plain, int rank, int steps) {
  // Every rank calls every collective getter, even after a local comparison fails.
  const auto rho = mpi.Rho();
  const auto p = mpi.P();
  const auto t = mpi.T();
  const auto u = mpi.U();
  bool ok = compare(rho, plain.Rho(), "rho", rank, steps);
  ok = compare(p, plain.P(), "p", rank, steps) && ok;
  ok = compare(t, plain.T(), "T", rank, steps) && ok;
  return compare(u, plain.U(), "u", rank, steps) && ok;
}
} // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool ok = true;
  for (const auto [nx, ny] : { std::pair{ 64, 4 }, std::pair{ 33, 7 }, std::pair{ 6, 5 } }) {
    for (const auto [ux, uy] : { std::pair{ 0.25, 0.0 }, std::pair{ 0.1, 0.0 },
                                std::pair{ 0.3, 0.15 }, std::pair{ 0.0, 0.0 } }) {
      lbmini::FluidData<Scalar> fluid{};
      fluid.densityL = 1.0;
      fluid.pressureL = 1.0;
      fluid.densityR = 0.125;
      fluid.pressureR = 0.1;
      fluid.viscosity = 0.01;
      fluid.specificHeatCv = 2.5;
      fluid.specificHeatCp = 3.5;
      fluid.constant = 1.0;
      fluid.conductivity = fluid.specificHeatCp * fluid.viscosity / 0.71;
      lbmini::MeshData<Scalar, 2> mesh{};
      mesh.size[0] = nx;
      mesh.size[1] = ny;
      lbmini::ControlData<Scalar> control{};
      control.Ux = ux;
      control.Uy = uy;
      control.idw = std::sqrt(1.4);
      lbmini::PerformanceData perf{};
      PlainTube plain(fluid, mesh, control, perf);
      MpiTube mpi(fluid, mesh, control, perf);
      plain.Init();
      mpi.Init();
      ok = sameState(mpi, plain, rank, 0) && ok;
      int done = 0;
      for (const int chunk : { 1, 4, 7, 1, 12 }) {
        plain.Run(chunk, false);
        mpi.Run(chunk, false);
        done += chunk;
        if (!physical(plain))
          break;
        ok = sameState(mpi, plain, rank, done) && ok;
      }
      mpi.Init();
      plain.Init();
      ok = sameState(mpi, plain, rank, 0) && ok;
      mpi.Run(5, false);
      plain.Run(5, false);
      if (physical(plain))
        ok = sameState(mpi, plain, rank, 5) && ok;
    }
  }
  int local = ok ? 1 : 0, all = 0;
  MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (rank == 0)
    std::cout << (all ? "MPI equivalence passed\n" : "MPI equivalence FAILED\n");
  MPI_Finalize();
  return all ? 0 : 1;
}