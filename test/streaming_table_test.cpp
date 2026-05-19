/**
 * @file streaming_table_test.cpp
 * @brief Direct bounds proof for the precomputed streaming (gather) tables.
 *
 * Streaming is the only place in the solver where memory is addressed
 * *indirectly*: `buildStreamingTables()` precomputes, for every
 * `(direction, cell)` pair, four source cell indices plus four inverse-distance
 * weights, and the hot kernel then does
 *
 * @code
 * fOut[cell] = w0 * fplane[i0] + w1 * fplane[i1] + w2 * fplane[i2] + w3 * fplane[i3];
 * @endcode
 *
 * with no bounds check whatsoever — by design, since a branch there would kill
 * vectorisation and warp uniformity. Consequently a single bad table entry is a
 * silent out-of-bounds read (or, on a shifted plane, a read far outside the
 * allocation) that a benchmark would happily "measure".
 *
 * The tables depend only on the mesh geometry and on the lattice shift
 * `control.U`, so they can be validated exhaustively and cheaply: this test
 * sweeps mesh shapes against shift velocities and checks the invariants the
 * kernel silently assumes:
 *
 *  - the tables are exactly `nx * ny * Q * 4` long;
 *  - every index lies in `[0, nx * ny)`;
 *  - every weight is finite, non-negative, and the four weights sum to 1
 *    (a convex combination — otherwise mass is created or destroyed);
 *  - `nx * ny` still fits into the `std::int32_t` the table is made of.
 *
 * Covers the two backends that keep the table in host memory with the
 * `(idc * N + cell) * 4 + k` layout. The GPU-oriented backends mirror the same
 * host-side construction into device storage (Kokkos/RAJA use
 * `(idc * 4 + k) * N + cell`); validating those requires a device copy-back and
 * is covered instead by running the suite under `compute-sanitizer`, see
 * `docs/memory-safety.md`.
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "MemorySafetyHarness.hpp"

#include "Lbm/Plain/LatticeD2Q9.hpp"
#include "Lbm/Plain/LbmTube.hpp"

#ifdef LBMINI_TEST_OPENMP_CPU
#include "Lbm/OpenMp/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenMp/Cpu/LbmTube.hpp"
#endif

#if defined(LBMINI_TEST_OPENACC_CPU) && defined(_OPENACC)
#include "Lbm/OpenAcc/Cpu/LatticeD2Q9.hpp"
#include "Lbm/OpenAcc/Cpu/LbmTube.hpp"
#endif

namespace {
using lbmini_test::Index;
using lbmini_test::MeshShape;
using lbmini_test::Scalar;

/// Lattice shifts exercised for every mesh shape. `Ux == 0` takes the "snap to
/// an exact lattice node" path of the table builder, a non-integer shift takes
/// the four-way inverse-distance path, and the negative one flips the gather
/// direction — all three compute their source indices differently.
constexpr Scalar kShiftsX[] = { Scalar{ 0 }, Scalar{ 0.25 }, Scalar{ 0.5 }, Scalar{ 1 }, Scalar{ -0.25 } };
constexpr Scalar kShiftsY[] = { Scalar{ 0 }, Scalar{ 0.25 } };

/**
 * @brief Test-only accessor for the gather tables of a host backend.
 *
 * The members are `protected` in the solvers precisely so this probe can read
 * them without any cast or `friend` trickery.
 */
template<typename TubeT>
struct TableProbe : TubeT {
  using TubeT::TubeT;

  using TubeT::N_;
  using TubeT::distSize_;
  using TubeT::nx_;
  using TubeT::ny_;
  using TubeT::streamIdx_;
  using TubeT::streamW_;
};

struct PlainTables {
  using Lattice = lbmini::plain::LatticeD2Q9<Scalar>;
  using Tube = lbmini::plain::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::Plain, lbmini::TargetEnum::CPU);
  }
  static const char* Name() { return "plain"; }
};

#ifdef LBMINI_TEST_OPENMP_CPU
struct OpenMpCpuTables {
  using Lattice = lbmini::openmp::cpu::LatticeD2Q9<Scalar>;
  using Tube = lbmini::openmp::cpu::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::OpenMP, lbmini::TargetEnum::CPU);
  }
  static const char* Name() { return "openmp_cpu"; }
};
#endif

#if defined(LBMINI_TEST_OPENACC_CPU) && defined(_OPENACC)
struct OpenAccCpuTables {
  using Lattice = lbmini::openacc::cpu::LatticeD2Q9<Scalar>;
  using Tube = lbmini::openacc::cpu::LbmTube<Scalar, Lattice>;
  static lbmini::PerformanceData Performance() {
    return lbmini_test::MakePerformance(lbmini::BackendEnum::OpenACC, lbmini::TargetEnum::CPU);
  }
  static const char* Name() { return "openacc_cpu"; }
};
#endif

using TableBackends = ::testing::Types<
  PlainTables
#ifdef LBMINI_TEST_OPENMP_CPU
  , OpenMpCpuTables
#endif
#if defined(LBMINI_TEST_OPENACC_CPU) && defined(_OPENACC)
  , OpenAccCpuTables
#endif
>;

class TableBackendNames {
public:
  template<typename BackendT>
  static std::string GetName(int) { return BackendT::Name(); }
};

template<typename BackendT>
class StreamingTableBounds : public ::testing::Test {};

TYPED_TEST_SUITE(StreamingTableBounds, TableBackends, TableBackendNames);

TYPED_TEST(StreamingTableBounds, EveryGatherIndexIsInsideTheDomain) {
  using Probe = TableProbe<typename TypeParam::Tube>;
  constexpr Index kQ = TypeParam::Tube::kQ_;

  for (const auto& shape : lbmini_test::MeshShapes()) {
    for (const Scalar ux : kShiftsX) {
      for (const Scalar uy : kShiftsY) {
        const std::string where = std::string(TypeParam::Name()) + " mesh " + shape.Name() +
          " shift (" + std::to_string(ux) + ", " + std::to_string(uy) + ")";
        SCOPED_TRACE(where);

        // Heap allocated: an overrun of the tables then lands in
        // AddressSanitizer's redzone instead of in a neighbouring stack slot.
        const auto probe = std::make_unique<Probe>(
          lbmini_test::MakeFluid(),
          lbmini_test::MakeMesh(shape.nx, shape.ny),
          lbmini_test::MakeControl(ux, uy),
          TypeParam::Performance()
        );

        const Index cells = shape.Cells();
        ASSERT_EQ(probe->nx_, shape.nx);
        ASSERT_EQ(probe->ny_, shape.ny);
        ASSERT_EQ(probe->N_, cells);
        ASSERT_EQ(probe->distSize_, cells * kQ);

        // The table is made of `std::int32_t`; a mesh beyond 2^31 cells would
        // wrap around into negative indices and gather from wild addresses.
        ASSERT_LE(cells, static_cast<Index>(std::numeric_limits<std::int32_t>::max()))
          << "mesh too large for the int32 streaming table";

        ASSERT_EQ(static_cast<Index>(probe->streamIdx_.size()), cells * kQ * 4);
        ASSERT_EQ(static_cast<Index>(probe->streamW_.size()), cells * kQ * 4);

        for (Index idc = 0; idc < kQ; ++idc) {
          for (Index cell = 0; cell < cells; ++cell) {
            const Index base = (idc * cells + cell) * 4;
            Scalar weightSum = Scalar{ 0 };

            for (Index k = 0; k < 4; ++k) {
              const Index source = static_cast<Index>(probe->streamIdx_[base + k]);
              const Scalar weight = probe->streamW_[base + k];

              ASSERT_GE(source, Index{ 0 })
                << "negative gather index at idc " << idc << ", cell " << cell << ", k " << k;
              ASSERT_LT(source, cells)
                << "out-of-bounds gather index " << source << " (domain has " << cells
                << " cells) at idc " << idc << ", cell " << cell << ", k " << k;

              ASSERT_TRUE(lbmini::IsFinite(weight))
                << "non-finite gather weight at idc " << idc << ", cell " << cell << ", k " << k;
              EXPECT_GE(weight, Scalar{ 0 })
                << "negative gather weight at idc " << idc << ", cell " << cell << ", k " << k;
              weightSum += weight;
            }

            // A non-convex gather does not corrupt memory by itself, but it
            // means the table was built from a geometry the kernel does not
            // agree with — the same root cause that produces bad indices.
            if (std::fabs(weightSum - Scalar{ 1 }) > Scalar{ 1.0e-12 }) {
              FAIL() << "gather weights do not sum to one (" << weightSum << ") at idc "
                     << idc << ", cell " << cell;
            }
          }
        }
      }
    }
  }
}
} // namespace
