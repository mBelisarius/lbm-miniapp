/**
 * @file backend_equivalence_test.cpp
 * @brief Every backend must reproduce the model, and `plain::LbmTube` is its serial form.
 *
 * Two chains of comparisons are made.
 *
 *  1. `plain::LbmTube` against the independent reference of `test/ReferenceTube.hpp`: a literal
 *     transcription of the equations (general per-direction energy closure with 36 exponentials per
 *     cell and step, three separate passes, per-`(cell, direction)` streaming tables). `plain::LbmTube`
 *     shares the factorised closure and the per-direction stencil with the optimised backends, so
 *     this is the one check that is independent of that algebra.
 *  2. Every optimised backend in `BackendTypes` against `plain::LbmTube`.
 *
 * The optimised backends fuse streaming, macroscopic recovery and collision into one pull kernel,
 * keep post-collision populations as their state, store the macroscopic fields only on the last step
 * of a `Run()`, and replace the per-cell streaming tables by one per-direction stencil. Each of those
 * is a place where a backend could silently diverge, so all macroscopic fields are compared after
 * `Init()` and after several `Run()` chunks of different lengths, for lattice shifts that exercise
 * every streaming path:
 *
 *  - `U = (0.25, 0)`: the benchmark shift (fractional x, integer y);
 *  - `U = (0.1, 0)`: a non-dyadic shift (weights equal only to rounding);
 *  - `U = (0.3, 0.15)`: fractional in both directions (four live corners, y wrap);
 *  - `U = (0, 0)`: every source is a lattice node (the "snap" path).
 *
 * Comparisons stop for a configuration once the *reference* leaves the physical
 * regime (non-finite values or a non-positive temperature): the unshifted
 * lattice, for instance, is unstable for this Sod setup and blows up within a
 * few tens of steps, after which round-off differences (e.g. FMA contraction
 * on a GPU) grow without bound and no two correct implementations agree.
 *
 * The MPI backend needs several processes and is compared against the plain
 * solver by `test/run-tests.sh` instead.
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "BackendTraits.hpp"
#include "ReferenceTube.hpp"
#include "Utils.hpp"

namespace {
using lbmini_test::BackendNames;
using lbmini_test::Index;
using lbmini_test::MeshShape;
using lbmini_test::Scalar;

using PlainTube = lbmini_test::PlainBackend::Tube;
using ReferenceTube = lbmini_test::reference::LbmTube<Scalar, lbmini::plain::LatticeD2Q9<Scalar>>;

/// Relative tolerance: same algorithm, different evaluation order / FMA
/// contraction / device math library.
constexpr Scalar kTol = 1.0e-9;

struct Shift {
  Scalar ux;
  Scalar uy;

  [[nodiscard]] std::string Name() const { return "U=(" + std::to_string(ux) + ", " + std::to_string(uy) + ")"; }
};

const std::vector<Shift>& Shifts() {
  static const std::vector<Shift> shifts = { { 0.25, 0.0 }, { 0.1, 0.0 }, { 0.3, 0.15 }, { 0.0, 0.0 } };
  return shifts;
}

const std::vector<MeshShape>& Shapes() {
  static const std::vector<MeshShape> shapes = { { 64, 4 }, { 63, 5 }, { 128, 2 }, { 33, 7 } };
  return shapes;
}

/// `Run()` chunk lengths: odd and even counts, and single steps, so both
/// ping-pong parities and the "store macroscopics on the last step" logic are
/// observed.
constexpr Index kChunks[] = { 1, 4, 7, 1, 12 };

template<typename Tensor>
void ExpectClose(const Tensor& actual, const Tensor& expected, const std::string& what) {
  ASSERT_EQ(actual.size(), expected.size()) << what;
  Index mismatches = 0;
  for (Index k = 0; k < expected.size(); ++k) {
    const Scalar e = expected.data()[k];
    const Scalar a = actual.data()[k];
    const Scalar tol = kTol * std::max(Scalar{ 1 }, std::fabs(e));
    if (!(std::fabs(a - e) <= tol)) {
      if (mismatches++ < 5)
        ADD_FAILURE() << what << " differs at flat index " << k << ": solver " << a << ", reference " << e;
    }
  }
  EXPECT_EQ(mismatches, 0) << what << ": " << mismatches << " of " << expected.size() << " values differ";
}

template<typename Tube, typename Reference>
void ExpectSameState(const Tube& tube, const Reference& reference, const std::string& where) {
  ExpectClose(tube.Rho(), reference.Rho(), where + " rho");
  ExpectClose(tube.P(), reference.P(), where + " p");
  ExpectClose(tube.T(), reference.T(), where + " T");
  ExpectClose(tube.U(), reference.U(), where + " u");
}

/// Whether the reference state is still a meaningful solution to compare against.
template<typename Reference>
bool IsPhysical(const Reference& reference) {
  const auto tem = reference.T();
  const auto rho = reference.Rho();
  for (Index k = 0; k < tem.size(); ++k)
    if (!lbmini::IsFinite(tem.data()[k]) || !(tem.data()[k] > Scalar{ 0 }) || !lbmini::IsFinite(rho.data()[k]))
      return false;
  return true;
}

/// Every shift x mesh shape x `Run()` chunk of `Tube` against `Reference`.
template<typename Tube, typename Reference>
void CompareOverShiftsAndShapes(
  const std::string& name,
  const lbmini::PerformanceData& performance,
  const lbmini::PerformanceData& referencePerformance
) {
  for (const auto& shift : Shifts()) {
    for (const auto& shape : Shapes()) {
      const std::string where = name + " " + shift.Name() + " mesh " + shape.Name();
      SCOPED_TRACE(where);

      const auto fluid = lbmini_test::MakeFluid();
      const auto mesh = lbmini_test::MakeMesh(shape.nx, shape.ny);
      const auto control = lbmini_test::MakeControl(shift.ux, shift.uy);

      Reference reference(fluid, mesh, control, referencePerformance);
      auto tube = std::make_unique<Tube>(fluid, mesh, control, performance);

      reference.Init();
      tube->Init();
      ExpectSameState(*tube, reference, where + " after Init");

      Index done = 0;
      for (const Index chunk : kChunks) {
        reference.Run(chunk, false);
        tube->Run(chunk, false);
        done += chunk;
        if (!IsPhysical(reference))
          break;
        ExpectSameState(*tube, reference, where + " after " + std::to_string(done) + " steps");
        if (::testing::Test::HasFailure())
          return; // one diverging configuration is enough to diagnose
      }
      // At least the first chunks must have been compared.
      EXPECT_GT(done, kChunks[0]) << where << ": the reference left the physical regime immediately";
    }
  }
}

/// Longer runs of three stable shock tubes (pressure ratios 5, 10 and 20) of `Tube` against `Reference`.
template<typename Tube, typename Reference>
void CompareThroughStableShockRuns(
  const std::string& name,
  const lbmini::PerformanceData& performance,
  const lbmini::PerformanceData& referencePerformance
) {
  for (const Scalar ratio : { Scalar{ 5 }, Scalar{ 10 }, Scalar{ 20 } }) {
    SCOPED_TRACE("pressure ratio " + std::to_string(ratio));
    auto fluid = lbmini_test::MakeFluid();
    fluid.pressureR = fluid.pressureL / ratio;
    fluid.densityR *= std::pow(fluid.pressureR / Scalar{ 0.1 }, Scalar{ 1 } / fluid.gamma);
    fluid.viscosity = Scalar{ 0.1 };
    fluid.conductivity = fluid.specificHeatCp * fluid.viscosity / fluid.prandtl;
    const auto mesh = lbmini_test::MakeMesh(257, 3);
    const auto control = lbmini_test::MakeControl();
    Reference reference(fluid, mesh, control, referencePerformance);
    Tube tube(fluid, mesh, control, performance);
    reference.Init();
    tube.Init();
    for (int chunk = 0; chunk < 4; ++chunk) {
      reference.Run(40, false);
      tube.Run(40, false);
      ASSERT_TRUE(IsPhysical(reference));
      ExpectSameState(tube, reference, name + " stable shock");
      if (::testing::Test::HasFailure())
        return;
    }
  }
}

template<typename BackendT>
class BackendEquivalence : public ::testing::Test {
protected:
  void SetUp() override {
    if (!BackendT::IsAvailable())
      GTEST_SKIP() << BackendT::Name() << " backend is not available in current runtime environment.";
  }
};

TYPED_TEST_SUITE(BackendEquivalence, lbmini_test::OptimisedBackendTypes, BackendNames);

TYPED_TEST(BackendEquivalence, MatchesPlainReference) {
  CompareOverShiftsAndShapes<typename TypeParam::Tube, PlainTube>(
    TypeParam::Name(), TypeParam::Performance(), lbmini_test::PlainBackend::Performance()
  );
}

TYPED_TEST(BackendEquivalence, MatchesPlainThroughLongerStableShockRuns) {
  CompareThroughStableShockRuns<typename TypeParam::Tube, PlainTube>(
    TypeParam::Name(), TypeParam::Performance(), lbmini_test::PlainBackend::Performance()
  );
}

/// The serial solver of the program against the literal transcription of the model.
TEST(PlainEquivalence, MatchesIndependentReference) {
  CompareOverShiftsAndShapes<PlainTube, ReferenceTube>(
    "plain", lbmini_test::PlainBackend::Performance(), lbmini_test::PlainBackend::Performance()
  );
}

TEST(PlainEquivalence, MatchesIndependentReferenceThroughLongerStableShockRuns) {
  CompareThroughStableShockRuns<PlainTube, ReferenceTube>(
    "plain", lbmini_test::PlainBackend::Performance(), lbmini_test::PlainBackend::Performance()
  );
}

/// `Init()` must restart from the initial condition, whatever ran before.
template<typename Tube>
void ExpectReInitRestarts(const std::string& name, const lbmini::PerformanceData& performance) {
  const MeshShape shape{ 64, 4 };
  auto tube = std::make_unique<Tube>(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(shape.nx, shape.ny),
    lbmini_test::MakeControl(),
    performance
  );
  tube->Init();
  tube->Run(5, false);
  const auto expected = tube->Rho();

  tube->Run(3, false);
  tube->Init();
  tube->Run(5, false);
  ExpectClose(tube->Rho(), expected, name + " rho after re-Init");
}

TYPED_TEST(BackendEquivalence, ReInitRestartsFromInitialCondition) {
  ExpectReInitRestarts<typename TypeParam::Tube>(TypeParam::Name(), TypeParam::Performance());
}

TEST(PlainEquivalence, ReInitRestartsFromInitialCondition) {
  ExpectReInitRestarts<PlainTube>("plain", lbmini_test::PlainBackend::Performance());
}

/// Largest difference between two final states, relative to `max(1, |value|)`, over rho, p, T and u.
template<typename Tube, typename Reference>
Scalar MaxRelativeDifference(const Tube& tube, const Reference& reference) {
  Scalar worst = 0;
  const auto accumulate = [&worst](const auto& actual, const auto& expected) {
    for (Index k = 0; k < expected.size(); ++k) {
      const Scalar e = expected.data()[k];
      worst = std::max(worst, std::fabs(actual.data()[k] - e) / std::max(Scalar{ 1 }, std::fabs(e)));
    }
  };
  accumulate(tube.Rho(), reference.Rho());
  accumulate(tube.P(), reference.P());
  accumulate(tube.T(), reference.T());
  accumulate(tube.U(), reference.U());
  return worst;
}

/// The Sod tube of the thesis (1000 x 10 nodes, 346 steps, a pressure ratio of 10) of `Tube` against `Reference`;
/// the difference is printed so that the end-to-end agreement of every backend is on record.
template<typename Tube, typename Reference>
void CompareEndToEndSod(
  const std::string& name,
  const lbmini::PerformanceData& performance,
  const lbmini::PerformanceData& referencePerformance
) {
  const auto fluid = lbmini_test::MakeFluid();
  const auto mesh = lbmini_test::MakeMesh(1000, 10);
  const auto control = lbmini_test::MakeControl();
  Reference reference(fluid, mesh, control, referencePerformance);
  Tube tube(fluid, mesh, control, performance);
  reference.Init();
  tube.Init();
  reference.Run(346, false);
  tube.Run(346, false);
  ASSERT_TRUE(IsPhysical(reference));
  const Scalar difference = MaxRelativeDifference(tube, reference);
  std::cout << "[ end-to-end ] " << name << ": largest relative difference to the reference " << difference << std::endl;
  EXPECT_LT(difference, kTol);
}

TYPED_TEST(BackendEquivalence, EndToEndSodTube) {
  CompareEndToEndSod<typename TypeParam::Tube, ReferenceTube>(
    TypeParam::Name(), TypeParam::Performance(), lbmini_test::PlainBackend::Performance()
  );
}

TEST(PlainEquivalence, EndToEndSodTube) {
  CompareEndToEndSod<PlainTube, ReferenceTube>(
    "plain", lbmini_test::PlainBackend::Performance(), lbmini_test::PlainBackend::Performance()
  );
}
} // namespace
