/**
 * @file backend_equivalence_test.cpp
 * @brief Every optimised backend must reproduce the plain reference solver.
 *
 * The optimised backends are not line-by-line ports of the reference: they
 * fuse streaming, macroscopic recovery and collision into one pull kernel,
 * keep post-collision populations as their state, store the macroscopic
 * fields only on the last step of a `Run()`, and replace the per-cell
 * streaming tables by one per-direction stencil. Each of those is a place
 * where a backend could silently diverge, so this suite compares every
 * macroscopic field of every backend in `BackendTypes` against
 * `plain::LbmTube` after `Init()` and after several `Run()` chunks of
 * different lengths, for lattice shifts that exercise every streaming path:
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
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "BackendTraits.hpp"
#include "Utils.hpp"

namespace {
using lbmini_test::BackendNames;
using lbmini_test::Index;
using lbmini_test::MeshShape;
using lbmini_test::Scalar;

using PlainTube = lbmini_test::PlainBackend::Tube;

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
        ADD_FAILURE() << what << " differs at flat index " << k << ": backend " << a << ", plain " << e;
    }
  }
  EXPECT_EQ(mismatches, 0) << what << ": " << mismatches << " of " << expected.size() << " values differ";
}

template<typename Tube>
void ExpectSameState(const Tube& tube, const PlainTube& plain, const std::string& where) {
  ExpectClose(tube.Rho(), plain.Rho(), where + " rho");
  ExpectClose(tube.P(), plain.P(), where + " p");
  ExpectClose(tube.T(), plain.T(), where + " T");
  ExpectClose(tube.U(), plain.U(), where + " u");
}

/// Whether the reference state is still a meaningful solution to compare against.
bool IsPhysical(const PlainTube& plain) {
  const auto tem = plain.T();
  const auto rho = plain.Rho();
  for (Index k = 0; k < tem.size(); ++k)
    if (!lbmini::IsFinite(tem.data()[k]) || !(tem.data()[k] > Scalar{ 0 }) || !lbmini::IsFinite(rho.data()[k]))
      return false;
  return true;
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
  for (const auto& shift : Shifts()) {
    for (const auto& shape : Shapes()) {
      const std::string where = std::string(TypeParam::Name()) + " " + shift.Name() + " mesh " + shape.Name();
      SCOPED_TRACE(where);

      const auto fluid = lbmini_test::MakeFluid();
      const auto mesh = lbmini_test::MakeMesh(shape.nx, shape.ny);
      const auto control = lbmini_test::MakeControl(shift.ux, shift.uy);

      PlainTube plain(fluid, mesh, control, lbmini_test::PlainBackend::Performance());
      auto tube = std::make_unique<typename TypeParam::Tube>(fluid, mesh, control, TypeParam::Performance());

      plain.Init();
      tube->Init();
      ExpectSameState(*tube, plain, where + " after Init");

      Index done = 0;
      for (const Index chunk : kChunks) {
        plain.Run(chunk, false);
        tube->Run(chunk, false);
        done += chunk;
        if (!IsPhysical(plain))
          break;
        ExpectSameState(*tube, plain, where + " after " + std::to_string(done) + " steps");
        if (::testing::Test::HasFailure())
          return; // one diverging configuration is enough to diagnose
      }
      // At least the first chunks must have been compared.
      EXPECT_GT(done, kChunks[0]) << where << ": the reference left the physical regime immediately";
    }
  }
}

/// `Init()` must restart from the initial condition, whatever ran before.
TYPED_TEST(BackendEquivalence, ReInitRestartsFromInitialCondition) {
  const MeshShape shape{ 64, 4 };
  auto tube = std::make_unique<typename TypeParam::Tube>(
    lbmini_test::MakeFluid(),
    lbmini_test::MakeMesh(shape.nx, shape.ny),
    lbmini_test::MakeControl(),
    TypeParam::Performance()
  );
  tube->Init();
  tube->Run(5, false);
  const auto expected = tube->Rho();

  tube->Run(3, false);
  tube->Init();
  tube->Run(5, false);
  ExpectClose(tube->Rho(), expected, std::string(TypeParam::Name()) + " rho after re-Init");
}
} // namespace
