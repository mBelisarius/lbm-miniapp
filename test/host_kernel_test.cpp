/**
 * @file host_kernel_test.cpp
 * @brief Unit tests of the SIMD lane type and of the vectorised CPU kernel.
 *
 * The CPU backends are compared against the plain reference by
 * `backend_equivalence_test.cpp`, but only at the lane count the test machine
 * happens to have (4 on AVX2, 8 on AVX-512). The kernel is written once for
 * every width, so this suite instantiates it for 1 (scalar), 2, 4, 8 and 16 lanes
 * and requires them to agree, which covers the AVX-512 and the SSE/NEON paths
 * on any machine, and checks the pieces they share: the exponential, the
 * padded layout and the segment partition.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "Lbm/HostKernel.hpp"
#include "Lbm/OpenMp/Cpu/LatticeD2Q9.hpp"
#include "Lbm/Simd.hpp"
#include "MemorySafetyHarness.hpp"

namespace {
using lbmini::simd::Ops;

template<int W>
struct Width {
  static constexpr int kLanes = W;
};

template<typename WidthT>
class SimdOps : public ::testing::Test {};

using Widths = ::testing::Types<Width<1>, Width<2>, Width<4>, Width<8>, Width<16>>;

TYPED_TEST_SUITE(SimdOps, Widths);

/// Every lane of `v`, as an array.
template<int W>
std::vector<double> Lanes(const typename Ops<double, W>::V v) {
  std::vector<double> out(W);
  for (int l = 0; l < W; ++l)
    out[l] = Ops<double, W>::Lane(v, l);
  return out;
}

TYPED_TEST(SimdOps, ExpMatchesTheLibraryToAFewUlp) {
  constexpr int W = TypeParam::kLanes;
  using S = Ops<double, W>;

  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> wide(-700.0, 700.0);
  std::uniform_real_distribution<double> narrow(-2.0, 2.0);

  double worst = 0.0;
  for (int trial = 0; trial < 20000; ++trial) {
    double in[W];
    double out[W];
    for (int l = 0; l < W; ++l)
      in[l] = (trial % 2 == 0) ? wide(rng) : narrow(rng);
    S::Store(out, S::Exp(S::Load(in)));
    for (int l = 0; l < W; ++l) {
      const double expected = std::exp(in[l]);
      const double rel = std::fabs(out[l] - expected) / expected;
      worst = std::max(worst, rel);
      ASSERT_TRUE(std::isfinite(out[l])) << "exp(" << in[l] << ")";
    }
  }
  // 1 ulp is 1.1e-16 .. 2.2e-16 relative; the reduction and the Horner chain add a few.
  EXPECT_LT(worst, 1.0e-15);
}

TYPED_TEST(SimdOps, ExpAtTheEdgesOfTheClampedRange) {
  constexpr int W = TypeParam::kLanes;
  using S = Ops<double, W>;

  for (const double x : { -700.0, -699.999, -354.2, -1.0, -0.5, 0.0, 0.25, 0.34657359, 1.0, 354.2, 699.999, 700.0 }) {
    const double v = S::Lane(S::Exp(S::Splat(x)), W - 1);
    EXPECT_NEAR(v / std::exp(x), 1.0, 1.0e-15) << "exp(" << x << ")";
  }
  EXPECT_EQ(S::Lane(S::Exp(S::Splat(0.0)), 0), 1.0);
}

TYPED_TEST(SimdOps, SelectMinMaxAbsAndMasks) {
  constexpr int W = TypeParam::kLanes;
  using S = Ops<double, W>;

  double a[W];
  double b[W];
  for (int l = 0; l < W; ++l) {
    a[l] = (l % 2 == 0) ? -1.5 - l : 2.5 + l;
    b[l] = (l % 3 == 0) ? 0.5 : -4.0 + l;
  }
  const auto va = S::Load(a);
  const auto vb = S::Load(b);
  const auto less = Lanes<W>(S::Select(va < vb, va, vb));
  const auto mx = Lanes<W>(S::Max(va, vb));
  const auto mn = Lanes<W>(S::Min(va, vb));
  const auto ab = Lanes<W>(S::Abs(va));
  for (int l = 0; l < W; ++l) {
    EXPECT_EQ(less[l], a[l] < b[l] ? a[l] : b[l]);
    EXPECT_EQ(mx[l], a[l] > b[l] ? a[l] : b[l]);
    EXPECT_EQ(mn[l], a[l] < b[l] ? a[l] : b[l]);
    EXPECT_EQ(ab[l], std::fabs(a[l]));
  }

  const auto always = S::True();
  EXPECT_TRUE(S::All(always));
  EXPECT_TRUE(S::Any(always));
  const auto never = S::And(always, va < va);
  EXPECT_FALSE(S::Any(never));
  EXPECT_FALSE(S::All(never));
  if constexpr (W > 1) {
    const auto some = (va > S::Splat(0.0));
    EXPECT_TRUE(S::Any(some));
    EXPECT_FALSE(S::All(some));
    EXPECT_TRUE(S::All(S::Or(some, va <= S::Splat(0.0))));
  }
}

TEST(HostLayout, PitchAndPlaneAvoidPowerOfTwoAliasing) {
  using lbmini::host::ChoosePitch;
  using lbmini::host::ChoosePlane;
  for (lbmini::host::Index n = 1; n <= 20000; ++n) {
    const auto pitch = ChoosePitch(n);
    ASSERT_GE(pitch, n);
    ASSERT_EQ(pitch % 16, 0);
    ASSERT_LT(pitch - n, 32);
    ASSERT_NE((pitch * 8) % 4096, 0) << "rows of " << n << " columns alias in the 4K check";
  }
  for (const lbmini::host::Index cells : { 1, 8, 100, 4096, 40960, 512 * 64, 123457 }) {
    const auto plane = ChoosePlane(cells);
    ASSERT_GE(plane, cells);
    ASSERT_EQ(plane % 8, 0);
    ASSERT_EQ(plane % 512, 320);
    ASSERT_LT(plane - cells, 512 + 8);
  }
}

TEST(HostPartition, EverySegmentCoversDistinctBlocksAndTheirUnionIsTheMesh) {
  using lbmini::host::Index;
  for (const Index columns : { Index{ 16 }, Index{ 48 }, Index{ 112 }, Index{ 512 }, Index{ 5008 }, Index{ 10016 } }) {
    for (const Index rows : { Index{ 1 }, Index{ 2 }, Index{ 5 }, Index{ 100 } }) {
      for (const int lanes : { 1, 2, 4, 8, 16 }) {
        for (const int threads : { 1, 8, 28 }) {
          const auto part = lbmini::host::MakePartition(columns, rows, lanes, threads);
          std::vector<int> seen(static_cast<std::size_t>(rows * (columns / lanes)), 0);
          for (Index s = 0; s < part.Segments(); ++s) {
            ASSERT_GE(part.BlockCount(s), 1);
            for (Index b = 0; b < part.BlockCount(s); ++b)
              ++seen[static_cast<std::size_t>(part.Row(s) * (columns / lanes) + part.FirstBlock(s) + b)];
          }
          ASSERT_TRUE(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }))
            << columns << "x" << rows << " lanes " << lanes << " threads " << threads;
        }
      }
    }
  }
}

/// Plain single-threaded driver of the kernel at a fixed lane count.
template<int W>
class Driver {
public:
  using Lattice = lbmini::openmp::cpu::LatticeD2Q9<double>;
  using Kernel = lbmini::host::Kernel<double, Lattice, W>;
  using Index = lbmini::host::Index;

  Driver(const Index nx, const Index ny, const double ux, const double uy)
    : geo_(lbmini::host::MakeGeometry(nx, ny)),
      params_(lbmini::host::MakeParams<double, Lattice>(
        lbmini_test::MakeFluid(), lbmini_test::MakeMesh(nx, ny), lbmini_test::MakeControl(ux, uy))) {
    const auto plane = static_cast<std::size_t>(geo_.plane);
    f_[0].assign(9 * plane, 0.0);
    f_[1].assign(9 * plane, 0.0);
    g_[0].assign(9 * plane, 0.0);
    g_[1].assign(9 * plane, 0.0);
    xi_.assign(2 * plane, 0.0);
    rho_.assign(plane, 0.0);
    p_.assign(plane, 0.0);
    tem_.assign(plane, 0.0);
    u_.assign(2 * plane, 0.0);
    lbmini::host::Fields<double> fl;
    fl.fOut = f_[0].data();
    fl.gOut = g_[0].data();
    fl = Bind(fl);
    for (Index j = 0; j < ny; ++j)
      Kernel::InitRow(params_, geo_, fl, nx, 0, j, 0, geo_.pitch / W);
  }

  void Run(const int steps) {
    for (int t = 0; t < steps; ++t) {
      lbmini::host::Fields<double> fl;
      fl.fIn = f_[cur_].data();
      fl.gIn = g_[cur_].data();
      fl.fOut = f_[1 - cur_].data();
      fl.gOut = g_[1 - cur_].data();
      fl = Bind(fl);
      for (Index j = 0; j < geo_.ny; ++j)
        Kernel::template StepRow<true>(params_, geo_, fl, 0, geo_.nx - 1, j, 0, geo_.pitch / W);
      cur_ = 1 - cur_;
    }
  }

  /// Field value at `(i, j)`; `which` = 0 rho, 1 p, 2 T, 3 ux, 4 uy.
  [[nodiscard]] double At(const int which, const Index i, const Index j) const {
    const std::vector<double>* base[] = { &rho_, &p_, &tem_, &u_, &u_ };
    const Index off = (which == 4) ? geo_.plane : 0;
    return (*base[which])[static_cast<std::size_t>(off + j * geo_.pitch + i)];
  }

  [[nodiscard]] const lbmini::host::Geometry& Geo() const { return geo_; }

private:
  lbmini::host::Fields<double> Bind(lbmini::host::Fields<double> fl) {
    fl.xi = xi_.data();
    fl.rho = rho_.data();
    fl.p = p_.data();
    fl.tem = tem_.data();
    fl.u = u_.data();
    return fl;
  }

  lbmini::host::Geometry geo_;
  lbmini::host::Params<double> params_;
  lbmini::DeviceBuffer<double> f_[2];
  lbmini::DeviceBuffer<double> g_[2];
  std::vector<double> xi_, rho_, p_, tem_, u_;
  int cur_ = 0;
};

TEST(HostKernel, EveryLaneCountComputesTheSameSolution) {
  struct Case {
    lbmini::host::Index nx, ny;
    double ux, uy;
  };
  // A pitch that is not a multiple of the lane count would also exercise the tail, but
  // the pitch is always a multiple of 8 by construction; the shapes below differ in how
  // many blocks are interior, edge, pad-only, and in whether the y wrap is trivial.
  for (const Case c : { Case{ 37, 5, 0.25, 0.0 }, Case{ 64, 1, 0.25, 0.0 }, Case{ 100, 3, 0.3, 0.15 }, Case{ 9, 4, 0.1, 0.0 } }) {
    Driver<1> ref(c.nx, c.ny, c.ux, c.uy);
    Driver<2> d2(c.nx, c.ny, c.ux, c.uy);
    Driver<4> d4(c.nx, c.ny, c.ux, c.uy);
    Driver<8> d8(c.nx, c.ny, c.ux, c.uy);
    Driver<16> d16(c.nx, c.ny, c.ux, c.uy);
    for (int step = 0; step < 6; ++step) {
      ref.Run(3);
      d2.Run(3);
      d4.Run(3);
      d8.Run(3);
      d16.Run(3);
      for (int which = 0; which < 5; ++which) {
        for (lbmini::host::Index j = 0; j < c.ny; ++j) {
          for (lbmini::host::Index i = 0; i < c.nx; ++i) {
            const double e = ref.At(which, i, j);
            const double tol = 1.0e-12 * std::max(1.0, std::fabs(e));
            ASSERT_TRUE(std::isfinite(e)) << c.nx << "x" << c.ny;
            ASSERT_NEAR(d2.At(which, i, j), e, tol) << "W=2 field " << which << " at " << i << "," << j;
            ASSERT_NEAR(d4.At(which, i, j), e, tol) << "W=4 field " << which << " at " << i << "," << j;
            ASSERT_NEAR(d8.At(which, i, j), e, tol) << "W=8 field " << which << " at " << i << "," << j;
            ASSERT_NEAR(d16.At(which, i, j), e, tol) << "W=16 field " << which << " at " << i << "," << j;
          }
        }
      }
    }
  }
}

TEST(HostKernel, StreamingStoresGiveTheSameSolution) {
  // The populations are the only thing written with streaming stores, and only their store
  // instruction changes. The two paths are separate instantiations of the block code, though, and
  // `-ffast-math` may contract or reassociate them differently: the solutions agree to rounding
  // error, not bit for bit. The driver's buffers are 64-byte aligned, so the streaming path is
  // taken (it is skipped for unaligned buffers).
  for (const int mode : { 0, 1 }) {
    ASSERT_EQ(::setenv("LBMINI_HOST_STREAM", mode == 0 ? "0" : "1", 1), 0);
    Driver<16> plain(100, 3, 0.25, 0.0);
    ASSERT_EQ(::setenv("LBMINI_HOST_STREAM", mode == 0 ? "1" : "0", 1), 0);
    Driver<16> other(100, 3, 0.25, 0.0);
    ::unsetenv("LBMINI_HOST_STREAM");
    plain.Run(12);
    other.Run(12);
    for (int which = 0; which < 5; ++which)
      for (lbmini::host::Index j = 0; j < 3; ++j)
        for (lbmini::host::Index i = 0; i < 100; ++i) {
          const double expected = plain.At(which, i, j);
          ASSERT_NEAR(expected, other.At(which, i, j), 1e-12 * std::max(1.0, std::abs(expected)))
            << "field " << which << " at " << i << "," << j;
        }
  }
}

TEST(HostKernel, PadColumnsStayFinite) {
  Driver<4> d(37, 5, 0.25, 0.0);
  d.Run(20);
  const auto& geo = d.Geo();
  for (lbmini::host::Index j = 0; j < geo.ny; ++j)
    for (lbmini::host::Index i = 0; i < geo.pitch; ++i)
      for (int which = 0; which < 5; ++which)
        ASSERT_TRUE(std::isfinite(d.At(which, i, j))) << "field " << which << " at " << i << "," << j;
}
} // namespace
