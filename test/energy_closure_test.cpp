#include <algorithm>
#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "Lbm/EnergyClosure.hpp"
#include "Lbm/Simd.hpp"

namespace {
template<int W>
void CheckScaledMoments() {
  using S = lbmini::simd::Ops<double, W>;
  constexpr double tiny = 1.0e-12;
  std::mt19937_64 random(20261106);
  std::uniform_real_distribution<double> temperature(0.01, 1.2);
  std::uniform_real_distribution<double> exponent(-20.0, 20.0);
  std::uniform_real_distribution<double> target(-0.9, 0.9);
  for (int trial = 0; trial < 1000; ++trial) {
    double minus[W], zero[W], plus[W], residual[W];
    for (int lane = 0; lane < W; ++lane) {
      const double t = temperature(random);
      const double xi = exponent(random);
      const double e = std::exp(-std::fabs(xi));
      minus[lane] = 0.5 * t * (xi >= 0.0 ? e * e : 1.0);
      zero[lane] = (1.0 - t) * e;
      plus[lane] = 0.5 * t * (xi >= 0.0 ? 1.0 : e * e);
      residual[lane] = target(random);
    }
    const auto moments = lbmini::energy::ScaledAxis(
      S::Load(minus), S::Load(zero), S::Load(plus), S::Load(residual), S::Splat(tiny)
    );
    for (int lane = 0; lane < W; ++lane) {
      const double z = minus[lane] + zero[lane] + plus[lane];
      const double mean = (plus[lane] - minus[lane]) / z;
      const double jacobian = (plus[lane] + minus[lane]) / z - mean * mean + tiny;
      const double scaledJ = S::Lane(moments.jacobian, lane) / (z * z);
      const double scaledR = S::Lane(moments.residual, lane) / (z * z);
      EXPECT_NEAR(scaledJ, jacobian, 2.0e-14);
      EXPECT_NEAR(scaledR, mean + residual[lane], 2.0e-14);
      if (std::fabs(jacobian) > 1.0e-4) {
        const double expected = (mean + residual[lane]) / jacobian;
        EXPECT_NEAR(
          S::Lane(moments.residual, lane) / S::Lane(moments.jacobian, lane),
          expected, 2.0e-11 * std::max(1.0, std::fabs(expected))
        );
      }
    }
  }
}

TEST(EnergyClosure, ScaledMomentsPreserveTheRegularisedNewtonStep) {
  CheckScaledMoments<1>();
  CheckScaledMoments<2>();
  CheckScaledMoments<4>();
  CheckScaledMoments<8>();
  CheckScaledMoments<16>();
}

TEST(EnergyClosure, DeterminantThresholdMustScaleWithBothNormalisers) {
  constexpr double tiny = 1.0e-12;
  for (const double scale : { 1.0e-8, 0.01, 1.0, 100.0 }) {
    for (const double side : { 1.0e-8, 0.01, 0.2 }) {
      const auto x = lbmini::energy::ScaledAxis(side * scale, scale, side * scale, 0.1, tiny);
      const auto y = lbmini::energy::ScaledAxis(side, 1.0, side, -0.1, tiny);
      const double zxy = x.z * y.z;
      const double determinant = x.jacobian * y.jacobian;
      const double normalised = (x.jacobian / (x.z * x.z)) * (y.jacobian / (y.z * y.z));
      EXPECT_EQ(std::fabs(determinant) > tiny * (zxy * zxy), std::fabs(normalised) > tiny);
    }
  }
}
} // namespace
