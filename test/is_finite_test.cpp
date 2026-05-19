// The Release configurations are built with -ffast-math, which folds
// std::isfinite() to `true`. lbmini::IsFinite() is what the solvers and the
// memory-safety suite use instead; these cases fail if it ever regresses to a
// check the compiler is allowed to delete.

#include <bit>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "Utils.hpp"

TEST(IsFiniteTest, AcceptsOrdinaryValues) {
  EXPECT_TRUE(lbmini::IsFinite(0.0));
  EXPECT_TRUE(lbmini::IsFinite(-0.0));
  EXPECT_TRUE(lbmini::IsFinite(1.5));
  EXPECT_TRUE(lbmini::IsFinite(-1e300));
  EXPECT_TRUE(lbmini::IsFinite(std::numeric_limits<double>::denorm_min()));
  EXPECT_TRUE(lbmini::IsFinite(std::numeric_limits<double>::max()));
  EXPECT_TRUE(lbmini::IsFinite(1.5f));
  EXPECT_TRUE(lbmini::IsFinite(std::numeric_limits<float>::max()));
}

TEST(IsFiniteTest, RejectsInfinityAndNaN) {
  // Built from bit patterns so the values exist even if the compiler assumes
  // they never do.
  const double posInf = std::bit_cast<double>(std::uint64_t{ 0x7FF0000000000000ull });
  const double negInf = std::bit_cast<double>(std::uint64_t{ 0xFFF0000000000000ull });
  const double qNaN = std::bit_cast<double>(std::uint64_t{ 0x7FF8000000000000ull });
  const double sNaN = std::bit_cast<double>(std::uint64_t{ 0x7FF0000000000001ull });
  EXPECT_FALSE(lbmini::IsFinite(posInf));
  EXPECT_FALSE(lbmini::IsFinite(negInf));
  EXPECT_FALSE(lbmini::IsFinite(qNaN));
  EXPECT_FALSE(lbmini::IsFinite(sNaN));

  const float fInf = std::bit_cast<float>(std::uint32_t{ 0x7F800000u });
  const float fNaN = std::bit_cast<float>(std::uint32_t{ 0x7FC00000u });
  EXPECT_FALSE(lbmini::IsFinite(fInf));
  EXPECT_FALSE(lbmini::IsFinite(fNaN));
}
