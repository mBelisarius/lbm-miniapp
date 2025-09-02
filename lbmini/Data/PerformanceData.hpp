#ifndef LBMINI_DATA_PERFORMANCE_DATA_HPP_
#define LBMINI_DATA_PERFORMANCE_DATA_HPP_

#include <Eigen/Dense>
#include "Data/DataEnums.hpp"

namespace lbmini {
struct PerformanceData {
  using Index = Eigen::Index;

  TargetEnum target;   // Target to use (CPU; GPU)
  BackendEnum backend; // Backend to use (Plain; OpenMP)
  Index cores;         // Number of CPU cores to use (0 for auto)
};
} // namespace lbmini

#endif  // LBMINI_DATA_PERFORMANCE_DATA_HPP_
