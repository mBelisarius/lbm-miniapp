#ifndef LBMINI_DATA_PERFORMANCE_DATA_HPP_
#define LBMINI_DATA_PERFORMANCE_DATA_HPP_

#include <Eigen/Dense>

namespace lbmini {

struct PerformanceData {
  using Index = Eigen::Index;

  Index backend;   // Backend to use (0: plain; 1: OpenMP)
  Index cores;     // Number of cores to use (0 for auto)
};

}  // namespace lbmini

#endif  // LBMINI_DATA_PERFORMANCE_DATA_HPP_
