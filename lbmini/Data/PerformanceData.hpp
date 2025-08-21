#ifndef LBMINI_DATA_PERFORMANCE_DATA_HPP_
#define LBMINI_DATA_PERFORMANCE_DATA_HPP_

namespace lbmini {

struct PerformanceData {
  using Index = Eigen::Index;

  Index cores;     // Number of cores to use (0 for auto)
  Index tileSize;  // Tile size for cache blocking
};

}  // namespace lbmini

#endif  // LBMINI_DATA_PERFORMANCE_DATA_HPP_
