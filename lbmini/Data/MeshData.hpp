#ifndef LBMINI_DATA_MESH_DATA_HPP_
#define LBMINI_DATA_MESH_DATA_HPP_

#include <Eigen/Dense>

namespace lbmini {

template <typename Scalar_, Eigen::Index dim_>
struct MeshData {
  using Index = Eigen::Index;

  Eigen::Array<Scalar_, dim_, 1> lenght;
  Eigen::Array<Index, dim_, 1> size;

  Scalar_ lx() const { return lenght[0]; }

  Scalar_ ly() const {
    if constexpr (dim_ >= 2)
      return lenght[1];

    return 0;
  }

  Scalar_ lz() const {
    if constexpr (dim_ >= 3)
      return lenght[2];

    return 0;
  }

  Index nx() const { return size[0]; }

  Index ny() const {
    if constexpr (dim_ >= 2)
      return size[1];

    return 0;
  }

  Index nz() const {
    if constexpr (dim_ >= 3)
      return size[2];

    return 0;
  }
};

}  // namespace lbmini

#endif  // LBMINI_DATA_MESH_DATA_HPP_
