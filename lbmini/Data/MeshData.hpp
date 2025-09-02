#ifndef LBMINI_DATA_MESH_DATA_HPP_
#define LBMINI_DATA_MESH_DATA_HPP_

#include <Eigen/Dense>

namespace lbmini {

template <typename Scalar, Eigen::Index Dim>
struct MeshData {
  using Index = Eigen::Index;

  Scalar lenght[Dim];
  Index size[Dim];

  Scalar lx() const { return lenght[0]; }

  Scalar ly() const {
    if constexpr (Dim >= 2)
      return lenght[1];

    return 0;
  }

  Scalar lz() const {
    if constexpr (Dim >= 3)
      return lenght[2];

    return 0;
  }

  Index nx() const { return size[0]; }

  Index ny() const {
    if constexpr (Dim >= 2)
      return size[1];

    return 0;
  }

  Index nz() const {
    if constexpr (Dim >= 3)
      return size[2];

    return 0;
  }
};

}  // namespace lbmini

#endif  // LBMINI_DATA_MESH_DATA_HPP_
