#ifndef LBMINI_MESH_DATA_HPP_
#define LBMINI_MESH_DATA_HPP_

#include <Eigen/Dense>

namespace lbmini {

template <typename Scalar_, Eigen::Index dim_>
struct MeshData {
  using Index = Eigen::Index;

  Eigen::Vector<Scalar_, dim_> lenght;
  Eigen::Vector<Index, dim_> size;

  Scalar_ lx() { return lenght[0]; }

  Scalar_ ly() { return lenght[1]; }

  Scalar_ lz() { return lenght[2]; }

  Index nx() { return size[0]; }

  Index ny() { return size[1]; }

  Index nz() { return size[2]; }
};

}  // namespace lbmini

#endif  // LBMINI_MESH_DATA_HPP_
