#ifndef LBMINI_UTILS_HPP_
#define LBMINI_UTILS_HPP_

namespace lbmini {

template<typename Scalar_, typename Index>
Scalar_ Kronecker(Index a, Index b) {
  return (a == b) ? 1.0 : 0.0;
};

}

#endif // LBMINI_UTILS_HPP_
