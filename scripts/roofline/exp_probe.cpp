// Custo da exponencial vetorial própria (uma chamada sobre um registrador ymm de quatro valores), para a contagem de operações.
#include <cstdio>

#include "Lbm/Simd.hpp"

using S = lbmini::simd::Ops<double, 4>;

__attribute__((noinline)) S::V ExpProbe(const S::V x) {
  return S::Exp(x);
}

int main(int argc, char**) {
  S::V x = S::Splat(-0.37 * argc);
  S::V y = ExpProbe(x);
  std::printf("%g\n", S::Lane(y, 0));
  return 0;
}
