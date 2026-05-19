// Chamada única a exp/log da glibc, para contagem de instruções sob gdb
// (roofline-gdbcount.py). O resultado é impresso para impedir a eliminação.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
  const double x = std::atof(argv[2]);
  volatile double in = x;
  double r;
  if (std::strcmp(argv[1], "exp") == 0)
    r = std::exp(in);
  else
    r = std::log(in);
  std::printf("%.17g\n", r);
  return 0;
}
