// CPU roofline probes: AVX2 FMA peak (per thread and for T threads), triad / copy bandwidth, and the divider throughput.
//
//   g++ -O3 -march=native -fopenmp cpu_roof.cpp -o cpu_roof
//   OMP_PLACES=cores OMP_PROC_BIND=close ./cpu_roof [threads]
//
// The numbers are the ceilings that docs/performance.md compares the solver against. Run on a quiet machine.
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template<int OP>
static double chains(int threads, long iters) {
  double best = 1e30;
  for (int rep = 0; rep < 5; ++rep) {
    double t0 = now();
    double sink = 0;
#pragma omp parallel num_threads(threads) reduction(+ : sink)
    {
      __m256d x[12];
      for (int k = 0; k < 12; ++k) x[k] = _mm256_set1_pd(1.0 + k * 1e-3 + omp_get_thread_num());
      const __m256d a = _mm256_set1_pd(1.0000001), b = _mm256_set1_pd(1e-9);
      for (long i = 0; i < iters; ++i) {
#pragma GCC unroll 12
        for (int k = 0; k < 12; ++k) {
          if (OP == 0) x[k] = _mm256_fmadd_pd(x[k], a, b);
          if (OP == 1) x[k] = _mm256_add_pd(x[k], b);
          if (OP == 2) x[k] = _mm256_mul_pd(x[k], a);
          if (OP == 3) x[k] = _mm256_div_pd(b, x[k]);
        }
      }
      __m256d s = x[0];
      for (int k = 1; k < 12; ++k) s = _mm256_add_pd(s, x[k]);
      double tmp[4];
      _mm256_storeu_pd(tmp, s);
      sink += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
    double dt = now() - t0;
    if (dt < best) best = dt;
    if (sink == 12345.678) printf("x");
  }
  return best;
}

int main(int argc, char** argv) {
  int threads = argc > 1 ? atoi(argv[1]) : 1;
  long iters = 20000000;
  double t = chains<0>(threads, iters);
  double fmaFlops = 2.0 * 4 * 12 * iters * threads / t;
  printf("FMA   %2d thread(s): %7.1f GFLOP/s  (%.2f GHz-equivalent at 16 flop/cycle per thread)\n", threads, fmaFlops / 1e9,
         fmaFlops / threads / 16e9);
  t = chains<1>(threads, iters);
  printf("ADD   %2d thread(s): %7.1f G ymm-add/s\n", threads, 12.0 * iters * threads / t / 1e9);
  t = chains<2>(threads, iters);
  printf("MUL   %2d thread(s): %7.1f G ymm-mul/s\n", threads, 12.0 * iters * threads / t / 1e9);
  t = chains<3>(threads, iters / 20);
  printf("DIV   %2d thread(s): %7.1f G ymm-div/s  (%.2f cycles per ymm vdivpd at 4.5 GHz)\n", threads,
         12.0 * (iters / 20) * threads / t / 1e9, 4.5e9 * threads / (12.0 * (iters / 20) * threads / t));

  const size_t n = size_t(1) << 26; // 512 MiB per array
  std::vector<double> a(n), b(n), c(n);
#pragma omp parallel for num_threads(threads) schedule(static)
  for (size_t i = 0; i < n; ++i) { a[i] = 0; b[i] = 1; c[i] = 2; }
  double best = 1e30;
  for (int rep = 0; rep < 6; ++rep) {
    double t0 = now();
#pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < n; ++i) a[i] = b[i] + 3.0 * c[i];
    double dt = now() - t0;
    if (dt < best) best = dt;
  }
  printf("TRIAD %2d thread(s): %7.1f GB/s (24 B per element, write-allocate not counted)\n", threads, 24.0 * n / best / 1e9);
  best = 1e30;
  for (int rep = 0; rep < 6; ++rep) {
    double t0 = now();
#pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < n; ++i) a[i] = b[i];
    double dt = now() - t0;
    if (dt < best) best = dt;
  }
  printf("COPY  %2d thread(s): %7.1f GB/s (16 B per element)\n", threads, 16.0 * n / best / 1e9);
  return a[n / 2] == 12345 ? 1 : 0;
}
