// GPU roofline probes: FP64 FMA / ADD / MUL throughput, FP64 division and exp rate, device-memory copy bandwidth.
//
//   nvcc -O3 -arch=sm_89 gpu_roof.cu -o gpu_roof && ./gpu_roof
//
// The numbers are the ceilings that docs/performance.md compares the solver against. Run on an idle GPU.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

template<int OP>
__global__ void chains(double* out, int iters, double a, double b) {
  double x[8];
  for (int k = 0; k < 8; ++k) x[k] = threadIdx.x * 1e-3 + k;
#pragma unroll 4
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (OP == 0) x[k] = __fma_rn(x[k], a, b);
      if (OP == 1) x[k] = __dadd_rn(x[k], b);
      if (OP == 2) x[k] = __dmul_rn(x[k], a);
      if (OP == 3) x[k] = b / x[k];          // IEEE division
      if (OP == 4) x[k] = exp(x[k] * 1e-9);  // library exp
    }
  }
  double s = 0;
  for (int k = 0; k < 8; ++k) s += x[k];
  if (s == 123456789.0) out[0] = s;
}

__global__ void copyKernel(const double2* __restrict__ a, double2* __restrict__ b, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) b[i] = a[i];
}

template<int OP>
double run(const char* name, double opsPerIter, int iters, int sms) {
  double* out;
  CK(cudaMalloc(&out, 64));
  int blocks = sms * 12, threads = 128;
  chains<OP><<<blocks, threads>>>(out, 16, 1.0000001, 1e-7);
  CK(cudaDeviceSynchronize());
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    cudaEventRecord(t0);
    chains<OP><<<blocks, threads>>>(out, iters, 1.0000001, 1e-7);
    cudaEventRecord(t1);
    CK(cudaEventSynchronize(t1));
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    if (ms < best) best = ms;
  }
  double ops = (double)blocks * threads * iters * 8 * opsPerIter;
  printf("%-10s %9.2f Gop/s (thread-level instructions)\n", name, ops / best / 1e6);
  cudaFree(out);
  return ops / best / 1e6;
}

int main() {
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  int clk = 0, memclk = 0;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, 0);
  printf("%s, %d SMs, sm clock max %.0f MHz, mem clock %.0f MHz, bus %d bit, L2 %d KiB\n", prop.name, prop.multiProcessorCount,
         clk / 1e3, memclk / 1e3, prop.memoryBusWidth, prop.l2CacheSize / 1024);
  printf("theoretical DRAM bandwidth %.1f GB/s\n", 2.0 * memclk * 1e3 * prop.memoryBusWidth / 8 / 1e9);
  int sms = prop.multiProcessorCount;
  double fma = run<0>("DFMA", 1, 200000, sms);
  run<1>("DADD", 1, 200000, sms);
  run<2>("DMUL", 1, 200000, sms);
  run<3>("DDIV", 1, 20000, sms);
  run<4>("DEXP", 1, 20000, sms);
  printf("=> FP64 FMA peak %.1f GFLOP/s\n", fma * 2);

  size_t n = (size_t)1 << 26; // 2^26 double2 = 1 GiB each
  double2 *a, *b;
  CK(cudaMalloc(&a, n * sizeof(double2)));
  CK(cudaMalloc(&b, n * sizeof(double2)));
  CK(cudaMemset(a, 1, n * sizeof(double2)));
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  double best = 1e30;
  for (int r = 0; r < 8; ++r) {
    cudaEventRecord(t0);
    copyKernel<<<sms * 16, 256>>>(a, b, n);
    cudaEventRecord(t1);
    CK(cudaEventSynchronize(t1));
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    if (ms < best) best = ms;
  }
  printf("copy bandwidth %.1f GB/s (read+write)\n", 2.0 * n * sizeof(double2) / (best * 1e-3) / 1e9);
  return 0;
}
