// Harness de contagem de operações do miniaplicativo.
// Os corpos dos kernels são extraídos *literalmente* do código-fonte por
// roofline-count.py e incluídos abaixo; nada é reescrito à mão.
//
// Modo COUNTING: o tipo escalar conta +, -, *, / e sqrt (1 FLOP cada) e
//                chamadas a exp/log; comparações, fabs e floor valem 0.
// Modo double  : mesmo código com double nativo, para execução sob valgrind.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct Counters {
  unsigned long long add = 0, mul = 0, div = 0, sqrt = 0, exp = 0, log = 0;
  unsigned long long Basic() const { return add + mul + div + sqrt; }
};
static Counters g;

#ifdef COUNTING
struct CD {
  double v;
  constexpr CD() : v(0) {}
  constexpr CD(double x) : v(x) {}
  explicit operator int() const { return static_cast<int>(v); }
  explicit operator double() const { return v; }
};
inline CD operator+(CD a, CD b) { ++g.add; return CD(a.v + b.v); }
inline CD operator-(CD a, CD b) { ++g.add; return CD(a.v - b.v); }
inline CD operator*(CD a, CD b) { ++g.mul; return CD(a.v * b.v); }
inline CD operator/(CD a, CD b) { ++g.div; return CD(a.v / b.v); }
inline CD operator-(CD a) { return CD(-a.v); }
inline CD& operator+=(CD& a, CD b) { ++g.add; a.v += b.v; return a; }
inline CD& operator-=(CD& a, CD b) { ++g.add; a.v -= b.v; return a; }
inline CD& operator*=(CD& a, CD b) { ++g.mul; a.v *= b.v; return a; }
inline CD& operator/=(CD& a, CD b) { ++g.div; a.v /= b.v; return a; }
inline bool operator<(CD a, CD b) { return a.v < b.v; }
inline bool operator>(CD a, CD b) { return a.v > b.v; }
inline bool operator<=(CD a, CD b) { return a.v <= b.v; }
inline bool operator>=(CD a, CD b) { return a.v >= b.v; }
inline bool operator==(CD a, CD b) { return a.v == b.v; }
namespace std {
inline CD exp(CD x) { ++g.exp; return CD(::exp(x.v)); }
inline CD log(CD x) { ++g.log; return CD(::log(x.v)); }
inline CD sqrt(CD x) { ++g.sqrt; return CD(::sqrt(x.v)); }
inline CD fabs(CD x) { return CD(::fabs(x.v)); }
inline CD floor(CD x) { return CD(::floor(x.v)); }
} // namespace std
using S = CD;
inline double Val(S x) { return x.v; }
#else
using S = double;
inline double Val(S x) { return x; }
#endif

#define LBMINI_UNROLL(n)

#include "Lbm/OpenMp/Cpu/LatticeD2Q9.hpp"

using Index = long;

// ------------------------------------------------------------ parâmetros
// Valores de lbmini/config.yaml e de Data/DataReader.hpp.
struct Fluid {
  S viscosity{1.0e-2};
  S specificHeatCv{2.5};
  S specificHeatCp{3.5};
  S constant{1.0};
  S conductivity{3.5 * 1.0e-2 / 0.71};
};
struct Control {
  S Ux{0.25}, Uy{0.0};
  S U(Index d) const { return d == 0 ? Ux : Uy; }
};

static std::mt19937_64 rng(12345);
static double Rand() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng); }

// Estado de Sod: metade esquerda (rho=1, p=1), metade direita (rho=0,125, p=0,1);
// 10% das células recebem velocidade não nula (região da onda).
static void InitState(std::vector<S>& rho, std::vector<S>& T, std::vector<S>& u,
                      std::vector<S>& f, std::vector<S>& g, Index N, Index nQ) {
  for (Index c = 0; c < N; ++c) {
    const bool left = c < N / 2;
    const double r = left ? 1.0 : 0.125;
    const double t = left ? 1.0 : 0.8;
    const double ux = (Rand() < 0.1) ? 0.15 * (Rand() - 0.5) : 0.0;
    const double uy = (Rand() < 0.1) ? 0.05 * (Rand() - 0.5) : 0.0;
    rho[c] = r;
    T[c] = t;
    u[0 * N + c] = ux;
    u[1 * N + c] = uy;
    const double E = 2.5 * t + 0.5 * (ux * ux + uy * uy);
    for (Index i = 0; i < nQ; ++i) {
      f[i * N + c] = r / 9.0 * (1.0 + 0.02 * (Rand() - 0.5));
      g[i * N + c] = 2.0 * r * E / 9.0 * (1.0 + 0.02 * (Rand() - 0.5));
    }
  }
}

static void Report(const char* name, const Counters& a, const Counters& b, Index N) {
  const double n = static_cast<double>(N);
  std::printf("%s add=%.4f mul=%.4f div=%.4f sqrt=%.4f exp=%.4f log=%.4f basic=%.4f\n", name,
              (b.add - a.add) / n, (b.mul - a.mul) / n, (b.div - a.div) / n,
              (b.sqrt - a.sqrt) / n, (b.exp - a.exp) / n, (b.log - a.log) / n,
              (b.Basic() - a.Basic()) / n);
}

// ------------------------------------------------------------ caminho CPU
// Cópia fiel dos membros usados pelos corpos de OpenMp/Cpu/LbmTube.hpp.
struct CpuTube {
  using Scalar = S;
  using LatticeType = lbmini::openmp::cpu::LatticeD2Q9<S>;
  static constexpr Index kQ_ = 9, kDim_ = 2;
  static constexpr S kTiny_{1.0e-12};
  static constexpr S kMaxExp_{700.0};
  Fluid kFluid_;
  Control kControl_;
  Index N_ = 0;
  struct Buf {
    std::vector<S> v;
    S* data() { return v.data(); }
  };
  struct BufI {
    std::vector<std::int32_t> v;
    std::int32_t* data() { return v.data(); }
  };
  Buf rho_, p_, tem_, u_, f_, g_, faux_, gaux_, lastGx_, streamW_;
  BufI streamIdx_;
  S* fCur_ = nullptr;
  S* gCur_ = nullptr;
  S* fAlt_ = nullptr;
  S* gAlt_ = nullptr;

  void Init(Index n) {
    N_ = n;
    rho_.v.resize(N_);
    p_.v.resize(N_);
    tem_.v.resize(N_);
    u_.v.resize(2 * N_);
    f_.v.resize(kQ_ * N_);
    g_.v.resize(kQ_ * N_);
    faux_.v.resize(kQ_ * N_);
    gaux_.v.resize(kQ_ * N_);
    lastGx_.v.assign((kDim_ + 1) * N_, S{0});
    streamW_.v.resize(kQ_ * N_ * 4);
    streamIdx_.v.resize(kQ_ * N_ * 4);
    for (Index i = 0; i < kQ_; ++i)
      for (Index c = 0; c < N_; ++c)
        for (int k = 0; k < 4; ++k) {
          const Index off = (k == 0) ? 0 : (k == 1 ? 1 : (k == 2 ? 100 : 101));
          streamIdx_.v[(i * N_ + c) * 4 + k] = static_cast<std::int32_t>(std::min(c + off, N_ - 1));
          streamW_.v[(i * N_ + c) * 4 + k] = S{0.25};
        }
    InitState(rho_.v, tem_.v, u_.v, f_.v, g_.v, N_, kQ_);
    fCur_ = f_.data();
    gCur_ = g_.data();
    fAlt_ = faux_.data();
    gAlt_ = gaux_.data();
  }

  void collisionAndEquilibria() {
#include "cpu_collision.inc"
  }
  void stream() {
#include "cpu_stream.inc"
  }
  void streamAndMacroscopic() {
#include "cpu_streammacro.inc"
  }
};

// ------------------------------------------------------------ caminho CUDA
#define __global__
struct Dim1 { int x = 0; };
static Dim1 blockIdx, blockDim, threadIdx;
static int kD2Q9Cx[9] = { 0, 1, -1, 0, 0, 1, -1, 1, -1 };
static int kD2Q9Cy[9] = { 0, 0, 0, 1, -1, 1, -1, -1, 1 };

#include "cuda_collision.inc"
#include "cuda_stream.inc"

template<class F>
static void Launch(Index N, F&& kernel) {
  blockDim.x = 64;
  for (Index cell = 0; cell < N; ++cell) {
    blockIdx.x = static_cast<int>(cell / 64);
    threadIdx.x = static_cast<int>(cell % 64);
    kernel();
  }
}

// Pontos de entrada para a contagem de instruções executadas (gdb): cada um
// processa uma única célula, com o código otimizado do compilador.
struct ProbeState {
  CpuTube t;
  std::vector<S> rho, p, T, u, f, g, fa, ga, lastGx;
  int nx = 1, ny = 1, N = 1;
};
static ProbeState* ps;
extern "C" __attribute__((noinline)) void probe_cpu_collision() { ps->t.collisionAndEquilibria(); }
extern "C" __attribute__((noinline)) void probe_cpu_stream() { ps->t.stream(); }
extern "C" __attribute__((noinline)) void probe_cpu_streammacro() { ps->t.streamAndMacroscopic(); }
extern "C" __attribute__((noinline)) void probe_cuda_collision() {
  const Fluid fl;
  const Control ct;
  collisionAndEquilibriaKernel<S>(ps->f.data(), ps->g.data(), ps->lastGx.data(), ps->rho.data(), ps->T.data(),
                                  ps->u.data(), ct.Ux, ct.Uy, fl.viscosity, fl.conductivity,
                                  fl.specificHeatCp, fl.specificHeatCv, S{1.0e-12}, S{700.0}, ps->nx, ps->ny,
                                  ps->N);
}
extern "C" __attribute__((noinline)) void probe_cuda_streammacro() {
  const Fluid fl;
  const Control ct;
  streamAndMacroscopicKernel<S, 0>(ps->f.data(), ps->g.data(), ps->fa.data(), ps->ga.data(), ps->rho.data(),
                                   ps->p.data(), ps->T.data(), ps->u.data(), ct.Ux, ct.Uy,
                                   S{1} / fl.specificHeatCv, fl.constant, S{1.0e-12}, S{std::sqrt(1.4)}, ps->nx,
                                   ps->ny, ps->N);
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "all";
  const Index N = argc > 2 ? std::atol(argv[2]) : 20000;
  const int ny = 100;
  const int nx = static_cast<int>(N / ny);

  if (mode == "probe") {
    // probe <N=1> <estado: 0 = repouso, 1 = em movimento> <kernel>
    ps = new ProbeState;
    ps->N = 1;
    ps->t.Init(1);
    ps->rho.assign(1, S{1}); ps->p.resize(1); ps->T.assign(1, S{1});
    ps->u.assign(2, S{0});
    ps->f.resize(9); ps->g.resize(9); ps->fa.resize(9); ps->ga.resize(9);
    ps->lastGx.assign(3, S{0});
    const int moving = std::atoi(argv[3]);
    const std::string which = argv[4];
    InitState(ps->rho, ps->T, ps->u, ps->f, ps->g, 1, 9);
    if (moving) {
      ps->u[0] = 0.07;
      ps->u[1] = -0.02;
    } else {
      ps->u[0] = 0.0;
      ps->u[1] = 0.0;
    }
    ps->t.rho_.v[0] = ps->rho[0];
    ps->t.tem_.v[0] = ps->T[0];
    ps->t.u_.v[0] = ps->u[0];
    ps->t.u_.v[1] = ps->u[1];
    if (which == "cpu_collision") probe_cpu_collision();
    if (which == "cpu_stream") probe_cpu_stream();
    if (which == "cpu_streammacro") probe_cpu_streammacro();
    if (which == "cuda_collision") probe_cuda_collision();
    if (which == "cuda_streammacro") probe_cuda_streammacro();
    std::printf("%g\n", Val(ps->f[0]) + Val(ps->t.f_.v[0]) + Val(ps->fa[0]) + Val(ps->t.faux_.v[0]));
    return 0;
  }

  if (mode == "all" || mode == "cpu") {
    CpuTube t;
    t.Init(N);
    Counters a = g;
    t.collisionAndEquilibria();
    Counters b = g;
    Report("cpu_collision", a, b, N);
    a = g;
    t.stream();
    b = g;
    Report("cpu_stream", a, b, N);
    a = g;
    t.streamAndMacroscopic();
    b = g;
    Report("cpu_stream+macro", a, b, N);
  }

  if (mode == "all" || mode == "cuda") {
    std::vector<S> rho(N), p(N), T(N), u(2 * N), f(9 * N), gg(9 * N), fa(9 * N), ga(9 * N);
    std::vector<S> lastGx(3 * N, S{0});
    InitState(rho, T, u, f, gg, N, 9);
    const Fluid fl;
    const Control ct;
    const S kTiny{1.0e-12}, kMaxE{700.0};
    const S idw{std::sqrt(1.4)};
    const S invCv = S{1} / fl.specificHeatCv;
    Counters a = g;
    Launch(N, [&] {
      collisionAndEquilibriaKernel<S>(f.data(), gg.data(), lastGx.data(), rho.data(), T.data(), u.data(),
                                      ct.Ux, ct.Uy, fl.viscosity, fl.conductivity, fl.specificHeatCp,
                                      fl.specificHeatCv, kTiny, kMaxE, nx, ny, static_cast<int>(N));
    });
    Counters b = g;
    Report("cuda_collision", a, b, N);
    a = g;
    Launch(N, [&] {
      streamAndMacroscopicKernel<S, 0>(f.data(), gg.data(), fa.data(), ga.data(), rho.data(), p.data(),
                                       T.data(), u.data(), ct.Ux, ct.Uy, invCv, fl.constant, kTiny, idw, nx,
                                       ny, static_cast<int>(N));
    });
    b = g;
    Report("cuda_stream+macro", a, b, N);
  }
  double chk = 0;
  (void)chk;
  return 0;
}
