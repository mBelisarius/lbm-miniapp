#ifndef LBMINI_ALPAKA_LBMTUBEIMPL_HPP_
#define LBMINI_ALPAKA_LBMTUBEIMPL_HPP_

/**
 * @file LbmTubeImpl.hpp
 * @brief alpaka3 implementation behind `lbmini::alpaka::LbmTube` (`detail::Engine`).
 *
 * Only included by the two explicit-instantiation translation units
 * (`Cpu/LbmTube.cpp`, `Gpu/LbmTube.cpp`), which belong to `alpaka_finalize()`d
 * CMake targets. Never include it from code that is not compiled through
 * alpaka.
 *
 * ### One engine, two implementations
 *
 * `Engine::State` is chosen by the target:
 *
 *  - `TargetEnum::GPU`: `DeviceState`, defined here. One fused per-cell kernel
 *    per time step, described below; every thread owns one cell.
 *  - `TargetEnum::CPU`: `HostState`, defined in `Cpu/LbmTubeImpl.hpp` (which
 *    `Cpu/LbmTube.cpp` includes after this file). A CPU gains nothing from a
 *    thread per cell: it runs the vectorised row-segment kernel shared by every
 *    host backend (`Lbm/HostKernel.hpp`, x-fastest padded layout), and alpaka
 *    only allocates the buffers and launches one block per OpenMP thread. The
 *    per-cell code below is never instantiated for it, and the GCC
 *    vector-extension code of the host kernel is never seen by the CUDA compiler.
 *
 * ### Device target: data layout and memory traffic
 *
 * All per-cell state lives in device buffers in structure-of-arrays form:
 *
 * | buffer       | size     | layout                  | role                              |
 * | :----------- | :------- | :---------------------- | :-------------------------------- |
 * | `fA/fB`      | `Q * N`  | `f[idc * N + cell]`     | post-collision density (ping-pong) |
 * | `gA/gB`      | `Q * N`  | `g[idc * N + cell]`     | post-collision energy (ping-pong)  |
 * | `xi`         | `D * N`  | `xi[d * N + cell]`      | Newton-Raphson warm start          |
 * | `rho/p/tem`  | `N`      | `x[cell]`               | macroscopic output                 |
 * | `u`          | `D * N`  | `u[d * N + cell]`       | macroscopic output                 |
 *
 * `cell = i * ny + j` (row-major, `j` fastest), so a direction plane is
 * contiguous: consecutive GPU threads / consecutive iterations of a CPU thread
 * touch consecutive addresses for every load and store.
 *
 * One time step is a single kernel (`StepKernel`) that *pulls* the
 * post-collision populations of the previous step from the four IDW source
 * cells of each direction, reduces them to the macroscopic fields, evaluates
 * the equilibria and the Newton-Raphson energy closure and relaxes — all in
 * registers — and writes the new post-collision populations to the other
 * ping-pong buffer. Per cell and step that is 2 x 9 x 8 B of streamed reads
 * (the neighbour reuse is served by the caches), 2 x 9 x 8 B of writes and
 * 2 x 2 x 8 B for the warm start: ~320 B, against ~700 B for the classic
 * collide / stream+macroscopic pair plus 432 B of streaming tables.
 *
 * The state of the solver after `n` steps is therefore kept as *post*-collision
 * populations: `Init()` runs the first collision itself, and every step's
 * collision prepares the next one. The observable state — the macroscopic
 * fields after the n-th streaming — is identical to the reference
 * collide-then-stream sequence; only the internal warm start is one collision
 * ahead, which is exactly the value the next collision would read anyway.
 *
 * ### Streaming stencil
 *
 * The reference solver precomputes, for every `(cell, idc)`, four source
 * indices and four inverse-distance weights. With a uniform lattice shift `U`
 * those are the same for every cell up to a clamp in x and a wrap in y, so the
 * kernel reads the shared per-direction `lbmini::DirectionStencil`
 * (`Lbm/Stencil.hpp`) instead. Indices are in range by construction (clamp /
 * wrap / explicit range test), so there is no table to verify and nothing to
 * read per cell. The host kernel uses the same stencil.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "Guards.hpp"
#include "Lbm/Alpaka/LbmTube.hpp"
#include "Lbm/Stencil.hpp"

/// Unroll a short, fixed-trip loop (directions / dimensions). Fully unrolled
/// loops turn every lattice lookup into an immediate and keep the per-cell
/// arrays in registers instead of GPU local memory. `LBMINI_ALPAKA_NO_UNROLL`
/// keeps the Newton iteration rolled (code size; the iterations are sequential).
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__clang__)
#define LBMINI_ALPAKA_UNROLL _Pragma("unroll")
#define LBMINI_ALPAKA_NO_UNROLL _Pragma("unroll 1")
#elif defined(__GNUC__)
#define LBMINI_ALPAKA_UNROLL _Pragma("GCC unroll 16")
#define LBMINI_ALPAKA_NO_UNROLL _Pragma("GCC unroll 1")
#else
#define LBMINI_ALPAKA_UNROLL
#define LBMINI_ALPAKA_NO_UNROLL
#endif

namespace lbmini::alpaka::detail {
/**
 * @brief Maps a solver target onto an alpaka device, executor and queue.
 *
 * Each specialisation exists only where alpaka can provide it, so the CPU
 * translation unit never needs CUDA and the device pass of the GPU one never
 * needs OpenMP.
 */
template<TargetEnum Target>
struct TargetTraits;

#if ALPAKA_OMP
template<>
struct TargetTraits<TargetEnum::CPU> {
  static constexpr const char* kName = "alpaka::cpu";
  static constexpr auto kApi = ::alpaka::api::host;
  static constexpr auto kDeviceKind = ::alpaka::deviceKind::cpu;
  static constexpr auto kExecutor = ::alpaka::exec::cpuOmpBlocks;
  // Blocking: the kernel runs on the calling thread, so its OpenMP team obeys
  // the thread count `Executor` set with `omp_set_num_threads()` (an ICV of the
  // calling thread, which a non-blocking queue's worker thread would not
  // inherit) and `OMP_PLACES` / `OMP_PROC_BIND` pin it as for every other CPU
  // backend. It also makes a launch return only when every block has finished,
  // which `HostState` relies on to reset its work counter between steps.
  static constexpr auto kQueueKind = ::alpaka::queueKind::blocking;
};
#endif

#if ALPAKA_LANG_CUDA
template<>
struct TargetTraits<TargetEnum::GPU> {
  static constexpr const char* kName = "alpaka::gpu";
  static constexpr auto kApi = ::alpaka::api::cuda;
  static constexpr auto kDeviceKind = ::alpaka::deviceKind::nvidiaGpu;
  static constexpr auto kExecutor = ::alpaka::exec::gpuCuda;
  // Kernels and copies are enqueued asynchronously; `Run()` waits once at the
  // end so the benchmark timer measures finished device work.
  static constexpr auto kQueueKind = ::alpaka::queueKind::nonBlocking;

  /// One thread per cell. The fused step kernel needs ~150 registers per
  /// thread, so it must be launched with at most 65536 / 150 ~ 430 threads per
  /// block (alpaka's generic default of 512 would not be launchable). 128-thread
  /// blocks are the best fit: three of them (12 warps) are resident per SM,
  /// whereas a 256-thread block leaves registers for only one (8 warps), which
  /// hides less of the double-precision latency (measured ~5% slower).
  static constexpr std::int32_t kBlockSize = 128;

  static auto FrameSpec(const auto& /*device*/, const std::int32_t cells) {
    const std::int32_t blocks = (cells + kBlockSize - 1) / kBlockSize;
    return ::alpaka::onHost::FrameSpec{ ::alpaka::Vec{ blocks }, ::alpaka::Vec{ kBlockSize }, kExecutor };
  }
};
#endif

/// Everything a kernel needs besides the buffers; trivially copyable, copied
/// once to device memory and passed by pointer.
template<typename Scalar, int Q>
struct Params {
  std::int32_t nx;
  std::int32_t ny;
  std::int32_t cells;

  Scalar ushiftX;
  Scalar ushiftY;

  Scalar cv;
  Scalar invCv;
  Scalar rgas;
  Scalar viscosity;
  Scalar condOverCp; ///< `conductivity / cp`: `kappa / (rho cp) = condOverCp / rho`

  Scalar densityL;
  Scalar pressureL;
  Scalar densityR;
  Scalar pressureR;

  DirectionStencil<Scalar, Q> stencil;
};

/**
 * @brief Per-cell LBM physics, shared by the kernels.
 *
 * Every function operates on register arrays of one cell. The arithmetic is
 * that of the plain reference except for the energy closure, which uses its
 * factorised form (`AxisExp`) and agrees with the reference to rounding error.
 */
template<typename Scalar, typename LatticeType>
struct Physics {
  static constexpr int kQ = static_cast<int>(LatticeType::Speeds());
  static constexpr int kD = static_cast<int>(LatticeType::Dim());
  static constexpr Scalar kTiny = Scalar{ 1.0e-12 };
  static constexpr Scalar kMaxExp = Scalar{ 700.0 };
  static constexpr int kMaxIter = 3;

  static_assert(kD == 2, "the alpaka backend implements the 2-D closure");

  ALPAKA_FN_HOST_ACC static constexpr int Vx(const int idc) { return static_cast<int>(LatticeType::Velocity(idc, 0)); }

  ALPAKA_FN_HOST_ACC static constexpr int Vy(const int idc) { return static_cast<int>(LatticeType::Velocity(idc, 1)); }

  /**
   * @brief `LatticeType::Weights(idc, tem)` for all directions.
   *
   * The reference evaluates `w = 1; w *= wZero + isNonZero * (wNonZero - wZero)`
   * per dimension, with `isNonZero` in {0, 1}: the factor is exactly `wZero` for
   * a zero component and `wZero + (wNonZero - wZero)` otherwise.
   */
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static void Weights(const Scalar tem, Scalar (&w)[kQ]) {
    const Scalar wZero = Scalar{ 1 } - tem;
    const Scalar wNonZero = Scalar{ 0.5 } * tem;
    const Scalar factorNonZero = wZero + (wNonZero - wZero);
    LBMINI_ALPAKA_UNROLL
    for (int idc = 0; idc < kQ; ++idc)
      w[idc] = (Vx(idc) == 0 ? wZero : factorNonZero) * (Vy(idc) == 0 ? wZero : factorNonZero);
  }

  /// Density equilibrium, factorised product form.
  ALPAKA_FN_HOST_ACC static void Feq(
    const Scalar rhoc,
    const Scalar ux,
    const Scalar uy,
    const Scalar temc,
    const Scalar ushiftX,
    const Scalar ushiftY,
    Scalar (&feq)[kQ]
  ) {
    const Scalar uiaX = ux - ushiftX;
    const Scalar uiaY = uy - ushiftY;
    const Scalar uia2tX = uiaX * uiaX + temc;
    const Scalar uia2tY = uiaY * uiaY + temc;
    LBMINI_ALPAKA_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      Scalar phi = rhoc;
      LBMINI_ALPAKA_UNROLL
      for (int d = 0; d < kD; ++d) {
        const int vi = (d == 0) ? Vx(idc) : Vy(idc);
        const Scalar uia = (d == 0) ? uiaX : uiaY;
        const Scalar uia2t = (d == 0) ? uia2tX : uia2tY;
        Scalar pf;
        if (vi == 0)
          pf = Scalar{ 1 } - uia2t;
        else if (vi == 1)
          pf = Scalar{ 0.5 } * (uia + uia2t);
        else
          pf = Scalar{ 0.5 } * (-uia + uia2t);
        phi *= pf;
      }
      feq[idc] = phi;
    }
  }

  /// rho, 1 / rho, u, T, p from one cell's populations.
  ALPAKA_FN_HOST_ACC static void Macroscopic(
    const Scalar (&f)[kQ],
    const Scalar (&g)[kQ],
    const Params<Scalar, kQ>& p,
    Scalar& rho,
    Scalar& invRho,
    Scalar& ux,
    Scalar& uy,
    Scalar& tem,
    Scalar& pres
  ) {
    Scalar rhoSum = Scalar{ 0 };
    Scalar nrg = Scalar{ 0 };
    Scalar momX = Scalar{ 0 };
    Scalar momY = Scalar{ 0 };
    LBMINI_ALPAKA_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      rhoSum += f[idc];
      nrg += g[idc];
      momX += LatticeType::Cshift(idc, 0, p.ushiftX) * f[idc];
      momY += LatticeType::Cshift(idc, 1, p.ushiftY) * f[idc];
    }
    invRho = Scalar{ 1 } / rhoSum;
    ux = momX * invRho;
    uy = momY * invRho;
    const Scalar kin = Scalar{ 0.5 } * (ux * ux + uy * uy);
    tem = (Scalar{ 0.5 } * nrg * invRho - kin) * p.invCv;
    rho = rhoSum;
    pres = p.rgas * rhoSum * tem;
  }

  /**
   * @brief `p[c + 1] = W_c * exp(xi * c - max_c' (xi * c'))` for `c = -1, 0, 1`: one axis of the closure.
   *
   * Double precision runs at 1/64 of the single-precision rate on consumer
   * GPUs and the kernel is bound by it, not by memory, so the number of FP64
   * instructions per cell is the cost. The D2Q9 weights and the exponents of
   * the energy equilibrium are tensor products, `W_q = Wx(c_x) Wy(c_y)` and
   * `s_q = xi_0 (c_x + Ux) + xi_1 (c_y + Uy)`, hence
   * `W_q exp(s_q - smax) = Px(c_x) Py(c_y)` (the `U` shift and the common maximum
   * cancel), with a single exponential per axis, `E = exp(-|xi|)`: the exponents
   * `xi c - max` are `0`, `-|xi|` and `-2 |xi|` in the order of the sign of `xi`,
   * i.e. `1`, `E` and `E^2`. That is 2 exponentials per evaluation (8 per cell and
   * step; the reference evaluates 9 per evaluation, 36 per cell and step) and it
   * turns the nine-term sums of the Newton step into three-term ones.
   *
   * It equals the reference to rounding error while neither of its clamps
   * (`|s - smax| <= 700`) is active; beyond that every omitted term is below
   * `exp(-700)` relative to the largest one.
   */
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static void AxisExp(
    const Scalar xi,
    const Scalar wn,
    const Scalar wZero,
    Scalar (&p)[3]
  ) {
    const Scalar t = -::alpaka::math::abs(xi);
    const Scalar e = ::alpaka::math::exp((t > -kMaxExp) ? t : -kMaxExp);
    const Scalar e2 = e * e;
    const bool pos = xi >= Scalar{ 0 };
    p[0] = wn * (pos ? e2 : Scalar{ 1 });
    p[1] = wZero * e;
    p[2] = wn * (pos ? Scalar{ 1 } : e2);
  }

  /// Equilibrium energy populations from the Newton solution `xi`.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static void GeqFromXi(
    const Scalar xi0,
    const Scalar xi1,
    const Scalar wn,
    const Scalar wZero,
    const Scalar targetE,
    Scalar (&geq)[kQ]
  ) {
    Scalar px[3];
    Scalar py[3];
    AxisExp(xi0, wn, wZero, px);
    AxisExp(xi1, wn, wZero, py);
    const Scalar scale = targetE / ((px[0] + px[1] + px[2]) * (py[0] + py[1] + py[2]));
    Scalar sx[3];
    LBMINI_ALPAKA_UNROLL
    for (int k = 0; k < 3; ++k)
      sx[k] = scale * px[k];
    LBMINI_ALPAKA_UNROLL
    for (int idc = 0; idc < kQ; ++idc)
      geq[idc] = sx[Vx(idc) + 1] * py[Vy(idc) + 1];
  }

  /**
   * @brief Equilibria + Newton-Raphson energy closure + BGK collision, in place.
   *
   * `f`/`g` enter as the pre-collision populations of the cell and leave as the
   * post-collision ones; `xi` is the Newton warm start (read and updated). The
   * Newton loop runs a fixed number of iterations with clamped reciprocals and
   * a sticky `solverOk` predicate, so control flow only diverges in the (rare)
   * degenerate fallback.
   */
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static void Collide(
    const Params<Scalar, kQ>& p,
    const Scalar rhoc,
    const Scalar invRho,
    const Scalar ux,
    const Scalar uy,
    const Scalar temc,
    Scalar (&f)[kQ],
    Scalar (&g)[kQ],
    Scalar (&xi)[kD]
  ) {
    const Scalar ushiftX = p.ushiftX;
    const Scalar ushiftY = p.ushiftY;
    const Scalar u2 = ux * ux + uy * uy;

    Scalar feqLocal[kQ];
    Feq(rhoc, ux, uy, temc, ushiftX, ushiftY, feqLocal);

    Scalar wi[kQ];
    Weights(temc, wi);
    // The per-axis factors of `wi` (`W_q = Wx(c_x) Wy(c_y)`), see `AxisExp`.
    const Scalar wZero = Scalar{ 1 } - temc;
    const Scalar wn = wZero + (Scalar{ 0.5 } * temc - wZero);

    // Geq via Newton-Raphson on the normalised energy flux.
    //
    // The closure distribution `W_q exp(xi . (c_q + U))` is a product of an x and a
    // y factor, so x and y are independent: the Jacobian of the reference is
    // diagonal (its off-diagonal term is exactly `S2xy / Z - (S1x / Z) (S1y / Z) = 0`)
    // and the iteration is two scalar Newton steps,
    // `xi_a -= alpha (m1_a - target_a) / var_a`, each from three weighted
    // exponentials of one axis (see `AxisExp`). Divisions are 8 FP64 slots each,
    // so the two axes share one reciprocal per stage.
    const Scalar E = p.cv * temc + Scalar{ 0.5 } * u2;
    const Scalar targetE = Scalar{ 2 } * rhoc * E;
    const Scalar invE = Scalar{ 1 } / E;
    const Scalar targetM0 = ux * (E + temc) * invE; // 2 rho u (E + T) / (2 rho E)
    const Scalar targetM1 = uy * (E + temc) * invE;

    Scalar alpha = Scalar{ 1 };
    bool solverOk = true;

    LBMINI_ALPAKA_NO_UNROLL
    for (int iter = 0; iter < kMaxIter; ++iter) {
      Scalar px[3];
      Scalar py[3];
      AxisExp(xi[0], wn, wZero, px);
      AxisExp(xi[1], wn, wZero, py);
      const Scalar zx = px[0] + px[1] + px[2];
      const Scalar zy = py[0] + py[1] + py[2];
      const Scalar zxy = zx * zy;
      solverOk = solverOk && (zxy > kTiny);
      const Scalar rz = Scalar{ 1 } / zxy;
      const Scalar invZx = zy * rz;
      const Scalar invZy = zx * rz;

      // Moments of the unshifted velocity; `c_a + U_a` shifts the mean only.
      const Scalar m1x = (px[2] - px[0]) * invZx;
      const Scalar m1y = (py[2] - py[0]) * invZy;
      const Scalar jxx = ((px[2] + px[0]) * invZx - m1x * m1x) + kTiny;
      const Scalar jyy = ((py[2] + py[0]) * invZy - m1y * m1y) + kTiny;
      const Scalar jxy = jxx * jyy;
      solverOk = solverOk && (::alpaka::math::abs(jxy) > kTiny);

      const Scalar rj = alpha * (Scalar{ 1 } / jxy);
      const Scalar r0 = (m1x + ushiftX) - targetM0;
      const Scalar r1 = (m1y + ushiftY) - targetM1;
      xi[0] -= r0 * jyy * rj; // alpha r0 / jxx
      xi[1] -= r1 * jxx * rj; // alpha r1 / jyy
      alpha *= Scalar{ 0.5 };
    }

    Scalar geqLocal[kQ];
    if (solverOk) {
      GeqFromXi(xi[0], xi[1], wn, wZero, targetE, geqLocal);
      // The reference also stores `log(scale / rho) - smax` as warm-start
      // component 0; nothing ever reads it, so it is neither computed nor stored.
    } else {
      // Fallback: normalised-Wi distribution, cold restart of the warm start.
      xi[0] = Scalar{ 0 };
      xi[1] = Scalar{ 0 };
      Scalar sumW = Scalar{ 0 };
      LBMINI_ALPAKA_UNROLL
      for (int idc = 0; idc < kQ; ++idc)
        sumW += wi[idc];
      if (sumW <= kTiny) {
        const Scalar uni = targetE / Scalar(kQ);
        LBMINI_ALPAKA_UNROLL
        for (int idc = 0; idc < kQ; ++idc)
          geqLocal[idc] = uni;
      } else {
        const Scalar sc = targetE / sumW;
        LBMINI_ALPAKA_UNROLL
        for (int idc = 0; idc < kQ; ++idc)
          geqLocal[idc] = wi[idc] * sc;
      }
    }

    // Collision (BGK + Knudsen sensor + thermal-relaxation correction).
    // `tau` and `tauT` share `1 / (rho T)`, `omega` and `omegaT` one division.
    const Scalar invT = Scalar{ 1 } / temc;
    const Scalar tau = p.viscosity * invRho * invT + Scalar{ 0.5 };
    const Scalar tauT = p.condOverCp * invRho * invT + Scalar{ 0.5 };
    const Scalar rTau = Scalar{ 1 } / (tau * tauT);
    const Scalar omega = tauT * rTau;
    Scalar omegaT = tau * rTau;

    Scalar eps = Scalar{ 0 };
    LBMINI_ALPAKA_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar diff = f[idc] - feqLocal[idc];
      const Scalar den = (feqLocal[idc] > kTiny) ? feqLocal[idc] : kTiny;
      eps += ::alpaka::math::abs(diff) / den;
    }
    eps *= Scalar{ 1 } / Scalar(kQ);
    // `omega / sigma` with `sigma` in {omega, 1.35, 1.05, 1}, without a division.
    Scalar omegaL = omega;
    if (eps >= Scalar{ 1 })
      omegaL = Scalar{ 1 };
    else if (eps >= Scalar{ 0.1 })
      omegaL = omega * (Scalar{ 1 } / Scalar{ 1.35 });
    else if (eps >= Scalar{ 0.01 })
      omegaL = omega * (Scalar{ 1 } / Scalar{ 1.05 });

    omegaL = (omegaL > Scalar{ 1 }) ? omegaL : Scalar{ 1 };
    omegaL = (omegaL < (Scalar{ 2 } - Scalar{ 1e-7 })) ? omegaL : (Scalar{ 2 } - Scalar{ 1e-7 });
    omegaT = (omegaT > Scalar{ 1 }) ? omegaT : Scalar{ 1 };
    omegaT = (omegaT < (Scalar{ 2 } - Scalar{ 1e-7 })) ? omegaT : (Scalar{ 2 } - Scalar{ 1e-7 });

    // L[a] = sum_i 2 * (u . c_i) * c_i_a * (f - feq)   (c_i unshifted)
    Scalar L[kD] = { Scalar{ 0 }, Scalar{ 0 } };
    LBMINI_ALPAKA_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar cix0 = static_cast<Scalar>(Vx(idc));
      const Scalar ciy0 = static_cast<Scalar>(Vy(idc));
      const Scalar uvi = ux * cix0 + uy * ciy0;
      const Scalar aux = Scalar{ 2 } * uvi * (f[idc] - feqLocal[idc]);
      L[0] += aux * cix0;
      L[1] += aux * ciy0;
    }

    LBMINI_ALPAKA_UNROLL
    for (int idc = 0; idc < kQ; ++idc) {
      const Scalar fOld = f[idc];
      f[idc] = fOld + omegaL * (feqLocal[idc] - fOld);
      const Scalar cidotL = L[0] * LatticeType::Cshift(idc, 0, ushiftX)
        + L[1] * LatticeType::Cshift(idc, 1, ushiftY);
      const Scalar gDiff = wi[idc] * cidotL * invT;
      const Scalar gOld = g[idc];
      g[idc] = gOld + omegaL * (geqLocal[idc] - gOld) + (omegaL - omegaT) * gDiff;
    }
  }
};

/**
 * @brief Initial state: Sod left/right macroscopics, equilibrium seeding, the
 *        macroscopic recovery of the reference `Init()`, and the first
 *        collision — one pass, nothing transferred from the host.
 *
 * Writing every buffer from the compute device also places the pages
 * (first touch) next to the threads that will use them on a CPU.
 */
template<typename Scalar, typename LatticeType>
struct InitKernel {
  using Phys = Physics<Scalar, LatticeType>;
  static constexpr int kQ = Phys::kQ;
  static constexpr int kD = Phys::kD;

  ALPAKA_FN_ACC void operator()(
    ::alpaka::onAcc::concepts::Acc auto const& acc,
    const Params<Scalar, kQ>* __restrict__ params,
    Scalar* __restrict__ fOut,
    Scalar* __restrict__ gOut,
    Scalar* __restrict__ xiOut,
    Scalar* __restrict__ rhoOut,
    Scalar* __restrict__ pOut,
    Scalar* __restrict__ temOut,
    Scalar* __restrict__ uOut
  ) const {
    const Params<Scalar, kQ>& p = *params;
    const std::ptrdiff_t N = p.cells;
    for (auto [cell] : ::alpaka::onAcc::makeIdxMap(
           acc,
           ::alpaka::onAcc::worker::threadsInGrid,
           ::alpaka::IdxRange{ ::alpaka::Vec{ p.cells } }
         )) {
      const std::int32_t i = cell / p.ny;

      // Sod-like left/right state (reference `Init()` host loop).
      const bool left = i < p.nx / 2;
      const Scalar rho0 = left ? p.densityL : p.densityR;
      const Scalar p0 = left ? p.pressureL : p.pressureR;
      const Scalar tem0 = LatticeType::Cs2 * p0 / (rho0 * p.rgas);
      const Scalar ux0 = Scalar{ 0 };
      const Scalar uy0 = Scalar{ 0 };

      // seedEquilibria(): f = feq, g = Wi(T)-normalised geq (xi == 0).
      Scalar f[kQ];
      Scalar g[kQ];
      Phys::Feq(rho0, ux0, uy0, tem0, p.ushiftX, p.ushiftY, f);
      Scalar wi[kQ];
      Phys::Weights(tem0, wi);
      const Scalar u2 = ux0 * ux0 + uy0 * uy0;
      const Scalar E = p.cv * tem0 + Scalar{ 0.5 } * u2;
      const Scalar targetE = Scalar{ 2 } * rho0 * E;
      Scalar sumW = Scalar{ 0 };
      LBMINI_ALPAKA_UNROLL
      for (int idc = 0; idc < kQ; ++idc)
        sumW += wi[idc];
      const Scalar sc = (sumW > Phys::kTiny) ? targetE / sumW : targetE / Scalar(kQ);
      LBMINI_ALPAKA_UNROLL
      for (int idc = 0; idc < kQ; ++idc)
        g[idc] = ((sumW > Phys::kTiny) ? wi[idc] : Scalar{ 1 }) * sc;

      // computeMacroscopic(): the state every getter sees after Init().
      Scalar rho;
      Scalar invRho;
      Scalar ux;
      Scalar uy;
      Scalar tem;
      Scalar pres;
      Phys::Macroscopic(f, g, p, rho, invRho, ux, uy, tem, pres);
      rhoOut[cell] = rho;
      pOut[cell] = pres;
      temOut[cell] = tem;
      uOut[0 * N + cell] = ux;
      uOut[1 * N + cell] = uy;

      // First collision, cold Newton start.
      Scalar xi[kD] = { Scalar{ 0 }, Scalar{ 0 } };
      Phys::Collide(p, rho, invRho, ux, uy, tem, f, g, xi);

      LBMINI_ALPAKA_UNROLL
      for (int idc = 0; idc < kQ; ++idc) {
        fOut[idc * N + cell] = f[idc];
        gOut[idc * N + cell] = g[idc];
      }
      LBMINI_ALPAKA_UNROLL
      for (int d = 0; d < kD; ++d)
        xiOut[d * N + cell] = xi[d];
    }
  }
};

/**
 * @brief One full time step: pull streaming + macroscopics + collision.
 *
 * @tparam kStoreMacro write `rho`, `p`, `T`, `u` (only needed on the last step
 *         of a `Run()`, the only point where a getter can observe them).
 */
template<typename Scalar, typename LatticeType, bool kStoreMacro>
struct StepKernel {
  using Phys = Physics<Scalar, LatticeType>;
  static constexpr int kQ = Phys::kQ;
  static constexpr int kD = Phys::kD;

  ALPAKA_FN_ACC void operator()(
    ::alpaka::onAcc::concepts::Acc auto const& acc,
    const Params<Scalar, kQ>* __restrict__ params,
    const Scalar* __restrict__ fIn,
    const Scalar* __restrict__ gIn,
    Scalar* __restrict__ fOut,
    Scalar* __restrict__ gOut,
    Scalar* __restrict__ xiIo,
    Scalar* __restrict__ rhoOut,
    Scalar* __restrict__ pOut,
    Scalar* __restrict__ temOut,
    Scalar* __restrict__ uOut
  ) const {
    const Params<Scalar, kQ>& p = *params;
    const std::ptrdiff_t N = p.cells;
    const std::int32_t nx = p.nx;
    const std::int32_t ny = p.ny;
    const auto& st = p.stencil;

    for (auto [cell] : ::alpaka::onAcc::makeIdxMap(
           acc,
           ::alpaka::onAcc::worker::threadsInGrid,
           ::alpaka::IdxRange{ ::alpaka::Vec{ p.cells } }
         )) {
      const std::int32_t i = cell / ny;
      const std::int32_t j = cell - i * ny;

      // Pull the post-collision populations of the previous step.
      Scalar f[kQ];
      Scalar g[kQ];
      LBMINI_ALPAKA_UNROLL
      for (int idc = 0; idc < kQ; ++idc) {
        const Scalar* __restrict__ fp = fIn + idc * N;
        const Scalar* __restrict__ gp = gIn + idc * N;
        if (st.nearest[idc]) {
          // Degenerate-weight fallback of the reference table (uniform branch:
          // `nearest` depends on the direction only).
          const auto xN = static_cast<std::int32_t>(
            ::alpaka::math::round(static_cast<Scalar>(i) - st.cix[idc]));
          const auto yN = static_cast<std::int32_t>(
            ::alpaka::math::round(static_cast<Scalar>(j) - st.ciy[idc]));
          std::int32_t yW = yN % ny;
          yW += (yW < 0) ? ny : 0;
          const std::int32_t src = (xN < 0 || xN >= nx) ? cell : xN * ny + yW;
          f[idc] = fp[src];
          g[idc] = gp[src];
        } else {
          const std::int32_t xa = i + st.ox[idc];
          const std::int32_t rowA = ((xa < 0) ? 0 : ((xa >= nx) ? nx - 1 : xa)) * ny;
          const std::int32_t xb = xa + 1;
          const std::int32_t rowB = ((xb < 0) ? 0 : ((xb >= nx) ? nx - 1 : xb)) * ny;
          std::int32_t ya = j + st.oy0[idc];
          ya -= (ya >= ny) ? ny : 0;
          std::int32_t yb = j + st.oy1[idc];
          yb -= (yb >= ny) ? ny : 0;

          const Scalar w0 = st.w[idc][0];
          const Scalar w1 = st.w[idc][1];
          const Scalar w2 = st.w[idc][2];
          const Scalar w3 = st.w[idc][3];
          f[idc] = w0 * fp[rowA + ya] + w1 * fp[rowB + ya] + w2 * fp[rowA + yb] + w3 * fp[rowB + yb];
          g[idc] = w0 * gp[rowA + ya] + w1 * gp[rowB + ya] + w2 * gp[rowA + yb] + w3 * gp[rowB + yb];
        }
      }

      Scalar rho;
      Scalar invRho;
      Scalar ux;
      Scalar uy;
      Scalar tem;
      Scalar pres;
      Phys::Macroscopic(f, g, p, rho, invRho, ux, uy, tem, pres);
      if constexpr (kStoreMacro) {
        rhoOut[cell] = rho;
        pOut[cell] = pres;
        temOut[cell] = tem;
        uOut[0 * N + cell] = ux;
        uOut[1 * N + cell] = uy;
      }

      Scalar xi[kD];
      LBMINI_ALPAKA_UNROLL
      for (int d = 0; d < kD; ++d)
        xi[d] = xiIo[d * N + cell];

      Phys::Collide(p, rho, invRho, ux, uy, tem, f, g, xi);

      LBMINI_ALPAKA_UNROLL
      for (int idc = 0; idc < kQ; ++idc) {
        fOut[idc * N + cell] = f[idc];
        gOut[idc * N + cell] = g[idc];
      }
      LBMINI_ALPAKA_UNROLL
      for (int d = 0; d < kD; ++d)
        xiIo[d * N + cell] = xi[d];
    }
  }
};
} // namespace lbmini::alpaka::detail

namespace lbmini::alpaka::detail {
/// All alpaka state of one solver on the device target: device, queue, launch
/// shape and buffers of the per-cell kernels.
template<typename Scalar, typename LatticeType, TargetEnum Target>
struct DeviceState {
  using Index = Eigen::Index;
  using Traits = TargetTraits<Target>;
  using Phys = Physics<Scalar, LatticeType>;
  static constexpr int kQ = Phys::kQ;
  static constexpr int kD = Phys::kD;
  using ParamsT = Params<Scalar, kQ>;
  using Extent = ::alpaka::Vec<Index, 1>;

  using Device = decltype(::alpaka::onHost::makeDeviceSelector(Traits::kApi, Traits::kDeviceKind).makeDevice(0));
  using Queue = decltype(std::declval<Device&>().makeQueue(Traits::kQueueKind));
  using Buffer = decltype(::alpaka::onHost::alloc<Scalar>(std::declval<Device&>(), Extent{}));
  using HostBuffer = decltype(::alpaka::onHost::allocHost<Scalar>(Extent{}));
  using ParamsBuffer = decltype(::alpaka::onHost::alloc<ParamsT>(std::declval<Device&>(), Extent{}));
  using FrameSpecT = decltype(Traits::FrameSpec(std::declval<const Device&>(), std::int32_t{}));

  DeviceState(
    const FluidData<Scalar>& fluid,
    const MeshData<Scalar, kD>& mesh,
    const ControlData<Scalar>& control
  )
    : params_(MakeParams(fluid, mesh, control)),
      nx_(mesh.size[0]),
      ny_(mesh.size[1]),
      N_(nx_ * ny_),
      device_(MakeDevice()),
      queue_(device_.makeQueue(Traits::kQueueKind)),
      frameSpec_(Traits::FrameSpec(device_, params_.cells)),
      fA_(::alpaka::onHost::alloc<Scalar>(device_, Extent{ N_ * kQ })),
      fB_(::alpaka::onHost::alloc<Scalar>(device_, Extent{ N_ * kQ })),
      gA_(::alpaka::onHost::alloc<Scalar>(device_, Extent{ N_ * kQ })),
      gB_(::alpaka::onHost::alloc<Scalar>(device_, Extent{ N_ * kQ })),
      xi_(::alpaka::onHost::alloc<Scalar>(device_, Extent{ N_ * kD })),
      rho_(::alpaka::onHost::alloc<Scalar>(device_, Extent{ N_ })),
      p_(::alpaka::onHost::alloc<Scalar>(device_, Extent{ N_ })),
      tem_(::alpaka::onHost::alloc<Scalar>(device_, Extent{ N_ })),
      u_(::alpaka::onHost::alloc<Scalar>(device_, Extent{ N_ * kD })),
      uHost_(::alpaka::onHost::allocHost<Scalar>(Extent{ N_ * kD })),
      paramsDev_(::alpaka::onHost::alloc<ParamsT>(device_, Extent{ 1 })),
      fCur_(::alpaka::onHost::data(fA_)),
      fNext_(::alpaka::onHost::data(fB_)),
      gCur_(::alpaka::onHost::data(gA_)),
      gNext_(::alpaka::onHost::data(gB_)) {
    // The parameters travel once to device memory and are passed by pointer:
    // every thread reads the same addresses (one broadcast load per warp, then
    // L1). As a by-value kernel argument the ~0.6 kB block would be copied
    // into every thread's stack frame by clang-CUDA, since the kernel binds it
    // by reference.
    auto hostView = ::alpaka::makeView(::alpaka::api::host, &params_, Extent{ 1 });
    ::alpaka::onHost::memcpy(queue_, paramsDev_, hostView);
    ::alpaka::onHost::wait(queue_);
  }

  DeviceState(const DeviceState&) = delete;

  DeviceState& operator=(const DeviceState&) = delete;

  DeviceState(DeviceState&&) = delete;

  DeviceState& operator=(DeviceState&&) = delete;

  ~DeviceState() {
    // Nothing may still be reading or writing a buffer when it is released.
    ::alpaka::onHost::wait(queue_);
  }

  void Init() {
    // The buffers are only ever written by kernels, so there is no stale state
    // to clear: a re-Init() simply restarts from the initial condition.
    fCur_ = ::alpaka::onHost::data(fA_);
    fNext_ = ::alpaka::onHost::data(fB_);
    gCur_ = ::alpaka::onHost::data(gA_);
    gNext_ = ::alpaka::onHost::data(gB_);
    queue_.enqueue(
      frameSpec_,
      ::alpaka::KernelBundle{
        InitKernel<Scalar, LatticeType>{},
        static_cast<const ParamsT*>(::alpaka::onHost::data(paramsDev_)),
        fCur_,
        gCur_,
        ::alpaka::onHost::data(xi_),
        ::alpaka::onHost::data(rho_),
        ::alpaka::onHost::data(p_),
        ::alpaka::onHost::data(tem_),
        ::alpaka::onHost::data(u_)
      }
    );
    ::alpaka::onHost::wait(queue_);
  }

  void Run(const Index steps) {
    for (Index t = 0; t < steps; ++t) {
      if (t + 1 == steps)
        enqueueStep<true>();
      else
        enqueueStep<false>();
      std::swap(fCur_, fNext_);
      std::swap(gCur_, gNext_);
    }
    // Asynchronous on the GPU: wait so the caller's wall clock covers the
    // device work, not just the enqueue (a no-op on the blocking CPU queue).
    ::alpaka::onHost::wait(queue_);
  }

  void CopyRho(Scalar* dst) const { CopyOut(rho_, dst); }

  void CopyP(Scalar* dst) const { CopyOut(p_, dst); }

  void CopyT(Scalar* dst) const { CopyOut(tem_, dst); }

  void CopyU(Scalar* dst) const {
    ::alpaka::onHost::memcpy(queue_, uHost_, u_);
    ::alpaka::onHost::wait(queue_);
    // SoA d-major -> AoS [cell][d] (RowMajor [i][j][d]).
    const Scalar* src = ::alpaka::onHost::data(uHost_);
    for (Index cell = 0; cell < N_; ++cell)
      for (Index d = 0; d < kD; ++d)
        dst[cell * kD + d] = src[d * N_ + cell];
  }

private:
  /// Copy a scalar field straight into caller-owned host memory (`N` values).
  void CopyOut(const Buffer& field, Scalar* dst) const {
    auto view = ::alpaka::makeView(::alpaka::api::host, dst, Extent{ N_ });
    ::alpaka::onHost::memcpy(queue_, view, field);
    ::alpaka::onHost::wait(queue_);
  }

  template<bool kStoreMacro>
  void enqueueStep() {
    queue_.enqueue(
      frameSpec_,
      ::alpaka::KernelBundle{
        StepKernel<Scalar, LatticeType, kStoreMacro>{},
        static_cast<const ParamsT*>(::alpaka::onHost::data(paramsDev_)),
        static_cast<const Scalar*>(fCur_),
        static_cast<const Scalar*>(gCur_),
        fNext_,
        gNext_,
        ::alpaka::onHost::data(xi_),
        ::alpaka::onHost::data(rho_),
        ::alpaka::onHost::data(p_),
        ::alpaka::onHost::data(tem_),
        ::alpaka::onHost::data(u_)
      }
    );
  }

  static Device MakeDevice() {
    auto selector = ::alpaka::onHost::makeDeviceSelector(Traits::kApi, Traits::kDeviceKind);
    if (!selector.isAvailable())
      throw std::runtime_error(std::string(Traits::kName) + ": no alpaka device available for this target.");
    return selector.makeDevice(0);
  }

  static ParamsT MakeParams(
    const FluidData<Scalar>& fluid,
    const MeshData<Scalar, kD>& mesh,
    const ControlData<Scalar>& control
  ) {
    ParamsT p{};
    p.nx = static_cast<std::int32_t>(mesh.size[0]);
    p.ny = static_cast<std::int32_t>(mesh.size[1]);
    p.cells = static_cast<std::int32_t>(mesh.size[0] * mesh.size[1]);
    p.ushiftX = control.U(0);
    p.ushiftY = control.U(1);
    p.cv = fluid.specificHeatCv;
    p.invCv = Scalar{ 1 } / fluid.specificHeatCv;
    p.rgas = fluid.constant;
    p.viscosity = fluid.viscosity;
    p.condOverCp = fluid.conductivity / fluid.specificHeatCp;
    p.densityL = fluid.densityL;
    p.pressureL = fluid.pressureL;
    p.densityR = fluid.densityR;
    p.pressureR = fluid.pressureR;
    p.stencil = BuildDirectionStencil<LatticeType>(p.ny, control.U(0), control.U(1), control.idw);
    return p;
  }

  ParamsT params_;
  Index nx_;
  Index ny_;
  Index N_;

  // Declaration order is construction order: the device and its queue outlive
  // every buffer allocated from them.
  Device device_;
  mutable Queue queue_;
  FrameSpecT frameSpec_;

  Buffer fA_;
  Buffer fB_;
  Buffer gA_;
  Buffer gB_;
  Buffer xi_;
  Buffer rho_;
  Buffer p_;
  Buffer tem_;
  Buffer u_;
  mutable HostBuffer uHost_;
  ParamsBuffer paramsDev_;

  // Ping-pong roles of the distribution buffers (swapped, never copied).
  Scalar* fCur_;
  Scalar* fNext_;
  Scalar* gCur_;
  Scalar* gNext_;
};

template<typename Scalar, typename LatticeType>
class HostState;

/// The implementation `Engine::State` is built on: the vectorised host state
/// for the CPU target, the per-cell kernels for a device.
template<typename Scalar, typename LatticeType, TargetEnum Target>
using StateImpl = std::conditional_t<
  Target == TargetEnum::CPU,
  HostState<Scalar, LatticeType>,
  DeviceState<Scalar, LatticeType, Target>>;

/// `HostState` is defined in `Cpu/LbmTubeImpl.hpp`, which only the CPU
/// translation unit includes (before it instantiates the engine); the device
/// compiler never sees it.
template<typename Scalar, typename LatticeType, TargetEnum Target>
struct Engine<Scalar, LatticeType, Target>::State : StateImpl<Scalar, LatticeType, Target> {
  using Base = StateImpl<Scalar, LatticeType, Target>;
  using Base::Base;
};

template<typename Scalar, typename LatticeType, TargetEnum Target>
Engine<Scalar, LatticeType, Target>::Engine(
  const FluidData<Scalar>& fluid,
  const MeshData<Scalar, LatticeType::Dim()>& mesh,
  const ControlData<Scalar>& control
)
  : state_(std::make_unique<State>(fluid, mesh, control)) {}

template<typename Scalar, typename LatticeType, TargetEnum Target>
Engine<Scalar, LatticeType, Target>::~Engine() = default;

template<typename Scalar, typename LatticeType, TargetEnum Target>
void Engine<Scalar, LatticeType, Target>::Init() {
  state_->Init();
}

template<typename Scalar, typename LatticeType, TargetEnum Target>
void Engine<Scalar, LatticeType, Target>::Run(const Index steps) {
  state_->Run(steps);
}

template<typename Scalar, typename LatticeType, TargetEnum Target>
void Engine<Scalar, LatticeType, Target>::CopyRho(Scalar* dst) const {
  state_->CopyRho(dst);
}

template<typename Scalar, typename LatticeType, TargetEnum Target>
void Engine<Scalar, LatticeType, Target>::CopyP(Scalar* dst) const {
  state_->CopyP(dst);
}

template<typename Scalar, typename LatticeType, TargetEnum Target>
void Engine<Scalar, LatticeType, Target>::CopyT(Scalar* dst) const {
  state_->CopyT(dst);
}

template<typename Scalar, typename LatticeType, TargetEnum Target>
void Engine<Scalar, LatticeType, Target>::CopyU(Scalar* dst) const {
  state_->CopyU(dst);
}
} // namespace lbmini::alpaka::detail

#endif // LBMINI_ALPAKA_LBMTUBEIMPL_HPP_
