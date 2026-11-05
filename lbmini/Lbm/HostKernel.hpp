#ifndef LBMINI_LBM_HOSTKERNEL_HPP_
#define LBMINI_LBM_HOSTKERNEL_HPP_

/**
 * @file HostKernel.hpp
 * @brief Vectorised fused LBM step of every CPU solver (data layout + kernel).
 *
 * One implementation of the compressible D2Q9 step, shared by `openmp::cpu`,
 * `mpi::cpu`, `kokkos::cpu`, `raja::cpu` and `alpaka::cpu`. The backends differ
 * only in what the programming model owns: how the buffers are allocated, which
 * construct distributes the row segments over the cores, and how the ranks
 * exchange halos. The arithmetic per cell is that of `lbmini::plain::LbmTube`,
 * operation for operation; only `exp` (see `Simd.hpp`) differs in the last bit.
 *
 * ### Why x-fastest
 *
 * The tube is `nx x ny` with `ny = nx / 100` in the benchmark, so `ny` is tiny
 * (1 ... a few hundred) and `nx` is long. The former layout `cell = i * ny + j`
 * made `j` the contiguous direction: a SIMD register then spans a handful of
 * cells, wraps around the periodic row at `ny`, and every shifted gather of the
 * streaming step becomes a permutation. Here the cell index is
 *
 *     cell = j * pitch + i          (i = x, contiguous; j = y, periodic)
 *
 * so every lane of a register is a neighbour in `x`, every source of the shifted
 * gather is a plain unaligned vector load at `i + ox` (the periodic wrap in `y`
 * is a per-row pointer, the clamp in `x` only touches the first and last block
 * of a row), and the vector length is independent of `ny`.
 *
 * ### Layout
 *
 * Structure of arrays, `x[plane_index * plane + j * pitch + i]`:
 *  - `pitch` is `nx` rounded up to 16 columns (two cache lines), so a row is a whole
 *    number of vector blocks for every lane count up to 16; the pad columns are
 *    ordinary cells (they stream from clamped sources, collide, and nothing ever
 *    reads them), so the kernel needs no masked loads, stores or scalar tails;
 *  - `plane` (the stride between populations) is padded so that consecutive
 *    planes sit 2560 B apart modulo 4096. With a power-of-two mesh the 18 input
 *    and 18 output streams of one block would otherwise all map to the same L1
 *    sets (and alias in the 4K store-to-load check); and `pitch` is bumped by 16
 *    when it is itself a multiple of 4096 B, which would alias the rows.
 *
 * Streaming is a pull gather from the previous *post-collision* state; the work
 * unit is a *row segment* (`StepRow`): consecutive vector blocks of one row.
 * Consecutive segments of the same column chunk are consecutive rows, so the
 * rows `j - 2 ... j + 2` a segment gathers from are still in L2 when the next
 * row reuses them.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

#include "Data.hpp"
#include "Lbm/DeviceBuffer.hpp"
#include "Lbm/Simd.hpp"
#include "Lbm/Stencil.hpp"

#define LBMINI_HOST_INLINE LBMINI_SIMD_INLINE

namespace lbmini::host {
using Index = std::int64_t;

inline constexpr int kQ = 9;                ///< D2Q9 speeds
inline constexpr int kD = 2;                ///< spatial dimensions
inline constexpr Index kPadColumns = 16;    ///< rows are padded to a multiple of the widest supported block (16 lanes)
inline constexpr Index kStaggerDoubles = 320; ///< 2560 B: plane `k` starts at `k * 2560 (mod 4096)`

/// Rounds `n` up to a multiple of `m`.
constexpr Index RoundUp(const Index n, const Index m) {
  return (n + m - 1) / m * m;
}

/**
 * @brief Padded x-fastest geometry of one plane (see the file comment).
 */
struct Geometry {
  Index nx = 0;    ///< physical columns
  Index ny = 0;    ///< physical rows
  Index pitch = 0; ///< padded columns per row, a multiple of `kPadColumns`
  Index plane = 0; ///< stride between two planes (populations, components)

  [[nodiscard]] Index Cells() const { return ny * pitch; }
};

/// Row length for `columns` cells: whole cache lines, never a multiple of 4 KiB.
constexpr Index ChoosePitch(const Index columns) {
  Index pitch = RoundUp(columns, kPadColumns);
  if (pitch % 512 == 0)
    pitch += kPadColumns;
  return pitch;
}

/// Plane stride for `cells` cells per plane: planes 2560 B apart modulo 4096.
constexpr Index ChoosePlane(const Index cells) {
  Index plane = RoundUp(cells, kPadColumns);
  plane += (kStaggerDoubles - plane % 512 + 512) % 512;
  return plane;
}

inline Geometry MakeGeometry(const Index nx, const Index ny, const Index columns) {
  Geometry g;
  g.nx = nx;
  g.ny = ny;
  g.pitch = ChoosePitch(columns);
  g.plane = ChoosePlane(ny * g.pitch);
  return g;
}

inline Geometry MakeGeometry(const Index nx, const Index ny) {
  return MakeGeometry(nx, ny, nx);
}

/**
 * @brief Fluid constants, lattice shift and gather stencil. Plain old data.
 */
template<typename Scalar>
struct Params {
  Scalar ushiftX;
  Scalar ushiftY;
  Scalar cs2;
  Scalar cv;
  Scalar invCv;
  Scalar cp;
  Scalar rgas;
  Scalar viscosity;
  Scalar conductivity;

  Scalar densityL;
  Scalar pressureL;
  Scalar densityR;
  Scalar pressureR;

  DirectionStencil<Scalar, kQ> stencil;

  Index minOx = 0;  ///< smallest source-column offset over the directions (corner 0)
  Index maxOx1 = 0; ///< largest source-column offset over the directions (corner 1, `ox + 1`)
  bool anyNearest = false;
  bool stream = false; ///< write the post-collision populations with streaming stores (`UseStreamingStores`)
};

/**
 * @brief Whether the populations of a mesh of `cells` cells are better written around the cache.
 *
 * A time step reads and writes `4 * 9` doubles per cell, the output of one step being the input of the
 * next. When that working set is much larger than the last-level cache every output line is evicted
 * before it is reused, and an ordinary store first *reads* the line (write-allocate): 464 instead of
 * 320 B per cell and step, i.e. 45 % more of the bandwidth that bounds the solver at that size. Streaming
 * stores write the line directly. For meshes that fit the cache they would only force the next step to
 * fetch the data from memory again, so they are used when the working set exceeds four times the
 * last-level cache. `LBMINI_HOST_STREAM=0|1` overrides the decision.
 */
inline bool UseStreamingStores(const Index cells) {
  if (const char* env = std::getenv("LBMINI_HOST_STREAM"))
    return env[0] != '0';
  long llc = 0;
#if defined(_SC_LEVEL3_CACHE_SIZE)
  llc = ::sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
  if (llc <= 0)
    llc = 32L << 20;
  return cells * Index{ 4 * kQ } * static_cast<Index>(sizeof(double)) > Index{ 4 } * llc;
}

template<typename Scalar, typename LatticeType>
Params<Scalar> MakeParams(
  const FluidData<Scalar>& fluid,
  const MeshData<Scalar, 2>& mesh,
  const ControlData<Scalar>& control
) {
  static_assert(LatticeType::Speeds() == kQ && LatticeType::Dim() == kD, "D2Q9 only");
  Params<Scalar> p{};
  p.ushiftX = control.U(0);
  p.ushiftY = control.U(1);
  p.cs2 = LatticeType::Cs2;
  p.cv = fluid.specificHeatCv;
  p.invCv = Scalar{ 1 } / fluid.specificHeatCv;
  p.cp = fluid.specificHeatCp;
  p.rgas = fluid.constant;
  p.viscosity = fluid.viscosity;
  p.conductivity = fluid.conductivity;
  p.densityL = fluid.densityL;
  p.pressureL = fluid.pressureL;
  p.densityR = fluid.densityR;
  p.pressureR = fluid.pressureR;
  p.stencil = BuildDirectionStencil<LatticeType>(mesh.size[1], control.U(0), control.U(1), control.idw);
  p.minOx = p.stencil.ox[0];
  p.maxOx1 = p.stencil.ox[0] + 1;
  for (int q = 0; q < kQ; ++q) {
    p.minOx = std::min<Index>(p.minOx, p.stencil.ox[q]);
    p.maxOx1 = std::max<Index>(p.maxOx1, p.stencil.ox[q] + 1);
    p.anyNearest = p.anyNearest || (p.stencil.nearest[q] != 0);
  }
  p.stream = UseStreamingStores(mesh.size[0] * mesh.size[1]);
  return p;
}

/// Pointers of one time step. `fIn`/`gIn` are the previous post-collision state.
template<typename Scalar>
struct Fields {
  const Scalar* fIn = nullptr;
  const Scalar* gIn = nullptr;
  Scalar* fOut = nullptr;
  Scalar* gOut = nullptr;
  Scalar* xi = nullptr;  ///< Newton warm start, 2 planes
  Scalar* rho = nullptr; ///< macroscopic output, 1 plane
  Scalar* p = nullptr;
  Scalar* tem = nullptr;
  Scalar* u = nullptr; ///< 2 planes
};

/**
 * @brief Splits the cells into row segments, the unit of parallel work.
 *
 * A segment is `chunkBlocks` consecutive vector blocks of one row. Segments are
 * numbered `s = chunk * ny + row`, so a contiguous range of segments (a static
 * schedule) sweeps consecutive rows of the same column chunk: the gather window
 * of five rows stays in L2. The chunk width targets ~256 cells, and shrinks until
 * there are at least `kSegmentsPerThread` segments per thread (small meshes).
 */
struct Partition {
  Index blocksPerRow = 0;
  Index chunkBlocks = 0;
  Index chunks = 0;
  Index rows = 0;

  [[nodiscard]] Index Segments() const { return chunks * rows; }

  [[nodiscard]] Index Row(const Index s) const { return s % rows; }

  [[nodiscard]] Index FirstBlock(const Index s) const { return (s / rows) * chunkBlocks; }

  [[nodiscard]] Index BlockCount(const Index s) const {
    return std::min(chunkBlocks, blocksPerRow - FirstBlock(s));
  }
};

#ifndef LBMINI_HOST_CHUNK_CELLS
#define LBMINI_HOST_CHUNK_CELLS 256
#endif
inline constexpr Index kTargetChunkCells = LBMINI_HOST_CHUNK_CELLS;
inline constexpr Index kSegmentsPerThread = 8;

inline Partition MakePartition(const Index columns, const Index rows, const Index lanes, const int threads) {
  Partition part;
  part.rows = rows;
  part.blocksPerRow = std::max<Index>(1, columns / lanes);
  Index chunks = std::max<Index>(1, (columns + kTargetChunkCells / 2) / kTargetChunkCells);
  const Index wanted = std::max<Index>(1, threads) * kSegmentsPerThread;
  if (chunks * rows < wanted)
    chunks = (wanted + rows - 1) / rows;
  chunks = std::min(chunks, part.blocksPerRow);
  part.chunkBlocks = (part.blocksPerRow + chunks - 1) / chunks;
  part.chunks = (part.blocksPerRow + part.chunkBlocks - 1) / part.chunkBlocks;
  return part;
}

/**
 * @brief The per-cell physics for `W` cells at once.
 *
 * `Collide` and `Macroscopic` are the operation-for-operation image of the
 * scalar solver: branches become lane selects, and the branch the reference
 * takes on `solverOk` is taken per lane (`Select`) unless every lane agrees.
 */
template<typename Scalar, typename LatticeType, int W>
struct Kernel {
  static_assert(std::is_same_v<Scalar, double>, "the host SIMD kernel is double precision");
  static_assert(LatticeType::Speeds() == kQ && LatticeType::Dim() == kD, "D2Q9 only");
  static_assert(kPadColumns % W == 0, "rows are padded to a whole number of blocks");

  using S = simd::Ops<Scalar, W>;
  using V = typename S::V;
  using M = typename S::M;
  using P = Params<Scalar>;
  using F = Fields<Scalar>;

  static constexpr int kLanes = W;
  static constexpr Scalar kTiny = Scalar{ 1.0e-12 };
  static constexpr Scalar kMaxExp = Scalar{ 700.0 };
  static constexpr int kMaxIter = 3;

  static constexpr int Cx(const int q) { return static_cast<int>(LatticeType::Velocity(q, 0)); }

  static constexpr int Cy(const int q) { return static_cast<int>(LatticeType::Velocity(q, 1)); }

  /// Splatted constants of one segment.
  struct Consts {
    V ushiftX, ushiftY, cs2, cv, invCv, cp, rgas, viscosity, conductivity;
    V condOverCp; ///< `conductivity / cp`, the thermal diffusivity times `rho`
    V one, half, two, tiny, maxExp;
    V csx[3]; ///< `c_x + Ux` for `c_x = -1, 0, 1`
    V csy[3];
  };

  LBMINI_HOST_INLINE static Consts MakeConsts(const P& p) {
    Consts c;
    c.ushiftX = S::Splat(p.ushiftX);
    c.ushiftY = S::Splat(p.ushiftY);
    c.cs2 = S::Splat(p.cs2);
    c.cv = S::Splat(p.cv);
    c.invCv = S::Splat(p.invCv);
    c.cp = S::Splat(p.cp);
    c.rgas = S::Splat(p.rgas);
    c.viscosity = S::Splat(p.viscosity);
    c.conductivity = S::Splat(p.conductivity);
    c.condOverCp = S::Splat(p.conductivity / p.cp);
    c.one = S::Splat(Scalar{ 1 });
    c.half = S::Splat(Scalar{ 0.5 });
    c.two = S::Splat(Scalar{ 2 });
    c.tiny = S::Splat(kTiny);
    c.maxExp = S::Splat(kMaxExp);
    for (int k = 0; k < 3; ++k) {
      c.csx[k] = S::Splat(static_cast<Scalar>(k - 1) + p.ushiftX);
      c.csy[k] = S::Splat(static_cast<Scalar>(k - 1) + p.ushiftY);
    }
    return c;
  }

  /// `Weight(q) = Wx * Wy`, `W = wZero` for a zero velocity component, `wNonZero` otherwise.
  LBMINI_HOST_INLINE static void Weights(const Consts& c, const V temc, V (&w)[kQ]) {
    const V wZero = c.one - temc;
    const V wNonZero = c.half * temc;
    const V wn = wZero + (wNonZero - wZero);
    LBMINI_UNROLL(9)
    for (int q = 0; q < kQ; ++q)
      w[q] = (Cx(q) != 0 ? wn : wZero) * (Cy(q) != 0 ? wn : wZero);
  }

  LBMINI_HOST_INLINE static void Feq(const Consts& c, const V rhoc, const V ux, const V uy, const V temc, V (&feq)[kQ]) {
    const V uax = ux - c.ushiftX;
    const V uay = uy - c.ushiftY;
    const V tx = uax * uax + temc;
    const V ty = uay * uay + temc;
    const V px[3] = { c.half * ((-uax) + tx), c.one - tx, c.half * (uax + tx) }; // c = -1, 0, +1
    const V py[3] = { c.half * ((-uay) + ty), c.one - ty, c.half * (uay + ty) };
    LBMINI_UNROLL(9)
    for (int q = 0; q < kQ; ++q)
      feq[q] = (rhoc * px[Cx(q) + 1]) * py[Cy(q) + 1];
  }

  LBMINI_HOST_INLINE static void Macroscopic(
    const Consts& c,
    const V (&f)[kQ],
    const V (&g)[kQ],
    V& rho,
    V& invRho,
    V& ux,
    V& uy,
    V& tem,
    V& pres
  ) {
    V rhoSum = c.one - c.one;
    V nrg = rhoSum;
    V momX = rhoSum;
    V momY = rhoSum;
    LBMINI_UNROLL(9)
    for (int q = 0; q < kQ; ++q) {
      rhoSum += f[q];
      nrg += g[q];
      momX += c.csx[Cx(q) + 1] * f[q];
      momY += c.csy[Cy(q) + 1] * f[q];
    }
    invRho = c.one / rhoSum;
    ux = momX * invRho;
    uy = momY * invRho;
    const V kin = c.half * (ux * ux + uy * uy);
    tem = (c.half * nrg * invRho - kin) * c.invCv;
    rho = rhoSum;
    pres = c.rgas * rhoSum * tem;
  }

  /**
   * @brief `p[c + 1] = W_c * exp(xi * c - max_c' (xi * c'))` for `c = -1, 0, 1`: one axis of the closure.
   *
   * The D2Q9 weights and the exponents of the energy equilibrium are tensor
   * products, `W_q = Wx(c_x) Wy(c_y)` and `s_q = xi_0 (c_x + Ux) + xi_1 (c_y + Uy)`,
   * so `W_q exp(s_q - smax)` of the reference is `Px(c_x) Py(c_y)` with one
   * factor per axis (the `Ux`/`Uy` shift and the common maximum cancel in the
   * product). An axis needs a single exponential, `E = exp(-|xi|)`: the three
   * exponents `xi * c - max` are `0`, `-|xi|` and `-2 |xi|` in the order of the
   * sign of `xi`, i.e. `1`, `E` and `E^2`. That is 2 exponentials per
   * evaluation instead of the 9 of the per-direction form (`Exponents` +
   * `Exp` in the scalar reference), which was the largest cost of the kernel.
   *
   * It equals the reference to rounding error as long as neither of its clamps
   * (`|s - smax| <= 700`) is active; beyond that every omitted term is below
   * `exp(-700)` relative to the largest one.
   */
  LBMINI_HOST_INLINE static void AxisExp(const Consts& c, const V xi, const V wn, const V wZero, V (&p)[3]) {
    const V e = S::Exp(S::Max(-S::Abs(xi), -c.maxExp));
    const V e2 = e * e;
    const M pos = (xi >= (c.one - c.one));
    p[0] = wn * S::Select(pos, e2, c.one);
    p[1] = wZero * e;
    p[2] = wn * S::Select(pos, c.one, e2);
  }

  /// Equilibrium energy populations from the Newton solution `xi`.
  LBMINI_HOST_INLINE static void GeqFromXi(
    const Consts& c,
    const V (&xi)[kD],
    const V wn,
    const V wZero,
    const V targetE,
    V (&geq)[kQ]
  ) {
    V px[3];
    V py[3];
    AxisExp(c, xi[0], wn, wZero, px);
    AxisExp(c, xi[1], wn, wZero, py);
    const V Z = (px[0] + px[1] + px[2]) * (py[0] + py[1] + py[2]);
    const V scale = targetE / Z;
    V sx[3];
    LBMINI_UNROLL(3)
    for (int k = 0; k < 3; ++k)
      sx[k] = scale * px[k];
    LBMINI_UNROLL(9)
    for (int q = 0; q < kQ; ++q)
      geq[q] = sx[Cx(q) + 1] * py[Cy(q) + 1];
  }

  /// Normalised-weight energy populations (the reference's fallback).
  LBMINI_HOST_INLINE static void GeqFallback(const Consts& c, const V (&w)[kQ], const V targetE, V (&geq)[kQ]) {
    V sumW = c.one - c.one;
    LBMINI_UNROLL(9)
    for (int q = 0; q < kQ; ++q)
      sumW += w[q];
    const V uni = targetE / S::Splat(static_cast<Scalar>(kQ));
    const V sc = targetE / sumW;
    const M degenerate = (sumW <= c.tiny);
    LBMINI_UNROLL(9)
    for (int q = 0; q < kQ; ++q)
      geq[q] = S::Select(degenerate, uni, w[q] * sc);
  }

  /**
   * @brief Equilibria, Newton-Raphson energy closure and BGK/Knudsen collision, in place.
   *
   * `f`/`g` enter pre-collision and leave post-collision; `xi` is the Newton warm
   * start (read and updated). The Newton loop runs a fixed number of iterations
   * with clamped reciprocals and a sticky per-lane `ok` predicate, exactly as the
   * reference; lanes that are not `ok` restart from `xi = 0` with the fallback
   * energy equilibrium.
   */
  LBMINI_HOST_INLINE static void Collide(
    const Consts& c,
    const V rhoc,
    const V invRho,
    const V ux,
    const V uy,
    const V temc,
    V (&f)[kQ],
    V (&g)[kQ],
    V (&xi)[kD]
  ) {
    const V u2 = ux * ux + uy * uy;
    V w[kQ];
    Weights(c, temc, w);
    // The per-axis factors of `w` (`W_q = Wx(c_x) Wy(c_y)`), see `AxisExp`.
    const V wZero = c.one - temc;
    const V wn = wZero + (c.half * temc - wZero);

    V feq[kQ];
    Feq(c, rhoc, ux, uy, temc, feq);

    // Geq via Newton-Raphson on the normalised energy flux.
    //
    // The closure distribution `W_q exp(xi . (c_q + U))` is a product of an x and a
    // y factor, so x and y are independent: the Jacobian of the reference is
    // diagonal (its off-diagonal term cancels exactly, `S2xy / Z - (S1x / Z) (S1y / Z) = 0`)
    // and the two moments of the iteration come from three weighted
    // exponentials per axis. Each Newton step is then two scalar steps,
    // `xi_a -= alpha (m1_a - target_a) / var_a`, with 2 exponentials per
    // iteration instead of 9, and without the 9-term sums (see `AxisExp`).
    const V E = c.cv * temc + c.half * u2;
    const V targetE = c.two * rhoc * E;
    // `2 rho u (E + T) / (2 rho E)`: one reciprocal for both components.
    const V invE = c.one / E;
    const V targetMx = ux * (E + temc) * invE;
    const V targetMy = uy * (E + temc) * invE;

    V alpha = c.one;
    M ok = S::True();
    LBMINI_UNROLL(3)
    for (int iter = 0; iter < kMaxIter; ++iter) {
      V px[3];
      V py[3];
      AxisExp(c, xi[0], wn, wZero, px);
      AxisExp(c, xi[1], wn, wZero, py);
      const V zx = px[0] + px[1] + px[2];
      const V zy = py[0] + py[1] + py[2];
      const V zxy = zx * zy;
      ok = S::And(ok, (zxy > c.tiny));
      // One division gives both `1 / zx` and `1 / zy`.
      const V rz = c.one / zxy;
      const V invZx = zy * rz;
      const V invZy = zx * rz;

      // First and second moments of the unshifted velocity; `c_a + U_a` shifts the mean only.
      const V m1x = (px[2] - px[0]) * invZx;
      const V m1y = (py[2] - py[0]) * invZy;
      const V jxx = ((px[2] + px[0]) * invZx - m1x * m1x) + c.tiny;
      const V jyy = ((py[2] + py[0]) * invZy - m1y * m1y) + c.tiny;
      const V jxy = jxx * jyy;
      ok = S::And(ok, (S::Abs(jxy) > c.tiny));

      // `r / j` for both axes from one division: `r0 / jxx = r0 jyy / (jxx jyy)`.
      const V rj = alpha * (c.one / jxy);
      const V r0 = (m1x + c.ushiftX) - targetMx;
      const V r1 = (m1y + c.ushiftY) - targetMy;
      xi[0] -= r0 * jyy * rj;
      xi[1] -= r1 * jxx * rj;
      alpha *= c.half;
    }

    V geq[kQ];
    if (S::All(ok)) {
      GeqFromXi(c, xi, wn, wZero, targetE, geq);
    } else if (!S::Any(ok)) {
      const V zero = c.one - c.one;
      xi[0] = zero;
      xi[1] = zero;
      GeqFallback(c, w, targetE, geq);
    } else {
      V geqOk[kQ];
      V geqFb[kQ];
      GeqFromXi(c, xi, wn, wZero, targetE, geqOk);
      GeqFallback(c, w, targetE, geqFb);
      const V zero = c.one - c.one;
      xi[0] = S::Select(ok, xi[0], zero);
      xi[1] = S::Select(ok, xi[1], zero);
      LBMINI_UNROLL(9)
      for (int q = 0; q < kQ; ++q)
        geq[q] = S::Select(ok, geqOk[q], geqFb[q]);
    }

    // Collision (BGK + Knudsen sensor + thermal-relaxation correction).
    //
    // `tau = mu / (rho T) + 1/2` and `tauT = kappa / (rho cp T) + 1/2` share `1 / (rho T)`;
    // `omega = 1 / tau` and `omegaT = 1 / tauT` share one division.
    const V invT = c.one / temc;
    const V tau = c.viscosity * invRho * invT + c.half;
    const V tauT = c.condOverCp * invRho * invT + c.half;
    const V rTau = c.one / (tau * tauT);
    const V omega = tauT * rTau;
    V omegaT = tau * rTau;

    V eps = c.one - c.one;
    LBMINI_UNROLL(9)
    for (int q = 0; q < kQ; ++q) {
      const V diff = f[q] - feq[q];
      const V den = S::Select(feq[q] > c.tiny, feq[q], c.tiny);
      eps += S::Abs(diff) / den;
    }
    eps *= S::Splat(Scalar{ 1 } / static_cast<Scalar>(kQ));
    // `omega / sigma` with `sigma` in {omega, 1.35, 1.05, 1}, without a division.
    V omegaL = S::Select(
      eps >= c.one,
      c.one,
      S::Select(
        eps >= S::Splat(Scalar{ 0.1 }),
        omega * S::Splat(Scalar{ 1 } / Scalar{ 1.35 }),
        S::Select(eps >= S::Splat(Scalar{ 0.01 }), omega * S::Splat(Scalar{ 1 } / Scalar{ 1.05 }), omega)
      )
    );

    const V omegaMax = S::Splat(Scalar{ 2 } - Scalar{ 1e-7 });
    omegaL = S::Select(omegaL > c.one, omegaL, c.one);
    omegaL = S::Select(omegaL < omegaMax, omegaL, omegaMax);
    omegaT = S::Select(omegaT > c.one, omegaT, c.one);
    omegaT = S::Select(omegaT < omegaMax, omegaT, omegaMax);

    // L[a] = sum_i 2 * (u . c_i) * c_i_a * (f - feq)   (c_i unshifted)
    V Lx = c.one - c.one;
    V Ly = Lx;
    LBMINI_UNROLL(9)
    for (int q = 0; q < kQ; ++q) {
      const Scalar cix0 = static_cast<Scalar>(Cx(q));
      const Scalar ciy0 = static_cast<Scalar>(Cy(q));
      const V uvi = ux * cix0 + uy * ciy0;
      const V aux = c.two * uvi * (f[q] - feq[q]);
      Lx += aux * cix0;
      Ly += aux * ciy0;
    }

    const V dOmega = omegaL - omegaT;
    LBMINI_UNROLL(9)
    for (int q = 0; q < kQ; ++q) {
      const V fOld = f[q];
      f[q] = fOld + omegaL * (feq[q] - fOld);
      const V cidotL = Lx * c.csx[Cx(q) + 1] + Ly * c.csy[Cy(q) + 1];
      const V gDiff = w[q] * cidotL * invT;
      const V gOld = g[q];
      g[q] = gOld + omegaL * (geq[q] - gOld) + dOmega * gDiff;
    }
  }

  /// State of one row: the source rows of every direction and the output pointers.
  struct Row {
    Index j;
    Index pitch;
    Index plane;
    Index ny;
    Index xLo; ///< first valid source column (local index)
    Index xHi; ///< last valid source column (local index)
    Index cellRow;
    Index src[kQ][2]; ///< `q * plane + wrappedRow * pitch` of the two y corners
  };

  LBMINI_HOST_INLINE static void BeginRow(const P& p, const Index pitch, const Index plane, const Index ny, const Index xLo, const Index xHi, const Index j, Row& r) {
    r.j = j;
    r.pitch = pitch;
    r.plane = plane;
    r.ny = ny;
    r.xLo = xLo;
    r.xHi = xHi;
    r.cellRow = j * pitch;
    LBMINI_UNROLL(9)
    for (int q = 0; q < kQ; ++q) {
      Index y0 = j + p.stencil.oy0[q];
      y0 -= (y0 >= ny) ? ny : 0;
      Index y1 = j + p.stencil.oy1[q];
      y1 -= (y1 >= ny) ? ny : 0;
      r.src[q][0] = static_cast<Index>(q) * plane + y0 * pitch;
      r.src[q][1] = static_cast<Index>(q) * plane + y1 * pitch;
    }
  }

  /// `W` source columns `[x, x + W)` of one row, each clamped into `[xLo, xHi]`.
  LBMINI_HOST_INLINE static V LoadClamped(const Scalar* row, const Index x, const Index xLo, const Index xHi) {
    alignas(64) Scalar tmp[W];
    for (int l = 0; l < W; ++l) {
      Index xs = x + l;
      xs = (xs < xLo) ? xLo : ((xs > xHi) ? xHi : xs);
      tmp[l] = row[xs];
    }
    return S::Load(tmp);
  }

  /// Nearest-node fallback of the gather (`lbmini::GatherSources`), lane by lane.
  LBMINI_HOST_INLINE static void GatherNearest(const P& p, const F& fl, const Row& r, const int q, const Index i0, V& f, V& g) {
    alignas(64) Scalar tf[W];
    alignas(64) Scalar tg[W];
    for (int l = 0; l < W; ++l) {
      const Index i = i0 + l;
      const auto xN = static_cast<Index>(std::round(static_cast<Scalar>(i) - p.stencil.cix[q]));
      const auto yN = static_cast<Index>(std::round(static_cast<Scalar>(r.j) - p.stencil.ciy[q]));
      Index yW = yN % r.ny;
      yW += (yW < 0) ? r.ny : 0;
      const Index src = (xN < r.xLo || xN > r.xHi) ? r.cellRow + i : yW * r.pitch + xN;
      tf[l] = fl.fIn[static_cast<Index>(q) * r.plane + src];
      tg[l] = fl.gIn[static_cast<Index>(q) * r.plane + src];
    }
    f = S::Load(tf);
    g = S::Load(tg);
  }

  /// Pulls the post-collision populations of the cells `[i0, i0 + W)` of the row.
  template<bool kInterior>
  LBMINI_HOST_INLINE static void Pull(const P& p, const F& fl, const Row& r, const Index i0, V (&f)[kQ], V (&g)[kQ]) {
    const auto& st = p.stencil;
    LBMINI_UNROLL(9)
    for (int q = 0; q < kQ; ++q) {
      if (st.nearest[q] != 0) {
        GatherNearest(p, fl, r, q, i0, f[q], g[q]);
        continue;
      }
      const Index x0 = i0 + st.ox[q];
      const Scalar* fa = fl.fIn + r.src[q][0];
      const Scalar* fb = fl.fIn + r.src[q][1];
      const Scalar* ga = fl.gIn + r.src[q][0];
      const Scalar* gb = fl.gIn + r.src[q][1];
      const V w0 = S::Splat(st.w[q][0]);
      const V w1 = S::Splat(st.w[q][1]);
      const V w2 = S::Splat(st.w[q][2]);
      const V w3 = S::Splat(st.w[q][3]);
      V fa0, fa1, fb0, fb1, ga0, ga1, gb0, gb1; // corner (x0 | x1, y0 | y1)
      if constexpr (kInterior) {
        fa0 = S::Load(fa + x0);
        fa1 = S::Load(fa + x0 + 1);
        fb0 = S::Load(fb + x0);
        fb1 = S::Load(fb + x0 + 1);
        ga0 = S::Load(ga + x0);
        ga1 = S::Load(ga + x0 + 1);
        gb0 = S::Load(gb + x0);
        gb1 = S::Load(gb + x0 + 1);
      } else {
        fa0 = LoadClamped(fa, x0, r.xLo, r.xHi);
        fa1 = LoadClamped(fa, x0 + 1, r.xLo, r.xHi);
        fb0 = LoadClamped(fb, x0, r.xLo, r.xHi);
        fb1 = LoadClamped(fb, x0 + 1, r.xLo, r.xHi);
        ga0 = LoadClamped(ga, x0, r.xLo, r.xHi);
        ga1 = LoadClamped(ga, x0 + 1, r.xLo, r.xHi);
        gb0 = LoadClamped(gb, x0, r.xLo, r.xHi);
        gb1 = LoadClamped(gb, x0 + 1, r.xLo, r.xHi);
      }
      // corners in reference order: (x0, y0), (x1, y0), (x0, y1), (x1, y1)
      f[q] = w0 * fa0 + w1 * fa1 + w2 * fb0 + w3 * fb1;
      g[q] = w0 * ga0 + w1 * ga1 + w2 * gb0 + w3 * gb1;
    }
  }

  /// One time step of the `W` cells `[i0, i0 + W)` of the row.
  template<bool kStore, bool kInterior, bool kStream>
  LBMINI_HOST_INLINE static void StepBlock(const P& p, const Consts& c, const F& fl, const Row& r, const Index i0) {
    V f[kQ];
    V g[kQ];
    Pull<kInterior>(p, fl, r, i0, f, g);

    V rho, invRho, ux, uy, tem, pres;
    Macroscopic(c, f, g, rho, invRho, ux, uy, tem, pres);
    const Index cell = r.cellRow + i0;
    if constexpr (kStore) {
      S::Store(fl.rho + cell, rho);
      S::Store(fl.p + cell, pres);
      S::Store(fl.tem + cell, tem);
      S::Store(fl.u + cell, ux);
      S::Store(fl.u + r.plane + cell, uy);
    }

    V xi[kD] = { S::Load(fl.xi + cell), S::Load(fl.xi + r.plane + cell) };
    Collide(c, rho, invRho, ux, uy, tem, f, g, xi);
    LBMINI_UNROLL(9)
    for (int q = 0; q < kQ; ++q) {
      if constexpr (kStream) {
        S::StoreStream(fl.fOut + static_cast<Index>(q) * r.plane + cell, f[q]);
        S::StoreStream(fl.gOut + static_cast<Index>(q) * r.plane + cell, g[q]);
      } else {
        S::Store(fl.fOut + static_cast<Index>(q) * r.plane + cell, f[q]);
        S::Store(fl.gOut + static_cast<Index>(q) * r.plane + cell, g[q]);
      }
    }
    S::Store(fl.xi + cell, xi[0]);
    S::Store(fl.xi + r.plane + cell, xi[1]);
  }

  /// The blocks of one row segment, interior ones on the unclamped gather.
  template<bool kStore, bool kStream>
  LBMINI_HOST_INLINE static void RowBlocks(
    const P& p,
    const Consts& c,
    const F& fl,
    const Row& r,
    const Index i0,
    const Index count
  ) {
    // A block is interior if every direction's gather window stays inside the tube.
    const Index interiorLo = r.xLo - p.minOx;
    const Index interiorHi = r.xHi - (W - 1) - p.maxOx1;
    for (Index b = 0; b < count; ++b) {
      const Index i = i0 + b * W;
      if (!p.anyNearest && i >= interiorLo && i <= interiorHi)
        StepBlock<kStore, true, kStream>(p, c, fl, r, i);
      else
        StepBlock<kStore, false, kStream>(p, c, fl, r, i);
    }
  }

  /// Alignment, in bytes, that a streaming store needs: one native register.
  static constexpr std::size_t kStreamAlign = (W >= simd::kNativeLanes ? simd::kNativeLanes : W) * sizeof(Scalar);

  /**
   * @brief One time step of `count` consecutive blocks of row `j`, the first at column `i0`.
   *
   * Source columns are clamped into `[xLo, xHi]` (the tube, in local indices):
   * only the blocks whose gather window leaves that range take the clamped path.
   * With `Params::stream` the post-collision populations are written with streaming
   * stores (when the buffers are aligned for them), see `UseStreamingStores`.
   */
  template<bool kStore>
  static void StepRow(
    const P& p,
    const Geometry& geo,
    const F& fl,
    const Index xLo,
    const Index xHi,
    const Index j,
    const Index i0,
    const Index count
  ) {
    const Consts c = MakeConsts(p);
    Row r;
    BeginRow(p, geo.pitch, geo.plane, geo.ny, xLo, xHi, j, r);
    const bool aligned =
      ((reinterpret_cast<std::uintptr_t>(fl.fOut) | reinterpret_cast<std::uintptr_t>(fl.gOut)) % kStreamAlign) == 0
      && ((static_cast<std::size_t>(geo.plane | geo.pitch | i0) * sizeof(Scalar)) % kStreamAlign) == 0;
    if (p.stream && aligned) {
      RowBlocks<kStore, true>(p, c, fl, r, i0, count);
      S::StreamFence();
    } else {
      RowBlocks<kStore, false>(p, c, fl, r, i0, count);
    }
  }

  /**
   * @brief Sod initial state of `count` blocks of row `j`: seeding, macroscopics and first collision.
   *
   * `gx0` is the global column of local column `i0`; columns past the tube (pad
   * lanes) take the right state.
   */
  static void InitRow(
    const P& p,
    const Geometry& geo,
    const Fields<Scalar>& fl,
    const Index nxGlobal,
    const Index gx0,
    const Index j,
    const Index i0,
    const Index count
  ) {
    const Consts c = MakeConsts(p);
    for (Index b = 0; b < count; ++b) {
      const Index i = i0 + b * W;
      alignas(64) Scalar rl[W];
      alignas(64) Scalar pl[W];
      for (int l = 0; l < W; ++l) {
        const Index gi = std::min(gx0 + b * W + l, nxGlobal - 1);
        const bool left = gi < nxGlobal / 2;
        rl[l] = left ? p.densityL : p.densityR;
        pl[l] = left ? p.pressureL : p.pressureR;
      }
      const V rho0 = S::Load(rl);
      const V p0 = S::Load(pl);
      const V tem0 = c.cs2 * p0 / (rho0 * c.rgas);
      const V zero = c.one - c.one;

      V f[kQ];
      V g[kQ];
      Feq(c, rho0, zero, zero, tem0, f);
      V w[kQ];
      Weights(c, tem0, w);
      const V u2 = zero;
      const V E = c.cv * tem0 + c.half * u2;
      const V targetE = c.two * rho0 * E;
      V sumW = zero;
      LBMINI_UNROLL(9)
      for (int q = 0; q < kQ; ++q)
        sumW += w[q];
      const M useW = (sumW > c.tiny);
      const V sc = S::Select(useW, targetE / sumW, targetE / S::Splat(static_cast<Scalar>(kQ)));
      LBMINI_UNROLL(9)
      for (int q = 0; q < kQ; ++q)
        g[q] = S::Select(useW, w[q], c.one) * sc;

      V rho, invRho, ux, uy, tem, pres;
      Macroscopic(c, f, g, rho, invRho, ux, uy, tem, pres);
      const Index cell = j * geo.pitch + i;
      S::Store(fl.rho + cell, rho);
      S::Store(fl.p + cell, pres);
      S::Store(fl.tem + cell, tem);
      S::Store(fl.u + cell, ux);
      S::Store(fl.u + geo.plane + cell, uy);

      V xi[kD] = { zero, zero };
      Collide(c, rho, invRho, ux, uy, tem, f, g, xi);
      LBMINI_UNROLL(9)
      for (int q = 0; q < kQ; ++q) {
        S::Store(fl.fOut + static_cast<Index>(q) * geo.plane + cell, f[q]);
        S::Store(fl.gOut + static_cast<Index>(q) * geo.plane + cell, g[q]);
      }
      S::Store(fl.xi + cell, xi[0]);
      S::Store(fl.xi + geo.plane + cell, xi[1]);
    }
  }
};

/// Copies one plane of the padded x-fastest layout into a row-major `(nx, ny)` field.
template<typename Scalar>
void ExportField(const Scalar* plane, const Geometry& geo, const Index xFirst, Scalar* out) {
  for (Index j = 0; j < geo.ny; ++j) {
    const Scalar* row = plane + j * geo.pitch + xFirst;
    for (Index i = 0; i < geo.nx; ++i)
      out[i * geo.ny + j] = row[i];
  }
}

/// Copies two planes into a row-major `(nx, ny, 2)` field.
template<typename Scalar>
void ExportVector(const Scalar* plane0, const Scalar* plane1, const Geometry& geo, const Index xFirst, Scalar* out) {
  for (Index j = 0; j < geo.ny; ++j) {
    const Scalar* r0 = plane0 + j * geo.pitch + xFirst;
    const Scalar* r1 = plane1 + j * geo.pitch + xFirst;
    for (Index i = 0; i < geo.nx; ++i) {
      out[(i * geo.ny + j) * 2 + 0] = r0[i];
      out[(i * geo.ny + j) * 2 + 1] = r1[i];
    }
  }
}
} // namespace lbmini::host

#endif // LBMINI_LBM_HOSTKERNEL_HPP_
