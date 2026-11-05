# Lattice Boltzmann Method for HPC

This repository includes mini-apps with different framworks for heterogeneous computing of the Lattice Boltzmann Method.

The benchmark uses the Sod shock tube problem.

## Backends

The backend and the target device are selected in `config.yaml`:

```yaml
Performance:
    target: CPU     # Target device (CPU; GPU)
    backend: Plain  # Backend to use (Plain; OpenMP; OpenACC; CUDA; OpenCL; Kokkos; RAJA; MPI; Alpaka)
```

| Backend   | CPU | GPU |
|-----------|-----|-----|
| `Plain`   | yes | no  |
| `OpenMP`  | yes | yes |
| `OpenACC` | yes | yes |
| `OpenCL`  | yes | yes |
| `CUDA`    | no  | yes |
| `Kokkos`  | yes | yes |
| `RAJA`    | yes | yes |
| `MPI`     | yes | no  |
| `Alpaka`  | yes | yes |

`cores` in `config.yaml` is the number of cores of the whole run. The `MPI` backend starts one
rank per core when it is launched without `mpirun`; under an external `mpirun -np R` each rank
runs `cores / R` OpenMP threads (at least one), so the ranks never oversubscribe the cores.

## Building

```bash
./cmake-build.sh Release
```

Kokkos is fetched at configure time and is enabled by default. Its CPU target runs the
vectorised host solver (`lbmini/Lbm/Kokkos/Cpu/LbmTube.hpp`) on
`Kokkos::DefaultHostExecutionSpace` (OpenMP/Threads/Serial) and its GPU target uses the
single-source per-cell solver on `Kokkos::DefaultExecutionSpace`.
The CUDA device backend of Kokkos is opt-in, because it requires the host compiler to
also act as a CUDA compiler:

```bash
./cmake-build.sh Release -DENABLE_KOKKOS_CUDA=ON            # Kokkos with CUDA
./cmake-build.sh Release -DENABLE_KOKKOS=OFF                # skip Kokkos entirely
```

RAJA is fetched at configure time as well and is also enabled by default. Its CPU target
runs the vectorised host solver (`lbmini/Lbm/Raja/Cpu/LbmTube.hpp`, `RAJA::forall` with
`RAJA::omp_parallel_for_dynamic_exec<1>`, or `RAJA::seq_exec` when OpenMP is unavailable) and its
GPU target uses `RAJA::cuda_exec` from the single-source per-cell solver. Because RAJA only
abstracts loops, data placement of the GPU solver is handled by the backend tags in
`lbmini/Lbm/Raja/Backends.hpp`. The CUDA device backend is opt-in, since RAJA device
lambdas require the whole translation unit to be compiled by `nvcc`:

```bash
./cmake-build.sh Release -DENABLE_RAJA_CUDA=ON              # RAJA with CUDA
./cmake-build.sh Release -DENABLE_RAJA=OFF                  # skip RAJA entirely
```

Alpaka3 is used from the clone in `deps/alpaka3` (no download) and is enabled by default; it
needs GCC >= 12 or Clang >= 17 and is skipped with a warning otherwise. Its CPU target uses
`api::host` + `exec::cpuOmpBlocks` (OpenMP) and runs the vectorised host kernel
(`lbmini/Lbm/Alpaka/Cpu/LbmTubeImpl.hpp`); its GPU target uses `api::cuda` + `exec::gpuCuda`
with a single fused pull kernel per time step (streaming + macroscopics + collision, no
streaming tables, `lbmini/Lbm/Alpaka/LbmTubeImpl.hpp`). Because
alpaka compiles every source of a CUDA target as CUDA, the GPU flavour lives in its own
library and is opt-in. It works with `nvcc` or with `clang++` as CUDA compiler (the latter
is the way to go when the host GCC is newer than `nvcc` supports):

```bash
./cmake-build.sh Release -DENABLE_ALPAKA_CUDA=ON            # Alpaka with CUDA (nvcc)
./cmake-build.sh Release -DENABLE_ALPAKA_CUDA=ON -DCMAKE_CUDA_COMPILER=clang++ \
    -DCMAKE_CUDA_ARCHITECTURES=89 "-DCMAKE_CUDA_FLAGS=--cuda-path=$CUDA_HOME"  # ... clang-CUDA
./cmake-build.sh Release -DENABLE_ALPAKA=OFF                # skip Alpaka entirely
```

The project's own code (not the fetched dependencies) is compiled with `-ffast-math` in the
`Release` and `RelWithDebInfo` configurations (GCC/Clang), which lets the compiler call
glibc's libmvec (`_ZGVdN4v_exp`, ...) for `exp`/`log` in loops it vectorizes; those variants
are only declared when `__FAST_MATH__` is defined. The vector CPU kernel (next section) does
not depend on it: it evaluates `exp` with its own polynomial (`lbmini/Lbm/Simd.hpp`), written
to survive `-ffast-math` (its rounding trick sits behind an `asm` barrier that the optimizer
cannot fold away).
The scalar `Plain` reference and the GPU kernels still call libm/CUDA `exp`. Results differ
from a strict-IEEE build in the last bits (in regimes with a low relaxation time on the high-pressure side,
e.g. a pressure ratio of 2 with `nu = 1e-2`, rounding differences are amplified by the scheme itself, see
`docs/performance.md`). Pass `-DLBMINI_FAST_MATH=OFF` to disable it. Note
that fast-math folds `std::isfinite` to `true`; use `lbmini::IsFinite` (`lbmini/Utils.hpp`)
for every NaN/Inf check.

### CPU backends: SIMD kernel, memory layout, scheduling

`Plain` is the scalar, serial reference (and the baseline of every speed-up). All the other
CPU backends (`OpenMP`, `OpenACC`, `OpenCL`, `Kokkos`, `RAJA`, `MPI`, `Alpaka`) run a
vectorized kernel on the same data layout, and differ in how they spread it over the cores:

- **Layout.** Structure of arrays with the *x* direction contiguous (`cell = j * pitch + i`),
  `pitch` being `nx` rounded up to a multiple of 16 doubles and the planes of a field
  staggered so that they do not alias in the L1 cache (`lbmini/Lbm/HostKernel.hpp`,
  `host::Geometry`). The pad columns are ordinary cells, so the vector loops need no masks
  or scalar tails.
- **Kernel.** One fused pull kernel (streaming, macroscopics, Newton-Raphson energy closure,
  collision) over `kLanes` cells at a time (`lbmini/Lbm/Simd.hpp`: 16 doubles on AVX2, built
  from four independent 4-wide registers to hide latency; override with
  `-DLBMINI_SIMD_BYTES=` / `-DLBMINI_SIMD_UNROLL=`). The arithmetic is the one of the
  reference, cell by cell, except `exp` and the closure (next item).
- **Energy closure.** The Newton-Raphson equilibrium `W_q exp(xi . c_q)` is a product of an x and
  a y factor, so its Jacobian is exactly diagonal and each step is two scalar steps with one
  `exp(-|xi|)` per axis: 8 exponentials and ~850 flops per cell and step instead of 36 and
  ~2200. Every backend (CPU and GPU) uses this form; `Plain` keeps the general one as the
  reference.
- **Stores.** Meshes whose working set exceeds four times the last-level cache write the
  post-collision populations with streaming stores (`LBMINI_HOST_STREAM=0|1` overrides),
  saving the write-allocate traffic.
- **Scheduling.** The unit of work is a *segment*: a run of `LBMINI_HOST_CHUNK_CELLS` (256)
  cells of one row, so even a `ny = 1` mesh is spread over all cores. The segments are handed
  out dynamically (`schedule(dynamic,1)`; `OMP_SCHEDULE` overrides it for the OpenMP backend),
  which absorbs slow cores (E-cores, SMT siblings, time-sliced vCPUs); the initial condition
  is written with a static schedule so memory is first-touched by its eventual owner.
- **nvc++ (OpenACC).** nvc++ cannot vectorize the monolithic kernel nor accept GCC vector
  types in `acc` regions, so the OpenACC CPU backend uses an equivalent formulation made of
  innermost lane loops (`lbmini/Lbm/OpenAcc/Cpu/LaneKernel.hpp`) and needs `-Mvect=simd:256`,
  which `CMakeLists.txt` adds.

See [`docs/performance.md`](docs/performance.md) for the analysis of the campaign
results that led to this design, the roofline position of every CPU and GPU backend, the
backend limitations and the measured scaling;
[`scripts/cpu-scaling.py`](scripts/cpu-scaling.py) reproduces the scaling table and the
independent-copies throughput ceiling that parallel efficiency should be judged against.

## Testing & Verification

All tests are consolidated into unified runners for Docker, Apptainer, and native environments:

```bash
# Docker:
docker run --rm lbm-miniapp:latest test unit        # Unit tests (GoogleTest)
docker run --rm lbm-miniapp:latest test cpu         # CPU memory-safety (ASan, UBSan, Valgrind, TSan)
docker run --rm --gpus all lbm-miniapp:latest test gpu # GPU compute-sanitizer suite
docker run --rm lbm-miniapp:latest test mpi         # MPI multi-rank sweep
docker run --rm --gpus all lbm-miniapp:latest test all # Full end-to-end verification

# Direct host:
test/run-tests.sh unit
test/run-tests.sh cpu
test/run-tests.sh gpu
test/run-tests.sh mpi
test/run-tests.sh all
```

See [`docs/memory-safety.md`](docs/memory-safety.md) for full memory verification procedures.

## HPC Container Deployment (Docker & Apptainer)

For deployment on cloud HPC clusters (e.g. Slurm, AWS ParallelCluster, Azure CycleCloud), pre-configured container definitions based on **Rocky Linux 9** and the **NVIDIA HPC SDK (`nvcr.io/nvidia/nvhpc`)** are available:

- `Dockerfile`: Rocky Linux 9 + NVIDIA HPC SDK + CUDA + OpenCL + MPI container.
- `Apptainer.def` / `Singularity.def`: Native definition file for rootless execution on HPC clusters.
- `docs/container.md`: Complete guide with Slurm job scripts, multi-node MPI, and GPU execution.

### Quick Start with Docker

```bash
# Build container image
docker build -t lbm-miniapp:latest .

# Run CFD simulation with GPU acceleration
docker run --rm --gpus all -v $(pwd)/config.yaml:/opt/lbm-miniapp/config.yaml lbm-miniapp:latest
```

### Quick Start with Apptainer (Singularity)

```bash
# Build SIF image
apptainer build lbm-miniapp.sif Apptainer.def

# Run CFD simulation with GPU acceleration
apptainer run --nv --bind $(pwd):/work -W /work lbm-miniapp.sif

# Run test suite
apptainer run --app test --nv lbm-miniapp.sif

# Run benchmarking campaign
apptainer run --app campaign --nv --bind $(pwd)/results:/opt/lbm-miniapp/results lbm-miniapp.sif run
```
