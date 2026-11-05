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

## Building

```bash
./cmake-build.sh Release
```

Kokkos is fetched at configure time and is enabled by default. Its CPU target uses
`Kokkos::DefaultHostExecutionSpace` (OpenMP/Serial) and its GPU target uses
`Kokkos::DefaultExecutionSpace`, so both come from the same single-source solver.
The CUDA device backend of Kokkos is opt-in, because it requires the host compiler to
also act as a CUDA compiler:

```bash
./cmake-build.sh Release -DENABLE_KOKKOS_CUDA=ON            # Kokkos with CUDA
./cmake-build.sh Release -DENABLE_KOKKOS=OFF                # skip Kokkos entirely
```

RAJA is fetched at configure time as well and is also enabled by default. Its CPU target
uses `RAJA::omp_parallel_for_exec` (or `RAJA::seq_exec` when OpenMP is unavailable) and its
GPU target uses `RAJA::cuda_exec`, again from a single-source solver. Because RAJA only
abstracts loops, data placement is handled by the backend tags in
`lbmini/Lbm/Raja/Backends.hpp`. The CUDA device backend is opt-in, since RAJA device
lambdas require the whole translation unit to be compiled by `nvcc`:

```bash
./cmake-build.sh Release -DENABLE_RAJA_CUDA=ON              # RAJA with CUDA
./cmake-build.sh Release -DENABLE_RAJA=OFF                  # skip RAJA entirely
```

Alpaka3 is used from the clone in `deps/alpaka3` (no download) and is enabled by default; it
needs GCC >= 12 or Clang >= 17 and is skipped with a warning otherwise. Its CPU target uses
`api::host` + `exec::cpuOmpBlocks` (OpenMP), its GPU target `api::cuda` + `exec::gpuCuda`,
from one implementation (`lbmini/Lbm/Alpaka/LbmTubeImpl.hpp`) that runs each time step as a
single fused pull kernel (streaming + macroscopics + collision, no streaming tables). Because
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
are only declared when `__FAST_MATH__` is defined. The fused per-cell kernels of the CPU
backends are not vectorized by GCC, so there `exp`/`log` stay scalar (check with
`-fopt-info-vec`). Results differ from a strict-IEEE build in the last bits. Pass
`-DLBMINI_FAST_MATH=OFF` to disable it. Note that fast-math folds `std::isfinite` to `true`;
use `lbmini::IsFinite` (`lbmini/Utils.hpp`) for every NaN/Inf check.

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
