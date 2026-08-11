# Lattice Boltzmann Method for HPC

This repository includes mini-apps with different framworks for heterogeneous computing of the Lattice Boltzmann Method.

The benchmark uses the Sod shock tube problem.

## Backends

The backend and the target device are selected in `config.yaml`:

```yaml
Performance:
    target: CPU     # Target device (CPU; GPU)
    backend: Plain  # Backend to use (Plain; OpenMP; OpenACC; CUDA; OpenCL; Kokkos; RAJA; MPI)
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
