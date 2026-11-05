# Container Deployment Guide for Cloud HPC Clusters

This document describes the containerization setup for the **LBM Mini-App CFD Solver** using **Docker** and **Apptainer (Singularity)** on **Rocky Linux 9** and the official **NVIDIA HPC SDK (`nvcr.io/nvidia/nvhpc`)** base image.

---

## 1. Overview & Acceleration Backends

The container encapsulates all compilers, runtimes, and dependencies needed to benchmark HPC technologies across CPU and GPU architectures:

| Backend | CPU Support | GPU Support | Technology Stack |
| :--- | :--- | :--- | :--- |
| `Plain` | Yes | No | ISO C++20 Serial |
| `OpenMP` | Yes | Yes | OpenMP 4.5+ / GPU Target Offloading (`-mp`, `-mp=gpu`) |
| `OpenACC` | Yes | Yes | OpenACC 2.7+ Multicore & GPU (`-acc=multicore`, `-acc=gpu`) |
| `OpenCL` | Yes | Yes | OpenCL C++ / OpenCL Headers + NVIDIA GPU ICD + Intel CPU ICD |
| `CUDA` | No | Yes | NVIDIA CUDA C++ (`nvcc` / `nvc++`) |
| `Kokkos` | Yes | Yes | Kokkos Framework (OpenMP CPU & CUDA Device Backends) |
| `RAJA` | Yes | Yes | LLNL RAJA Framework (OpenMP CPU & CUDA Device Backends) |
| `MPI` | Yes | No | OpenMPI / HPC-X Multi-Node Domain Decomposition |
| `Alpaka` | Yes | Yes | Alpaka3 (`deps/alpaka3`): `exec::cpuOmpBlocks` CPU & `exec::gpuCuda` device (`-DENABLE_ALPAKA_CUDA=ON`) |

> **Alpaka3 needs GCC >= 12.** The Rocky Linux 9 system compiler is GCC 11.5, so with the
> default image the Alpaka backend is skipped at configure time (CMake prints a warning and
> the solver reports `Alpaka backend not enabled.`, which the test runners treat as a skip).
> To include it, build with a newer toolset, e.g. install `gcc-toolset-13` and configure with
> `-DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/g++
> -DCMAKE_CUDA_HOST_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/g++ -DENABLE_ALPAKA_CUDA=ON`.

---

## 2. Container Files Reference

- **`Dockerfile`**: Builds a Rocky Linux 9 + NVIDIA HPC SDK image containing the full compiler toolchain (`nvc++`, `nvcc`, `nvfortran`), CUDA Toolkit, HPC-X OpenMPI, OpenCL (OpenCL headers, NVIDIA GPU ICD, and Intel OpenCL CPU ICD), Python scientific suite, and pre-compiled LBM mini-app binaries.
- **`Apptainer.def`** / **`Singularity.def`**: Native Apptainer/Singularity definition file designed for rootless execution on Cloud HPC clusters (Slurm, PBS, AWS ParallelCluster, Azure CycleCloud, GCP HPC Toolkit).
- **`entrypoint.sh`**: Intelligent entrypoint script that dynamically discovers NVHPC paths, sets OpenMP thread affinities, and routes CLI commands (`main`, `test`, `memory-check`, `campaign`, `bash`).
- **`.dockerignore`**: Excludes build caches, results data, and VCS directories to keep images lightweight.

---

## 3. Building the Container

### Option A: Build with Docker / Podman
```bash
# Build using official NVIDIA HPC SDK on Rocky Linux 9
docker build -t lbm-miniapp:latest .

# (Optional) Build behind corporate proxy / SSL inspection (bypasses SSL cert error)
docker build --build-arg INSECURE_SSL=true -t lbm-miniapp:latest .

# (Optional) Build with custom base image version
docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/nvhpc:26.1-devel-cuda13.1-rockylinux9 -t lbm-miniapp:latest .

# (Optional) Build the device backends for the compute capability of the target GPU
docker build --build-arg CUDA_ARCHITECTURES=80 -t lbm-miniapp:latest .
```

#### GPU architecture (`CUDA_ARCHITECTURES`)

`CUDA_ARCHITECTURES` (default `89`) is the single place the compute capability is
chosen: `CMakeLists.txt` derives `CMAKE_CUDA_ARCHITECTURES`, Kokkos'
`Kokkos_ARCH_<NAME>` and RAJA/BLT's `CUDA_ARCH=sm_<nn>` from it, so the CUDA,
Kokkos, and RAJA device backends can never disagree. Kokkos and BLT accept a
single architecture, so this is one capability, not a list:

| Value | GPUs |
| :--- | :--- |
| `80` | A100 (Ampere) |
| `86` | RTX 30xx, A40 |
| `89` | RTX 40xx, L40, L4 (Ada) — default |
| `90` | H100, H200 (Hopper) |
| `100` / `120` | Blackwell |

A mismatch is not a build error: the driver JIT-compiles the embedded PTX and
the run only warns — `Kokkos::Cuda::initialize WARNING: running kernels compiled
for compute capability 8.0 on device with compute capability 8.9` — while losing
performance, which is precisely what a benchmark must not do silently.

The same override works for Apptainer builds:

```bash
CUDA_ARCHITECTURES=90 apptainer build lbm-miniapp.sif Apptainer.def
```

### Option B: Build Apptainer SIF directly from definition file
```bash
# Set temporary and cache directories to a partition with ample free space (recommended on HPC clusters)
export APPTAINER_TMPDIR=/var/tmp
export APPTAINER_CACHEDIR=/var/tmp

# On a machine with root/fakeroot access:
apptainer build lbm-miniapp.sif Apptainer.def
```

### Option C: Convert Docker image to Apptainer SIF (Recommended on clusters)
```bash
# Method 1: Convert directly from local Docker daemon:
apptainer build lbm-miniapp.sif docker-daemon://lbm-miniapp:latest

# Method 2: Convert from exported Docker tar archive (bypasses daemon streaming/socket issues):
docker save lbm-miniapp:latest -o lbm-miniapp.tar
apptainer build lbm-miniapp.sif docker-archive:lbm-miniapp.tar
rm -f lbm-miniapp.tar

# Method 3: Pull directly from container registry:
apptainer build lbm-miniapp.sif docker://your-registry/lbm-miniapp:latest
```

---

## 4. Running on Cloud HPC & Workstations

### 4.1 Running with Apptainer (Singularity)

Apptainer is the standard runtime on HPC clusters as it does not require root privileges.

#### Running Main CFD Solver Simulation:
```bash
# Run on CPU with local config.yaml in the working directory
apptainer run --bind $(pwd):/work -W /work lbm-miniapp.sif

# Run on CPU with explicit thread count
OMP_NUM_THREADS=16 apptainer run --bind $(pwd):/work -W /work lbm-miniapp.sif

# Run on NVIDIA GPU (CUDA / OpenACC / OpenMP GPU / Kokkos / RAJA / OpenCL)
apptainer run --nv --bind $(pwd):/work -W /work lbm-miniapp.sif

# Run with explicit configuration and output paths
apptainer run --nv \
  --bind $(pwd)/lbmini/config.yaml:/opt/lbm-miniapp/lbmini/config.yaml \
  --bind $(pwd)/out:/opt/lbm-miniapp/out \
  lbm-miniapp.sif main --config /opt/lbm-miniapp/lbmini/config.yaml --outpath /opt/lbm-miniapp/out
```

#### Running Test Suites & Verification:
```bash
# 1. Run Unit tests (GoogleTest)
apptainer run lbm-miniapp.sif test unit
# (Or using app syntax: apptainer run --app test lbm-miniapp.sif unit)

# 2. Run CPU memory safety & race check (ASan + UBSan + Valgrind + TSan)
apptainer run lbm-miniapp.sif test cpu

# 3. Run GPU diagnostic suite (compute-sanitizer memcheck, racecheck, initcheck, synccheck)
apptainer run --nv lbm-miniapp.sif test gpu

# 4. Run MPI multi-rank domain decomposition tests
apptainer run lbm-miniapp.sif test mpi

# 5. Run full end-to-end verification pipeline
apptainer run --nv lbm-miniapp.sif test all
```

#### Running Benchmark Campaign & Simulation Analysis:
```bash
# 1. Run full benchmark campaign
apptainer run --app campaign --nv --bind $(pwd)/results:/opt/lbm-miniapp/results lbm-miniapp.sif run

# 2. Analyse a single simulation output directory
apptainer run --bind $(pwd)/out:/opt/lbm-miniapp/out lbm-miniapp.sif campaign single --outpath /opt/lbm-miniapp/out

# 3. Generate figures and report from campaign ledger
apptainer run --app campaign --bind $(pwd)/results:/opt/lbm-miniapp/results lbm-miniapp.sif report
```

#### Interactive Shell:
```bash
apptainer shell --nv --bind $(pwd):/work -W /work lbm-miniapp.sif
```

---

### 4.2 Running with Docker

#### Running Main CFD Solver Simulation:
```bash
# Run on CPU:
docker run --rm \
    -v $(pwd)/lbmini/config.yaml:/opt/lbm-miniapp/lbmini/config.yaml \
    -v $(pwd)/out:/opt/lbm-miniapp/out \
    lbm-miniapp:latest main --config /opt/lbm-miniapp/lbmini/config.yaml --outpath /opt/lbm-miniapp/out

# Run on NVIDIA GPU (requires NVIDIA Container Toolkit):
docker run --rm --gpus all \
    -v $(pwd)/lbmini/config.yaml:/opt/lbm-miniapp/lbmini/config.yaml \
    -v $(pwd)/out:/opt/lbm-miniapp/out \
    lbm-miniapp:latest main --config /opt/lbm-miniapp/lbmini/config.yaml --outpath /opt/lbm-miniapp/out
```

#### Running Test Suites & Verification:
```bash
# Unit tests
docker run --rm lbm-miniapp:latest test unit

# CPU memory-safety & race checks (ASan + UBSan + Valgrind + TSan)
docker run --rm lbm-miniapp:latest test cpu

# GPU diagnostic suite (compute-sanitizer); ~40 s on an RTX 4050 Laptop GPU.
# Every step is bounded by LBMINI_SANITIZER_TIMEOUT (default 900 s), and a
# container that has the driver but gets no device fails - set
# LBMINI_ALLOW_NO_GPU=1 to make that a skip on purpose (see section 7.6).
docker run --rm --gpus all lbm-miniapp:latest test gpu

# MPI multi-rank domain decomposition tests
docker run --rm lbm-miniapp:latest test mpi

# Full verification pipeline (~40 min on a 16-core host; every memory-safety
# stage runs in parallel - override the degree with LBMINI_TEST_JOBS)
docker run --rm --gpus all lbm-miniapp:latest test all
docker run --rm --gpus all -e LBMINI_TEST_JOBS=8 lbm-miniapp:latest test all
```

#### Running Benchmark Campaign & Simulation Analysis:
```bash
# Run full benchmark campaign
docker run --rm --gpus all -v $(pwd)/results:/opt/lbm-miniapp/results lbm-miniapp:latest campaign run

# Analyse a single simulation output directory
docker run --rm -v $(pwd)/out:/opt/lbm-miniapp/out lbm-miniapp:latest campaign single --outpath /opt/lbm-miniapp/out

# Generate figures and report from campaign ledger
docker run --rm -v $(pwd)/results:/opt/lbm-miniapp/results lbm-miniapp:latest campaign report
```

#### Interactive Shell:
```bash
docker run --rm -it --gpus all lbm-miniapp:latest bash
```

---

## 5. Slurm Cluster Job Script Examples

### 5.1 Single-Node Multi-GPU Benchmark (CUDA / OpenACC / Kokkos)
```bash
#!/bin/bash
#SBATCH --job-name=lbm_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

module load apptainer nvidia

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

apptainer run --nv \
    --bind ${SLURM_SUBMIT_DIR}:/work \
    -W /work \
    /path/to/lbm-miniapp.sif
```

### 5.2 Multi-Node MPI Simulation
```bash
#!/bin/bash
#SBATCH --job-name=lbm_mpi
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=compute

module load apptainer openmpi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Launch MPI ranks with host launcher calling containerized binary
mpirun -np ${SLURM_NTASKS} \
    apptainer exec \
    --bind ${SLURM_SUBMIT_DIR}:/work \
    -W /work \
    /path/to/lbm-miniapp.sif \
    lbm-miniapp-main
```

---

## 6. CPU / GPU Backend Isolation

The mini-app exists to compare processors, so a backend is never allowed to run
on the hardware it is not measuring. This is enforced, not assumed:

| Guarantee | Where it is enforced |
| :--- | :--- |
| An OpenCL CPU run uses an OpenCL **CPU** device, or fails | `lbmini/Lbm/OpenCl/Cpu/LbmTube.hpp` requests `CL_DEVICE_TYPE_CPU` only — the former `CL_DEVICE_TYPE_ALL` fallback could hand it the GPU |
| An OpenCL GPU run uses an OpenCL **GPU** device, or fails | `lbmini/Lbm/OpenCl/Gpu/LbmTube.hpp`, symmetrically |
| A Kokkos / RAJA GPU run really executes on a device | `lbmini/Executor.cpp` throws when the framework was built without a device backend, instead of degrading to the host execution space |
| `Plain` and `MPI` never accept a GPU target; `CUDA` never accepts a CPU target | `lbmini/Executor.cpp` |
| The CPU test stage cannot touch a GPU | `test/memory-check.sh` builds every tree with `-DENABLE_CUDA=OFF` and exports an empty `CUDA_VISIBLE_DEVICES`; Valgrind runs the CUDA-free `build-memcheck` binary |
| The MPI stage cannot touch a GPU | `test/run-tests.sh` exports an empty `CUDA_VISIBLE_DEVICES` before `mpirun` (the MPI backend is CPU-only) |
| A failing GPU backend is reported, not skipped | `test/run-tests.sh` no longer discards the solver's stderr; only an explicit "not available / not enabled" message counts as a skip |
| A backend that skips is a failure, not a pass | `test/memory-check.sh` fails on any host-backend skip; `test/run-tests.sh` fails on any device-backend skip once a GPU has been detected |
| A run that advances zero time steps cannot pass | `lbmini/Executor.cpp` throws when `tmax` is shorter than the derived `dt`; `test/run-tests.sh` additionally requires each GPU and MPI run to report time steps |
| A `compute-sanitizer` report with findings fails the stage | `test/run-tests.sh` passes `--error-exitcode 1` and audits the `ERROR`/`LEAK`/`RACECHECK` summaries |
| A GPU that is installed but unusable fails, it is not skipped | `test/run-tests.sh` skips only when the container has no driver stack at all; a present driver that yields no device is an error (section 7.6) |
| A memory tool that stops making progress fails the stage | `test/run-tests.sh` bounds every `compute-sanitizer` step and device sweep with `LBMINI_SANITIZER_TIMEOUT` (default 900 s) |
| The OpenCL GPU backend is really leak-checked | `test/run-tests.sh` runs its sweep from `build-asan` with `protect_shadow_gap=0`, without which the NVIDIA platform is invisible to a sanitized process (section 7.7) |

Running a container without `--gpus all` (Docker) or `--nv` (Apptainer) is
therefore a supported way to verify the host backends: the GPU targets fail
loudly instead of producing host numbers labelled as device numbers.

---

## 7. Troubleshooting

### 7.1 `undefined symbol: pmix_framework_names` when launching MPI ranks

```
lbm-miniapp-main: symbol lookup error: .../hpcx-2.50/ompi/lib/libopen-pal.so.80:
    undefined symbol: pmix_framework_names
```

HPC-X OpenMPI 5 resolved PMIx somewhere other than its own installation — usually
the Rocky system `libpmix`, or a second copy of the SDK reached through a
different directory name (the NVIDIA HPC SDK exposes the same release as both
`Linux_x86_64/2026` and `Linux_x86_64/26.1`, and its MPI through the
`comm_libs/mpi` symlink). A binary linked through one path and launched with the
other dies in the dynamic loader, before `main()`.

`scripts/nvhpc-env.sh` is the fix and the reference implementation: it resolves
one SDK tree (`pwd -P`, newest first), exports `OPAL_PREFIX` /
`PMIX_INSTALL_PREFIX` — which a relocated Open MPI needs to find its own MCA
components — and prepends the HPC-X `ompi`, `pmix`, `ucx`, `ucc` and `hcoll`
library directories to `LD_LIBRARY_PATH`. The image build, `entrypoint.sh` and
the Apptainer/Singularity `%environment` all source that one script, and the
build additionally registers those directories in
`/etc/ld.so.conf.d/000-nvhpc-mpi.conf` for processes started without it
(`apptainer exec ... lbm-miniapp-main`).

Outside the container, source it before launching:

```bash
. /opt/lbm-miniapp/scripts/nvhpc-env.sh
mpirun -np 4 lbm-miniapp-main --config lbmini/config.yaml
```

### 7.2 `Backend OpenACC/OpenMP GPU target not available ... skipping`

The image compiles with GCC by default, and GCC offers no NVIDIA offloading
here: the OpenACC and OpenMP **GPU** targets exist only when the project is
built with the NVIDIA HPC SDK compilers (`-DCMAKE_CXX_COMPILER=nvc++`). The CUDA,
Kokkos, RAJA and OpenCL GPU backends are unaffected. This is a skip, not a
failure.

### 7.3 `LeakSanitizer: detected memory leaks` in every OpenCL test

```
Direct leak of 25600 byte(s) in 1 object(s) allocated from:
    #0 ... in calloc
    #1 ...  (<unknown module>)
    #2 ... in __pthread_once_slow
```

The allocation belongs to Intel's oneTBB arena, created once inside the OpenCL
ICD loader's `pthread_once` the moment a platform is enumerated — it is not the
solver's. `test/lsan_suppressions.supp` covers vendor runtimes, but a
suppression matches on module names, and the vendor libraries carry no frame
pointers, so ASan's default fast unwinder produced the `<unknown module>` frame
above and nothing could match it. `test/memory-check.sh` runs the ASan stage
with `fast_unwind_on_malloc=0`, which restores the real stack
(`libtbbmalloc → libtbb → libOpenCL.so.1 → clGetPlatformIDs`). If the report
reappears after a base-image change, run the stage once without the suppression
file to see which library the new stack names, and add it there — do not turn
`detect_leaks` off.

### 7.4 `Failed to locate <Sphinx|AStyle|ClangFormat|...> executable`

Informational output from RAJA's BLT build system probing optional developer
tools. It has no effect on the build or on the solver, and it is no longer
printed: `CMakeLists.txt` forces the corresponding `ENABLE_<TOOL>` switches off
before RAJA is configured, since ten such lines per configure — three trees per
memory-safety run — only train a reader to skim past real output.

### 7.5 `Warning: noted but unhandled ioctl 0x3000000x` in the Valgrind stage

Valgrind does not model the NVIDIA driver's private `ioctl` numbers, which the
OpenCL ICD loader issues while enumerating vendors — including the NVIDIA one,
which is installed in this image even when the stage only uses the Intel CPU
device. The warning describes Valgrind's knowledge of the kernel interface, not
the solver, and the stage keeps its `ERROR SUMMARY: 0 errors` verdict.

Pinning the loader to a single vendor file (`OCL_ICD_VENDORS=.../intel64.icd`)
removes the lines but also hides the CPU device — the OpenCL cases then skip,
which this suite treats as a failure, correctly: an unchecked backend is worse
than a warning.

### 7.6 GPU stage finds no device on a machine that has one

```
Unable to determine the device handle for GPU0: 0000:01:00.0: Unknown Error
```

NVML found the driver but could not talk to the GPU. On a laptop with hybrid
graphics — an Intel iGPU driving the display and an NVIDIA dGPU for compute,
which is the configuration this project is developed on — there are two common
causes, and neither is a container problem:

1. **A driver upgrade with the old kernel modules still loaded.** The user-space
   libraries the container gets injected are the *new* ones; the running modules
   are the old ones, and the two do not talk to each other. Compare them on the
   host:

   ```bash
   cat /proc/driver/nvidia/version      # what is loaded right now
   modinfo -F version nvidia            # what is installed on disk
   ```

   If they differ, reload the modules (`sudo rmmod nvidia_uvm nvidia_drm
   nvidia_modeset nvidia`, then any CUDA program reloads them) or reboot. This
   is what produced the message above; after a reboot the same
   `docker run --gpus all ... test gpu` ran the whole device sweep.

2. **The dGPU is runtime-suspended.** Laptops enable fine-grained Runtime D3, so
   the card is powered down whenever nothing holds it open:

   ```bash
   grep DynamicPowerManagement /proc/driver/nvidia/params   # 3 = fine-grained
   cat /sys/bus/pci/devices/0000:01:00.0/power/runtime_status
   ```

   Waking it across PCIe takes longer than one immediate NVML call, so the GPU
   stage now retries three times over ~10 s. Keeping `nvidia-persistenced`
   running (`systemctl enable --now nvidia-persistenced`) holds the device
   resident and removes the resume latency entirely.

A third cause is not about the GPU at all: **Docker Desktop's daemon runs inside
a VM and exposes no GPU**. `docker context ls` must show `default` (the native
`unix:///var/run/docker.sock` daemon, which needs `docker.service` running and
the NVIDIA Container Toolkit registered) rather than `desktop-linux`, otherwise
`--gpus all` fails or silently yields no device.

The Intel iGPU is irrelevant to all of this: the container is never given
`/dev/dri`, so it sees exactly one OpenCL GPU platform — NVIDIA's — and the CPU
OpenCL device always comes from Intel's CPU runtime, never from the iGPU.

**What the stage does about it.** A container without the driver stack (started
without `--gpus all` / `--nv`) still skips, since verifying the host backends is
what such a run is for. A container that *has* the driver stack and still gets
no device now **fails**: the GPU backends would otherwise go unchecked while the
pipeline printed "All Test Suites Passed". Use `LBMINI_ALLOW_NO_GPU=1` to
downgrade that back to a skip deliberately. A Kokkos built with the CUDA backend
— which cannot initialise at all without a device — is still caught at start-up
(`Kokkos::initialize() failed, Kokkos-backed cases will be skipped`) instead of
aborting the whole test binary before its first case.

### 7.7 `opencl_gpu` cases skip only under AddressSanitizer

The OpenCL GPU sweep runs from the sanitized tree (`build-asan`), and there — and
only there — the NVIDIA platform can disappear: the sanitized process enumerates
`Intel(R) OpenCL` alone, finds no `CL_DEVICE_TYPE_GPU` device, and skips all 13
cases, which the stage reports as a failure.

The NVIDIA driver maps its unified address space over the region AddressSanitizer
reserves as an unmapped `PROT_NONE` *shadow gap*. The mapping fails, the vendor's
`clGetPlatformIDs` init fails with it, and the ICD loader drops that vendor
without a word — two platforms become one. `test/run-tests.sh` therefore runs
that one pass with `protect_shadow_gap=0` (`LBMINI_ASAN_OPTIONS_DEVICE`). No
allocation ever lives in the gap, so redzones, use-after-free, and leak detection
are exactly as strict as everywhere else.

The related `AddressSanitizer CHECK failed: ... sigaltstack` aborts from the same
driver's threads are why every sanitized invocation in this project also carries
`use_sigaltstack=0`.

### 7.8 GPU stage appears to hang for tens of minutes

`compute-sanitizer --tool racecheck` cannot follow kernels launched concurrently
from several host threads. On the device sweep's
`MultiThreadedConcurrentExecutionIsSafe` case — four threads, four solvers — the
Kokkos instantiation intermittently ends with `Internal Sanitizer Error: Detected
a failure to track a kernel launch` counted as a hazard, and the RAJA one
deadlocks with no output at all. Both are properties of the tool.

That case is now excluded from `racecheck` only (it still runs uninstrumented and
under `memcheck`, `initcheck` and `synccheck`), and every step of the stage is
bounded by `LBMINI_SANITIZER_TIMEOUT` (default 900 s) so a tool that stops making
progress fails the stage instead of consuming the night. For reference, the whole
GPU stage takes about 40 s on an RTX 4050 Laptop GPU.

---

## 8. Verification and Health Check

To verify all backends inside the container environment:

```bash
# Verify binary availability
apptainer exec lbm-miniapp.sif which lbm-miniapp-main lbm-miniapp-test

# Run unit tests
apptainer run --app test lbm-miniapp.sif

# Inspect available OpenCL platforms (NVIDIA)
apptainer exec --nv lbm-miniapp.sif clinfo
```
