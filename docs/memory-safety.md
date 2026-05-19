# Memory-safety verification of the LBM backends

**This document is a gate, not a suggestion.** No backend enters a benchmarking
campaign (`results/campaign.py`) before it has passed every check listed here.

The mini-app exists to compare backends, so every backend re-implements the same
solver against a different memory model: aligned host buffers (`Plain`,
`OpenMP CPU`, `MPI`), `omp_target_alloc` device memory (`OpenMP GPU`,
`OpenACC GPU`), `cudaMalloc` (`CUDA`), `cl::Buffer` (`OpenCL`),
`Kokkos::View` (`Kokkos`) and hand-rolled allocations (`RAJA`). A wrong number
is an inconvenience; an out-of-bounds write, a use-after-free or a double-free
on a shared GPU node can hang the device, take down other users' jobs, force a
node reboot and — with a mis-programmed device — damage hardware. Benchmarks
also run long, wide and unattended, which is exactly the regime where a latent
corruption stops being latent.

---

## 1. What is being defended against

| Bug class | Where it lives in this code base |
|---|---|
| Out-of-bounds read | `stream()` gathers `f[i0..i3]` through the precomputed `streamIdx_` table with no bounds check, in every backend |
| Out-of-bounds write | Buffers sized from `nx * ny` while kernels index `i * ny + j`, `d * N + cell`, `idc * N + cell`; zero/negative extents and overflowing products break the assumptions, while one-cell and odd extents exercise boundary paths |
| Integer overflow into a wild address | The streaming tables are `std::int32_t`; `nx * ny > 2^31` wraps to a negative index. `cellIndex()` casts extents to `int` in some backends |
| Use-after-free | Global runtimes must outlive every solver allocation; cached `fCur_`/`fAlt_` pointers must never outlive their owning object |
| Double free | Raw `cudaMalloc` / `omp_target_alloc` / command-buffer handles must have exclusive ownership and non-copyable owners |
| Reading uninitialised / stale memory | Device buffers read back on the host without a fence / `deep_copy` / `enqueueReadBuffer` |
| Data race (a corruption under a different name) | Shared mutable state between instances, unsynchronised halo exchange in the MPI backend |
| Invalid polymorphic destruction | Every solver must be owned through the genuine lattice-independent `ILbmTubeNd` base with a virtual destructor |

---

## 2. The automated part: the memory-safety suite

| File | What it proves |
|---|---|
| `test/MemorySafetyHarness.hpp` | Shared fixtures and the mesh-shape sweep (canonical, odd, single-row, single-column, 1x1) |
| `test/memory_safety_test.cpp` | Per-backend: in-bounds allocation/stepping/read-back for every shape, odd/even step counts, `Run(0)` as a no-op, getters returning owning copies, repeated construct/destruct, one solver surviving another's destruction, two live solvers not sharing state, destruction through the interface pointer, and the layout guard for the `Executor` type-erasure |
| `test/streaming_table_test.cpp` | Every gather index of the host backends is inside `[0, nx * ny)`, table lengths are exact, weights are finite and convex, cell count fits `int32` |
| `test/kokkos_lbm_test.cpp`, `test/raja_lbm_test.cpp` | Numerical agreement with the `Plain` reference (a corrupted run usually also diverges) |

Adding a backend to the suite means adding one traits struct to the type list in
`test/memory_safety_test.cpp` — no test body is backend-specific.

Which backends the binary actually contains is decided at configure time:
`Kokkos` / `RAJA` follow `ENABLE_KOKKOS` / `ENABLE_RAJA`, the OpenMP CPU solver
follows `ENABLE_TEST_OPENMP_BACKEND` **and** the presence of an
`OpenMP::OpenMP_CXX` target, and the GPU-targeted instantiations need
`ENABLE_TEST_DEVICE_BACKENDS=ON` plus an accessible NVIDIA GPU at run time. Each
backend appears in the test name
(`BackendMemorySafety/kokkos_cpu.EveryMeshShapeStaysInBounds`), so
`ctest -N` or `--gtest_list_tests` tells you exactly what a green run covered.

**A green test run on its own proves nothing.** The suite is written to *touch*
every byte a backend allocates; the verdict comes from the tool it runs under.

**A skipped backend is an unchecked backend, and that is a failure.** Every
case in the suite skips itself when its backend is compiled in but unavailable
at run time — which is correct behaviour for a test and useless as a gate,
because a skip is green. The stage that owns each backend therefore audits the
skips instead of trusting the exit code:

| Stage | Rule |
|---|---|
| ASan/UBSan (`test/memory-check.sh`) | Any **host** backend that skips fails the stage; the GPU-targeted cases are expected to skip, because the stage hides the devices on purpose. `ctest --no-tests=error` additionally rejects an empty test set. |
| GPU (`test/run-tests.sh gpu`) | Reached only after a device was found, so **any** skip in the device sweep fails; a binary built without `ENABLE_TEST_DEVICE_BACKENDS` fails immediately rather than reporting success over zero device tests. |

This is how an OpenCL ICD that disappears from `/etc/OpenCL/vendors`, or a
Kokkos that cannot initialise, stops being a silent loss of coverage.

---

## 3. Unified Test Commands (Primary Gate)

All tests are consolidated into a few high-level test runners executable via Docker, Apptainer, or direct host scripts:

| Test Target | What it Runs | Docker Command | Apptainer Command |
|---|---|---|---|
| **Unit** | Full GoogleTest suite (unit, numerical agreement, and memory safety fixtures) | `docker run --rm lbm-miniapp:latest test unit` | `apptainer run --app test lbm-miniapp.sif unit` |
| **CPU** | AddressSanitizer + UBSan, Valgrind memcheck, and ThreadSanitizer data-race checks, run in parallel | `docker run --rm lbm-miniapp:latest test cpu` | `apptainer run --app test lbm-miniapp.sif cpu` |
| **GPU** | NVIDIA `compute-sanitizer` (`memcheck`, `racecheck`, `initcheck`, `synccheck`) across GPU solvers & shapes, plus the OpenCL GPU sweep under ASan | `docker run --rm --gpus all lbm-miniapp:latest test gpu` | `apptainer run --app test --nv lbm-miniapp.sif gpu` |
| **MPI** | Multi-rank domain decomposition & halo exchange sweep (1, 2, 3, 4 ranks), then 2 ranks under ASan + LSan | `docker run --rm lbm-miniapp:latest test mpi` | `apptainer run --app test lbm-miniapp.sif mpi` |
| **All** | Complete end-to-end verification pipeline (Unit + CPU + GPU + MPI) | `docker run --rm --gpus all lbm-miniapp:latest test all` | `apptainer run --app test --nv lbm-miniapp.sif all` |

---

## 4. Test Suites in Detail

### 4.1 Unit Tests (`test unit`)
Runs the comprehensive GoogleTest binary (`lbm-miniapp-test`) verifying functional correctness, solver numerical convergence against the `Plain` ISO C++ reference, boundary conditions, and memory-safety test matrices across all compiled backends:
```bash
# Run all unit tests (CPU backends):
docker run --rm lbm-miniapp:latest test unit

# Run all unit tests including GPU device backends (requires NVIDIA GPU):
docker run --rm --gpus all lbm-miniapp:latest test unit
```

### 4.2 CPU Memory Safety & Concurrency (`test cpu`)
Executes the automated 3-stage host-side analysis pipeline (`test/memory-check.sh --tsan`):
1. **Stage 1: AddressSanitizer (ASan) + UndefinedBehaviorSanitizer (UBSan)** — detects out-of-bounds access, use-after-free, integer overflows, alignment issues, `malloc`/`delete` mismatches and leaks.
2. **Stage 2: Valgrind Memcheck** — verifies absolute absence of direct and indirect memory leaks or uninitialized memory reads.
3. **Stage 3: ThreadSanitizer (TSan)** — detects thread data-races in multi-threaded host backends (OpenMP, Kokkos OpenMP, RAJA OpenMP).
```bash
docker run --rm lbm-miniapp:latest test cpu
```

**Every stage runs in parallel.** An instrumented process spends its time
inside one instrumented thread, so a serial run of this gate left a 16-core
machine 15/16 idle for the better part of an hour — the OpenCL CPU cases alone
spend minutes each inside the ICD's JIT. The ASan stage is driven through
`ctest --parallel`, and the Valgrind and TSan stages — one GoogleTest process
each — are split into independent GTest shards (`GTEST_TOTAL_SHARDS` /
`GTEST_SHARD_INDEX`) that run concurrently, with each shard's output buffered
and replayed in order so a failure stays readable. Sharding changes nothing
about what is checked: every case still runs, in its own fresh process, under
the same tool, and the stage fails if any shard fails. Parallelism defaults to
half the core count (two hardware threads per job, since each job also starts a
thread team) and is overridable with `-p N` or `LBMINI_TEST_JOBS`; the TSan
stage uses wider, fewer shards because a race is only visible to a real thread
team. Measured on a 16-core host: the ASan stage 335 s of wall time for 106
tests whose serial sum is over 1,700 s, and the whole `test all` pipeline about
40 minutes instead of a night.

This stage is **host-only by construction**: every tree it uses is configured
with `-DENABLE_CUDA=OFF` and the script exports an empty `CUDA_VISIBLE_DEVICES`,
so no stage can open a device context and the NVIDIA OpenCL ICD stops
advertising its GPU. Valgrind therefore runs the dedicated uninstrumented
`build-memcheck` tree (pre-built inside the container), **not** the CUDA-enabled
`build`: pointing memcheck at the latter drags the NVIDIA driver through the
tool and buries the report under tens of thousands of suppressed driver errors.

**Vendor allocations and the unwinder.** Enumerating OpenCL platforms arms the
ICD loader's one-time init, in which Intel's oneTBB allocates a ~25 kB arena
that lives until the process exits. `test/lsan_suppressions.supp` covers such
third-party allocations, but a suppression can only match a stack that carries
module names: the vendor libraries have no frame pointers, so ASan's default
fast unwinder emitted a single `<unknown module>` frame and every OpenCL test
aborted on a leak that is not this project's. `test/memory-check.sh` therefore
runs the ASan stage with `fast_unwind_on_malloc=0` (bounded by
`malloc_context_size=15`), which recovers the real chain
`libtbbmalloc → libtbb → libOpenCL.so.1 → clGetPlatformIDs` and lets the
suppression apply. Leak detection itself stays on — this makes stacks readable,
it does not hide anything. For the same reason Valgrind runs with
`--num-callers=30`: at the default depth of 12 the vendor frames a suppression
has to match fall off the end of the stack. Its error set is
`definite,indirect`, an indirectly lost block being the remainder of the same
leak.

### 4.3 GPU Verification & Sanitization (`test gpu`)
Runs NVIDIA's `compute-sanitizer` across all enabled GPU backends (`CUDA`, `Kokkos GPU`, `RAJA GPU`, `OpenACC GPU`, `OpenMP GPU`) over all representative mesh shapes:
1. `memcheck` — out-of-bounds device memory access, shared memory bounds, and allocation leak checks.
2. `racecheck` — shared memory and global memory hazard / race detection.
3. `initcheck` — uninitialized global memory access detection.
4. `synccheck` — stream synchronization and barrier hazards.
```bash
docker run --rm --gpus all lbm-miniapp:latest test gpu
```

The stage runs two things, not one:

1. **The device-backend memory-safety sweep** — the same 13 cases the host
   backends get from §4.2 (every mesh shape, odd/even step counts, repeated
   construction, two live solvers, concurrent instances, getter copy-back),
   instantiated for `cuda_gpu`, `kokkos_gpu`, `raja_gpu` and `opencl_gpu`, run
   under `memcheck`, `initcheck`, `racecheck` and `synccheck`. This requires
   `ENABLE_TEST_DEVICE_BACKENDS=ON`, which the container's `build` and
   `build-asan` trees both set — without it the stage has no device test to run
   and now says so instead of passing.
2. **The solver itself, per backend**, over the four tools, which is what
   exercises the real time loop, the output path and the teardown.

Three properties of this stage are deliberate:

- **A clean report, not merely a zero exit code.** `compute-sanitizer` forwards
  the application's exit status unless told otherwise, so a run that printed
  `ERROR SUMMARY: 12 errors` used to pass. Every invocation now carries
  `--error-exitcode 1` *and* has its summary lines audited (`ERROR SUMMARY`,
  `LEAK SUMMARY`, `RACECHECK SUMMARY`), because a leak-only finding does not
  always reach the exit-code path.
- **The OpenCL GPU backend is sanitized by AddressSanitizer instead.**
  `compute-sanitizer` instruments the CUDA driver/runtime API; OpenCL reaches
  the same device through the NVIDIA ICD, so every tool aborts with *"Target
  application terminated before first instrumented API call"* — an error that
  says nothing about the code. Leaving it there made OpenCL GPU the one backend
  with no memory tool at all, so the stage runs its sweep from the sanitized
  tree (`build-asan`, which is CUDA-free but keeps the OpenCL GPU cases): ASan
  covers the host-side buffer sizing, the SoA/AoS transpose in the getters and
  the release of contexts, queues and command buffers.
- **A GPU backend that cannot use a GPU fails, it does not fall back.** Kokkos
  and RAJA built without a device backend, and OpenCL with no GPU device, now
  throw at construction instead of quietly executing the "GPU" run on the host,
  and the runner reports the solver's own message instead of swallowing it.
- **A driver that is present but hands out no device fails the stage.** A
  container started without `--gpus all` / `--nv` legitimately skips: host
  backends are what such a run is for. A container that *has* the driver stack
  (`/dev/nvidiactl`, `libcuda.so.1`) and still gets no device from `nvidia-smi`
  is a broken host, and reporting "all test suites passed" for a pipeline that
  never touched a GPU is exactly the silent loss of coverage this suite exists
  to prevent. The probe retries three times over ~10 s first, because a laptop
  dGPU in fine-grained Runtime D3 has to be resumed across PCIe before it
  answers; `LBMINI_ALLOW_NO_GPU=1` restores the skip on purpose.
- **Every step is time-bounded** (`LBMINI_SANITIZER_TIMEOUT`, default 900 s).
  A tool that stops making progress is a failed check, not a licence to run all
  night — see the `racecheck` limitation below, which deadlocked this stage for
  over 40 minutes.

**Two vendor-tool limitations shape this stage, and both are worked around
without losing coverage:**

| Symptom | Cause | What the stage does |
|---|---|---|
| `racecheck` reports one hazard with `Internal Sanitizer Error: Detected a failure to track a kernel launch` (Kokkos), or never returns (RAJA) | The tool cannot follow kernels launched concurrently from several host threads, which is exactly what `MultiThreadedConcurrentExecutionIsSafe` does with four | That one case is excluded from **`racecheck` only**. It still runs uninstrumented, and under `memcheck`, `initcheck` and `synccheck`; `racecheck` looks for races *inside* a kernel and those kernels are the ones the other twelve cases launch, while the host-side concurrency is ThreadSanitizer's job in §4.2 |
| The sanitized binary enumerates one OpenCL platform (`Intel(R) OpenCL`) instead of two, and every `opencl_gpu` case skips | The NVIDIA driver maps its unified address space over the region AddressSanitizer keeps as an unmapped `PROT_NONE` *shadow gap*; the mapping fails and the ICD loader drops the vendor silently | The OpenCL GPU pass — and only that pass — adds `protect_shadow_gap=0`. The gap holds no allocation, so redzones, use-after-free and leak detection are unaffected |

> **Note on Windows / WSL2 Host:**
> In Windows / WSL2 environments with WDDM graphics drivers, NVIDIA `compute-sanitizer` requires the WDDM debugger interface to be enabled on the Windows host by running `EnableDebuggerInterface.bat` as Administrator. On native Linux HPC clusters, cloud servers, and GPUs running in TCC mode, `compute-sanitizer` runs natively without WDDM restrictions.

### 4.4 MPI Multi-Rank Sweep (`test mpi`)
Validates domain decomposition, non-blocking halo exchanges, boundary synchronization, and buffer lifetimes under `mpirun` across 1, 2, 3, and 4 MPI ranks with memory leak and bounds analysis:
```bash
docker run --rm lbm-miniapp:latest test mpi
```

The rank sweep proves that the ranks agree; it looks at no memory, which left
the MPI backend — the one backend whose halo exchange, pack/unpack buffers and
teardown path are written by hand — covered by nothing. Valgrind cannot be
pointed at a launcher without a suppression file per MCA component, so the
stage finishes with **two ranks of the sanitized binary** (`build-asan`) under
ASan + LSan, with the launcher's own allocations (`libmpi`, `libpmix`, UCX,
UCC, HCOLL) named in `test/lsan_suppressions.supp` so the solver's own
allocations are what remains visible.

**A run that advances zero steps is not a passing run.** These stages generated
their configuration with `tmax: 0.005`, while `dt` is derived from the mesh
(`dx · uLuRef / uPhysRef` ≈ 0.009 at `nx: 64`): `kSteps = tmax / dt` truncated
to **0**, so every GPU backend and every MPI rank wrote its step-0 snapshot,
printed "Simulation completed" and exited without a single kernel launch or
halo exchange — and all four `compute-sanitizer` tools dutifully reported a
clean run over it. `lbmini::Executor::Run` now refuses such a configuration
with an explanatory error, the generated configurations use the same `tmax` as
`lbmini/config.yaml`, and both stages additionally require the run to report
time steps before accepting it.

### 4.5 Full Pipeline (`test all`)
Runs all four verification suites sequentially to provide a complete pass/fail verdict before benchmarking:
```bash
docker run --rm --gpus all lbm-miniapp:latest test all
```

---

## 5. Direct Host Execution (Without Containers)

If developing directly on a Linux/HPC workstation outside containers, the same consolidated runner is used:

```bash
# 1. Run unit tests
test/run-tests.sh unit

# 2. Run CPU memory-safety & race analysis
test/run-tests.sh cpu

# 3. Run GPU compute-sanitizer suite (requires NVIDIA GPU)
test/run-tests.sh gpu

# 4. Run MPI multi-rank tests
test/run-tests.sh mpi

# 5. Run full test suite
test/run-tests.sh all
```

---

## 6. Pre-benchmark checklist

- [ ] Unit tests pass: `docker run --rm lbm-miniapp:latest test unit`.
- [ ] CPU memory-safety suite passes: `docker run --rm lbm-miniapp:latest test cpu` (ASan + UBSan + Valgrind + TSan).
- [ ] GPU diagnostic suite passes: `docker run --rm --gpus all lbm-miniapp:latest test gpu` (`memcheck`, `initcheck`, `racecheck`, `synccheck`).
- [ ] MPI multi-rank suite passes: `docker run --rm lbm-miniapp:latest test mpi` (1, 2, 3, 4 ranks).
- [ ] Full pipeline passes: `docker run --rm --gpus all lbm-miniapp:latest test all`.
- [ ] Every mesh shape in `MeshShapes()` accepted or rejected *explicitly* — a rejected configuration must throw `std::invalid_argument`, never index out of bounds.
- [ ] No backend **skipped**: a skip is green and proves nothing (see §2). The stages enforce this, but read the summary anyway.
- [ ] Every run in the pipeline actually advanced time steps — a configuration whose `tmax` is shorter than the derived `dt` measures and checks nothing (see §4.4).
- [ ] Benchmark binaries rebuilt from a clean, **uninstrumented** `Release` tree: sanitizer flags change both timing and layout.

---

## 7. Rules for new or modified backend code

1. Every buffer's size is derived from a single named expression (`N_`, `uSize_`, `distSize_`), never open-coded at the allocation site.
2. Any indirect index (streaming tables, halo maps, rank neighbours) must be provably inside its target buffer, and that proof must be a test — see `test/streaming_table_test.cpp`.
3. Ownership is RAII or explicitly deleted: a class that owns a raw handle deletes its copy and move operations, or implements them.
4. Global runtimes (Kokkos, MPI, CUDA/OpenCL contexts) are initialised and finalised **once per process**, outside the solver lifetime.
5. Getters return owning copies. Never hand out a pointer or a view into solver storage — the memory-safety suite asserts this.
6. Host code never reads a device buffer without an explicit fence / copy-back.
7. A new backend joins the type list in `test/memory_safety_test.cpp` in the same commit that introduces it, and its device runner is added to the table in section 4.
8. Finiteness is checked with `lbmini::IsFinite` (`lbmini/Utils.hpp`), never `std::isfinite`/`std::isnan`: the Release configurations use `-ffast-math` (`-ffinite-math-only`), which folds those to constants and would turn the NaN/Inf assertions into no-ops.
9. A backend never substitutes one processor for another. A CPU-targeted solver
   requests host execution only, a GPU-targeted solver requests a device only,
   and an unavailable target throws with an actionable message. A silent
   fallback (OpenCL picking `CL_DEVICE_TYPE_ALL`, Kokkos/RAJA degrading to the
   host execution space) turns a benchmark into a fabricated measurement.
