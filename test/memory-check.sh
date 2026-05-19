#!/bin/bash
# Pre-benchmark memory-safety gate. See docs/memory-safety.md for what each
# stage proves and for the checks that cannot be automated from here (GPU and
# MPI backends need a device / a launcher).
#
# Usage:
#   test/memory-check.sh                 # ASan+UBSan build + full ctest, then Valgrind if available
#   test/memory-check.sh --tsan          # additionally build and run with ThreadSanitizer
#   test/memory-check.sh --no-valgrind   # skip the Valgrind stage
#   test/memory-check.sh -j 8            # build parallelism
#   test/memory-check.sh -p 4            # test parallelism (ctest jobs / GTest shards)
#
# Any extra arguments are forwarded to every `cmake` configure call, e.g.
#   test/memory-check.sh -DENABLE_KOKKOS=OFF -DENABLE_RAJA=OFF
#
# Every stage runs in parallel. A sanitized or Valgrinded process spends its
# time inside one instrumented thread, so a single serial run leaves a 16-core
# machine 15/16 idle for the better part of an hour: the ASan stage is driven
# through `ctest --parallel`, and the Valgrind and TSan stages - which are one
# GoogleTest process each - are split into independent GTest shards
# (`GTEST_TOTAL_SHARDS` / `GTEST_SHARD_INDEX`) that run concurrently. Sharding
# changes nothing about what is checked: every test still executes, in its own
# fresh process, under the same tool. Per-shard output is buffered and replayed
# shard by shard so a failure is still readable, and the stage fails if *any*
# shard fails.

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SOURCE_DIR}"

RUN_TSAN=0
RUN_VALGRIND=1
JOBS="$(nproc 2>/dev/null || echo 4)"
TEST_JOBS=""
# `-DENABLE_TEST_DEVICE_BACKENDS` is deliberately *not* forced here: the trees
# built by the container enable it so the same binary can also be used for the
# device-backend suite (see test/run-tests.sh), and the GPU-targeted cases skip
# themselves anyway while `CUDA_VISIBLE_DEVICES` is empty (below). Forcing it
# off would only re-configure and re-link those trees on every run.
CMAKE_ARGS=(
    "-DENABLE_CUDA=OFF"
)

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --tsan) RUN_TSAN=1; shift ;;
        --no-valgrind) RUN_VALGRIND=0; shift ;;
        -j) JOBS="$2"; shift 2 ;;
        -p|--test-jobs) TEST_JOBS="$2"; shift 2 ;;
        *) CMAKE_ARGS+=("$1"); shift ;;
    esac
done

# Test parallelism. Each job is a whole instrumented process that also starts a
# thread team, so the default keeps two hardware threads per job instead of
# oversubscribing the machine with (jobs x threads) runnable threads.
if [[ -z "${TEST_JOBS}" ]]; then
    TEST_JOBS="${LBMINI_TEST_JOBS:-$(( JOBS / 2 ))}"
fi
[[ "${TEST_JOBS}" -lt 1 ]] && TEST_JOBS=1

# Under Valgrind / TSan only the memory-safety suite is worth the slowdown; the
# ASan stage above runs the whole ctest set, agreement tests included.
MEMORY_FILTER='BackendMemorySafety*:StreamingTableBounds*'

# This gate covers the host backends only (every tree below is configured with
# -DENABLE_CUDA=OFF). Hiding the GPUs makes that explicit: no stage here may
# open a device context, and the NVIDIA OpenCL ICD stops advertising its GPU,
# so a CPU backend cannot quietly end up running on the accelerator.
export CUDA_VISIBLE_DEVICES=""

# Runs one GoogleTest command as `${1}` independent shards in parallel.
#
# Usage: run_sharded <shards> <label> <command...>
#
# The command is executed verbatim (prefix it with `env VAR=...` to set
# per-shard variables); only GTEST_TOTAL_SHARDS / GTEST_SHARD_INDEX are added.
run_sharded() {
    local shards="$1"
    local label="$2"
    shift 2

    local pids=()
    local logs=()
    local idx
    local status=0

    for (( idx = 0; idx < shards; ++idx )); do
        local log
        log="$(mktemp "/tmp/lbmini_${label}_shard${idx}_XXXXXX.log")"
        logs+=("${log}")
        GTEST_TOTAL_SHARDS="${shards}" GTEST_SHARD_INDEX="${idx}" \
            "$@" > "${log}" 2>&1 &
        pids+=("$!")
    done

    for (( idx = 0; idx < shards; ++idx )); do
        if ! wait "${pids[idx]}"; then
            status=1
        fi
        if [[ "${shards}" -gt 1 ]]; then
            echo "-------- ${label}: shard $(( idx + 1 ))/${shards} --------"
        fi
        cat "${logs[idx]}"
        rm -f "${logs[idx]}"
    done

    return "${status}"
}

echo "==> Stage 1/3: AddressSanitizer + UndefinedBehaviorSanitizer"
# Sanitized builds must not be reused for benchmarking, hence the separate tree.
cmake -B build-asan -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLBMINI_SANITIZER=address+undefined \
    "${CMAKE_ARGS[@]}"
cmake --build build-asan --parallel "${JOBS}"

# `abort_on_error` turns the first violation into a non-zero exit code instead
# of a report buried in the log; leak detection must stay on.
# `alloc_dealloc_mismatch` / `new_delete_type_mismatch` catch a `free()` of a
# `new[]` allocation and a `delete` through the wrong type - both corrupt the
# heap silently in a release build, and both are exactly what a backend written
# around raw device buffers gets wrong.
#
# `fast_unwind_on_malloc=0` is what makes the suppression file below work at
# all. The ICD loader arms its one-time init (`clGetPlatformIDs` ->
# `pthread_once`) the moment any platform is enumerated, and Intel's oneTBB
# allocates its 25,600-byte arena in there, never freeing it. Neither library
# carries frame pointers, so the default fast unwinder walked out of the stack
# and produced a single `<unknown module>` frame - and a leak whose stack
# carries no module name matches no `leak:` template, which is why every
# OpenCL test used to abort on a third-party allocation. The slow (CFI)
# unwinder recovers the real chain (`libtbbmalloc` -> `libtbb` ->
# `libOpenCL.so.1` -> `clGetPlatformIDs`), which the suppression file below
# covers. `malloc_context_size` bounds what that costs: the frames that decide
# the match sit at depth 5-7, so 15 keeps a wide margin while halving the
# per-allocation unwinding work. This trades unwinding time for nothing:
# leaks in this project's own code are still reported, with a usable stack.
export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:strict_string_checks=1:detect_stack_use_after_return=1:alloc_dealloc_mismatch=1:new_delete_type_mismatch=1:use_sigaltstack=0:fast_unwind_on_malloc=0:malloc_context_size=15"
if [[ -f "${SOURCE_DIR}/test/lsan_suppressions.supp" ]]; then
    export LSAN_OPTIONS="suppressions=${SOURCE_DIR}/test/lsan_suppressions.supp"
fi
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"

# `--no-tests=error`: an empty test set is a configuration failure, not a pass.
# The output is kept so the skip audit below can read it.
ASAN_CTEST_LOG="$(mktemp /tmp/lbmini_asan_ctest_XXXXXX.log)"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" \
    ctest --test-dir build-asan \
        --output-on-failure \
        --no-tests=error \
        --parallel "${TEST_JOBS}" \
        2>&1 | tee "${ASAN_CTEST_LOG}"

# A host backend that is compiled in but unavailable at run time skips its
# whole memory-safety sweep, and a skip is green: that is how a backend stops
# being memory-checked without anyone noticing (an OpenCL ICD that disappeared
# from /etc/OpenCL/vendors is enough). GPU-targeted cases are expected to skip
# here - this gate hides the devices on purpose - so only host skips fail.
HOST_SKIPS="$(grep -E '\*\*\*Skipped' "${ASAN_CTEST_LOG}" | grep -vEi '_gpu|Gpu' || true)"
GPU_SKIPS="$(grep -cE '\*\*\*Skipped' "${ASAN_CTEST_LOG}" || true)"
rm -f "${ASAN_CTEST_LOG}"
if [[ -n "${HOST_SKIPS}" ]]; then
    echo "Error: a host backend skipped its memory-safety sweep instead of running it:" >&2
    echo "${HOST_SKIPS}" | sed 's/^/    /' >&2
    echo "    A skipped backend is an unchecked backend. Fix its runtime" >&2
    echo "    availability (ICD, runtime library, device visibility) or remove it" >&2
    echo "    from the build; do not let the suite report green for it." >&2
    exit 1
fi
echo "==> ASan/UBSan stage complete (${GPU_SKIPS} GPU-targeted case(s) skipped, as expected here)."

if [[ "${RUN_VALGRIND}" -eq 1 ]] && command -v valgrind >/dev/null 2>&1; then
    echo "==> Stage 2/3: Valgrind memcheck (uninstrumented build), ${TEST_JOBS} shard(s)"
    # Valgrind and ASan are mutually exclusive.
    # The uninstrumented tree is reused across runs to avoid a rebuild.
    # Only the dedicated CUDA-free tree qualifies. The default ./build (and the
    # /usr/local/bin symlink pointing at it) links the CUDA runtime and the
    # Kokkos CUDA backend: running that under Valgrind drags the NVIDIA driver
    # through memcheck, which produced 33,574 suppressed errors and 273 MB of
    # suppressed allocations - noise that hides real findings, in a stage that
    # is not supposed to touch the GPU at all.
    VALGRIND_BIN=""
    if [[ -x "./build-memcheck/lbm-miniapp-test" ]]; then
        VALGRIND_BIN="./build-memcheck/lbm-miniapp-test"
    else
        cmake -B build-memcheck -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo "${CMAKE_ARGS[@]}"
        cmake --build build-memcheck --parallel "${JOBS}"
        VALGRIND_BIN="./build-memcheck/lbm-miniapp-test"
    fi

    VALGRIND_SUPP_OPT=()
    if [[ -f "${SOURCE_DIR}/test/valgrind_suppressions.supp" ]]; then
        VALGRIND_SUPP_OPT=("--suppressions=${SOURCE_DIR}/test/valgrind_suppressions.supp")
    fi

    # `indirect` joins `definite` in the error set: an indirectly lost block is
    # a block only reachable from a leaked one, i.e. the rest of the same leak.
    # `--num-callers=30` keeps the vendor frames a suppression has to match
    # from falling off the end of the stack (the default 12 is not enough once
    # a call goes through the OpenCL ICD into oneTBB).
    # OMP_NUM_THREADS=2 keeps the concurrent code paths covered without paying
    # Valgrind's thread-serialisation cost for a full team per shard.
    run_sharded "${TEST_JOBS}" valgrind \
        env OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" valgrind \
            --tool=memcheck \
            --error-exitcode=1 \
            --leak-check=full \
            --errors-for-leak-kinds=definite,indirect \
            --num-callers=30 \
            "${VALGRIND_SUPP_OPT[@]}" \
            "${VALGRIND_BIN}" --gtest_filter="${MEMORY_FILTER}"
else
    echo "==> Stage 2/3: Valgrind skipped"
fi

if [[ "${RUN_TSAN}" -eq 1 ]]; then
    # ThreadSanitizer needs a real thread team per process to see a race at
    # all, so its shards are wider (4 threads each) and correspondingly fewer.
    TSAN_SHARDS="$(( JOBS / 4 ))"
    [[ "${TSAN_SHARDS}" -lt 1 ]] && TSAN_SHARDS=1
    echo "==> Stage 3/3: ThreadSanitizer, ${TSAN_SHARDS} shard(s)"
    cmake -B build-tsan -S . \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DLBMINI_SANITIZER=thread \
        -DENABLE_OPENCL=OFF \
        -DENABLE_MPI=OFF \
        "${CMAKE_ARGS[@]}"
    cmake --build build-tsan --parallel "${JOBS}"
    TSAN_OPTS="halt_on_error=1:ignore_noninstrumented_modules=1:history_size=7"
    if [[ -f "${SOURCE_DIR}/test/tsan_suppressions.txt" ]]; then
        TSAN_OPTS="${TSAN_OPTS}:suppressions=${SOURCE_DIR}/test/tsan_suppressions.txt"
    fi
    run_sharded "${TSAN_SHARDS}" tsan \
        env OMP_NUM_THREADS=4 TSAN_OPTIONS="${TSAN_OPTS}" \
            ./build-tsan/lbm-miniapp-test --gtest_filter="${MEMORY_FILTER}"
else
    echo "==> Stage 3/3: ThreadSanitizer skipped (pass --tsan to enable)"
fi

echo
echo "Host-side memory checks passed."
echo "GPU and MPI backends are NOT covered here - test/run-tests.sh runs the"
echo "compute-sanitizer and MPI stages; see docs/memory-safety.md section 4."
