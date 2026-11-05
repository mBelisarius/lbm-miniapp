#!/bin/bash
# Unified test runner for LBM Mini-App
# Usage:
#   test/run-tests.sh unit          # Run GoogleTest unit and functional tests
#   test/run-tests.sh cpu           # Run CPU memory-safety suite (ASan, UBSan, Valgrind, TSan)
#   test/run-tests.sh gpu           # Run GPU compute-sanitizer suite (memcheck, racecheck, initcheck, synccheck)
#   test/run-tests.sh mpi           # Run MPI multi-rank domain decomposition tests
#   test/run-tests.sh all           # Run all test suites in sequence
#   test/run-tests.sh [gtest flags] # Defaults to unit tests with passed arguments

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SOURCE_DIR}"

find_binary() {
    local name="$1"
    if command -v "${name}" >/dev/null 2>&1; then
        echo "${name}"
    elif [[ -x "${SOURCE_DIR}/build/${name}" ]]; then
        echo "${SOURCE_DIR}/build/${name}"
    elif [[ -x "${SOURCE_DIR}/cmake-build-release/${name}" ]]; then
        echo "${SOURCE_DIR}/cmake-build-release/${name}"
    elif [[ -x "${SOURCE_DIR}/cmake-build-debug/${name}" ]]; then
        echo "${SOURCE_DIR}/cmake-build-debug/${name}"
    elif [[ -x "/usr/local/bin/${name}" ]]; then
        echo "/usr/local/bin/${name}"
    else
        echo ""
    fi
}

TEST_BIN="$(find_binary lbm-miniapp-test)"
MAIN_BIN="$(find_binary lbm-miniapp-main)"

# The sanitized tree, when the image built one. It is what covers the two
# backends no other tool reaches: the OpenCL GPU backend (compute-sanitizer
# instruments the CUDA API only) and the MPI backend (Valgrind cannot be
# pointed at a launcher without a suppression file per MCA component).
ASAN_TEST_BIN="${SOURCE_DIR}/build-asan/lbm-miniapp-test"
ASAN_MAIN_BIN="${SOURCE_DIR}/build-asan/lbm-miniapp-main"

# The very ASan/LSan configuration the CPU gate uses; see test/memory-check.sh
# for why `fast_unwind_on_malloc=0` is what makes the vendor suppressions match
# at all.
LBMINI_ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:strict_string_checks=1:detect_stack_use_after_return=1:alloc_dealloc_mismatch=1:new_delete_type_mismatch=1:use_sigaltstack=0:fast_unwind_on_malloc=0:malloc_context_size=15"
LBMINI_LSAN_OPTIONS=""
if [[ -f "${SOURCE_DIR}/test/lsan_suppressions.supp" ]]; then
    LBMINI_LSAN_OPTIONS="suppressions=${SOURCE_DIR}/test/lsan_suppressions.supp"
fi

# The same options plus `protect_shadow_gap=0`, for the one pass that runs a
# sanitized process against the NVIDIA driver (the OpenCL GPU sweep below).
# Without it the driver cannot initialise: it maps its unified address space
# over the region AddressSanitizer keeps as an unmapped, PROT_NONE "shadow
# gap", the mapping fails, and the ICD loader drops the vendor silently - the
# process then enumerates one platform ("Intel(R) OpenCL") instead of two, the
# OpenCL GPU cases skip themselves for want of a CL_DEVICE_TYPE_GPU device, and
# the backend goes unchecked while everything looks green. Unprotecting the gap
# costs nothing that matters here: the gap is a region no allocation is ever
# placed in, so redzones, use-after-free and leak detection are untouched.
LBMINI_ASAN_OPTIONS_DEVICE="${LBMINI_ASAN_OPTIONS}:protect_shadow_gap=0"

run_unit() {
    echo "================================================================================"
    echo "==> [1/4] Running Unit & Functional Test Suite (GoogleTest)"
    echo "================================================================================"
    if [[ -z "${TEST_BIN}" ]]; then
        echo "Error: lbm-miniapp-test binary not found. Please build the project first." >&2
        return 1
    fi
    "${TEST_BIN}" "$@"
}

run_cpu() {
    echo "================================================================================"
    echo "==> [2/4] Running CPU Memory-Safety & Multi-Threading Suite"
    echo "================================================================================"
    if [[ -x "${SOURCE_DIR}/test/memory-check.sh" ]]; then
        "${SOURCE_DIR}/test/memory-check.sh" --tsan "$@"
    else
        echo "Error: ${SOURCE_DIR}/test/memory-check.sh not found." >&2
        return 1
    fi
}

run_gpu() {
    echo "================================================================================"
    echo "==> [3/4] Running GPU Diagnostic & Sanitization Suite"
    echo "================================================================================"
    
    if ! command -v compute-sanitizer >/dev/null 2>&1; then
        echo "Warning: compute-sanitizer not found in PATH."
        echo "GPU verification requires NVIDIA HPC SDK / CUDA Toolkit with compute-sanitizer."
        return 0
    fi

    # Probing the device is retried, and a failure is classified rather than
    # waved through. The machine this is developed on is the awkward case: a
    # laptop with an Intel iGPU driving the display and an NVIDIA dGPU in
    # fine-grained Runtime D3 (`DynamicPowerManagement: 3`), i.e. powered down
    # whenever nothing holds it open. The first NVML call after a cold period
    # has to resume the device across PCIe, which can take longer than the
    # single immediate `nvidia-smi` this used to do - and while the device is
    # mid-resume, or while a driver upgrade has left the running kernel modules
    # older than the user-space libraries, NVML answers
    # "Unable to determine the device handle for GPU0: Unknown Error".
    local NVIDIA_SMI_LOG="/tmp/lbm_miniapp_nvidia_smi.log"
    local gpu_visible=0
    local attempt
    for attempt in 1 2 3; do
        if command -v nvidia-smi >/dev/null 2>&1 \
                && nvidia-smi -L > "${NVIDIA_SMI_LOG}" 2>&1 \
                && grep -q "^GPU [0-9]" "${NVIDIA_SMI_LOG}"; then
            gpu_visible=1
            break
        fi
        if [[ "${attempt}" -lt 3 ]]; then
            sleep 5
        fi
    done

    if [[ "${gpu_visible}" -eq 0 ]]; then
        # No driver stack at all - a container started without `--gpus all` /
        # `--nv`, or a machine with no NVIDIA hardware - is a legitimate skip:
        # the host backends are what such a run is for. A driver stack that is
        # present but hands out no device is not: something is wrong with the
        # host, and reporting "all test suites passed" for a pipeline that
        # never touched a GPU is exactly the silent loss of coverage this suite
        # exists to prevent.
        local driver_present=0
        if [[ -e /dev/nvidiactl || -e /dev/nvidia0 ]] \
                || ldconfig -p 2>/dev/null | grep -q "libcuda\.so\.1"; then
            driver_present=1
        fi

        if [[ "${driver_present}" -eq 0 || "${LBMINI_ALLOW_NO_GPU:-0}" == "1" ]]; then
            echo "Warning: No accessible NVIDIA GPU detected (nvidia-smi failed or unavailable)."
            echo "To run GPU tests in Docker, ensure NVIDIA GPU driver and --gpus all are passed."
            return 0
        fi

        echo "Error: the NVIDIA driver stack is present in this container, but no device" >&2
        echo "    answered after 3 attempts over ~10 s:" >&2
        sed 's/^/        /' "${NVIDIA_SMI_LOG}" >&2
        rm -f "${NVIDIA_SMI_LOG}"
        echo "    This is a host condition, not a solver defect, and it is reported as a" >&2
        echo "    failure because the GPU backends would otherwise go unchecked while the" >&2
        echo "    pipeline reported success. On the host:" >&2
        echo "      * nvidia-smi                     - does it fail there too?" >&2
        echo "      * cat /proc/driver/nvidia/version vs. modinfo -F version nvidia" >&2
        echo "        - a driver upgrade with the old modules still loaded is the most" >&2
        echo "          common cause; reload the modules or reboot." >&2
        echo "      * laptops with hybrid graphics: the dGPU may be runtime-suspended" >&2
        echo "        (cat /proc/driver/nvidia/params | grep DynamicPowerManagement)." >&2
        echo "        nvidia-persistenced keeps it resident; see docs/container.md 7.6." >&2
        echo "      * docker context ls - Docker Desktop's VM exposes no GPU; the" >&2
        echo "        native daemon (context 'default') does." >&2
        echo "    Set LBMINI_ALLOW_NO_GPU=1 to downgrade this to a skip on purpose." >&2
        return 1
    fi

    echo "==> Detected NVIDIA GPU device(s):"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    rm -f "${NVIDIA_SMI_LOG}"
    echo

    # A clean run is one whose *report* is clean, not merely one that exited 0.
    # compute-sanitizer forwards the application's exit status unless it is told
    # otherwise, so a kernel that wrote out of bounds printed
    # "ERROR SUMMARY: 12 errors" and this stage still reported success. Both
    # gates are applied now: `--error-exitcode` below, and this summary audit,
    # because a leak-only finding does not always reach the exit-code path.
    sanitizer_log_is_clean() {
        local log_file="$1"
        if grep -Eq "ERROR SUMMARY: [1-9][0-9]* error" "${log_file}"; then
            return 1
        fi
        if grep -Eq "LEAK SUMMARY: [1-9][0-9]* bytes leaked" "${log_file}"; then
            return 1
        fi
        if grep -Eq "RACECHECK SUMMARY: .*\([1-9][0-9]* error" "${log_file}"; then
            return 1
        fi
        # The tool losing track of the application is not a clean run either:
        # "Internal Sanitizer Error: Detected a failure to track a kernel
        # launch. Future records might be incomplete" means the rest of the
        # report covers an unknown subset of the kernels.
        if grep -q "Internal Sanitizer Error" "${log_file}"; then
            return 1
        fi
        return 0
    }

    # Helper function to execute compute-sanitizer and handle WSL2/WDDM host limitations
    local wddm_restricted=false
    run_sanitizer_step() {
        local tool="$1"
        shift
        if [[ "${wddm_restricted}" == "true" ]]; then
            return 0
        fi
        local log_file
        log_file="$(mktemp /tmp/compute_sanitizer_${tool}_XXXXXX.log)"
        
        echo "--> Running compute-sanitizer (${tool})..."
        # Bounded, because these tools can stop making progress: `racecheck`
        # deadlocked here for over 40 minutes on kernels launched concurrently
        # from several host threads, with no output and no exit. A hang is a
        # failure of the check, and an unbounded one turns a 40-minute pipeline
        # into an overnight one; every step in this stage otherwise finishes in
        # seconds on real hardware.
        local step_status=0
        timeout --signal=KILL "${LBMINI_SANITIZER_TIMEOUT:-900}" \
            compute-sanitizer --tool "${tool}" --error-exitcode 1 "$@" \
            > "${log_file}" 2>&1 || step_status=$?
        if [[ "${step_status}" -eq 124 || "${step_status}" -eq 137 ]]; then
            cat "${log_file}"
            echo "Error: compute-sanitizer (${tool}) did not finish within" >&2
            echo "    ${LBMINI_SANITIZER_TIMEOUT:-900}s and was killed. Raise" >&2
            echo "    LBMINI_SANITIZER_TIMEOUT if the machine is genuinely that slow." >&2
            rm -f "${log_file}"
            return 1
        fi
        if [[ "${step_status}" -eq 0 ]] && sanitizer_log_is_clean "${log_file}"; then
            cat "${log_file}"
            rm -f "${log_file}"
        else
            cat "${log_file}"
            if grep -Eqi "WDDM debugger interface|Device not supported" "${log_file}"; then
                wddm_restricted=true
                echo
                echo "==> Notice: compute-sanitizer encountered WDDM debugger interface restrictions under Windows/WSL2."
                echo "    To enable compute-sanitizer under Windows host: Run 'EnableDebuggerInterface.bat' as Administrator."
                echo "    On Linux HPC clusters / Bare-Metal nodes (TCC mode), compute-sanitizer operates without restrictions."
                echo
            elif grep -Eqi "Target application terminated before first instrumented API call" "${log_file}"; then
                echo "==> Notice: compute-sanitizer skipped (no CUDA device kernels or API calls were invoked by this backend configuration)."
            else
                echo "Error: compute-sanitizer (${tool}) reported findings (see the report above)." >&2
                rm -f "${log_file}"
                return 1
            fi
            rm -f "${log_file}"
        fi
    }

    # 1. The backend-agnostic memory-safety sweep, instantiated for the device
    #    backends. This is the only place where the GPU solvers get the same
    #    13-case sweep the host backends get from test/memory-check.sh -
    #    degenerate meshes, odd extents, repeated construction/destruction,
    #    concurrent instances, getter copy-back - and the only place where a
    #    device-side out-of-bounds write is caught by a tool instead of by a
    #    benchmark number that looks plausible.
    if [[ -z "${TEST_BIN}" ]]; then
        echo "Error: lbm-miniapp-test binary not found. Please build the project first." >&2
        return 1
    fi

    # The listing is taken once, with its exit status checked: a binary that
    # cannot even start (the CUDA-enabled test binary needs `libcuda.so.1`,
    # which the container only has when it was started with `--gpus all` /
    # `--nv`) must say that, not be reported as "built without device
    # backends".
    local test_listing
    local listing_status=0
    test_listing="$("${TEST_BIN}" --gtest_list_tests 2>&1)" || listing_status=$?
    if [[ "${listing_status}" -ne 0 ]]; then
        echo "Error: ${TEST_BIN} could not list its tests:" >&2
        printf '%s\n' "${test_listing}" | sed 's/^/    /' >&2
        echo "    A GPU was detected above, so the container is missing the driver" >&2
        echo "    libraries: start it with --gpus all (Docker) or --nv (Apptainer)." >&2
        return 1
    fi

    local device_tests
    device_tests="$(printf '%s\n' "${test_listing}" | grep -Ei "_gpu\." || true)"
    if [[ -z "${device_tests}" ]]; then
        echo "Error: the test binary contains no device-backend cases, so no GPU backend is" >&2
        echo "    memory-checked here and this stage would only ever report success." >&2
        echo "    Rebuild with -DENABLE_TEST_DEVICE_BACKENDS=ON." >&2
        return 1
    fi

    echo "--> Running the device-backend memory-safety sweep..."
    local DEVICE_LOG="/tmp/lbm_miniapp_device_suite.log"
    if ! timeout --signal=KILL "${LBMINI_SANITIZER_TIMEOUT:-900}" \
            "${TEST_BIN}" --gtest_filter="*_gpu.*" > "${DEVICE_LOG}" 2>&1; then
        cat "${DEVICE_LOG}"
        echo "Error: the device-backend memory-safety sweep failed." >&2
        rm -f "${DEVICE_LOG}"
        return 1
    fi
    # A skip is green, which is precisely how a device backend stops being
    # checked without anyone noticing. A GPU is present at this point (verified
    # above), so every compiled-in device backend must actually run.
    if grep -q "SKIPPED" "${DEVICE_LOG}"; then
        cat "${DEVICE_LOG}"
        echo "Error: a device backend skipped its memory-safety sweep instead of running it" >&2
        echo "    on the GPU detected above. A skipped backend is an unchecked backend." >&2
        rm -f "${DEVICE_LOG}"
        return 1
    fi
    grep -E "^\[(=|-)*\]|\[  PASSED  \]" "${DEVICE_LOG}" | tail -n 4
    rm -f "${DEVICE_LOG}"

    # The same sweep under compute-sanitizer. `memcheck` and `initcheck` are
    # the memory tools (out-of-bounds and leaked device allocations,
    # uninitialised device reads); `racecheck` and `synccheck` matter here too
    # because the sweep is the only test that runs several solvers from several
    # host threads. The OpenCL GPU backend is excluded: it reaches the device
    # through the NVIDIA ICD, never through CUDA, so every tool would abort
    # with "Target application terminated before first instrumented API call";
    # it is covered by the AddressSanitizer pass below instead.
    local CUDA_DEVICE_FILTER="*_gpu.*-*opencl_gpu*"
    # `racecheck` alone drops MultiThreadedConcurrentExecutionIsSafe, the one
    # case that drives four solvers from four host threads at once. The tool
    # cannot follow kernels launched concurrently from several host threads: on
    # this hardware the Kokkos instantiation intermittently ends with
    # "Internal Sanitizer Error: Detected a failure to track a kernel launch"
    # counted as one hazard, and the RAJA one deadlocks outright - a false
    # positive and a hang, both of them properties of the tool rather than of
    # the solver. Nothing is lost: racecheck looks for races *inside* a kernel,
    # the kernels are the very ones the other twelve cases launch, and the
    # host-side concurrency this case exists for is covered by ThreadSanitizer
    # in test/memory-check.sh. The case still runs uninstrumented above, and
    # under memcheck, initcheck and synccheck below.
    local RACECHECK_FILTER="${CUDA_DEVICE_FILTER}:*MultiThreadedConcurrentExecutionIsSafe*"
    if printf '%s\n' "${test_listing}" | grep -Eqi "(cuda|kokkos|raja|openacc|openmp|alpaka)_gpu\."; then
        run_sanitizer_step memcheck --leak-check full "${TEST_BIN}" --gtest_filter="${CUDA_DEVICE_FILTER}"
        run_sanitizer_step initcheck "${TEST_BIN}" --gtest_filter="${CUDA_DEVICE_FILTER}"
        run_sanitizer_step racecheck "${TEST_BIN}" --gtest_filter="${RACECHECK_FILTER}"
        run_sanitizer_step synccheck "${TEST_BIN}" --gtest_filter="${CUDA_DEVICE_FILTER}"
    fi

    # The OpenCL GPU backend, under AddressSanitizer + LeakSanitizer. Without
    # this it is the one backend with no memory tool at all: compute-sanitizer
    # cannot see it, and the CPU gate hides the GPUs by design. ASan covers
    # exactly what breaks in this backend - host-side buffer sizing, the SoA /
    # AoS transpose in the getters, and the release of contexts, queues and
    # command buffers - while the vendor one-time allocations stay covered by
    # test/lsan_suppressions.supp.
    if [[ -x "${ASAN_TEST_BIN}" ]] && "${ASAN_TEST_BIN}" --gtest_list_tests 2>/dev/null | grep -q "opencl_gpu\."; then
        echo "--> Running the OpenCL GPU sweep under AddressSanitizer + LeakSanitizer..."
        local OPENCL_LOG="/tmp/lbm_miniapp_opencl_gpu_asan.log"
        if ASAN_OPTIONS="${LBMINI_ASAN_OPTIONS_DEVICE}" LSAN_OPTIONS="${LBMINI_LSAN_OPTIONS}" \
                timeout --signal=KILL "${LBMINI_SANITIZER_TIMEOUT:-900}" \
                "${ASAN_TEST_BIN}" --gtest_filter="*opencl_gpu.*" > "${OPENCL_LOG}" 2>&1 \
                && ! grep -q "SKIPPED" "${OPENCL_LOG}"; then
            grep -E "\[  PASSED  \]|\[  FAILED  \]" "${OPENCL_LOG}" | tail -n 2
            rm -f "${OPENCL_LOG}"
        else
            cat "${OPENCL_LOG}"
            echo "Error: the OpenCL GPU sweep failed or was skipped under AddressSanitizer." >&2
            if grep -q "SKIPPED" "${OPENCL_LOG}"; then
                echo "    A GPU was detected above and the uninstrumented sweep used it, so" >&2
                echo "    'not available' here means the sanitized process saw no OpenCL GPU" >&2
                echo "    device. That is what protect_shadow_gap=0 exists for (see the" >&2
                echo "    LBMINI_ASAN_OPTIONS_DEVICE comment at the top of this script);" >&2
                echo "    check that the NVIDIA ICD is still listed in /etc/OpenCL/vendors." >&2
            fi
            rm -f "${OPENCL_LOG}"
            return 1
        fi
    else
        echo "Note: no sanitized build with OpenCL GPU cases (build-asan); the OpenCL GPU"
        echo "      backend is executed above but not leak-checked in this run."
    fi

    # 2. Direct GPU execution and compute-sanitizer suite via main solver across available GPU backends
    if [[ -n "${MAIN_BIN}" ]]; then
        local GPU_CONFIG="/tmp/lbm_miniapp_gpu_check.yaml"
        local GPU_OUT="/tmp/lbm_miniapp_gpu_out"
        local GPU_ERR="/tmp/lbm_miniapp_gpu_err.log"
        local GPU_STDOUT="/tmp/lbm_miniapp_gpu_out.log"
        for gpu_backend in CUDA Kokkos RAJA Alpaka OpenACC OpenMP OpenCL; do
            cat << EOF > "${GPU_CONFIG}"
Fluid:
    densityL:     1.0
    pressureL:    1.0
    densityR:     0.125
    pressureR:    0.1
    viscosity:    1.0e-2
    prandtl:      0.71
    gamma:        1.4

Mesh:
    lx: 1.0
    ly: 1.0
    lz: 1.0
    nx: 64
    ny: 4
    nz: 1

Control:
    tmax: 0.2
    Ux: 0.25
    Uy: 0.0
    Uz: 0.0
    printStep: 10

Performance:
    target: GPU
    backend: ${gpu_backend}
    cores: 0
EOF
            echo "--> Testing GPU solver execution with backend ${gpu_backend}..."
            rm -rf "${GPU_OUT}"
            # The solver's stderr decides whether this is a backend that was
            # compiled out (a skip) or a backend that is present and broke (a
            # failure). Discarding stderr, as this loop used to, reported every
            # real GPU fault as "not enabled ... skipping".
            if "${MAIN_BIN}" --config "${GPU_CONFIG}" --outpath "${GPU_OUT}" >"${GPU_STDOUT}" 2>"${GPU_ERR}"; then
                # Proof that the solver actually stepped. `tmax` above used to
                # be 0.005, shorter than the derived lattice time step, so
                # `kSteps` came out 0: every backend wrote its step-0 snapshot,
                # printed "Simulation completed" and exited without launching a
                # single kernel - and all four compute-sanitizer tools then
                # reported a clean run on a solver that had not executed.
                # lbmini::Executor now refuses such a configuration; this is
                # the second line of defence, and it is cheap.
                if ! grep -q "Time step:" "${GPU_STDOUT}"; then
                    echo "Error: backend ${gpu_backend} exited successfully without advancing a" >&2
                    echo "    single time step, so nothing was executed or checked on the device:" >&2
                    sed 's/^/        /' "${GPU_STDOUT}" >&2
                    rm -f "${GPU_ERR}" "${GPU_STDOUT}"
                    rm -rf "${GPU_OUT}" "${GPU_CONFIG}"
                    return 1
                fi
                echo "    Advanced $(grep -c 'Time step:' "${GPU_STDOUT}") output chunk(s) on the device."
                rm -rf "${GPU_OUT}"
                # compute-sanitizer instruments the CUDA driver/runtime API only.
                # The OpenCL backend reaches the same GPU through the NVIDIA ICD,
                # never through CUDA, so every tool would abort with "Target
                # application terminated before first instrumented API call":
                # running them proves nothing and only emits spurious errors.
                if [[ "${gpu_backend}" == "OpenCL" ]]; then
                    echo "    Executed on the GPU. compute-sanitizer not applicable to OpenCL (it instruments the CUDA API only)."
                else
                    run_sanitizer_step memcheck --leak-check full "${MAIN_BIN}" --config "${GPU_CONFIG}" --outpath "${GPU_OUT}"
                    rm -rf "${GPU_OUT}"
                    run_sanitizer_step racecheck "${MAIN_BIN}" --config "${GPU_CONFIG}" --outpath "${GPU_OUT}"
                    rm -rf "${GPU_OUT}"
                    run_sanitizer_step initcheck "${MAIN_BIN}" --config "${GPU_CONFIG}" --outpath "${GPU_OUT}"
                    rm -rf "${GPU_OUT}"
                    run_sanitizer_step synccheck "${MAIN_BIN}" --config "${GPU_CONFIG}" --outpath "${GPU_OUT}"
                    rm -rf "${GPU_OUT}"
                fi
            elif grep -Eqi "not enabled|not available|is unavailable|not supported|only support|no .*(device|platform)s? (are )?available|without a device backend" "${GPU_ERR}"; then
                echo "    Backend ${gpu_backend} GPU target not available in this build or on this hardware, skipping:"
                sed 's/^/        /' "${GPU_ERR}"
            else
                echo "Error: backend ${gpu_backend} failed on the GPU target:" >&2
                sed 's/^/    /' "${GPU_ERR}" >&2
                rm -f "${GPU_ERR}" "${GPU_STDOUT}"
                rm -rf "${GPU_OUT}" "${GPU_CONFIG}"
                return 1
            fi
            rm -f "${GPU_ERR}" "${GPU_STDOUT}"
            rm -rf "${GPU_OUT}" "${GPU_CONFIG}"
        done
    fi

    echo "==> GPU sanitization checks completed successfully."
}

run_mpi() {
    echo "================================================================================"
    echo "==> [4/4] Running MPI Multi-Rank Domain Decomposition Suite"
    echo "================================================================================"
    
    if ! command -v mpirun >/dev/null 2>&1; then
        echo "Warning: mpirun not found. Skipping MPI verification."
        return 0
    fi

    if [[ -z "${MAIN_BIN}" ]]; then
        echo "Error: lbm-miniapp-main binary not found. Please build the project first." >&2
        return 1
    fi

    # The MPI backend is CPU-only: hiding the GPUs keeps a stray device context
    # out of the ranks and makes that guarantee observable instead of assumed.
    export CUDA_VISIBLE_DEVICES=""

    # The base image exports CUDA-IPC tuning variables that this CPU-only,
    # single-node stage never consumes, and UCX prints two warning lines per
    # rank about each of them. The runs set no UCX variable of their own, so
    # there is nothing left for this warning to protect.
    export UCX_WARN_UNUSED_ENV_VARS=n

    local MPI_LOG="/tmp/lbm_miniapp_mpi_run.log"

    for ranks in 1 2 3 4; do
        echo "--> Testing MPI simulation with ${ranks} rank(s)..."
        local MPI_CONFIG="/tmp/lbm_miniapp_mpi_${ranks}.yaml"
        local MPI_OUT="/tmp/lbm_miniapp_mpi_out_${ranks}"
        cat << 'EOF' > "${MPI_CONFIG}"
Fluid:
    densityL:     1.0
    pressureL:    1.0
    densityR:     0.125
    pressureR:    0.1
    viscosity:    1.0e-2
    prandtl:      0.71
    gamma:        1.4

Mesh:
    lx: 1.0
    ly: 1.0
    lz: 1.0
    nx: 64
    ny: 4
    nz: 1

Control:
    tmax: 0.2
    Ux: 0.25
    Uy: 0.0
    Uz: 0.0
    printStep: 10

Performance:
    target: CPU
    backend: MPI
    cores: 0
EOF

        rm -rf "${MPI_OUT}"
        if mpirun --allow-run-as-root --bind-to none --oversubscribe --mca coll_hcoll_enable 0 -np "${ranks}" \
                "${MAIN_BIN}" --config "${MPI_CONFIG}" --outpath "${MPI_OUT}" 2>&1 | tee "${MPI_LOG}"; then
            # Proof that the ranks actually stepped, i.e. that the halo
            # exchange ran. `tmax` above used to be 0.005, shorter than the
            # derived lattice time step, so every rank wrote its step-0
            # snapshot and exited without exchanging a single halo.
            if ! grep -q "Time step:" "${MPI_LOG}"; then
                echo "Error: the ${ranks}-rank run completed without advancing a single time" >&2
                echo "    step, so no halo exchange was exercised." >&2
                rm -f "${MPI_LOG}"
                rm -rf "${MPI_OUT}" "${MPI_CONFIG}"
                return 1
            fi
        else
            # A mismatched MPI installation (OpenMPI resolving a foreign PMIx,
            # a stale OPAL_PREFIX) dies in the dynamic loader before main(), so
            # the failure has nothing to do with the solver: say so, and print
            # what the ranks were actually launched against.
            if grep -Eqi "MPI backend not enabled|MPI backend only supports" "${MPI_LOG}"; then
                echo "    The binary was built without the MPI backend (-DENABLE_MPI=ON), skipping."
                rm -f "${MPI_LOG}"
                rm -rf "${MPI_OUT}" "${MPI_CONFIG}"
                return 0
            fi
            if grep -Eqi "symbol lookup error|undefined symbol|error while loading shared libraries" "${MPI_LOG}"; then
                echo "Error: the MPI runtime is not functional in this environment." >&2
                echo "    mpirun:          $(command -v mpirun)" >&2
                echo "    OPAL_PREFIX:     ${OPAL_PREFIX:-<unset>}" >&2
                echo "    LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}" >&2
                echo "    Source scripts/nvhpc-env.sh so that OpenMPI, PMIx and UCX all come" >&2
                echo "    from the same HPC-X installation (see docs/container.md section 7)." >&2
            fi
            rm -f "${MPI_LOG}"
            rm -rf "${MPI_OUT}" "${MPI_CONFIG}"
            return 1
        fi
        rm -f "${MPI_LOG}"
        rm -rf "${MPI_OUT}" "${MPI_CONFIG}"
    done

    # The sweep above only proves the ranks agree; nothing in it looks at
    # memory, so the MPI backend used to be the one backend covered by no tool
    # whatsoever - its halo exchange, its pack/unpack buffers and its teardown
    # path are exactly where a domain-decomposition bug writes out of bounds.
    # Valgrind cannot be pointed at a launcher without a suppression file per
    # MCA component, so the instrumented binary is used instead: two ranks,
    # under AddressSanitizer and LeakSanitizer, with the vendor MPI
    # allocations covered by test/lsan_suppressions.supp.
    if [[ -x "${ASAN_MAIN_BIN}" ]]; then
        echo "--> Re-running 2 ranks under AddressSanitizer + LeakSanitizer..."
        local ASAN_CONFIG="/tmp/lbm_miniapp_mpi_asan.yaml"
        local ASAN_OUT="/tmp/lbm_miniapp_mpi_asan_out"
        cat << 'EOF' > "${ASAN_CONFIG}"
Fluid:
    densityL:     1.0
    pressureL:    1.0
    densityR:     0.125
    pressureR:    0.1
    viscosity:    1.0e-2
    prandtl:      0.71
    gamma:        1.4

Mesh:
    lx: 1.0
    ly: 1.0
    lz: 1.0
    nx: 64
    ny: 4
    nz: 1

Control:
    tmax: 0.2
    Ux: 0.25
    Uy: 0.0
    Uz: 0.0
    printStep: 10

Performance:
    target: CPU
    backend: MPI
    cores: 0
EOF
        rm -rf "${ASAN_OUT}"
        if ASAN_OPTIONS="${LBMINI_ASAN_OPTIONS}" LSAN_OPTIONS="${LBMINI_LSAN_OPTIONS}" \
                mpirun --allow-run-as-root --bind-to none --oversubscribe --mca coll_hcoll_enable 0 -np 2 \
                "${ASAN_MAIN_BIN}" --config "${ASAN_CONFIG}" --outpath "${ASAN_OUT}" 2>&1 | tee "${MPI_LOG}"; then
            if ! grep -q "Time step:" "${MPI_LOG}"; then
                echo "Error: the sanitized ranks completed without advancing a single time step." >&2
                rm -f "${MPI_LOG}" "${ASAN_CONFIG}"
                rm -rf "${ASAN_OUT}"
                return 1
            fi
            echo "    2 sanitized ranks completed with no AddressSanitizer/LeakSanitizer report."
        else
            echo "Error: the sanitized MPI run reported a memory defect (see the report above)." >&2
            rm -f "${MPI_LOG}" "${ASAN_CONFIG}"
            rm -rf "${ASAN_OUT}"
            return 1
        fi
        rm -f "${MPI_LOG}" "${ASAN_CONFIG}"
        rm -rf "${ASAN_OUT}"
    else
        echo "Note: no sanitized build (build-asan/lbm-miniapp-main); the MPI ranks ran"
        echo "      without a memory tool in this environment."
    fi

    echo "==> MPI multi-rank checks completed successfully."
}

run_all() {
    echo "================================================================================"
    echo "==> Starting Full End-to-End Verification Pipeline (Unit, CPU, GPU, MPI)"
    echo "================================================================================"
    run_unit
    echo
    run_cpu
    echo
    run_gpu
    echo
    run_mpi
    echo
    echo "================================================================================"
    echo "==> All Test Suites (Unit, CPU, GPU, MPI) Passed Successfully!"
    echo "================================================================================"
}

TARGET="${1:-unit}"

case "${TARGET}" in
    unit)
        shift 1 2>/dev/null || true
        run_unit "$@"
        ;;
    cpu)
        shift 1 2>/dev/null || true
        run_cpu "$@"
        ;;
    gpu)
        shift 1 2>/dev/null || true
        run_gpu "$@"
        ;;
    mpi)
        shift 1 2>/dev/null || true
        run_mpi "$@"
        ;;
    all)
        shift 1 2>/dev/null || true
        run_all
        ;;
    -h|--help|help)
        echo "LBM Mini-App Test Runner"
        echo
        echo "Usage:"
        echo "  test/run-tests.sh unit [gtest_args...]   Run GoogleTest unit & functional tests"
        echo "  test/run-tests.sh cpu [options...]       Run CPU memory-safety suite (ASan, UBSan, Valgrind, TSan)"
        echo "  test/run-tests.sh gpu                    Run GPU compute-sanitizer suite"
        echo "  test/run-tests.sh mpi                    Run MPI multi-rank sweep"
        echo "  test/run-tests.sh all                    Run all test suites in sequence"
        ;;
    *)
        # If unknown target or gtest flag passed (e.g. --gtest_filter=...), default to unit test runner
        run_unit "$@"
        ;;
esac
