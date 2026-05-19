#!/bin/sh
# Resolves one - and always the same - NVIDIA HPC SDK installation and exports
# the environment the LBM mini-app needs from it.
#
# Why this exists: the SDK ships several directory names for the same release
# (e.g. Linux_x86_64/2026 and Linux_x86_64/26.1, one being a symlink of the
# other) and the MPI stack is reached through the comm_libs/mpi symlink. When
# the image build and the runtime pick different names, a binary linked against
# one tree loads plugins from the other: HPC-X OpenMPI 5 then resolves
# `pmix_framework_names` against a foreign (or the Rocky system) PMIx and dies
# with a symbol lookup error before main(). Resolving the tree once, through
# `pwd -P`, and exporting OPAL_PREFIX / PMIX_INSTALL_PREFIX - which a relocated
# OpenMPI needs to find its own components - keeps build and run consistent.
#
# Usage: `. /opt/lbm-miniapp/scripts/nvhpc-env.sh` (POSIX sh, no bashisms).

: "${NVHPC_ROOT:=/opt/nvidia/hpc_sdk}"

lbmini_prepend_path() {
    # $1 = variable name, $2 = directory. Skips missing directories and keeps
    # the SDK ahead of /usr/lib64, so the system PMIx can never win.
    lbmini_new="$2"
    [ -d "${lbmini_new}" ] || return 0
    eval "lbmini_cur=\${$1:-}"
    case ":${lbmini_cur}:" in
        *":${lbmini_new}:"*) return 0 ;;
    esac
    if [ -n "${lbmini_cur}" ]; then
        eval "export $1=\"\${lbmini_new}:\${lbmini_cur}\""
    else
        eval "export $1=\"\${lbmini_new}\""
    fi
}

# --- 1. The compiler / CUDA tree.
LBMINI_NVHPC_DIR=""
if [ -d "${NVHPC_ROOT}/Linux_x86_64" ]; then
    # Newest first, deterministically: `sort -V` orders 2026 above 26.1, and
    # `pwd -P` collapses whichever of the two is a symlink onto the real tree.
    for lbmini_candidate in $(ls -1 "${NVHPC_ROOT}/Linux_x86_64" 2>/dev/null | sort -V -r); do
        lbmini_dir="${NVHPC_ROOT}/Linux_x86_64/${lbmini_candidate}"
        [ -d "${lbmini_dir}/compilers/bin" ] || continue
        LBMINI_NVHPC_DIR="$(cd "${lbmini_dir}" && pwd -P)"
        break
    done
fi

if [ -n "${LBMINI_NVHPC_DIR}" ]; then
    export NVHPC_ROOT
    export LBMINI_NVHPC_DIR

    # --- 2. The MPI stack.
    #
    # `comm_libs/mpi` is NOT usable as a library prefix: it is a symlink to
    # `comm_libs/hpcx`, which holds the NVHPC compiler wrappers in `bin/` and
    # nothing else - no `lib/` at all. Putting `comm_libs/mpi/lib` on
    # LD_LIBRARY_PATH (as the container used to) adds a directory that does not
    # exist, so `libopen-pal.so.80` resolves `libpmix.so.2` against the Rocky
    # system PMIx and aborts with `undefined symbol: pmix_framework_names`.
    # The real installation is `comm_libs/<cuda>/hpcx/hpcx-<ver>/ompi`, and the
    # one to pick is the newest under the CUDA version this SDK defaults to.
    lbmini_cuda_version=""
    if [ -d "${LBMINI_NVHPC_DIR}/cuda/bin" ]; then
        lbmini_cuda_version="$(basename "$(dirname "$(cd "${LBMINI_NVHPC_DIR}/cuda/bin" && pwd -P)")")"
    fi

    # A candidate only counts when it ships both the launcher and the runtime.
    lbmini_find_mpi() {
        for lbmini_mpi_candidate in $(ls -d $1 2>/dev/null | sort -V -r); do
            if [ -x "${lbmini_mpi_candidate}/bin/mpirun" ] &&
               [ -n "$(ls "${lbmini_mpi_candidate}"/lib/libmpi.so* 2>/dev/null | head -n 1)" ]; then
                (cd "${lbmini_mpi_candidate}" && pwd -P)
                return 0
            fi
        done
        return 1
    }

    LBMINI_MPI_DIR=""
    for lbmini_pattern in \
        "${LBMINI_NVHPC_DIR}/comm_libs/${lbmini_cuda_version}/hpcx/hpcx-*/ompi" \
        "${LBMINI_NVHPC_DIR}/comm_libs/*/hpcx/hpcx-*/ompi" \
        "${LBMINI_NVHPC_DIR}/comm_libs/${lbmini_cuda_version}/openmpi4/openmpi-*" \
        "${LBMINI_NVHPC_DIR}/comm_libs/openmpi4/openmpi-*" \
        "${LBMINI_NVHPC_DIR}/comm_libs/mpi"; do
        LBMINI_MPI_DIR="$(lbmini_find_mpi "${lbmini_pattern}")" && break
    done

    if [ -n "${LBMINI_MPI_DIR}" ]; then
        export LBMINI_MPI_DIR
        # A relocated Open MPI reads its plugin directory from OPAL_PREFIX; with
        # the wrong prefix it loads mismatched MCA components.
        export OPAL_PREFIX="${LBMINI_MPI_DIR}"
        lbmini_prepend_path PATH "${LBMINI_MPI_DIR}/bin"
        lbmini_prepend_path LD_LIBRARY_PATH "${LBMINI_MPI_DIR}/lib"

        lbmini_hpcx_dir="$(dirname "${LBMINI_MPI_DIR}")"
        for lbmini_component in pmix ucx ucc hcoll sharp; do
            lbmini_prepend_path LD_LIBRARY_PATH "${lbmini_hpcx_dir}/${lbmini_component}/lib"
        done
        if [ -d "${lbmini_hpcx_dir}/pmix" ]; then
            export PMIX_INSTALL_PREFIX="${lbmini_hpcx_dir}/pmix"
        elif [ -f "${LBMINI_MPI_DIR}/lib/libpmix.so" ] || [ -f "${LBMINI_MPI_DIR}/lib/libpmix.so.2" ]; then
            export PMIX_INSTALL_PREFIX="${LBMINI_MPI_DIR}"
        fi
        if [ -d "${lbmini_hpcx_dir}" ]; then
            export HPCX_DIR="${lbmini_hpcx_dir}"
        fi
    fi

    # --- 3. Compilers, CUDA and math libraries.
    lbmini_prepend_path PATH "${LBMINI_NVHPC_DIR}/cuda/bin"
    lbmini_prepend_path PATH "${LBMINI_NVHPC_DIR}/compilers/bin"
    lbmini_prepend_path LD_LIBRARY_PATH "${LBMINI_NVHPC_DIR}/math_libs/lib64"
    lbmini_prepend_path LD_LIBRARY_PATH "${LBMINI_NVHPC_DIR}/cuda/lib64"
    lbmini_prepend_path LD_LIBRARY_PATH "${LBMINI_NVHPC_DIR}/compilers/lib"
    lbmini_prepend_path CMAKE_PREFIX_PATH "${LBMINI_NVHPC_DIR}/cmake"
fi

# --- 4. The Intel OpenCL CPU runtime (the CPU ICD of the OpenCL backend).
lbmini_prepend_path LD_LIBRARY_PATH "/opt/intel/oneapi/lib"
lbmini_prepend_path LD_LIBRARY_PATH "/opt/intel/oneapi/redist/lib"

unset lbmini_candidate lbmini_dir lbmini_component lbmini_hpcx_dir lbmini_cur lbmini_new
unset lbmini_cuda_version lbmini_pattern lbmini_mpi_candidate
