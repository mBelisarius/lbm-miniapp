#!/bin/bash
set -e

# Resolve the NVHPC / HPC-X environment through the shared script. The image
# build sources the very same script, so the MPI stack found at run time is the
# one the binaries were linked against: picking a different SDK directory here
# (both Linux_x86_64/2026 and Linux_x86_64/26.1 exist) makes HPC-X OpenMPI load
# a foreign PMIx and abort with `undefined symbol: pmix_framework_names`.
LBMINI_HOME="$(cd "$(dirname "$0")" && pwd)"
if [ -r "${LBMINI_HOME}/scripts/nvhpc-env.sh" ]; then
    . "${LBMINI_HOME}/scripts/nvhpc-env.sh"
elif [ -r /opt/lbm-miniapp/scripts/nvhpc-env.sh ]; then
    . /opt/lbm-miniapp/scripts/nvhpc-env.sh
fi

# Set default OpenMP threading policy if not explicitly set
if [ -z "${OMP_NUM_THREADS:-}" ]; then
    export OMP_NUM_THREADS=$(nproc 2>/dev/null || echo 1)
fi

# Command dispatcher
case "$1" in
    lbm-miniapp-main|main)
        shift 1 2>/dev/null || true
        exec lbm-miniapp-main "$@"
        ;;
    lbm-miniapp-test|test)
        shift 1 2>/dev/null || true
        exec /opt/lbm-miniapp/test/run-tests.sh "$@"
        ;;
    unit)
        shift 1 2>/dev/null || true
        exec /opt/lbm-miniapp/test/run-tests.sh unit "$@"
        ;;
    cpu|memory-check)
        shift 1 2>/dev/null || true
        exec /opt/lbm-miniapp/test/run-tests.sh cpu "$@"
        ;;
    gpu)
        shift 1 2>/dev/null || true
        exec /opt/lbm-miniapp/test/run-tests.sh gpu "$@"
        ;;
    mpi)
        shift 1 2>/dev/null || true
        exec /opt/lbm-miniapp/test/run-tests.sh mpi "$@"
        ;;
    campaign)
        shift 1 2>/dev/null || true
        cd /opt/lbm-miniapp/results
        exec python3 campaign.py "$@"
        ;;
    bash|sh)
        exec "$@"
        ;;
    *)
        if [ "$#" -eq 0 ]; then
            exec lbm-miniapp-main
        else
            exec "$@"
        fi
        ;;
esac
