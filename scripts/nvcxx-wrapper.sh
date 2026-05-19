#!/usr/bin/env bash
# Wrapper around nvc++ that silently drops flags unsupported by NVHPC
# but injected by CLion's compiler detection mechanism.

REAL_COMPILER="/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvc++"

args=()
for arg in "$@"; do
    case "$arg" in
        -xc++|-fpch-preprocess)
            # Drop GCC/Clang-only flags that nvc++ does not understand
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

exec "$REAL_COMPILER" "${args[@]}"
