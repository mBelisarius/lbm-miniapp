#!/bin/bash
set -e

# Default configuration and arguments
CONFIG="DEBUG"
CMAKE_ARGS=("$@")

# Use the first argument as the configuration, default to DEBUG
if [[ "$#" -gt 0 ]]; then
    FIRST_ARG_UPPER=$(echo "$1" | tr '[:lower:]' '[:upper:]')
    case "$FIRST_ARG_UPPER" in
        RELEASE|RELWITHDEBINFO|MINSIZEREL|DEBUG)
            CONFIG=$1
            shift
            CMAKE_ARGS=("$@")
            ;;
    esac
fi

# Define build directory
CONFIG_LOWER=$(echo "$CONFIG" | tr '[:upper:]' '[:lower:]')
CONFIG_UPPER=$(echo "$CONFIG" | tr '[:lower:]' '[:upper:]')
BUILD_DIR="cmake-build-${CONFIG_LOWER}"

echo "Configuration: ${CONFIG_UPPER}"
echo "Build directory: ${BUILD_DIR}"
if [ ${#CMAKE_ARGS[@]} -ne 0 ]; then
    echo "Additional CMake flags: ${CMAKE_ARGS[*]}"
fi

# Configure and build the CMake project
cmake -B "${BUILD_DIR}" -S . -DCMAKE_BUILD_TYPE="${CONFIG_UPPER}" "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --parallel

echo "Build finished successfully."
