# =========================================================
# LBM mini-app

cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0060)
    cmake_policy(SET CMP0060 NEW)
endif()
set(CMAKE_POLICY_DEFAULT_CMP0060 NEW)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

set(ProjectName "lbm-miniapp")
set(ProjectNamespace "lbmini")

option(ENABLE_CUDA "Enable CUDA for the project" ON)
option(ENABLE_OPENCL "Enable the OpenCL backend" ON)
option(ENABLE_MPI "Enable the MPI backend" ON)
option(ENABLE_KOKKOS "Enable the Kokkos backend" ON)
option(ENABLE_KOKKOS_CUDA "Build Kokkos with its CUDA device backend" OFF)
option(ENABLE_RAJA "Enable the RAJA backend" ON)
option(ENABLE_RAJA_CUDA "Build RAJA with its CUDA device backend" OFF)

# RAJA and its BLT dependencies also use broad ENABLE_* cache names. Preserve
# the application choice before configuring dependencies so they cannot turn a
# backend back on during CMake's automatic regeneration.
set(LBMINI_CUDA_REQUESTED "${ENABLE_CUDA}")
set(LBMINI_MPI_REQUESTED "${ENABLE_MPI}")
set(LBMINI_OPENCL_REQUESTED "${ENABLE_OPENCL}")
set(LBMINI_RAJA_REQUESTED "${ENABLE_RAJA}")

# --- Memory-safety verification (see docs/memory-safety.md)
#
# LBMINI_SANITIZER builds the whole project (and its fetched dependencies, so
# the instrumentation is consistent) with the requested sanitizer. Combine
# several with `+`, e.g. -DLBMINI_SANITIZER=address+undefined. Sanitized builds
# are for the test suite, never for benchmark numbers.
set(LBMINI_SANITIZER "" CACHE STRING
        "Sanitizer for the memory-safety builds: address, undefined, thread, leak, or a '+'-separated combination")
option(ENABLE_TEST_OPENMP_BACKEND "Include the OpenMP CPU backend in the test binary" ON)
option(ENABLE_TEST_DEVICE_BACKENDS "Include the GPU-targeted Kokkos/RAJA solvers in the test binary" OFF)

# Builds the project's own code (not the fetched dependencies) with -ffast-math
# in the Release and RelWithDebInfo configurations. The point is SIMD exp/log:
# glibc only declares the libmvec vector variants (_ZGVdN4v_exp, ...) when
# __FAST_MATH__ is defined, so without it every `exp`/`log` in the collision
# kernel stays a scalar libm call and blocks vectorization of the cell loop.
# Results differ from a strict-IEEE build in the last bits (see docs).
option(LBMINI_FAST_MATH "Compile the LBM code with -ffast-math (SIMD exp/log through libmvec)" ON)

set(CMAKE_CXX_STANDARD 20)
project("${ProjectName}" LANGUAGES CXX)

if (ENABLE_CUDA)
    if (NOT CMAKE_CUDA_HOST_COMPILER)
        if (CMAKE_CXX_COMPILER)
            set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "CUDA host compiler" FORCE)
        elseif (DEFINED ENV{CXX})
            set(CMAKE_CUDA_HOST_COMPILER "$ENV{CXX}" CACHE FILEPATH "CUDA host compiler" FORCE)
        endif ()
    endif ()
    if (CMAKE_CUDA_HOST_COMPILER)
        set(BLT_CUDA_HOST_COMPILER "${CMAKE_CUDA_HOST_COMPILER}" CACHE FILEPATH "CUDA host compiler for BLT" FORCE)
    endif ()
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures")
    endif ()
    if (NOT DEFINED CUDA_ARCH)
        set(CUDA_ARCH "sm_80" CACHE STRING "CUDA architecture for BLT" FORCE)
    endif ()
    if (NOT DEFINED CUB_DIR AND NOT DEFINED CUB_INCLUDE_DIRS)
        find_package(CUDAToolkit QUIET)
        file(GLOB _nvhpc_cub_hints
            "/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/targets/x86_64-linux/include"
            "/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/targets/x86_64-linux/include/cccl"
            "/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/include"
            "/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/include/cccl"
            "/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/include"
            "/opt/nvidia/hpc_sdk/Linux_x86_64/*/math_libs/*/include"
            "$ENV{NVHPC_ROOT}/Linux_x86_64/*/cuda/*/targets/x86_64-linux/include"
            "$ENV{NVHPC_ROOT}/Linux_x86_64/*/cuda/*/targets/x86_64-linux/include/cccl"
            "$ENV{NVHPC_ROOT}/Linux_x86_64/*/cuda/*/include"
        )
        find_path(_cub_found_dir
            NAMES cub/cub.cuh
            HINTS
                ${CUDAToolkit_INCLUDE_DIRS}
                ${_nvhpc_cub_hints}
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                $ENV{CUDA_HOME}
                $ENV{CUDA_PATH}
                /usr/local/cuda
            PATH_SUFFIXES
                include
                include/cccl
                targets/x86_64-linux/include
                targets/x86_64-linux/include/cccl
                cccl
                cub
        )
        if (_cub_found_dir)
            set(CUB_DIR "${_cub_found_dir}" CACHE PATH "Path to CUB" FORCE)
            set(CUB_INCLUDE_DIRS "${_cub_found_dir}" CACHE PATH "Path to CUB include directory" FORCE)
            message(STATUS "Found CUB at: ${_cub_found_dir}")
        endif ()
    endif ()
    include(CheckLanguage)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
        set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
        add_compile_options(
            $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
        message(STATUS "CUDA language enabled with compiler: ${CMAKE_CUDA_COMPILER}")
    else ()
        set(ENABLE_CUDA OFF CACHE BOOL "Enable CUDA for the project" FORCE)
        set(LBMINI_CUDA_REQUESTED OFF)
        message(WARNING "CUDA compiler not found. Disabling CUDA for this build.")
    endif ()
endif ()

if(MINGW AND LBMINI_RAJA_REQUESTED)
    message(WARNING "RAJA is not compatible with MinGW; disabling the RAJA backend for this build.")
    set(LBMINI_RAJA_REQUESTED OFF)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
    message(STATUS "Configuring for NVIDIA HPC SDK")
    set(ARCH_FLAG "$<$<COMPILE_LANGUAGE:CXX>:-tp=native>")
    set(DEBUG_OPT_FLAG "-g")
    file(GLOB NVHPC_CMAKE_DIRS "/opt/nvidia/hpc_sdk/Linux_x86_64/*/cmake")
    if(NVHPC_CMAKE_DIRS)
        list(APPEND CMAKE_PREFIX_PATH ${NVHPC_CMAKE_DIRS})
    endif()
    if(DEFINED ENV{NVHPC_ROOT})
        file(GLOB NVHPC_ENV_CMAKE_DIRS "$ENV{NVHPC_ROOT}/Linux_x86_64/*/cmake" "$ENV{NVHPC_ROOT}/cmake")
        if(NVHPC_ENV_CMAKE_DIRS)
            list(APPEND CMAKE_PREFIX_PATH ${NVHPC_ENV_CMAKE_DIRS})
        endif()
    endif()
    find_package(NVHPC REQUIRED)
    if(ENABLE_CUDA)
        set(NVHPC_GPU_ARCH "ccnative" CACHE STRING "NVIDIA GPU architecture to target, e.g. ccnative")
    endif()
else ()
    set(ARCH_FLAG "$<$<COMPILE_LANGUAGE:CXX>:-march=native>")
    set(DEBUG_OPT_FLAG "-Og")
endif()

add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-error>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-O3>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-DNDEBUG>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:${ARCH_FLAG}>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELWITHDEBINFO>>:-O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELWITHDEBINFO>>:-DNDEBUG>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELWITHDEBINFO>>:-g>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELWITHDEBINFO>>:${ARCH_FLAG}>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:MINSIZEREL>>:-Oz>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:MINSIZEREL>>:-DNDEBUG>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:MINSIZEREL>>:${ARCH_FLAG}>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:${DEBUG_OPT_FLAG}>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-g>
)

# =========================================================
# Sanitizers
#
# Applied globally (before any target or FetchContent dependency is declared) so
# every translation unit that ends up in the test binary carries the same
# instrumentation — mixing sanitized and non-sanitized objects produces false
# negatives at best and link errors at worst. `-fno-sanitize-recover=all` makes
# the first violation abort, which is what turns a silent out-of-bounds access
# into a failing test. CUDA objects are left untouched: device code is verified
# with `compute-sanitizer` instead (see docs/memory-safety.md).

if (LBMINI_SANITIZER)
    string(REPLACE "+" "," LBMINI_SANITIZER_LIST "${LBMINI_SANITIZER}")
    if (CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
        message(WARNING
                "LBMINI_SANITIZER=${LBMINI_SANITIZER} requested with nvc++, whose sanitizer support is partial. "
                "Run the memory-safety builds with GCC or Clang.")
    endif ()
    message(STATUS "Sanitizers enabled: ${LBMINI_SANITIZER_LIST}")
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${LBMINI_SANITIZER_LIST}>
            $<$<COMPILE_LANGUAGE:CXX>:-fno-sanitize-recover=all>
            $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
            $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
    add_link_options(
            $<$<LINK_LANGUAGE:CXX>:-fsanitize=${LBMINI_SANITIZER_LIST}>
            $<$<LINK_LANGUAGE:C>:-fsanitize=${LBMINI_SANITIZER_LIST}>
    )
endif ()

# =========================================================
# gtest

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(GTEST_FORCE_SHARED_CRT ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

set_property(TARGET gmock PROPERTY FOLDER "gtest")
set_property(TARGET gmock_main PROPERTY FOLDER "gtest")
set_property(TARGET gtest PROPERTY FOLDER "gtest")
set_property(TARGET gtest_main PROPERTY FOLDER "gtest")

# =========================================================
# Eigen

include(FetchContent)
FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5.0.1
)

set(EIGEN_FORCE_SHARED_CRT ON CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen)

# =========================================================
# OpenMP

if (NOT NVHPC_FOUND)
    find_package(OpenMP REQUIRED)
endif ()

# =========================================================
# MPI

if (ENABLE_MPI)
    if (CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
        find_package(MPI QUIET)
    endif ()

    if (NOT MPI_CXX_FOUND)
        find_path(MPI_CXX_HEADER_DIR mpi.h
            PATHS
                $ENV{NVHPC_ROOT}/Linux_x86_64/26.1/comm_libs/mpi/include
                $ENV{NVHPC_ROOT}/Linux_x86_64/*/comm_libs/*/hpcx/*/ompi/include
                $ENV{NVHPC_ROOT}/Linux_x86_64/*/comm_libs/mpi/include
                /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi/include
                /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/mpi/include
                /usr/include
                /usr/include/openmpi-x86_64
                /usr/include/mpich-x86_64
        )
        find_library(MPI_CXX_LIB mpi
            PATHS
                $ENV{NVHPC_ROOT}/Linux_x86_64/26.1/comm_libs/mpi/lib
                $ENV{NVHPC_ROOT}/Linux_x86_64/*/comm_libs/*/hpcx/*/ompi/lib
                $ENV{NVHPC_ROOT}/Linux_x86_64/*/comm_libs/mpi/lib
                /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi/lib
                /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/mpi/lib
                /usr/lib64
                /usr/lib64/openmpi/lib
                /usr/lib64/mpich/lib
        )
        if (MPI_CXX_HEADER_DIR AND MPI_CXX_LIB)
            set(MPI_CXX_FOUND TRUE)
            set(MPI_CXX_INCLUDE_PATH "${MPI_CXX_HEADER_DIR}")
            set(MPI_CXX_LIBRARIES "${MPI_CXX_LIB}")
            if (NOT TARGET MPI::MPI_CXX)
                add_library(MPI::MPI_CXX UNKNOWN IMPORTED)
                set_target_properties(MPI::MPI_CXX PROPERTIES
                    IMPORTED_LOCATION "${MPI_CXX_LIB}"
                    INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_HEADER_DIR}"
                )
            endif ()
        else ()
            set(MPI_CXX_FOUND FALSE)
        endif ()
    endif ()
else ()
    set(MPI_CXX_FOUND FALSE)
endif ()

# =========================================================
# yaml-cpp

include(FetchContent)
FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.9.0
)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
    foreach(_yaml_target yaml-cpp yaml-cpp-sandbox yaml-cpp-parse yaml-cpp-read)
        if(TARGET ${_yaml_target})
            target_compile_options(${_yaml_target} PRIVATE --diag_suppress=code_is_unreachable)
        endif()
    endforeach()
endif()

# =========================================================
# GPU architecture

# CMAKE_CUDA_ARCHITECTURES is the single place the compute capability is chosen;
# Kokkos (Kokkos_ARCH_<NAME>) and RAJA/BLT (CUDA_ARCH=sm_<nn>) each want it in
# their own spelling. Deriving both from it keeps the three device backends on
# the same architecture: a mismatch is not a build error, it only shows up at
# run time as `Kokkos::Cuda::initialize WARNING: running kernels compiled for
# compute capability 8.0 on device with compute capability 8.9`, plus the
# silent performance loss of JIT-recompiled PTX.
set(LBMINI_CUDA_ARCH "")
if (DEFINED CMAKE_CUDA_ARCHITECTURES)
    foreach (_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        # Strip the -real / -virtual suffixes and skip the symbolic values
        # (native, all, all-major), which have no Kokkos equivalent.
        string(REGEX REPLACE "-(real|virtual)$" "" _arch "${_arch}")
        if (_arch MATCHES "^[0-9]+$")
            set(LBMINI_CUDA_ARCH "${_arch}")
            break()
        endif ()
    endforeach ()
endif ()

set(LBMINI_KOKKOS_ARCH_NAME "")
if (LBMINI_CUDA_ARCH)
    set(_kokkos_arch_map
            "70=VOLTA70" "72=VOLTA72" "75=TURING75"
            "80=AMPERE80" "86=AMPERE86" "89=ADA89"
            "90=HOPPER90" "100=BLACKWELL100" "120=BLACKWELL120"
    )
    foreach (_entry IN LISTS _kokkos_arch_map)
        string(REPLACE "=" ";" _pair "${_entry}")
        list(GET _pair 0 _capability)
        list(GET _pair 1 _name)
        if (_capability STREQUAL LBMINI_CUDA_ARCH)
            set(LBMINI_KOKKOS_ARCH_NAME "${_name}")
            break()
        endif ()
    endforeach ()
    if (NOT LBMINI_KOKKOS_ARCH_NAME)
        message(WARNING
                "CMAKE_CUDA_ARCHITECTURES=${LBMINI_CUDA_ARCH} has no known Kokkos_ARCH_* name; "
                "set Kokkos_ARCH_<NAME>=ON explicitly to avoid a compute capability mismatch.")
    endif ()
endif ()

# =========================================================
# Kokkos

# The Kokkos backend is single-source: the very same solver is instantiated on
# Kokkos::DefaultHostExecutionSpace (CPU target) and on
# Kokkos::DefaultExecutionSpace (GPU target). Which spaces exist is decided
# here, at Kokkos configuration time. The CUDA device backend is opt-in
# (-DENABLE_KOKKOS_CUDA=ON) because it requires the host compiler to double as
# a CUDA compiler (nvcc / nvcc_wrapper / nvc++).
if (ENABLE_KOKKOS)
    include(FetchContent)
    FetchContent_Declare(
            kokkos
            GIT_REPOSITORY https://github.com/kokkos/kokkos.git
            GIT_TAG 5.2.1
    )

    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
    if (TARGET OpenMP::OpenMP_CXX OR CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
        set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
    endif ()
    if (ENABLE_KOKKOS_CUDA)
        set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "" FORCE)
        get_cmake_property(_all_variables VARIABLES)
        set(_kokkos_arch_defined FALSE)
        if(DEFINED Kokkos_ARCH OR DEFINED KOKKOS_ARCH)
            set(_kokkos_arch_defined TRUE)
        else()
            foreach(_var ${_all_variables})
                if(_var MATCHES "^Kokkos_ARCH_" AND ${_var})
                    set(_kokkos_arch_defined TRUE)
                    break()
                endif()
            endforeach()
        endif()
        if(NOT _kokkos_arch_defined)
            if (LBMINI_KOKKOS_ARCH_NAME)
                set(Kokkos_ARCH_${LBMINI_KOKKOS_ARCH_NAME} ON CACHE BOOL "Kokkos GPU Architecture" FORCE)
                message(STATUS "Kokkos GPU architecture: Kokkos_ARCH_${LBMINI_KOKKOS_ARCH_NAME} "
                        "(from CMAKE_CUDA_ARCHITECTURES=${LBMINI_CUDA_ARCH})")
            else()
                set(Kokkos_ARCH_AMPERE80 ON CACHE BOOL "Kokkos GPU Architecture" FORCE)
            endif()
        endif()
    endif ()

    FetchContent_MakeAvailable(kokkos)

    foreach(_kokkos_target kokkoscore kokkoscontainers kokkosalgorithms kokkossimd)
        if(TARGET ${_kokkos_target})
            set_property(TARGET ${_kokkos_target} PROPERTY FOLDER "kokkos")
        endif()
    endforeach()
endif ()

# =========================================================
# RAJA

# The RAJA backend is single-source too: the very same solver is instantiated
# with the host backend tag (OpenMP / sequential) and with the device backend
# tag (CUDA / HIP). Which policies exist is decided here, at RAJA configuration
# time. The CUDA device backend is opt-in (-DENABLE_RAJA_CUDA=ON) because RAJA
# device lambdas require the whole translation unit to be compiled by nvcc.
if (LBMINI_RAJA_REQUESTED)
    include(FetchContent)
    FetchContent_Declare(
            RAJA
            GIT_REPOSITORY https://github.com/LLNL/RAJA.git
            GIT_TAG v2026.07.0
    )

    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    # BLT (RAJA's build system) propagates BLT_CXX_STD to CMAKE_CXX_STANDARD;
    # pin it so RAJA does not downgrade the project to C++14.
    set(BLT_CXX_STD "c++20" CACHE STRING "" FORCE)
    # BLT would otherwise register its own bundled googletest and collide with
    # the one fetched above.
    set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(ENABLE_DOCS OFF CACHE BOOL "" FORCE)
    set(ENABLE_MPI OFF CACHE BOOL "" FORCE)
    set(ENABLE_FIND_MPI OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_MPI OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_EXERCISES OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_REPRODUCERS OFF CACHE BOOL "" FORCE)
    # BLT probes for a dozen optional developer tools (Sphinx, AStyle,
    # ClangFormat, Uncrustify, Yapf, CMakeFormat, Cppcheck, ClangQuery,
    # ClangTidy, ClangApplyReplacements) and prints a "Failed to locate ..."
    # line for each one it cannot find. None of them affects this build, and
    # the ten lines per configure - three trees per memory-safety run - are
    # exactly the kind of noise that trains a reader to skim past real output.
    foreach(_blt_tool ASTYLE CLANGFORMAT UNCRUSTIFY YAPF CMAKEFORMAT
            CPPCHECK CLANGQUERY CLANGTIDY CLANGAPPLYREPLACEMENTS
            SPHINX DOXYGEN)
        set(ENABLE_${_blt_tool} OFF CACHE BOOL "" FORCE)
    endforeach()
    set(RAJA_ENABLE_DESUBLIMATED_TENSORS OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_VECTORIZATION OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_VECTOR_STATS OFF CACHE BOOL "" FORCE)
    if (TARGET OpenMP::OpenMP_CXX OR CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
        set(ENABLE_OPENMP ON CACHE BOOL "" FORCE)
        set(RAJA_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
    endif ()
    if (ENABLE_RAJA_CUDA)
        set(ENABLE_CUDA ON CACHE BOOL "" FORCE)
        set(RAJA_ENABLE_CUDA ON CACHE BOOL "" FORCE)
        if (NOT CMAKE_CUDA_HOST_COMPILER)
            if (CMAKE_CXX_COMPILER)
                set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "CUDA host compiler" FORCE)
            elseif (DEFINED ENV{CXX})
                set(CMAKE_CUDA_HOST_COMPILER "$ENV{CXX}" CACHE FILEPATH "CUDA host compiler" FORCE)
            endif ()
        endif ()
        if (CMAKE_CUDA_HOST_COMPILER)
            set(BLT_CUDA_HOST_COMPILER "${CMAKE_CUDA_HOST_COMPILER}" CACHE FILEPATH "CUDA host compiler for BLT" FORCE)
        endif ()
        if (NOT DEFINED CUDA_ARCH)
            if (LBMINI_CUDA_ARCH)
                set(CUDA_ARCH "sm_${LBMINI_CUDA_ARCH}" CACHE STRING "CUDA architecture for BLT" FORCE)
            else ()
                set(CUDA_ARCH "sm_80" CACHE STRING "CUDA architecture for BLT" FORCE)
            endif ()
        endif ()
        if (NOT DEFINED CUB_DIR AND NOT DEFINED CUB_INCLUDE_DIRS)
            find_package(CUDAToolkit QUIET)
            file(GLOB _nvhpc_cub_hints
                "/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/targets/x86_64-linux/include"
                "/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/targets/x86_64-linux/include/cccl"
                "/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/include"
                "/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/include/cccl"
                "/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/include"
                "/opt/nvidia/hpc_sdk/Linux_x86_64/*/math_libs/*/include"
                "$ENV{NVHPC_ROOT}/Linux_x86_64/*/cuda/*/targets/x86_64-linux/include"
                "$ENV{NVHPC_ROOT}/Linux_x86_64/*/cuda/*/targets/x86_64-linux/include/cccl"
                "$ENV{NVHPC_ROOT}/Linux_x86_64/*/cuda/*/include"
            )
            find_path(_cub_found_dir
                NAMES cub/cub.cuh
                HINTS
                    ${CUDAToolkit_INCLUDE_DIRS}
                    ${_nvhpc_cub_hints}
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                    $ENV{CUDA_HOME}
                    $ENV{CUDA_PATH}
                    /usr/local/cuda
                PATH_SUFFIXES
                    include
                    include/cccl
                    targets/x86_64-linux/include
                    targets/x86_64-linux/include/cccl
                    cccl
                    cub
            )
            if (_cub_found_dir)
                set(CUB_DIR "${_cub_found_dir}" CACHE PATH "Path to CUB" FORCE)
                set(CUB_INCLUDE_DIRS "${_cub_found_dir}" CACHE PATH "Path to CUB include directory" FORCE)
            endif ()
        endif ()
    else ()
        set(ENABLE_CUDA OFF CACHE BOOL "" FORCE)
        set(RAJA_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
    endif ()

    # BLT writes CMAKE_*_OUTPUT_DIRECTORY into the cache and would relocate
    # *every* project binary into RAJA's build tree. Claiming those cache
    # entries first (without FORCE, so an explicit user setting still wins)
    # keeps the mini-app executables where the rest of the tooling expects them.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" CACHE PATH "")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" CACHE PATH "")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}" CACHE PATH "")

    FetchContent_MakeAvailable(RAJA)

    # Restore the parent project's ENABLE_CUDA, ENABLE_MPI, and ENABLE_OPENCL settings
    set(ENABLE_CUDA "${LBMINI_CUDA_REQUESTED}" CACHE BOOL "Enable CUDA for the project" FORCE)
    set(ENABLE_MPI "${LBMINI_MPI_REQUESTED}" CACHE BOOL "Enable the MPI backend" FORCE)
    set(ENABLE_OPENCL "${LBMINI_OPENCL_REQUESTED}" CACHE BOOL "Enable the OpenCL backend" FORCE)

    foreach(_raja_target RAJA camp blt_cuda blt_cuda_runtime)
        if(TARGET ${_raja_target})
            set_property(TARGET ${_raja_target} PROPERTY FOLDER "raja")
        endif()
    endforeach()
endif ()

# =========================================================
# Targets

# --- Common

add_library("${ProjectName}-common" STATIC
        lbmini/Data.hpp
        lbmini/Data/ControlData.hpp
        lbmini/Data/DataEnums.hpp
        lbmini/Data/DataReader.hpp
        lbmini/Data/FluidData.hpp
        lbmini/Data/MeshData.hpp
        lbmini/Lbm/CountCores.hpp
        lbmini/Lbm/ILattice.hpp
        lbmini/Lbm/ILbmTube.hpp
        lbmini/AsyncWriter.hpp
        lbmini/Executor.hpp
        lbmini/Executor.cpp
        lbmini/Utils.hpp
)

target_include_directories("${ProjectName}-common" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")

# Every other project target links `-common` PUBLIC, so attaching the fast-math
# flags here reaches all of them (and the test binary) while the fetched
# dependencies keep strict IEEE semantics. nvc++ and nvcc are left alone: they
# have their own vector math and do not take the GCC spelling of the flag.
# Fast-math folds std::isfinite() to true; the project uses lbmini::IsFinite()
# (Utils.hpp) for every NaN/Inf check for that reason.
if (LBMINI_FAST_MATH)
    if (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
        set(_lbmini_fast_math_flags -ffast-math)
        # Clang needs to be told which vector math library to emit calls to.
        if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
                AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag(-fveclib=libmvec LBMINI_HAS_FVECLIB_LIBMVEC)
            if (LBMINI_HAS_FVECLIB_LIBMVEC)
                list(APPEND _lbmini_fast_math_flags -fveclib=libmvec)
            endif ()
        endif ()
        foreach (_flag IN LISTS _lbmini_fast_math_flags)
            target_compile_options("${ProjectName}-common" PUBLIC
                    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release,RelWithDebInfo>>:${_flag}>")
        endforeach ()
        message(STATUS "Fast math enabled for Release/RelWithDebInfo: ${_lbmini_fast_math_flags}")
    else ()
        message(STATUS "LBMINI_FAST_MATH has no effect with ${CMAKE_CXX_COMPILER_ID}")
    endif ()
endif ()

set_target_properties("${ProjectName}-common" PROPERTIES
        LINKER_LANGUAGE CXX
)

if(ENABLE_MPI AND MPI_CXX_FOUND AND TARGET MPI::MPI_CXX)
    target_compile_definitions("${ProjectName}-common" PUBLIC USE_MPI)
    target_link_libraries("${ProjectName}-common" PUBLIC MPI::MPI_CXX)
endif()

target_link_libraries("${ProjectName}-common"
        PUBLIC Eigen3::Eigen
        PUBLIC yaml-cpp::yaml-cpp
)

# Executor.cpp instantiates every backend, so the Kokkos headers and the
# USE_KOKKOS switch have to be visible to the common target as well.
if(ENABLE_KOKKOS AND TARGET Kokkos::kokkos)
    target_compile_definitions("${ProjectName}-common" PUBLIC USE_KOKKOS)
    target_link_libraries("${ProjectName}-common" PUBLIC Kokkos::kokkos)
endif()

# Same for RAJA: the factory in Executor.cpp needs the policies and the
# USE_RAJA switch.
if(LBMINI_RAJA_REQUESTED AND TARGET RAJA)
    target_compile_definitions("${ProjectName}-common" PUBLIC USE_RAJA)
    target_link_libraries("${ProjectName}-common" PUBLIC RAJA)
endif()


# --- Lbm Plain

add_library("${ProjectName}-lbm-plain" STATIC
        lbmini/Lbm/Plain/LatticeD2Q9.hpp
        lbmini/Lbm/Plain/LbmTube.hpp
)
target_include_directories("${ProjectName}-lbm-plain" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")
set_target_properties("${ProjectName}-lbm-plain" PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions("${ProjectName}-lbm-plain" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})
target_link_libraries("${ProjectName}-lbm-plain" PUBLIC Eigen3::Eigen PUBLIC "${ProjectName}-common")

# --- Lbm OpenMP

add_library("${ProjectName}-lbm-openmp" STATIC
        lbmini/Lbm/OpenMp/Cpu/LatticeD2Q9.hpp
        lbmini/Lbm/OpenMp/Cpu/LbmTube.hpp
        lbmini/Lbm/OpenMp/Gpu/LatticeD2Q9.hpp
        lbmini/Lbm/OpenMp/Gpu/LbmTube.hpp
)

target_include_directories("${ProjectName}-lbm-openmp" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")
set_target_properties("${ProjectName}-lbm-openmp" PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions("${ProjectName}-lbm-openmp" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})
target_link_libraries("${ProjectName}-lbm-openmp" PUBLIC Eigen3::Eigen PUBLIC "${ProjectName}-common")

if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
    target_compile_options("${ProjectName}-lbm-openmp" PUBLIC
            $<$<COMPILE_LANGUAGE:CXX>:-mp>
            $<$<COMPILE_LANGUAGE:CXX>:-acc>
            $<$<COMPILE_LANGUAGE:CXX>:-Minfo=accel>
    )
    target_link_options("${ProjectName}-lbm-openmp" INTERFACE 
            $<$<LINK_LANGUAGE:CXX>:-mp> 
            $<$<LINK_LANGUAGE:CXX>:-acc>
    )
    if(ENABLE_CUDA)
        target_compile_options("${ProjectName}-lbm-openmp" PUBLIC
                $<$<COMPILE_LANGUAGE:CXX>:-mp=gpu>
        )
        target_link_options("${ProjectName}-lbm-openmp" INTERFACE 
                $<$<LINK_LANGUAGE:CXX>:-mp=gpu>
        )
    endif()
endif()

if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries("${ProjectName}-lbm-openmp" PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- Lbm OpenACC CPU

add_library("${ProjectName}-lbm-openacc-cpu" STATIC
        lbmini/Lbm/OpenAcc/Cpu/LatticeD2Q9.hpp
        lbmini/Lbm/OpenAcc/Cpu/LbmTube.hpp
)

target_include_directories("${ProjectName}-lbm-openacc-cpu" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")
set_target_properties("${ProjectName}-lbm-openacc-cpu" PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions("${ProjectName}-lbm-openacc-cpu" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})
target_link_libraries("${ProjectName}-lbm-openacc-cpu" PUBLIC Eigen3::Eigen PUBLIC "${ProjectName}-common")

# --- Lbm OpenACC GPU

add_library("${ProjectName}-lbm-openacc-gpu" STATIC
        lbmini/Lbm/OpenAcc/Gpu/LatticeD2Q9.hpp
        lbmini/Lbm/OpenAcc/Gpu/LbmTube.hpp
)

target_include_directories("${ProjectName}-lbm-openacc-gpu" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")
set_target_properties("${ProjectName}-lbm-openacc-gpu" PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions("${ProjectName}-lbm-openacc-gpu" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})
target_link_libraries("${ProjectName}-lbm-openacc-gpu" PUBLIC Eigen3::Eigen PUBLIC "${ProjectName}-common")

if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
    target_compile_options("${ProjectName}-lbm-openacc-cpu" PUBLIC
            "-acc=multicore"
            "-Minfo=accel,vect"
    )
    target_link_options("${ProjectName}-lbm-openacc-cpu" INTERFACE
            "-acc=multicore"
    )
    target_compile_options("${ProjectName}-lbm-openacc-gpu" PUBLIC
            "-acc=gpu"
    )
    target_link_options("${ProjectName}-lbm-openacc-gpu" INTERFACE
            "-acc=gpu"
    )
endif()

if(ENABLE_OPENCL)
    find_package(OpenCL QUIET)
else()
    set(OpenCL_FOUND FALSE)
endif()

if(ENABLE_OPENCL AND OpenCL_FOUND)
    target_compile_definitions("${ProjectName}-common" PUBLIC USE_OPENCL)
    if(TARGET OpenCL::OpenCL)
        target_link_libraries("${ProjectName}-common" PUBLIC OpenCL::OpenCL)
    endif()
    if(DEFINED OpenCL_INCLUDE_DIRS)
        target_include_directories("${ProjectName}-common" PUBLIC ${OpenCL_INCLUDE_DIRS})
    endif()
    if(DEFINED OpenCL_LIBRARIES)
        foreach(_lib ${OpenCL_LIBRARIES})
            if(EXISTS "${_lib}")
                get_filename_component(_opencl_lib_dir "${_lib}" DIRECTORY)
                if(_opencl_lib_dir AND NOT "${_opencl_lib_dir}" MATCHES "^(/usr)?/lib(64)?$")
                    list(PREPEND CMAKE_BUILD_RPATH "${_opencl_lib_dir}")
                    list(PREPEND CMAKE_INSTALL_RPATH "${_opencl_lib_dir}")
                endif()
                break()
            endif()
        endforeach()
    endif()
else()
    message(STATUS "OpenCL not found; OpenCL backends will not be built.")
endif()

# --- Lbm Cuda

if(ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    target_compile_definitions("${ProjectName}-common" PUBLIC USE_CUDA)
    if(TARGET CUDA::cudart_static)
        target_link_libraries("${ProjectName}-common" PUBLIC CUDA::cudart_static)
    elseif(TARGET CUDA::cudart)
        target_link_libraries("${ProjectName}-common" PUBLIC CUDA::cudart)
    endif()
    if(DEFINED CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
        target_include_directories("${ProjectName}-common" PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
    if(DEFINED CUDAToolkit_INCLUDE_DIRS)
        target_include_directories("${ProjectName}-common" PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
        set_source_files_properties(lbmini/Lbm/Cuda/Gpu/LbmTube.cu PROPERTIES LANGUAGE CXX)
    else()
        set_source_files_properties(lbmini/Lbm/Cuda/Gpu/LbmTube.cu PROPERTIES LANGUAGE CUDA)
    endif()
    add_library("${ProjectName}-lbm-cuda" STATIC
            lbmini/Lbm/Cuda/Gpu/LatticeD2Q9.hpp
            lbmini/Lbm/Cuda/Gpu/LbmTube.hpp
            lbmini/Lbm/Cuda/Gpu/LbmTube.cu
    )

    target_include_directories("${ProjectName}-lbm-cuda" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")
    set_target_properties("${ProjectName}-lbm-cuda" PROPERTIES
            LINKER_LANGUAGE CXX
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_RUNTIME_LIBRARY Static
    )

    target_compile_definitions("${ProjectName}-lbm-cuda" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})
    target_link_libraries("${ProjectName}-lbm-cuda" PUBLIC Eigen3::Eigen PUBLIC "${ProjectName}-common")
    if(TARGET CUDA::cudart_static)
        target_link_libraries("${ProjectName}-lbm-cuda" PUBLIC CUDA::cudart_static)
    elseif(TARGET CUDA::cudart)
        target_link_libraries("${ProjectName}-lbm-cuda" PUBLIC CUDA::cudart)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
        target_compile_options("${ProjectName}-lbm-cuda" PUBLIC
                $<$<COMPILE_LANGUAGE:CXX>:-cuda>
                $<$<COMPILE_LANGUAGE:CXX>:-gpu=${NVHPC_GPU_ARCH}>
        )
        target_link_options("${ProjectName}-lbm-cuda" INTERFACE
                $<$<LINK_LANGUAGE:CXX>:-cuda>
        )
    endif()
endif()


# --- Lbm OpenCL CPU

if(ENABLE_OPENCL AND OpenCL_FOUND)
    add_library("${ProjectName}-lbm-opencl-cpu" STATIC
            lbmini/Lbm/OpenCl/Cpu/LatticeD2Q9.hpp
            lbmini/Lbm/OpenCl/Cpu/LbmTube.hpp
    )

    target_include_directories("${ProjectName}-lbm-opencl-cpu" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")
    set_target_properties("${ProjectName}-lbm-opencl-cpu" PROPERTIES LINKER_LANGUAGE CXX)
    target_compile_definitions("${ProjectName}-lbm-opencl-cpu" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})
    target_link_libraries("${ProjectName}-lbm-opencl-cpu" PUBLIC Eigen3::Eigen PUBLIC "${ProjectName}-common" PUBLIC OpenCL::OpenCL)
endif()

# --- Lbm OpenCL GPU

if(ENABLE_OPENCL AND OpenCL_FOUND)
    add_library("${ProjectName}-lbm-opencl-gpu" STATIC
            lbmini/Lbm/OpenCl/Gpu/LatticeD2Q9.hpp
            lbmini/Lbm/OpenCl/Gpu/LbmTube.hpp
    )

    target_include_directories("${ProjectName}-lbm-opencl-gpu" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")
    set_target_properties("${ProjectName}-lbm-opencl-gpu" PROPERTIES LINKER_LANGUAGE CXX)
    target_compile_definitions("${ProjectName}-lbm-opencl-gpu" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})
    target_link_libraries("${ProjectName}-lbm-opencl-gpu" PUBLIC Eigen3::Eigen PUBLIC "${ProjectName}-common" PUBLIC OpenCL::OpenCL)
endif()

# --- Lbm MPI CPU

if(ENABLE_MPI AND MPI_CXX_FOUND)
    add_library("${ProjectName}-lbm-mpi" STATIC
            lbmini/Lbm/Mpi/Cpu/LatticeD2Q9.hpp
            lbmini/Lbm/Mpi/Cpu/LbmTube.hpp
    )

    target_include_directories("${ProjectName}-lbm-mpi" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")
    set_target_properties("${ProjectName}-lbm-mpi" PROPERTIES LINKER_LANGUAGE CXX)
    target_compile_definitions("${ProjectName}-lbm-mpi" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})
    target_link_libraries("${ProjectName}-lbm-mpi" PUBLIC Eigen3::Eigen PUBLIC "${ProjectName}-common" PUBLIC MPI::MPI_CXX)
endif()

# --- Lbm Kokkos

if(ENABLE_KOKKOS AND TARGET Kokkos::kokkos)
    add_library("${ProjectName}-lbm-kokkos" STATIC
            lbmini/Lbm/Kokkos/LatticeD2Q9.hpp
            lbmini/Lbm/Kokkos/LbmTube.hpp
            lbmini/Lbm/Kokkos/Runtime.hpp
            lbmini/Lbm/Kokkos/Cpu/LatticeD2Q9.hpp
            lbmini/Lbm/Kokkos/Cpu/LbmTube.hpp
            lbmini/Lbm/Kokkos/Gpu/LatticeD2Q9.hpp
            lbmini/Lbm/Kokkos/Gpu/LbmTube.hpp
    )

    target_include_directories("${ProjectName}-lbm-kokkos" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")
    set_target_properties("${ProjectName}-lbm-kokkos" PROPERTIES LINKER_LANGUAGE CXX)
    target_compile_definitions("${ProjectName}-lbm-kokkos" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})
    target_link_libraries("${ProjectName}-lbm-kokkos"
            PUBLIC Eigen3::Eigen
            PUBLIC "${ProjectName}-common"
            PUBLIC Kokkos::kokkos
    )
endif()

# --- Lbm RAJA

if(LBMINI_RAJA_REQUESTED AND TARGET RAJA)
    add_library("${ProjectName}-lbm-raja" STATIC
            lbmini/Lbm/Raja/Backends.hpp
            lbmini/Lbm/Raja/LatticeD2Q9.hpp
            lbmini/Lbm/Raja/LbmTube.hpp
            lbmini/Lbm/Raja/Cpu/LatticeD2Q9.hpp
            lbmini/Lbm/Raja/Cpu/LbmTube.hpp
            lbmini/Lbm/Raja/Gpu/LatticeD2Q9.hpp
            lbmini/Lbm/Raja/Gpu/LbmTube.hpp
    )

    target_include_directories("${ProjectName}-lbm-raja" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")
    set_target_properties("${ProjectName}-lbm-raja" PROPERTIES LINKER_LANGUAGE CXX)
    target_compile_definitions("${ProjectName}-lbm-raja" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})
    target_link_libraries("${ProjectName}-lbm-raja"
            PUBLIC Eigen3::Eigen
            PUBLIC "${ProjectName}-common"
            PUBLIC RAJA
    )
endif()

# --- Main

add_executable("${ProjectName}-main" lbmini/main.cpp)
set_target_properties("${ProjectName}-main" PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RUNTIME_LIBRARY Static
)
if(CMAKE_BUILD_RPATH)
    set_target_properties("${ProjectName}-main" PROPERTIES BUILD_RPATH "${CMAKE_BUILD_RPATH}")
endif()
target_link_libraries("${ProjectName}-main"
        PUBLIC Eigen3::Eigen
        PUBLIC "${ProjectName}-common"
        PUBLIC "${ProjectName}-lbm-plain"
        PUBLIC "${ProjectName}-lbm-openmp"
        PUBLIC "${ProjectName}-lbm-openacc-cpu"
        PUBLIC "${ProjectName}-lbm-openacc-gpu"
)

if(TARGET "${ProjectName}-lbm-cuda")
    target_link_libraries("${ProjectName}-main" PUBLIC "${ProjectName}-lbm-cuda")
endif()

if(ENABLE_MPI AND MPI_CXX_FOUND AND TARGET "${ProjectName}-lbm-mpi")
    target_link_libraries("${ProjectName}-main" PUBLIC MPI::MPI_CXX PUBLIC "${ProjectName}-lbm-mpi")
endif()

if(ENABLE_OPENCL AND OpenCL_FOUND)
    target_link_libraries("${ProjectName}-main"
            PUBLIC "${ProjectName}-lbm-opencl-cpu"
            PUBLIC "${ProjectName}-lbm-opencl-gpu"
    )
endif()

if(TARGET "${ProjectName}-lbm-kokkos")
    target_link_libraries("${ProjectName}-main" PUBLIC "${ProjectName}-lbm-kokkos")
endif()

if(TARGET "${ProjectName}-lbm-raja")
    target_link_libraries("${ProjectName}-main" PUBLIC "${ProjectName}-lbm-raja")
endif()

# --- GTest

enable_testing()

set(TestSources
        test/sanity_test.cpp
        test/is_finite_test.cpp
        test/backend_presence_test.cpp
        # Memory-safety suite: mandatory (and meant to run under sanitizers)
        # before any benchmarking campaign. See docs/memory-safety.md.
        test/MemorySafetyHarness.hpp
        test/memory_safety_test.cpp
        test/streaming_table_test.cpp
)

add_executable("${ProjectName}-test" ${TestSources})
set_target_properties("${ProjectName}-test" PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RUNTIME_LIBRARY Static
)
if(CMAKE_BUILD_RPATH)
    set_target_properties("${ProjectName}-test" PROPERTIES BUILD_RPATH "${CMAKE_BUILD_RPATH}")
endif()
target_include_directories("${ProjectName}-test" PRIVATE "${CMAKE_SOURCE_DIR}/test")
target_link_libraries("${ProjectName}-test"
        PUBLIC Eigen3::Eigen
        PUBLIC gtest_main
        PUBLIC "${ProjectName}-common"
        PUBLIC "${ProjectName}-lbm-plain"
)

if(ENABLE_MPI AND MPI_CXX_FOUND AND TARGET "${ProjectName}-lbm-mpi")
    target_compile_definitions("${ProjectName}-test" PRIVATE LBMINI_TEST_MPI)
    target_link_libraries("${ProjectName}-test" PUBLIC MPI::MPI_CXX)
endif()

# The memory-safety suite is backend-agnostic: each backend it can instantiate
# in-process is switched on by appropriate definitions.
if(ENABLE_TEST_OPENMP_BACKEND AND TARGET OpenMP::OpenMP_CXX)
    target_compile_definitions("${ProjectName}-test" PRIVATE LBMINI_TEST_OPENMP_CPU)
    if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
        target_compile_definitions("${ProjectName}-test" PRIVATE LBMINI_TEST_OPENMP_GPU)
    endif()
    target_link_libraries("${ProjectName}-test" PUBLIC "${ProjectName}-lbm-openmp")
endif()

if(TARGET "${ProjectName}-lbm-openacc-cpu")
    target_compile_definitions("${ProjectName}-test" PRIVATE LBMINI_TEST_OPENACC_CPU)
    target_link_libraries("${ProjectName}-test" PUBLIC "${ProjectName}-lbm-openacc-cpu")
endif()

if(TARGET "${ProjectName}-lbm-openacc-gpu")
    target_compile_definitions("${ProjectName}-test" PRIVATE LBMINI_TEST_OPENACC_GPU)
    target_link_libraries("${ProjectName}-test" PUBLIC "${ProjectName}-lbm-openacc-gpu")
endif()

if(ENABLE_OPENCL AND OpenCL_FOUND)
    target_compile_definitions("${ProjectName}-test" PRIVATE LBMINI_TEST_OPENCL_CPU LBMINI_TEST_OPENCL_GPU)
    target_link_libraries("${ProjectName}-test"
            PUBLIC "${ProjectName}-lbm-opencl-cpu"
            PUBLIC "${ProjectName}-lbm-opencl-gpu"
    )
endif()

if(TARGET "${ProjectName}-lbm-kokkos")
    target_compile_definitions("${ProjectName}-test" PRIVATE LBMINI_TEST_KOKKOS)
    target_link_libraries("${ProjectName}-test" PUBLIC "${ProjectName}-lbm-kokkos")
endif()

if(TARGET "${ProjectName}-lbm-raja")
    target_compile_definitions("${ProjectName}-test" PRIVATE LBMINI_TEST_RAJA)
    target_link_libraries("${ProjectName}-test" PUBLIC "${ProjectName}-lbm-raja")
endif()

if(ENABLE_CUDA AND TARGET "${ProjectName}-lbm-cuda")
    target_compile_definitions("${ProjectName}-test" PRIVATE LBMINI_TEST_CUDA)
    target_link_libraries("${ProjectName}-test" PUBLIC "${ProjectName}-lbm-cuda")
endif()

# The GPU-targeted instantiations need a device to be present at run time, so
# they stay opt-in even when the frameworks were built with a device backend.
if(ENABLE_TEST_DEVICE_BACKENDS)
    target_compile_definitions("${ProjectName}-test" PRIVATE LBMINI_TEST_DEVICE_BACKENDS)
endif()

include(GoogleTest)
gtest_discover_tests("${ProjectName}-test" DISCOVERY_MODE PRE_TEST)

# Convenience entry point for the pre-benchmark gate: runs only the
# memory-safety suite, which is the part worth paying the sanitizer /
# Valgrind slowdown for.
add_custom_target("${ProjectName}-memory-check"
        COMMAND "$<TARGET_FILE:${ProjectName}-test>"
        --gtest_filter=BackendMemorySafety*:StreamingTableBounds*
        DEPENDS "${ProjectName}-test"
        COMMENT "Running the LBM memory-safety suite (see docs/memory-safety.md)"
        USES_TERMINAL
)

# =========================================================
