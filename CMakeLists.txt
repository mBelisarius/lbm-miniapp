# =========================================================
# LBM mini-app

cmake_minimum_required(VERSION 3.20)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

set(ProjectName "lbm-miniapp")
set(ProjectNamespace "lbmini")

option(ENABLE_CUDA "Enable CUDA for the project" ON)
option(ENABLE_KOKKOS "Enable the Kokkos backend" ON)
option(ENABLE_KOKKOS_CUDA "Build Kokkos with its CUDA device backend" OFF)
option(ENABLE_RAJA "Enable the RAJA backend" ON)
option(ENABLE_RAJA_CUDA "Build RAJA with its CUDA device backend" OFF)

set(CMAKE_CXX_STANDARD 17)
project("${ProjectName}" LANGUAGES CXX)

if (ENABLE_CUDA)
    enable_language(CUDA OPTIONAL)
    if (CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        message(STATUS "CUDA language enabled with compiler: ${CMAKE_CUDA_COMPILER}")
    else ()
        set(ENABLE_CUDA OFF CACHE BOOL "Enable CUDA for the project" FORCE)
        message(WARNING "CUDA compiler not found. Disabling CUDA for this build.")
    endif ()
endif ()

if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
    message(STATUS "Configuring for NVIDIA HPC SDK")
    set(ARCH_FLAG "$<$<COMPILE_LANGUAGE:CXX>:-tp=native>")
    set(DEBUG_OPT_FLAG "-g")
    list(APPEND CMAKE_PREFIX_PATH "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cmake")
    find_package(NVHPC REQUIRED)
    if(ENABLE_CUDA)
        set(NVHPC_GPU_ARCH "ccnative" CACHE STRING "NVIDIA GPU architecture to target, e.g. ccnative")
    endif()
else ()
    set(ARCH_FLAG "$<$<COMPILE_LANGUAGE:CXX>:-march=native>")
    set(DEBUG_OPT_FLAG "-Og")
endif()

add_compile_options(
        -Wall
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-error>
        $<$<CONFIG:RELEASE>:-O3>
        $<$<CONFIG:RELEASE>:-DNDEBUG>
        $<$<CONFIG:RELEASE>:${ARCH_FLAG}>
        $<$<CONFIG:RELWITHDEBINFO>:-O2>
        $<$<CONFIG:RELWITHDEBINFO>:-DNDEBUG>
        $<$<CONFIG:RELWITHDEBINFO>:-g>
        $<$<CONFIG:RELWITHDEBINFO>:${ARCH_FLAG}>
        $<$<CONFIG:MINSIZEREL>:-Oz>
        $<$<CONFIG:MINSIZEREL>:-DNDEBUG>
        $<$<CONFIG:MINSIZEREL>:${ARCH_FLAG}>
        $<$<CONFIG:DEBUG>:${DEBUG_OPT_FLAG}>
        $<$<CONFIG:DEBUG>:-g>
)

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
        GIT_TAG 5.0.0
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

find_package(MPI)

# =========================================================
# yaml-cpp

include(FetchContent)
FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.9.0
)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
FetchContent_MakeAvailable(yaml-cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
    foreach(_yaml_target yaml-cpp yaml-cpp-sandbox yaml-cpp-parse yaml-cpp-read)
        if(TARGET ${_yaml_target})
            target_compile_options(${_yaml_target} PRIVATE --diag_suppress=code_is_unreachable)
        endif()
    endforeach()
endif()

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
            GIT_TAG 4.4.01
    )

    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
    if (TARGET OpenMP::OpenMP_CXX OR CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
        set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
    endif ()
    if (ENABLE_KOKKOS_CUDA)
        set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
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
if (ENABLE_RAJA)
    include(FetchContent)
    FetchContent_Declare(
            RAJA
            GIT_REPOSITORY https://github.com/LLNL/RAJA.git
            GIT_TAG v2024.07.0
    )

    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    # BLT (RAJA's build system) propagates BLT_CXX_STD to CMAKE_CXX_STANDARD;
    # pin it so RAJA does not downgrade the project to C++14.
    set(BLT_CXX_STD "c++17" CACHE STRING "" FORCE)
    # BLT would otherwise register its own bundled googletest and collide with
    # the one fetched above.
    set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(ENABLE_DOCS OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_EXERCISES OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_REPRODUCERS OFF CACHE BOOL "" FORCE)
    if (TARGET OpenMP::OpenMP_CXX OR CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
        set(ENABLE_OPENMP ON CACHE BOOL "" FORCE)
        set(RAJA_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
    endif ()
    if (ENABLE_RAJA_CUDA)
        set(ENABLE_CUDA ON CACHE BOOL "" FORCE)
        set(RAJA_ENABLE_CUDA ON CACHE BOOL "" FORCE)
    endif ()

    # BLT writes CMAKE_*_OUTPUT_DIRECTORY into the cache and would relocate
    # *every* project binary into RAJA's build tree. Claiming those cache
    # entries first (without FORCE, so an explicit user setting still wins)
    # keeps the mini-app executables where the rest of the tooling expects them.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" CACHE PATH "")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" CACHE PATH "")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}" CACHE PATH "")

    FetchContent_MakeAvailable(RAJA)

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
        lbmini/Executor.hpp
        lbmini/Executor.cpp
        lbmini/Utils.hpp
)

target_include_directories("${ProjectName}-common" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")

set_target_properties("${ProjectName}-common" PROPERTIES
        LINKER_LANGUAGE CXX
)

if(MPI_CXX_FOUND)
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
if(ENABLE_RAJA AND TARGET RAJA)
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

if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
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
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options("${ProjectName}-lbm-openacc-cpu" PUBLIC
            "-fopenacc"
    )
    target_link_options("${ProjectName}-lbm-openacc-cpu" INTERFACE
            "-fopenacc"
    )
    target_compile_options("${ProjectName}-lbm-openacc-gpu" PUBLIC
            "-fopenacc"
    )
    target_link_options("${ProjectName}-lbm-openacc-gpu" INTERFACE
            "-fopenacc"
    )
endif()

find_package(OpenCL REQUIRED)

# --- Lbm Cuda

set_source_files_properties(lbmini/Lbm/Cuda/Gpu/LbmTube.cu PROPERTIES LANGUAGE CXX)
add_library("${ProjectName}-lbm-cuda" STATIC
        lbmini/Lbm/Cuda/Gpu/LatticeD2Q9.hpp
        lbmini/Lbm/Cuda/Gpu/LbmTube.hpp
        lbmini/Lbm/Cuda/Gpu/LbmTube.cu
)

target_include_directories("${ProjectName}-lbm-cuda" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")
set_target_properties("${ProjectName}-lbm-cuda" PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION OFF
)

target_compile_definitions("${ProjectName}-lbm-cuda" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})
target_link_libraries("${ProjectName}-lbm-cuda" PUBLIC Eigen3::Eigen PUBLIC "${ProjectName}-common")

if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
    if(ENABLE_CUDA)
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

add_library("${ProjectName}-lbm-opencl-cpu" STATIC
        lbmini/Lbm/OpenCl/Cpu/LatticeD2Q9.hpp
        lbmini/Lbm/OpenCl/Cpu/LbmTube.hpp
)

target_include_directories("${ProjectName}-lbm-opencl-cpu" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")
set_target_properties("${ProjectName}-lbm-opencl-cpu" PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions("${ProjectName}-lbm-opencl-cpu" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})
target_link_libraries("${ProjectName}-lbm-opencl-cpu" PUBLIC Eigen3::Eigen PUBLIC "${ProjectName}-common" PUBLIC OpenCL::OpenCL)

# --- Lbm OpenCL GPU

add_library("${ProjectName}-lbm-opencl-gpu" STATIC
        lbmini/Lbm/OpenCl/Gpu/LatticeD2Q9.hpp
        lbmini/Lbm/OpenCl/Gpu/LbmTube.hpp
)

target_include_directories("${ProjectName}-lbm-opencl-gpu" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")
set_target_properties("${ProjectName}-lbm-opencl-gpu" PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions("${ProjectName}-lbm-opencl-gpu" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})
target_link_libraries("${ProjectName}-lbm-opencl-gpu" PUBLIC Eigen3::Eigen PUBLIC "${ProjectName}-common" PUBLIC OpenCL::OpenCL)

# --- Lbm MPI CPU

add_library("${ProjectName}-lbm-mpi" STATIC
        lbmini/Lbm/Mpi/Cpu/LatticeD2Q9.hpp
        lbmini/Lbm/Mpi/Cpu/LbmTube.hpp
)

target_include_directories("${ProjectName}-lbm-mpi" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")
set_target_properties("${ProjectName}-lbm-mpi" PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions("${ProjectName}-lbm-mpi" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})
target_link_libraries("${ProjectName}-lbm-mpi" PUBLIC Eigen3::Eigen PUBLIC "${ProjectName}-common" PUBLIC MPI::MPI_CXX)

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

if(ENABLE_RAJA AND TARGET RAJA)
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
set_target_properties("${ProjectName}-main" PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_link_libraries("${ProjectName}-main"
        PUBLIC MPI::MPI_CXX
        PUBLIC Eigen3::Eigen
        PUBLIC "${ProjectName}-common"
        PUBLIC "${ProjectName}-lbm-plain"
        PUBLIC "${ProjectName}-lbm-openmp"
        PUBLIC "${ProjectName}-lbm-openacc-cpu"
        PUBLIC "${ProjectName}-lbm-openacc-gpu"
        PUBLIC "${ProjectName}-lbm-opencl-cpu"
        PUBLIC "${ProjectName}-lbm-opencl-gpu"
        PUBLIC "${ProjectName}-lbm-cuda"
        PUBLIC "${ProjectName}-lbm-mpi"
)

if(TARGET "${ProjectName}-lbm-kokkos")
    target_link_libraries("${ProjectName}-main" PUBLIC "${ProjectName}-lbm-kokkos")
endif()

if(TARGET "${ProjectName}-lbm-raja")
    target_link_libraries("${ProjectName}-main" PUBLIC "${ProjectName}-lbm-raja")
endif()

# --- GTest

enable_testing()

set(TestSources test/sanity_test.cpp)
if(TARGET "${ProjectName}-lbm-kokkos")
    list(APPEND TestSources test/kokkos_lbm_test.cpp)
endif()
if(TARGET "${ProjectName}-lbm-raja")
    list(APPEND TestSources test/raja_lbm_test.cpp)
endif()

add_executable("${ProjectName}-test" ${TestSources})
set_target_properties("${ProjectName}-test" PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_link_libraries("${ProjectName}-test"
        PUBLIC MPI::MPI_CXX
        PUBLIC Eigen3::Eigen
        PUBLIC gtest_main
        PUBLIC "${ProjectName}-common"
        PUBLIC "${ProjectName}-lbm-plain"
)

if(TARGET "${ProjectName}-lbm-kokkos")
    target_link_libraries("${ProjectName}-test" PUBLIC "${ProjectName}-lbm-kokkos")
endif()

if(TARGET "${ProjectName}-lbm-raja")
    target_link_libraries("${ProjectName}-test" PUBLIC "${ProjectName}-lbm-raja")
endif()

include(GoogleTest)
gtest_discover_tests("${ProjectName}-test")

# =========================================================
