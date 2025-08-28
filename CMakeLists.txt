# =========================================================
# LBM mini-app

cmake_minimum_required(VERSION 3.20)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

set(ProjectName "lbm-miniapp")
set(ProjectNamespace "lbmini")

option(ENABLE_CUDA "Enable CUDA for the project" ON)

set(PROJECT_LANGUAGES CXX)
if (ENABLE_CUDA)
    list(APPEND PROJECT_LANGUAGES CUDA)
endif ()

set(CMAKE_CXX_STANDARD 17)
if (ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif ()

project("${ProjectName}" LANGUAGES ${PROJECT_LANGUAGES})

if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
    message(STATUS "Configuring for NVIDIA HPC SDK")
    set(ARCH_FLAG "-tp=native")
    set(DEBUG_OPT_FLAG "-g")
    list(APPEND CMAKE_PREFIX_PATH "/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cmake")
    find_package(NVHPC REQUIRED)
    if(ENABLE_CUDA)
        set(NVHPC_GPU_ARCH "ccnative" CACHE STRING "NVIDIA GPU architecture to target, e.g. ccnative")
    endif()
else ()
    set(ARCH_FLAG "-march=native")
    set(DEBUG_OPT_FLAG "-Og")
endif()

add_compile_options(
        -Wall
        -Werror
        -Wno-error
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
        GIT_TAG 3.4.0
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
# yaml-cpp

include(FetchContent)
FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 2f86d13775d119edbb69af52e5f566fd65c6953b
)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
FetchContent_MakeAvailable(yaml-cpp)

# =========================================================
# Targets

# --- Data

add_library("${ProjectName}-data" STATIC
        lbmini/Data.hpp
        lbmini/Data/ControlData.hpp
        lbmini/Data/DataReader.hpp
        lbmini/Data/FluidData.hpp
        lbmini/Data/MeshData.hpp
)

target_include_directories("${ProjectName}-data" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")

set_target_properties("${ProjectName}-data" PROPERTIES
        LINKER_LANGUAGE CXX
)

target_compile_definitions("${ProjectName}-data" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})

target_link_libraries("${ProjectName}-data"
        PUBLIC Eigen3::Eigen
        PUBLIC yaml-cpp::yaml-cpp
)

# --- Lbm

add_library(
        "${ProjectName}-lbm" STATIC
        lbmini/Lbm/LbmBase.hpp
        lbmini/Lbm/OpenMp/LbmD2Q9.hpp
        lbmini/Lbm/OpenMp/LbmTube.hpp
        lbmini/Lbm/Plain/LbmD2Q9.hpp
        lbmini/Lbm/Plain/LbmTube.hpp
        lbmini/Utils.hpp
)

target_include_directories("${ProjectName}-lbm" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")

set_target_properties("${ProjectName}-lbm" PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION ON
)

target_compile_definitions("${ProjectName}-lbm" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})

if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA|NVHPC")
    target_compile_options("${ProjectName}-lbm" PUBLIC -mp -acc -Minfo=accel)
    target_link_options("${ProjectName}-lbm" INTERFACE -mp -acc)
    if(ENABLE_CUDA)
        target_compile_options("${ProjectName}-lbm" PUBLIC -cuda -gpu=${NVHPC_GPU_ARCH} -mp=gpu)
        target_link_options("${ProjectName}-lbm" INTERFACE -cuda)
    endif()
endif()

target_link_libraries("${ProjectName}-lbm"
        PUBLIC Eigen3::Eigen
)

# --- Main

add_executable("${ProjectName}-main" lbmini/main.cpp)
target_link_libraries("${ProjectName}-main"
        PUBLIC "${ProjectName}-data"
        PUBLIC "${ProjectName}-lbm"
)

# --- GTest

enable_testing()

add_executable("${ProjectName}-test"
        test/sanity_test.cpp
)

target_link_libraries("${ProjectName}-test"
        PUBLIC gtest_main
        PUBLIC "${ProjectName}-data"
        PUBLIC "${ProjectName}-lbm"
)

include(GoogleTest)
gtest_discover_tests("${ProjectName}-test")

# =========================================================
