# =========================================================
# LBM mini-app

cmake_minimum_required(VERSION 3.20)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

set(ProjectName "lbm-miniapp")
set(ProjectNamespace "lbmini")
set(CMAKE_CXX_STANDARD 23)

project("${ProjectName}")

add_compile_options(
        -Wall
        -Werror
        -Wno-error=uninitialized
        $<$<CONFIG:RELEASE>:-O3>
        $<$<CONFIG:RELEASE>:-DNDEBUG>
        $<$<CONFIG:RELEASE>:-march=native>
        $<$<CONFIG:RELWITHDEBINFO>:-O2>
        $<$<CONFIG:RELWITHDEBINFO>:-DNDEBUG>
        $<$<CONFIG:RELWITHDEBINFO>:-g>
        $<$<CONFIG:RELWITHDEBINFO>:-march=native>
        $<$<CONFIG:MINSIZEREL>:-Oz>
        $<$<CONFIG:MINSIZEREL>:-DNDEBUG>
        $<$<CONFIG:MINSIZEREL>:-march=native>
        $<$<CONFIG:DEBUG>:-Og>
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

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(EIGEN_FORCE_SHARED_CRT ON CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
FetchContent_MakeAvailable(Eigen)

# =========================================================
# OpenMP

find_package(OpenMP REQUIRED)

# =========================================================
# yaml-cpp

include(FetchContent)
FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG c2680200486572baf8221ba052ef50b58ecd816e
)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
FetchContent_MakeAvailable(yaml-cpp)

# =========================================================
# Targets

# --- project

add_library(
        "${ProjectName}" STATIC
        lbmini/Data.hpp
        lbmini/Data/ControlData.hpp
        lbmini/Data/DataReader.hpp
        lbmini/Data/FluidData.hpp
        lbmini/Data/MeshData.hpp
        lbmini/Lbm/LbmBase.hpp
        lbmini/Lbm/Plain/LbmD2Q9.hpp
        lbmini/Lbm/Plain/LbmTube.hpp
        lbmini/Utils.hpp
)

target_include_directories("${ProjectName}" PUBLIC "${CMAKE_SOURCE_DIR}/lbmini")

set_target_properties(${ProjectName} PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries("${ProjectName}"
        PUBLIC Eigen3::Eigen
        PUBLIC yaml-cpp::yaml-cpp
)

target_compile_definitions("${ProjectName}"
        PRIVATE PROJECT_NAMESPACE=${ProjectNamespace}
)

# --- Plain

add_executable("${ProjectName}-plain" lbmini/plain.cpp)
target_link_libraries("${ProjectName}-plain" PUBLIC "${ProjectName}")

# --- OpenMP

add_executable("${ProjectName}-openmp" lbmini/openmp.cpp)
target_link_libraries("${ProjectName}-openmp"
        PUBLIC "${ProjectName}"
        PUBLIC OpenMP::OpenMP_CXX
)

# --- GTest

enable_testing()

add_executable("${ProjectName}-test"
        test/sanity_test.cpp
)

target_link_libraries("${ProjectName}-test" gtest_main "${ProjectName}")

include(GoogleTest)
gtest_discover_tests("${ProjectName}-test")

# =========================================================
