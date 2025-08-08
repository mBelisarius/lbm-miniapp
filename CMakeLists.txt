# =========================================================
# LBM mini-app

cmake_minimum_required(VERSION 3.20)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

set(ProjectName "lbm-miniapp")
set(ProjectNamespace "lma")
set(CMAKE_CXX_STANDARD 23)

project("${ProjectName}")

add_compile_options(
        -Wall -Werror -Wno-error=uninitialized
        $<$<CONFIG:RELEASE>:-O3>
        $<$<CONFIG:RELEASE>:-DNDEBUG>
        $<$<CONFIG:RELWITHDEBINFO>:-O2>
        $<$<CONFIG:RELWITHDEBINFO>:-DNDEBUG>
        $<$<CONFIG:RELWITHDEBINFO>:-g>
        $<$<CONFIG:MINSIZEREL>:-Oz>
        $<$<CONFIG:MINSIZEREL>:-DNDEBUG>
        $<$<CONFIG:DEBUG>:-Og>
        $<$<CONFIG:DEBUG>:-g>
)

# =========================================================
# gtest

include(FetchContent)
FetchContent_Declare(
        googletest
        URL
        https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
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
        URL
        https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(EIGEN_FORCE_SHARED_CRT ON CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
FetchContent_MakeAvailable(Eigen)

# =========================================================
# Targets

# --- project

add_library(
        "${ProjectName}" STATIC
        src/information.cpp
        include/information.h
        include/FluidProperties.hpp
        include/LbmBase.hpp
        include/LbmD3Q19.hpp
        include/LbmTube.hpp
)

target_include_directories("${ProjectName}" PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_include_directories("${ProjectName}" PUBLIC "${CMAKE_SOURCE_DIR}/src")

target_link_libraries("${ProjectName}" PUBLIC Eigen3::Eigen)

target_compile_definitions("${ProjectName}" PRIVATE PROJECT_NAMESPACE=${ProjectNamespace})

# --- main

add_executable("${ProjectName}-main" src/main.cpp)
target_link_libraries("${ProjectName}-main" PUBLIC "${ProjectName}")

# --- gtest

enable_testing()

add_executable("${ProjectName}-test"
        test/sanity_test.cpp
)

target_link_libraries("${ProjectName}-test" gtest_main "${ProjectName}")

include(GoogleTest)
gtest_discover_tests("${ProjectName}-test")

# =========================================================
