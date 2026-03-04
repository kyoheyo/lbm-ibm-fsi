cmake_minimum_required(VERSION 3.20)
project(lbm_ibm_fsi VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(ENABLE_MPI    "Enable MPI parallelism"   ON)
option(ENABLE_OPENMP "Enable OpenMP parallelism" ON)
option(ENABLE_CUDA   "Enable CUDA GPU backend"   OFF)
option(BUILD_TESTS   "Build unit tests"          ON)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# MPI
# ---------------------------------------------------------------------------
if(ENABLE_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
endif()

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP found: version ${OpenMP_CXX_VERSION}")
endif()

# ---------------------------------------------------------------------------
# CUDA (optional)
# ---------------------------------------------------------------------------
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_subdirectory(core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(core/tests)
endif()
