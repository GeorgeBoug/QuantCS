# 3.18 is required for CMAKE_CUDA_ARCHITECTURES (used below when the CUDA
# backend is enabled); the CPU-only path would work with an older CMake too.
cmake_minimum_required(VERSION 3.18)
project(QuantCS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

find_package(OpenMP REQUIRED)

# --- Optional CUDA GPU backend -------------------------------------------
# QUANTCS_ENABLE_CUDA lets a CUDA-less machine (or CI runner) build the
# CPU-only backend by passing -DQUANTCS_ENABLE_CUDA=OFF; otherwise we
# probe for a working CUDA compiler with CMake's standard check_language()
# and silently fall back to CPU-only if none is found -- no hardcoded nvcc
# path, no assumption about where the toolkit lives.
option(QUANTCS_ENABLE_CUDA "Build the CUDA GPU backend if a CUDA toolkit is available" ON)

set(QUANTCS_CUDA_ENABLED FALSE)
if(QUANTCS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        # Covers Pascal through Ada Lovelace (GTX 10-series up through
        # RTX 40-series / current data-center parts) so the same build
        # works across most CUDA-capable machines without per-GPU tuning.
        # Override with -DCMAKE_CUDA_ARCHITECTURES=<arch> for something
        # older/newer, or "native" to autodetect the build machine's GPU.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89)
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(QUANTCS_CUDA_ENABLED TRUE)
        message(STATUS "CUDA compiler found (${CMAKE_CUDA_COMPILER}) -- building GPU backend")
    else()
        message(STATUS "No CUDA compiler found -- building CPU-only (OpenMP) backend")
    endif()
endif()

# Core simulator library, shared by the demo binary, tests, and benchmarks.
add_library(quantcs
    src/state_vector.cpp
    src/circuit.cpp
    src/grover.cpp
)
target_include_directories(quantcs PUBLIC include)
target_link_libraries(quantcs PUBLIC OpenMP::OpenMP_CXX)
if(MSVC)
    target_compile_options(quantcs PUBLIC /EHsc)
endif()

if(QUANTCS_CUDA_ENABLED)
    target_sources(quantcs PRIVATE src/cuda/state_vector_gpu.cu)
    target_compile_definitions(quantcs PUBLIC QUANTCS_WITH_CUDA)
    set_target_properties(quantcs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Demo command-line binary: builds and prints a Bell state.
add_executable(quantcs_demo src/main.cpp)
target_link_libraries(quantcs_demo PRIVATE quantcs)

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)

# --- invmul: bidirectional array multiplier / factoring by local search ---
# A self-contained module that shares this build (compiler settings, OpenMP,
# CTest) but neither links against nor modifies the simulator above. See
# invmul/README.md.
add_subdirectory(invmul)
