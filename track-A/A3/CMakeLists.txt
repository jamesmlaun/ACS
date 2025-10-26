# ============================================================
# Project A3: Approximate Membership Filters
# ============================================================
cmake_minimum_required(VERSION 3.10)
project(ProjectA3_Filters LANGUAGES C CXX)

# ------------------------------------------------------------
# C++ standard and compile options
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
    -O3
    -march=native
    -Wall
    -Wextra
    -Wpedantic
    -fno-exceptions
    -fno-rtti
)

# ------------------------------------------------------------
# Source discovery
# ------------------------------------------------------------
file(GLOB_RECURSE FILTER_SRC "filters/*.cpp")
file(GLOB_RECURSE BENCH_SRC  "benchmark/bench.cpp")
file(GLOB_RECURSE TEST_SRC   "benchmark/test_filters.cpp")

# ------------------------------------------------------------
# Include directories
# ------------------------------------------------------------
include_directories(
    ${PROJECT_SOURCE_DIR}/filters
    ${PROJECT_SOURCE_DIR}/benchmark
    ${PROJECT_SOURCE_DIR}/tests
    ${PROJECT_SOURCE_DIR}/external
)

# ------------------------------------------------------------
# External xxHash library (fast non-cryptographic hash)
# ------------------------------------------------------------
add_library(xxhash STATIC external/xxhash.c)
set_target_properties(xxhash PROPERTIES LINKER_LANGUAGE C)
target_include_directories(xxhash PUBLIC external)

# ------------------------------------------------------------
# Benchmark executable
# ------------------------------------------------------------
add_executable(bench
    ${BENCH_SRC}
    ${FILTER_SRC}
)

set_target_properties(bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
)

# ------------------------------------------------------------
# Test executable for filter correctness
# ------------------------------------------------------------
add_executable(test_filters
    ${TEST_SRC}
    ${FILTER_SRC}
)

set_target_properties(test_filters PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
)

# ------------------------------------------------------------
# Threads linkage (required for benchmark concurrency)
# ------------------------------------------------------------
find_package(Threads REQUIRED)

# ------------------------------------------------------------
# Link libraries (consistent keyword signature)
# ------------------------------------------------------------
target_link_libraries(bench PRIVATE Threads::Threads xxhash)
target_link_libraries(test_filters PRIVATE Threads::Threads xxhash)

# ------------------------------------------------------------
# Build summary
# ------------------------------------------------------------
message(STATUS "-------------------------------------------------------")
message(STATUS " Project A3 Approximate Membership Filters Build Config")
message(STATUS " Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS " C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS " Output binaries: ${PROJECT_SOURCE_DIR}/build/")
message(STATUS "   - bench")
message(STATUS "   - test_filters")
message(STATUS "-------------------------------------------------------")
