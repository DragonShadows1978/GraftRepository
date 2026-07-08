cmake_minimum_required(VERSION 3.16)
project(grm_runtime LANGUAGES CXX)

# Default to an optimized build when the caller does not specify one.
# An empty CMAKE_BUILD_TYPE compiles with no optimization at all (-O0):
# measured 2026-07-08 (MLA route P1 follow-on, 100k nodes, topk=3), the
# unoptimized library routes ~21x slower than -O3 (int4 M=128: 222ms vs
# 10.5ms; fp32: 233ms vs 19.7ms) — every receipted router number assumes
# an optimized build. OpenMP remains opt-in (harness --openmp / explicit
# toolchain flags): it changes threading behavior and is gated separately
# (grm_router_tsan_gate).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Build type (defaults to Release; was previously empty = -O0)" FORCE)
endif()

add_library(grm_runtime STATIC grm_runtime.cpp)
target_include_directories(grm_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(grm_runtime PUBLIC cxx_std_17)

add_library(grm_runtime_shared SHARED grm_runtime.cpp)
set_target_properties(grm_runtime_shared PROPERTIES OUTPUT_NAME grm_runtime)
target_include_directories(grm_runtime_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(grm_runtime_shared PUBLIC cxx_std_17)
