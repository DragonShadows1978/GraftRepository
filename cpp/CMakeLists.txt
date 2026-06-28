cmake_minimum_required(VERSION 3.16)
project(grm_runtime LANGUAGES CXX)

add_library(grm_runtime STATIC grm_runtime.cpp)
target_include_directories(grm_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(grm_runtime PUBLIC cxx_std_17)

add_library(grm_runtime_shared SHARED grm_runtime.cpp)
set_target_properties(grm_runtime_shared PROPERTIES OUTPUT_NAME grm_runtime)
target_include_directories(grm_runtime_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(grm_runtime_shared PUBLIC cxx_std_17)
