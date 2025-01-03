cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

project(fftconv C CXX)

# only works for Makefile and Ninja
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 

# Build test
add_subdirectory(test)

# Build benchmark
add_subdirectory(benchmark)
