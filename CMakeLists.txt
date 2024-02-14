cmake_minimum_required(VERSION 3.20)
project(fftconv C CXX)
set(CMAKE_CXX_STANDARD 20)

# only works for Makefile and Ninja
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 

# Build fftconv
add_subdirectory(fftconv_fftw)
add_subdirectory(fftconv_pocketfft)

# Build test
add_subdirectory(test)

# Build benchmark
add_subdirectory(benchmark)
