cmake_minimum_required(VERSION 3.20)
project(fftconv C CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(BENCHMARK CONFIG REQUIRED)
find_package(Armadillo CONFIG REQUIRED)
find_package(FFTW3 CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)

add_executable(bench_fftconv 
    bench_fftconv.cpp
    bench_hilbert.cpp
)

target_include_directories(bench_fftconv
    PRIVATE
    ../include
)

target_link_libraries(bench_fftconv 
    PRIVATE 
    FFTW3::fftw3
    FFTW3::fftw3f
    benchmark::benchmark_main
    armadillo
)
