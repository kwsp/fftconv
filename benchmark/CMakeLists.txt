cmake_minimum_required(VERSION 3.20)
project(fftconv C CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(BENCHMARK CONFIG REQUIRED)
find_package(Armadillo CONFIG REQUIRED)

add_executable(bench_fftconv)
target_sources(bench_fftconv
    PRIVATE 
    bench_fftconv.cpp
)

target_link_libraries(bench_fftconv PRIVATE fftconv fftconv_pocket)

target_link_libraries(bench_fftconv
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    armadillo
)