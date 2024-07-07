cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)

find_package(Armadillo CONFIG)

add_executable(test_fftconv)
target_sources(test_fftconv
    PRIVATE
    test.cpp
)

target_link_libraries(test_fftconv
    PRIVATE
    armadillo
    fftconv
    fftconv_pocket
)