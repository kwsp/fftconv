cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)

find_package(FFTW3 CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)

add_library(fftconv STATIC)
target_sources(fftconv
    PRIVATE 
    fftconv.cpp
)
target_include_directories(fftconv PUBLIC .)
target_link_libraries(fftconv PUBLIC FFTW3::fftw3)