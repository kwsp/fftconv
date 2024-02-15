cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)

find_package(FFTW3 CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)

add_library(fftconv INTERFACE)
target_include_directories(fftconv INTERFACE .)
target_link_libraries(fftconv INTERFACE FFTW3::fftw3 FFTW3::fftw3f)
