cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)

add_library(fftconv_pocket STATIC)
target_sources(fftconv_pocket
    PRIVATE 
    pocketfft.c
)
target_include_directories(fftconv_pocket PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})