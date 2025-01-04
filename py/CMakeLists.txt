cmake_minimum_required(VERSION 3.26)
project(pyfftconv LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(FFTW3 CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)

pybind11_add_module(_pyfftconv main.cpp)
target_include_directories(_pyfftconv PRIVATE ../include)
target_link_libraries(_pyfftconv PRIVATE FFTW3::fftw3 FFTW3::fftw3f)
install(TARGETS _pyfftconv DESTINATION pyfftconv)