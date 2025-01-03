set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(FFTW3 CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)

pybind11_add_module(pyfftconv wrapper.cpp)
target_include_directories(pyfftconv
    PRIVATE 
    ../include
)
target_link_libraries(pyfftconv 
    PRIVATE
    FFTW3::fftw3
    FFTW3::fftw3f
)
install(TARGETS pyfftconv DESTINATION .)