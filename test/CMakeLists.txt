cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)

enable_testing()

find_package(Armadillo CONFIG)
find_package(GTest CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(FFTW3 CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)


function(add_exe TARGET)

    add_executable(${TARGET} ${ARGN})
    set_target_properties(${TARGET} PROPERTIES
        CXX_STANDARD 20
        CXX_EXTENSIONS OFF
    )
    target_link_libraries(${TARGET}
        PRIVATE
        FFTW3::fftw3
        FFTW3::fftw3f
        fmt::fmt
        armadillo
    )
    target_include_directories(${TARGET} 
        PRIVATE
        ../include
    )

endfunction()

add_exe(test_script_fftconv test_script.cpp)
add_exe(test_fftconv 
    test_fftw.cpp
    test_fftconv.cpp
    test_hilbert.cpp
)
    
# On Windows, linking GTest::gmock causes gtest to not discover any tests
# https://github.com/google/googletest/issues/2157
target_link_libraries(test_fftconv
    PRIVATE
    GTest::gtest
)

include(GoogleTest)
gtest_discover_tests(test_fftconv)