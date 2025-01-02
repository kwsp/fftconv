cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)

enable_testing()

find_package(Armadillo CONFIG)
find_package(GTest CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)


add_executable(test_script_fftconv)
set_target_properties(test_script_fftconv PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)
target_sources(test_script_fftconv
    PRIVATE
    test_script.cpp
)
target_link_libraries(test_script_fftconv
    PRIVATE
    fmt::fmt
    armadillo
    fftconv
    fftconv_pocket
)


add_executable(test_fftconv)
target_sources(test_fftconv
    PRIVATE
    test_fftconv.cpp
)
# On Windows, linking GTest::gmock causes gtest to not discover any tests
# https://github.com/google/googletest/issues/2157
target_link_libraries(test_fftconv
    PRIVATE
    armadillo
    fftconv
    fmt::fmt
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_fftconv)