#include "test_common.hpp"
#include <array>
#include <fftconv/hilbert.hpp>
#include <gtest/gtest.h>

TEST(TestHilbertFFTW, Correct) {
  const auto fn = [&]<typename T>() {
    const std::array<T, 10> inp = {
        -0.999984, -0.736924, 0.511211, -0.0826997, 0.0655345,
        -0.562082, -0.905911, 0.357729, 0.358593,   0.869386,
    };
    const std::array<T, 10> expect = {
        1.45197493, 1.15365169, 0.54703078, 0.27346519, 0.15097965,
        0.83696245, 1.1476185,  0.71885109, 0.46089151, 1.07384968};
    std::array<T, 10> out{};

    fftconv::hilbert<T>(inp, out);

    ExpectArraysNear<T>(expect.data(), out.data(), expect.size(), 1e-6);
  };

  fn.template operator()<double>();
  fn.template operator()<float>();
}