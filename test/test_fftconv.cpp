// NOLINTBEGIN(*-magic-numbers,*-array-index)
#include <armadillo>
#include <array>
#include <complex>
#include <fftw3.h>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <span>
#include <vector>

#include <fftconv/fftconv.hpp>

#include "test_common.hpp"

using fftconv::ConvMode;

// Test internal utils
// Assuming the function is defined here or included from another header
TEST(CopyToPaddedBuffer, SameTypeLargerDestination) {
  const std::vector<int> src = {1, 2, 3};
  // Initialize with a non-zero value to test padding
  std::vector<int> dst(5, -1);

  fftconv::internal::copy_to_padded_buffer(std::span(src), std::span(dst));

  EXPECT_EQ(dst[0], 1);
  EXPECT_EQ(dst[1], 2);
  EXPECT_EQ(dst[2], 3);
  EXPECT_EQ(dst[3], 0); // Check padding
  EXPECT_EQ(dst[4], 0); // Check padding
}

TEST(CopyToPaddedBuffer, SameSizeNoPaddingNeeded) {
  const std::vector<int> src = {4, 5, 6};
  std::vector<int> dst(3, -1); // Initialize with a non-zero value

  fftconv::internal::copy_to_padded_buffer(std::span(src), std::span(dst));

  EXPECT_EQ(dst[0], 4);
  EXPECT_EQ(dst[1], 5);
  EXPECT_EQ(dst[2], 6);
  // No padding necessary, so no checks for zeros beyond this point
}

TEST(ElementwiseMultiply, CorrectMultiplication) {
  // Setup test data
  std::array<std::complex<double>, 2> complex1{{{1, 2}, {3, 4}}};

  std::array<std::complex<double>, 2> complex2{{{5, 6}, {7, 8}}};

  std::array<std::complex<double>, 2> result;

  // Perform element-wise multiplication
  fftconv::internal::multiply_cx<double>(complex1, complex2, result);

  // Verify the results
  EXPECT_DOUBLE_EQ(result[0].real(), -7);  // (1*5 - 2*6) = 5 - 12 = -7
  EXPECT_DOUBLE_EQ(result[0].imag(), 16);  // (2*5 + 1*6) = 10 + 6 = 16
  EXPECT_DOUBLE_EQ(result[1].real(), -11); // (3*7 - 4*8) = 21 - 32 = -11
  EXPECT_DOUBLE_EQ(result[1].imag(), 52);  // (4*7 + 3*8) = 28 + 24 = 52
}

TEST(ElementwiseMultiply, HandlesDifferentSizes) {
  // Setup test data
  std::array<std::complex<double>, 3> complex1{{
      {1, 1},
      {2, 2},
      {3, 3},
  }};

  std::array<std::complex<double>, 2> complex2{{
      {1, -1},
      {1, -1},
  }};

  std::array<std::complex<double>, 3> result{};

  // Perform element-wise multiplication
  fftconv::internal::multiply_cx<double>(complex1, complex2, result);

  // Only the first 2 elements should be computed
  EXPECT_DOUBLE_EQ(result[0].real(), 2);
  EXPECT_DOUBLE_EQ(result[0].imag(), 0);
  EXPECT_DOUBLE_EQ(result[1].real(), 4);
  EXPECT_DOUBLE_EQ(result[1].imag(), 0);

  // The third element of complex1 and result should not be computed (size
  // mismatch)
  EXPECT_DOUBLE_EQ(result[2].real(), 0);
  EXPECT_DOUBLE_EQ(result[2].imag(), 0);
}

TEST(FFTConvEngine, OAConvFull) {
  using T = double;
  arma::Col<T> kernel(95, arma::fill::randn);

  auto &plans = fftconv::FFTConvEngine<T>::get_for_ksize(kernel.size());
  for (const auto arr_size : {1000, 2000, 3000}) {
    arma::Col<T> arr(arr_size, arma::fill::randn);
    arma::Col<T> expected = arma::conv(arr, kernel);
    arma::Col<T> res(arr.size() + kernel.size() - 1, arma::fill::zeros);

    plans.oaconvolve<ConvMode::Full>(arr, kernel, res);

    ExpectVectorsNear(std::span<const T>(res), std::span<const T>(expected),
                      getTol<T>());
  }
}

TEST(FFTConvEngine, OAConvSame) {
  using T = double;
  arma::Col<T> kernel(95, arma::fill::randn);

  auto &plans = fftconv::FFTConvEngine<T>::get_for_ksize(kernel.size());
  for (const auto arr_size : {1000, 2000, 3000}) {
    arma::Col<T> arr(arr_size, arma::fill::randn);
    arma::Col<T> expected = arma::conv(arr, kernel, "same");
    arma::Col<T> res(arr.size(), arma::fill::zeros);

    plans.oaconvolve<ConvMode::Same>(arr, kernel, res);

    ExpectVectorsNear<T>(res, expected, getTol<T>());
  }
}

// Test convolution
template <fftconv::Floating T, ConvMode Mode, typename Func>
void test_conv(Func conv_func) {
  for (int real_size = 4; real_size < 100; real_size += 9) {
    arma::Col<T> a(real_size, arma::fill::randn);
    arma::Col<T> k(real_size, arma::fill::randn);

    size_t outSize{};
    arma::Col<T> expected;
    if constexpr (Mode == ConvMode::Full) {
      outSize = a.size() + k.size() - 1;
      expected = arma::conv(a, k, "full");
    } else if constexpr (Mode == ConvMode::Same) {
      outSize = a.size();
      expected = arma::conv(a, k, "same");
    }

    arma::Col<T> out(outSize, arma::fill::zeros);
    conv_func(a, k, out);

    ExpectVectorsNear<T>(out, expected, getTol<T>());
  }
}

// Test convolution
template <fftconv::Floating T, ConvMode Mode, typename Func>
void test_oaconv(Func conv_func) {
  const int ksize = 95;
  const arma::Col<T> kernel(ksize, arma::fill::randn);

  for (int real_size = 200; real_size < 501; real_size += 150) {
    const arma::Col<T> a(real_size, arma::fill::randn);

    size_t outSize{};
    arma::Col<T> expected;
    if constexpr (Mode == ConvMode::Full) {
      outSize = a.size() + kernel.size() - 1;
      expected = arma::conv(a, kernel, "full");
    } else if constexpr (Mode == ConvMode::Same) {
      outSize = a.size();
      expected = arma::conv(a, kernel, "same");
    }

    arma::Col<T> res(outSize, arma::fill::zeros);
    conv_func(a, kernel, res);

    ExpectVectorsNear(std::span<const T>(res), std::span<const T>(expected),
                      getTol<T>());
  }
}

TEST(Convolve, Full) {
  constexpr auto mode = ConvMode::Full;
  test_conv<double, mode>(fftconv::convolve_fftw<double, mode>);
  test_conv<float, mode>(fftconv::convolve_fftw<float, mode>);
}

TEST(Convolve, Same) {
  constexpr auto mode = ConvMode::Same;
  test_conv<double, mode>(fftconv::convolve_fftw<double, mode>);
  test_conv<float, mode>(fftconv::convolve_fftw<float, mode>);
}


TEST(OAConvolve, Full) {
  constexpr auto mode = ConvMode::Full;
  test_conv<double, mode>(fftconv::oaconvolve_fftw<double, mode>);
  test_conv<float, mode>(fftconv::oaconvolve_fftw<float, mode>);
}

TEST(OAConvolve, Same) {
  constexpr auto mode = ConvMode::Same;

  test_oaconv<double, mode>(fftconv::oaconvolve_fftw<double, mode>);
  test_oaconv<float, mode>(fftconv::oaconvolve_fftw<float, mode>);
}


// NOLINTEND(*-magic-numbers,*-array-index)
