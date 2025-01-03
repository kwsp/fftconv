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

TEST(CopyToPaddedBuffer, DifferentTypeConversion) {
  const std::vector<int> src = {1, 2, 3};
  // Initialize with a non-zero value to test padding
  std::vector<float> dst(5, -1.0F);

  fftconv::internal::copy_to_padded_buffer(std::span(src), std::span(dst));

  EXPECT_FLOAT_EQ(dst[0], 1.0F);
  EXPECT_FLOAT_EQ(dst[1], 2.0F);
  EXPECT_FLOAT_EQ(dst[2], 3.0F);
  EXPECT_FLOAT_EQ(dst[3], 0.0F); // Check padding
  EXPECT_FLOAT_EQ(dst[4], 0.0F); // Check padding
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

template <fftconv::Floating T> struct Tol {};
template <> struct Tol<float> {
  auto operator()() { return FloatTol; }
};
template <> struct Tol<double> {
  auto operator()() { return DoubleTol; }
};

TEST(FFTConvEngine, ExecutesOAConvFull) {
  using T = double;
  arma::Col<T> kernel(95, arma::fill::randn);

  auto &plans = fftconv::FFTConvEngine<T>::get_for_ksize(kernel.size());

  // for (const auto arr_size : {1000, 2000, 3000}) {
  for (const auto arr_size : {1000}) {
    arma::Col<T> arr(arr_size, arma::fill::randn);
    arma::Col<T> expected = arma::conv(arr, kernel);
    arma::Col<T> res(arr.size() + kernel.size() - 1, arma::fill::zeros);

    plans.oaconvolve_full(std::span<const T>(arr), std::span<const T>(kernel),
                          std::span<T>(res));

    expected.save(fmt::format("full_expected_{}.bin", arr_size),
                  arma::raw_binary);
    arr.save(fmt::format("full_arr_{}.bin", arr_size), arma::raw_binary);
    kernel.save(fmt::format("full_kernel_{}.bin", arr_size), arma::raw_binary);
    res.save(fmt::format("full_res_{}.bin", arr_size), arma::raw_binary);

    ExpectVectorsNear(std::span<const T>(res), std::span<const T>(expected),
                      Tol<T>()());
  }
}

TEST(FFTConvEngine, ExecutesOAConvSame) {
  using T = double;
  arma::Col<T> kernel(95, arma::fill::randn);

  auto &plans = fftconv::FFTConvEngine<T>::get_for_ksize(kernel.size());

  for (const auto arr_size : {1000, 2000, 3000}) {
    arma::Col<T> arr(arr_size, arma::fill::randn);
    arma::Col<T> expected = arma::conv(arr, kernel, "same");
    arma::Col<T> res(arr.size(), arma::fill::zeros);

    plans.oaconvolve_same(std::span<const T>(arr), std::span<const T>(kernel),
                          std::span<T>(res));

    expected.save(fmt::format("same_expected_{}.bin", arr_size),
                  arma::raw_binary);
    arr.save(fmt::format("same_arr_{}.bin", arr_size), arma::raw_binary);
    kernel.save(fmt::format("same_kernel_{}.bin", arr_size), arma::raw_binary);
    res.save(fmt::format("same_res_{}.bin", arr_size), arma::raw_binary);

    for (size_t i = 0; i < expected.size(); ++i) {
      ASSERT_NEAR(expected[i], res[i], DoubleTol)
          << "Vectors differ at index " << i;
    }
  }
}

// Test convolution
template <fftconv::Floating T, typename Func>
void execute_conv_full_correctly(Func conv_func) {
  for (int real_size = 4; real_size < 100; real_size += 9) {
    arma::Col<T> arr1(real_size, arma::fill::randn);
    arma::Col<T> arr2(real_size, arma::fill::randn);
    arma::Col<T> expected = arma::conv(arr1, arr2);
    arma::Col<T> res(arr1.size() + arr2.size() - 1, arma::fill::zeros);

    conv_func(std::span<const T>(arr1), std::span<const T>(arr2),
              std::span<T>(res));

    ExpectVectorsNear(std::span<const T>(res), std::span<const T>(expected),
                      Tol<T>()());
  }
}

TEST(ConvolveFFTW, ExecuteCorrectly) {
  execute_conv_full_correctly<double>(fftconv::convolve_fftw<double>);

  execute_conv_full_correctly<float>(fftconv::convolve_fftw<float>);
}

TEST(Convolution, ExecuteOAConvCorrectlyDifferentSizes) {
  execute_conv_full_correctly<double>(
      fftconv::oaconvolve_fftw<double, fftconv::Full>);

  execute_conv_full_correctly<float>(
      fftconv::oaconvolve_fftw<float, fftconv::Full>);
}

// Test convolution
template <fftconv::Floating T, typename Func>
void execute_oaconv_same_correctly(Func conv_func) {
  const int ksize = 65;
  const arma::Col<T> kernel(ksize, arma::fill::randn);
  for (int real_size = 200; real_size < 501; real_size += 150) {
    const arma::Col<T> arr(real_size, arma::fill::randn);
    const arma::Col<T> expected = arma::conv(arr, kernel, "same");

    arma::Col<T> res(arr.size(), arma::fill::zeros);

    conv_func(std::span<const T>(arr), std::span<const T>(kernel),
              std::span<T>(res));

    ExpectVectorsNear(std::span<const T>(res), std::span<const T>(expected),
                      Tol<T>()());
  }
}

TEST(OAConvolveSame, ExecuteCorrectly) {
  execute_oaconv_same_correctly<double>(
      fftconv::oaconvolve_fftw<double, fftconv::Same>);

  execute_oaconv_same_correctly<float>(
      fftconv::oaconvolve_fftw<float, fftconv::Same>);
}

// NOLINTEND(*-magic-numbers,*-array-index)
