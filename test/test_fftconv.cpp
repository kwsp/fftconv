// NOLINTBEGIN(*-magic-numbers,*-array-index)
#include <armadillo>
#include <array>
#include <complex>
#include <fftw3.h>
#include <gtest/gtest.h>
#include <span>
#include <vector>

#include "fftconv.hpp"

#include "test_common.hpp"

// Test internal utils
// Assuming the function is defined here or included from another header
TEST(CopyToPaddedBufferTest, SameTypeLargerDestination) {
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

TEST(CopyToPaddedBufferTest, DifferentTypeConversion) {
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

TEST(CopyToPaddedBufferTest, SameSizeNoPaddingNeeded) {
  const std::vector<int> src = {4, 5, 6};
  std::vector<int> dst(3, -1); // Initialize with a non-zero value

  fftconv::internal::copy_to_padded_buffer(std::span(src), std::span(dst));

  EXPECT_EQ(dst[0], 4);
  EXPECT_EQ(dst[1], 5);
  EXPECT_EQ(dst[2], 6);
  // No padding necessary, so no checks for zeros beyond this point
}

// Test FFTW buffer management
TEST(FFTWBuffer, AllocatesAndFreesMemoryCorrectly) {
  const size_t size = 128;

  {
    fftconv::fftw_buffer<double> buffer(size);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), size);

    // Verify can access last element without crash
    EXPECT_NO_THROW(buffer[size - 1] = 0.0);
  }
  {
    fftconv::fftw_buffer<float> buffer(size);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), size);

    // Verify can access last element without crash
    EXPECT_NO_THROW(buffer[size - 1] = 0.0);
  }

  {
    fftconv::fftw_buffer<std::complex<double>> buffer(size);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), size);

    // Verify can access last element without crash
    EXPECT_NO_THROW(const auto last = buffer[size - 1]);
  }

  {
    fftconv::fftw_buffer<std::complex<double>> buffer(size);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), size);

    // Verify can access last element without crash
    EXPECT_NO_THROW(const auto last = buffer[size - 1]);
  }
}

// Test for correct allocation and size reporting
TEST(FFTWBuffer, CorrectAllocationAndSize) {
  const auto sizes = std::to_array({16, 128, 1024, 2048});

  for (auto size : sizes) {
    fftconv::fftw_buffer<float> buffer(size);
    EXPECT_EQ(buffer.size(), size)
        << "Buffer size should match the requested size.";
    EXPECT_NE(buffer.data(), nullptr)
        << "Data pointer should not be null after allocation.";
  }
}

TEST(ElementwiseMultiplyTest, CorrectMultiplication) {
  // Setup test data
  fftconv::fftw_buffer<std::complex<double>> complex1{{{1, 2}, {3, 4}}};

  fftconv::fftw_buffer<std::complex<double>> complex2{{{5, 6}, {7, 8}}};

  fftconv::fftw_buffer<std::complex<double>> result(2);

  // Perform element-wise multiplication
  fftconv::internal::multiply_cx<double>(complex1, complex2, result);

  // Verify the results
  EXPECT_DOUBLE_EQ(result[0].real(), -7);  // (1*5 - 2*6) = 5 - 12 = -7
  EXPECT_DOUBLE_EQ(result[0].imag(), 16);  // (2*5 + 1*6) = 10 + 6 = 16
  EXPECT_DOUBLE_EQ(result[1].real(), -11); // (3*7 - 4*8) = 21 - 32 = -11
  EXPECT_DOUBLE_EQ(result[1].imag(), 52);  // (4*7 + 3*8) = 28 + 24 = 52
}

TEST(ElementwiseMultiplyTest, HandlesDifferentSizes) {
  // Setup test data
  fftconv::fftw_buffer<std::complex<double>> complex1{{
      {1, 1},
      {2, 2},
      {3, 3},
  }};

  fftconv::fftw_buffer<std::complex<double>> complex2{{
      {1, -1},
      {1, -1},
  }};

  fftconv::fftw_buffer<std::complex<double>> result(3);

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

// Test FFT plan creation
TEST(FFTPlans1D, CreatesPlansCorrectly) {
  const size_t size = 128;

  {
    fftconv::fftw_buffer<float> real(size);
    fftconv::fftw_buffer<std::complex<float>> complex(size / 2 + 1);
    fftconv::internal::Plans1d<float> plans(real, complex);
    EXPECT_NE(plans.plan_forward.plan, nullptr);
    EXPECT_NE(plans.plan_backward.plan, nullptr);
  }
  {
    fftconv::fftw_buffer<double> real(size);
    fftconv::fftw_buffer<std::complex<double>> complex(size / 2 + 1);
    fftconv::internal::Plans1d<double> plans(real, complex);
    EXPECT_NE(plans.plan_forward.plan, nullptr);
    EXPECT_NE(plans.plan_backward.plan, nullptr);
  }
}

TEST(FFTWPlans1DFloatTest, ForwardTransform) {
  using traits = fftw::Traits<float>;
  constexpr size_t size = 16; // Example size
  constexpr size_t size_cx = size / 2 + 1;

  // Initialize real_input
  fftconv::fftw_buffer<traits::Real> real_buffer{{
      0.422464,
      0.32053405,
      0.3295821,
      0.58121234,
      0.38011023,
      0.07002003,
      0.9369484,
      0.8599247,
      0.3834778,
      0.44842938,
      0.846668,
      0.2881548,
      0.28737324,
      0.55736756,
      0.33394003,
      0.19078194,
  }};
  fftconv::fftw_buffer<traits::Cx> complex_buffer(size_cx);
  fftconv::internal::Plans1d<float> fft_plans(real_buffer, complex_buffer);

  // Expected fft
  using namespace std::complex_literals;
  std::array<std::complex<traits::Real>, size_cx> expected{
      {{7.23698863F + 0.if},
       {-1.19075913F - 0.18111922if},
       {0.36679393F + 0.12275547if},
       {-0.19500105F - 0.54241835if},
       {-0.97371325F + 0.52372275if},
       {1.85702601F - 0.60637959if},
       {-0.08987725F - 0.06652122if},
       {-0.315321F + 0.1258675if},
       {0.60413907F + 0.if}}};

  // Execute the forward FFT
  fft_plans.forward(real_buffer, complex_buffer);

  for (int i = 0; i < size_cx; i++) {
    EXPECT_NEAR(complex_buffer[i].real(), expected[i].real(), FloatTol);
    EXPECT_NEAR(complex_buffer[i].imag(), expected[i].imag(), FloatTol);
  }
}

template <fftconv::FloatOrDouble Real> struct Tol {};
template <> struct Tol<float> {
  auto operator()() { return FloatTol; }
};
template <> struct Tol<double> {
  auto operator()() { return DoubleTol; }
};

// Test convolution
template <fftconv::FloatOrDouble Real, typename Func>
void execute_conv_correctly_different_sizes(Func conv_func) {
  for (int real_size = 4; real_size < 100; real_size += 9) {
    arma::Col<Real> arr1(real_size, arma::fill::randn);
    arma::Col<Real> arr2(real_size, arma::fill::randn);
    arma::Col<Real> expected = arma::conv(arr1, arr2);
    arma::Col<Real> res(arr1.size() + arr2.size() - 1, arma::fill::zeros);

    conv_func(std::span<const Real>(arr1), std::span<const Real>(arr2),
              std::span<Real>(res));

    ExpectVectorsNear(std::span<const Real>(res),
                      std::span<const Real>(expected), Tol<Real>()());
  }
}

TEST(Convolution, ExecutesOAConvolutionCorrectlySameKernel) {
  using T = double;
  arma::Col<T> kernel(65, arma::fill::randn);

  auto &plans =
      fftconv::fftconv_plans<T>::get_for_kernel(std::span<const T>(kernel));

  for (const auto arr_size : {1000, 2000, 3000}) {
    arma::Col<T> arr(arr_size, arma::fill::randn);
    arma::Col<T> expected = arma::conv(arr, kernel);
    arma::Col<T> res(arr.size() + kernel.size() - 1, arma::fill::zeros);

    plans.oaconvolve(std::span<const T>(arr), std::span<const T>(kernel),
                     std::span<T>(res));

    ExpectVectorsNear(std::span<const T>(res), std::span<const T>(expected),
                      Tol<T>()());
  }
}

TEST(Convolution, ExecuteConvCorrectlyDifferentSizes) {
  execute_conv_correctly_different_sizes<double>(
      fftconv::convolve_fftw<double>);

  execute_conv_correctly_different_sizes<float>(fftconv::convolve_fftw<float>);
}

TEST(Convolution, ExecuteOAConvCorrectlyDifferentSizes) {
  execute_conv_correctly_different_sizes<double>(
      fftconv::oaconvolve_fftw<double>);

  execute_conv_correctly_different_sizes<float>(
      fftconv::oaconvolve_fftw<float>);
}

// Test convolution
template <fftconv::FloatOrDouble Real, typename Func>
void execute_oaconv_same_correctly(Func conv_func) {
  const int ksize = 65;
  const arma::Col<Real> kernel(ksize, arma::fill::randn);
  for (int real_size = 200; real_size < 501; real_size += 150) {
    const arma::Col<Real> arr(real_size, arma::fill::randn);
    const arma::Col<Real> expected = arma::conv(arr, kernel, "same");

    arma::Col<Real> res(arr.size(), arma::fill::zeros);

    conv_func(std::span<const Real>(arr), std::span<const Real>(kernel),
              std::span<Real>(res));

    ExpectVectorsNear(std::span<const Real>(res),
                      std::span<const Real>(expected), Tol<Real>()());
  }
}

TEST(OAConvolveSame, ExecuteCorrectly) {
  execute_oaconv_same_correctly<double>(fftconv::oaconvolve_fftw_same<double>);

  execute_oaconv_same_correctly<float>(fftconv::oaconvolve_fftw_same<float>);
}

// NOLINTEND(*-magic-numbers,*-array-index)
