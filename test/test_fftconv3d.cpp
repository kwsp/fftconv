// NOLINTBEGIN(*-magic-numbers,*-array-index)
#include <armadillo>
#include <complex>
#include <fftw3.h>
#include <gtest/gtest.h>
#include <span>

#include <fftconv/fftconv.hpp>

#include "test_common.hpp"

using fftconv::ConvMode;

// Helper to compute 3D convolution manually (for testing small sizes only) - ROW-MAJOR
template <typename T>
std::vector<T> manual_conv3d_rowmajor(const std::vector<T>& a, const std::array<size_t, 3>& a_dims,
                                      const std::vector<T>& b, const std::array<size_t, 3>& b_dims) {
  const size_t aDepth = a_dims[0];
  const size_t aRows = a_dims[1];
  const size_t aCols = a_dims[2];

  const size_t bDepth = b_dims[0];
  const size_t bRows = b_dims[1];
  const size_t bCols = b_dims[2];

  const size_t outDepth = aDepth + bDepth - 1;
  const size_t outRows = aRows + bRows - 1;
  const size_t outCols = aCols + bCols - 1;

  std::vector<T> result(outDepth * outRows * outCols, static_cast<T>(0));

  for (size_t d = 0; d < aDepth; ++d) {
    for (size_t r = 0; r < aRows; ++r) {
      for (size_t c = 0; c < aCols; ++c) {
        const T val = a[d * aRows * aCols + r * aCols + c];
        if (val == 0) continue;

        for (size_t bd = 0; bd < bDepth; ++bd) {
          for (size_t br = 0; br < bRows; ++br) {
            for (size_t bc = 0; bc < bCols; ++bc) {
              const size_t od = d + bd;
              const size_t orow = r + br;
              const size_t ocol = c + bc;

              result[od * outRows * outCols + orow * outCols + ocol] += val * b[bd * bRows * bCols + br * bCols + bc];
            }
          }
        }
      }
    }
  }

  return result;
}

// Helper to compare two row-major 3D arrays
template <typename T>
void ExpectCubesNearRowMajor(const std::vector<T>& actual, const std::array<size_t, 3>& actual_dims,
                              const std::vector<T>& expected, const std::array<size_t, 3>& expected_dims,
                              T tolerance) {
  ASSERT_EQ(actual_dims[0], expected_dims[0]);
  ASSERT_EQ(actual_dims[1], expected_dims[1]);
  ASSERT_EQ(actual_dims[2], expected_dims[2]);

  ExpectVectorsNear(std::span<const T>(actual.data(), actual.size()),
                    std::span<const T>(expected.data(), expected.size()),
                    tolerance);
}

TEST(Convolve3D, SimplePattern) {
  // Create data in row-major order
  const std::array<size_t, 3> a_dims = {2, 2, 2};
  const std::array<size_t, 3> k_dims = {2, 2, 2};
  std::vector<double> a_rowmajor(a_dims[0] * a_dims[1] * a_dims[2], 1.0);
  std::vector<double> k_rowmajor(k_dims[0] * k_dims[1] * k_dims[2], 1.0);

  const std::array<size_t, 3> expected_dims = {
    a_dims[0] + k_dims[0] - 1,
    a_dims[1] + k_dims[1] - 1,
    a_dims[2] + k_dims[2] - 1
  };
  std::vector<double> expected = manual_conv3d_rowmajor<double>(a_rowmajor, a_dims,
                                                                 k_rowmajor, k_dims);

  std::vector<double> res(expected_dims[0] * expected_dims[1] * expected_dims[2], 0.0);

  fftconv::convolve_fftw_3d<double, ConvMode::Full>(
      std::span<const double>(a_rowmajor.data(), a_rowmajor.size()),
      a_dims[0], a_dims[1], a_dims[2],
      std::span<const double>(k_rowmajor.data(), k_rowmajor.size()),
      k_dims[0], k_dims[1], k_dims[2],
      std::span<double>(res.data(), res.size()),
      expected_dims[0], expected_dims[1], expected_dims[2]);

  ExpectCubesNearRowMajor<double>(res, expected_dims, expected, expected_dims, getTol<double>());
}

TEST(Convolve3D, Full) {
  for (int depth : {4, 8}) {
    for (int rows : {4, 8}) {
      for (int cols : {4, 8}) {
        const std::array<size_t, 3> a_dims = {static_cast<size_t>(depth), static_cast<size_t>(rows), static_cast<size_t>(cols)};
        const std::array<size_t, 3> k_dims = {3, 3, 3};
        std::vector<double> a_rowmajor(depth * rows * cols);
        std::vector<double> k_rowmajor(3 * 3 * 3);

        // Fill with random data
        for (auto& val : a_rowmajor) val = static_cast<double>(rand()) / RAND_MAX - 0.5;
        for (auto& val : k_rowmajor) val = static_cast<double>(rand()) / RAND_MAX - 0.5;

        const std::array<size_t, 3> expected_dims = {
          a_dims[0] + k_dims[0] - 1,
          a_dims[1] + k_dims[1] - 1,
          a_dims[2] + k_dims[2] - 1
        };
        std::vector<double> expected = manual_conv3d_rowmajor<double>(a_rowmajor, a_dims,
                                                                       k_rowmajor, k_dims);

        std::vector<double> res(expected_dims[0] * expected_dims[1] * expected_dims[2], 0.0);

        fftconv::convolve_fftw_3d<double, ConvMode::Full>(
            std::span<const double>(a_rowmajor.data(), a_rowmajor.size()),
            a_dims[0], a_dims[1], a_dims[2],
            std::span<const double>(k_rowmajor.data(), k_rowmajor.size()),
            k_dims[0], k_dims[1], k_dims[2],
            std::span<double>(res.data(), res.size()),
            expected_dims[0], expected_dims[1], expected_dims[2]);

        ExpectCubesNearRowMajor<double>(res, expected_dims, expected, expected_dims, getTol<double>());
      }
    }
  }
}

TEST(Convolve3D, Same) {
  for (int depth : {5, 9}) {
    for (int rows : {5, 9}) {
      for (int cols : {5, 9}) {
        const std::array<size_t, 3> a_dims = {static_cast<size_t>(depth), static_cast<size_t>(rows), static_cast<size_t>(cols)};
        const std::array<size_t, 3> k_dims = {3, 3, 3};
        std::vector<double> a_rowmajor(depth * rows * cols);
        std::vector<double> k_rowmajor(3 * 3 * 3);

        // Fill with random data
        for (auto& val : a_rowmajor) val = static_cast<double>(rand()) / RAND_MAX - 0.5;
        for (auto& val : k_rowmajor) val = static_cast<double>(rand()) / RAND_MAX - 0.5;

        const std::array<size_t, 3> full_dims = {
          a_dims[0] + k_dims[0] - 1,
          a_dims[1] + k_dims[1] - 1,
          a_dims[2] + k_dims[2] - 1
        };

        std::vector<double> full = manual_conv3d_rowmajor<double>(a_rowmajor, a_dims,
                                                                   k_rowmajor, k_dims);

        // Extract "same" region (center)
        const size_t padD = k_dims[0] / 2;
        const size_t padR = k_dims[1] / 2;
        const size_t padC = k_dims[2] / 2;
        std::vector<double> expected_rowmajor(a_dims[0] * a_dims[1] * a_dims[2], 0.0);

        for (size_t d = 0; d < a_dims[0]; ++d) {
          for (size_t r = 0; r < a_dims[1]; ++r) {
            for (size_t c = 0; c < a_dims[2]; ++c) {
              const size_t full_d = d + padD;
              const size_t full_r = r + padR;
              const size_t full_c = c + padC;

              expected_rowmajor[d * a_dims[1] * a_dims[2] + r * a_dims[2] + c] =
                  full[full_d * full_dims[1] * full_dims[2] + full_r * full_dims[2] + full_c];
            }
          }
        }

        std::vector<double> res(a_dims[0] * a_dims[1] * a_dims[2], 0.0);

        fftconv::convolve_fftw_3d<double, ConvMode::Same>(
            std::span<const double>(a_rowmajor.data(), a_rowmajor.size()),
            a_dims[0], a_dims[1], a_dims[2],
            std::span<const double>(k_rowmajor.data(), k_rowmajor.size()),
            k_dims[0], k_dims[1], k_dims[2],
            std::span<double>(res.data(), res.size()),
            a_dims[0], a_dims[1], a_dims[2]);

        ExpectCubesNearRowMajor<double>(res, a_dims, expected_rowmajor, a_dims, getTol<double>());
      }
    }
  }
}

TEST(Convolve3D, Float) {
  const std::array<size_t, 3> a_dims = {6, 6, 6};
  const std::array<size_t, 3> k_dims = {3, 3, 3};
  std::vector<float> a_rowmajor(a_dims[0] * a_dims[1] * a_dims[2]);
  std::vector<float> k_rowmajor(k_dims[0] * k_dims[1] * k_dims[2]);

  // Fill with random data
  for (auto& val : a_rowmajor) val = static_cast<float>(rand()) / RAND_MAX - 0.5f;
  for (auto& val : k_rowmajor) val = static_cast<float>(rand()) / RAND_MAX - 0.5f;

  const std::array<size_t, 3> expected_dims = {
    a_dims[0] + k_dims[0] - 1,
    a_dims[1] + k_dims[1] - 1,
    a_dims[2] + k_dims[2] - 1
  };
  std::vector<float> expected = manual_conv3d_rowmajor<float>(a_rowmajor, a_dims,
                                                               k_rowmajor, k_dims);

  std::vector<float> res(expected_dims[0] * expected_dims[1] * expected_dims[2], 0.0f);

  fftconv::convolve_fftw_3d<float, ConvMode::Full>(
      std::span<const float>(a_rowmajor.data(), a_rowmajor.size()),
      a_dims[0], a_dims[1], a_dims[2],
      std::span<const float>(k_rowmajor.data(), k_rowmajor.size()),
      k_dims[0], k_dims[1], k_dims[2],
      std::span<float>(res.data(), res.size()),
      expected_dims[0], expected_dims[1], expected_dims[2]);

  ExpectCubesNearRowMajor<float>(res, expected_dims, expected, expected_dims, getTol<float>());
}

TEST(Convolve3D, SingleElementKernel) {
  const std::array<size_t, 3> a_dims = {4, 4, 4};
  const std::array<size_t, 3> k_dims = {1, 1, 1};
  std::vector<double> a_rowmajor(a_dims[0] * a_dims[1] * a_dims[2]);
  std::vector<double> k_rowmajor(k_dims[0] * k_dims[1] * k_dims[2], 1.0);

  // Fill with random data
  for (auto& val : a_rowmajor) val = static_cast<double>(rand()) / RAND_MAX - 0.5;

  std::vector<double> res(a_dims[0] * a_dims[1] * a_dims[2], 0.0);

  fftconv::convolve_fftw_3d<double, ConvMode::Same>(
      std::span<const double>(a_rowmajor.data(), a_rowmajor.size()),
      a_dims[0], a_dims[1], a_dims[2],
      std::span<const double>(k_rowmajor.data(), k_rowmajor.size()),
      k_dims[0], k_dims[1], k_dims[2],
      std::span<double>(res.data(), res.size()),
      a_dims[0], a_dims[1], a_dims[2]);

  ExpectCubesNearRowMajor<double>(res, a_dims, a_rowmajor, a_dims, getTol<double>());
}

// NOLINTEND(*-magic-numbers,*-array-index)
