// NOLINTBEGIN(*-magic-numbers,*-array-index)
#include <armadillo>
#include <complex>
#include <fftw3.h>
#include <gtest/gtest.h>
#include <span>

#include <fftconv/fftconv.hpp>

#include "test_common.hpp"

using fftconv::ConvMode;

// Test 2D convolution with known simple pattern
TEST(Convolve2D, SimplePattern) {
  // Armadillo is column-major, our code is row-major
  // So create transposed matrices for the input
  arma::Mat<double> a_t = arma::ones<arma::Mat<double>>(4, 4);
  arma::Mat<double> k_t = arma::ones<arma::Mat<double>>(2, 2);

  // Armadillo's conv2 does: conv2(a, k) = conv2(a_t, k_t).t()
  // So we need to compute expected correctly by matching memory layout
  // Create our own row-major inputs
  const size_t rows = 4, cols = 4;
  const size_t krows = 2, kcols = 2;
  const size_t orows = rows + krows - 1;
  const size_t ocols = cols + kcols - 1;

  std::vector<double> a_rowmajor(rows * cols, 1.0);
  std::vector<double> k_rowmajor(krows * kcols, 1.0);
  std::vector<double> res_rowmajor(orows * ocols, 0.0);

  fftconv::convolve_fftw_2d<double, ConvMode::Full>(
      std::span<const double>(a_rowmajor.data(), a_rowmajor.size()),
      rows, cols,
      std::span<const double>(k_rowmajor.data(), k_rowmajor.size()),
      krows, kcols,
      std::span<double>(res_rowmajor.data(), res_rowmajor.size()),
      orows, ocols);

  // Expected result for row-major 4x4 ones convolved with 2x2 ones
  const double expected_vals[5][5] = {
      {1, 2, 2, 2, 1},
      {2, 4, 4, 4, 2},
      {2, 4, 4, 4, 2},
      {2, 4, 4, 4, 2},
      {1, 2, 2, 2, 1}
  };

  for (size_t r = 0; r < orows; ++r) {
    for (size_t c = 0; c < ocols; ++c) {
      EXPECT_NEAR(res_rowmajor[r * ocols + c], expected_vals[r][c], 1e-9);
    }
  }
}

TEST(Convolve2D, Full) {
  for (int real_rows = 8; real_rows < 32; real_rows += 8) {
    for (int real_cols = 8; real_cols < 32; real_cols += 8) {
      // Create data in row-major order
      std::vector<double> a_rowmajor(real_rows * real_cols);
      std::vector<double> k_rowmajor(5 * 5);

      // Fill with random data
      for (auto& val : a_rowmajor) val = static_cast<double>(rand()) / RAND_MAX - 0.5;
      for (auto& val : k_rowmajor) val = static_cast<double>(rand()) / RAND_MAX - 0.5;

      // Create transposed for Armadillo (column-major)
      arma::Mat<double> a_t(real_cols, real_rows);
      arma::Mat<double> k_t(5, 5);

      for (int r = 0; r < real_rows; ++r) {
        for (int c = 0; c < real_cols; ++c) {
          a_t(c, r) = a_rowmajor[r * real_cols + c];
        }
      }
      for (int kr = 0; kr < 5; ++kr) {
        for (int kc = 0; kc < 5; ++kc) {
          k_t(kc, kr) = k_rowmajor[kr * 5 + kc];
        }
      }

      // Compute expected using Armadillo (transpose both, convolve, transpose result)
      arma::Mat<double> expected_t = arma::conv2(a_t, k_t);
      int orows = real_rows + 5 - 1;
      int ocols = real_cols + 5 - 1;
      std::vector<double> res_rowmajor(orows * ocols, 0.0);

      fftconv::convolve_fftw_2d<double, ConvMode::Full>(
          std::span<const double>(a_rowmajor.data(), a_rowmajor.size()),
          static_cast<size_t>(real_rows),
          static_cast<size_t>(real_cols),
          std::span<const double>(k_rowmajor.data(), k_rowmajor.size()),
          5, 5,
          std::span<double>(res_rowmajor.data(), res_rowmajor.size()),
          static_cast<size_t>(orows),
          static_cast<size_t>(ocols));

      // Convert expected to row-major
      std::vector<double> expected_rowmajor(orows * ocols);
      for (int r = 0; r < orows; ++r) {
        for (int c = 0; c < ocols; ++c) {
          expected_rowmajor[r * ocols + c] = expected_t(c, r);
        }
      }

      ExpectVectorsNear<double>(
          std::span<const double>(res_rowmajor.data(), res_rowmajor.size()),
          std::span<const double>(expected_rowmajor.data(), expected_rowmajor.size()),
          getTol<double>());
    }
  }
}

TEST(Convolve2D, Same) {
  for (int real_rows = 8; real_rows < 32; real_rows += 8) {
    for (int real_cols = 8; real_cols < 32; real_cols += 8) {
      // Create data in row-major order
      std::vector<double> a_rowmajor(real_rows * real_cols);
      std::vector<double> k_rowmajor(5 * 5);

      // Fill with random data
      for (auto& val : a_rowmajor) val = static_cast<double>(rand()) / RAND_MAX - 0.5;
      for (auto& val : k_rowmajor) val = static_cast<double>(rand()) / RAND_MAX - 0.5;

      // Create transposed for Armadillo (column-major)
      arma::Mat<double> a_t(real_cols, real_rows);
      arma::Mat<double> k_t(5, 5);

      for (int r = 0; r < real_rows; ++r) {
        for (int c = 0; c < real_cols; ++c) {
          a_t(c, r) = a_rowmajor[r * real_cols + c];
        }
      }
      for (int kr = 0; kr < 5; ++kr) {
        for (int kc = 0; kc < 5; ++kc) {
          k_t(kc, kr) = k_rowmajor[kr * 5 + kc];
        }
      }

      // Compute expected using Armadillo
      arma::Mat<double> expected_t = arma::conv2(a_t, k_t, "same");
      std::vector<double> res_rowmajor(real_rows * real_cols, 0.0);

      fftconv::convolve_fftw_2d<double, ConvMode::Same>(
          std::span<const double>(a_rowmajor.data(), a_rowmajor.size()),
          static_cast<size_t>(real_rows),
          static_cast<size_t>(real_cols),
          std::span<const double>(k_rowmajor.data(), k_rowmajor.size()),
          5, 5,
          std::span<double>(res_rowmajor.data(), res_rowmajor.size()),
          static_cast<size_t>(real_rows),
          static_cast<size_t>(real_cols));

      // Convert expected to row-major
      std::vector<double> expected_rowmajor(real_rows * real_cols);
      for (int r = 0; r < real_rows; ++r) {
        for (int c = 0; c < real_cols; ++c) {
          expected_rowmajor[r * real_cols + c] = expected_t(c, r);
        }
      }

      ExpectVectorsNear<double>(
          std::span<const double>(res_rowmajor.data(), res_rowmajor.size()),
          std::span<const double>(expected_rowmajor.data(), expected_rowmajor.size()),
          getTol<double>());
    }
  }
}

TEST(Convolve2D, Float) {
  // Create data in row-major order
  const size_t rows = 16, cols = 16;
  const size_t krows = 3, kcols = 3;

  std::vector<float> a_rowmajor(rows * cols);
  std::vector<float> k_rowmajor(krows * kcols);

  // Fill with random data
  for (auto& val : a_rowmajor) val = static_cast<float>(rand()) / RAND_MAX - 0.5f;
  for (auto& val : k_rowmajor) val = static_cast<float>(rand()) / RAND_MAX - 0.5f;

  // Create transposed for Armadillo (column-major)
  arma::Mat<float> a_t(cols, rows);
  arma::Mat<float> k_t(kcols, krows);

  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      a_t(c, r) = a_rowmajor[r * cols + c];
    }
  }
  for (size_t kr = 0; kr < krows; ++kr) {
    for (size_t kc = 0; kc < kcols; ++kc) {
      k_t(kc, kr) = k_rowmajor[kr * kcols + kc];
    }
  }

  // Full mode
  size_t orows = rows + krows - 1;
  size_t ocols = cols + kcols - 1;
  arma::Mat<float> expected_t = arma::conv2(a_t, k_t);
  std::vector<float> res_rowmajor(orows * ocols, 0.0f);

  fftconv::convolve_fftw_2d<float, ConvMode::Full>(
      std::span<const float>(a_rowmajor.data(), a_rowmajor.size()),
      rows, cols,
      std::span<const float>(k_rowmajor.data(), k_rowmajor.size()),
      krows, kcols,
      std::span<float>(res_rowmajor.data(), res_rowmajor.size()),
      orows, ocols);

  std::vector<float> expected_rowmajor(orows * ocols);
  for (size_t r = 0; r < orows; ++r) {
    for (size_t c = 0; c < ocols; ++c) {
      expected_rowmajor[r * ocols + c] = expected_t(c, r);
    }
  }

  ExpectVectorsNear<float>(
      std::span<const float>(res_rowmajor.data(), res_rowmajor.size()),
      std::span<const float>(expected_rowmajor.data(), expected_rowmajor.size()),
      getTol<float>());

  // Same mode
  std::vector<float> res_same(rows * cols, 0.0f);
  arma::Mat<float> expected_same_t = arma::conv2(a_t, k_t, "same");

  fftconv::convolve_fftw_2d<float, ConvMode::Same>(
      std::span<const float>(a_rowmajor.data(), a_rowmajor.size()),
      rows, cols,
      std::span<const float>(k_rowmajor.data(), k_rowmajor.size()),
      krows, kcols,
      std::span<float>(res_same.data(), res_same.size()),
      rows, cols);

  std::vector<float> expected_same_rowmajor(rows * cols);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      expected_same_rowmajor[r * cols + c] = expected_same_t(c, r);
    }
  }

  ExpectVectorsNear<float>(
      std::span<const float>(res_same.data(), res_same.size()),
      std::span<const float>(expected_same_rowmajor.data(), expected_same_rowmajor.size()),
      getTol<float>());
}

// Test 2D vs 1D convolution for separable kernels
TEST(Convolve2D, SeparableKernel) {
  // Create data in row-major order
  const size_t rows = 32, cols = 32;
  const size_t ksize = 5;

  std::vector<double> a_rowmajor(rows * cols);
  std::vector<double> row_k(ksize);
  std::vector<double> col_k(ksize);

  // Fill with random data
  for (auto& val : a_rowmajor) val = static_cast<double>(rand()) / RAND_MAX - 0.5;
  for (auto& val : row_k) val = static_cast<double>(rand()) / RAND_MAX - 0.5;
  for (auto& val : col_k) val = static_cast<double>(rand()) / RAND_MAX - 0.5;

  // Create 2D kernel as outer product (separable)
  std::vector<double> k_rowmajor(ksize * ksize);
  for (size_t kr = 0; kr < ksize; ++kr) {
    for (size_t kc = 0; kc < ksize; ++kc) {
      k_rowmajor[kr * ksize + kc] = col_k[kr] * row_k[kc];
    }
  }

  // Create transposed for Armadillo
  arma::Mat<double> a_t(cols, rows);
  arma::Mat<double> k_t(ksize, ksize);

  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      a_t(c, r) = a_rowmajor[r * cols + c];
    }
  }
  for (size_t kr = 0; kr < ksize; ++kr) {
    for (size_t kc = 0; kc < ksize; ++kc) {
      k_t(kc, kr) = k_rowmajor[kr * ksize + kc];
    }
  }

  arma::Mat<double> expected_t = arma::conv2(a_t, k_t, "same");
  std::vector<double> res_rowmajor(rows * cols, 0.0);

  fftconv::convolve_fftw_2d<double, ConvMode::Same>(
      std::span<const double>(a_rowmajor.data(), a_rowmajor.size()),
      rows, cols,
      std::span<const double>(k_rowmajor.data(), k_rowmajor.size()),
      ksize, ksize,
      std::span<double>(res_rowmajor.data(), res_rowmajor.size()),
      rows, cols);

  std::vector<double> expected_rowmajor(rows * cols);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      expected_rowmajor[r * cols + c] = expected_t(c, r);
    }
  }

  ExpectVectorsNear<double>(
      std::span<const double>(res_rowmajor.data(), res_rowmajor.size()),
      std::span<const double>(expected_rowmajor.data(), expected_rowmajor.size()),
      getTol<double>());
}

// NOLINTEND(*-magic-numbers,*-array-index)
