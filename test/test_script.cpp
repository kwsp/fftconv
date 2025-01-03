#include <armadillo>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <iostream>
#include <span>

#include "test_helpers.hpp"
#include <fftconv/fftconv.hpp>
#include <fftconv/fftw.hpp>

constexpr int N_RUNS = 5000;

template <fftconv::Floating T>
void bench(const arma::Col<T> &input, const arma::Col<T> &kernel) {
  arma::Col<T> output(input.size() + kernel.size() - 1);

  if constexpr (std::is_same_v<T, double>) {
    timeit(
        "convolve_fftw_ref",
        [&]() { fftconv::convolve_fftw_ref<double>(input, kernel, output); },
        N_RUNS);
  }

  timeit(
      "convolve_fftw",
      [&]() { fftconv::convolve_fftw<T>(input, kernel, output); }, N_RUNS);

  timeit(
      "oaconvolve_fftw",
      [&]() { fftconv::oaconvolve_fftw<T>(input, kernel, output); }, N_RUNS);
}

template <typename T> void run_bench() {
  constexpr std::array<std::array<size_t, 2>, 4> test_sizes{{
      {1664, 65},
      {2816, 65},
      {2304, 65},
      {4352, 65},
  }};

  T tol{};
  if constexpr (std::is_same_v<T, double>) {
    tol = 1e-9;
  } else {
    tol = 1e-5f;
  }

  for (const auto [size1, size2] : test_sizes) {
    arma::Col<T> input(size1, arma::fill::randn);
    arma::Col<T> kernel(size2, arma::fill::randn);

    fmt::println("=== test case ({}, {}) ===", size1, size2);

    arma::Col<T> expected_arma = arma::conv(input, kernel, "same");
    {
      // Test fftconv against armadillo

      arma::Col<T> res(size1, arma::fill::zeros);
      fftconv::oaconvolve_fftw<T, fftconv::Same>(std::span<const T>(input),
                                                 std::span<const T>(kernel),
                                                 std::span<T>(res));

      const auto equal = arma::approx_equal(res, expected_arma, "absdiff", tol);
      if (!equal) {
        fmt::println("Test failed.");
      } else {
        fmt::println("Test passed.");
      }
    }

    bench(input, kernel);
  }
};

auto main() -> int {
  fftw::WisdomSetup wisdom(false);

  std::cout << "Testing double ...\n";
  run_bench<double>();

  std::cout << "\nTesting float ...\n";
  run_bench<float>();

  return 0;
}
