#include <armadillo>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <iostream>
#include <kfr/all.hpp>
#include <span>

#include "fftconv.hpp" // fftw impl
// #include "fftconv_pocket.hpp" // pocketfft impl
#include "test_helpers.hpp"

constexpr int N_RUNS = 5000;

template <fftconv::FloatOrDouble T>
void bench(const arma::Col<T> &vec1, const arma::Col<T> &vec2) {
  arma::Col<T> _res(vec1.size() + vec2.size() - 1);

  const std::span v1s(vec1);
  const std::span v2s(vec2);
  const std::span res(_res);

  if constexpr (std::is_same_v<T, double>) {
    timeit(
        "convolve_fftw_ref",
        [&]() { fftconv::convolve_fftw_ref<double>(vec1, vec2, res); }, N_RUNS);
  }

  timeit(
      "convolve_fftw", [&]() { fftconv::convolve_fftw<T>(vec1, vec2, res); },
      N_RUNS);

  timeit(
      "oaconvolve_fftw", [&]() { fftconv::oaconvolve_fftw<T>(v1s, v2s, res); },
      N_RUNS);

  // timeit(
  //     "oaconvolve_fftw_advanced",
  //     [&]() { fftconv::oaconvolve_fftw_advanced(v1s, v2s, res); }, N_RUNS);

  kfr::univector<T> expected(vec1.size());
  timeit(
      "kfr_fir",
      [&]() {
        // Test fftconv against KFR
        auto inData = kfr::make_univector(vec1.memptr(), vec1.size());
        auto taps = kfr::make_univector(vec2.memptr(), vec2.size());

        kfr::filter_fir<T> filter(taps);
        // kfr::convolve_filter<T> filter(taps);
        filter.apply(expected, inData);
      },
      N_RUNS);
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
    arma::Col<T> arr1(size1, arma::fill::randn);
    arma::Col<T> arr2(size2, arma::fill::randn);

    fmt::println("=== test case ({}, {}) ===", size1, size2);

    arma::Col<T> expected_arma = arma::conv(arr1, arr2, "same");
    {
      // Test fftconv against armadillo

      arma::Col<T> res(size1, arma::fill::zeros);
      fftconv::oaconvolve_fftw_same(std::span<const T>(arr1),
                                    std::span<const T>(arr2),
                                    std::span<T>(res));

      const auto equal = arma::approx_equal(res, expected_arma, "absdiff", tol);
      if (!equal) {
        fmt::println("Test failed.");
      } else {
        fmt::println("Test passed.");
      }
    }

    {
      // Test fftconv against KFR
      auto inData = kfr::make_univector(arr1.memptr(), arr1.size());
      auto taps = kfr::make_univector(arr2.memptr(), arr2.size());
      kfr::univector<T> expected(size1);

      kfr::filter_fir<T> filter(taps);
      filter.apply(expected, inData);

      arma::Col<T> expected_kfr(expected.data(), expected.size(), false, true);

      expected_kfr.save("expected_kfr.bin", arma::raw_binary);
      expected_arma.save("expected_arma.bin", arma::raw_binary);

      const auto equal =
          arma::approx_equal(expected_kfr, expected_arma, "absdiff", tol);
      if (!equal) {
        fmt::println("Test failed.");
      } else {
        fmt::println("Test passed.");
      }
    }

    bench(arr1, arr2);
  }
};

auto main() -> int {

  std::cout << "Testing double ...\n";
  run_bench<double>();

  std::cout << "\nTesting float ...\n";
  run_bench<float>();

  return 0;
}
