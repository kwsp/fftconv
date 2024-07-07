#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>

#include <armadillo>

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

  timeit(
      "convolve_fftw_ref", [&]() { fftconv::convolve_fftw_ref(v1s, v2s, res); },
      N_RUNS);

  timeit(
      "convolve_fftw", [&]() { fftconv::convolve_fftw(v1s, v2s, res); },
      N_RUNS);

  timeit(
      "oaconvolve_fftw", [&]() { fftconv::oaconvolve_fftw(v1s, v2s, res); },
      N_RUNS);

  // timeit(
  //     "oaconvolve_fftw_advanced",
  //     [&]() { fftconv::oaconvolve_fftw_advanced(v1s, v2s, res); }, N_RUNS);

  // TIMEIT(convolve_pocketfft, vec1, vec2);
  // TIMEIT(oaconvolve_pocketfft, vec1, vec2);
  // TIMEIT(convolve_pocketfft_hdr, vec1, vec2);
  // TIMEIT(oaconvolve_pocketfft_hdr, vec1, vec2);
}

auto main() -> int {
  // test_a_case({0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3});
  // test_a_case(arange(25), arange(10));

  constexpr std::array<std::array<size_t, 2>, 4> test_sizes{{
      {1664, 65},
      {2816, 65},
      {2304, 65},
      {4352, 65},
  }};

  using T = double;
  for (const auto &pair : test_sizes) {
    // for (int real_size = 4; real_size < 100; real_size += 9) {
    // arma::Col<T> arr1(real_size, arma::fill::randn);
    // arma::Col<T> arr2(real_size, arma::fill::randn);

    arma::Col<T> arr1(pair[0], arma::fill::randn);
    arma::Col<T> arr2(pair[1], arma::fill::randn);
    arma::Col<T> expected = arma::conv(arr1, arr2);
    arma::Col<T> res(arr1.size() + arr2.size() - 1, arma::fill::zeros);

    std::cout << "=== test case (" << arr1.size() << ", " << arr2.size()
              << ") ===\n";

    fftconv::oaconvolve_fftw(std::span<const T>(arr1), std::span<const T>(arr2),
                             std::span<T>(res));

    auto equal = arma::approx_equal(res, expected, "absdiff", 1e-9);
    if (!equal) {
      std::cout << "Test failed.\n";
    } else {
      std::cout << "Test passed.\n";
    }

    bench(arr1, arr2);
  }

  return 0;
}
