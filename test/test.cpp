#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <armadillo>

#include "fftconv.hpp" // fftw impl
// #include "fftconv_pocket.hpp" // pocketfft impl
#include "test_helpers.hpp"

using std::cout;
using std::string;
using std::vector;

// This function computes the discrete convolution of two arrays:
// result[i] = vec1[i]*vec2[0] + vec1[i-1]*vec2[1] + ... + vec1[0]*vec2[i]
auto convolve_naive(const vector<double> &vec1, const vector<double> &vec2)
    -> vector<double> {
  const size_t n_a = vec1.size();
  const size_t n_b = vec2.size();
  const size_t conv_sz = n_a + n_b - 1;

  vector<double> result(conv_sz);

  for (int i = 0; i < conv_sz; ++i) {
    double sum = 0.0;
    for (int j = 0; j <= i; ++j) {
      sum += ((j < n_a) && (i - j < n_b)) ? vec1[j] * vec2[i - j] : 0.0;
    }
    result[i] = sum;
  }
  return result;
}

// Wrapper to prevent it from being optimized away
void convolve_armadillo(const arma::vec &vec1, const arma::vec &vec2) {
  volatile arma::vec res = arma::conv(vec1, vec2);
}

void test(const vector<double> &vec1, const vector<double> &vec2) {
  using namespace test_helpers;
  // Ground true
  auto ground_truth = convolve_naive(vec1, vec2);
  auto cmp = [&](const std::string &name, const vector<double> &res) {
    if (!cmp_vec(ground_truth, res)) {
      cout << "ground_truth vs " << name << "\n";
      cout << "ground truth: ";
      print_vec(ground_truth);
      cout << name << ":";
      print_vec(res);
      return false;
    }
    return true;
  };

  bool passed = true;
#define CMP(FUNC) passed = cmp(#FUNC, FUNC(vec1, vec2))

  using namespace fftconv;
  CMP(convolve_fftw);
  CMP(convolve_fftw_advanced);
  CMP(oaconvolve_fftw);
  CMP(oaconvolve_fftw_advanced);

  // CMP(convolve_pocketfft);
  // CMP(oaconvolve_pocketfft);
  // CMP(convolve_pocketfft_hdr);
  // CMP(oaconvolve_pocketfft_hdr);

  if (passed) {
    cout << "All tests passed.\n";
  } else {
    cout << "Some tests failed.\n";
  }
}

constexpr int N_RUNS = 5000;
#define TIMEIT(FUNC, V1, V2)                                                   \
  test_helpers::_timeit(                                                       \
      #FUNC, [&]() { return FUNC(V1, V2); }, N_RUNS);

void bench(const vector<double> &vec1, const vector<double> &vec2) {
  auto arma_a = arma::vec(vec1);
  auto arma_b = arma::vec(vec2);
  // auto result_arma = arma::conv(arma_a, arma_b);

  // TIMEIT(convolve_naive, vec1, vec2);
  // TIMEIT(fftconv::fftconv_ref, vec1, vec2);
  using namespace fftconv;
  TIMEIT(convolve_fftw, vec1, vec2);
  TIMEIT(convolve_fftw_advanced, vec1, vec2);
  TIMEIT(oaconvolve_fftw, vec1, vec2);
  TIMEIT(oaconvolve_fftw_advanced, vec1, vec2);

  // TIMEIT(convolve_pocketfft, vec1, vec2);
  // TIMEIT(oaconvolve_pocketfft, vec1, vec2);
  // TIMEIT(convolve_pocketfft_hdr, vec1, vec2);
  // TIMEIT(oaconvolve_pocketfft_hdr, vec1, vec2);

  TIMEIT(convolve_armadillo, arma_a, arma_b);
}

// Run vec1 test case
void test_a_case(const vector<double> &vec1, const vector<double> &vec2) {
  printf("=== test case (%lu, %lu) ===\n", vec1.size(), vec2.size());
  test(vec1, vec2);
  bench(vec1, vec2);
}

static auto get_vec(size_t size) -> vector<double> {
  std::random_device rand_device;
  std::default_random_engine engine(rand_device());
  std::uniform_real_distribution<double> dist(-1, 1);

  vector<double> res(size);
  for (size_t i = 0; i < size; i++) {
    res[i] = dist(engine);
  }
  return res;
}

template <typename T> auto arange(size_t size) -> vector<T> {
  vector<double> res(size);
  for (size_t i = 0; i < size; i++) {
    res[i] = static_cast<T>(i);
  }
  return res;
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

  for (const auto &pair : test_sizes) {
    test_a_case(get_vec(pair[0]), get_vec(pair[1]));
  }
}
