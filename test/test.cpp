#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <armadillo>

#include "fftconv.hpp"        // fftw impl
// #include "fftconv_pocket.hpp" // pocketfft impl
#include "test_helpers.hpp"

using std::cout;
using std::string;
using std::vector;

// This function computes the discrete convolution of two arrays:
// result[i] = a[i]*b[0] + a[i-1]*b[1] + ... + a[0]*b[i]
vector<double> convolve_naive(const vector<double> &a,
                              const vector<double> &b) {
  const int n_a = a.size();
  const int n_b = b.size();
  const size_t conv_sz = n_a + n_b - 1;

  vector<double> result(conv_sz);

  for (int i = 0; i < conv_sz; ++i) {
    double sum = 0.0;
    for (int j = 0; j <= i; ++j)
      sum += ((j < n_a) && (i - j < n_b)) ? a[j] * b[i - j] : 0.0;
    result[i] = sum;
  }
  return result;
}

// Wrapper to prevent it from being optimized away
void convolve_armadillo(const arma::vec &a, const arma::vec &b) {
  volatile arma::vec res = arma::conv(a, b);
}

void _test(const vector<double> &a, const vector<double> &b) {
  using namespace test_helpers;
  // Ground true
  auto gt = convolve_naive(a, b);
  auto cmp = [&](const std::string &name, vector<double> res) {
    if (!cmp_vec(gt, res)) {
      cout << "gt vs " << name << "\n";
      cout << "ground truth: ";
      print_vec(gt);
      cout << name << ":";
      print_vec(res);
      return false;
    }
    return true;
  };

  bool res = true;
#define CMP(FUNC) res &= cmp(#FUNC, FUNC(a, b))

  using namespace fftconv;
  CMP(convolve_fftw);
  CMP(convolve_fftw_advanced);
  CMP(oaconvolve_fftw);
  CMP(oaconvolve_fftw_advanced);

  // CMP(convolve_pocketfft);
  // CMP(oaconvolve_pocketfft);
  // CMP(convolve_pocketfft_hdr);
  // CMP(oaconvolve_pocketfft_hdr);

  if (res)
    cout << "All tests passed.\n";
  else
    cout << "Some tests failed.\n";
}

constexpr int N_RUNS = 5000;
#define TIMEIT(FUNC, V1, V2)                                                   \
  test_helpers::_timeit(                                                       \
      #FUNC, [&]() { return FUNC(V1, V2); }, N_RUNS);

void _bench(const vector<double> &a, const vector<double> &b) {
  auto arma_a = arma::vec(a);
  auto arma_b = arma::vec(b);
  // auto result_arma = arma::conv(arma_a, arma_b);

  // TIMEIT(convolve_naive, a, b);
  // TIMEIT(fftconv::fftconv_ref, a, b);
  using namespace fftconv;
  TIMEIT(convolve_fftw, a, b);
  TIMEIT(convolve_fftw_advanced, a, b);
  TIMEIT(oaconvolve_fftw, a, b);
  TIMEIT(oaconvolve_fftw_advanced, a, b);

  // TIMEIT(convolve_pocketfft, a, b);
  // TIMEIT(oaconvolve_pocketfft, a, b);
  // TIMEIT(convolve_pocketfft_hdr, a, b);
  // TIMEIT(oaconvolve_pocketfft_hdr, a, b);

  TIMEIT(convolve_armadillo, arma_a, arma_b);
}

// Run a test case
void test_a_case(vector<double> a, vector<double> b) {
  printf("=== test case (%llu, %llu) ===\n", a.size(), b.size());
  _test(a, b);
  _bench(a, b);
}

static vector<double> get_vec(size_t size) {
  vector<double> res(size);
  for (size_t i = 0; i < size; i++)
    res[i] = (double)(std::rand() % 10);
  return res;
}

static vector<double> arange(size_t size) {
  vector<double> res(size);
  for (size_t i = 0; i < size; i++)
    res[i] = i;
  return res;
}

int main() {
  // test_a_case({0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3});
  // test_a_case(arange(25), arange(10));
  test_a_case(get_vec(1664), get_vec(65));
  test_a_case(get_vec(2816), get_vec(65));
  test_a_case(get_vec(2304), get_vec(65));
  test_a_case(get_vec(4352), get_vec(65));
  //   test_a_case(get_vec(2000), get_vec(2000));
}
