#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "fftconv.h"
#include <armadillo>

using std::cout;
using std::string;
using std::vector;

constexpr int N_RUNS = 5000;
constexpr double err = 1e-6;

// This function computes the discrete convolution of two arrays:
// result[i] = a[i]*b[0] + a[i-1]*b[1] + ... + a[0]*b[i]
vector<double> convolve_naive(const vector<double> &a,
                              const vector<double> &b) {
  int n_a = a.size();
  int n_b = b.size();
  size_t conv_sz = n_a + n_b - 1;

  vector<double> result(conv_sz);

  for (int i = 0; i < conv_sz; ++i) {
    double sum = 0.0;
    for (int j = 0; j <= i; ++j)
      sum += ((j < n_a) && (i - j < n_b)) ? a[j] * b[i - j] : 0.0;
    result[i] = sum;
  }
  return result;
}

// Run the `callable` `n_runs` times and print the time.
void timeit(string name, std::function<void()> callable, int n_runs = N_RUNS) {
  using namespace std::chrono;
  cout << "    (" << n_runs << " runs) " << name;
  auto start = high_resolution_clock::now();
  for (int i = 0; i < N_RUNS; i++)
    callable();
  auto elapsed =
      duration_cast<milliseconds>(high_resolution_clock::now() - start);
  cout << " took " << elapsed.count() << "ms\n";
}

// Compare two vectors
template <class T> bool cmp_vec(const vector<T> &a, const vector<T> &b) {
  assert(a.size() == b.size());
  for (auto i = 0; i < a.size(); ++i)
    if (abs(a[i] - b[i]) > err) {
      printf("Vectors are different: v1[%d]=%f, v2[%d]=%f\n", i, a[i], i, b[i]);
      return false;
    }
  printf("Vectors are equal.\n");
  return true;
}

// Wrapper to prevent it from being optimized away
void arma_conv(const arma::vec &a, const arma::vec &b) {
  volatile arma::vec res = arma::conv(a, b);
}

// Run a test case
void test_a_case(vector<double> a, vector<double> b) {
  using namespace std::chrono;
  printf("=== test case (%lu, %lu) ===\n", a.size(), b.size());

  auto result_naive = convolve_naive(a, b);
  auto result_fft_ref = fftconv::convolve1d_ref(a, b);
  cmp_vec(result_naive, result_fft_ref);
  auto result_fft = fftconv::convolve1d(a, b);
  cmp_vec(result_naive, result_fft);

  auto arma_v1 = arma::vec(a);
  auto arma_v2 = arma::vec(b);
  auto result_arma = arma::conv(arma_v1, arma_v2);

  // timeit("convolve_naive", [&]() { return convolve_naive(a, b); });
  timeit("ffconv::convolve1d_ref",
         [&]() { return fftconv::convolve1d_ref(a, b); });
  timeit("ffconv::convolve1d", [&]() { return fftconv::convolve1d(a, b); });
  timeit("arma::conv", [&]() { return arma_conv(arma_v1, arma_v2); });
}

vector<double> get_vec(size_t size) {
  vector<double> res(size);
  for (size_t i = 0; i < size; i++) {
    res[i] = (double)(std::rand() % 256);
  }
  return res;
}

int main() {
  test_a_case(get_vec(1664), get_vec(65));
  test_a_case(get_vec(2816), get_vec(65));
  // test_a_case(get_vec(2000), get_vec(2000));
}
