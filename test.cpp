#include <cassert>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <omp.h>

#include "fftconv.h"
#include "test_utils.h"

using std::cout;
using std::vector;

constexpr int N_RUNS = 1000;

// This function computes the discrete convolution of two arrays:
// result[i] = a[i]*b[0] + a[i-1]*b[1] + ... + a[0]*b[i]
// a and b can be vectors of different lengths, this function is careful to
// never exceed the bounds of the vectors.
vector<double> convolve_naive(const vector<double> &a,
                              const vector<double> &b) {
  int n_a = a.size();
  int n_b = b.size();
  size_t conv_sz = n_a + n_b - 1;

  vector<double> result(conv_sz);

  for (int i = 0; i < conv_sz; ++i) {
    double sum = 0.0;
    for (int j = 0; j <= i; ++j) {
      sum += ((j < n_a) && (i - j < n_b)) ? a[j] * b[i - j] : 0.0;
    }
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

// Run a test case
void test_a_case(string fname) {
  using namespace std::chrono;
  TestCase tc(fname);
  cout << "=== " << fname << " (" << tc.v1.size() << ", " << tc.v2.size()
       << ") "
       << " ===\n";

  auto result_naive = convolve_naive(tc.v1, tc.v2);
  auto result_fft_ref = fftconv::convolve1d_ref(tc.v1, tc.v2);
  cmp_vec(result_naive, result_fft_ref);
  auto result_fft = fftconv::convolve1d(tc.v1, tc.v2);
  cmp_vec(result_naive, result_fft);

  timeit("convolve_naive", [&tc]() { convolve_naive(tc.v1, tc.v2); });
  timeit("ffconv::convolve1d_ref",
         [&tc]() { fftconv::convolve1d_ref(tc.v1, tc.v2); });
  timeit("ffconv::convolve1d", [&tc]() { fftconv::convolve1d(tc.v1, tc.v2); });
}

int main() {
  test_a_case("test_case_1.txt");
  test_a_case("test_case_2.txt");
  test_a_case("test_case_3.txt");
}
