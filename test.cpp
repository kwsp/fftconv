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
template <class T>
bool cmp_vec(T *a, const size_t l_a, T *b, const size_t l_b) {
  assert(l_a == l_b);
  for (auto i = 0; i < l_a; ++i)
    if (abs(a[i] - b[i]) > err) {
      printf("Vectors are different: v1[%d]=%f, v2[%d]=%f\n", i, a[i], i, b[i]);
      return false;
    }
  printf("Vectors are equal.\n");
  return true;
}
template <class T> bool cmp_vec(const vector<T> &a, const vector<T> &b) {
  return cmp_vec(a.data(), a.size(), b.data(), b.size());
}
bool cmp_vec(const arma::vec &a, const arma::vec &b) {
  return cmp_vec(a.memptr(), a.size(), b.memptr(), b.size());
}
bool cmp_vec(const vector<double> &a, const arma::vec &b) {
  return cmp_vec(a.data(), a.size(), b.memptr(), b.size());
}
bool cmp_vec(const arma::vec &a, const vector<double> &b) {
  return cmp_vec(a.memptr(), a.size(), b.data(), b.size());
}

// Wrapper to prevent it from being optimized away
void arma_conv(const arma::vec &a, const arma::vec &b) {
  volatile arma::vec res = arma::conv(a, b);
}

void make_same_length(vector<double> &vec, size_t len) {
  assert(vec.size() >= len);
  auto offset = (vec.size() - len) / 2;
  vec.erase(vec.begin(), vec.begin() + offset);
  vec.erase(vec.end() - offset, vec.end());
}

template <class T> void print_vec(T *arr, size_t sz) {
  for (int i = 0; i < sz; ++i)
    std::cout << std::setw(4) << arr[i] << ", ";
  std::cout << "\n";
}

template <class T> void print_vec(std::vector<T> vec) {
  print_vec(vec.data(), vec.size());
}

void print_vec(arma::vec vec) { print_vec(vec.memptr(), vec.size()); }

// Run a test case
void test_a_case(vector<double> a, vector<double> b) {
  using namespace std::chrono;
  printf("=== test case (%lu, %lu) ===\n", a.size(), b.size());

  // Ground true
  auto result_naive = convolve_naive(a, b);

  //auto result_fft_ref = fftconv::convolve1d_ref(a, b);
  //std::cout << "naive vs fft_ref ";
  //if (!cmp_vec(result_naive, result_fft_ref)) {
    //std::cout << "naive   : ";
    //print_vec(result_naive);
    //std::cout << "fft_ref : ";
    //print_vec(result_fft_ref);
  //}

  auto result_fft = fftconv::convolve1d(a, b);
  std::cout << "naive vs fft ";
  if (!cmp_vec(result_naive, result_fft)) {
    std::cout << "naive     : ";
    print_vec(result_naive);
    std::cout << "convolve1d: ";
    print_vec(result_fft);
  }

  auto arma_a = arma::vec(a);
  auto arma_b = arma::vec(b);
  auto result_arma = arma::conv(arma_a, arma_b);

  auto result_fftfilt = fftconv::fftfilt(a, b);
  std::cout << "naive vs fftfilt ";
  if (!cmp_vec(result_naive, result_fftfilt)) {
    std::cout << "naive  : ";
    print_vec(result_naive);
    std::cout << "fftfilt: ";
    print_vec(result_fftfilt);
  }

  // timeit("convolve_naive", [&]() { return convolve_naive(a, b); });
  //timeit("fftconv::convolve1d_ref",
         //[&]() { return fftconv::convolve1d_ref(a, b); });
  timeit("fftconv::convolve1d", [&]() { return fftconv::convolve1d(a, b); });
  timeit("fftconv::fftfilt",
         [&]() { return fftconv::fftfilt(a, b); });
  timeit("arma::conv", [&]() { return arma_conv(arma_a, arma_b); });
}

vector<double> get_vec(size_t size) {
  vector<double> res(size);
  for (size_t i = 0; i < size; i++) {
    res[i] = (double)(std::rand() % 10);
  }
  return res;
}

int main() {
  // test_a_case(get_vec(22), get_vec(10));
  // test_a_case({0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3});
  test_a_case(get_vec(8), get_vec(4));
  test_a_case(get_vec(1664), get_vec(65));
  test_a_case(get_vec(2816), get_vec(65));
  test_a_case(get_vec(2304), get_vec(65));
  test_a_case(get_vec(4352), get_vec(65));
  // test_a_case(get_vec(2000), get_vec(2000));
}
