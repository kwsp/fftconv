#include <benchmark/benchmark.h>
#include <functional>
#include <vector>

#include "fftconv.h"
#include "fftconv_pocket.h"
#include <armadillo>

using std::vector;

//------------------ Helper functions

// This function computes the discrete convolution of two arrays:
// result[i] = a[i]*b[0] + a[i-1]*b[1] + ... + a[0]*b[i]
static vector<double> convolve_naive(const vector<double> &a,
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

static vector<double> get_vec(size_t size) {
  vector<double> res(size);
  for (size_t i = 0; i < size; i++) {
    res[i] = (double)(std::rand() % 10);
  }
  return res;
}

// Wrapper to prevent arma::conv from being optimized away
void convolve_armadillo(vector<double> &a, vector<double> &b) {
  const arma::vec _a(a.data(), a.size(), false, true);
  const arma::vec _b(b.data(), b.size(), false, true);
  volatile arma::vec res = arma::conv(_a, _b);
}

//------------------ Benchmarks
template <class F> void BM_ConvVec(benchmark::State &state) {
  auto a = get_vec(state.range(0));
  auto b = get_vec(state.range(1));
  for (auto _ : state) {
    F(a, b);
  }
}

#define MAKE_BM_FUNC(NAME, FUNC)                                               \
  void NAME(benchmark::State &state) {                                         \
    auto a = get_vec(state.range(0));                                          \
    auto b = get_vec(state.range(1));                                          \
    for (auto _ : state)                                                       \
      FUNC(a, b);                                                              \
  }

// First arg is signal length, second arg is kernel length
#define REGISTER_BENCHMARK(BM_FUNC)                                            \
  BENCHMARK(BM_FUNC)->ArgsProduct(                                             \
      {{1000, 1664, 2304, 2816, 3326, 4352}, {15, 35, 65, 95}});

#define REGISTER(FUNC)                                                         \
  MAKE_BM_FUNC(BM_##FUNC, FUNC)                                                \
  REGISTER_BENCHMARK(BM_##FUNC)

using namespace fftconv;

REGISTER(convolve_naive)
REGISTER(convolve_fftw)
REGISTER(oaconvolve_fftw)
REGISTER(convolve_pocketfft)
REGISTER(oaconvolve_pocketfft)
REGISTER(convolve_pocketfft_hdr)
REGISTER(oaconvolve_pocketfft_hdr)
REGISTER(convolve_armadillo)

BENCHMARK_MAIN();
