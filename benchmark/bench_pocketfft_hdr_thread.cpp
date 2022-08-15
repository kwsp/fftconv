#include <benchmark/benchmark.h>
#include <cstdlib>
#include <functional>
#include <vector>

#include "fftconv_pocket.h"

using std::vector;

template <class T> static vector<T> get_vec(size_t size) {
  vector<T> res(size);
  for (size_t i = 0; i < size; i++) {
    res[i] = static_cast<T>(std::rand() % 10);
  }
  return res;
}

void BM_convolve_pocketfft_hdr_thread(benchmark::State &state) {
  const auto a = get_vec<double>(state.range(0));
  const auto b = get_vec<double>(state.range(1));
  const size_t nthreads = state.range(2);
  for (auto _ : state)
    fftconv::convolve_pocketfft_hdr(a, b, nthreads);
}

BENCHMARK(BM_convolve_pocketfft_hdr_thread)
    ->ArgsProduct({{1000, 1664, 2304, 2816, 3326, 4352}, {65}, {1, 2, 4, 8}});

BENCHMARK_MAIN();
