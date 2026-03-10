#include <armadillo>
#include <benchmark/benchmark.h>
#include <fftconv/aligned_vector.hpp>
#include <fftconv/fftconv.hpp>
#include <fftconv/fftw.hpp>
#include <span>

// NOLINTBEGIN(*-identifier-length)

using fftconv::AlignedVector;

// const std::vector<std::vector<int64_t>> args{
//     {{1000, 1664, 2304, 2816, 3326, 4352}, {15, 35, 65, 95}}};

const std::vector<std::vector<int64_t>> ARGS_FIR{{{2304, 4352}, {165}}};

//------------------ Helper functions

// Wrapper to prevent arma::conv from being optimized away
// Not storing results back in res.
template <fftconv::Floating T, fftconv::ConvMode Mode>
void arma_conv(std::span<const T> span1, std::span<const T> span2,
               std::span<T> span_res) {
  // NOLINTBEGIN(*-const-cast)
  const arma::Col<T> vec1(const_cast<T *>(span1.data()), span1.size(), false,
                          true);
  const arma::Col<T> vec2(const_cast<T *>(span2.data()), span2.size(), false,
                          true);
  // NOLINTEND(*-const-cast)
  if constexpr (Mode == fftconv::ConvMode::Same) {
    arma::Col<T> res = arma::conv(vec1, vec2, "same");
    benchmark::DoNotOptimize(res);
  } else {
    arma::Col<T> res = arma::conv(vec1, vec2);
    benchmark::DoNotOptimize(res);
  }
}

template <fftconv::Floating T, typename Func>
void conv_bench_full(benchmark::State &state, Func conv_func) {
  AlignedVector<T> a(state.range(0));
  AlignedVector<T> k(state.range(1));
  AlignedVector<T> out(a.size() + k.size() - 1);

  conv_func(a, k, out);
  for (auto _ : state) {
    conv_func(a, k, out);
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(T));
}

template <fftconv::Floating T, typename Func>
void conv_bench_same(benchmark::State &state, Func conv_func) {
  AlignedVector<T> a(state.range(0));
  AlignedVector<T> k(state.range(1));
  AlignedVector<T> out(a.size());

  conv_func(a, k, out);
  for (auto _ : state) {
    conv_func(a, k, out);
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(T));
}

template <fftconv::Floating T> void BM_oaconvolve(benchmark::State &state) {
  conv_bench_full<T>(state, fftconv::oaconvolve_fftw<T, fftconv::Full>);
}

BENCHMARK(BM_oaconvolve<double>)->ArgsProduct(ARGS_FIR);
BENCHMARK(BM_oaconvolve<float>)->ArgsProduct(ARGS_FIR);

template <fftconv::Floating T>
void BM_oaconvolve_same(benchmark::State &state) {
  conv_bench_same<T>(state, fftconv::oaconvolve_fftw<T, fftconv::Same>);
}
BENCHMARK(BM_oaconvolve_same<double>)->ArgsProduct(ARGS_FIR);
BENCHMARK(BM_oaconvolve_same<float>)->ArgsProduct(ARGS_FIR);

template <fftconv::Floating T> void BM_convolve(benchmark::State &state) {
  conv_bench_full<T>(state, fftconv::convolve_fftw<T>);
}
BENCHMARK(BM_convolve<double>)->ArgsProduct(ARGS_FIR);
BENCHMARK(BM_convolve<float>)->ArgsProduct(ARGS_FIR);

template <fftconv::Floating T> void BM_arma_conv(benchmark::State &state) {
  conv_bench_full<T>(state, arma_conv<T, fftconv::ConvMode::Full>);
}
BENCHMARK(BM_arma_conv<double>)->ArgsProduct(ARGS_FIR);
BENCHMARK(BM_arma_conv<float>)->ArgsProduct(ARGS_FIR);

template <fftconv::Floating T> void BM_arma_conv_same(benchmark::State &state) {
  conv_bench_same<T>(state, arma_conv<T, fftconv::ConvMode::Same>);
}
BENCHMARK(BM_arma_conv_same<double>)->ArgsProduct(ARGS_FIR);
BENCHMARK(BM_arma_conv_same<float>)->ArgsProduct(ARGS_FIR);

// NOLINTEND(*-identifier-length)

// BENCHMARK_MAIN();

int main(int argc, char **argv) {
  fftw::WisdomSetup wisdom(false);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}