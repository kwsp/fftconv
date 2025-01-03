#include <armadillo>
#include <benchmark/benchmark.h>
#include <span>

#include <fftconv/fftconv.hpp>
#include <fftconv/fftw.hpp>

// NOLINTBEGIN(*-identifier-length)

//------------------ Helper functions

// Wrapper to prevent arma::conv from being optimized away
// Not storing results back in res.
template <fftconv::Floating T>
void arma_conv_full(std::span<const T> span1, std::span<const T> span2,
                    std::span<T> span_res) {
  // NOLINTBEGIN(*-const-cast)
  const arma::Col<T> vec1(const_cast<T *>(span1.data()), span1.size(), false,
                          true);
  const arma::Col<T> vec2(const_cast<T *>(span2.data()), span2.size(), false,
                          true);
  // NOLINTEND(*-const-cast)
  volatile arma::Col<T> res = arma::conv(vec1, vec2);
}

template <fftconv::Floating T, typename Func>
void conv_bench_full(benchmark::State &state, Func conv_func) {
  arma::Col<T> input(state.range(0), arma::fill::randn);
  arma::Col<T> kernel(state.range(1), arma::fill::randn);
  arma::Col<T> output(input.size() + kernel.size() - 1);

  for (auto _ : state) {
    conv_func(input, kernel, output);
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(T));
}

// Wrapper to prevent arma::conv from being optimized away
// Not storing results back in res.
template <fftconv::Floating T>
void arma_conv_same(const std::span<const T> span1,
                    const std::span<const T> span2, std::span<T> span_res) {
  // NOLINTBEGIN(*-const-cast)
  const arma::Col<T> vec1(const_cast<T *>(span1.data()), span1.size(), false,
                          true);
  const arma::Col<T> vec2(const_cast<T *>(span2.data()), span2.size(), false,
                          true);
  // NOLINTEND(*-const-cast)
  volatile arma::Col<T> res = arma::conv(vec1, vec2, "same");
}

template <fftconv::Floating T, typename Func>
void conv_bench_same(benchmark::State &state, Func conv_func) {
  arma::Col<T> vec1(state.range(0), arma::fill::randn);
  arma::Col<T> vec2(state.range(1), arma::fill::randn);
  arma::Col<T> res(vec1.size());

  const std::span<const T> span1(vec1);
  const std::span<const T> span2(vec2);
  std::span<T> span_res(res);

  for (auto _ : state) {
    conv_func(vec1, vec2, res);
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(T));
}

// const std::vector<std::vector<int64_t>> args{
//     {{1000, 1664, 2304, 2816, 3326, 4352}, {15, 35, 65, 95}}};

const std::vector<std::vector<int64_t>> args{{{2304, 4352}, {95}}};

template <fftconv::Floating T> void BM_oaconvolve(benchmark::State &state) {
  conv_bench_full<T>(state, fftconv::oaconvolve_fftw<T, fftconv::Same>);
}

BENCHMARK(BM_oaconvolve<double>)->ArgsProduct(args);
BENCHMARK(BM_oaconvolve<float>)->ArgsProduct(args);

template <fftconv::Floating T>
void BM_oaconvolve_same(benchmark::State &state) {
  conv_bench_same<T>(state, fftconv::oaconvolve_fftw<T, fftconv::Same>);
}
BENCHMARK(BM_oaconvolve_same<double>)->ArgsProduct(args);
BENCHMARK(BM_oaconvolve_same<float>)->ArgsProduct(args);

// template <fftconv::Floating T>
// void BM_convolve(benchmark::State &state) {
//   conv_bench_full<T>(state, fftconv::convolve_fftw<T>);
// }
// BENCHMARK(BM_convolve<double>)->ArgsProduct(args)->Setup(DoSetup)->Teardown(DoTeardown);
// BENCHMARK(BM_convolve<float>)->ArgsProduct(args)->Setup(DoSetup)->Teardown(DoTeardown);

// template <fftconv::Floating T>
// void BM_arma_conv(benchmark::State &state) {
//   conv_bench_full<T>(state, arma_conv_full<T>);
// }
// BENCHMARK(BM_arma_conv<double>)->ArgsProduct(args);
// BENCHMARK(BM_arma_conv<float>)->ArgsProduct(args);

// template <fftconv::Floating T>
// void BM_arma_conv_same(benchmark::State &state) {
//   conv_bench_same<T>(state, arma_conv_same<T>);
// }
// BENCHMARK(BM_arma_conv_same<double>)->ArgsProduct(args);
// BENCHMARK(BM_arma_conv_same<float>)->ArgsProduct(args);

// NOLINTEND(*-identifier-length)

// BENCHMARK_MAIN();

int main(int argc, char **argv) {
  fftw::WisdomSetup wisdom(false);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}