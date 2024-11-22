#include <armadillo>
#include <benchmark/benchmark.h>
#include <kfr/all.hpp>
#include <kfr/base/univector.hpp>
#include <span>

#include "fftconv.hpp"
#include "fftw.hpp"
// #include "fftconv_pocket.hpp"

// NOLINTBEGIN(*-identifier-length)

//------------------ Helper functions

// Wrapper to prevent arma::conv from being optimized away
// Not storing results back in res.
template <fftconv::FloatOrDouble Real>
void arma_conv_full(const std::span<const Real> span1,
                    const std::span<const Real> span2,
                    std::span<Real> span_res) {
  // NOLINTBEGIN(*-const-cast)
  const arma::Col<Real> vec1(const_cast<Real *>(span1.data()), span1.size(),
                             false, true);
  const arma::Col<Real> vec2(const_cast<Real *>(span2.data()), span2.size(),
                             false, true);
  // NOLINTEND(*-const-cast)
  volatile arma::Col<Real> res = arma::conv(vec1, vec2);
}

template <fftconv::FloatOrDouble Real, typename Func>
void conv_bench_full(benchmark::State &state, Func conv_func) {
  arma::Col<Real> input(state.range(0), arma::fill::randn);
  arma::Col<Real> kernel(state.range(1), arma::fill::randn);
  arma::Col<Real> output(input.size() + kernel.size() - 1);

  for (auto _ : state) {
    conv_func(input, kernel, output);
  }
}

template <fftconv::FloatOrDouble Real>
void kfr_conv(const std::span<const Real> span1,
              const std::span<const Real> span2, std::span<Real> span_res) {

  auto inData = kfr::make_univector(span1.data(), span1.size());
  auto taps = kfr::make_univector(span2.data(), span2.size());
  auto res = kfr::make_univector(span_res.data(), span_res.size());

  kfr::filter_fir<Real> filter(taps);
  // kfr::convolve_filter<T> filter(taps);
  filter.apply(res, inData);
}

// Wrapper to prevent arma::conv from being optimized away
// Not storing results back in res.
template <fftconv::FloatOrDouble Real>
void arma_conv_same(const std::span<const Real> span1,
                    const std::span<const Real> span2,
                    std::span<Real> span_res) {
  // NOLINTBEGIN(*-const-cast)
  const arma::Col<Real> vec1(const_cast<Real *>(span1.data()), span1.size(),
                             false, true);
  const arma::Col<Real> vec2(const_cast<Real *>(span2.data()), span2.size(),
                             false, true);
  // NOLINTEND(*-const-cast)
  volatile arma::Col<Real> res = arma::conv(vec1, vec2, "same");
}

template <fftconv::FloatOrDouble Real, typename Func>
void conv_bench_same(benchmark::State &state, Func conv_func) {
  arma::Col<Real> vec1(state.range(0), arma::fill::randn);
  arma::Col<Real> vec2(state.range(1), arma::fill::randn);
  arma::Col<Real> res(vec1.size());

  const std::span<const Real> span1(vec1);
  const std::span<const Real> span2(vec2);
  std::span<Real> span_res(res);

  for (auto _ : state) {
    conv_func(vec1, vec2, res);
  }
}

// const std::vector<std::vector<int64_t>> args{
//     {{1000, 1664, 2304, 2816, 3326, 4352}, {15, 35, 65, 95}}};

const std::vector<std::vector<int64_t>> args{{{2304, 4352}, {65, 95}}};

template <fftconv::FloatOrDouble Real>
void BM_oaconvolve(benchmark::State &state) {
  conv_bench_full<Real>(state, fftconv::oaconvolve_fftw<Real>);
}

BENCHMARK(BM_oaconvolve<double>)->ArgsProduct(args);
BENCHMARK(BM_oaconvolve<float>)->ArgsProduct(args);

template <fftconv::FloatOrDouble Real>
void BM_oaconvolve_same(benchmark::State &state) {
  conv_bench_same<Real>(state, fftconv::oaconvolve_fftw_same<Real>);
}
BENCHMARK(BM_oaconvolve_same<double>)->ArgsProduct(args);
BENCHMARK(BM_oaconvolve_same<float>)->ArgsProduct(args);

// template <fftconv::FloatOrDouble Real>
// void BM_convolve(benchmark::State &state) {
//   conv_bench_full<Real>(state, fftconv::convolve_fftw<Real>);
// }
// BENCHMARK(BM_convolve<double>)->ArgsProduct(args)->Setup(DoSetup)->Teardown(DoTeardown);
// BENCHMARK(BM_convolve<float>)->ArgsProduct(args)->Setup(DoSetup)->Teardown(DoTeardown);

// template <fftconv::FloatOrDouble Real>
// void BM_arma_conv(benchmark::State &state) {
//   conv_bench_full<Real>(state, arma_conv_full<Real>);
// }
// BENCHMARK(BM_arma_conv<double>)->ArgsProduct(args);
// BENCHMARK(BM_arma_conv<float>)->ArgsProduct(args);

// template <fftconv::FloatOrDouble Real>
// void BM_arma_conv_same(benchmark::State &state) {
//   conv_bench_same<Real>(state, arma_conv_same<Real>);
// }
// BENCHMARK(BM_arma_conv_same<double>)->ArgsProduct(args);
// BENCHMARK(BM_arma_conv_same<float>)->ArgsProduct(args);

// template <fftconv::FloatOrDouble Real>
// void BM_kfr_conv_same(benchmark::State &state) {
//   conv_bench_same<Real>(state, kfr_conv<Real>);
// }
// BENCHMARK(BM_kfr_conv_same<double>)->ArgsProduct(args);
// BENCHMARK(BM_kfr_conv_same<float>)->ArgsProduct(args);

// NOLINTEND(*-identifier-length)

// BENCHMARK_MAIN();

int main(int argc, char **argv) {
  fftw::FFTWGlobalSetup fftwSetup;

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}