#include <benchmark/benchmark.h>
#include <span>

#include "fftconv.hpp"
// #include "fftconv_pocket.hpp"
#include <armadillo>

using std::vector;

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
  arma::Col<Real> vec1(state.range(0), arma::fill::randn);
  arma::Col<Real> vec2(state.range(1), arma::fill::randn);
  arma::Col<Real> res(vec1.size() + vec2.size() - 1);

  const std::span<const Real> span1(vec1);
  const std::span<const Real> span2(vec2);
  std::span<Real> span_res(res);

  for (auto _ : state) {
    conv_func(vec1, vec2, res);
  }
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

const std::vector<std::vector<int64_t>> args{
    {{1000, 1664, 2304, 2816, 3326, 4352}, {15, 35, 65, 95}}};

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

template <fftconv::FloatOrDouble Real>
void BM_convolve(benchmark::State &state) {
  conv_bench_full<Real>(state, fftconv::convolve_fftw<Real>);
}
BENCHMARK(BM_convolve<double>)->ArgsProduct(args);
BENCHMARK(BM_convolve<float>)->ArgsProduct(args);

template <fftconv::FloatOrDouble Real>
void BM_arma_conv(benchmark::State &state) {
  conv_bench_full<Real>(state, arma_conv_full<Real>);
}
BENCHMARK(BM_arma_conv<double>)->ArgsProduct(args);
BENCHMARK(BM_arma_conv<float>)->ArgsProduct(args);

template <fftconv::FloatOrDouble Real>
void BM_arma_conv_same(benchmark::State &state) {
  conv_bench_same<Real>(state, arma_conv_same<Real>);
}
BENCHMARK(BM_arma_conv_same<double>)->ArgsProduct(args);
BENCHMARK(BM_arma_conv_same<float>)->ArgsProduct(args);

// REGISTER(convolve_pocketfft)
// REGISTER(oaconvolve_pocketfft)
// REGISTER(convolve_pocketfft_hdr)
// REGISTER(oaconvolve_pocketfft_hdr)

// NOLINTEND(*-identifier-length)

BENCHMARK_MAIN();
