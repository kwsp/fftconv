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

// ==========================================
// 2D Convolution Benchmarks
// ==========================================

// Args: {rows, cols, krows, kcols}
const std::vector<std::vector<int64_t>> ARGS_2D{{{64, 256}, {64, 256}, {5, 15}, {5, 15}}};

template <fftconv::Floating T> void BM_convolve_2d_full(benchmark::State &state) {
  const auto rows = static_cast<size_t>(state.range(0));
  const auto cols = static_cast<size_t>(state.range(1));
  const auto krows = static_cast<size_t>(state.range(2));
  const auto kcols = static_cast<size_t>(state.range(3));
  const auto orows = rows + krows - 1;
  const auto ocols = cols + kcols - 1;

  AlignedVector<T> a(rows * cols);
  AlignedVector<T> k(krows * kcols);
  AlignedVector<T> out(orows * ocols);

  fftconv::convolve_fftw_2d<T, fftconv::Full>(a, rows, cols, k, krows, kcols, out, orows, ocols);
  for (auto _ : state) {
    fftconv::convolve_fftw_2d<T, fftconv::Full>(a, rows, cols, k, krows, kcols, out, orows, ocols);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(rows * cols));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(rows * cols) * sizeof(T));
}
BENCHMARK(BM_convolve_2d_full<double>)->ArgsProduct(ARGS_2D);
BENCHMARK(BM_convolve_2d_full<float>)->ArgsProduct(ARGS_2D);

template <fftconv::Floating T> void BM_convolve_2d_same(benchmark::State &state) {
  const auto rows = static_cast<size_t>(state.range(0));
  const auto cols = static_cast<size_t>(state.range(1));
  const auto krows = static_cast<size_t>(state.range(2));
  const auto kcols = static_cast<size_t>(state.range(3));

  AlignedVector<T> a(rows * cols);
  AlignedVector<T> k(krows * kcols);
  AlignedVector<T> out(rows * cols);

  fftconv::convolve_fftw_2d<T, fftconv::Same>(a, rows, cols, k, krows, kcols, out, rows, cols);
  for (auto _ : state) {
    fftconv::convolve_fftw_2d<T, fftconv::Same>(a, rows, cols, k, krows, kcols, out, rows, cols);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(rows * cols));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(rows * cols) * sizeof(T));
}
BENCHMARK(BM_convolve_2d_same<double>)->ArgsProduct(ARGS_2D);
BENCHMARK(BM_convolve_2d_same<float>)->ArgsProduct(ARGS_2D);

// ==========================================
// 3D Convolution Benchmarks
// ==========================================

// Args: {depth, rows, cols, kdepth, krows, kcols}
const std::vector<std::vector<int64_t>> ARGS_3D{{{16, 64}, {16, 64}, {16, 64}, {3}, {3}, {3}}};

template <fftconv::Floating T> void BM_convolve_3d_full(benchmark::State &state) {
  const auto depth = static_cast<size_t>(state.range(0));
  const auto rows = static_cast<size_t>(state.range(1));
  const auto cols = static_cast<size_t>(state.range(2));
  const auto kdepth = static_cast<size_t>(state.range(3));
  const auto krows = static_cast<size_t>(state.range(4));
  const auto kcols = static_cast<size_t>(state.range(5));
  const auto odepth = depth + kdepth - 1;
  const auto orows = rows + krows - 1;
  const auto ocols = cols + kcols - 1;

  AlignedVector<T> a(depth * rows * cols);
  AlignedVector<T> k(kdepth * krows * kcols);
  AlignedVector<T> out(odepth * orows * ocols);

  fftconv::convolve_fftw_3d<T, fftconv::Full>(a, depth, rows, cols, k, kdepth, krows, kcols, out, odepth, orows, ocols);
  for (auto _ : state) {
    fftconv::convolve_fftw_3d<T, fftconv::Full>(a, depth, rows, cols, k, kdepth, krows, kcols, out, odepth, orows, ocols);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(depth * rows * cols));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(depth * rows * cols) * sizeof(T));
}
BENCHMARK(BM_convolve_3d_full<double>)->ArgsProduct(ARGS_3D);
BENCHMARK(BM_convolve_3d_full<float>)->ArgsProduct(ARGS_3D);

template <fftconv::Floating T> void BM_convolve_3d_same(benchmark::State &state) {
  const auto depth = static_cast<size_t>(state.range(0));
  const auto rows = static_cast<size_t>(state.range(1));
  const auto cols = static_cast<size_t>(state.range(2));
  const auto kdepth = static_cast<size_t>(state.range(3));
  const auto krows = static_cast<size_t>(state.range(4));
  const auto kcols = static_cast<size_t>(state.range(5));

  AlignedVector<T> a(depth * rows * cols);
  AlignedVector<T> k(kdepth * krows * kcols);
  AlignedVector<T> out(depth * rows * cols);

  fftconv::convolve_fftw_3d<T, fftconv::Same>(a, depth, rows, cols, k, kdepth, krows, kcols, out, depth, rows, cols);
  for (auto _ : state) {
    fftconv::convolve_fftw_3d<T, fftconv::Same>(a, depth, rows, cols, k, kdepth, krows, kcols, out, depth, rows, cols);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(depth * rows * cols));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(depth * rows * cols) * sizeof(T));
}
BENCHMARK(BM_convolve_3d_same<double>)->ArgsProduct(ARGS_3D);
BENCHMARK(BM_convolve_3d_same<float>)->ArgsProduct(ARGS_3D);

// NOLINTEND(*-identifier-length)

// BENCHMARK_MAIN();

int main(int argc, char **argv) {
  fftw::WisdomSetup wisdom(false);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}