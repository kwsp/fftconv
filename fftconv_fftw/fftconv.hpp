// Author: Taylor Nie
// 2022 - 2024
// https://github.com/kwsp/fftconv
//
// Version 0.2.1
#pragma once

#include "fftw.hpp"

#include <array>
#include <cassert>
#include <complex>
#include <memory>
// #include <mutex>
#include <span>
#include <type_traits>
#include <unordered_map>

// NOLINTBEGIN(*-reinterpret-cast, *-const-cast)

namespace fftconv {
using fftw::fftw_buffer;
using fftw::FloatOrDouble;
using fftw::Plan;

namespace internal {

// NOLINTBEGIN(*-global-variables)
// static std::mutex *fftconv_fftw_mutex = nullptr;
// NOLINTEND(*-global-variables)

// static int nextpow2(int x) { return 1 << (int)(std::log2(x) + 1); }
// Lookup table of {max_filter_size, optimal_fft_size}
constexpr std::array<std::array<size_t, 2>, 9> optimal_oa_fft_size{
    {{7, 16},
     {12, 32},
     {21, 64},
     {36, 128},
     {65, 256},
     {120, 512},
     {221, 1024},
     {411, 2048},
     {769, 4096}},
};
constexpr size_t max_oaconv_fft_size = 8192;

// Given a filter_size, return the optimal fft size for the overlap-add
// convolution method
inline auto get_optimal_fft_size(const size_t filter_size) -> size_t {
  for (const auto &pair : optimal_oa_fft_size) {
    if (filter_size < pair[0]) {
      return pair[1];
    }
  }
  return max_oaconv_fft_size;
}

// In memory cache with key type K and value type V
template <class K, class V> auto get_cached(K key) {
  static thread_local std::unordered_map<K, std::unique_ptr<V>> _cache;

  auto &val = _cache[key];
  if (val == nullptr) {
    val = std::make_unique<V>(key);
  }

  return val.get();
}

// // In memory cache with key type K and value type V
// // additionally accepts a mutex to guard the V constructor
// template <class Key, class Val>
// auto get_cached_vlock(Key key, std::mutex *V_mutex) {
//   static thread_local std::unordered_map<Key, std::unique_ptr<Val>> _cache;

//   auto &val = _cache[key];
//   if (val == nullptr) {
//     // Using unique_lock here for RAII locking since the mutex is optional.
//     // If we have a non-optional mutex, prefer scoped_lock
//     const auto lock = V_mutex == nullptr ? std::unique_lock<std::mutex>{}
//                                          : std::unique_lock(*V_mutex);

//     val = std::make_unique<Val>(key);
//   }

//   return val.get();
// }

// Copy data from src to dst and padded the extra with zero
// dst_size must be greater than src_size
// Deprecated
template <typename Real>
inline void copy_to_padded_buffer(const Real *src, const size_t src_size,
                                  Real *dst, const size_t dst_size) {
  assert(src_size <= dst_size);
  std::copy(src, src + src_size, dst);
  std::fill(dst + src_size, dst + dst_size, 0);
}

template <typename Tin, typename Tout = Tin>
inline void copy_to_padded_buffer(const std::span<const Tin> src,
                                  const std::span<Tout> dst) {
  // Assert that destination span is larger or equal to the source span
  assert(src.size() <= dst.size());

  // Copy data from source to destination
  std::copy(src.begin(), src.end(), dst.begin());

  // Fill the remaining part of the destination with zeros
  auto dst_rest = dst.subspan(src.size());
  std::fill(dst_rest.begin(), dst_rest.end(), 0);
}

// static inline void elementwise_multiply(const fftw_complex *a,
// const fftw_complex *b,
// const size_t length,
// fftw_complex *result) {
//// fftw_complex in C89 mode is double[2], which is binary compatible with
//// C99's <complex.h> and C++'s complex<double> template class
//// http://www.fftw.org/doc/Complex-numbers.html
// const auto _a = reinterpret_cast<const std::complex<double> *>(a);
// const auto _b = reinterpret_cast<const std::complex<double> *>(b);
// auto _res = reinterpret_cast<std::complex<double> *>(result);

// for (size_t i = 0; i < length; ++i)
//_res[i] = _a[i] * _b[i];
//}

template <fftw::FFTWComplex T>
inline void elementwise_multiply_fftw_cx(std::span<const T> complex1,
                                         std::span<const T> complex2,
                                         std::span<T> result) {
  // implement naive complex multiply. This is much faster than the
  // std version, but doesn't handle infinities.
  // https://stackoverflow.com/questions/49438158/why-is-muldc3-called-when-two-stdcomplex-are-multiplied

  const auto size = std::min(complex1.size(), complex2.size());
  for (size_t i = 0; i < size; ++i) {
    const auto a_1 = complex1[i][0];
    const auto a_2 = complex1[i][1];
    const auto b_1 = complex2[i][0];
    const auto b_2 = complex2[i][1];
    result[i][0] = a_1 * b_1 - a_2 * b_2;
    result[i][1] = a_2 * b_1 + a_1 * b_2;
  }
}

#if defined(__ARM_NEON__)
#include <arm_neon.h>

inline void multiply_cx_neon_f32(std::span<const std::complex<float>> cx1,
                                 std::span<const std::complex<float>> cx2,
                                 std::span<std::complex<float>> result) {
  const size_t size = std::min(cx1.size(), cx2.size());
  const size_t step =
      4; // NEON can process 4 floats at a time (2 complex numbers)

  size_t i = 0;
  for (; i + step <= size; i += step) {
    // Load interleaved real and imaginary parts
    float32x4x2_t c1 =
        vld2q_f32(reinterpret_cast<const float *>(cx1.data() + i));
    float32x4x2_t c2 =
        vld2q_f32(reinterpret_cast<const float *>(cx2.data() + i));

    // Perform complex multiplication
    float32x4_t real_part =
        vmlsq_f32(vmulq_f32(c1.val[0], c2.val[0]), c1.val[1], c2.val[1]);
    float32x4_t imag_part =
        vmlaq_f32(vmulq_f32(c1.val[1], c2.val[0]), c1.val[0], c2.val[1]);

    // Interleave results back into complex form
    float32x4x2_t result_vec = {real_part, imag_part};

    // Store result
    vst2q_f32(reinterpret_cast<float *>(result.data() + i), result_vec);
  }

  // Handle remaining elements (scalar fallback)
  for (; i < size; ++i) {
    const auto a_1 = cx1[i].real();
    const auto a_2 = cx1[i].imag();
    const auto b_1 = cx2[i].real();
    const auto b_2 = cx2[i].imag();
    const auto real = a_1 * b_1 - a_2 * b_2;
    const auto imag = a_2 * b_1 + a_1 * b_2;
    result[i] = {real, imag};
  }
}
#endif

template <typename T>
inline void multiply_cx_serial(std::span<const std::complex<T>> cx1,
                               std::span<const std::complex<T>> cx2,
                               std::span<std::complex<T>> out) {
  // implement naive complex multiply. This is much faster than the
  // std version, but doesn't handle infinities.
  // https://stackoverflow.com/questions/49438158/why-is-muldc3-called-when-two-stdcomplex-are-multiplied

  const auto size = std::min(cx1.size(), cx2.size());
  for (size_t i = 0; i < size; ++i) {
    const auto a_1 = cx1[i].real();
    const auto a_2 = cx1[i].imag();
    const auto b_1 = cx2[i].real();
    const auto b_2 = cx2[i].imag();
    const auto real = a_1 * b_1 - a_2 * b_2;
    const auto imag = a_2 * b_1 + a_1 * b_2;
    out[i] = {real, imag};
  }
}

template <typename T>
inline void multiply_cx(std::span<const std::complex<T>> cx1,
                        std::span<const std::complex<T>> cx2,
                        std::span<std::complex<T>> out) {
#if defined(__ARM_NEON__)
  if constexpr (std::is_same_v<T, float>) {
    multiply_cx_neon_f32(cx1, cx2, out);
  } else if constexpr (std::is_same_v<T, double>) {
    multiply_cx_serial(cx1, cx2, out);
  }
#else
  elementwise_multiply_cx_serial(cx1, cx2, out);
#endif
}

template <typename T>
inline void normalize_add_results_serial(T *out, T *real, size_t len, T fct) {
  for (size_t i = 0; i < len; ++i) {
    out[i] += real[i] * fct;
  }
}

inline void normalize_add_results_serial_(float *out, float const *inp,
                                          size_t n, float fct) {
  for (size_t i = 0; i < n; ++i) {
    out[i] += inp[i] * fct;
  }
}

#if defined(__ARM_NEON__)

#include <arm_neon.h>

inline void normalize_add_neon_f32(float *out, float const *inp, size_t n,
                                   float fct) {
  const float32x4_t fct_vec = vdupq_n_f32(fct);
  constexpr size_t n_step = 4;

  size_t i = 0;
  for (; i + n_step <= n; i += n_step) {
    float32x4_t out_vec = vld1q_f32(out + i);
    float32x4_t inp_vec = vld1q_f32(inp + i);
    out_vec = vfmaq_f32(out_vec, inp_vec, fct_vec);
    vst1q_f32(out + i, out_vec);
  }
  for (; i < n; ++i) {
    out[i] += inp[i] * fct;
  }
}

inline void normalize_add_neon_f64(double *out, double const *inp, size_t n,
                                   double fct) {
  const float64x2_t fct_vec = vdupq_n_f64(fct);
  constexpr size_t n_step = 2;

  size_t i = 0;
  for (; i + n_step <= n; i += n_step) {
    auto out_vec = vld1q_f64(out + i);
    auto inp_vec = vld1q_f64(inp + i);
    out_vec = vfmaq_f64(out_vec, inp_vec, fct_vec);
    vst1q_f64(out + i, out_vec);
  }
  for (; i < n; ++i) {
    out[i] += inp[i] * fct;
  }
}

#endif

template <typename T> void normalize_add(T *out, T *inp, size_t n, T fct) {

#if defined(__ARM_NEON__)
  if constexpr (std::is_same_v<T, float>) {
    normalize_add_neon_f32(out, inp, n, fct);
  } else if constexpr (std::is_same_v<T, double>) {
    normalize_add_neon_f64(out, inp, n, fct);
  } else {
    static_assert(false, "Not implemented");
  }
#else
  normalize_add_results_serial<T>(out, inp, n, fct);
#endif
}

template <FloatOrDouble Real> struct PlansBase {
  using Cx = std::complex<Real>;

  Plan<Real> plan_forward;
  Plan<Real> plan_backward;

  PlansBase(Plan<Real> &&forward, Plan<Real> &&backward)
      : plan_forward(std::move(forward)), plan_backward(std::move(backward)) {}

  PlansBase() = default;
  PlansBase(PlansBase &&other) = delete;
  PlansBase(const PlansBase &other) = delete;
  auto operator=(PlansBase &&other) -> PlansBase = delete;
  auto operator=(const PlansBase &other) -> PlansBase = delete;
  ~PlansBase() = default;
};

template <FloatOrDouble Real> struct Plans1d : public PlansBase<Real> {
  using Cx = std::complex<Real>;

  explicit Plans1d(const fftw_buffer<Real> &real,
                   const fftw_buffer<Cx> &complex)
      : PlansBase<Real>(Plan<Real>::plan_dft_r2c_1d(real, complex),
                        Plan<Real>::plan_dft_c2r_1d(complex, real)) {}

  constexpr void forward(const fftw_buffer<Real> &real,
                         fftw_buffer<Cx> &complex) const {
    this->plan_forward.execute_dft_r2c(real, complex);
  }

  constexpr void backward(const fftw_buffer<Cx> &complex,
                          fftw_buffer<Real> &real) const {
    this->plan_backward.execute_dft_c2r(complex, real);
  }
};

template <FloatOrDouble Real> struct Plans1dMany : public Plans1d<Real> {
  using Cx = std::complex<Real>;

  Plans1dMany(const fftw_buffer<Real> &real, const fftw_buffer<Cx> &complex,
              int n_arrays) {
    this->plan_forward = Plan<Real>::plan_many_dft_r2c(real, complex, n_arrays);
    this->plan_backward =
        Plan<Real>::plan_many_dft_c2r(complex, real, n_arrays);
  }
};

} // namespace internal

// // Since FFTW planners are not thread-safe, you can pass a pointer to a
// // std::mutex to fftconv and all calls to the planner with be guarded by the
// // mutex.
// inline void use_fftw_mutex(std::mutex *fftw_mutex) {
//   internal::fftconv_fftw_mutex = fftw_mutex;
// }
// inline auto get_fftw_mutex() -> std::mutex * {
//   return internal::fftconv_fftw_mutex;
// };

// fft_plans manages the memory of the forward and backward fft plans
// and the fftw buffers
//
// Not using half complex transforms before it's non-trivial to do complex
// multiplications with FFTW's half complex format
template <fftw::FloatOrDouble Real> class fftconv_plans {
public:
  using Cx = std::complex<Real>;

  // Get the fftconv_plans object for a specific kernel size
  static auto get(const size_t padded_length) -> auto & {
    return *internal::get_cached<size_t, fftconv_plans<Real>>(padded_length);
  }

  // Get the fftconv_plans object for a specific kernel size
  static auto get_for_kernel(const std::span<const Real> kernel) -> auto & {
    const auto fft_size = internal::get_optimal_fft_size(kernel.size());
    return fftconv_plans<Real>::get(fft_size);
  }

  // Constructors
  // Compute the fftw plans and allocate buffers
  explicit fftconv_plans(const int padded_length)
      : real(padded_length), complex1(padded_length / 2 + 1),
        complex2(padded_length / 2 + 1), plans(real, complex1) {}

  // Convolve real arrays a and b
  // Results saved in real_buf
  void convolve(const std::span<const Real> input,
                const std::span<const Real> kernel,
                const std::span<Real> output) {
    std::fill(output.begin(), output.end(), static_cast<Real>(0));

    std::span<Real> real_span(real);

    // Copy a to buffer
    internal::copy_to_padded_buffer<Real>(input, real_span);

    // A = fft(a)
    plans.forward(real, complex1);

    // Copy b to buffer
    internal::copy_to_padded_buffer<Real>(kernel, real_span);

    // B = fft(b)
    plans.forward(real, complex2);

    // Complex elementwise multiple, A = A * B
    internal::multiply_cx<Real>(complex1, complex2, complex1);

    // a = ifft(A)
    plans.backward(complex1, real);

    // divide each result elem by real_sz
    for (size_t i = 0; i < real.size(); ++i) {
      real[i] /= real.size();
    }

    std::copy(real.begin(), real.end(), output.begin());
  }

  void oaconvolve(const std::span<const Real> input,
                  const std::span<const Real> kernel, std::span<Real> output) {
    assert(real.size() == internal::get_optimal_fft_size(kernel.size()));
    std::fill(output.begin(), output.end(), static_cast<Real>(0));

    const auto fft_size = real.size();
    const auto step_size = fft_size - (kernel.size() - 1);

    std::span<Real> real_span(real);

    // forward fft of kernel and save to complex2
    internal::copy_to_padded_buffer<Real>(kernel, real_span);
    plans.forward(real, complex2);

    // create forward/backward ffts for x
    // Normalization factor
    const auto fct = static_cast<Real>(1. / fft_size);
    for (size_t pos = 0; pos < input.size(); pos += step_size) {
      size_t len =
          std::min<size_t>(input.size() - pos, step_size); // bound check

      internal::copy_to_padded_buffer<Real>(input.subspan(pos, len), real_span);
      plans.forward(real, complex1);

      internal::multiply_cx<Real>(complex1, complex2, complex1);

      plans.backward(complex1, real);

      // normalize output and add to result
      len = std::min<size_t>(output.size() - pos, fft_size);
      internal::normalize_add<Real>(output.data() + pos, real.data(), len, fct);
    }
  }

  void oaconvolve_same(const std::span<const Real> input,
                       const std::span<const Real> kernel,
                       std::span<Real> output) {
    assert(real.size() == internal::get_optimal_fft_size(kernel.size()));
    assert(input.size() == output.size());
    std::fill(output.begin(), output.end(), static_cast<Real>(0));

    const auto fft_size = real.size();
    const auto step_size = fft_size - (kernel.size() - 1);

    std::span<Real> real_span(real);

    // forward fft of kernel and save to complex2
    internal::copy_to_padded_buffer<Real>(kernel, real_span);
    plans.forward(real, complex2);

    const int64_t copy_start = kernel.size() / 2;

    // create forward/backward ffts for x
    // Normalization factor
    const auto fct = static_cast<Real>(1. / fft_size);
    const int64_t ksize_half = kernel.size() / 2;
    for (int64_t pos = 0; pos < input.size(); pos += step_size) {
      const int64_t len = std::min<size_t>(input.size() - pos, step_size);

      internal::copy_to_padded_buffer<Real>(input.subspan(pos, len), real_span);
      plans.forward(real, complex1);

      internal::multiply_cx<Real>(complex1, complex2, complex1);

      plans.backward(complex1, real);

      // normalize output and add to result
      const int64_t loop_start = std::max<int64_t>(copy_start - pos, 0LL);
      const int64_t loop_end =
          std::min<int64_t>(output.size() - pos + copy_start, fft_size);
      const int64_t n = loop_end - loop_start;
      internal::normalize_add<Real>(
          output.data() + pos - copy_start + loop_start,
          real.data() + loop_start, loop_end - loop_start, fct);
    }
  }

private:
  // FFTW buffers
  fftw_buffer<Real> real;
  fftw_buffer<Cx> complex1;
  fftw_buffer<Cx> complex2;

  // FFTW plans
  internal::Plans1d<Real> plans;
};

// 1D convolution using the FFT
//
// Optimizations:
//    * Cache fftw_plan
//    * Reuse buffers (no malloc on second call to the same convolution size)
// https://en.wikipedia.org/w/index.php?title=Convolution#Fast_convolution_algorithms
template <FloatOrDouble Real>
void convolve_fftw(const std::span<const Real> input,
                   const std::span<const Real> kernel, std::span<Real> output) {
  // length of the real arrays, including the final convolution output
  const size_t padded_length = input.size() + kernel.size() - 1;

  // Get cached plans
  auto &plan = fftconv_plans<Real>::get(padded_length);

  // Execute FFT convolution and copy normalized result
  plan.convolve(input, kernel, output);
}

// 1D Overlap-Add convolution ("full" mode)
//
// input is a 1D signal
// kernel is a kernel, input_size >> kernel_size
// res is the results buffer. output_size >= input_size + kernel_size - 1
//
// 1. Split arr into blocks of step_size.
// 2. convolve with kernel using fft of length N.
// 3. add blocks together
template <FloatOrDouble Real>
void oaconvolve_fftw(const std::span<const Real> input,
                     const std::span<const Real> kernel,
                     std::span<Real> output) {
  assert(input.size() + kernel.size() - 1 == output.size());

  // Get cached plans
  auto &plan = fftconv_plans<Real>::get_for_kernel(kernel);

  // Execute FFT convolution and copy normalized result
  plan.oaconvolve(input, kernel, output);
}

// 1D Overlap-Add convolution ("same" mode)
//
// input is a 1D signal
// kernel is a kernel, input_size >> kernel_size
// res is the results buffer. output_size >= input_size + kernel_size - 1
//
// 1. Split arr into blocks of step_size.
// 2. convolve with kernel using fft of length N.
// 3. add blocks together
template <FloatOrDouble Real>
void oaconvolve_fftw_same(const std::span<const Real> input,
                          const std::span<const Real> kernel,
                          std::span<Real> output) {
  assert(input.size() == output.size());

  // Get cached plans
  auto &plan = fftconv_plans<Real>::get_for_kernel(kernel);

  // Execute FFT convolution and copy normalized result
  plan.oaconvolve_same(input, kernel, output);
}

// Reference implementation of fft convolution with minimal optimizations
// Only supports double
template <FloatOrDouble Real>
void convolve_fftw_ref(const std::span<const double> input,
                       const std::span<const double> kernel,
                       std::span<double> output)
  requires(std::is_same_v<Real, double>)
{
  std::fill(output.begin(), output.end(), static_cast<Real>(0));

  // length of the real arrays, including the final convolution output
  const size_t padded_length = input.size() + kernel.size() - 1;
  // length of the complex arrays
  const auto complex_length = padded_length / 2 + 1;

  // Allocate fftw buffers for a
  double *a_buf = fftw_alloc_real(padded_length);
  fftw_complex *A_buf = fftw_alloc_complex(complex_length);

  // Compute forward fft plan
  fftw_plan plan_forward = fftw_plan_dft_r2c_1d(static_cast<int>(padded_length),
                                                a_buf, A_buf, FFTW_ESTIMATE);

  // Copy a to buffer
  internal::copy_to_padded_buffer(input,
                                  std::span<double>(a_buf, padded_length));

  // Compute Fourier transform of vector a
  fftw_execute_dft_r2c(plan_forward, a_buf, A_buf);

  // Allocate fftw buffers for b
  double *b_buf = fftw_alloc_real(padded_length);
  fftw_complex *B_buf = fftw_alloc_complex(complex_length);

  // Copy b to buffer
  internal::copy_to_padded_buffer(kernel,
                                  std::span<double>(b_buf, padded_length));

  // Compute Fourier transform of vector b
  fftw_execute_dft_r2c(plan_forward, b_buf, B_buf);

  // Compute backward fft plan
  fftw_complex *input_buffer = fftw_alloc_complex(complex_length);
  double *output_buffer = fftw_alloc_real(padded_length);
  fftw_plan plan_backward =
      fftw_plan_dft_c2r_1d(static_cast<int>(padded_length), input_buffer,
                           output_buffer, FFTW_ESTIMATE);

  // Perform element-wise product of FFT(a) and FFT(b)
  // then compute inverse fourier transform.
  internal::elementwise_multiply_fftw_cx(
      std::span<fftw_complex const>(A_buf, complex_length),
      std::span<fftw_complex const>(B_buf, complex_length),
      std::span(input_buffer, complex_length));

  // A_buf becomes input to inverse conv
  fftw_execute_dft_c2r(plan_backward, input_buffer, output_buffer);

  // Normalize output
  for (int i = 0; i < std::min<size_t>(padded_length, output.size()); i++) {
    // NOLINTBEGIN(*-pointer-arithmetic)
    output[i] = output_buffer[i] / static_cast<double>(padded_length);
    // NOLINTEND(*-pointer-arithmetic)
  }

  fftw_free(a_buf);
  fftw_free(b_buf);
  fftw_free(A_buf);
  fftw_free(B_buf);
  fftw_free(input_buffer);
  fftw_free(output_buffer);
  fftw_destroy_plan(plan_forward);
  fftw_destroy_plan(plan_backward);
}

} // namespace fftconv

// NOLINTEND(*-reinterpret-cast, *-const-cast)