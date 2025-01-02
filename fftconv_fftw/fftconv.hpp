// Author: Taylor Nie
// 2022 - 2024
// https://github.com/kwsp/fftconv
//
// Version 0.3.0
#pragma once

#include "fftw.hpp"
#include <array>
#include <cassert>
#include <complex>
#include <memory>
#include <ranges>
#include <span>
#include <type_traits>
#include <unordered_map>

// NOLINTBEGIN(*-reinterpret-cast, *-const-cast)

namespace fftconv {
using fftw::Floating;
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
  std::ranges::copy(src, dst.begin());

  // Fill the remaining part of the destination with zeros
  std::ranges::fill(dst.subspan(src.size()), 0);
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

template <Floating T>
inline void
elementwise_multiply_fftw_cx(std::span<const fftw::Complex<T>> complex1,
                             std::span<const fftw::Complex<T>> complex2,
                             std::span<fftw::Complex<T>> result) {
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

#if defined(__AVX2__)

#include <immintrin.h>

__m256d mult_c128_avx2(__m256d vec1, __m256d vec2) {
  // vec1 and vec2 each have 2 128bit complex
  const __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

  // 1. Multiply
  auto vec3 = _mm256_mul_pd(vec1, vec2);

  // 2. switch the real and imag parts of vec2
  vec2 = _mm256_permute_pd(vec2, 0x5);

  // 3. negate the imag parts of vec2
  vec2 = _mm256_mul_pd(vec2, neg);

  // 4. multiply vec1 and the modified vec2
  auto vec4 = _mm256_mul_pd(vec1, vec2);

  // horizontally subtract the elements in vec3 and vec4
  vec1 = _mm256_hsub_pd(vec3, vec4);

  return vec1;
}

inline __m256 mult_c64_avx2(__m256 vec1, __m256 vec2) {
  // vec1 and vec2 each have 4 64bit complex
  const __m256 neg =
      _mm256_setr_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f);

  __m256 vec3 = _mm256_mul_ps(vec1, vec2);
  vec2 = _mm256_permute_ps(vec2, 0b10110001);
  vec2 = _mm256_mul_ps(vec2, neg);
  __m256 vec4 = _mm256_mul_ps(vec1, vec2);
  vec1 = _mm256_hsub_ps(vec3, vec4);
  vec1 = _mm256_permute_ps(vec1, 0b11011000);
  return vec1;
}

template <typename T>
inline void multiply_cx_haswell(std::span<const std::complex<T>> cx1,
                                std::span<const std::complex<T>> cx2,
                                std::span<std::complex<T>> out) {
  constexpr size_t simd_width = 256 / (8 * sizeof(T));
  const size_t vec_size = std::min({cx1.size(), cx2.size(), out.size()});
  const size_t vec_end = vec_size / (simd_width / 2) * (simd_width / 2);
  // Process pairs of complex numbers

  if constexpr (std::is_same_v<T, float>) {

    const __m256 neg =
        _mm256_setr_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f);

    for (size_t i = 0; i < vec_end; i += simd_width / 2) {
      // Load interleaved real and imaginary parts
      __m256 vec1 = _mm256_load_ps(reinterpret_cast<const float *>(&cx1[i]));
      __m256 vec2 = _mm256_load_ps(reinterpret_cast<const float *>(&cx2[i]));

      // auto res = mult_c64_avx2(vec1, vec2);
      // _mm256_storeu_ps(reinterpret_cast<float *>(&out[i]), res);

      __m256 vec3 = _mm256_mul_ps(vec1, vec2);
      vec2 = _mm256_permute_ps(vec2, 0b10110001);
      vec2 = _mm256_mul_ps(vec2, neg);
      __m256 vec4 = _mm256_mul_ps(vec1, vec2);
      vec1 = _mm256_hsub_ps(vec3, vec4);
      vec1 = _mm256_permute_ps(vec1, 0b11011000);
      _mm256_store_ps(reinterpret_cast<float *>(&out[i]), vec1);
    }
  } else if constexpr (std::is_same_v<T, double>) {

    const __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    for (size_t i = 0; i < vec_end; i += simd_width / 2) {
      // Load interleaved real and imaginary parts
      __m256d vec1 = _mm256_load_pd(reinterpret_cast<const double *>(&cx1[i]));
      __m256d vec2 = _mm256_load_pd(reinterpret_cast<const double *>(&cx2[i]));

      // auto res = mult_c128_avx2(vec1, vec2);
      // _mm256_storeu_pd(reinterpret_cast<double *>(&out[i]), res);

      auto vec3 = _mm256_mul_pd(vec1, vec2);
      vec2 = _mm256_permute_pd(vec2, 0x5);
      vec2 = _mm256_mul_pd(vec2, neg);
      auto vec4 = _mm256_mul_pd(vec1, vec2);
      vec1 = _mm256_hsub_pd(vec3, vec4);

      _mm256_store_pd(reinterpret_cast<double *>(&out[i]), vec1);
    }
  }

  for (size_t i = vec_end; i < vec_size; ++i) {
    const auto a_1 = cx1[i].real();
    const auto a_2 = cx1[i].imag();
    const auto b_1 = cx2[i].real();
    const auto b_2 = cx2[i].imag();
    const auto real = a_1 * b_1 - a_2 * b_2;
    const auto imag = a_2 * b_1 + a_1 * b_2;
    out[i] = {real, imag};
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

#elif defined(__AVX2__)

  multiply_cx_haswell<T>(cx1, cx2, out);

#else
  multiply_cx_serial(cx1, cx2, out);
#endif
}

// out = cx1 * cx2
template <typename T>
inline void multiply_cx(std::span<const fftw::Complex<T>> cx1,
                        std::span<const fftw::Complex<T>> cx2,
                        std::span<fftw::Complex<T>> out) {

  multiply_cx(
      std::span{reinterpret_cast<const std::complex<T> *>(cx1.data()),
                cx1.size()},
      std::span{reinterpret_cast<const std::complex<T> *>(cx2.data()),
                cx2.size()},
      std::span{reinterpret_cast<std::complex<T> *>(out.data()), out.size()});
}

} // namespace internal

template <typename T> struct FFTConvBuffer {
  using Cx = fftw::Complex<T>;

  std::span<T> real;
  std::span<Cx> cx1;
  std::span<Cx> cx2;

  FFTConvBuffer(const FFTConvBuffer &) = delete;
  FFTConvBuffer(FFTConvBuffer &&) = delete;
  FFTConvBuffer &operator=(const FFTConvBuffer &) = delete;
  FFTConvBuffer &operator=(FFTConvBuffer &&) = delete;

  explicit FFTConvBuffer(size_t real_sz)
      : real(fftw::alloc_real<T>(real_sz), real_sz),
        cx1(fftw::alloc_complex<T>(real_sz / 2 + 1), real_sz / 2 + 1),
        cx2(fftw::alloc_complex<T>(real_sz / 2 + 1), real_sz / 2 + 1) {}

  ~FFTConvBuffer() {
    fftw::free<T>(real.data());
    fftw::free<T>(cx1.data());
    fftw::free<T>(cx2.data());
  }
};

// EngineFFTConv manages the memory of the forward and backward fft plans
// and the fftw buffers
template <typename T>
struct FFTConvEngine : public fftw::cache_mixin<FFTConvEngine<T>> {
  using Plan = fftw::Plan<T>;
  using Cx = fftw::Complex<T>;

  FFTConvBuffer<T> buf;
  Plan forward;
  Plan backward;

  // n is padded_length
  explicit FFTConvEngine(size_t n)
      : buf(n), forward(Plan::dft_r2c_1d(n, buf.real.data(), buf.cx1.data(),
                                         fftw::FLAGS)),
        backward(Plan::dft_c2r_1d(n, buf.cx1.data(), buf.real.data(),
                                  fftw::FLAGS)) {}

  // Get the fftconv_plans object for a specific kernel size
  static auto get_for_ksize(size_t ksize) -> FFTConvEngine<T> & {
    const auto fft_size = internal::get_optimal_fft_size(ksize);
    return FFTConvEngine<T>::get(fft_size);
  }

  void convolve(const std::span<const T> input, const std::span<const T> kernel,
                const std::span<T> output) {
    std::fill(output.begin(), output.end(), static_cast<T>(0));

    // Copy input to buffer
    // TODO assume input is aligned and don't copy
    internal::copy_to_padded_buffer<T>(input, buf.real);

    // A = fft(a)
    forward.execute_dft_r2c(buf.real.data(), buf.cx1.data());

    // Copy b to buffer
    internal::copy_to_padded_buffer<T>(kernel, buf.real);

    // B = fft(b)
    forward.execute_dft_r2c(buf.real.data(), buf.cx2.data());

    // Complex elementwise multiple, A = A * B
    internal::multiply_cx<T>(buf.cx1, buf.cx2, buf.cx1);

    // a = ifft(A)
    backward.execute_dft_c2r(buf.cx1.data(), buf.real.data());

    // divide each result elem by real_sz
    fftw::normalize<T>(buf.real.data(), buf.real.size(), 1. / buf.real.size());

    std::copy(buf.real.begin(), buf.real.end(), output.begin());
  }

  void oaconvolve_full(const std::span<const T> input,
                       const std::span<const T> kernel, std::span<T> output) {
    assert(buf.real.size() == internal::get_optimal_fft_size(kernel.size()));
    std::fill(output.begin(), output.end(), 0);

    const auto &real = buf.real;
    const auto &cx1 = buf.cx1;
    const auto &cx2 = buf.cx2;

    const auto fft_size = real.size();
    const auto step_size = fft_size - (kernel.size() - 1);

    // forward fft of kernel and save to complex2
    internal::copy_to_padded_buffer<T>(kernel, buf.real);
    forward.execute_dft_r2c(real.data(), buf.cx2.data());

    // create forward/backward ffts for x
    // Normalization factor
    const auto fct = static_cast<T>(1. / fft_size);
    for (size_t pos = 0; pos < input.size(); pos += step_size) {
      size_t len =
          std::min<size_t>(input.size() - pos, step_size); // bound check
      internal::copy_to_padded_buffer<T>(input.subspan(pos, len), buf.real);
      forward.execute_dft_r2c(real.data(), buf.cx1.data());
      internal::multiply_cx<T>(cx1, buf.cx2, buf.cx1);
      backward.execute_dft_c2r(cx1.data(), buf.real.data());

      // normalize output and add to result
      fftw::normalize_add<T>(output.subspan(pos), real, fct);
    }
  }

  void oaconvolve_same(const std::span<const T> input,
                       const std::span<const T> kernel, std::span<T> output) {
    assert(buf.real.size() == internal::get_optimal_fft_size(kernel.size()));
    std::fill(output.begin(), output.end(), 0);

    const auto &real = buf.real;
    const auto &cx1 = buf.cx1;
    const auto &cx2 = buf.cx2;

    const auto fft_size = real.size();
    const auto step_size = fft_size - (kernel.size() - 1);

    // forward fft of kernel and save to complex2
    internal::copy_to_padded_buffer<T>(kernel, real);
    forward.execute_dft_r2c(real.data(), cx2.data());

    const size_t padding = kernel.size() / 2;

    // create forward/backward ffts for x
    // Normalization factor
    const auto fct = static_cast<T>(1. / fft_size);
    for (size_t pos = 0; pos < input.size(); pos += step_size) {
      size_t len =
          std::min<size_t>(input.size() - pos, step_size); // bound check

      internal::copy_to_padded_buffer<T>(input.subspan(pos, len), real);
      forward.execute_dft_r2c(real.data(), cx1.data());
      internal::multiply_cx<T>(cx1, cx2, cx1);
      backward.execute_dft_c2r(cx1.data(), real.data());

      // normalize output and add to result
      if (pos < padding) {
        fftw::normalize_add<T>(output, real.subspan(padding), fct);
      } else {
        fftw::normalize_add<T>(output.subspan(pos - padding), real, fct);
      }
    }
  }
};

// 1D convolution using the FFT
//
// Optimizations:
//    * Cache fftw_plan
//    * Reuse buffers (no malloc on second call to the same convolution size)
// https://en.wikipedia.org/w/index.php?title=Convolution#Fast_convolution_algorithms
template <Floating T>
void convolve_fftw(const std::span<const T> input,
                   const std::span<const T> kernel, std::span<T> output) {
  // length of the real arrays, including the final convolution output
  const size_t padded_length = input.size() + kernel.size() - 1;

  auto &plan = FFTConvEngine<T>::get(padded_length);

  // Execute FFT convolution and copy normalized result
  plan.convolve(input, kernel, output);
}

enum ConvMode { Full, Same };

/**
1D Overlap-Add convolution
// input_size >> kernel_size

For "Full" mode, output_size >= input_size + kernel_size - 1
For "Same" mode, output_size == input_size

1. Split arr into blocks of step_size.
2. convolve with kernel using fft of length N.
3. add blocks together
 */
template <Floating T, ConvMode Mode = ConvMode::Full>
void oaconvolve_fftw(std::span<const T> input, std::span<const T> kernel,
                     std::span<T> output) {
  if constexpr (Mode == ConvMode::Full) {
    assert(input.size() + kernel.size() - 1 == output.size());
  } else if constexpr (Mode == ConvMode::Same) {
    assert(input.size() == output.size());
  }

  // Get cached plans
  auto &plan = FFTConvEngine<T>::get_for_ksize(kernel.size());

  // Execute FFT convolution and copy normalized result
  if constexpr (Mode == ConvMode::Full) {
    plan.oaconvolve_full(input, kernel, output);
  } else if constexpr (Mode == ConvMode::Same) {
    plan.oaconvolve_same(input, kernel, output);
  } else {
    static_assert(false, "Unsupported mode.");
  }
}

// Reference implementation of fft convolution with minimal optimizations
// Only supports double
template <Floating T>
void convolve_fftw_ref(const std::span<const double> input,
                       const std::span<const double> kernel,
                       std::span<double> output)
  requires(std::is_same_v<T, double>)
{
  std::fill(output.begin(), output.end(), static_cast<T>(0));

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
  internal::elementwise_multiply_fftw_cx<T>(
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
