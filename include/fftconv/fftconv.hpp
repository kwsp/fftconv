// Author: Taylor Nie
// 2022 - 2025
// https://github.com/kwsp/fftconv
#pragma once

#include <array>
#include <cassert>
#include <complex>
#include <fftconv/aligned_vector.hpp>
#include <fftconv/fftw.hpp>
#include <memory>
#include <span>
#include <type_traits>
#include <unordered_map>

// NOLINTBEGIN(*-reinterpret-cast, *-const-cast, *-pointer-arithmetic)

#define FFTCONV_VERSION "0.5.1"

namespace fftconv {
using fftw::Floating;
using fftw::Plan;
enum ConvMode { Full, Same };

// Function to check if a pointer is SIMD-aligned
template <std::size_t Alignment> bool isSIMDAligned(const void *ptr) {
  return reinterpret_cast<std::uintptr_t>(ptr) % Alignment == 0;
}

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

// Copy data from src to dst and padded the extra with zero
// dst_size must be greater than src_size
template <typename T>
inline void copy_to_padded_buffer(const std::span<const T> src,
                                  const std::span<T> dst) {
  // Assert that destination span is larger or equal to the source span
  assert(src.size() <= dst.size());

  // Copy data from source to destination
  std::copy(src.begin(), src.end(), dst.begin());

  // Fill the remaining part of the destination with zeros
  const auto dst_ = dst.subspan(src.size());
  std::fill(dst_.begin(), dst_.end(), 0);
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

inline __m256d mult_c128_avx2(__m256d vec1, __m256d vec2) {
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
  // using Cx = fftw::Complex<T>;
  using Cx = std::complex<T>;

  AlignedVector<T> real;
  AlignedVector<Cx> cx1;
  AlignedVector<Cx> cx2;

  explicit FFTConvBuffer(size_t real_sz)
      : real(real_sz), cx1(real_sz / 2 + 1), cx2(real_sz / 2 + 1) {}

  auto real_ptr() -> T * { return real.data(); }
  auto cx1_ptr() -> fftw::Complex<T> * {
    return reinterpret_cast<fftw::Complex<T> *>(cx1.data());
  }
  auto cx2_ptr() -> fftw::Complex<T> * {
    return reinterpret_cast<fftw::Complex<T> *>(cx2.data());
  }

  auto real_span() -> std::span<T> { return real; }
  auto cx1_span() -> std::span<Cx> { return cx1; }
  auto cx2_span() -> std::span<Cx> { return cx2; }
};

// EngineFFTConv manages the memory of the forward and backward fft plans
// and the fftw buffers
template <typename T, int PlannerFlag = FFTW_ESTIMATE>
struct FFTConvEngine : public fftw::cache_mixin<FFTConvEngine<T>> {
  using Plan = fftw::Plan<T>;
  using Cx = fftw::Complex<T>;
  using View = const std::span<const T>;
  using MutView = std::span<T>;

  FFTConvBuffer<T> buf;
  Plan forward;
  Plan backward;

  // n is padded_length
  explicit FFTConvEngine(size_t n)
      : buf(n), forward(Plan::dft_r2c_1d(n, buf.real_ptr(), buf.cx1_ptr(),
                                         PlannerFlag)),
        backward(
            Plan::dft_c2r_1d(n, buf.cx1_ptr(), buf.real_ptr(), PlannerFlag)) {}

  // Get the fftconv_plans object for a specific kernel size
  static auto get_for_ksize(size_t ksize) -> FFTConvEngine<T> & {
    const auto fft_size = internal::get_optimal_fft_size(ksize);
    return FFTConvEngine<T>::get(fft_size);
  }

  template <ConvMode Mode = ConvMode::Full>
  void convolve(View a, View k, MutView out) {
    std::fill(out.begin(), out.end(), static_cast<T>(0));

    // Copy input to buffer
    // A = fft(a)
    internal::copy_to_padded_buffer<T>(a, buf.real);
    forward.execute_dft_r2c(buf.real_ptr(), buf.cx1_ptr());

    // Copy b to buffer
    // B = fft(b)
    internal::copy_to_padded_buffer<T>(k, buf.real);
    forward.execute_dft_r2c(buf.real_ptr(), buf.cx2_ptr());

    // Complex elementwise multiple, A = A * B
    internal::multiply_cx<T>(buf.cx1, buf.cx2, buf.cx1);

    // a = ifft(A)
    const T fct = 1. / buf.real.size();
    if constexpr (Mode == ConvMode::Full) {

      if (isSIMDAligned<64>(out.data())) {
        backward.execute_dft_c2r(buf.cx1_ptr(), out.data());
        fftw::normalize<T>(out, fct);
      } else {
        backward.execute_dft_c2r(buf.cx1_ptr(), buf.real_ptr());
        fftw::normalize<T>(buf.real, fct);
        std::copy(buf.real.begin(), buf.real.end(), out.begin());
      }

    } else if constexpr (Mode == ConvMode::Same) {
      const size_t padding = k.size() / 2;

      // Always use buf.real_ptr() for Same mode to ensure buffer size is
      // correct The FFTW plan is for real_sz which is larger than out.size(),
      // so we can't write directly to out.data()
      backward.execute_dft_c2r(buf.cx1_ptr(), buf.real_ptr());
      fftw::normalize<T>(buf.real, fct);

      const auto *const real_ptr = buf.real_ptr() + padding;
      std::copy(real_ptr, real_ptr + out.size(), out.data());
    }
  }

  template <ConvMode Mode = ConvMode::Full>
  void oaconvolve(View a, View k, MutView out) {
    assert(buf.real.size() == internal::get_optimal_fft_size(k.size()));
    std::fill(out.begin(), out.end(), 0);

    const size_t fft_size = buf.real.size();
    const size_t step_size = fft_size - (k.size() - 1);

    // forward fft of kernel and save to complex2
    internal::copy_to_padded_buffer<T>(k, buf.real);
    forward.execute_dft_r2c(buf.real_ptr(), buf.cx2_ptr());

    const auto fct = static_cast<T>(1. / fft_size);

    if constexpr (Mode == ConvMode::Full) {
      assert(a.size() + k.size() - 1 == out.size());

      // create forward/backward ffts for x
      for (size_t pos = 0; pos < a.size(); pos += step_size) {
        size_t len = std::min<size_t>(a.size() - pos, step_size); // bound check

        internal::copy_to_padded_buffer<T>(a.subspan(pos, len), buf.real);
        forward.execute_dft_r2c(buf.real_ptr(), buf.cx1_ptr());
        internal::multiply_cx<T>(buf.cx1, buf.cx2, buf.cx1);
        backward.execute_dft_c2r(buf.cx1_ptr(), buf.real_ptr());

        // normalize output and add to result
        fftw::normalize_add<T>(out.subspan(pos), buf.real, fct);
      }

    } else if constexpr (Mode == ConvMode::Same) {
      assert(a.size() == out.size());

      const size_t padding = k.size() / 2;

      for (size_t pos = 0; pos < a.size(); pos += step_size) {
        size_t len = std::min<size_t>(a.size() - pos, step_size); // bound check

        internal::copy_to_padded_buffer<T>(a.subspan(pos, len), buf.real);
        forward.execute_dft_r2c(buf.real_ptr(), buf.cx1_ptr());
        internal::multiply_cx<T>(buf.cx1, buf.cx2, buf.cx1);
        backward.execute_dft_c2r(buf.cx1_ptr(), buf.real_ptr());

        // normalize output and add to result
        if (pos < padding) {
          fftw::normalize_add<T>(out.subspan(pos),
                                 buf.real_span().subspan(padding - pos), fct);
        } else {
          fftw::normalize_add<T>(out.subspan(pos - padding),
                                 buf.real_span().subspan(0), fct);
        }
      }
    } else {
      static_assert(Mode == ConvMode::Full || Mode == ConvMode::Same,
                    "Unsupported mode.");
    }
  }

  void oaconvolve_same(View a, View k, MutView out) {
    oaconvolve<ConvMode::Same>(a, k, out);
  }
  void oaconvolve_full(View a, View k, MutView out) {
    oaconvolve<ConvMode::Full>(a, k, out);
  }
};

// 1D convolution using the FFT
//
// Optimizations:
//    * Cache fftw_plan
//    * Reuse buffers (no malloc on second call to the same convolution size)
// https://en.wikipedia.org/w/index.php?title=Convolution#Fast_convolution_algorithms
template <Floating T, ConvMode Mode = ConvMode::Same,
          int PlannerFlag = FFTW_ESTIMATE>
void convolve_fftw(const std::span<const T> input,
                   const std::span<const T> kernel, std::span<T> output) {
  // length of the real arrays, including the final convolution output
  const size_t padded_length = input.size() + kernel.size() - 1;
  auto &plan = FFTConvEngine<T, PlannerFlag>::get(padded_length);
  plan.template convolve<Mode>(input, kernel, output);
}

/**
1D Overlap-Add convolution
// input_size >> kernel_size

For "Full" mode, output_size >= input_size + kernel_size - 1
For "Same" mode, output_size == input_size

1. Split arr into blocks of step_size.
2. convolve with kernel using fft of length N.
3. add blocks together
 */
template <Floating T, ConvMode Mode = ConvMode::Same,
          int PlannerFlag = FFTW_ESTIMATE>
void oaconvolve_fftw(std::span<const T> input, std::span<const T> kernel,
                     std::span<T> output) {
  auto &plan = FFTConvEngine<T, PlannerFlag>::get_for_ksize(kernel.size());
  plan.template oaconvolve<Mode>(input, kernel, output);
}

} // namespace fftconv

// NOLINTEND(*-reinterpret-cast, *-const-cast, *-pointer-arithmetic)
