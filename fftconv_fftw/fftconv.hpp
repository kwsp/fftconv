// Author: Taylor Nie
// 2022 - 2024
// https://github.com/kwsp/fftconv
#pragma once

#include <array>
#include <cassert>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>

#include <fftw3.h>
#include <type_traits>

namespace fftconv {

template <typename T>
concept FloatOrDouble = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
concept FFTWComplex =
    std::is_same_v<T, fftw_complex> || std::is_same_v<T, fftwf_complex>;

template <typename T>
concept FFTWBufferSupported = FloatOrDouble<T> || FFTWComplex<T>;

namespace internal {

static std::mutex *fftconv_fftw_mutex = nullptr;

// static int nextpow2(int x) { return 1 << (int)(std::log2(x) + 1); }
// Lookup table of {max_filter_size, optimal_fft_size}
constexpr std::array<std::array<size_t, 2>, 9> _optimal_oa_fft_size{
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
  for (const auto &pair : _optimal_oa_fft_size) {
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

// In memory cache with key type K and value type V
// additionally accepts a mutex to guard the V constructor
template <class K, class V> auto get_cached_vlock(K key, std::mutex *V_mutex) {
  static thread_local std::unordered_map<K, std::unique_ptr<V>> _cache;

  auto &val = _cache[key];
  if (val == nullptr) {
    // Using unique_lock here for RAII locking since the mutex is optional.
    // If we have a non-optional mutex, prefer scoped_lock
    const auto lock = V_mutex == nullptr ? std::unique_lock<std::mutex>{}
                                         : std::unique_lock(*V_mutex);

    val = std::make_unique<V>(key);
  }

  return val.get();
}

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

template <typename Tin, typename Tout>
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

template <FFTWComplex T>
inline void elementwise_multiply(std::span<const T> complex1,
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

/// @brief Encapsulate FFTW allocated buffer
/// @tparam T
template <FFTWBufferSupported T> class fftw_buffer {
public:
  fftw_buffer(const fftw_buffer &) = delete; // Copy constructor
  fftw_buffer(fftw_buffer &&) = delete;      // Move constructor
  auto operator=(const fftw_buffer &)
      -> fftw_buffer & = delete;                            // Copy assignment
  auto operator=(fftw_buffer &&) -> fftw_buffer & = delete; // Move assignment

  explicit fftw_buffer(size_t size) : m_size(size), m_data(fftw_alloc(size)) {}

  fftw_buffer(std::initializer_list<T> list) : fftw_buffer(list.size()) {
    std::copy(list.begin(), list.end(), this->begin());
  }

  // Destructor
  ~fftw_buffer() {
    if (m_data != nullptr) {
      if constexpr (std::is_same_v<T, double> ||
                    std::is_same_v<T, fftw_complex>) {
        fftw_free(m_data);
      } else if constexpr (std::is_same_v<T, float> ||
                           std::is_same_v<T, fftwf_complex>) {
        fftwf_free(m_data);
      }
    }
  }

  using iterator = T *;
  using const_iterator = const T *;
  auto begin() noexcept { return iterator(m_data); }
  auto end() noexcept { return iterator(m_data + m_size); }
  [[nodiscard]] auto begin() const noexcept { return iterator(m_data); }
  [[nodiscard]] auto end() const noexcept { return iterator(m_data + m_size); }

  auto operator[](size_t idx) -> auto & { return m_data[idx]; }
  auto operator[](size_t idx) const -> const auto & { return m_data[idx]; }

  [[nodiscard]] auto size() const { return m_size; }
  [[nodiscard]] auto data() const { return m_data; }

  // return a span-like view of the data
  [[nodiscard]] auto cspan() const -> std::span<const T> {
    return {m_data, m_size}; // Create a span pointing to data with size
  }
  [[nodiscard]] auto span() -> std::span<T> {
    return {m_data, m_size}; // Create a span pointing to data with size
  }

private:
  static constexpr auto fftw_alloc(const size_t size) {
    if constexpr (std::is_same_v<T, double>) {
      return static_cast<T *>(fftw_alloc_real(size));
    } else if constexpr (std::is_same_v<T, fftw_complex>) {
      return static_cast<T *>(fftw_alloc_complex(size));
    } else if constexpr (std::is_same_v<T, float>) {
      return static_cast<T *>(fftwf_alloc_real(size));
    } else if constexpr (std::is_same_v<T, fftwf_complex>) {
      return static_cast<T *>(fftwf_alloc_complex(size));
    }
  }

  T *m_data;
  size_t m_size{};
};

template <FloatOrDouble> struct fftw_plans_traits {};

template <> struct fftw_plans_traits<float> {
  using Real = float;
  using Complex = fftwf_complex;
};

template <> struct fftw_plans_traits<double> {
  using Real = double;
  using Complex = fftw_complex;
};

template <FloatOrDouble> struct fftw_plans_1d {};

template <> struct fftw_plans_1d<float> {
  using real_t = typename fftw_plans_traits<float>::Real;
  using complex_t = typename fftw_plans_traits<float>::Complex;

  explicit fftw_plans_1d(const fftw_buffer<real_t> &real,
                         const fftw_buffer<complex_t> &complex)
      : plan_f(fftwf_plan_dft_r2c_1d(static_cast<int>(real.size()), real.data(),
                                     complex.data(), FFTW_ESTIMATE)),
        plan_b(fftwf_plan_dft_c2r_1d(static_cast<int>(real.size()),
                                     complex.data(), real.data(),
                                     FFTW_ESTIMATE)) {}
  fftw_plans_1d(fftw_plans_1d &&other) = delete;
  fftw_plans_1d(const fftw_plans_1d &other) = delete;
  auto operator=(fftw_plans_1d &&other) -> fftw_plans_1d = delete;
  auto operator=(const fftw_plans_1d &other) -> fftw_plans_1d = delete;

  ~fftw_plans_1d() {
    if (plan_f != nullptr) {
      fftwf_destroy_plan(plan_f);
    }
    if (plan_b != nullptr) {
      fftwf_destroy_plan(plan_b);
    }
  }

  void forward(fftw_buffer<real_t> &real,
               fftw_buffer<complex_t> &complex) const {
    fftwf_execute_dft_r2c(plan_f, real.data(), complex.data());
  }

  void backward(fftw_buffer<complex_t> &complex,
                fftw_buffer<real_t> &real) const {
    fftwf_execute_dft_c2r(plan_b, complex.data(), real.data());
  }

  fftwf_plan plan_f;
  fftwf_plan plan_b;
};

template <> struct fftw_plans_1d<double> {
  using real_t = typename fftw_plans_traits<double>::Real;
  using complex_t = typename fftw_plans_traits<double>::Complex;

  explicit fftw_plans_1d(const fftw_buffer<real_t> &real,
                         const fftw_buffer<complex_t> &complex)
      : plan_f(fftw_plan_dft_r2c_1d(static_cast<int>(real.size()), real.data(),
                                    complex.data(), FFTW_ESTIMATE)),
        plan_b(fftw_plan_dft_c2r_1d(static_cast<int>(real.size()),
                                    complex.data(), real.data(), FFTW_ESTIMATE))

  {}
  fftw_plans_1d(fftw_plans_1d &&other) = delete;
  fftw_plans_1d(const fftw_plans_1d &other) = delete;
  auto operator=(fftw_plans_1d &&other) -> fftw_plans_1d = delete;
  auto operator=(const fftw_plans_1d &other) -> fftw_plans_1d = delete;

  ~fftw_plans_1d() {
    if (plan_f != nullptr) {
      fftw_destroy_plan(plan_f);
    }
    if (plan_b != nullptr) {
      fftw_destroy_plan(plan_b);
    }
  }

  void forward(fftw_buffer<real_t> &real,
               fftw_buffer<complex_t> &complex) const {
    fftw_execute_dft_r2c(plan_f, real.data(), complex.data());
  }
  void backward(fftw_buffer<complex_t> &complex,
                fftw_buffer<real_t> &real) const {
    fftw_execute_dft_c2r(plan_b, complex.data(), real.data());
  }

  fftw_plan plan_f;
  fftw_plan plan_b;
};

template <FloatOrDouble> struct fftw_plans_many {};

template <> class fftw_plans_many<float> {
public:
  using real_t = typename fftw_plans_traits<float>::Real;
  using complex_t = typename fftw_plans_traits<float>::Complex;

  fftw_plans_many(const fftw_plans_many &) = delete;
  fftw_plans_many(fftw_plans_many &&) = delete;
  auto operator=(fftw_plans_many &&) -> fftw_plans_many & = delete;
  auto operator=(const fftw_plans_many &) -> fftw_plans_many & = delete;

  fftw_plans_many(const fftw_buffer<real_t> &real,
                  const fftw_buffer<complex_t> &complex, int n_arrays) {
    const int real_size = static_cast<int>(real.size());
    int rank = 1;
    int stride = 1;
    plan_f = fftwf_plan_many_dft_r2c(
        rank, &real_size, n_arrays, real.data(), nullptr, stride,
        static_cast<int>(real.size()), complex.data(), nullptr, stride,
        static_cast<int>(complex.size()), FFTW_ESTIMATE);

    plan_b = fftwf_plan_many_dft_c2r(
        rank, &real_size, n_arrays, complex.data(), nullptr, stride,
        static_cast<int>(complex.size()), real.data(), nullptr, stride,
        static_cast<int>(real.size()), FFTW_ESTIMATE);
  }

  ~fftw_plans_many() {
    if (plan_f != nullptr) {
      fftw_free(plan_f);
    }
    if (plan_b != nullptr) {
      fftw_free(plan_b);
    }
  }

  constexpr void forward(fftw_buffer<real_t> &real,
                         fftw_buffer<complex_t> &complex) const {
    fftwf_execute_dft_r2c(plan_f, real.data(), complex.data());
  }

  constexpr void backward(fftw_buffer<complex_t> &complex,
                          fftw_buffer<real_t> &real) const {
    fftwf_execute_dft_c2r(plan_b, complex.data(), real.data());
  }

private:
  fftwf_plan plan_f;
  fftwf_plan plan_b;
};

template <> class fftw_plans_many<double> {
public:
  using real_t = typename fftw_plans_traits<double>::Real;
  using complex_t = typename fftw_plans_traits<double>::Complex;

  fftw_plans_many(const fftw_plans_many &) = delete;
  fftw_plans_many(fftw_plans_many &&) = delete;
  auto operator=(fftw_plans_many &&) -> fftw_plans_many & = delete;
  auto operator=(const fftw_plans_many &) -> fftw_plans_many & = delete;

  fftw_plans_many(const fftw_buffer<real_t> &real,
                  const fftw_buffer<complex_t> &complex, int n_arrays) {
    const int real_size = static_cast<int>(real.size());
    int rank = 1;
    int stride = 1;
    plan_f = fftw_plan_many_dft_r2c(
        rank, &real_size, n_arrays, real.data(), nullptr, stride,
        static_cast<int>(real.size()), complex.data(), nullptr, stride,
        static_cast<int>(complex.size()), FFTW_ESTIMATE);

    plan_b = fftw_plan_many_dft_c2r(
        rank, &real_size, n_arrays, complex.data(), nullptr, stride,
        static_cast<int>(complex.size()), real.data(), nullptr, stride,
        static_cast<int>(real.size()), FFTW_ESTIMATE);
  }

  ~fftw_plans_many() {
    if (plan_f != nullptr) {
      fftw_free(plan_f);
    }
    if (plan_b != nullptr) {
      fftw_free(plan_b);
    }
  }

  constexpr void forward(fftw_buffer<real_t> &real,
                         fftw_buffer<complex_t> &complex) const {
    fftw_execute_dft_r2c(plan_f, real.data(), complex.data());
  }

  constexpr void backward(fftw_buffer<complex_t> &complex,
                          fftw_buffer<real_t> &real) const {
    fftw_execute_dft_c2r(plan_b, complex.data(), real.data());
  }

private:
  fftw_plan plan_f;
  fftw_plan plan_b;
};
} // namespace internal

// Since FFTW planners are not thread-safe, you can pass a pointer to a
// std::mutex to fftconv and all calls to the planner with be guarded by the
// mutex.
void use_fftw_mutex(std::mutex *fftw_mutex) {
  internal::fftconv_fftw_mutex = fftw_mutex;
}
auto get_fftw_mutex() -> std::mutex * { return internal::fftconv_fftw_mutex; };

// Reference implementation of fft convolution with minimal optimizations
void convolve_fftw_ref(std::span<const double> arr1,
                       std::span<const double> arr2, std::span<double> res);

// fft_plans manages the memory of the forward and backward fft plans
// and the fftw buffers
//
// Not using half complex transforms before it's non-trivial to do complex
// multiplications with FFTW's half complex format
template <FloatOrDouble Real> class fftconv_plans {
public:
  using Complex = typename internal::fftw_plans_traits<Real>::Complex;

  // Get the fftconv_plans object for a specific kernel size
  static auto get(const size_t padded_length) -> auto & {
    thread_local static auto fftconv_plans_cache =
        internal::get_cached_vlock<size_t, fftconv_plans<Real>>;
    return *fftconv_plans_cache(padded_length, get_fftw_mutex());
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
  void convolve(const std::span<const Real> arr,
                const std::span<const Real> kernel, const std::span<Real> res) {
    const auto real_span = real.span();

    // Copy a to buffer
    internal::copy_to_padded_buffer(arr, real_span);

    // A = fft(a)
    plans.forward(real, complex1);

    // Copy b to buffer
    internal::copy_to_padded_buffer(kernel, real_span);

    // B = fft(b)
    plans.forward(real, complex2);

    // Complex elementwise multiple, A = A * B
    internal::elementwise_multiply(complex1.cspan(), complex2.cspan(),
                                   complex1.span());

    // a = ifft(A)
    plans.backward(complex1, real);

    // divide each result elem by real_sz
    for (size_t i = 0; i < real.size(); ++i) {
      real[i] /= real.size();
    }

    std::copy(real_span.begin(), real_span.end(), res.begin());
  }

  void oaconvolve(const std::span<const Real> arr,
                  const std::span<const Real> kernel, std::span<Real> res) {
    assert(real.size() == internal::get_optimal_fft_size(kernel.size()));
    const auto fft_size = real.size();
    const auto step_size = fft_size - (kernel.size() - 1);

    // forward fft of kernel and save to complex2
    const auto real_span = real.span();
    internal::copy_to_padded_buffer(kernel, real_span);
    plans.forward(real, complex2);

    // create forward/backward ffts for x
    // Normalization factor
    const double fct = 1. / static_cast<Real>(fft_size);
    for (size_t pos = 0; pos < arr.size(); pos += step_size) {
      size_t len = std::min<size_t>(arr.size() - pos, step_size); // bound check

      internal::copy_to_padded_buffer(arr.subspan(pos, len), real_span);
      plans.forward(real, complex1);

      internal::elementwise_multiply(complex1.cspan(), complex2.cspan(),
                                     complex1.span());

      plans.backward(complex1, real);

      // normalize output and add to result
      len = std::min<size_t>(res.size() - pos, fft_size);
      for (size_t i = 0; i < len; ++i) {
        res[pos + i] += real[i] * fct;
      }
    }
  }

private:
  // FFTW buffers
  internal::fftw_buffer<Real> real;
  internal::fftw_buffer<Complex> complex1;
  internal::fftw_buffer<Complex> complex2;

  // FFTW plans
  internal::fftw_plans_1d<Real> plans;
};

// fftconv_plans manages the memory of the forward and backward fft plans
// and the fftw buffers
// Plans are for the FFTW New-Array Execute Functions
// https://www.fftw.org/fftw3_doc/New_002darray-Execute-Functions.html
template <FloatOrDouble Real> class fftconv_plans_advanced {
  using Complex = internal::fftw_plans_traits<Real>::Complex;

public:
  static auto get(const size_t padded_length, const int n_arrays)
      -> fftconv_plans_advanced<Real> & {
    static thread_local std::unordered_map<
        size_t, std::unique_ptr<fftconv_plans_advanced<Real>>>
        _cache;
    const size_t _hash = (padded_length << 4) ^ n_arrays;

    auto &plan = _cache[_hash];
    if (plan == nullptr || plan->kernel_real.size() != padded_length) {
      plan = std::make_unique<fftconv_plans_advanced<Real>>(padded_length,
                                                            n_arrays);
    }
    return *plan;
  }

  // Use advanced interface
  explicit fftconv_plans_advanced(const size_t padded_length,
                                  const size_t n_arrays = 1)
      : n_arrays(n_arrays), kernel_real(padded_length),
        kernel_cx(padded_length / 2 + 1), signal_real(padded_length * n_arrays),
        signal_cx(padded_length * n_arrays),
        plan_kernel(kernel_real, kernel_cx),
        plan_signal(signal_real, signal_cx, n_arrays) {
    // `howmany` is the (nonnegative) number of transforms to compute.
    // - The resulting plan computs `howmany` transforms, where the input of
    //   the k-th transform is at location `in+k*idist`
    // - Each of `howmany` has rank `rank` and size `n`
  }

  void oaconvolve(const std::span<const Real> arr,
                  const std::span<const Real> kernel, std::span<Real> res) {

    auto kernel_real_span = kernel_real.span();
    auto signal_real_span = signal_real.span();

    // Forward kernel
    internal::copy_to_padded_buffer(kernel, kernel_real_span);
    plan_kernel.forward(kernel_real, kernel_cx);

    // Forward signal
    const auto fft_size = kernel_real.size();
    const auto step_size = fft_size - (kernel.size() - 1);
    for (size_t pos = 0, idx = 0; pos < arr.size(); pos += step_size, idx++) {
      size_t len = std::min<size_t>(arr.size() - pos, step_size); // bound check
      internal::copy_to_padded_buffer(arr.subspan(pos, len), signal_real_span);
    }
    plan_signal.forward(signal_real, signal_cx);

    // Complex multiply
    auto signal_cx_span = signal_cx.span();
    auto kernel_cx_span = kernel_cx.cspan();
    for (int i = 0; i < n_arrays; ++i) {
      auto signal_cx_subspan = signal_cx_span.subspan(i * kernel_cx.size());
      internal::elementwise_multiply(
          std::span<const Complex>(signal_cx_subspan), kernel_cx_span,
          signal_cx_subspan);
    }

    // Backward
    plan_signal.backward(signal_cx, signal_real);

    // Copy results
    const double fct = 1. / kernel_real.size();
    for (size_t pos = 0, idx = 0; pos < res.size(); pos += step_size, idx++) {
      size_t len = std::min<size_t>(res.size() - pos, fft_size); // bound check
      auto res_subspan = res.subspan(pos, len);

      const auto size2 = std::min<size_t>(kernel_real.size(), len);
      const size_t pos2 = idx * kernel_real.size();
      for (int i = 0; i < size2; ++i) {
        res_subspan[i] += signal_real[pos + i] * fct;
      }
    }
  }

private:
  // FFTW buffer sizes
  size_t n_arrays;

  // FFTW padded buffers
  internal::fftw_buffer<Real> signal_real;
  internal::fftw_buffer<Complex> signal_cx;
  internal::fftw_buffer<Real> kernel_real;
  internal::fftw_buffer<Complex> kernel_cx;

  // FFTW plans
  internal::fftw_plans_1d<Real> plan_kernel;
  internal::fftw_plans_many<Real> plan_signal;
};

// 1D convolution using the FFT
// Optimizations:
//    * Cache fftw_plan
//    * Reuse buffers (no malloc on second call to the same convolution size)
// https://en.wikipedia.org/w/index.php?title=Convolution#Fast_convolution_algorithms

template <FloatOrDouble Real>
void convolve_fftw(const std::span<const Real> arr1,
                   const std::span<const Real> arr2, std::span<Real> res) {
  // length of the real arrays, including the final convolution output
  const size_t padded_length = arr1.size() + arr2.size() - 1;

  // Get cached plans
  auto &plan = fftconv_plans<Real>::get(padded_length);

  // Execute FFT convolution and copy normalized result
  plan.convolve(arr1, arr2, res);
}

// 1D Overlap-Add convolution of x and h
//
// arr is a long signal
// kernel is a kernel, arr_size >> kernel_size
// res is the results buffer. res_size >= arr_size + kernel_size - 1
//
// 1. Split arr into blocks of step_size.
// 2. convolve with kernel using fft of length N.
// 3. add blocks together
template <FloatOrDouble Real>
void oaconvolve_fftw(const std::span<const Real> arr,
                     const std::span<const Real> kernel, std::span<Real> res) {
  // Get cached plans
  auto &plan = fftconv_plans<Real>::get_for_kernel(kernel);

  // Execute FFT convolution and copy normalized result
  plan.oaconvolve(arr, kernel, res);
}

// reference implementation of fftconv with no optimizations
void convolve_fftw_ref(const std::span<const double> arr1,
                       const std::span<const double> arr2,
                       std::span<double> res) {
  // length of the real arrays, including the final convolution output
  const size_t padded_length = arr1.size() + arr2.size() - 1;
  // length of the complex arrays
  const auto complex_length = padded_length / 2 + 1;

  // Allocate fftw buffers for a
  double *a_buf = fftw_alloc_real(padded_length);
  fftw_complex *A_buf = fftw_alloc_complex(complex_length);

  // Compute forward fft plan
  fftw_plan plan_forward = fftw_plan_dft_r2c_1d(static_cast<int>(padded_length),
                                                a_buf, A_buf, FFTW_ESTIMATE);

  // Copy a to buffer
  internal::copy_to_padded_buffer(arr1,
                                  std::span<double>(a_buf, padded_length));

  // Compute Fourier transform of vector a
  fftw_execute_dft_r2c(plan_forward, a_buf, A_buf);

  // Allocate fftw buffers for b
  double *b_buf = fftw_alloc_real(padded_length);
  fftw_complex *B_buf = fftw_alloc_complex(complex_length);

  // Copy b to buffer
  internal::copy_to_padded_buffer(arr2,
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
  internal::elementwise_multiply(
      std::span<fftw_complex const>(A_buf, complex_length),
      std::span<fftw_complex const>(B_buf, complex_length),
      std::span(input_buffer, complex_length));

  // A_buf becomes input to inverse conv
  fftw_execute_dft_c2r(plan_backward, input_buffer, output_buffer);

  // Normalize output
  for (int i = 0; i < std::min<size_t>(padded_length, res.size()); i++) {
    res[i] = output_buffer[i] / static_cast<double>(padded_length);
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

template <FloatOrDouble Real>
void oaconvolve_fftw_advanced(const std::span<const Real> arr,
                              const std::span<const Real> kernel,
                              std::span<Real> res) {
  // more optimal size for each fft
  const auto fft_size = internal::get_optimal_fft_size(kernel.size());
  const auto step_size = fft_size - (kernel.size() - 1);
  const auto n_arrays = arr.size() / step_size + 1; // last batch zero pad

  auto &plans = fftconv_plans_advanced<Real>::get(fft_size, n_arrays);
  plans.oaconvolve(arr, kernel, res);
}

} // namespace fftconv
