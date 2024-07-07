// Author: Taylor Nie
// 2022
// https://github.com/kwsp/fftconv

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <fftw3.h>

#include "fftconv.hpp"

namespace {

std::mutex *fftconv_fftw_mutex = nullptr;

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

// Given a filter_size, return the optimal fft size for the overlap-add
// convolution method
auto get_optimal_fft_size(const size_t filter_size) -> size_t {
  for (const auto &pair : _optimal_oa_fft_size) {
    if (filter_size < pair[0]) {
      return pair[1];
    }
  }
  return 8192;
}

// Copy data from src to dst and padded the extra with zero
// dst_size must be greater than src_size
template <class T>
inline void copy_to_padded_buffer(const T *src, const size_t src_size, T *dst,
                                  const size_t dst_size) {
  assert(src_size <= dst_size);
  std::copy(src, src + src_size, dst);
  std::fill(dst + src_size, dst + dst_size, 0);
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

inline void elementwise_multiply(const fftw_complex *a, const fftw_complex *b,
                                 const size_t length, fftw_complex *result) {
  // implement naive complex multiply. This is much faster than the
  // std version, but doesn't handle infinities.
  // https://stackoverflow.com/questions/49438158/why-is-muldc3-called-when-two-stdcomplex-are-multiplied

  for (size_t i = 0; i < length; ++i) {
    const double a_1 = a[i][0];
    const double a_2 = a[i][1];
    const double b_1 = b[i][0];
    const double b_2 = b[i][1];
    result[i][0] = a_1 * b_1 - a_2 * b_2;
    result[i][1] = a_2 * b_1 + a_1 * b_2;
  }
}

// enum class fft_direction { Forward = 0b01, Backward = 0b10 };

// struct fft_plan {
// fftw_plan plan;
// double *real_buf;
// fftw_complex *complex_buf;

//// Padded length must be an even power of 2
//// The lowest 2 bits are used as direction flags
// fft_plan(size_t padded_length) {
//// length of the complex arrays
// size_t complex_length = padded_length / 2 + 1;
// real_buf = (double *)fftw_alloc_real(padded_length);
// complex_buf = (fftw_complex *)fftw_alloc_complex(complex_length);

//// Check direction
// if ((int)padded_length & (int)fft_direction::Forward)
// plan = fftw_plan_dft_r2c_1d(padded_length, real_buf, complex_buf,
// FFTW_ESTIMATE);
// else
// plan = fftw_plan_dft_c2r_1d(padded_length, complex_buf, real_buf,
// FFTW_ESTIMATE);
//}

//~fft_plan() {
// fftw_destroy_plan(plan);
// fftw_free(real_buf);
// fftw_free(complex_buf);
//}
//};

template <typename T>
concept is_fftw_buffer_supported = requires() {
  std::is_same_v<T, float> || std::is_same_v<T, double> ||
      std::is_same_v<T, fftw_complex> || std::is_same_v<T, fftwf_complex>;
};

/// @brief Encapsulate FFTW allocated buffer
/// @tparam T
template <is_fftw_buffer_supported T> class fftw_buffer {
public:
  fftw_buffer(const fftw_buffer &) = delete;
  fftw_buffer(fftw_buffer &&) = delete;
  auto operator=(const fftw_buffer &) -> fftw_buffer & = delete;
  auto operator=(fftw_buffer &&) -> fftw_buffer & = delete;

  explicit fftw_buffer(size_t size) : m_size(size) {
    if constexpr (std::is_same_v<T, double>) {
      m_data = static_cast<T *>(fftw_alloc_real(size));
    } else if constexpr (std::is_same_v<T, fftw_complex>) {
      m_data = static_cast<T *>(fftw_alloc_complex(size));
    } else if constexpr (std::is_same_v<T, float>) {
      m_data = static_cast<T *>(fftwf_alloc_real(size));
    } else if (std::is_same_v<T, fftwf_complex>) {
      m_data = static_cast<T *>(fftwf_alloc_complex(size));
    } else {
      // Throw an exception or handle error in case of invalid type
      throw std::invalid_argument("Unsupported type for fftw_buffer.");
    }
  }
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

private:
  size_t m_size;
  T *m_data;
};

// fftconv_plans manages the memory of the forward and backward fft plans
// and the fftw buffers
// Plans are for the FFTW New-Array Execute Functions
// https://www.fftw.org/fftw3_doc/New_002darray-Execute-Functions.html
//
// Not using half complex transforms before it's non-trivial to do complex
// multiplications with FFTW's half complex format
class fftconv_plans {
private:
  // FFTW buffer sizes
  const size_t real_sz;
  const size_t complex_sz;

  // FFTW buffers corresponding to the above plans
  double *real_buf;
  fftw_complex *complex_buf_a;
  fftw_complex *complex_buf_b;

  // FFTW plans
  fftw_plan plan_f; // forward
  fftw_plan plan_b; // backward

public:
  // Constructors
  // Compute the fftw plans and allocate buffers
  explicit fftconv_plans(int padded_length)
      : real_sz(padded_length), complex_sz(padded_length / 2 + 1) {

    if (fftconv_fftw_mutex != nullptr) {
      fftconv_fftw_mutex->lock();
    }

    real_buf = fftw_alloc_real(real_sz);
    complex_buf_a = fftw_alloc_complex(complex_sz);
    complex_buf_b = fftw_alloc_complex(complex_sz);

    plan_f = fftw_plan_dft_r2c_1d(padded_length, real_buf, complex_buf_a,
                                  FFTW_ESTIMATE);
    plan_b = fftw_plan_dft_c2r_1d(padded_length, complex_buf_a, real_buf,
                                  FFTW_ESTIMATE);

    if (fftconv_fftw_mutex != nullptr) {
      fftconv_fftw_mutex->unlock();
    }
  }

  fftconv_plans() = delete;                      // default constructor
  fftconv_plans(fftconv_plans &&) = delete;      // move constructor
  fftconv_plans(const fftconv_plans &) = delete; // copy constructor
  auto operator=(const fftconv_plans)
      -> fftconv_plans & = delete; // copy assignment
  auto operator=(fftconv_plans &&)
      -> fftconv_plans & = delete; // move assignment

  // Destructor
  ~fftconv_plans() {
    fftw_free(real_buf);
    fftw_free(complex_buf_a);
    fftw_free(complex_buf_b);
    fftw_destroy_plan(plan_f);
    fftw_destroy_plan(plan_b);
  }

  void set_real_buf(const double *inp, size_t size) {
    copy_to_padded_buffer(inp, size, this->real_buf, this->real_sz);
  }
  [[nodiscard]] constexpr auto get_real_buf() const { return this->real_buf; }
  [[nodiscard]] constexpr auto get_real_sz() const { return this->real_sz; }

  void forward_a() { fftw_execute_dft_r2c(plan_f, real_buf, complex_buf_a); }
  void forward_b() { fftw_execute_dft_r2c(plan_f, real_buf, complex_buf_b); }
  void backward() { fftw_execute_dft_c2r(plan_b, complex_buf_a, real_buf); }
  void normalize() {
    for (size_t i = 0; i < real_sz; ++i) {
      real_buf[i] /= real_sz;
    }
  }

  // Complex element-wise multiply a and b, save results to a
  void complex_multiply_to_a() {
    elementwise_multiply(complex_buf_a, complex_buf_b, complex_sz,
                         complex_buf_a);
  }

  // Convolve real arrays a and b
  // Results saved in real_buf
  void execute_conv(const double *a, const size_t a_size, const double *b,
                    const size_t b_size) {
    set_real_buf(a, a_size); // Copy a to buffer
    forward_a();             // A = fft(a)
    set_real_buf(b, b_size); // Copy b to buffer
    forward_b();             // B = fft(b)
    complex_multiply_to_a(); // Complex elementwise multiple, A = A * B
    backward();              // a = ifft(A)
    normalize();             // divide each result elem by real_sz
  }
};

constexpr auto fftconv_plans_cache =
    fftconv::_get_cached<size_t, fftconv_plans>;

// fftconv_plans manages the memory of the forward and backward fft plans
// and the fftw buffers
// Plans are for the FFTW New-Array Execute Functions
// https://www.fftw.org/fftw3_doc/New_002darray-Execute-Functions.html
class fftconv_plans_advanced {

public:
  fftconv_plans_advanced(const fftconv_plans_advanced &) = default;
  fftconv_plans_advanced(fftconv_plans_advanced &&) = delete;
  auto operator=(const fftconv_plans_advanced &)
      -> fftconv_plans_advanced & = delete;
  auto operator=(fftconv_plans_advanced &&)
      -> fftconv_plans_advanced & = delete;

  // Use advanced interface
  explicit fftconv_plans_advanced(const int padded_length,
                                  const int n_arrays = 1)
      : real_sz(padded_length), complex_sz(padded_length / 2 + 1),
        n_arrays(n_arrays) {

    if (fftconv_fftw_mutex != nullptr) {
      fftconv_fftw_mutex->lock();
    }

    real_buf_signal = fftw_alloc_real(static_cast<size_t>(real_sz) * n_arrays);
    cx_buf_signal =
        fftw_alloc_complex(static_cast<size_t>(complex_sz) * n_arrays);

    real_buf_kernel = fftw_alloc_real(real_sz);
    cx_buf_kernel = fftw_alloc_complex(complex_sz);

    // `howmany` is the (nonnegative) number of transforms to compute.
    // - The resulting plan computs `howmany` transforms, where the input of
    //   the k-th transform is at location `in+k*idist`
    // - Each of `howmany` has rank `rank` and size `n`
    //
    // int howmany;

    int rank = 1; // Dimensions of each transform.
    const int n[] = {static_cast<int>(real_sz)};

    double *in = real_buf_signal;
    fftw_complex *out = cx_buf_signal;

    // {i,o}nembed must be arrays of size `rank`
    // or NULL (equivalent to passing `n`)
    const int *nembed = nullptr;

    const int stride = 1;
    const int idist = real_sz;
    const int odist = complex_sz;

    plan_forward_kernel = fftw_plan_dft_r2c_1d(real_sz, real_buf_kernel,
                                               cx_buf_kernel, FFTW_ESTIMATE);

    plan_forward_signal =
        fftw_plan_many_dft_r2c(rank, n, n_arrays, in, nembed, stride, idist,
                               out, nembed, stride, odist, FFTW_ESTIMATE);

    plan_backward_signal =
        fftw_plan_many_dft_c2r(rank, n, n_arrays, out, nembed, stride, odist,
                               in, nembed, stride, idist, FFTW_ESTIMATE);

    if (fftconv_fftw_mutex != nullptr) {
      fftconv_fftw_mutex->unlock();
    }
  }

  ~fftconv_plans_advanced() {
    fftw_free(real_buf_signal);
    fftw_free(cx_buf_signal);

    fftw_free(real_buf_kernel);
    fftw_free(cx_buf_kernel);

    fftw_destroy_plan(plan_forward_kernel);
    fftw_destroy_plan(plan_forward_signal);
    fftw_destroy_plan(plan_backward_signal);
  }

  void set_kernel(const double *arr, size_t size) const {
    assert(size <= real_sz);
    copy_to_padded_buffer(arr, size, real_buf_kernel, real_sz);
  }

  // i-th `howmany`
  void set_signal(const double *arr, const size_t size,
                  const size_t idx) const {
    assert(size <= real_sz);
    assert(idx <= n_arrays);
    copy_to_padded_buffer(arr, size, real_buf_signal + idx * real_sz, real_sz);
  }

  void forward_kernel() const {
    fftw_execute_dft_r2c(this->plan_forward_kernel, real_buf_kernel,
                         cx_buf_kernel);
  }
  void forward_signal() const {
    fftw_execute_dft_r2c(this->plan_forward_signal, real_buf_signal,
                         cx_buf_signal);
  }
  void backward() const {
    fftw_execute_dft_c2r(this->plan_backward_signal, cx_buf_signal,
                         real_buf_signal);
  }

  void complex_multiply() const {
    for (int i = 0; i < n_arrays; ++i) {
      elementwise_multiply(cx_buf_signal + i * complex_sz, cx_buf_kernel,
                           complex_sz, cx_buf_signal + i * complex_sz);
    }
  }

  void get_output(double *arr, size_t size, const size_t idx) const {
    const double fct = 1. / real_sz;
    size = std::min<size_t>(real_sz, size);

    const size_t pos = idx * real_sz;
    for (int i = 0; i < size; ++i) {
      arr[i] += real_buf_signal[pos + i] * fct;
    }
  }

  // FFTW buffer sizes
  const int real_sz;
  const int complex_sz;
  const int n_arrays;

  // FFTW buffers corresponding to the above plans
  double *real_buf_signal;
  fftw_complex *cx_buf_signal;

  double *real_buf_kernel;
  fftw_complex *cx_buf_kernel;

  // FFTW plans
  fftw_plan plan_forward_kernel;
  fftw_plan plan_forward_signal;
  fftw_plan plan_backward_signal;

  // void debug_print() {
  // #define PRINT(NAME) std::cout << #NAME << " (" << NAME << ") "
  // PRINT(real_sz);
  // PRINT(complex_sz);
  // PRINT(howmany);
  // std::cout << "\n";

  // std::cout << "real_buf_kernel";
  // print(real_buf_kernel, real_sz);

  // std::cout << "cx_buf_kernel";
  // print(cx_buf_kernel, complex_sz);

  // std::cout << "real_buf_signal";
  // print(real_buf_signal, real_sz * howmany);

  // std::cout << "cx_buf_signal";
  // print(cx_buf_signal, complex_sz * howmany);
  //}
};

auto fftconv_plans_advanced_cache(const size_t padded_length, const int howmany)
    -> fftconv_plans_advanced * {
  static thread_local std::unordered_map<
      size_t, std::unique_ptr<fftconv_plans_advanced>>
      _cache;
  const size_t _hash = (padded_length << 4) ^ howmany;

  auto &plan = _cache[_hash];
  if (plan == nullptr || plan->real_sz != padded_length) {
    plan = std::make_unique<fftconv_plans_advanced>(padded_length, howmany);
  }
  return plan.get();
}

} // namespace

namespace fftconv {

void use_fftw_mutex(std::mutex *fftw_mutex) { fftconv_fftw_mutex = fftw_mutex; }

void convolve_fftw(const double *arr1, const size_t size1, const double *arr2,
                   const size_t size2, double *res, const size_t res_sz) {
  // length of the real arrays, including the final convolution output
  const size_t padded_length = size1 + size2 - 1;

  // Get cached plans
  fftconv_plans *plan = fftconv_plans_cache(padded_length);
  plan->execute_conv(arr1, size1, arr2, size2);

  // copy normalized to result
  const auto *real_buf = plan->get_real_buf();
  const size_t end = std::min<size_t>(padded_length, res_sz);
  for (int i = 0; i < end; i++) {
    res[i] = real_buf[i];
  }
}

void convolve_fftw_advanced(const double *arr1, const size_t size1,
                            const double *arr2, const size_t size2, double *res,
                            const size_t res_sz) {
  const size_t padded_length = size1 + size2 - 1;
  const auto *plans = fftconv_plans_advanced_cache(padded_length, 1);

  plans->set_kernel(arr2, size2);
  plans->forward_kernel();

  plans->set_signal(arr1, size1, 0);
  plans->forward_signal();

  plans->complex_multiply();
  plans->backward();
  plans->get_output(res, res_sz, 0);
}

// reference implementation of fftconv with no optimizations
void convolve_fftw_ref(const double *arr1, const int size1, const double *arr2,
                       const int size2, double *res, const int res_size) {
  // length of the real arrays, including the final convolution output
  const auto padded_length = size1 + size2 - 1;
  // length of the complex arrays
  const auto complex_length = padded_length / 2 + 1;

  // Allocate fftw buffers for a
  double *a_buf = fftw_alloc_real(padded_length);
  fftw_complex *A_buf = fftw_alloc_complex(complex_length);

  // Compute forward fft plan
  fftw_plan plan_forward =
      fftw_plan_dft_r2c_1d(padded_length, a_buf, A_buf, FFTW_ESTIMATE);

  // Copy a to buffer
  copy_to_padded_buffer(arr1, size1, a_buf, padded_length);

  // Compute Fourier transform of vector a
  fftw_execute_dft_r2c(plan_forward, a_buf, A_buf);

  // Allocate fftw buffers for b
  double *b_buf = fftw_alloc_real(padded_length);
  fftw_complex *B_buf = fftw_alloc_complex(complex_length);

  // Copy b to buffer
  copy_to_padded_buffer(arr2, size2, b_buf, padded_length);

  // Compute Fourier transform of vector b
  fftw_execute_dft_r2c(plan_forward, b_buf, B_buf);

  // Compute backward fft plan
  fftw_complex *input_buffer = fftw_alloc_complex(complex_length);
  double *output_buffer = fftw_alloc_real(padded_length);
  fftw_plan plan_backward = fftw_plan_dft_c2r_1d(padded_length, input_buffer,
                                                 output_buffer, FFTW_ESTIMATE);

  // Perform element-wise product of FFT(a) and FFT(b)
  // then compute inverse fourier transform.
  elementwise_multiply(A_buf, B_buf, complex_length,
                       input_buffer); // A_buf becomes input to inverse conv

  fftw_execute_dft_c2r(plan_backward, input_buffer, output_buffer);

  // Normalize output
  for (int i = 0; i < std::min<size_t>(padded_length, res_size); i++) {
    res[i] = output_buffer[i] / padded_length;
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

// Overlap-Add convolution of x and h
//
// x is a long signal
// h is a kernel, x_size >> h_size
// y is the results buffer. y_size >= x_size + b_size - 1
//
// 1. Split x into blocks of step_size.
// 2. convolve with kernel b using fft of length N.
// 3. add blocks together
void oaconvolve_fftw(const double *arr, const size_t arr_size,
                     const double *kernel, const size_t kernel_size,
                     double *res, const size_t res_size) {

  // more optimal size for each fft
  const auto fft_size = get_optimal_fft_size(kernel_size);
  const auto step_size = fft_size - (kernel_size - 1);

  // forward fft of h
  auto *plan = fftconv_plans_cache(fft_size);
  plan->set_real_buf(kernel, kernel_size);
  plan->forward_b();

  // create forward/backward ffts for x
  const auto *real_buf = plan->get_real_buf();
  const double fct = 1. / fft_size;
  for (size_t pos = 0; pos < arr_size; pos += step_size) {
    size_t len = std::min<size_t>(arr_size - pos, step_size); // bound check
    plan->set_real_buf(arr + pos, len);
    plan->forward_a();
    plan->complex_multiply_to_a();
    plan->backward();
    // plan->normalize(); // normalize later in the copy loop

    // normalize output and add to result
    len = std::min<size_t>(res_size - pos, fft_size);
    for (size_t i = 0; i < len; ++i) {
      res[pos + i] += real_buf[i] * fct;
    }
  }
}

void oaconvolve_fftw_advanced(const double *arr, const size_t arr_size,
                              const double *kernel, const size_t kernel_size,
                              double *res, const size_t res_size) {
  // more optimal size for each fft
  const auto fft_size = get_optimal_fft_size(kernel_size);
  const auto step_size = fft_size - (kernel_size - 1);
  const auto batch_sz = arr_size / step_size;

  auto *plans = fftconv_plans_advanced_cache(
      fft_size, batch_sz + 1); // last batch zero pad
  plans->set_kernel(kernel, kernel_size);
  plans->forward_kernel();

  // Copy data to plan
  for (size_t pos = 0, idx = 0; pos < arr_size; pos += step_size, idx++) {
    size_t len = std::min<size_t>(arr_size - pos, step_size); // bound check
    plans->set_signal(arr + pos, len, idx);
  }

  plans->forward_signal();

  plans->complex_multiply();

  plans->backward();

  for (size_t pos = 0, idx = 0; pos < res_size; pos += step_size, idx++) {
    size_t len = std::min<size_t>(res_size - pos, fft_size); // bound check
    plans->get_output(res + pos, len, idx);
  }
}

} // namespace fftconv
