// Author: Tiger Nie
// 2022

#include "fftconv.h"
#include <memory>
#include <shared_mutex>
#include <unordered_map>

using std::vector;

// fftconv_plans manages the memory of the forward and backward fft plans
// and the fftw buffers
struct fftconv_plans {

  // FFTW plans
  fftw_plan forward;
  fftw_plan backward;

  // FFTW buffers corresponding to the above plans
  double *real_buf;
  fftw_complex *complex_buf_a;
  fftw_complex *complex_buf_b;

  // Constructors
  fftconv_plans(size_t padded_length);
  fftconv_plans() = delete; // default constructor

  fftconv_plans(fftconv_plans &&) = delete;      // move constructor
  fftconv_plans(const fftconv_plans &) = delete; // copy constructor

  fftconv_plans &operator=(const fftconv_plans) = delete; // copy assignment
  fftconv_plans &operator=(fftconv_plans &&) = delete;    // move assignment

  // Destructor
  ~fftconv_plans() {

    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);

    fftw_free(real_buf);
    fftw_free(complex_buf_a);
    fftw_free(complex_buf_b);
  }
};

// Compute the fftw plans and allocate buffers
fftconv_plans::fftconv_plans(size_t padded_length) {

  // length of the complex arrays
  size_t complex_length = padded_length / 2 + 1;

  // Allocate buffers for the purposes of creating the plans...
  real_buf = fftw_alloc_real(padded_length);
  complex_buf_a = fftw_alloc_complex(complex_length);
  complex_buf_b = fftw_alloc_complex(complex_length);

  // Compute the plans
  forward = fftw_plan_dft_r2c_1d(padded_length, real_buf, complex_buf_a,
                                 FFTW_ESTIMATE);
  backward = fftw_plan_dft_c2r_1d(padded_length, complex_buf_a, real_buf,
                                  FFTW_ESTIMATE);
}

// Thread-local hash map cache to store fftw plans and buffers.
// The thread-local is mainly to make the buffers reusable
// This does mean we need to compute the same plan in all threads
thread_local std::unordered_map<size_t, std::shared_ptr<fftconv_plans>> _cache;
// Mutex - fftw plan computation is not thread-safe by default
std::shared_mutex _mutex;

// Get fftconv_plans object.
// The cached object will be returned if available.
// Otherwise, a new one will be constructed.
std::shared_ptr<fftconv_plans> _get_plans(size_t size) {
  std::shared_ptr<fftconv_plans> plans;
  {
    // _cache is thread_local so we don't need a read lock to access
    // std::shared_lock read_lock(_mutex);
    auto it = _cache.find(size);
    if (it != _cache.end())
      plans = it->second;
  }
  if (plans == nullptr) {
    // We need an exclusive lock to access the cache because creation of
    // fftw_plans is not thread-safe by default
    std::unique_lock write_lock(_mutex);
    plans = std::make_shared<fftconv_plans>(size);
    _cache[size] = plans;
  }
  return plans;
}

void vector_elementwise_multiply(const fftw_complex *a, const fftw_complex *b,
                                 const size_t length, fftw_complex *result) {
  for (auto i = 0; i < length; ++i) {
    std::complex<double> _a(a[i][0], a[i][1]);
    std::complex<double> _b(b[i][0], b[i][1]);
    _a *= _b;
    result[i][0] = _a.real();
    result[i][1] = _a.imag();
  }
}

// Copy data from src to dst and padded the extra with zero
// dst_size must be greater than src_size
template <class T>
void _copy_to_padded_buffer(const T *src, const size_t src_size, T *dst,
                            const size_t dst_size) {
  assert(src_size <= dst_size);
  std::copy(src, src + src_size, dst);
  std::fill(dst + src_size, dst + dst_size, 0);
}

namespace fftconv {

void convolve1d(const double *a, const size_t a_size, const double *b,
                const size_t b_size, double *result) {
  // length of the real arrays, including the final convolution output
  size_t padded_length = a_size + b_size - 1;
  // length of the complex arrays
  size_t complex_length = padded_length / 2 + 1;

  // Get cached plans
  std::shared_ptr<fftconv_plans> plans = _get_plans(padded_length);

  // Get fftw buffers
  double *const real_buf = plans->real_buf;
  fftw_complex *const complex_buf_a = plans->complex_buf_a;
  fftw_complex *const complex_buf_b = plans->complex_buf_b;

  // Copy a to buffer
  _copy_to_padded_buffer(a, a_size, real_buf, padded_length);

  // Compute Fourier transform of vector a
  fftw_execute_dft_r2c(plans->forward, real_buf, complex_buf_a);

  // Copy b to buffer. Reuse real buffer
  _copy_to_padded_buffer(b, b_size, real_buf, padded_length);

  // Compute Fourier transform of vector b
  fftw_execute_dft_r2c(plans->forward, real_buf, complex_buf_b);

  // Reuse buffers for ifft
  //
  // Perform element-wise product of FFT(a) and FFT(b)
  // then compute inverse fourier transform.
  // Multiply INPLACE
  vector_elementwise_multiply(
      complex_buf_a, complex_buf_b, complex_length,
      complex_buf_a); // A_buf becomes input to inverse conv

  // Compute ifft
  fftw_execute_dft_c2r(plans->backward, complex_buf_a, real_buf);

  // Normalize output and copy to result
  for (int i = 0; i < padded_length; i++) {
    result[i] = real_buf[i] / padded_length;
  }
}

void convolve1d_ref(const double *a, const size_t a_size, const double *b,
                    const size_t b_size, double *result) {
  // length of the real arrays, including the final convolution output
  size_t padded_length = a_size + b_size - 1;
  // length of the complex arrays
  size_t complex_length = padded_length / 2 + 1;

  // Allocate fftw buffers for a
  double *a_buf = fftw_alloc_real(padded_length);
  fftw_complex *A_buf = fftw_alloc_complex(complex_length);

  // Compute forward fft plan
  fftw_plan plan_forward =
      fftw_plan_dft_r2c_1d(padded_length, a_buf, A_buf, FFTW_ESTIMATE);

  // Copy a to buffer
  _copy_to_padded_buffer(a, a_size, a_buf, padded_length);

  // Compute Fourier transform of vector a
  fftw_execute_dft_r2c(plan_forward, a_buf, A_buf);

  // Allocate fftw buffers for b
  double *b_buf = fftw_alloc_real(padded_length);
  fftw_complex *B_buf = fftw_alloc_complex(complex_length);

  // Copy b to buffer
  _copy_to_padded_buffer(b, b_size, b_buf, padded_length);

  // Compute Fourier transform of vector b
  fftw_execute_dft_r2c(plan_forward, b_buf, B_buf);

  // Compute backward fft plan
  fftw_complex *input_buffer = fftw_alloc_complex(complex_length);
  double *output_buffer = fftw_alloc_real(padded_length);
  fftw_plan plan_backward = fftw_plan_dft_c2r_1d(padded_length, input_buffer,
                                                 output_buffer, FFTW_ESTIMATE);

  // Perform element-wise product of FFT(a) and FFT(b)
  // then compute inverse fourier transform.
  vector_elementwise_multiply(
      A_buf, B_buf, complex_length,
      input_buffer); // A_buf becomes input to inverse conv

  fftw_execute_dft_c2r(plan_backward, input_buffer, output_buffer);

  // Normalize output
  for (int i = 0; i < padded_length; i++) {
    result[i] = output_buffer[i] / padded_length;
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

std::vector<double> convolve1d_ref(const vector<double> &a,
                                   const vector<double> &b) {
  int padded_length = a.size() + b.size() - 1;
  vector<double> result(padded_length);
  convolve1d_ref(a.data(), a.size(), b.data(), b.size(), result.data());
  return result;
}

} // namespace fftconv
