#ifndef __FFTCONV_H__
#define __FFTCONV_H__
#include <cassert>
#include <complex>
#include <map>
#include <memory>
#include <shared_mutex>
#include <vector>

#include <fftw3.h>

namespace fftconv {
using std::vector;

// fftconv_plans manages the memory of the forward and backward fft plans
// Assume the same FFTW plan will be used many times (use FFTW_MEASURE to
// compute optimal plan)
struct fftconv_plans {
  fftw_plan forward;
  fftw_plan backward;

  // NOT THREAD-SAFE RIGHT NOW
  double* real_buf;
  fftw_complex* complex_buf_a;
  fftw_complex* complex_buf_b;

  fftconv_plans(size_t padded_length) {

    // length of the complex arrays
    size_t complex_length = padded_length / 2 + 1;

    // Allocate temporary buffers for the purposes of creating the plans...
    real_buf = fftw_alloc_real(padded_length);
    complex_buf_a = fftw_alloc_complex(complex_length);
    complex_buf_b = fftw_alloc_complex(complex_length);

    // Compute the plans
    this->forward = fftw_plan_dft_r2c_1d(padded_length, real_buf, complex_buf_a,
                                         FFTW_ESTIMATE);
    this->backward = fftw_plan_dft_c2r_1d(padded_length, complex_buf_a, real_buf,
                                          FFTW_ESTIMATE);

  }
  ~fftconv_plans() {
    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
    fftw_free(real_buf);
    fftw_free(complex_buf_a);
    fftw_free(complex_buf_b);
  }
};

// Global cache of FFTW plans
// Thread-safe
class FFTW_PLAN_STORE {
public:
  //static FFTW_PLAN_STORE &Instance() {
    //return instance;
  //}

  static std::shared_ptr<fftconv_plans> get(size_t size) {
    static FFTW_PLAN_STORE instance;
    //auto& instance = FFTW_PLAN_STORE::Instance();
    if (auto plan = instance._get(size))
      return plan;
    return instance._get_set(size);
  }

private:
  FFTW_PLAN_STORE() = default;
  ~FFTW_PLAN_STORE() = default;


  std::shared_ptr<fftconv_plans> _get(size_t size) {
    std::shared_lock read_lock(mutex_);
    auto it = cache.find(size);
    if (it != cache.end())
      return it->second;
    return nullptr;
  }

  std::shared_ptr<fftconv_plans> _get_set(size_t size) {
    std::unique_lock write_lock(mutex_);
    auto plan = std::make_shared<fftconv_plans>(size);
    cache[size] = plan;
    return plan;
  }

  // cache and mutex
  std::map<size_t, std::shared_ptr<fftconv_plans>> cache;
  mutable std::shared_mutex mutex_;
};

template <class T>
vector<T> vector_elementwise_multiply(const vector<T> a, const vector<T> b) {
  assert(a.size() == b.size());
  vector<T> result(a.size());
  for (int i = 0; i < result.size(); ++i) {
    result[i] = a[i] * b[i];
  }
  return result;
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
  memcpy(dst, src, sizeof(T) * src_size);
  memset(&dst[src_size], 0, sizeof(T) * (dst_size - src_size));
}

// http://en.wikipedia.org/w/index.php?title=Convolution&oldid=630841165#Fast_convolution_algorithms
// size(a) >= size(b). size(a) must be greater or equal to size(b);
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

#ifdef DEBUG
  printf("FFT(a) ");
  print_complex_array(A_buf, complex_length);
  printf("FFT(b) ");
  print_complex_array(B_buf, complex_length);
#endif

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

#ifdef DEBUG
  printf("element multiply ");
  print_complex_array(input_buffer, complex_length);
#endif

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

vector<double> convolve1d_ref(const vector<double> &a,
                              const vector<double> &b) {
  int padded_length = a.size() + b.size() - 1;
  vector<double> result(padded_length);
  convolve1d_ref(a.data(), a.size(), b.data(), b.size(), result.data());
  return result;
}

// size(a) >= size(b). size(a) must be greater or equal to size(b);
// Faster version of convolve_ref.
// Optimizations:
//    * Cache fftw_plan
//    * Reuse buffers (3 fftw_mallocs vs 6 in convolve_ref)
// https://en.wikipedia.org/w/index.php?title=Convolution#Fast_convolution_algorithms
void convolve1d(const double *a, const size_t a_size, const double *b,
                const size_t b_size, double *result) {
  // length of the real arrays, including the final convolution output
  size_t padded_length = a_size + b_size - 1;
  // length of the complex arrays
  size_t complex_length = padded_length / 2 + 1;

  // Get cached plans
  auto plans = FFTW_PLAN_STORE::get(padded_length);

  // Allocate fftw buffers for a
  //double *real_buf = fftw_alloc_real(padded_length);
  //fftw_complex *complex_buf_a = fftw_alloc_complex(complex_length);
  double* const real_buf = plans->real_buf;
  fftw_complex* const complex_buf_a = plans->complex_buf_a;
  fftw_complex* const complex_buf_b = plans->complex_buf_b;

  // Copy a to buffer
  _copy_to_padded_buffer(a, a_size, real_buf, padded_length);

  // Compute Fourier transform of vector a
  fftw_execute_dft_r2c(plans->forward, real_buf, complex_buf_a);

  // Allocate fftw buffers for b
  //fftw_complex *complex_buf_b = fftw_alloc_complex(complex_length);

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

  //fftw_free(real_buf);
  //fftw_free(complex_buf_a);
  //fftw_free(complex_buf_b);
}

vector<double> convolve1d(const vector<double> &a, const vector<double> &b) {
  int padded_length = a.size() + b.size() - 1;
  vector<double> result(padded_length);
  convolve1d(a.data(), a.size(), b.data(), b.size(), result.data());
  return result;
}

} // namespace fftconv
#endif  // __FFTCONV_H__
