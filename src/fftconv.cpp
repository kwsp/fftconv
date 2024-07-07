#include "fftconv.h"

namespace fftconv {

// fftconv_plans manages the memory of the forward and backward fft plans
// Assume the same FFTW plan will be used many times (use FFTW_MEASURE to
// compute optimal plan)
struct fftconv_plans {
  fftw_plan forward;
  fftw_plan backward;

  // NOT THREAD-SAFE RIGHT NOW
  double *real_buf;
  fftw_complex *complex_buf_a;
  fftw_complex *complex_buf_b;

  fftconv_plans(size_t padded_length);

  ~fftconv_plans() {
    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
    fftw_free(real_buf);
    fftw_free(complex_buf_a);
    fftw_free(complex_buf_b);
  }
};

fftconv_plans::fftconv_plans(size_t padded_length) {

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

thread_local std::unordered_map<size_t, std::shared_ptr<fftconv_plans>> _cache;
std::shared_mutex _mutex;

std::shared_ptr<fftconv_plans> _get_plans(size_t size) {
  std::shared_ptr<fftconv_plans> plans;
  {
    // _cache is thread_local so we don't need a read lock to access
    //std::shared_lock read_lock(_mutex);
    auto it = _cache.find(size);
    if (it != _cache.end())
      plans = it->second;
  }
  if (plans == nullptr) {
    // We need an exclusive lock to access the cache because creation of fftw_plans
    // is not thread-safe by default
    std::unique_lock write_lock(_mutex);
    plans = std::make_shared<fftconv_plans>(size);
    _cache[size] = plans;
  }
  return plans;
}

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
} // namespace fftconv
