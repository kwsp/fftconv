#ifndef __FFTCONV_H__
#define __FFTCONV_H__
#include <cassert>
#include <complex>
#include <cstring>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include <fftw3.h>

namespace fftconv {
using std::vector;

inline void vector_elementwise_multiply(const fftw_complex *a,
                                        const fftw_complex *b,
                                        const size_t length,
                                        fftw_complex *result) {
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

// http://en.wikipedia.org/w/index.php?title=Convolution&oldid=630841165#Fast_convolution_algorithms
// size(a) >= size(b). size(a) must be greater or equal to size(b);
inline void convolve1d_ref(const double *a, const size_t a_size,
                           const double *b, const size_t b_size,
                           double *result) {
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

inline vector<double> convolve1d_ref(const vector<double> &a,
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
                const size_t b_size, double *result);

inline vector<double> convolve1d(const vector<double> &a,
                                 const vector<double> &b) {
  int padded_length = a.size() + b.size() - 1;
  vector<double> result(padded_length);
  convolve1d(a.data(), a.size(), b.data(), b.size(), result.data());
  return result;
}

} // namespace fftconv
#endif // __FFTCONV_H__
