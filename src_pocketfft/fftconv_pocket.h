// Author: Tiger Nie
// 2022
// https://github.com/kwsp/fftconv
#pragma once

#include "pocketfft_hdronly.h"
//#include <pocketfft.h>

#include <cassert>
#include <complex>
#include <iostream> // DEBUG
#include <vector>

// std::vector wrapper for fftconv routines
#define VECTOR_WRAPPER(CONV_FUNC)                                              \
  inline std::vector<double> CONV_FUNC(const std::vector<double> &x,           \
                                       const std::vector<double> &b) {         \
    std::vector<double> y(b.size() + x.size() - 1);                            \
    CONV_FUNC(x.data(), x.size(), b.data(), b.size(), y.data(), y.size());     \
    return y;                                                                  \
  }

// Copy data from src to dst and padded the extra with zero
// dst_size must be greater than src_size
template <class T>
static inline void _copy_to_padded_buffer(const T *src, const size_t src_size,
                                          T *dst, const size_t dst_size) {
  assert(src_size <= dst_size);
  std::copy(src, src + src_size, dst);
  std::fill(dst + src_size, dst + dst_size, 0);
}

template <class T>
static inline void elementwise_multiply(const T *a, const T *b,
                                        const size_t length, T *result) {
  for (size_t i = 0; i < length; ++i)
    result[i] = a[i] * b[i];
}

template <class T> inline void print(const T *a, const size_t sz) {
  for (size_t i = 0; i < sz; ++i)
    std::cout << a[i] << ", ";
  std::cout << "\n";
}
template <class T>
inline void print(const std::complex<T> *a, const size_t sz) {
  for (size_t i = 0; i < sz; ++i)
    std::cout << "(" << a[i].real() << ", " << a[i].imag() << "), ";
  std::cout << "\n";
}
template <class T> inline void print(const std::vector<T> &a) {
  print(a.data(), a.size());
}

namespace fftconv {

inline void fftconv_ref_pocket(const double *a, const size_t a_sz,
                               const double *b, const size_t b_sz, double *res,
                               const size_t res_sz) {

  // length of the real arrays, including the final convolution output
  const size_t padded_length = a_sz + b_sz - 1;
  // length of the complex arrays
   const size_t complex_length = (padded_length >> 1) + 1;

  // Allocate fftw buffers for a and b
  auto a_buf = new double[padded_length]();
  auto b_buf = new double[padded_length]();
  auto A_buf = new std::complex<double>[complex_length]();

  // Compute fft plan
  // auto plan = make_rfft_plan(padded_length);

  // Copy a to buffer
  _copy_to_padded_buffer(a, a_sz, a_buf, padded_length);
  std::cout << "a    : ";
  print(a, a_sz);
  std::cout << "a_buf: ";
  print(a_buf, padded_length);

  // Compute Fourier transform of vector a
  // rfft_forward(plan, a_buf, 1.);
  //pocketfft::r2c({padded_length}, {1}, {1}, 0, true, a_buf, A_buf, 1.);
  pocketfft::r2c({a_sz}, {1}, {1}, 0, true, a, A_buf, 1.);
  std::cout << "A_buf: ";
  print(A_buf, padded_length);
  // fftw_execute_dft_r2c(plan_forward, a_buf, A_buf);

  // Copy b to buffer
  _copy_to_padded_buffer(b, b_sz, b_buf, padded_length);

  // Compute Fourier transform of vector b
  //rfft_forward(plan, b_buf, 1.);

  // Perform element-wise product of FFT(a) and FFT(b)
  // then compute inverse fourier transform.
  // elementwise_multiply(A_buf, B_buf, complex_length,
  // input_buffer); // A_buf becomes input to inverse conv

  // fftw_execute_dft_c2r(plan_backward, input_buffer, output_buffer);

  // auto end = std::max(conv_size, res_sz);
  // for (int i=0; i<end; ++i)
  // res[i] = real_buf[i];

  delete[] a_buf;
  delete[] b_buf;
  //destroy_rfft_plan(plan);
}

VECTOR_WRAPPER(fftconv_ref_pocket);

} // namespace fftconv
