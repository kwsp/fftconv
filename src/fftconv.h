// Author: Tiger Nie
// 2022

#ifndef __FFTCONV_H__
#define __FFTCONV_H__

#include <cassert>
#include <complex>
#include <cstring>
#include <vector>

#include <fftw3.h>

namespace fftconv {

// Fast 1D convolution using the FFT
// Optimizations:
//    * Cache fftw_plan
//    * Reuse buffers (no malloc on second call to the same convolution size)
// https://en.wikipedia.org/w/index.php?title=Convolution#Fast_convolution_algorithms
void convolve1d(const double *a, const size_t a_size, const double *b,
                const size_t b_size, double *result);

// Vector interface to the fft convolution implementation
inline std::vector<double> convolve1d(const std::vector<double> &a,
                                 const std::vector<double> &b) {
  int padded_length = a.size() + b.size() - 1;
  std::vector<double> result(padded_length);
  convolve1d(a.data(), a.size(), b.data(), b.size(), result.data());
  return result;
}

// Reference implementation of fft convolution with minimal optimizations
void convolve1d_ref(const double *a, const size_t a_size, const double *b,
                    const size_t b_size, double *result);

// Vector interface to the above reference fft convolution implementation
std::vector<double> convolve1d_ref(const std::vector<double> &a, const std::vector<double> &b);

} // namespace fftconv
#endif // __FFTCONV_H__
