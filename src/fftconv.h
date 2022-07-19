#ifndef __FFTCONV_H__
#define __FFTCONV_H__
#include <cassert>
#include <complex>
#include <cstring>
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

void convolve1d_ref(const double *a, const size_t a_size, const double *b,
                    const size_t b_size, double *result);

vector<double> convolve1d_ref(const vector<double> &a, const vector<double> &b);

} // namespace fftconv
#endif // __FFTCONV_H__
