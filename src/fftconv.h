// Author: Tiger Nie
// 2022

#ifndef __FFTCONV_H__
#define __FFTCONV_H__

#include <cassert>
#include <complex>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

#include <fftw3.h>

// vector wrapper for fftconv routines
#define VECTOR_WRAPPER(CONV_FUNC)                                              \
  inline std::vector<double> CONV_FUNC(const std::vector<double> &x,           \
                                       const std::vector<double> &b) {         \
    std::vector<double> y(b.size() + x.size() - 1);                            \
    CONV_FUNC(x.data(), x.size(), b.data(), b.size(), y.data(), y.size());     \
    return y;                                                                  \
  }

// armadillo wrapper for fftconv routines
#ifdef ARMA_INCLUDES
  #define ARMA_WRAPPER(CONV_FUNC) \
  inline arma::vec CONV_FUNC(const arma::vec &x, const arma::vec &b) { \
    arma::vec y(b.size() + x.size() - 1);                                      \
    CONV_FUNC(x.memptr(), x.size(), b.memptr(), b.size(), y.memptr(),          \
              y.size());                                                       \
    return y;                                                                  \
  }
#endif

namespace fftconv {

// 1D convolution using the FFT
// Optimizations:
//    * Cache fftw_plan
//    * Reuse buffers (no malloc on second call to the same convolution size)
// https://en.wikipedia.org/w/index.php?title=Convolution#Fast_convolution_algorithms
void fftconv(const double *a, const size_t a_size, const double *b,
             const size_t b_size, double *y, const size_t y_sz);

// Reference implementation of fft convolution with minimal optimizations
void fftconv_ref(const double *a, const size_t a_size, const double *b,
                 const size_t b_size, double *y, const size_t y_sz);

// FFTconv with overlap-add method
void fftconv_oa(const double *x, const size_t x_size, const double *h,
                const size_t h_sz, double *y, const size_t y_sz);

// std::vector interface to the fftconv routines
VECTOR_WRAPPER(fftconv)
VECTOR_WRAPPER(fftconv_ref)
VECTOR_WRAPPER(fftconv_oa)

// arma::vec interface
#ifdef ARMA_WRAPPER
 ARMA_WRAPPER(fftconv)
 ARMA_WRAPPER(fftconv_ref)
 ARMA_WRAPPER(fftconv_oa)
#endif

// In memory cache with key type K and value type V
// V's constructor must take K as input
template <class K, class V> class Cache {
public:
  // Get a cached object if available.
  // Otherwise, a new one will be constructed.
  V *get(K key) {
    auto &val = _cache[key];
    if (val)
      return val.get();

    val = std::make_unique<V>(key);
    return val.get();
  }

  V *operator()(K key) { return get(key); }

private:
  // hash map cache to store fftw plans and buffers.
  std::unordered_map<K, std::unique_ptr<V>> _cache;
};

} // namespace fftconv

#ifdef ARMA_WRAPPER
#undef ARMA_WRAPPER
#endif

#undef VECTOR_WRAPPER

#endif // __FFTCONV_H__
