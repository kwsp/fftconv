// Author: Tiger Nie
// 2022
// https://github.com/kwsp/fftconv

#ifndef __FFTCONV_H__
#define __FFTCONV_H__

#include <cassert>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <fftw3.h>

// std::vector wrapper for fftconv routines
#define VECTOR_WRAPPER(CONV_FUNC)                                              \
  inline std::vector<double> CONV_FUNC(const std::vector<double> &x,           \
                                       const std::vector<double> &b) {         \
    std::vector<double> y(b.size() + x.size() - 1);                            \
    CONV_FUNC(x.data(), x.size(), b.data(), b.size(), y.data(), y.size());     \
    return y;                                                                  \
  }

// arma::vec wrapper for fftconv routines
#ifdef ARMA_INCLUDES
#define ARMA_WRAPPER(CONV_FUNC)                                                \
  inline arma::vec CONV_FUNC(const arma::vec &x, const arma::vec &b) {         \
    arma::vec y(b.size() + x.size() - 1);                                      \
    CONV_FUNC(x.memptr(), x.size(), b.memptr(), b.size(), y.memptr(),          \
              y.size());                                                       \
    return y;                                                                  \
  }
#endif

namespace fftconv {

// Since FFTW planners are not thread-safe, you can pass a pointer to a
// std::mutex to fftconv and all calls to the planner with be guarded by the
// mutex.
void use_fftw_mutex(std::mutex *fftw_mutex);

// 1D convolution using the FFT
// Optimizations:
//    * Cache fftw_plan
//    * Reuse buffers (no malloc on second call to the same convolution size)
// https://en.wikipedia.org/w/index.php?title=Convolution#Fast_convolution_algorithms
void convolve_fftw(const double *a, const size_t a_sz, const double *b,
             const size_t b_sz, double *result, const size_t res_sz);

// Reference implementation of fft convolution with minimal optimizations
void convolve_fftw_ref(const double *a, const size_t a_sz, const double *b,
                 const size_t b_sz, double *result, const size_t result_sz);

// 1D Overlap-Add convolution of x and h
//
// x is a long signal
// h is a kernel, x_size >> h_size
// y is the results buffer. y_size >= x_size + b_size - 1
//
// 1. Split x into blocks of step_size.
// 2. convolve with kernel b using fft of length N.
// 3. add blocks together
void oaconvolve_fftw(const double *x, const size_t x_sz, const double *h,
                const size_t h_sz, double *y, const size_t y_sz);

// std::vector interface to the fftconv routines
VECTOR_WRAPPER(convolve_fftw)
VECTOR_WRAPPER(convolve_fftw_ref)
VECTOR_WRAPPER(oaconvolve_fftw)

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
    if (val == nullptr)
      val = std::make_unique<V>(key);
    return val.get();
  }

  V *operator()(K key) { return get(key); }

private:
  std::unordered_map<K, std::unique_ptr<V>> _cache;
};

} // namespace fftconv

#ifdef ARMA_WRAPPER
#undef ARMA_WRAPPER
#endif

#undef VECTOR_WRAPPER

#endif // __FFTCONV_H__
