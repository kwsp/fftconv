// Author: Taylor Nie
// 2022
// https://github.com/kwsp/fftconv

#ifndef __FFTCONV_H__
#define __FFTCONV_H__

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <fftw3.h>

// std::vector wrapper for fftconv routines
#define VECTOR_WRAPPER(CONV_FUNC)                                              \
  inline std::vector<double> CONV_FUNC(const std::vector<double> &vec1,        \
                                       const std::vector<double> &vec2) {      \
    std::vector<double> res(vec1.size() + vec2.size() - 1);                    \
    CONV_FUNC(vec1.data(), vec1.size(), vec2.data(), vec2.size(), res.data(),  \
              res.size());                                                     \
    return res;                                                                \
  }

// arma::vec wrapper for fftconv routines
#ifdef ARMA_INCLUDES
#define ARMA_WRAPPER(CONV_FUNC)                                                \
  inline arma::vec CONV_FUNC(const arma::vec &vec1, const arma::vec &vec2) {   \
    arma::vec res(vec2.size() + vec1.size() - 1);                              \
    CONV_FUNC(vec1.memptr(), vec1.size(), vec2.memptr(), vec2.size(),          \
              res.memptr(), res.size());                                       \
    return res;                                                                \
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
void convolve_fftw(const double *arr1, size_t size1, const double *arr2,
                   size_t size2, double *res, size_t res_sz);

// Advanced interface to batch ffts
void convolve_fftw_advanced(const double *arr1, size_t size1,
                            const double *arr2, size_t size2, double *res,
                            size_t res_sz);

// Reference implementation of fft convolution with minimal optimizations
void convolve_fftw_ref(const double *arr1, size_t size1, const double *arr2,
                       size_t size2, double *res, size_t res_size);

// 1D Overlap-Add convolution of x and h
//
// arr is a long signal
// kernel is a kernel, arr_size >> kernel_size
// res is the results buffer. res_size >= arr_size + kernel_size - 1
//
// 1. Split arr into blocks of step_size.
// 2. convolve with kernel using fft of length N.
// 3. add blocks together
void oaconvolve_fftw(const double *arr, size_t arr_size, const double *kernel,
                     size_t kernel_size, double *res, size_t res_size);

void oaconvolve_fftw_advanced(const double *arr, size_t arr_size,
                              const double *kernel, size_t kernel_size,
                              double *res, size_t res_size);

// std::vector interface to the fftconv routines
VECTOR_WRAPPER(convolve_fftw)
VECTOR_WRAPPER(convolve_fftw_advanced)
VECTOR_WRAPPER(convolve_fftw_ref)
VECTOR_WRAPPER(oaconvolve_fftw)
VECTOR_WRAPPER(oaconvolve_fftw_advanced)

// arma::vec interface
#ifdef ARMA_WRAPPER
ARMA_WRAPPER(convolve_fftw)
ARMA_WRAPPER(convolve_fftw_advanced)
ARMA_WRAPPER(convolve_fftw_ref)
ARMA_WRAPPER(oaconvolve_fftw)
ARMA_WRAPPER(oaconvolve_fftw_advanced)
#endif

// In memory cache with key type K and value type V
template <class K, class V> auto _get_cached(K key) -> V * {
  static thread_local std::unordered_map<K, std::unique_ptr<V>> _cache;

  auto &val = _cache[key];
  if (val == nullptr) {
    val = std::make_unique<V>(key);
  }

  return val.get();
}

} // namespace fftconv

#ifdef ARMA_WRAPPER
#undef ARMA_WRAPPER
#endif

#undef VECTOR_WRAPPER

#endif // __FFTCONV_H__
