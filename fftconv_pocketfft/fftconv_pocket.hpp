// Author: Tiger Nie
// 2022
// https://github.com/kwsp/fftconv
#pragma once

#define POCKETFFT_NO_MULTITHREADING

#include <pocketfft_hdronly.h>
extern "C" {
#include <pocketfft.h>
}

#include <cassert>
#include <complex>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <vector>

#include "debug_utils.hpp"

// Lookup table of {max_filter_size, optimal_fft_size}
static constexpr std::array<std::array<size_t, 2>, 9> _optimal_fft_size{
    {{7, 16},
     {12, 32},
     {21, 64},
     {36, 128},
     {65, 256},
     {120, 512},
     {221, 1024},
     {411, 2048},
     {769, 4096}},
};

static size_t get_optimal_fft_size(const size_t filter_size) {
  for (const auto &pair : _optimal_fft_size)
    if (filter_size < pair[0])
      return pair[1];
  return 8192;
}

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

// std::vector wrapper for fftconv routines
#define VECTOR_WRAPPER(CONV_FUNC)                                              \
  inline std::vector<double> CONV_FUNC(const std::vector<double> &x,           \
                                       const std::vector<double> &b) {         \
    std::vector<double> y(b.size() + x.size() - 1);                            \
    CONV_FUNC(x.data(), x.size(), b.data(), b.size(), y.data(), y.size());     \
    return y;                                                                  \
  }
#define VECTOR_WRAPPER_FFTCONV_POCKETFFT_HDR(CONV_FUNC)                        \
  inline std::vector<double> CONV_FUNC(const std::vector<double> &x,           \
                                       const std::vector<double> &b,           \
                                       const size_t nthreads = 1) {            \
    std::vector<double> y(b.size() + x.size() - 1);                            \
    CONV_FUNC(x.data(), x.size(), b.data(), b.size(), y.data(), y.size(),      \
              nthreads);                                                       \
    return y;                                                                  \
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

template <class T>
void elementwise_multiply(const T *a, const T *b, const size_t length,
                          T *result) {
  for (size_t i = 0; i < length; ++i)
    result[i] = a[i] * b[i];
}

// template <class T>
// void elementwise_multiply_half_cx(const T *a, const T *b, const size_t
// length, T *result) {
// result[0] = a[0] * b[0];
// auto _a = reinterpret_cast<const std::complex<T> *>(a + 1);
// auto _b = reinterpret_cast<const std::complex<T> *>(b + 1);
// auto _c = reinterpret_cast<std::complex<T> *>(result + 1);

// const auto _len = (length - 1) >> 1;
// for (size_t i = 0; i < _len; ++i)
//_c[i] = _a[i] * _b[i];

// if (length % 2 == 0)
// result[length - 1] = a[length - 1] * b[length - 1];
//}

template <class T>
void elementwise_multiply_half_cx(const T *a, const T *b, const size_t length,
                                  T *result) {
  result[0] = a[0] * b[0];
  for (size_t i = 1; i < length - 1; i += 2) {
    const T a1 = a[i], a2 = a[i + 1];
    const T b1 = b[i], b2 = b[i + 1];
    result[i] = a1 * b1 - a2 * b2;
    result[i + 1] = a2 * b1 + a1 * b2;
  }

  if (length % 2 == 0)
    result[length - 1] = a[length - 1] * b[length - 1];
}

namespace fftconv {

template <class T> struct Bufs {
  size_t size_r;
  T *r_buf;
  std::complex<T> *A_buf;
  std::complex<T> *B_buf;
  const pocketfft::shape_t shape_r;
  const pocketfft::shape_t shape_c;
  const pocketfft::stride_t stride_r;
  const pocketfft::stride_t stride_c;

  Bufs(const size_t size_r, const size_t size_c)
      : size_r(size_r), r_buf(new double[size_r]),
        A_buf(new std::complex<double>[size_c]),
        B_buf(new std::complex<double>[size_c]), shape_r({size_r}),
        shape_c({size_c}), stride_r({sizeof(T)}),
        stride_c({sizeof(std::complex<T>)}) {}
  Bufs(const size_t size_r) : Bufs(size_r, (size_r >> 1) + 1) {}

  ~Bufs() {
    delete[] r_buf;
    delete[] A_buf;
    delete[] B_buf;
  }
};

static thread_local Cache<size_t, Bufs<double>> bufs_cache;

template <class T> struct Bufs_half_cx {
  const size_t size_r;
  T *A_buf;
  T *B_buf;
  const pocketfft::shape_t shape_r;
  const pocketfft::stride_t stride_r;
  const pocketfft::stride_t stride_c;

  Bufs_half_cx(const size_t size_r)
      : size_r(size_r), A_buf(new double[size_r]), B_buf(new double[size_r]),
        shape_r({size_r}), stride_r({sizeof(T)}), stride_c({sizeof(T)}) {}

  ~Bufs_half_cx() {
    delete[] A_buf;
    delete[] B_buf;
  }
};

static thread_local Cache<size_t, Bufs_half_cx<double>> bufs_half_cx_cache;

struct pocketfft_plan {
  double *a_buf;
  double *b_buf;
  rfft_plan plan;

  pocketfft_plan(const size_t size_r)
      : a_buf(new double[size_r]), b_buf(new double[size_r]),
        plan(make_rfft_plan(size_r)) {}

  pocketfft_plan() = delete;                       // default constructor
  pocketfft_plan(pocketfft_plan &&) = delete;      // move constructor
  pocketfft_plan(const pocketfft_plan &) = delete; // copy constructor
  pocketfft_plan &operator=(const pocketfft_plan) = delete; // copy assignment
  pocketfft_plan &operator=(pocketfft_plan &&) = delete;    // move assignment

  ~pocketfft_plan() {
    destroy_rfft_plan(plan);
    delete[] a_buf;
    delete[] b_buf;
  }
};

static thread_local Cache<size_t, pocketfft_plan> pocketfft_plan_cache;

inline void convolve_pocketfft(const double *a, const size_t a_sz,
                               const double *b, const size_t b_sz, double *res,
                               const size_t res_sz) {
  // length of the real arrays, including the final convolution output
  const size_t size_r = a_sz + b_sz - 1;

  auto _plan = pocketfft_plan_cache.get(size_r);
  auto a_buf = _plan->a_buf;
  auto b_buf = _plan->b_buf;
  const auto &plan = _plan->plan;

  _copy_to_padded_buffer(a, a_sz, a_buf, size_r);
  // std::cout << "a_buf: ";
  // print(a_buf, size_r);
  rfft_forward(plan, a_buf, 1.);
  // std::cout << "a_buf: ";
  // print(a_buf, size_r);

  _copy_to_padded_buffer(b, b_sz, b_buf, size_r);
  // std::cout << "b_buf: ";
  // print(b_buf, size_r);
  rfft_forward(plan, b_buf, 1.);
  // std::cout << "b_buf: ";
  // print(b_buf, size_r);

  elementwise_multiply_half_cx(a_buf, b_buf, size_r, a_buf);
  // std::cout << "a_buf: ";
  // print(a_buf, size_r);

  rfft_backward(plan, a_buf, 1. / size_r);

  const auto end = std::max(size_r, res_sz);
  for (size_t i = 0; i < end; i++)
    res[i] = a_buf[i];
}

inline void oaconvolve_pocketfft(const double *a, const size_t a_sz,
                                 const double *b, const size_t b_sz,
                                 double *res, const size_t res_sz) {
  const size_t size_r = get_optimal_fft_size(b_sz);
  const size_t step_size = size_r - (b_sz - 1);

  auto _plan = pocketfft_plan_cache.get(size_r);
  auto &a_buf = _plan->a_buf;
  auto &b_buf = _plan->b_buf;
  const auto &plan = _plan->plan;

  _copy_to_padded_buffer(b, b_sz, b_buf, size_r);
  rfft_forward(plan, b_buf, 1.);

  for (size_t pos = 0; pos < a_sz; pos += step_size) {
    size_t len = std::min(a_sz - pos, step_size);
    _copy_to_padded_buffer(a + pos, len, a_buf, size_r);

    rfft_forward(plan, a_buf, 1.);
    elementwise_multiply_half_cx(a_buf, b_buf, size_r, a_buf);
    rfft_backward(plan, a_buf, 1.); // normalize later

    len = std::min(res_sz - pos, size_r);
    for (size_t i = 0; i < len; ++i)
      res[pos + i] += a_buf[i] / size_r;
  }
}

inline void convolve_pocketfft_hdr(const double *a, const size_t a_sz,
                                   const double *b, const size_t b_sz,
                                   double *res, const size_t res_sz,
                                   size_t nthreads = 1) {
  using namespace pocketfft;
  using std::complex;

  const shape_t axis = {0};

  // length of the real arrays, including the final convolution output
  const size_t size_r = a_sz + b_sz - 1;
  // const size_t size_c = (size_r >> 1) + 1;

  // Allocate buffers
  auto bufs = bufs_half_cx_cache.get(size_r);
  auto A_buf = bufs->A_buf;
  auto B_buf = bufs->B_buf;
  const auto &shape_r = bufs->shape_r;
  const auto &stride_r = bufs->stride_r;
  const auto &stride_c = bufs->stride_c;

  _copy_to_padded_buffer(a, a_sz, A_buf, size_r);
  pocketfft::r2r_fftpack(shape_r, stride_r, stride_c, axis, true, FORWARD,
                         A_buf, A_buf, 1., nthreads);
  // std::cout << "A_buf: ";
  // print(A_buf, size_r);

  _copy_to_padded_buffer(b, b_sz, B_buf, size_r);
  pocketfft::r2r_fftpack(shape_r, stride_r, stride_c, axis, true, FORWARD,
                         B_buf, B_buf, 1., nthreads);
  // std::cout << "B_buf: ";
  // print(B_buf, size_r);

  elementwise_multiply_half_cx(A_buf, B_buf, size_r, A_buf);
  // std::cout << "A_buf: ";
  // print(A_buf, size_r);

  pocketfft::r2r_fftpack(shape_r, stride_c, stride_r, axis, false, BACKWARD,
                         A_buf, A_buf, 1., nthreads); // normalize later
  // std::cout << "r    : ";
  // print(r_buf, size_r);

  const auto end = std::min(size_r, res_sz);
  const double fct = 1. / size_r;
  for (size_t i = 0; i < end; ++i)
    res[i] = A_buf[i] * fct;
}

inline void oaconvolve_pocketfft_hdr(const double *a, const size_t a_sz,
                                     const double *b, const size_t b_sz,
                                     double *res, const size_t res_sz,
                                     size_t nthreads = 1) {
  using namespace pocketfft;

  const shape_t axis{0};

  const size_t size_r = get_optimal_fft_size(b_sz);
  const size_t step_size = size_r - (b_sz - 1);
  // const size_t size_c = (size_r >> 1) + 1;

  // Allocate buffers
  auto bufs = bufs_half_cx_cache.get(size_r);
  auto A_buf = bufs->A_buf;
  auto B_buf = bufs->B_buf;
  const auto &shape_r = bufs->shape_r;
  const auto &stride_r = bufs->stride_r;
  const auto &stride_c = bufs->stride_c;

  _copy_to_padded_buffer(b, b_sz, B_buf, size_r);
  pocketfft::r2r_fftpack(shape_r, stride_r, stride_c, axis, true, FORWARD,
                         B_buf, B_buf, 1., nthreads);

  const double fct = 1. / size_r;
  for (size_t pos = 0; pos < a_sz; pos += step_size) {
    size_t len = std::min(a_sz - pos, step_size);
    _copy_to_padded_buffer(a + pos, len, A_buf, size_r);
    pocketfft::r2r_fftpack(shape_r, stride_r, stride_c, axis, true, FORWARD,
                           A_buf, A_buf, 1., nthreads);
    elementwise_multiply_half_cx(A_buf, B_buf, size_r, A_buf);
    pocketfft::r2r_fftpack(shape_r, stride_c, stride_r, axis, false, BACKWARD,
                           A_buf, A_buf, 1., nthreads); // normalize later

    len = std::min(res_sz - pos, size_r);
    for (size_t i = 0; i < len; ++i)
      res[pos + i] += A_buf[i] * fct;
  }
}

VECTOR_WRAPPER(convolve_pocketfft)
VECTOR_WRAPPER(oaconvolve_pocketfft)
VECTOR_WRAPPER_FFTCONV_POCKETFFT_HDR(convolve_pocketfft_hdr)
VECTOR_WRAPPER_FFTCONV_POCKETFFT_HDR(oaconvolve_pocketfft_hdr)

} // namespace fftconv

#undef VECTOR_WRAPPER
#undef VECTOR_WRAPPER_FFTCONV_POCKETFFT_HDR
