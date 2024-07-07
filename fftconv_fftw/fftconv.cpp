// Author: Tiger Nie
// 2022
// https://github.com/kwsp/fftconv

#include "fftconv.hpp"
#include <algorithm>
#include <array>
#include <complex>
#include <cstring>
#include <debug_utils.hpp>

std::mutex *_fftw_mutex = nullptr;

// static int nextpow2(int x) { return 1 << (int)(std::log2(x) + 1); }

// Lookup table of {max_filter_size, optimal_fft_size}
static constexpr std::array<std::array<size_t, 2>, 9> _optimal_oa_fft_size{
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

// Given a filter_size, return the optimal fft size for the overlap-add
// convolution method
static size_t get_optimal_fft_size(const size_t filter_size) {
  for (const auto &pair : _optimal_oa_fft_size)
    if (filter_size < pair[0])
      return pair[1];
  return 8192;
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

// static inline void elementwise_multiply(const fftw_complex *a,
// const fftw_complex *b,
// const size_t length,
// fftw_complex *result) {
//// fftw_complex in C89 mode is double[2], which is binary compatible with
//// C99's <complex.h> and C++'s complex<double> template class
//// http://www.fftw.org/doc/Complex-numbers.html
// const auto _a = reinterpret_cast<const std::complex<double> *>(a);
// const auto _b = reinterpret_cast<const std::complex<double> *>(b);
// auto _res = reinterpret_cast<std::complex<double> *>(result);

// for (size_t i = 0; i < length; ++i)
//_res[i] = _a[i] * _b[i];
//}

static inline void elementwise_multiply(const fftw_complex *a,
                                        const fftw_complex *b,
                                        const size_t length,
                                        fftw_complex *result) {
  // implement naive complex multiply. This is much faster than the
  // std version, but doesn't handle infinities.
  // https://stackoverflow.com/questions/49438158/why-is-muldc3-called-when-two-stdcomplex-are-multiplied

  for (size_t i = 0; i < length; ++i) {
    const double a1 = a[i][0], a2 = a[i][1];
    const double b1 = b[i][0], b2 = b[i][1];
    result[i][0] = a1 * b1 - a2 * b2;
    result[i][1] = a2 * b1 + a1 * b2;
  }
}

// enum class fft_direction { Forward = 0b01, Backward = 0b10 };

// struct fft_plan {
// fftw_plan plan;
// double *real_buf;
// fftw_complex *complex_buf;

//// Padded length must be an even power of 2
//// The lowest 2 bits are used as direction flags
// fft_plan(size_t padded_length) {
//// length of the complex arrays
// size_t complex_length = padded_length / 2 + 1;
// real_buf = (double *)fftw_alloc_real(padded_length);
// complex_buf = (fftw_complex *)fftw_alloc_complex(complex_length);

//// Check direction
// if ((int)padded_length & (int)fft_direction::Forward)
// plan = fftw_plan_dft_r2c_1d(padded_length, real_buf, complex_buf,
// FFTW_ESTIMATE);
// else
// plan = fftw_plan_dft_c2r_1d(padded_length, complex_buf, real_buf,
// FFTW_ESTIMATE);
//}

//~fft_plan() {
// fftw_destroy_plan(plan);
// fftw_free(real_buf);
// fftw_free(complex_buf);
//}
//};

// fftconv_plans manages the memory of the forward and backward fft plans
// and the fftw buffers
// Plans are for the FFTW New-Array Execute Functions
// https://www.fftw.org/fftw3_doc/New_002darray-Execute-Functions.html
//
// Not using half complex transforms before it's non-trivial to do complex
// multiplications with FFTW's half complex format
class fftconv_plans {
private:
  // FFTW buffer sizes
  const size_t real_sz;
  const size_t complex_sz;

  // FFTW buffers corresponding to the above plans
  double *real_buf;
  fftw_complex *complex_buf_a;
  fftw_complex *complex_buf_b;

  // FFTW plans
  fftw_plan plan_f; // forward
  fftw_plan plan_b; // backward

public:
  // Constructors
  // Compute the fftw plans and allocate buffers
  fftconv_plans(size_t padded_length)
      : real_sz(padded_length), complex_sz(padded_length / 2 + 1) {
    if (_fftw_mutex)
      _fftw_mutex->lock();

    real_buf = fftw_alloc_real(real_sz);
    complex_buf_a = fftw_alloc_complex(complex_sz);
    complex_buf_b = fftw_alloc_complex(complex_sz);

    plan_f = fftw_plan_dft_r2c_1d(padded_length, real_buf, complex_buf_a,
                                  FFTW_ESTIMATE);
    plan_b = fftw_plan_dft_c2r_1d(padded_length, complex_buf_a, real_buf,
                                  FFTW_ESTIMATE);
    if (_fftw_mutex)
      _fftw_mutex->unlock();
  }

  fftconv_plans() = delete;                               // default constructor
  fftconv_plans(fftconv_plans &&) = delete;               // move constructor
  fftconv_plans(const fftconv_plans &) = delete;          // copy constructor
  fftconv_plans &operator=(const fftconv_plans) = delete; // copy assignment
  fftconv_plans &operator=(fftconv_plans &&) = delete;    // move assignment

  // Destructor
  ~fftconv_plans() {
    fftw_free(real_buf);
    fftw_free(complex_buf_a);
    fftw_free(complex_buf_b);
    fftw_destroy_plan(plan_f);
    fftw_destroy_plan(plan_b);
  }

  void set_real_buf(const double *inp, size_t sz) {
    _copy_to_padded_buffer(inp, sz, this->real_buf, this->real_sz);
  }
  constexpr double *get_real_buf() const { return this->real_buf; }
  constexpr size_t get_real_sz() const { return this->real_sz; }

  void forward_a() { fftw_execute_dft_r2c(plan_f, real_buf, complex_buf_a); }
  void forward_b() { fftw_execute_dft_r2c(plan_f, real_buf, complex_buf_b); }
  void backward() { fftw_execute_dft_c2r(plan_b, complex_buf_a, real_buf); }
  void normalize() {
    for (size_t i = 0; i < real_sz; ++i)
      real_buf[i] /= real_sz;
  }

  // Complex element-wise multiply a and b, save results to a
  void complex_multiply_to_a() {
    elementwise_multiply(complex_buf_a, complex_buf_b, complex_sz,
                         complex_buf_a);
  }

  // Convolve real arrays a and b
  // Results saved in real_buf
  void execute_conv(const double *a, const size_t a_size, const double *b,
                    const size_t b_size) {
    set_real_buf(a, a_size); // Copy a to buffer
    forward_a();             // A = fft(a)
    set_real_buf(b, b_size); // Copy b to buffer
    forward_b();             // B = fft(b)
    complex_multiply_to_a(); // Complex elementwise multiple, A = A * B
    backward();              // a = ifft(A)
    normalize();             // divide each result elem by real_sz
  }
};

static thread_local fftconv::Cache<size_t, fftconv_plans> fftconv_plans_cache;

// fftconv_plans manages the memory of the forward and backward fft plans
// and the fftw buffers
// Plans are for the FFTW New-Array Execute Functions
// https://www.fftw.org/fftw3_doc/New_002darray-Execute-Functions.html
struct fftconv_plans_advanced {
  // FFTW buffer sizes
  const size_t real_sz;
  const size_t complex_sz;
  const int howmany;

  // FFTW buffers corresponding to the above plans
  double *real_buf_signal;
  fftw_complex *cx_buf_signal;

  double *real_buf_kernel;
  fftw_complex *cx_buf_kernel;

  // FFTW plans
  fftw_plan plan_forward_kernel;
  fftw_plan plan_forward_signal;
  fftw_plan plan_backward_signal;

  void debug_print() {
#define PRINT(NAME) std::cout << #NAME << " (" << NAME << ") "
    PRINT(real_sz);
    PRINT(complex_sz);
    PRINT(howmany);
    std::cout << "\n";

    std::cout << "real_buf_kernel";
    print(real_buf_kernel, real_sz);

    std::cout << "cx_buf_kernel";
    print(cx_buf_kernel, complex_sz);

    std::cout << "real_buf_signal";
    print(real_buf_signal, real_sz * howmany);

    std::cout << "cx_buf_signal";
    print(cx_buf_signal, complex_sz * howmany);
  }

  // Use advanced interface
  fftconv_plans_advanced(const size_t padded_length, const int howmany)
      : real_sz(padded_length), complex_sz(padded_length / 2 + 1),
        howmany(howmany) {

    if (_fftw_mutex)
      _fftw_mutex->lock();

    real_buf_signal = fftw_alloc_real(real_sz * howmany);
    cx_buf_signal = fftw_alloc_complex(complex_sz * howmany);

    real_buf_kernel = fftw_alloc_real(real_sz);
    cx_buf_kernel = fftw_alloc_complex(complex_sz);

    // `howmany` is the (nonnegative) number of transforms to compute.
    // - The resulting plan computs `howmany` transforms, where the input of
    //   the k-th transform is at location `in+k*idist`
    // - Each of `howmany` has rank `rank` and size `n`
    //
    // int howmany;

    int rank = 1; // Dimensions of each transform.
    const int n[] = {(int)real_sz};

    double *in = real_buf_signal;
    fftw_complex *out = cx_buf_signal;

    // {i,o}nembed must be arrays of size `rank`
    // or NULL (equivalent to passing `n`)
    const int *nembed = NULL;

    int stride = 1;
    int idist = real_sz, odist = complex_sz;

    plan_forward_kernel = fftw_plan_dft_r2c_1d(real_sz, real_buf_kernel,
                                               cx_buf_kernel, FFTW_ESTIMATE);

    plan_forward_signal =
        fftw_plan_many_dft_r2c(rank, n, howmany, in, nembed, stride, idist, out,
                               nembed, stride, odist, FFTW_ESTIMATE);

    plan_backward_signal =
        fftw_plan_many_dft_c2r(rank, n, howmany, out, nembed, stride, odist, in,
                               nembed, stride, idist, FFTW_ESTIMATE);

    if (_fftw_mutex)
      _fftw_mutex->unlock();
  }

  ~fftconv_plans_advanced() {
    fftw_free(real_buf_signal);
    fftw_free(cx_buf_signal);

    fftw_free(real_buf_kernel);
    fftw_free(cx_buf_kernel);

    fftw_destroy_plan(plan_forward_kernel);
    fftw_destroy_plan(plan_forward_signal);
    fftw_destroy_plan(plan_backward_signal);
  }

  void set_kernel(const double *arr, size_t sz) {
    assert(sz <= real_sz);
    _copy_to_padded_buffer(arr, sz, real_buf_kernel, real_sz);
  }

  // i-th `howmany`
  void set_signal(const double *arr, const size_t sz, const size_t idx) {
    assert(sz <= real_sz);
    assert(idx <= howmany);
    _copy_to_padded_buffer(arr, sz, real_buf_signal + idx * real_sz, real_sz);
  }

  void forward_kernel() {
    fftw_execute_dft_r2c(this->plan_forward_kernel, real_buf_kernel,
                         cx_buf_kernel);
  }
  void forward_signal() {
    fftw_execute_dft_r2c(this->plan_forward_signal, real_buf_signal,
                         cx_buf_signal);
  }
  void backward() {
    fftw_execute_dft_c2r(this->plan_backward_signal, cx_buf_signal,
                         real_buf_signal);
  }

  void complex_multiply() {
    for (int i = 0; i < howmany; ++i)
      elementwise_multiply(cx_buf_signal + i * complex_sz, cx_buf_kernel,
                           complex_sz, cx_buf_signal + i * complex_sz);
  }

  void get_output(double *arr, size_t sz, const size_t idx) {
    const double fct = 1. / real_sz;
    sz = std::min(real_sz, sz);

    const size_t pos = idx * real_sz;
    for (int i = 0; i < sz; ++i)
      arr[i] += real_buf_signal[pos + i] * fct;
  }
};

fftconv_plans_advanced *fftconv_plans_advanced_cache(const size_t padded_length,
                                                     const int howmany) {
  static thread_local std::unordered_map<
      size_t, std::unique_ptr<fftconv_plans_advanced>>
      _cache;
  const size_t _hash = (padded_length << 4) ^ howmany;

  auto &plan = _cache[_hash];
  if (plan == nullptr || plan->real_sz != padded_length)
    plan = std::make_unique<fftconv_plans_advanced>(padded_length, howmany);
  return plan.get();
}

/////////////////////////////

namespace fftconv {

void use_fftw_mutex(std::mutex *fftw_mutex) { _fftw_mutex = fftw_mutex; }

void convolve_fftw(const double *a, const size_t a_sz, const double *b,
                   const size_t b_sz, double *result, const size_t res_sz) {
  // length of the real arrays, including the final convolution output
  const size_t padded_length = a_sz + b_sz - 1;

  // Get cached plans
  fftconv_plans *plan = fftconv_plans_cache(padded_length);
  plan->execute_conv(a, a_sz, b, b_sz);

  // copy normalized to result
  const auto real_buf = plan->get_real_buf();
  const size_t end = std::min(padded_length, res_sz);
  for (int i = 0; i < end; i++)
    result[i] = real_buf[i];
}

void convolve_fftw_advanced(const double *a, const size_t a_sz, const double *b,
                            const size_t b_sz, double *result,
                            const size_t res_sz) {
  const size_t padded_length = a_sz + b_sz - 1;
  auto plans = fftconv_plans_advanced_cache(padded_length, 1);

  plans->set_kernel(b, b_sz);
  plans->forward_kernel();

  plans->set_signal(a, a_sz, 0);
  plans->forward_signal();

  plans->complex_multiply();
  plans->backward();
  plans->get_output(result, res_sz, 0);
}

// reference implementation of fftconv with no optimizations
void convolve_fftw_ref(const double *a, const size_t a_sz, const double *b,
                       const size_t b_sz, double *result,
                       const size_t result_sz) {
  // length of the real arrays, including the final convolution output
  const size_t padded_length = a_sz + b_sz - 1;
  // length of the complex arrays
  const size_t complex_length = padded_length / 2 + 1;

  // Allocate fftw buffers for a
  double *a_buf = fftw_alloc_real(padded_length);
  fftw_complex *A_buf = fftw_alloc_complex(complex_length);

  // Compute forward fft plan
  fftw_plan plan_forward =
      fftw_plan_dft_r2c_1d(padded_length, a_buf, A_buf, FFTW_ESTIMATE);

  // Copy a to buffer
  _copy_to_padded_buffer(a, a_sz, a_buf, padded_length);

  // Compute Fourier transform of vector a
  fftw_execute_dft_r2c(plan_forward, a_buf, A_buf);

  // Allocate fftw buffers for b
  double *b_buf = fftw_alloc_real(padded_length);
  fftw_complex *B_buf = fftw_alloc_complex(complex_length);

  // Copy b to buffer
  _copy_to_padded_buffer(b, b_sz, b_buf, padded_length);

  // Compute Fourier transform of vector b
  fftw_execute_dft_r2c(plan_forward, b_buf, B_buf);

  // Compute backward fft plan
  fftw_complex *input_buffer = fftw_alloc_complex(complex_length);
  double *output_buffer = fftw_alloc_real(padded_length);
  fftw_plan plan_backward = fftw_plan_dft_c2r_1d(padded_length, input_buffer,
                                                 output_buffer, FFTW_ESTIMATE);

  // Perform element-wise product of FFT(a) and FFT(b)
  // then compute inverse fourier transform.
  elementwise_multiply(A_buf, B_buf, complex_length,
                       input_buffer); // A_buf becomes input to inverse conv

  fftw_execute_dft_c2r(plan_backward, input_buffer, output_buffer);

  // Normalize output
  for (int i = 0; i < std::min(padded_length, result_sz); i++)
    result[i] = output_buffer[i] / padded_length;

  fftw_free(a_buf);
  fftw_free(b_buf);
  fftw_free(A_buf);
  fftw_free(B_buf);
  fftw_free(input_buffer);
  fftw_free(output_buffer);
  fftw_destroy_plan(plan_forward);
  fftw_destroy_plan(plan_backward);
}

// Overlap-Add convolution of x and h
//
// x is a long signal
// h is a kernel, x_size >> h_size
// y is the results buffer. y_size >= x_size + b_size - 1
//
// 1. Split x into blocks of step_size.
// 2. convolve with kernel b using fft of length N.
// 3. add blocks together
void oaconvolve_fftw(const double *x, const size_t x_sz, const double *h,
                     const size_t h_sz, double *y, const size_t y_sz) {

  // const size_t N = 8 * nextpow2(h_size); // size for each fft
  const size_t N = get_optimal_fft_size(h_sz); // more optimal size for each fft
  const size_t step_size = N - (h_sz - 1);

  // forward fft of h
  auto plan = fftconv_plans_cache(N);
  plan->set_real_buf(h, h_sz);
  plan->forward_b();

  // create forward/backward ffts for x
  const auto real_buf = plan->get_real_buf();
  const double fct = 1. / N;
  for (size_t pos = 0; pos < x_sz; pos += step_size) {
    size_t len = std::min(x_sz - pos, step_size); // bound check
    plan->set_real_buf(x + pos, len);
    plan->forward_a();
    plan->complex_multiply_to_a();
    plan->backward();
    // plan->normalize(); // normalize later in the copy loop

    // normalize output and add to result
    len = std::min(y_sz - pos, N);
    for (size_t i = 0; i < len; ++i)
      y[pos + i] += real_buf[i] * fct;
  }
}

void oaconvolve_fftw_advanced(const double *x, const size_t x_sz,
                              const double *h, const size_t h_sz, double *y,
                              const size_t y_sz) {
  const size_t N = get_optimal_fft_size(h_sz); // more optimal size for each fft
  const size_t step_size = N - (h_sz - 1);
  const size_t batch_sz = x_sz / step_size;

  auto plans =
      fftconv_plans_advanced_cache(N, batch_sz + 1); // last batch zero pad
  plans->set_kernel(h, h_sz);
  plans->forward_kernel();

  // Copy data to plan
  for (size_t pos = 0, idx = 0; pos < x_sz; pos += step_size, idx++) {
    size_t len = std::min(x_sz - pos, step_size); // bound check
    plans->set_signal(x + pos, len, idx);
  }

  plans->forward_signal();

  plans->complex_multiply();

  plans->backward();

  for (size_t pos = 0, idx = 0; pos < y_sz; pos += step_size, idx++) {
    size_t len = std::min(y_sz - pos, N); // bound check
    plans->get_output(y + pos, len, idx);
  }
}

} // namespace fftconv
