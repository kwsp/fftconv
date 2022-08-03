// Author: Tiger Nie
// 2022

#include "fftconv.h"
#include <array>
#include <memory>
#include <shared_mutex>
#include <unordered_map>

using std::array;
using std::vector;

static int nextpow2(int x) { return 1 << (int)(std::log2(x) + 1); }

// Lookup table of {max_filter_size, optimal_fft_size}
static const std::array<std::array<size_t, 2>, 9> _optimal_fft_size{
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
static size_t get_optimal_fft_size(size_t filter_size) {
  for (const auto &pair : _optimal_fft_size)
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

static void elementwise_multiply(const fftw_complex *a, const fftw_complex *b,
                                 const size_t length, fftw_complex *result) {
  for (auto i = 0; i < length; ++i) {
    std::complex<double> _a(a[i][0], a[i][1]);
    std::complex<double> _b(b[i][0], b[i][1]);
    _a *= _b;
    result[i][0] = _a.real();
    result[i][1] = _a.imag();
  }
}

enum class fft_direction { Forward = 0b01, Backward = 0b10 };

struct fft_plan {
  fftw_plan plan;
  double *real_buf;
  fftw_complex *complex_buf;

  // Padded length must be an even power of 2
  // The lowest 2 bits are used as direction flags
  fft_plan(size_t padded_length) {
    // length of the complex arrays
    size_t complex_length = padded_length / 2 + 1;
    real_buf = (double *)fftw_alloc_real(padded_length);
    complex_buf = (fftw_complex *)fftw_alloc_complex(complex_length);

    // Check direction
    if ((int)padded_length & (int)fft_direction::Forward)
      plan = fftw_plan_dft_r2c_1d(padded_length, real_buf, complex_buf,
                                  FFTW_ESTIMATE);
    else
      plan = fftw_plan_dft_c2r_1d(padded_length, complex_buf, real_buf,
                                  FFTW_ESTIMATE);
  }

  ~fft_plan() {
    fftw_destroy_plan(plan);
    fftw_free(real_buf);
    fftw_free(complex_buf);
  }
};

// fftconv_plans manages the memory of the forward and backward fft plans
// and the fftw buffers
class fftconv_plans {
public:
  // FFTW plans
  fftw_plan plan_f; // forward
  fftw_plan plan_b; // backward

  // FFTW buffers corresponding to the above plans
  double *real_buf;
  fftw_complex *complex_buf_a;
  fftw_complex *complex_buf_b;

  // FFTW buffer sizes
  const size_t real_sz;
  const size_t complex_sz;

public:
  // Constructors
  fftconv_plans(size_t padded_length);
  fftconv_plans() = delete; // default constructor

  fftconv_plans(fftconv_plans &&) = delete;      // move constructor
  fftconv_plans(const fftconv_plans &) = delete; // copy constructor

  fftconv_plans &operator=(const fftconv_plans) = delete; // copy assignment
  fftconv_plans &operator=(fftconv_plans &&) = delete;    // move assignment

  // Destructor
  ~fftconv_plans() {

    fftw_destroy_plan(plan_f);
    fftw_destroy_plan(plan_b);

    fftw_free(real_buf); // real_buf is the only fftw_malloc
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

  void complex_multiply_to_a() {
    // A_buf becomes input to inverse conv
    elementwise_multiply(complex_buf_a, complex_buf_b, complex_sz,
                         complex_buf_a);
  }

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

// Compute the fftw plans and allocate buffers
fftconv_plans::fftconv_plans(size_t padded_length)
    : real_sz(padded_length), complex_sz(padded_length / 2 + 1) {
  // Allocate buffers for the purposes of creating the plans...
  real_buf = (double *)fftw_malloc(padded_length * sizeof(double) +
                                   2 * complex_sz * sizeof(fftw_complex));
  complex_buf_a = (fftw_complex *)(real_buf + padded_length);
  complex_buf_b = complex_buf_a + complex_sz;

  // Compute the plans
  plan_f = fftw_plan_dft_r2c_1d(padded_length, real_buf, complex_buf_a,
                                FFTW_ESTIMATE);
  plan_b = fftw_plan_dft_c2r_1d(padded_length, complex_buf_a, real_buf,
                                FFTW_ESTIMATE);
}

// In memory cache with key type K and value type V
// V's constructor must take K as input
template <class K, class V> class Cache {
public:
  // Get a cached object if available.
  // Otherwise, a new one will be constructed.
  V *get(K key) {
    // _cache is thread_local
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

static thread_local Cache<size_t, fft_plan> fft_plan_cache;
static thread_local Cache<size_t, fftconv_plans> fftconv_plans_cache;

namespace fftconv {

void convolve1d(const double *a, const size_t a_size, const double *b,
                const size_t b_size, double *result) {
  // length of the real arrays, including the final convolution output
  size_t padded_length = a_size + b_size - 1;

  // Get cached plans
  fftconv_plans *plan = fftconv_plans_cache(padded_length);
  plan->execute_conv(a, a_size, b, b_size);

  // copy normalized to result
  const auto real_buf = plan->get_real_buf();
  for (int i = 0; i < padded_length; i++) {
    result[i] = real_buf[i];
  }
}

void convolve1d_ref(const double *a, const size_t a_size, const double *b,
                    const size_t b_size, double *result) {
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
  elementwise_multiply(A_buf, B_buf, complex_length,
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

std::vector<double> convolve1d_ref(const vector<double> &a,
                                   const vector<double> &b) {
  int padded_length = a.size() + b.size() - 1;
  vector<double> result(padded_length);
  convolve1d_ref(a.data(), a.size(), b.data(), b.size(), result.data());
  return result;
}

// Overlap-Add convolution of x and h with block length B
//
// x is a long signal
// b is a kernel, x_size >> b_size
// y is the results buffer. y_size >= x_size + b_size - 1
//
// B is the block size
//
// 1. Split x into blocks of size B.
// 2. convolve with kernel b. The size of the convolution should be
//    (B + y_size - 1)
// 3. add blocks together
void fftfilt(const double *x, const size_t x_size, const double *h,
             const size_t h_size, double *y, const size_t y_size) {
  // Use overlap and add to convolve x and b
  // const size_t N = 8 * nextpow2(h_size); // size for each fft
  const size_t N = get_optimal_fft_size(h_size); // size for each fft
  const size_t step_size = N - (h_size - 1);

  // forward fft of b
  auto plan = fftconv_plans_cache(N);
  plan->set_real_buf(h, h_size);
  plan->forward_b();

  // create forward/backward ffts for x
  auto real_buf = plan->get_real_buf();
  for (size_t pos = 0; pos < x_size; pos += step_size) {
    size_t len = std::min(x_size - pos, step_size); // bound check
    plan->set_real_buf(x + pos, len);
    plan->forward_a();
    plan->complex_multiply_to_a();
    plan->backward();
    // plan->normalize(); // either normalize here or later in the copy loop

    // normalize output and add to result
    len = std::min(y_size - pos, N);
    for (size_t i = 0; i < len; ++i)
      y[pos + i] += real_buf[i] / N;
  }
}

std::vector<double> fftfilt(const std::vector<double> &x,
                            const std::vector<double> &b) {
  // Use overlap and add to convolve x and b
  const size_t y_size = b.size() + x.size() - 1;
  std::vector<double> y(y_size);
  fftfilt(x.data(), x.size(), b.data(), b.size(), y.data(), y_size);
  return y;
}

} // namespace fftconv
