/**
C++ wrapper of FFTW3

Author: Taylor Nie
*/
#pragma once

#include <fftw3.h>

#include <complex>
#include <span>
#include <type_traits>

// NOLINTBEGIN(*-reinterpret-cast, *-const-cast, *-identifier-length)

namespace fftw {

// Place this at the beginning of main() and RAII will take care of setting up
// and tearing down FFTW3 (threads and wisdom)
struct FFTWGlobalSetup {
  FFTWGlobalSetup() {
    static bool callSetup = true;
    if (callSetup) {
      fftw_make_planner_thread_safe();
      callSetup = false;
    }
    fftw_import_wisdom_from_filename(".fftw_wisdom");
    fftwf_import_wisdom_from_filename(".fftwf_wisdom");
  }
  FFTWGlobalSetup(const FFTWGlobalSetup &) = delete;
  FFTWGlobalSetup(FFTWGlobalSetup &&) = delete;
  FFTWGlobalSetup &operator=(const FFTWGlobalSetup &) = delete;
  FFTWGlobalSetup &operator=(FFTWGlobalSetup &&) = delete;
  ~FFTWGlobalSetup() {
    fftw_export_wisdom_to_filename(".fftw_wisdom");
    fftwf_export_wisdom_to_filename(".fftwf_wisdom");
  }
};

constexpr int DEFAULT_PLANNER_FLAG = FFTW_EXHAUSTIVE;
// constexpr int DEFAULT_PLANNER_FLAG = FFTW_ESTIMATE;

template <typename T>
concept FloatOrDouble = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
concept FFTWComplex =
    std::is_same_v<T, fftw_complex> || std::is_same_v<T, fftwf_complex>;

template <typename T>
concept FFTWBufferSupported =
    FloatOrDouble<T> || std::is_same_v<T, std::complex<float>> ||
    std::is_same_v<T, std::complex<double>>;

template <FloatOrDouble> struct Traits {};

template <> struct Traits<float> {
  using Real = float;
  using Cx = std::complex<float>;
  using FFTW_Cx = fftwf_complex;
};

template <> struct Traits<double> {
  using Real = double;
  using Cx = std::complex<double>;
  using FFTW_Cx = fftw_complex;
};

template <FFTWBufferSupported T> constexpr T *_fftw_malloc(const size_t size) {
  return static_cast<T *>(fftw_malloc(size * sizeof(T)));
}

template <FFTWBufferSupported T> constexpr void _fftw_free(T *ptr) {
  if (ptr != nullptr) {
    if constexpr (std::is_same_v<T, double> ||
                  std::is_same_v<T, fftw_complex> ||
                  std::is_same_v<T, std::complex<double>>) {
      fftw_free(ptr);
    } else if constexpr (std::is_same_v<T, float> ||
                         std::is_same_v<T, fftwf_complex> ||
                         std::is_same_v<T, std::complex<float>>) {
      fftwf_free(ptr);
    }
  }
}

template <FFTWBufferSupported T> struct FFTWBufferAllocator {
  using value_type = T;
  FFTWBufferAllocator() = default;

  template <typename U>
  constexpr explicit FFTWBufferAllocator(
      const FFTWBufferAllocator<U> & /*unused*/) noexcept {}

  T *allocate(std::size_t n) { return _fftw_malloc<T>(n); }
  void deallocate(T *ptr, std::size_t) noexcept { _fftw_free(ptr); }
};
} // namespace fftw

// Allocator equality comparisons
template <typename T, typename U>
bool operator==(const fftw::FFTWBufferAllocator<T> &,
                const fftw::FFTWBufferAllocator<U> &) {
  return true;
}

template <typename T, typename U>
bool operator!=(const fftw::FFTWBufferAllocator<T> &,
                const fftw::FFTWBufferAllocator<U> &) {
  return false;
}

namespace fftw {

/// Encapsulates FFTW allocated buffer
template <FFTWBufferSupported T>
using fftw_buffer = std::vector<T, FFTWBufferAllocator<T>>;

template <typename T> struct Plan {};

template <> struct Plan<double> : Traits<double> {
  fftw_plan plan;

  Plan() = delete;
  explicit Plan(fftw_plan plan) : plan(plan) {}
  Plan(Plan &&other) noexcept : plan(other.plan) { other.plan = nullptr; };
  Plan(const Plan &) = delete; // Copy constructor
  Plan &operator=(Plan &&) = delete;
  Plan &operator=(const Plan &) = delete;

  static Plan<Real> plan_dft_r2c_1d(std::span<const Real> real,
                                    std::span<const Cx> complex,
                                    unsigned int flags = DEFAULT_PLANNER_FLAG) {
    const int n = static_cast<int>(real.size());
    auto *real_p = const_cast<Real *>(real.data());
    auto *cx_p = reinterpret_cast<FFTW_Cx *>(const_cast<Cx *>(complex.data()));
    Plan<Real> ret{fftw_plan_dft_r2c_1d(n, real_p, cx_p, flags)};
    return std::move(ret);
  }

  static Plan<Real> plan_dft_c2r_1d(std::span<const Cx> complex,
                                    std::span<const Real> real,
                                    unsigned int flags = DEFAULT_PLANNER_FLAG) {
    const int n = static_cast<int>(real.size());
    auto *real_p = const_cast<Real *>(real.data());
    auto *cx_p = reinterpret_cast<FFTW_Cx *>(const_cast<Cx *>(complex.data()));
    Plan<Real> ret{fftw_plan_dft_c2r_1d(n, cx_p, real_p, flags)};
    return std::move(ret);
  }

  static Plan<Real> plan_many_dft_r2c(const fftw_buffer<Real> &real,
                                      const fftw_buffer<Cx> &complex,
                                      int n_arrays,
                                      int flags = DEFAULT_PLANNER_FLAG) {
    auto *real_p = const_cast<Real *>(real.data());
    auto *cx_p = reinterpret_cast<FFTW_Cx *>(const_cast<Cx *>(complex.data()));

    const int real_size = static_cast<int>(real.size());
    const int cx_size = static_cast<int>(complex.size());
    int rank = 1;
    int stride = 1;
    Plan<Real> ret{fftw_plan_many_dft_r2c(rank, &real_size, n_arrays, real_p,
                                          nullptr, stride, real_size, cx_p,
                                          nullptr, stride, cx_size, flags)};

    return std::move(ret);
  }

  static Plan<Real> plan_many_dft_c2r(const fftw_buffer<Cx> &complex,
                                      const fftw_buffer<Real> &real,
                                      int n_arrays,
                                      int flags = DEFAULT_PLANNER_FLAG) {
    auto *real_p = const_cast<Real *>(real.data());
    auto *cx_p = reinterpret_cast<FFTW_Cx *>(const_cast<Cx *>(complex.data()));

    const int real_size = static_cast<int>(real.size());
    const int cx_size = static_cast<int>(complex.size());
    int rank = 1;
    int stride = 1;
    Plan<Real> ret{fftw_plan_many_dft_c2r(rank, &real_size, n_arrays, cx_p,
                                          nullptr, stride, cx_size, real_p,
                                          nullptr, stride, real_size, flags)};

    return std::move(ret);
  }

  void execute_dft_r2c(std::span<const Real> real,
                       std::span<const Cx> complex) const {
    auto *real_p = const_cast<Real *>(real.data());
    auto *cx_p = reinterpret_cast<FFTW_Cx *>(const_cast<Cx *>(complex.data()));
    fftw_execute_dft_r2c(plan, real_p, cx_p);
  }

  void execute_dft_c2r(std::span<const Cx> complex,
                       std::span<const Real> real) const {
    auto *real_p = const_cast<Real *>(real.data());
    auto *cx_p = reinterpret_cast<FFTW_Cx *>(const_cast<Cx *>(complex.data()));
    fftw_execute_dft_c2r(plan, cx_p, real_p);
  }

  ~Plan() {
    if (plan != nullptr) {
      fftw_destroy_plan(plan);
    }
  }
};

template <> struct Plan<float> : Traits<float> {
  fftwf_plan plan;

  Plan() = delete;
  explicit Plan(fftwf_plan plan) : plan(plan) {}
  Plan(Plan &&other) noexcept : plan(other.plan) { other.plan = nullptr; };
  Plan(const Plan &) = delete; // Copy constructor
  Plan &operator=(Plan &&) = delete;
  Plan &operator=(const Plan &) = delete;

  static Plan<Real> plan_dft_r2c_1d(std::span<const Real> real,
                                    std::span<const Cx> complex,
                                    unsigned int flags = DEFAULT_PLANNER_FLAG) {
    const int n = static_cast<int>(real.size());
    auto *real_p = const_cast<Real *>(real.data());
    auto *cx_p = reinterpret_cast<FFTW_Cx *>(const_cast<Cx *>(complex.data()));
    Plan<Real> ret{fftwf_plan_dft_r2c_1d(n, real_p, cx_p, flags)};
    return std::move(ret);
  }

  static Plan<Real> plan_dft_c2r_1d(std::span<const Cx> complex,
                                    std::span<const Real> real,
                                    unsigned int flags = DEFAULT_PLANNER_FLAG) {
    const int n = static_cast<int>(real.size());
    auto *real_p = const_cast<Real *>(real.data());
    auto *cx_p = reinterpret_cast<FFTW_Cx *>(const_cast<Cx *>(complex.data()));
    Plan<Real> ret{fftwf_plan_dft_c2r_1d(n, cx_p, real_p, flags)};
    return std::move(ret);
  }

  static Plan<Real> plan_many_dft_r2c(const fftw_buffer<Real> &real,
                                      const fftw_buffer<Cx> &complex,
                                      int n_arrays,
                                      int flags = DEFAULT_PLANNER_FLAG) {
    auto *real_p = const_cast<Real *>(real.data());
    auto *cx_p = reinterpret_cast<FFTW_Cx *>(const_cast<Cx *>(complex.data()));

    const int real_size = static_cast<int>(real.size());
    const int cx_size = static_cast<int>(complex.size());
    int rank = 1;
    int stride = 1;
    Plan<Real> ret{fftwf_plan_many_dft_r2c(rank, &real_size, n_arrays, real_p,
                                           nullptr, stride, real_size, cx_p,
                                           nullptr, stride, cx_size, flags)};

    return std::move(ret);
  }

  static Plan<Real> plan_many_dft_c2r(const fftw_buffer<Cx> &complex,
                                      const fftw_buffer<Real> &real,
                                      int n_arrays,
                                      int flags = DEFAULT_PLANNER_FLAG) {
    auto *real_p = const_cast<Real *>(real.data());
    auto *cx_p = reinterpret_cast<FFTW_Cx *>(const_cast<Cx *>(complex.data()));

    const int real_size = static_cast<int>(real.size());
    const int cx_size = static_cast<int>(complex.size());
    int rank = 1;
    int stride = 1;
    Plan<Real> ret{fftwf_plan_many_dft_c2r(rank, &real_size, n_arrays, cx_p,
                                           nullptr, stride, cx_size, real_p,
                                           nullptr, stride, real_size, flags)};

    return std::move(ret);
  }

  void execute_dft_r2c(std::span<const Real> real,
                       std::span<const Cx> complex) const {
    auto *real_p = const_cast<Real *>(real.data());
    auto *cx_p = reinterpret_cast<FFTW_Cx *>(const_cast<Cx *>(complex.data()));
    fftwf_execute_dft_r2c(plan, real_p, cx_p);
  }

  void execute_dft_c2r(std::span<const Cx> complex,
                       std::span<const Real> real) const {
    auto *real_p = const_cast<Real *>(real.data());
    auto *cx_p = reinterpret_cast<FFTW_Cx *>(const_cast<Cx *>(complex.data()));
    fftwf_execute_dft_c2r(plan, cx_p, real_p);
  }

  ~Plan() {
    if (plan != nullptr) {
      fftwf_destroy_plan(plan);
    }
  }
};

} // namespace fftw

// NOLINTEND(*-reinterpret-cast, *-const-cast, *-identifier-length)