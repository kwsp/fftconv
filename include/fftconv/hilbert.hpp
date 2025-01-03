#pragma once

#include <cassert>
#include <fftconv/fftconv.hpp>
#include <fftconv/fftw.hpp>
#include <span>

// NOLINTBEGIN(*-pointer-arithmetic, *-magic-numbers)

namespace fftconv {

/**
@brief Compute the analytic signal, using the Hilbert transform.
Uses FFTW's r2c transform
*/
template <fftw::Floating T>
void hilbert(const std::span<const T> x, const std::span<T> env) {
  const auto n = x.size();
  assert(n > 0);
  assert(x.size() == env.size());

  fftw::EngineR2C1D<T> &engine = fftw::EngineR2C1D<T>::get(n);
  fftw::R2CBuffer<T> &buf = engine.buf;

  if (isSIMDAligned<64>(x.data())) {
    // Avoid a copy
    engine.forward(x.data(), buf.out);
  } else {
    // Copy input to real buffer
    for (int i = 0; i < n; ++i) {
      buf.in[i] = x[i];
    }

    // Execute r2c fft
    engine.forward();
  }

  //  Multiply by 1j
  const auto cx_size = n / 2 + 1;
  for (auto i = 0; i < cx_size; ++i) {
    const auto re = buf.out[i][0];
    const auto im = buf.out[i][1];
    buf.out[i][0] = im;
    buf.out[i][1] = -re;
  }

  // Execute c2r fft on modified spectrum
  engine.backward();

  // Take the abs of the analytic signal
  const T fct = static_cast<T>(1. / n);

  for (auto i = 0; i < n; ++i) {
    const auto real = x[i];
    const auto imag = buf.in[i] * fct;
    env[i] = std::sqrt(real * real + imag * imag);
  }

  // fftw::scale_and_magnitude<T>(buf.in, env.data(), n, fct);
}

namespace ipp {

#if defined(HAS_IPP)

#include <format>
#include <ipp.h>

void handleIppStatus(IppStatus status) {
  if (status != ippStsNoErr) {
    throw std::runtime_error(std::format("Ipp error: {}", status));
  }
}

namespace detail {

// In memory cache with key type K and value type V
// additionally accepts a mutex to guard the V constructor
template <class Key, class Val> auto get_cached(Key key) -> Val * {
  thread_local std::unordered_map<Key, std::unique_ptr<Val>> cache;

  auto &val = cache[key];
  if (val == nullptr) {
    val = std::make_unique<Val>(key);
  }
  return val.get();
}

struct HilbertIppBuf32fc {
  Ipp32fc *y;
  IppsHilbertSpec *pSpec;
  Ipp8u *pBuffer;

  HilbertIppBuf32fc() = delete;
  HilbertIppBuf32fc(const HilbertIppBuf32fc &) = delete;
  HilbertIppBuf32fc(HilbertIppBuf32fc &&) = delete;
  HilbertIppBuf32fc &operator=(const HilbertIppBuf32fc &) = delete;
  HilbertIppBuf32fc &operator=(HilbertIppBuf32fc &&) = delete;
  explicit HilbertIppBuf32fc(size_t n)
      : y(static_cast<Ipp32fc *>(
            ippMalloc(static_cast<int>(n * sizeof(Ipp32fc))))) {
    IppStatus status;
    int sizeSpec, sizeBuf;

    status = ippsHilbertGetSize_32f32fc(n, ippAlgHintNone, &sizeSpec, &sizeBuf);
    pSpec = static_cast<IppsHilbertSpec *>(ippMalloc(sizeSpec));
    pBuffer = static_cast<Ipp8u *>(ippMalloc(sizeBuf));
    status = ippsHilbertInit_32f32fc(n, ippAlgHintNone, pSpec, pBuffer);

    // TODO: handle status
  }
  ~HilbertIppBuf32fc() {
    ippFree(pSpec);
    ippFree(pBuffer);
    ippFree(y);
  }
};

struct HilbertIppBuf64fc {
  Ipp64fc *y;
  IppsHilbertSpec *pSpec;
  Ipp8u *pBuffer;

  HilbertIppBuf64fc() = delete;
  HilbertIppBuf64fc(const HilbertIppBuf64fc &) = delete;
  HilbertIppBuf64fc(HilbertIppBuf64fc &&) = default;
  HilbertIppBuf64fc &operator=(const HilbertIppBuf64fc &) = delete;
  HilbertIppBuf64fc &operator=(HilbertIppBuf64fc &&) = default;
  explicit HilbertIppBuf64fc(size_t n)
      : y(static_cast<Ipp64fc *>(
            ippMalloc(static_cast<int>(n * sizeof(Ipp64fc))))) {
    IppStatus status;
    int sizeSpec, sizeBuf;

    status = ippsHilbertGetSize_64f64fc(n, ippAlgHintNone, &sizeSpec, &sizeBuf);
    pSpec = static_cast<IppsHilbertSpec *>(ippMalloc(sizeSpec));
    pBuffer = static_cast<Ipp8u *>(ippMalloc(sizeBuf));
    status = ippsHilbertInit_64f64fc(n, ippAlgHintNone, pSpec, pBuffer);

    // TODO: handle status
  }
  ~HilbertIppBuf64fc() {
    ippFree(pSpec);
    ippFree(pBuffer);
    ippFree(y);
  }
};

} // namespace detail

template <fftw::Floating T>
void hilbert(const std::span<const T> x, const std::span<T> env) {
  const size_t n = x.size();
  IppStatus status;

  if constexpr (std::is_same_v<T, float>) {
    auto &buf = *fftw::get_cached<size_t, detail::HilbertIppBuf32fc>(n);

    status = ippsHilbert_32f32fc(x.data(), buf.y, buf.pSpec, buf.pBuffer);
    ippsMagnitude_32fc(buf.y, env.data(), n);

  } else if constexpr (std::is_same_v<T, double>) {
    auto &buf = *fftw::get_cached<size_t, detail::HilbertIppBuf64fc>(n);

    status = ippsHilbert_64f64fc(x.data(), buf.y, buf.pSpec, buf.pBuffer);
    ippsMagnitude_64fc(buf.y, env.data(), n);

  } else {
    static_assert(false, "Not supported.");
  }
}

#endif

} // namespace ipp

} // namespace fftconv

// NOLINTEND(*-pointer-arithmetic, *-magic-numbers)
