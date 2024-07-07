#include <complex>
#include <fftw3.h>
#include <iostream>
#include <vector>

template <class T> inline void print(const T *a, const size_t sz) {
  for (size_t i = 0; i < sz; ++i)
    std::cout << a[i] << ", ";
  std::cout << "\n";
}

template <class T> void print(const std::complex<T> *a, const size_t sz) {
  for (size_t i = 0; i < sz; ++i)
    std::cout << "(" << a[i].real() << ", " << a[i].imag() << "), ";
  std::cout << "\n";
}

template <class T> void print(const std::vector<T> &a) {
  print(a.data(), a.size());
}

inline void print(const fftw_complex *a, const size_t sz) {
  print(reinterpret_cast<const std::complex<double> *>(a), sz);
}
