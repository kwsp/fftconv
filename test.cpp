#include <cassert>
#include <iomanip>
#include <iostream>
#include <vector>

#include "pocketfft_hdronly.h"

using std::cout;
using std::vector;

template <class T> void print_vec(T *arr, size_t sz) {
  for (int i = 0; i < sz; ++i)
    cout << std::setw(4) << arr[i] << ", ";
  cout << "\n";
}
template <class T> void print_vec(vector<T> vec) {
  print_vec(vec.data(), vec.size());
}

vector<double> get_vec(size_t size) {
  vector<double> res(size);
  for (size_t i = 0; i < size; i++) {
    res[i] = (double)(std::rand() % 10);
  }
  return res;
}

int main() {
  std::vector<double> a = {0., 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> b(a.size());
  std::vector<std::complex<double>> A(a.size());

  std::cout << "a    : "; print_vec(a);
  pocketfft::r2c({a.size()}, {1}, {1}, 0, true, a.data(), A.data(), 1.);
  std::cout << "A    : "; print_vec(A);
  pocketfft::c2r({A.size()}, {1}, {1}, 0, true, A.data(), a.data(), 1./a.size());
  std::cout << "b    : "; print_vec(a);
}
