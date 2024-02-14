#include <armadillo>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <vector>

namespace test_helpers {

using std::cout;
using std::vector;
constexpr double err = 1e-6;

// Run the `callable` `n_runs` times and print the time.
inline void _timeit(const std::string &name,
                    const std::function<void()> &callable, int n_runs = 5000) {
  using namespace std::chrono; // NOLINT
  cout << "    (" << n_runs << " runs) " << name;
  auto start = high_resolution_clock::now();
  for (int i = 0; i < n_runs; i++) {
    callable();
  }
  auto elapsed =
      duration_cast<milliseconds>(high_resolution_clock::now() - start);
  cout << " took " << elapsed.count() << "ms\n";
}

// Compare two vectors
template <class T1, class T2>
auto cmp_vec(T1 *vec1, const size_t l_a, T2 *vec2, const size_t l_b) -> bool {
  assert(l_a == l_b);
  for (auto i = 0; i < l_a; ++i) {
    if (std::abs(vec1[i] - vec2[i]) > err) {
      printf("Vectors are different: v1[%d]=%f, v2[%d]=%f\n", i, vec1[i], i,
             vec2[i]);
      return false;
    }
  }
  return true;
}
template <class T>
auto cmp_vec(const vector<T> &vec1, const vector<T> &vec2) -> bool {
  return cmp_vec(vec1.data(), vec1.size(), vec2.data(), vec2.size());
}
template <class T>
auto cmp_vec(const arma::Col<T> &vec1, const arma::Col<T> &vec2) -> bool {
  return cmp_vec(vec1.data(), vec1.size(), vec2.data(), vec2.size());
}

template <class T> void make_same_length(vector<T> &vec, size_t len) {
  assert(vec.size() >= len);
  auto offset = (vec.size() - len) / 2;
  vec.erase(vec.begin(), vec.begin() + offset);
  vec.erase(vec.end() - offset, vec.end());
}

template <class T> void print_vec(T *arr, size_t size) {
  for (int i = 0; i < size; ++i) {
    cout << std::setw(4) << arr[i] << ", ";
  }
  cout << "\n";
}
template <class T> void print_vec(vector<T> vec) {
  print_vec(vec.data(), vec.size());
}
template <class T> void print_vec(arma::Col<T> vec) {
  print_vec(vec.memptr(), vec.size());
}

} // namespace test_helpers
