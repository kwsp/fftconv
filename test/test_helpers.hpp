#include <armadillo>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <type_traits>
#include <vector>

namespace test_helpers {

using std::cout;
using std::vector;
constexpr double err = 1e-6;

// Run the `callable` `n_runs` times and print the time.
inline void _timeit(const std::string &name, std::function<void()> callable,
                    int n_runs = 5000) {
  using namespace std::chrono;
  cout << "    (" << n_runs << " runs) " << name;
  auto start = high_resolution_clock::now();
  for (int i = 0; i < n_runs; i++)
    callable();
  auto elapsed =
      duration_cast<milliseconds>(high_resolution_clock::now() - start);
  cout << " took " << elapsed.count() << "ms\n";
}

// Compare two vectors
template <class T1, class T2>
bool cmp_vec(T1 *a, const size_t l_a, T2 *b, const size_t l_b) {
  assert(l_a == l_b);
  for (auto i = 0; i < l_a; ++i)
    if (std::abs(a[i] - b[i]) > err) {
      printf("Vectors are different: v1[%d]=%f, v2[%d]=%f\n", i, a[i], i, b[i]);
      return false;
    }
  return true;
}
template <class T> bool cmp_vec(const vector<T> &a, const vector<T> &b) {
  return cmp_vec(a.data(), a.size(), b.data(), b.size());
}
template <class T> bool cmp_vec(const arma::Col<T> &a, const arma::Col<T> &b) {
  return cmp_vec(a.data(), a.size(), b.data(), b.size());
}

template <class T> void make_same_length(vector<T> &vec, size_t len) {
  assert(vec.size() >= len);
  auto offset = (vec.size() - len) / 2;
  vec.erase(vec.begin(), vec.begin() + offset);
  vec.erase(vec.end() - offset, vec.end());
}

template <class T> void print_vec(T *arr, size_t sz) {
  for (int i = 0; i < sz; ++i)
    cout << std::setw(4) << arr[i] << ", ";
  cout << "\n";
}
template <class T> void print_vec(vector<T> vec) {
  print_vec(vec.data(), vec.size());
}
template <class T> void print_vec(arma::Col<T> vec) {
  print_vec(vec.memptr(), vec.size());
}

} // namespace test_helpers
