#include <armadillo>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fmt/core.h>
#include <span>

constexpr double Tol = 1e-6;

// Run the `callable` `n_runs` times and print the time.
inline void timeit(const std::string &name,
                   const std::function<void()> &callable, int n_runs) {
  using namespace std::chrono; // NOLINT
  const auto start = high_resolution_clock::now();
  for (int i = 0; i < n_runs; i++) {
    callable();
  }
  const auto elapsed =
      duration_cast<milliseconds>(high_resolution_clock::now() - start);
  fmt::println("    ({} runs) {} took {}ms", n_runs, name, elapsed.count());
}

// Compare two vectors
template <class T1, class T2>
auto cmp_vec(const std::span<const T1> vec1, const std::span<const T2> vec2)
    -> bool {
  assert(vec1.size() == vec2.size());
  for (auto i = 0; i < vec1.size(); ++i) {
    if (std::abs(vec1[i] - vec2[i]) > Tol) {
      printf("Vectors are different: v1[%d]=%f, v2[%d]=%f\n", i, vec1[i], i,
             vec2[i]);
      return false;
    }
  }
  return true;
}
