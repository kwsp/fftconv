#include <cassert>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>

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
  std::cout << "    (" << n_runs << " runs) " << name << " took " << elapsed.count() << "ms\n";
}