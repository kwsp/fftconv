#pragma once

#include <gtest/gtest.h>

// https://clangd.llvm.org/guides/include-cleaner
// Using symbol through template
#include <fftconv/fftconv.hpp> // IWYU pragma: keep

template <fftconv::Floating T> constexpr T getTol() {
  constexpr double DoubleTol{1e-9};
  constexpr float FloatTol{1e-5};

  if constexpr (std::is_same_v<T, float>) {
    return FloatTol;
  } else {
    return DoubleTol;
  }
}

template <fftconv::Floating T>
void ExpectVectorsNear(const std::span<const T> actual,
                       const std::span<const T> expected, const T tolerance) {
  ASSERT_EQ(expected.size(), actual.size()) << "Vectors are of unequal length";

  for (size_t i = 0; i < expected.size() && i < actual.size(); ++i) {
    ASSERT_NEAR(expected[i], actual[i], tolerance)
        << "Vectors differ at index " << i;
  }
}
