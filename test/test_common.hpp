#pragma once

#include <gtest/gtest.h>

// https://clangd.llvm.org/guides/include-cleaner
// Using symbol through template
#include "fftconv.hpp" // IWYU pragma: keep

constexpr double DoubleTol{1e-9};
constexpr float FloatTol{1e-5};

template <fftconv::FloatOrDouble Real>
void ExpectVectorsNear(const std::span<const Real> expected,
                       const std::span<const Real> actual,
                       const Real tolerance) {
  EXPECT_EQ(expected.size(), actual.size()) << "Vectors are of unequal length";

  for (size_t i = 0; i < expected.size() && i < actual.size(); ++i) {
    EXPECT_NEAR(expected[i], actual[i], tolerance)
        << "Vectors differ at index " << i;
  }
}
