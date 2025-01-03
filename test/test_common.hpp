#pragma once

#include <gtest/gtest.h>

// https://clangd.llvm.org/guides/include-cleaner
// Using symbol through template
#include <fftconv/fftconv.hpp> // IWYU pragma: keep

constexpr double DoubleTol{1e-9};
constexpr float FloatTol{1e-5};

template <fftconv::Floating T>
void ExpectVectorsNear(const std::span<const T> expected,
                       const std::span<const T> actual, const T tolerance) {
  ASSERT_EQ(expected.size(), actual.size()) << "Vectors are of unequal length";

  for (size_t i = 0; i < expected.size() && i < actual.size(); ++i) {
    ASSERT_NEAR(expected[i], actual[i], tolerance)
        << "Vectors differ at index " << i;
  }
}

template <typename T>
inline void ExpectArraysNear(const T *arr1, const T *arr2, size_t size,
                             T tolerance) {
  for (size_t i = 0; i < size; ++i) {
    if (std::abs(arr1[i] - arr2[i]) > tolerance) {
      GTEST_FAIL() << "Arrays differ at index " << i << ": expected " << arr1[i]
                   << " but got " << arr2[i] << ", tolerance = " << tolerance;
    }
  }
}