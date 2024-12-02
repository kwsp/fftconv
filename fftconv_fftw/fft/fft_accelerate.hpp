#pragma once

#ifdef __APPLE__

#include <Accelerate/Accelerate.h>
#include <concepts>
#include <vecLib/vecLib.h>

// vDSP_create_fftsetup(vDSP_Length Log2n, FFTRadix Radix)

class FFTInterface {
public:
};

template <typename T>
concept Floating = std::is_floating_point_v<T>;

template <Floating T> class FFTEngine {
public:
  FFTEngine(int realSize) {
    splitComplex.realp = new T[realSize / 2];
    splitComplex.imagp = new T[realSize / 2];

    vDSP_create_fftsetup(log2f(realSize), kFFTRadix2);
  }
  ~FFTEngine() {
    delete splitComplex.realp;
    delete splitComplex.imagp;
  }

private:
  FFTSetup setup;
  DSPSplitComplex splitComplex;
};

#endif