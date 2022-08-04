# fftconv

Extremely fast 1D discrete convolutions of real vectors in a header-only library. `fftconv` uses [FFTW 3](http://www.fftw.org/) under the hood.

It's well know that convolution in the time domain is equivalent to multiplication in the frequency domain (circular convolution). With the Fast Fourier Transform, we can reduce the time complexity of a discrete convolution from `O(n^2)` to `O(n log(n))`, where `n` is the larger of the two array sizes. The **[overlap-add method](https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method)** is a fast convolution method commonly use in FIR filtering, where the discrete signal is often much longer than the FIR filter kernel.

* `fftconv::fftconv` implements FFT convolution with memory optimizations.
* `fftconv:fftconv_oa` implements FFT convolution using the overlap-add method, much faster for FIR filtering.

In C++, All routines provide a C-array interface, a `std::vector` interface, and an `arma::vec` interface (if `<armadillo>` is included before `"fftconv.h"`).

Python bindings are provided through Cython.

## Build

**C++**

`fftconv` is header only, so just include the header with `#include "fftconv.h"`. However, you need to have `fftw3` installed, with the `fftw3.h` header visible to the compiler and `libfftw3` visible to the linker.

To run the test and benchmark, you need to install the [Armadillo library](http://arma.sourceforge.net/) (benchmarked against).

```
make       # build the test/benchmark
make test  # run the test/benchmark
```

**Python**

A Cython wrapper is provided. Similar to C++, make sure `fftw3` is installed and visible to the compiler on your system. Install Cython, then run 

```
python3 setup.py build_ext -i   # build extension
python3 test.py                 # run the test/benchmark
```

## Benchmark results

CPU: Apple M1

**C++**. 1D convolution in `fftconv` is more than 2x faster than the Armadillo implementation.

```
% make test
./build/fftconv_test
=== test case (8, 4) ===
gt vs fftconv Vectors are equal.
gt vs fftconv_oa Vectors are equal.
    (5000 runs) fftconv::fftconv took 0ms
    (5000 runs) fftconv::fftconv_oa took 0ms
    (5000 runs) arma_conv took 0ms
=== test case (1664, 65) ===
gt vs fftconv Vectors are equal.
gt vs fftconv_oa Vectors are equal.
    (5000 runs) fftconv::fftconv took 55ms
    (5000 runs) fftconv::fftconv_oa took 36ms
    (5000 runs) arma_conv took 109ms
=== test case (2816, 65) ===
gt vs fftconv Vectors are equal.
gt vs fftconv_oa Vectors are equal.
    (5000 runs) fftconv::fftconv took 103ms
    (5000 runs) fftconv::fftconv_oa took 62ms
    (5000 runs) arma_conv took 183ms
=== test case (2304, 65) ===
gt vs fftconv Vectors are equal.
gt vs fftconv_oa Vectors are equal.
    (5000 runs) fftconv::fftconv took 264ms
    (5000 runs) fftconv::fftconv_oa took 52ms
    (5000 runs) arma_conv took 150ms
=== test case (4352, 65) ===
gt vs fftconv Vectors are equal.
gt vs fftconv_oa Vectors are equal.
    (5000 runs) fftconv::fftconv took 315ms
    (5000 runs) fftconv::fftconv_oa took 86ms
    (5000 runs) arma_conv took 284ms
```

**Python**. 1D convolution with `fftconv` is more than 2x faster than Numpy, and the speed gains are more significant the large the input arrays.

```
% python3 test.py
=== test case (1664, 65) ===
Vectors are equal.
    (5000 runs) fftconv took 65ms
    (5000 runs) fftconv_oa took 43ms
    (5000 runs) np.convolve took 141ms
    (5000 runs) scipy.signal.convolve took 159ms
    (5000 runs) scipy.signal.fftconvolve took 194ms
=== test case (2816, 65) ===
Vectors are equal.
    (5000 runs) fftconv took 95ms
    (5000 runs) fftconv_oa took 67ms
    (5000 runs) np.convolve took 233ms
    (5000 runs) scipy.signal.convolve took 251ms
    (5000 runs) scipy.signal.fftconvolve took 254ms
=== test case (2304, 65) ===
Vectors are equal.
    (5000 runs) fftconv took 278ms
    (5000 runs) fftconv_oa took 59ms
    (5000 runs) np.convolve took 195ms
    (5000 runs) scipy.signal.convolve took 213ms
    (5000 runs) scipy.signal.fftconvolve took 240ms
=== test case (4352, 65) ===
Vectors are equal.
    (5000 runs) fftconv took 329ms
    (5000 runs) fftconv_oa took 94ms
    (5000 runs) np.convolve took 372ms
    (5000 runs) scipy.signal.convolve took 375ms
    (5000 runs) scipy.signal.fftconvolve took 358ms
```

The Python wrapper is almost as fast as the C++ code, as it has very little overhead.

## Implementation Details

TODO
