# fftconv

Extremely fast 1D discrete convolutions of real vectors in a header-only library. `fftconv` uses [FFTW 3](http://www.fftw.org/) under the hood.

It's well know that convolution in the time domain is equivalent to multiplication in the frequency domain (circular convolution). With the Fast Fourier Transform, we can reduce the time complexity of a discrete convolution from `O(n^2)` to `O(n log(n))`, where `n` is the larger of the two array sizes. The **[overlap-add method](https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method)** is a fast convolution method commonly use in FIR filtering, where the discrete signal is often much longer than the FIR filter kernel.

* `fftconv::fftconv` implements FFT convolution with memory optimizations.
* `fftconv:fftconv_oa` implements FFT convolution using the overlap-add method, much faster for FIR filtering.

In C++, All routines provide a C-array interface, a `std::vector` interface, and an `arma::vec` interface (if `<armadillo>` is included before `"fftconv.h"`).

Python bindings are provided through Cython.

## Build

**C++**

You must have `fftw3` installed, with the `fftw3.h` header visible to the compiler and `libfftw3` visible to the linker.

To run the test and benchmark, you also need to install the [Armadillo library](http://arma.sourceforge.net/) (benchmarked against).

```
make       # build the test/benchmark
make test  # run the test/benchmark
```

**Python**

A Cython wrapper is provided. Similar to C++, make sure `fftw3` is installed and visible to the compiler on your system. Install Cython, numpy and scipy (benchmarked against), then run 

```
python3 setup.py build_ext -i   # or `make python` to build extension
python3 test.py                 # run the test/benchmark
```

## Benchmark results

CPU: Intel i7 Comet Lake

**C++**.

```
% make test
./build/fftconv_test
=== test case (1664, 65) ===
gt vs fftconv Vectors are equal.
gt vs fftconv_oa Vectors are equal.
    (5000 runs) fftconv::fftconv took 85ms
    (5000 runs) fftconv::fftconv_oa took 36ms
    (5000 runs) arma_conv took 106ms
=== test case (2816, 65) ===
gt vs fftconv Vectors are equal.
gt vs fftconv_oa Vectors are equal.
    (5000 runs) fftconv::fftconv took 112ms
    (5000 runs) fftconv::fftconv_oa took 60ms
    (5000 runs) arma_conv took 172ms
=== test case (2304, 65) ===
gt vs fftconv Vectors are equal.
gt vs fftconv_oa Vectors are equal.
    (5000 runs) fftconv::fftconv took 523ms
    (5000 runs) fftconv::fftconv_oa took 51ms
    (5000 runs) arma_conv took 144ms
=== test case (4352, 65) ===
gt vs fftconv Vectors are equal.
gt vs fftconv_oa Vectors are equal.
    (5000 runs) fftconv::fftconv took 330ms
    (5000 runs) fftconv::fftconv_oa took 86ms
    (5000 runs) arma_conv took 267ms
```

**Python**.

```
% python3 test.py
=== test case (1664, 65) ===
Vectors are equal.
    (5000 runs) fftconv took 72ms
    (5000 runs) fftconv_oa took 49ms
    (5000 runs) np.convolve took 105ms
    (5000 runs) scipy.signal.convolve took 139ms
    (5000 runs) scipy.signal.fftconvolve took 351ms
    (5000 runs) scipy.signal.oaconvolve took 718ms
=== test case (2816, 65) ===
Vectors are equal.
    (5000 runs) fftconv took 126ms
    (5000 runs) fftconv_oa took 72ms
    (5000 runs) np.convolve took 162ms
    (5000 runs) scipy.signal.convolve took 197ms
    (5000 runs) scipy.signal.fftconvolve took 429ms
    (5000 runs) scipy.signal.oaconvolve took 762ms
=== test case (2304, 65) ===
Vectors are equal.
    (5000 runs) fftconv took 553ms
    (5000 runs) fftconv_oa took 66ms
    (5000 runs) np.convolve took 138ms
    (5000 runs) scipy.signal.convolve took 174ms
    (5000 runs) scipy.signal.fftconvolve took 407ms
    (5000 runs) scipy.signal.oaconvolve took 757ms
=== test case (4352, 65) ===
Vectors are equal.
    (5000 runs) fftconv took 355ms
    (5000 runs) fftconv_oa took 100ms
    (5000 runs) np.convolve took 251ms
    (5000 runs) scipy.signal.convolve took 285ms
    (5000 runs) scipy.signal.fftconvolve took 602ms
    (5000 runs) scipy.signal.oaconvolve took 833ms
```

The Python wrapper is almost as fast as the C++ code, as it has very little overhead.

## Implementation Details

TODO
