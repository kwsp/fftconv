# fftconv

Extremely fast 1D discrete convolutions of real vectors in a header-only library. `fftconv` uses [FFTW 3](http://www.fftw.org/) under the hood.

It's well know that convolution in the time domain is equivalent to multiplication in the frequency domain. With the Fast Fourier Transform, we can reduce the time complexity of a discrete convolution from `O(n^2)` to `O(n log(n))`, where `n` is the larger of the two array sizes.


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
=== test case (1664, 65) ===
naive vs fft_ref Vectors are equal.
naive vs fft Vectors are equal.
naive vs fftfilt Vectors are equal.
    (5000 runs) fftconv::convolve1d_ref took 207ms
    (5000 runs) fftconv::convolve1d took 71ms
    (5000 runs) fftconv::fftfilt took 57ms
    (5000 runs) arma::conv took 110ms
=== test case (2816, 65) ===
naive vs fft_ref Vectors are equal.
naive vs fft Vectors are equal.
naive vs fftfilt Vectors are equal.
    (5000 runs) fftconv::convolve1d_ref took 247ms
    (5000 runs) fftconv::convolve1d took 88ms
    (5000 runs) fftconv::fftfilt took 96ms
    (5000 runs) arma::conv took 182ms
=== test case (2304, 65) ===
naive vs fft_ref Vectors are equal.
naive vs fft Vectors are equal.
naive vs fftfilt Vectors are equal.
    (5000 runs) fftconv::convolve1d_ref took 492ms
    (5000 runs) fftconv::convolve1d took 265ms
    (5000 runs) fftconv::fftfilt took 79ms
    (5000 runs) arma::conv took 152ms
=== test case (4352, 65) ===
naive vs fft_ref Vectors are equal.
naive vs fft Vectors are equal.
naive vs fftfilt Vectors are equal.
    (5000 runs) fftconv::convolve1d_ref took 483ms
    (5000 runs) fftconv::convolve1d took 311ms
    (5000 runs) fftconv::fftfilt took 148ms
    (5000 runs) arma::conv took 281ms
```

**Python**. 1D convolution with `fftconv` is more than 2x faster than Numpy, and the speed gains are more significant the large the input arrays.

```
% python3 test.py
=== test case (1664, 65) ===
Vectors are equal.
    (5000 runs) fftconv took 70ms
    (5000 runs) fftfilt took 64ms
    (5000 runs) np.conv took 141ms
    (5000 runs) scipy.signal.convolve took 158ms
    (5000 runs) scipy.signal.fftconvolve took 194ms
=== test case (2816, 65) ===
Vectors are equal.
    (5000 runs) fftconv took 96ms
    (5000 runs) fftfilt took 102ms
    (5000 runs) np.conv took 232ms
    (5000 runs) scipy.signal.convolve took 250ms
    (5000 runs) scipy.signal.fftconvolve took 252ms
=== test case (2304, 65) ===
Vectors are equal.
    (5000 runs) fftconv took 270ms
    (5000 runs) fftfilt took 85ms
    (5000 runs) np.conv took 192ms
    (5000 runs) scipy.signal.convolve took 210ms
    (5000 runs) scipy.signal.fftconvolve took 235ms
=== test case (4352, 65) ===
Vectors are equal.
    (5000 runs) fftconv took 319ms
    (5000 runs) fftfilt took 157ms
    (5000 runs) np.conv took 375ms
    (5000 runs) scipy.signal.convolve took 378ms
    (5000 runs) scipy.signal.fftconvolve took 356ms
```

The Python wrapper is almost as fast as the C++ code, as it has very little overhead.

## Implementation Details

TODO
