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
./build/fftconv_test
=== test case (1664, 65) ===
Vectors are equal.
Vectors are equal.
    (5000 runs) ffconv::convolve1d_ref took 223ms
    (5000 runs) ffconv::convolve1d took 49ms
    (5000 runs) arma::conv took 110ms
=== test case (2816, 65) ===
Vectors are equal.
Vectors are equal.
    (5000 runs) ffconv::convolve1d_ref took 246ms
    (5000 runs) ffconv::convolve1d took 86ms
    (5000 runs) arma::conv took 183ms
=== test case (2000, 2000) ===
Vectors are equal.
Vectors are equal.
    (5000 runs) ffconv::convolve1d_ref took 886ms
    (5000 runs) ffconv::convolve1d took 667ms
    (5000 runs) arma::conv took 17440ms
```

**Python**. 1D convolution with `fftconv` is more than 2x faster than Numpy, and the speed gains are more significant the large the input arrays.

```
=== test case (1664, 65) ===
Vectors are equal.
    (5000 runs) fftconv took 56ms
    (5000 runs) np.conv took 149ms
=== test case (2816, 65) ===
Vectors are equal.
    (5000 runs) fftconv took 93ms
    (5000 runs) np.conv took 247ms
=== test case (2000, 2000) ===
Vectors are equal.
    (5000 runs) fftconv took 658ms
    (5000 runs) np.conv took 8285ms
```

The Python wrapper is almost as fast as the C++ code, as it has very little overhead.

## Implementation Details

TODO
