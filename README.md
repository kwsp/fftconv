# fftconv

Extremely fast 1D discrete convolutions of real vectors in a header-only library. `fftconv` uses [FFTW 3](http://www.fftw.org/) under the hood.

It's well know that convolution in the time domain is equivalent to multiplication in the frequency domain. With the Fast Fourier Transform, we can reduce the time complexity of a discrete convolution from `O(n^2)` to `O(n log(n))`, where `n` is the larger of the two array sizes.

To test, run `gen_input.py` to generate 3 test cases. Compile `test.c` with `make` (make sure libfftw3 is installed in a location visible to the compiler). Run `make test` to test for correctness and performance.

A Cython wrapper is provided so the routine can be called in Python.
## Build

**C++**

`fftconv` is header only, so just include the header with `#include "fftconv.h"`. However, you need to have `fftw3` installed, with the `fftw3.h` header visible to the compiler and `libfftw3` visible to the linker.

To run the test and benchmark, you need to install the [Armadillo library](http://arma.sourceforge.net/) (benchmarked against), and then run `make` to build and `make test` to run the test and benchmark.

**Python**

Install Cython, then run `python3 setup.py build_ext -i` to build the extension. Run the `test.py` script to test and benchmark.


## Benchmarks

**C++**. 1D convolution in `fftconv` is more than 2x faster than the Armadillo implementation.

```
=== test_case_1.txt (1664, 65)  ===
Vectors are equal.
Vectors are equal.
    (5000 runs) convolve_naive took 6821ms
    (5000 runs) ffconv::convolve1d took 45ms
    (5000 runs) arma::conv took 109ms
=== test_case_2.txt (2816, 65)  ===
Vectors are equal.
Vectors are equal.
    (5000 runs) convolve_naive took 19177ms
    (5000 runs) ffconv::convolve1d took 82ms
    (5000 runs) arma::conv took 183ms
=== test_case_3.txt (10, 5)  ===
Vectors are equal.
Vectors are equal.
    (5000 runs) convolve_naive took 0ms
    (5000 runs) ffconv::convolve1d took 0ms
    (5000 runs) arma::conv took 0ms
```

**Python**. 1D convolution with `fftconv` is more than 2x faster than Numpy, and the speed gains are more significant the large the input arrays.

```
=== test case (1664, 65) ===
Vectors are equal.
    (5000 runs) fftconv: 67.57 ms
    (5000 runs) np.conv: 151.07 ms
=== test case (2816, 65) ===
Vectors are equal.
    (5000 runs) fftconv: 92.30 ms
    (5000 runs) np.conv: 244.77 ms
=== test case (2000, 2000) ===
Vectors are equal.
    (5000 runs) fftconv: 669.90 ms
    (5000 runs) np.conv: 8267.93 ms
```

The Python wrapper is almost as fast as the C++ code, as it has very little overhead.

## Implementation Details

TODO
