# fftconv

Extremely fast 1D discrete convolutions of real vectors in a header-only library. `fftconv` uses [FFTW 3](http://www.fftw.org/) under the hood.

It's well know that convolution in the time domain is equivalent to multiplication in the frequency domain. With the Fast Fourier Transform, we can reduce the time complexity of a discrete convolution from `O(n^2)` to `O(n log(n))`, where `n` is the larger of the two array sizes.

To test, run `gen_input.py` to generate 3 test cases. Compile `test.c` with `make` (make sure libfftw3 is installed in a location visible to the compiler). Run `make test` to test for correctness and performance.
