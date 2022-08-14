# fftconv

Extremely fast 1D discrete convolutions of real vectors.

It's well know that convolution in the time domain is equivalent to multiplication in the frequency domain (circular convolution). With the Fast Fourier Transform, we can reduce the time complexity of a discrete convolution from `O(n^2)` to `O(n log(n))`, where `n` is the larger of the two array sizes. The **[overlap-add method](https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method)** is a fast convolution method commonly use in FIR filtering, where the discrete signal is often much longer than the FIR filter kernel.

* `fftconv::convolve_{fftw,pocketfft,pocketfft_hdr}` implement FFT convolution.
* `fftconv:oaconvolve_{fftw,pocketfft,pocketfft_hdr}` implements FFT convolution using the overlap-add method, much better when one sequence is much longer than the other (e.g. in FIR filtering).

In C++, All routines provide a C-array interface, a `std::vector` interface, and an `arma::vec` interface (if `<armadillo>` is included before `"fftconv.h"`).

Python bindings are provided through Cython.

## Build the test and benchmark

**C++**

`meson` and `ninja` are the build tools. The `fftw` implementation requires `fftw3`. The pocketfft source files are provided in the repo for convenience.

Benchmark and test dependencies:

* [fftw3](http://fftw.org/)
* [armadillo](http://arma.sourceforge.net/) (benchmarked against as a baseline)
* [google-benchmark](https://github.com/google/benchmark) used for benchmarking.
* [gperftools](https://github.com/gperftools/gperftools) used for profiling.

```
meson build
ninja -C build
```

**Python**

A Cython wrapper is provided. Dependencies:

* `Cython` for C++ bindings
* `numpy` (benchmarked against)
* `numba` (benchmarked against)
* `scipy` (benchmarked against)
* `matplotlib` (plot results)

```
python3 setup.py build_ext -i
python3 test.py # run the python test/benchmark
```

## Benchmark results

CPU: Intel i7 Comet Lake

**C++**.

The `fftconv_test` binary gives an easy benchmark that runs every test case 5000 times. The `fftconv_bench` uses `google-benchmark` and gives much more reliable measures. Use `./script/run_bench` to run the benchmark and generate figures.

Output from `fftconv_bench` (accurate bench) raw result saved in `./bench_result.json`. Plot generated from `plot_bench.py`:

![Comparison of the Overlap-Add method implemented with `fftw`, `pocketfft`, and `pocketfft_hdronly`](./bench_2022-08-14T01%3A47%3A26.svg)

Output from `fftconv_test` (simple bench)

```
% ./build/fftconv_test
=== test case (1664, 65) ===
All tests passed.
    (5000 runs) convolve_fftw took 82ms
    (5000 runs) oaconvolve_fftw took 36ms
    (5000 runs) convolve_pocketfft took 91ms
    (5000 runs) oaconvolve_pocketfft took 70ms
    (5000 runs) convolve_pocketfft_hdr took 111ms
    (5000 runs) oaconvolve_pocketfft_hdr took 105ms
    (5000 runs) convolve_armadillo took 108ms
=== test case (2816, 65) ===
All tests passed.
    (5000 runs) convolve_fftw took 111ms
    (5000 runs) oaconvolve_fftw took 60ms
    (5000 runs) convolve_pocketfft took 157ms
    (5000 runs) oaconvolve_pocketfft took 115ms
    (5000 runs) convolve_pocketfft_hdr took 187ms
    (5000 runs) oaconvolve_pocketfft_hdr took 166ms
    (5000 runs) convolve_armadillo took 174ms
=== test case (2304, 65) ===
All tests passed.
    (5000 runs) convolve_fftw took 536ms
    (5000 runs) oaconvolve_fftw took 52ms
    (5000 runs) convolve_pocketfft took 175ms
    (5000 runs) oaconvolve_pocketfft took 98ms
    (5000 runs) convolve_pocketfft_hdr took 206ms
    (5000 runs) oaconvolve_pocketfft_hdr took 143ms
    (5000 runs) convolve_armadillo took 147ms
=== test case (4352, 65) ===
All tests passed.
    (5000 runs) convolve_fftw took 335ms
    (5000 runs) oaconvolve_fftw took 86ms
    (5000 runs) convolve_pocketfft took 319ms
    (5000 runs) oaconvolve_pocketfft took 165ms
    (5000 runs) convolve_pocketfft_hdr took 369ms
    (5000 runs) oaconvolve_pocketfft_hdr took 235ms
    (5000 runs) convolve_armadillo took 276ms
```

**Python**.

```
% python3 test.py
=== test case (1664, 65) ===
Vectors are equal.
    (5000 runs) convolve_fftw took 66ms
    (5000 runs) convolve_pocketfft took 102ms
    (5000 runs) convolve_pocketfft_hdr took 132ms
    (5000 runs) oaconvolve_fftw took 46ms
    (5000 runs) oaconvolve_pocketfft took 81ms
    (5000 runs) oaconvolve_pocketfft_hdr took 113ms
    (5000 runs) np.convolve took 108ms
    (5000 runs) jit(np.convolve) took 1495ms
    (5000 runs) scipy.signal.convolve took 143ms
    (5000 runs) scipy.signal.fftconvolve took 356ms
    (5000 runs) scipy.signal.oaconvolve took 763ms
=== test case (2816, 65) ===
Vectors are equal.
    (5000 runs) convolve_fftw took 117ms
    (5000 runs) convolve_pocketfft took 179ms
    (5000 runs) convolve_pocketfft_hdr took 213ms
    (5000 runs) oaconvolve_fftw took 67ms
    (5000 runs) oaconvolve_pocketfft took 125ms
    (5000 runs) oaconvolve_pocketfft_hdr took 177ms
    (5000 runs) np.convolve took 173ms
    (5000 runs) jit(np.convolve) took 2448ms
    (5000 runs) scipy.signal.convolve took 214ms
    (5000 runs) scipy.signal.fftconvolve took 439ms
    (5000 runs) scipy.signal.oaconvolve took 831ms
=== test case (2304, 65) ===
Vectors are equal.
    (5000 runs) convolve_fftw took 537ms
    (5000 runs) convolve_pocketfft took 198ms
    (5000 runs) convolve_pocketfft_hdr took 243ms
    (5000 runs) oaconvolve_fftw took 63ms
    (5000 runs) oaconvolve_pocketfft took 111ms
    (5000 runs) oaconvolve_pocketfft_hdr took 155ms
    (5000 runs) np.convolve took 148ms
    (5000 runs) jit(np.convolve) took 2001ms
    (5000 runs) scipy.signal.convolve took 184ms
    (5000 runs) scipy.signal.fftconvolve took 410ms
    (5000 runs) scipy.signal.oaconvolve took 781ms
=== test case (4352, 65) ===
Vectors are equal.
    (5000 runs) convolve_fftw took 337ms
    (5000 runs) convolve_pocketfft took 329ms
    (5000 runs) convolve_pocketfft_hdr took 388ms
    (5000 runs) oaconvolve_fftw took 91ms
    (5000 runs) oaconvolve_pocketfft took 171ms
    (5000 runs) oaconvolve_pocketfft_hdr took 243ms
    (5000 runs) np.convolve took 262ms
    (5000 runs) jit(np.convolve) took 3920ms
    (5000 runs) scipy.signal.convolve took 330ms
    (5000 runs) scipy.signal.fftconvolve took 637ms
    (5000 runs) scipy.signal.oaconvolve took 885ms
```

The Python wrapper is almost as fast as the C++ code, as it has very little overhead.

## Implementation Details

TODO
