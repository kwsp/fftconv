# fftconv

Extremely fast CPU 1D discrete convolutions. [Faster than Intel IPP and Apple Accelerate on their respective platforms](https://github.com/kwsp/microbenchmarks/tree/main/src/conv1d)

*Kernel size = 245*

<p align="center">
  <img src="https://github.com/kwsp/microbenchmarks/blob/main/src/conv1d/plots/Conv1d%20Throughput%20Bar%20(k%3D245)%2013th%20Gen%20Intel(R)%20Core(TM)%20i9-13900K.svg" width="45%">
  <img src="https://github.com/kwsp/microbenchmarks/blob/main/src/conv1d/plots/Conv1d%20Throughput%20Line%20(k%3D245)%2013th%20Gen%20Intel(R)%20Core(TM)%20i9-13900K.svg" width="45%">
</p>

<p align="center">
  <img src="https://github.com/kwsp/microbenchmarks/blob/main/src/conv1d/plots/Conv1d%20Throughput%20Bar%20(k%3D245)%20Apple%20M1.svg" width="45%">
  <img src="https://github.com/kwsp/microbenchmarks/blob/main/src/conv1d/plots/Conv1d%20Throughput%20Line%20(k%3D245)%20Apple%20M1.svg" width="45%">
</p>

It's well know that convolution in the time domain is equivalent to multiplication in the frequency domain (circular convolution). With the Fast Fourier Transform, we can reduce the time complexity of a discrete convolution from `O(n^2)` to `O(n log(n))`, where `n` is the larger of the two array sizes. The **[overlap-add method](https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method)** is a fast convolution method commonly use in FIR filtering, where the discrete signal is often much longer than the FIR filter kernel.

## Usage

Check [this repo](https://github.com/kwsp/microbenchmarks) to see how to use fftconv as a custom port through VCPKG.

- `fftconv::convolve_fftw` implements FFT convolution.
- `fftconv::oaconvolve_fftw` implements FFT convolution using the overlap-add method, much faster when one sequence is much longer than the other (e.g. in FIR filtering).

All convolution functions support `float` and `double` and use a C++20 `std::span` interface.

```C++
template <FloatOrDouble Real>
void oaconvolve_fftw(const std::span<const Real> arr,
                     const std::span<const Real> kernel, std::span<Real> res);
```

Python bindings are provided through Cython.

## Build the test and benchmark

**This benchmark is out of date**. Check [this repo](https://github.com/kwsp/microbenchmarks) for the up-to-date benchmarks.

The only dependency of `fftconv` is [fftw3](http://fftw.org/). Since the float and double interface of `fftw3` are used, link with `-lfftw -lfftwf`.

Benchmark and test dependencies:

- [fftw3](http://fftw.org/)
- [armadillo](http://arma.sourceforge.net/) (benchmarked against as a baseline)
- [google-benchmark](https://github.com/google/benchmark) used for benchmarking.
- [gperftools](https://github.com/gperftools/gperftools) used for profiling.

**Python**

TODO The Python wrapper is currently out of date.

A Cython wrapper is provided. Dependencies:

- `Cython` for C++ bindings
- `numpy` (benchmarked against)
- `numba` (benchmarked against)
- `scipy` (benchmarked against)
- `matplotlib` (plot results)

```
python3 setup.py build_ext -i
python3 test.py # run the python test/benchmark
```

## Benchmark results

CPU: Intel i7 Comet Lake

**C++**.

The `test_fftconv` binary gives an easy benchmark that runs every test case 5000 times. The `bench_fftconv` uses `google-benchmark` and gives much more reliable measures. Use `./script/run_bench` to run the benchmark and generate figures.

Output from `bench_fftconv` (accurate bench) raw result saved in `./bench_result.json`. Plot generated from `plot_bench.py`:

![Comparison of the Overlap-Add method implemented with `fftw`, `pocketfft`, and `pocketfft_hdronly`](./bench_2022-08-21T23-11-01.svg)

Output from `test_fftconv` (simple bench)

```
% ./build/test_fftconv
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
    (5000 runs) convolve_fftw took 73ms
    (5000 runs) convolve_pocketfft took 70ms
    (5000 runs) oaconvolve_fftw took 38ms
    (5000 runs) oaconvolve_pocketfft took 53ms
    (5000 runs) np.convolve took 140ms
    (5000 runs) numba.njit(np.convolve) took 1409ms
    (5000 runs) scipy.signal.convolve took 162ms
    (5000 runs) scipy.signal.fftconvolve took 199ms
    (5000 runs) scipy.signal.oaconvolve took 321ms
=== test case (2816, 65) ===
Vectors are equal.
    (5000 runs) convolve_fftw took 96ms
    (5000 runs) convolve_pocketfft took 110ms
    (5000 runs) oaconvolve_fftw took 60ms
    (5000 runs) oaconvolve_pocketfft took 84ms
    (5000 runs) np.convolve took 236ms
    (5000 runs) numba.njit(np.convolve) took 2883ms
    (5000 runs) scipy.signal.convolve took 256ms
    (5000 runs) scipy.signal.fftconvolve took 256ms
    (5000 runs) scipy.signal.oaconvolve took 362ms
=== test case (2304, 65) ===
Vectors are equal.
    (5000 runs) convolve_fftw took 281ms
    (5000 runs) convolve_pocketfft took 132ms
    (5000 runs) oaconvolve_fftw took 53ms
    (5000 runs) oaconvolve_pocketfft took 75ms
    (5000 runs) np.convolve took 194ms
    (5000 runs) numba.njit(np.convolve) took 2215ms
    (5000 runs) scipy.signal.convolve took 213ms
    (5000 runs) scipy.signal.fftconvolve took 240ms
    (5000 runs) scipy.signal.oaconvolve took 346ms
=== test case (4352, 65) ===
Vectors are equal.
    (5000 runs) convolve_fftw took 326ms
    (5000 runs) convolve_pocketfft took 215ms
    (5000 runs) oaconvolve_fftw took 82ms
    (5000 runs) oaconvolve_pocketfft took 117ms
    (5000 runs) np.convolve took 358ms
    (5000 runs) numba.njit(np.convolve) took 3657ms
    (5000 runs) scipy.signal.convolve took 378ms
    (5000 runs) scipy.signal.fftconvolve took 365ms
    (5000 runs) scipy.signal.oaconvolve took 395ms
```

The Python wrapper is almost as fast as the C++ code, as it has very little overhead.

## Implementation Details

TODO
