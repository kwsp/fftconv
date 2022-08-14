from timeit import timeit
import numpy as np
from scipy import signal
from numba import jit

import fftconv


def test_conv(x, y):
    gt = np.convolve(x, y)

    def _test(func):
        np.allclose(gt, func(x, y))

    _test(fftconv.convolve_fftw)
    _test(fftconv.convolve_pocketfft)
    _test(fftconv.convolve_pocketfft_hdr)

    _test(fftconv.oaconvolve_fftw)
    _test(fftconv.oaconvolve_pocketfft)
    _test(fftconv.oaconvolve_pocketfft_hdr)

    print("Vectors are equal.")


N_RUNS = 5000


@jit
def numba_convolve(x, y):
    return np.convolve(x, y)


def run_bench(x, y):
    def _timeit(name, func):
        elapsed_ms = 1000 * timeit(
            func,
            number=N_RUNS,
        )
        print(f"    ({N_RUNS} runs) {name} took {round(elapsed_ms)}ms")

    _timeit("convolve_fftw", lambda: fftconv.convolve_fftw(x, y))
    _timeit("convolve_pocketfft", lambda: fftconv.convolve_pocketfft(x, y))
    _timeit("convolve_pocketfft_hdr", lambda: fftconv.convolve_pocketfft_hdr(x, y))

    _timeit("oaconvolve_fftw", lambda: fftconv.oaconvolve_fftw(x, y))
    _timeit("oaconvolve_pocketfft", lambda: fftconv.oaconvolve_pocketfft(x, y))
    _timeit("oaconvolve_pocketfft_hdr", lambda: fftconv.oaconvolve_pocketfft_hdr(x, y))

    numba_convolve(x, y)  # warm jit
    _timeit("np.convolve", lambda: np.convolve(x, y))
    _timeit("jit(np.convolve)", lambda: numba_convolve(x, y))
    _timeit("scipy.signal.convolve", lambda: signal.convolve(x, y))
    _timeit("scipy.signal.fftconvolve", lambda: signal.fftconvolve(x, y))
    _timeit("scipy.signal.oaconvolve", lambda: signal.oaconvolve(x, y))


def run_test_case(x, y):
    print(f"=== test case ({x.size}, {y.size}) ===")
    test_conv(x, y)
    run_bench(x, y)


def get_vec(n):
    return np.random.random(n)


if __name__ == "__main__":
    run_test_case(get_vec(1664), get_vec(65))
    run_test_case(get_vec(2816), get_vec(65))
    run_test_case(get_vec(2304), get_vec(65))
    run_test_case(get_vec(4352), get_vec(65))
