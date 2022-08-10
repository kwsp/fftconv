from timeit import timeit
import numpy as np
from scipy import signal
from numba import jit

import fftconv


def test_conv(x, y):
    gt = np.convolve(x, y)

    res = fftconv.fftconv(x, y)
    assert np.allclose(res, gt)

    res = fftconv.fftconv_oa(x, y)
    assert np.allclose(res, gt)

    print("Vectors are equal.")


N_RUNS = 5000


def _timeit(name, callable):
    elapsed_ms = 1000 * timeit(
        callable,
        number=N_RUNS,
    )
    print(f"    ({N_RUNS} runs) {name} took {round(elapsed_ms)}ms")

@jit
def numba_convolve(x, y):
    return np.convolve(x, y)

def run_bench(x, y):
    _timeit("fftconv", lambda: fftconv.fftconv(x, y))
    _timeit("fftconv_oa", lambda: fftconv.fftconv_oa(x, y))
    _timeit("np.convolve", lambda: np.convolve(x, y))

    numba_convolve(x, y)
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
