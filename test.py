from timeit import timeit
import numpy as np
from scipy import signal

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


def run_bench(x, y):
    _timeit("fftconv", lambda: fftconv.fftconv(x, y))
    _timeit("fftconv_oa", lambda: fftconv.fftconv_oa(x, y))
    _timeit("np.convolve", lambda: np.convolve(x, y))
    _timeit("scipy.signal.convolve", lambda: signal.convolve(x, y))
    _timeit("scipy.signal.fftconvolve", lambda: signal.fftconvolve(x, y))


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
