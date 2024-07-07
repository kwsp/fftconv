import numpy as np
from fftconv import convolve1d

x = np.arange(5, dtype=np.double)
y = np.arange(5, dtype=np.double)


def test_conv(x, y):
    res = convolve1d(x, x.size, y, y.size)
    gt = np.convolve(x, y)
    assert np.allclose(res, gt)
    print("Vectors are equal.")


from timeit import timeit

N_RUNS = 5000

def _timeit(name, callable):
    elapsed_ms = 1000 * timeit(
        callable,
        number=N_RUNS,
    )
    print(f"    ({N_RUNS} runs) {name} took {round(elapsed_ms)}ms")


def run_bench(x, y):
    _timeit("fftconv", lambda: convolve1d(x, x.size, y, y.size))
    _timeit("np.conv", lambda: np.convolve(x, y))


def run_test_case(x, y):
    print(f"=== test case ({x.size}, {y.size}) ===")
    test_conv(x, y)
    run_bench(x, y)


if __name__ == "__main__":
    x = np.linspace(0, 1, 1664)
    y = np.linspace(0, 1, 65)
    run_test_case(x, y)

    x = np.linspace(0, 1, 2816)
    y = np.linspace(0, 1, 65)
    run_test_case(x, y)

    x = np.linspace(0, 1, 2000)
    y = np.linspace(0, 1, 2000)
    run_test_case(x, y)
