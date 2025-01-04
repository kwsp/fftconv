from scipy import signal
import numpy as np
import pyfftconv

x = np.random.random(20)
expect = np.abs(signal.hilbert(x))

out = np.zeros_like(x)
pyfftconv.hilbert_(x, out)
if np.allclose(out, expect):
    print("Test passed")

out2 = pyfftconv.hilbert(x)
if np.allclose(out2, expect):
    print("Test passed")

out2 = pyfftconv.hilbert(x.astype(np.float32))
if np.allclose(out2, expect):
    print("Test passed")

import timeit
from typing import Callable


def measure_throughput_np(func: Callable, xsize: int, has_out=False, iterations=10000):
    """
    func takes a single arg (np.array of size N)
    """
    x = np.random.random(xsize)

    if has_out:
        out = np.zeros(xsize)
        func_ = lambda: func(x, out)
    else:
        func_ = lambda: func(x)

    func_()

    time = timeit.timeit(func_, number=iterations) / iterations
    throughput = xsize / time
    return throughput


fftconv_tp = measure_throughput_np(pyfftconv.hilbert, 1024 * 5, has_out=False)
print(f"fftconv.hilbert(a) throughput       : {fftconv_tp / 1e6:.1f} MS/s")

np_tp = measure_throughput_np(
    lambda a: np.abs(signal.hilbert(a)), 1024 * 5, has_out=False
)
print(f"np.abs(signal.hilbert(a)) throughput: {np_tp / 1e6:.1f} MS/s")
