import numpy as np
import pyfftconv

x = np.random.random(20)
k = np.random.random(5)
out = np.zeros_like(x)
expect = np.convolve(x, k, "same")

pyfftconv.oaconvolve_(x, k, out, "same")
if np.allclose(out, expect):
    print("Test passed")

out2 = pyfftconv.oaconvolve(x, k, "same")
if np.allclose(out2, expect):
    print("Test passed")

out2 = pyfftconv.oaconvolve(x.astype(np.float32), k.astype(np.float32), "same")
if np.allclose(out2, expect):
    print("Test passed")

import timeit
from typing import Callable


def measure_throughput_np(
    func: Callable, xsize: int, ksize: int, has_out=False, mode="same", iterations=10000
):
    """
    func takes a single arg (np.array of size N)
    """
    x = np.random.random(xsize)
    k = np.random.random(ksize)

    if has_out:
        if mode == "same":
            out = np.zeros(xsize)
        elif mode == "full":
            out = np.zeros(xsize + ksize - 1)
        else:
            raise ValueError("Unsupported conv mode")

        func_ = lambda: func(x, k, out, mode=mode)
    else:
        func_ = lambda: func(x, k, mode=mode)

    print(x.dtype, k.dtype, mode)
    func(x, k, mode=mode)
    func_()

    time = timeit.timeit(func_, number=iterations) / iterations
    throughput = xsize / time
    return throughput


fftconv_tp = measure_throughput_np(pyfftconv.oaconvolve, 1024 * 5, 165, has_out=False)
print(f"fftconv.oaconvolve throughput: {fftconv_tp / 1e6:.1f} MS/s")

np_tp = measure_throughput_np(np.convolve, 1024 * 5, 165, has_out=False)
print(f"np.convolve throughput       : {np_tp / 1e6:.1f} MS/s")
