"""Benchmark pyfftconv 2D/3D convolution against numpy and scipy."""

import time
import numpy as np
from scipy.signal import fftconvolve
from pyfftconv import convolve, convolve2d, convolve3d


def bench(fn, *args, n_warmup=3, n_iter=20):
    """Run fn(*args) n_iter times and return median time in microseconds."""
    for _ in range(n_warmup):
        fn(*args)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]


def fmt(us):
    if us >= 1e6:
        return f"{us / 1e6:8.2f} s "
    if us >= 1e3:
        return f"{us / 1e3:8.2f} ms"
    return f"{us:8.2f} us"


def print_row(label, t_fftconv, t_numpy=None, t_scipy=None):
    parts = [f"  {label:<40s}  {fmt(t_fftconv)}"]
    if t_numpy is not None:
        speedup = t_numpy / t_fftconv
        parts.append(f"  {fmt(t_numpy)}  ({speedup:5.2f}x)")
    if t_scipy is not None:
        speedup = t_scipy / t_fftconv
        parts.append(f"  {fmt(t_scipy)}  ({speedup:5.2f}x)")
    print("".join(parts))


def main():
    np.random.seed(0)

    # ── 1D benchmarks ──
    print("=" * 100)
    print(f"  {'1D Convolution':<40s}  {'pyfftconv':>11s}  {'np.convolve':>11s}  {'ratio':>8s}  {'scipy fftconv':>13s}  {'ratio':>8s}")
    print("-" * 100)

    for n, k in [(2304, 165), (4352, 165), (8192, 255), (65536, 1025)]:
        a = np.random.randn(n)
        kernel = np.random.randn(k)

        for mode in ("full", "same"):
            t_fc = bench(convolve, a, kernel, mode)
            t_np = bench(np.convolve, a, kernel, mode)
            t_sp = bench(fftconvolve, a, kernel, mode)
            label = f"[{n}] * [{k}] mode={mode}"
            print_row(label, t_fc, t_np, t_sp)

    # ── 2D benchmarks ──
    print()
    print("=" * 100)
    print(f"  {'2D Convolution':<40s}  {'pyfftconv':>11s}  {'scipy fftconv':>13s}  {'ratio':>8s}")
    print("-" * 100)

    for (nr, nc), (kr, kc) in [
        ((64, 64), (5, 5)),
        ((64, 64), (15, 15)),
        ((256, 256), (5, 5)),
        ((256, 256), (15, 15)),
        ((512, 512), (15, 15)),
    ]:
        a = np.random.randn(nr, nc)
        kernel = np.random.randn(kr, kc)

        for mode in ("full", "same"):
            t_fc = bench(convolve2d, a, kernel, mode)
            t_sp = bench(fftconvolve, a, kernel, mode)
            label = f"[{nr}x{nc}] * [{kr}x{kc}] mode={mode}"
            print_row(label, t_fc, t_scipy=t_sp)

    # ── 3D benchmarks ──
    print()
    print("=" * 100)
    print(f"  {'3D Convolution':<40s}  {'pyfftconv':>11s}  {'scipy fftconv':>13s}  {'ratio':>8s}")
    print("-" * 100)

    for (d, r, c), (kd, kr, kc) in [
        ((16, 16, 16), (3, 3, 3)),
        ((32, 32, 32), (3, 3, 3)),
        ((64, 64, 64), (3, 3, 3)),
        ((64, 64, 64), (5, 5, 5)),
    ]:
        a = np.random.randn(d, r, c)
        kernel = np.random.randn(kd, kr, kc)

        for mode in ("full", "same"):
            t_fc = bench(convolve3d, a, kernel, mode)
            t_sp = bench(fftconvolve, a, kernel, mode)
            label = f"[{d}x{r}x{c}] * [{kd}x{kr}x{kc}] mode={mode}"
            print_row(label, t_fc, t_scipy=t_sp)

    print("=" * 100)


if __name__ == "__main__":
    main()
