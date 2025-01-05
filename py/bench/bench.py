from google_benchmark import State
import google_benchmark as benchmark
import numpy as np

from pyfftconv import oaconvolve, oaconvolve_, convolve, convolve_, hilbert, hilbert_

ARGS_FIR = [[2304, 4352], [165]]


def BM_conv(fn, state: State, x, k, mode: str):
    fn(x, k, mode)
    while state:
        fn(x, k, mode)
    state.items_processed = state.iterations * x.size
    state.bytes_processed = state.items_processed * x.itemsize


def BM_conv_out(fn, state: State, x, k, out, mode: str):
    fn(x, k, out, mode)
    while state:
        fn(x, k, out, mode)
    state.items_processed = state.iterations * x.size
    state.bytes_processed = state.items_processed * x.itemsize


def BM_hilbert(fn, state: State, x):
    fn(x)
    while state:
        fn(x)
    state.items_processed = state.iterations * x.size
    state.bytes_processed = state.items_processed * x.itemsize


def BM_hilbert_out(fn, state: State, x, out):
    fn(x, out)
    while state:
        fn(x, out)
    state.items_processed = state.iterations * x.size
    state.bytes_processed = state.items_processed * x.itemsize


@benchmark.register
@benchmark.option.args_product(ARGS_FIR)
def BM_oaconvolve_same_double(state: State):
    x = np.random.random(state.range(0)).astype(np.float64)
    k = np.random.random(state.range(1)).astype(np.float64)
    BM_conv(oaconvolve, state, x, k, "same")


@benchmark.register
@benchmark.option.args_product(ARGS_FIR)
def BM_oaconvolve_same_float(state: State):
    x = np.random.random(state.range(0)).astype(np.float32)
    k = np.random.random(state.range(1)).astype(np.float32)
    BM_conv(oaconvolve, state, x, k, "same")


@benchmark.register
@benchmark.option.args_product(ARGS_FIR)
def BM_oaconvolve_same_out_double(state: State):
    x = np.random.random(state.range(0)).astype(np.float64)
    k = np.random.random(state.range(1)).astype(np.float64)
    out = np.zeros_like(x)
    BM_conv_out(oaconvolve_, state, x, k, out, "same")


@benchmark.register
@benchmark.option.args_product(ARGS_FIR)
def BM_oaconvolve_same_out_float(state: State):
    x = np.random.random(state.range(0)).astype(np.float32)
    k = np.random.random(state.range(1)).astype(np.float32)
    out = np.zeros_like(x)
    BM_conv_out(oaconvolve_, state, x, k, out, "same")


@benchmark.register
@benchmark.option.dense_range(2048, 6144, 1024)
def BM_hilbert_float(state: State):
    x = np.random.random(state.range(0)).astype(np.float32)
    BM_hilbert(hilbert, state, x)


@benchmark.register
@benchmark.option.dense_range(2048, 6144, 1024)
def BM_hilbert_double(state: State):
    x = np.random.random(state.range(0)).astype(np.float64)
    BM_hilbert(hilbert, state, x)


@benchmark.register
@benchmark.option.dense_range(2048, 6144, 1024)
def BM_hilbert_out_float(state: State):
    x = np.random.random(state.range(0)).astype(np.float32)
    out = np.zeros_like(x)
    BM_hilbert_out(hilbert_, state, x, out)


@benchmark.register
@benchmark.option.dense_range(2048, 6144, 1024)
def BM_hilbert_out_double(state: State):
    x = np.random.random(state.range(0)).astype(np.float64)
    out = np.zeros_like(x)
    BM_hilbert_out(hilbert_, state, x, out)


if __name__ == "__main__":
    benchmark.main()
