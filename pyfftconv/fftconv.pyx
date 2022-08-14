# distutils: language=c++
# cython: language_level=3

cimport cfftconv
import cython
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def convolve_fftw(double[:] x, double[:] h):
    """
    1D convolution using the FFT
    """

    cdef:
        int x_size = x.size
        int h_size = h.size
        int padded_length = x_size + h_size - 1

    result = np.zeros(padded_length, dtype=np.double)
    cdef double[:] res_view = result
    cfftconv.convolve_fftw(&x[0], x_size, &h[0], h_size, &res_view[0], padded_length)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def oaconvolve_fftw(double[:] x, double[:] h):
    """
    1D convolution using the overlap-add method. Extremely efficient
    for FIR filtering where `x` is a long signal and `h` is the FIR kernel
    """

    cdef:
        int x_size = x.size
        int h_size = h.size
        int padded_length = x_size + h_size - 1

    result = np.zeros(padded_length, dtype=np.double)
    cdef double[:] res_view = result
    cfftconv.oaconvolve_fftw(&x[0], x_size, &h[0], h_size, &res_view[0], padded_length)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def convolve_pocketfft(double[:] x, double[:] h):
    cdef:
        int x_size = x.size
        int h_size = h.size
        int padded_length = x_size + h_size - 1

    result = np.zeros(padded_length, dtype=np.double)
    cdef double[:] res_view = result
    cfftconv.convolve_pocketfft(&x[0], x_size, &h[0], h_size, &res_view[0], padded_length)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def oaconvolve_pocketfft(double[:] x, double[:] h):
    cdef:
        int x_size = x.size
        int h_size = h.size
        int padded_length = x_size + h_size - 1

    result = np.zeros(padded_length, dtype=np.double)
    cdef double[:] res_view = result
    cfftconv.oaconvolve_pocketfft(&x[0], x_size, &h[0], h_size, &res_view[0], padded_length)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def convolve_pocketfft_hdr(double[:] x, double[:] h):
    cdef:
        int x_size = x.size
        int h_size = h.size
        int padded_length = x_size + h_size - 1

    result = np.zeros(padded_length, dtype=np.double)
    cdef double[:] res_view = result
    cfftconv.convolve_pocketfft_hdr(&x[0], x_size, &h[0], h_size, &res_view[0], padded_length)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def oaconvolve_pocketfft_hdr(double[:] x, double[:] h):
    cdef:
        int x_size = x.size
        int h_size = h.size
        int padded_length = x_size + h_size - 1

    result = np.zeros(padded_length, dtype=np.double)
    cdef double[:] res_view = result
    cfftconv.oaconvolve_pocketfft_hdr(&x[0], x_size, &h[0], h_size, &res_view[0], padded_length)
    return result

