# distutils: language=c++
# cython: language_level=3

cimport cfftconv
import cython
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def convolve1d(double[:] a, double[:] b):

    cdef:
        int a_size = a.size
        int b_size = b.size
        int padded_length = a_size + b_size - 1

    result = np.zeros(padded_length, dtype=np.double)
    cdef double[:] res_view = result

    cfftconv.convolve1d(&a[0], a_size, &b[0], b_size, &res_view[0])

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def fftfilt(double[:] a, double[:] b):

    cdef:
        int a_size = a.size
        int b_size = b.size
        int padded_length = a_size + b_size - 1

    result = np.zeros(padded_length, dtype=np.double)
    cdef double[:] res_view = result

    cfftconv.fftfilt(&a[0], a_size, &b[0], b_size, &res_view[0], padded_length)

    return result


