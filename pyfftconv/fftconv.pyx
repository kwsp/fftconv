# distutils: language=c++
# cython: language_level=3

cimport cfftconv
import cython
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def fftconv(double[:] a, double[:] b):

    cdef:
        int a_size = a.size
        int b_size = b.size
        int padded_length = a_size + b_size - 1

    result = np.zeros(padded_length, dtype=np.double)
    cdef double[:] res_view = result

    cfftconv.fftconv(&a[0], a_size, &b[0], b_size, &res_view[0], padded_length)

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def fftconv_oa(double[:] a, double[:] b):

    cdef:
        int a_size = a.size
        int b_size = b.size
        int padded_length = a_size + b_size - 1

    result = np.zeros(padded_length, dtype=np.double)
    cdef double[:] res_view = result

    cfftconv.fftconv_oa(&a[0], a_size, &b[0], b_size, &res_view[0], padded_length)

    return result


