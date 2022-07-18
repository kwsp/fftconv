# distutils: language = c++
# cython: language_level=3

cimport cfftconv
import cython
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def convolve1d(double[:] a, int a_size, double[:] b, int b_size):

    cdef int padded_length = a_size + b_size - 1

    cdef double[:] a_view = a
    cdef double[:] b_view = b
    result = np.zeros(padded_length, dtype=np.double)
    cdef double[:] res_view = result

    cfftconv.convolve1d(&a_view[0], a_size, &b_view[0], b_size, &res_view[0])

    return result

