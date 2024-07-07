# distutils: language=C++
import cython

cdef extern from "fftconv.h" namespace "fftconv":
    void convolve1d(const double *a, const size_t a_size, const double *b, const size_t b_size, double *result)
