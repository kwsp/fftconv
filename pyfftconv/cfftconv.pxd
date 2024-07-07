# distutils: language=c++
import cython
from libcpp.vector cimport vector

cdef extern from "fftconv.h" namespace "fftconv":

    void fftconv(const double *a, const size_t a_size, const double *b,
                 const size_t b_size, double *y, const size_t y_sz);

    void fftconv_oa(const double *x, const size_t x_size, const double *h,
                    const size_t h_sz, double *y, const size_t y_sz);
