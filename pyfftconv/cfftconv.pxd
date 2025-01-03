# distutils: language=c++
import cython

cdef extern from "fftconv.hpp" namespace "fftconv":

    void convolve_fftw(const double *a, const size_t a_size, const double *b, 
            const size_t b_size, double *y, const size_t y_sz);

    void oaconvolve_fftw(const double *x, const size_t x_size, const double *h, 
            const size_t h_sz, double *y, const size_t y_sz);
