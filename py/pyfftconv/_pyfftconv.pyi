"""
Python wrapper for fftconv
"""
from __future__ import annotations
import numpy
import typing
__all__ = ['convolve', 'convolve_', 'hilbert', 'hilbert_', 'oaconvolve', 'oaconvolve_']
@typing.overload
def convolve(a: numpy.ndarray[numpy.float64], k: numpy.ndarray[numpy.float64], mode: str = 'full') -> numpy.ndarray[numpy.float64]:
    """
    Performs convolution using FFTW. API compatible with np.convolve
    """
@typing.overload
def convolve(a: numpy.ndarray[numpy.float32], k: numpy.ndarray[numpy.float32], mode: str = 'full') -> numpy.ndarray[numpy.float32]:
    """
    Performs convolution using FFTW. API compatible with np.convolve
    """
@typing.overload
def convolve_(a: numpy.ndarray[numpy.float64], k: numpy.ndarray[numpy.float64], out: numpy.ndarray[numpy.float64], mode: str = 'full') -> None:
    """
    Performs convolution using FFTW. API compatible with np.convolve
    """
@typing.overload
def convolve_(a: numpy.ndarray[numpy.float32], k: numpy.ndarray[numpy.float32], out: numpy.ndarray[numpy.float32], mode: str = 'full') -> None:
    """
    Performs convolution using FFTW. API compatible with np.convolve
    """
@typing.overload
def hilbert(a: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.float64]:
    """
    Performs envelope detection using the Hilbert transform.
    Equivalent to `np.abs(signal.hilbert(a))`
    """
@typing.overload
def hilbert(a: numpy.ndarray[numpy.float32]) -> numpy.ndarray[numpy.float32]:
    """
    Performs envelope detection using the Hilbert transform.
    Equivalent to `np.abs(signal.hilbert(a))`
    """
@typing.overload
def hilbert_(a: numpy.ndarray[numpy.float64], out: numpy.ndarray[numpy.float64]) -> None:
    """
    Performs envelope detection using the Hilbert transform.
    Equivalent to `np.abs(signal.hilbert(a))`
    """
@typing.overload
def hilbert_(a: numpy.ndarray[numpy.float32], out: numpy.ndarray[numpy.float32]) -> None:
    """
    Performs envelope detection using the Hilbert transform.
    Equivalent to `np.abs(signal.hilbert(a))`
    """
@typing.overload
def oaconvolve(a: numpy.ndarray[numpy.float64], k: numpy.ndarray[numpy.float64], mode: str = 'full') -> numpy.ndarray[numpy.float64]:
    """
    Performs overlap-add convolution using FFTW. API compatible with np.convolve
    """
@typing.overload
def oaconvolve(a: numpy.ndarray[numpy.float32], k: numpy.ndarray[numpy.float32], mode: str = 'full') -> numpy.ndarray[numpy.float32]:
    """
    Performs overlap-add convolution using FFTW. API compatible with np.convolve
    """
@typing.overload
def oaconvolve_(a: numpy.ndarray[numpy.float64], k: numpy.ndarray[numpy.float64], out: numpy.ndarray[numpy.float64], mode: str = 'full') -> None:
    """
    Performs overlap-add convolution using FFTW. API compatible with np.convolve
    """
@typing.overload
def oaconvolve_(a: numpy.ndarray[numpy.float32], k: numpy.ndarray[numpy.float32], out: numpy.ndarray[numpy.float32], mode: str = 'full') -> None:
    """
    Performs overlap-add convolution using FFTW. API compatible with np.convolve
    """
__version__: str = '0.5.1'
