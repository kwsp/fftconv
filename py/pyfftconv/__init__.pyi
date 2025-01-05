"""
Python wrapper for fftconv
"""
from __future__ import annotations
from pyfftconv._pyfftconv import convolve
from pyfftconv._pyfftconv import convolve_
from pyfftconv._pyfftconv import hilbert
from pyfftconv._pyfftconv import hilbert_
from pyfftconv._pyfftconv import oaconvolve
from pyfftconv._pyfftconv import oaconvolve_
from . import _pyfftconv
__all__ = ['convolve', 'convolve_', 'hilbert', 'hilbert_', 'oaconvolve', 'oaconvolve_']
__version__: str = '0.5.1'
