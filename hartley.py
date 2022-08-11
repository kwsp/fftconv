import scipy.signal as signal
import scipy.fft as fft
import numpy as np

a = np.arange(8)
b = np.arange(4)
gt = np.convolve(a, b)

conv_size = a.size + b.size - 1
pa = np.zeros(conv_size)
pa[:a.size] = a
pb = np.zeros(conv_size)
pb[:b.size] = b


def fftconv(pa, pb):
    A = fft.fft(pa)
    B = fft.fft(pb)
    prod = A * B
    return fft.ifft(prod)
fftconv(pa, pb).real.round()

def rfftconv(pa, pb):
    A = fft.rfft(pa)
    B = fft.rfft(pb)
    return fft.irfft(A * B)

rfftconv(pa, pb).real.round()
fft.rfft(pa) * fft.rfft(pb)
fft.fft(pa) * fft.fft(pb)

fft._pocketfft.r2c(True, a)
fft.rfft(a)
