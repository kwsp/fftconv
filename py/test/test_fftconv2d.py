import unittest
import numpy as np
from scipy.signal import fftconvolve
from numpy.testing import assert_allclose

from pyfftconv import convolve2d, convolve2d_


class TestConvolve2D(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.a = np.random.randn(32, 32).astype(np.float64)
        self.k = np.random.randn(5, 5).astype(np.float64)

    def test_full_double(self):
        result = convolve2d(self.a, self.k, mode="full")
        expected = fftconvolve(self.a, self.k, mode="full")
        assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_same_double(self):
        result = convolve2d(self.a, self.k, mode="same")
        expected = fftconvolve(self.a, self.k, mode="same")
        assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_full_float(self):
        a = self.a.astype(np.float32)
        k = self.k.astype(np.float32)
        result = convolve2d(a, k, mode="full")
        expected = fftconvolve(a.astype(np.float64), k.astype(np.float64), mode="full")
        assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_same_float(self):
        a = self.a.astype(np.float32)
        k = self.k.astype(np.float32)
        result = convolve2d(a, k, mode="same")
        expected = fftconvolve(a.astype(np.float64), k.astype(np.float64), mode="same")
        assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_full_out_double(self):
        rows, cols = self.a.shape
        krows, kcols = self.k.shape
        out = np.empty((rows + krows - 1, cols + kcols - 1), dtype=np.float64)
        convolve2d_(self.a, self.k, out, mode="full")
        expected = fftconvolve(self.a, self.k, mode="full")
        assert_allclose(out, expected, rtol=1e-10, atol=1e-10)

    def test_same_out_double(self):
        out = np.empty_like(self.a)
        convolve2d_(self.a, self.k, out, mode="same")
        expected = fftconvolve(self.a, self.k, mode="same")
        assert_allclose(out, expected, rtol=1e-10, atol=1e-10)

    def test_large_input(self):
        a = np.random.randn(128, 128).astype(np.float64)
        k = np.random.randn(15, 15).astype(np.float64)
        result = convolve2d(a, k, mode="full")
        expected = fftconvolve(a, k, mode="full")
        assert_allclose(result, expected, rtol=1e-9, atol=1e-9)

    def test_nonsquare(self):
        a = np.random.randn(64, 32).astype(np.float64)
        k = np.random.randn(7, 3).astype(np.float64)
        result = convolve2d(a, k, mode="full")
        expected = fftconvolve(a, k, mode="full")
        assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
