import unittest
import numpy as np
from scipy import signal
from numpy.testing import assert_allclose

from pyfftconv import convolve, convolve_, hilbert, hilbert_, oaconvolve, oaconvolve_


class TestFFTConv(unittest.TestCase):
    def setUp(self):
        # Example test data
        self.a = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        self.k = np.array([0.2, 0.5, 0.2], dtype=np.float64)
        self.out_full = np.empty(self.a.size + self.k.size - 1)
        self.out_same = np.empty(self.a.size)

    def test_convolve_full(self):
        result = convolve(self.a, self.k, mode="full")
        expected = np.convolve(self.a, self.k, mode="full")
        assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

    def test_convolve_out_full(self):
        convolve_(self.a, self.k, self.out_full, mode="full")
        expected = np.convolve(self.a, self.k, mode="full")
        assert_allclose(self.out_full, expected, rtol=1e-5, atol=1e-8)

    def test_oaconvolve_full(self):
        result = oaconvolve(self.a, self.k, mode="full")
        expected = np.convolve(self.a, self.k, mode="full")
        assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

    def test_oaconvolve_same(self):
        result = oaconvolve(self.a, self.k, mode="same")
        expected = np.convolve(self.a, self.k, mode="same")
        assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

    def test_oaconvolve_out_full(self):
        oaconvolve_(self.a, self.k, self.out_full, mode="full")
        expected = np.convolve(self.a, self.k, mode="full")
        assert_allclose(self.out_full, expected, rtol=1e-5, atol=1e-8)

    def test_oaconvolve_out_same(self):
        oaconvolve_(self.a, self.k, self.out_same, mode="same")
        expected = np.convolve(self.a, self.k, mode="same")
        assert_allclose(self.out_same, expected, rtol=1e-5, atol=1e-8)

    def test_hilbert(self):
        result = hilbert(self.a)
        expected = np.abs(signal.hilbert(self.a))
        assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

    def test_hilbert_out(self):
        hilbert_(self.a, self.out_same)
        expected = np.abs(signal.hilbert(self.a))
        assert_allclose(self.out_same, expected, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
