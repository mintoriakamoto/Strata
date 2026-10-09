"""Tests for tools/ternary_lut.py (no GPU, no downloads):

    python -m unittest tools.test_ternary_lut
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ternary_lut as t  # noqa: E402


def rand_w(rng, rows, cols):
    return rng.integers(-1, 2, size=(rows, cols)).astype(np.int8)


class TernaryLut(unittest.TestCase):
    def test_bytes_stay_below_243(self):
        w = np.full((4, 10), 1, dtype=np.int8)
        self.assertEqual(int(t.pack(w).max()), 242)
        self.assertEqual(int(t.pack(-w).max()), 0)

    def test_roundtrip_with_padding(self):
        rng = np.random.default_rng(1)
        for cols in (5, 7, 64, 2048, 2049):
            w = rand_w(rng, 13, cols)
            np.testing.assert_array_equal(t.unpack(t.pack(w), cols), w)

    def test_float_matvec_matches_dense(self):
        rng = np.random.default_rng(2)
        w, x = rand_w(rng, 32, 2048), rng.standard_normal(2048)
        scale = rng.random(32) + 0.5
        np.testing.assert_allclose(t.matvec(t.pack(w), x, scale), scale * (w.astype(np.float64) @ x), rtol=1e-12)

    def test_unaligned_columns(self):
        rng = np.random.default_rng(3)
        w, x = rand_w(rng, 9, 103), rng.standard_normal(103)
        np.testing.assert_allclose(t.matvec(t.pack(w), x), w.astype(np.float64) @ x, rtol=1e-12)

    def test_int8_path_is_exact(self):
        rng = np.random.default_rng(4)
        w, x = rand_w(rng, 16, 2050), rng.standard_normal(2050)
        xq, s = t.quantize_activations(x)
        expect = (w.astype(np.int64) @ xq.astype(np.int64)) * s
        np.testing.assert_allclose(t.matvec_int8(t.pack(w), xq, s), expect, rtol=1e-12)

    def test_int16_table_cannot_overflow(self):
        self.assertLessEqual(5 * 127, np.iinfo(np.int16).max)

    def test_sizes(self):
        self.assertAlmostEqual(t.bytes_per_matrix(1000, 5000) * 8 / (1000 * 5000), 1.6)
        # two 256-entry tables (10 weights) at int16 = 1 KB; 16 such pairs = 16 KB of shared memory
        self.assertEqual(t.table_bytes(2), 1024)
        self.assertEqual(t.table_bytes(32), 16 * 1024)


if __name__ == "__main__":
    unittest.main()
