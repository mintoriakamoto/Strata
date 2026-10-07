"""Reference for ternary (-1, 0, +1) weights with a lookup-table matrix-vector product.

Five ternary weights pack into one byte (3^5 = 243 <= 256, so 1.6 bits per weight). For every group of five
activations the product with all 243 possible weight groups is precomputed into a 256-entry table; a row's dot
product is then one table lookup and one add per byte, with no multiplies. The CUDA kernel in
src/kernels/cuda/ternary_lut.cu follows this layout. NumPy only, no GPU.

    python -m unittest tools.test_ternary_lut
"""
from __future__ import annotations

import numpy as np

GROUP = 5                    # weights per byte
TABLE = 256                  # table entries per group (243 used)
USED = 3 ** GROUP            # 243


def _digits() -> np.ndarray:
    """[256, 5] ternary value of every byte; unused bytes (243..255) decode to zeros."""
    d = np.zeros((TABLE, GROUP), dtype=np.int8)
    for b in range(USED):
        v = b
        for i in range(GROUP):
            d[b, i] = v % 3 - 1
            v //= 3
    return d


DIGITS = _digits()


def pack(w: np.ndarray) -> np.ndarray:
    """Pack ternary w [rows, cols] into bytes [cols_padded / 5, rows] (group-major, so a warp reading consecutive
    rows of one group makes one coalesced load). cols is zero-padded to a multiple of 5."""
    w = np.asarray(w)
    if w.ndim != 2 or not np.isin(w, (-1, 0, 1)).all():
        raise ValueError("w must be a 2-D array of -1, 0, +1")
    rows, cols = w.shape
    pad = -cols % GROUP
    if pad:
        w = np.pad(w, ((0, 0), (0, pad)))
    g = (w.astype(np.int16) + 1).reshape(rows, -1, GROUP)
    scale = 3 ** np.arange(GROUP, dtype=np.int16)
    return np.ascontiguousarray((g * scale).sum(axis=2).astype(np.uint8).T)


def unpack(packed: np.ndarray, cols: int) -> np.ndarray:
    """Inverse of pack: [rows, cols] int8."""
    groups, rows = packed.shape
    w = DIGITS[packed.T].reshape(rows, groups * GROUP)
    return w[:, :cols]


def build_tables(x: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Tables [groups, 256]: entry b of group g is the dot product of byte b's five weights with x[5g:5g+5]."""
    pad = -len(x) % GROUP
    if pad:
        x = np.pad(x, (0, pad))
    return x.reshape(-1, GROUP).astype(dtype) @ DIGITS.T.astype(dtype)


def matvec(packed: np.ndarray, x: np.ndarray, scale: np.ndarray | float = 1.0) -> np.ndarray:
    """y = scale * (W @ x) with W given as pack(W). One lookup and add per byte."""
    tables = build_tables(x)
    g = np.arange(packed.shape[0])[:, None]
    return tables[g, packed].sum(axis=0) * scale


def quantize_activations(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric int8 quantization: x ~ q * s."""
    s = float(np.abs(x).max()) / 127.0 or 1.0
    return np.clip(np.rint(x / s), -127, 127).astype(np.int8), s


def matvec_int8(packed: np.ndarray, xq: np.ndarray, x_scale: float, scale: np.ndarray | float = 1.0) -> np.ndarray:
    """Integer path the GPU kernel uses: int16 tables (|entry| <= 5 * 127), int32 accumulation, scales applied last."""
    tables = build_tables(xq, dtype=np.int16)
    g = np.arange(packed.shape[0])[:, None]
    acc = tables[g, packed].astype(np.int32).sum(axis=0)
    return acc * (x_scale * scale)


def bytes_per_matrix(rows: int, cols: int) -> int:
    return rows * (-(-cols // GROUP))


def table_bytes(groups: int, entry_bytes: int = 2) -> int:
    """Shared memory for the tables of `groups` activation groups (int16 entries by default)."""
    return groups * TABLE * entry_bytes
