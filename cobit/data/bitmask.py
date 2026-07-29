"""Expand big-int toggle bitmasks into sparse COO bit features.

Each pkl cell is a Python int whose bit ``k`` (LSB = 0) flags a toggle of the
net's RTL bit ``lo + k`` during that cycle. Nets can be up to 311 bits wide,
so expansion must handle arbitrary-precision ints. Only nonzero cells are
touched; output is COO triplets in the global feature space.
"""

from __future__ import annotations

import numpy as np

# Above this width, use the bytes/unpackbits path instead of uint64 shifts.
_SHIFT_WIDTH_MAX = 64


def _coerce_masks(values: np.ndarray, nz_local: np.ndarray, context: str) -> list[int]:
    """Convert nonzero cells to ints, validating float-typed masks.

    The real DB stores some cells as floats (verified on wide data nets).
    Exact-integer floats are coerced; NaN or fractional values are corrupt
    and abort the build immediately with context. Floats >= 2**53 already
    lost low mantissa bits upstream - warn once, the DB is the culprit.
    """
    out: list[int] = []
    warned = False
    for i in nz_local:
        v = values[i]
        if isinstance(v, float):
            if v != v or v != int(v):  # NaN or fractional
                raise ValueError(f"{context}: corrupt bitmask cell {v!r} at row {i}")
            if v >= 2.0**53 and not warned:
                from ..utils import log

                log.warning(
                    "%s: float bitmask >= 2^53 (%r) - low toggle bits may have "
                    "been lost when the DB was dumped", context, v,
                )
                warned = True
        out.append(int(v))
    return out


def expand_net_column(
    values: np.ndarray,
    width: int,
    base_col: int,
    row_offset: int,
    rows_out: list[np.ndarray],
    cols_out: list[np.ndarray],
    context: str = "",
) -> int:
    """Append COO triplets for one net's row-chunk; returns nnz added.

    values: 1-D object/int ndarray of bitmasks for consecutive cycles.
    Row indices emitted are ``row_offset + local_row``; column indices are
    ``base_col + k`` for each set mask bit k.
    """
    nz_local = np.flatnonzero(values != 0)
    if nz_local.size == 0:
        return 0
    masks = _coerce_masks(values, nz_local, context)

    if width <= _SHIFT_WIDTH_MAX:
        a = np.fromiter(masks, dtype=np.uint64, count=nz_local.size)
        bits = (a[:, None] >> np.arange(width, dtype=np.uint64)) & np.uint64(1)
    else:
        nbytes = (width + 7) // 8
        buf = b"".join(m.to_bytes(nbytes, "little") for m in masks)
        packed = np.frombuffer(buf, dtype=np.uint8).reshape(nz_local.size, nbytes)
        bits = np.unpackbits(packed, axis=1, bitorder="little")[:, :width]

    r, k = np.nonzero(bits)
    rows_out.append((row_offset + nz_local[r]).astype(np.int64))
    cols_out.append((base_col + k).astype(np.int64))
    return int(r.size)


def expand_reference(value: int, width: int) -> list[int]:
    """Slow, obviously-correct expansion used by unit tests."""
    return [k for k in range(width) if (value >> k) & 1]
