"""Vectorized per-signal transforms (step 3) and scaling (step 4).

Input values arrive as a 1-D numpy *object* array of non-negative Python ints
(one per cycle).  All signals fit their declared width (value <= 2^width-1).

Fast paths (measured on 1.44M rows, numpy 2.4):
  - object -> uint64 via ``.astype(np.uint64)`` covers the full 0..2^64-1 range.
  - popcount via ``np.bitwise_count`` (~0.001 s), not a per-cell ``int.bit_count``.
  - signals wider than 64 bits are handled per 64-bit *lane*; XOR and popcount are
    bitwise, so per-lane toggle+popcount summed over lanes equals the full-width
    result.
"""

from __future__ import annotations

import numpy as np

_MASK64 = (1 << 64) - 1
_U1 = np.uint64(1)


def to_lanes(values: np.ndarray, width: int) -> list[np.ndarray]:
    """Split an object array of ints into little-endian 64-bit uint64 lanes."""
    n = values.shape[0]
    if width <= 64:
        try:
            return [values.astype(np.uint64)]
        except (OverflowError, ValueError, TypeError):
            return [np.fromiter((int(v) & _MASK64 for v in values),
                                dtype=np.uint64, count=n)]
    nlanes = (width + 63) // 64
    lanes = []
    for L in range(nlanes):
        shift = 64 * L
        lanes.append(np.fromiter(((int(v) >> shift) & _MASK64 for v in values),
                                 dtype=np.uint64, count=n))
    return lanes


def split_bits(values: np.ndarray, path: str, width: int, lo: int,
               invert: bool = False) -> list[tuple[str, np.ndarray]]:
    """Bit-split a (possibly multi-bit) signal into single-bit uint8 features.

    Value of bit ``k`` (k in 0..width-1) is ``(intval >> k) & 1``; its label is
    ``path[lo+k]``.  A width-1 signal keeps its *original* column name (never
    ``name[0]``) -- matching the RTL net registry convention.  ``invert`` flips
    each bit (1-x) for type-A ``_stall``/``_idle`` signals.
    """
    n = values.shape[0]
    lanes = to_lanes(values, width)
    out: list[tuple[str, np.ndarray]] = []
    for k in range(width):
        lane = lanes[k // 64]
        b = np.uint64(k % 64)
        bit = ((lane >> b) & _U1).astype(np.uint8)
        if invert:
            bit = np.uint8(1) - bit
        name = path if width == 1 else f"{path}[{lo + k}]"
        out.append((name, bit))
    return out


def toggle_hamming(values: np.ndarray, width: int,
                   count_initial: bool = True) -> np.ndarray:
    """Hamming distance of the per-cycle toggle of a bus.

    ``toggle[t] = state[t] XOR state[t-1]`` with ``state[-1] = 0`` (per case),
    then ``hamming[t] = popcount(toggle[t])`` = number of bus bits that flipped.
    With ``count_initial=True`` the initial state counts as a toggle-from-reset
    (``hamming[0] = popcount(state[0])``); otherwise ``hamming[0] = 0``.
    """
    n = values.shape[0]
    ham = np.zeros(n, dtype=np.int32)
    for lane in to_lanes(values, width):
        tog = np.empty_like(lane)
        tog[0] = lane[0] if count_initial else np.uint64(0)
        if n > 1:
            tog[1:] = lane[1:] ^ lane[:-1]
        ham += np.bitwise_count(tog).astype(np.int32)
    return ham


def scale_bits(bits: np.ndarray) -> np.ndarray:
    """Single-bit features: max value is 1, so scaling is a no-op (return float)."""
    return bits.astype(np.float64)


def scale_hamming(ham: np.ndarray, bus_width: int) -> np.ndarray:
    """Hamming in [0, W] -> divide by bus width W for a clean [0,1] fraction."""
    return ham.astype(np.float64) / float(bus_width)


def scale_raw_int(values: np.ndarray, width: int) -> np.ndarray:
    """Type-D raw integer scaled by its max representable value 2^width - 1.

    A few signals in this DB carry a dump artifact value of exactly 2^width
    (e.g. ``mepc_value[63:0]`` = 2^64 on ~0.003% of cycles); clip guarantees the
    [0,1] invariant regardless of float64 rounding of the denominator.
    """
    denom = float((1 << width) - 1)
    return np.clip(values.astype(np.float64) / denom, 0.0, 1.0)
