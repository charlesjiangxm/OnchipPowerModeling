"""Streaming raw-value statistics over the training cases (step 1 + report).

A feature is "zero-variance" iff it is *constant* across the whole training set.
We detect this exactly via ``global_min == global_max`` (robust for 128-bit ints;
no float64 precision loss).  The same streaming pass yields the raw min/max/mean
for the report and validates the type-D denominator (max <= 2^width-1), and
optionally a per-column value fingerprint for exact-duplicate detection.

Only training cases are streamed -- the kept-column set, denominators and every
other decision are a train-only "fit" applied verbatim to the test case, so there
is no test leakage.
"""

from __future__ import annotations

import hashlib

import numpy as np

from transform import to_lanes


class RawStats:
    """Accumulate per-column min/max/sum/count (+ optional dup hash) case-by-case."""

    def __init__(self, columns: list[str], widths: dict[str, int],
                 detect_duplicates: bool = True):
        self.columns = columns
        self.widths = widths
        self.detect_duplicates = detect_duplicates
        self.min: dict[str, int] = {}
        self.max: dict[str, int] = {}
        self.sum: dict[str, float] = {c: 0.0 for c in columns}
        self.count: int = 0
        self._hash: dict[str, "hashlib._Hash"] = (
            {c: hashlib.blake2b(digest_size=16) for c in columns}
            if detect_duplicates else {}
        )

    def update(self, df) -> None:
        """Fold one aligned case DataFrame (columns already canonical) into stats."""
        self.count += df.shape[0]
        for col in self.columns:
            width = self.widths[col]
            v = df[col].to_numpy()
            if width <= 64:
                try:
                    u = v.astype(np.uint64)
                    mn = int(u.min()); mx = int(u.max())
                    self.sum[col] += float(u.astype(np.float64).sum())
                    if self.detect_duplicates:
                        self._hash[col].update(u.tobytes())
                    self._merge(col, mn, mx)
                    continue
                except (OverflowError, ValueError, TypeError):
                    pass
            # wide (>64b) or non-castable: object min/max + lane hashing
            ints = [int(x) for x in v]
            self._merge(col, min(ints), max(ints))
            self.sum[col] += float(sum(ints))
            if self.detect_duplicates:
                for lane in to_lanes(v, width):
                    self._hash[col].update(lane.tobytes())

    def _merge(self, col: str, mn: int, mx: int) -> None:
        self.min[col] = mn if col not in self.min else min(self.min[col], mn)
        self.max[col] = mx if col not in self.max else max(self.max[col], mx)

    def finalize(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for col in self.columns:
            mn, mx = self.min[col], self.max[col]
            rec = {
                "raw_min": mn,
                "raw_max": mx,
                "raw_mean": self.sum[col] / self.count if self.count else 0.0,
                "constant": (mn == mx),
            }
            if self.detect_duplicates:
                rec["value_hash"] = self._hash[col].hexdigest()
            out[col] = rec
        return out


def duplicate_groups(finalized: dict[str, dict], kept: list[str]) -> list[list[str]]:
    """Group kept columns that share an identical value fingerprint.

    Only groups of size >= 2 are returned.  Members are ordered canonically
    (shortest path first, then lexicographic) so the representative is members[0].
    """
    by_hash: dict[str, list[str]] = {}
    for col in kept:
        h = finalized[col].get("value_hash")
        if h is None:
            continue
        by_hash.setdefault(h, []).append(col)
    groups = []
    for cols in by_hash.values():
        if len(cols) >= 2:
            groups.append(sorted(cols, key=lambda c: (len(c), c)))
    groups.sort(key=lambda g: (len(g[0]), g[0]))
    return groups
