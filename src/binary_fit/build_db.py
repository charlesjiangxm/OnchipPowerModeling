"""Stage 0 (``--build_db``): materialize the single-bit feature dataset.

Reads the raw per-cycle signal-STATE func pkls (columns ``path[hi:lo]`` holding
Python ints) for the configured module(s) and writes one
``<bench>_func.pkl.zst`` of single-bit uint8 columns per benchmark, in one
canonical order shared by every benchmark. Bit ``k`` of a net is ``(int(v)>>k)&1``
named ``path[lo+k]`` (a width-1 net keeps its bare name).

Two passes so the column set is split-independent: pass 1 scans ALL benchmarks
to find the globally-live bits (a bit is kept iff it is 1 in some cycle AND 0 in
some cycle across the whole corpus -- this drops globally constant-0 and
constant-1 bits, the latter being useless and collinear with a model intercept);
pass 2 slices and writes. The power side is untouched -- the target ``pwr/`` is
already populated -- so only ``func/`` is produced.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import _FUNC_SUFFIXES, _find, discover_benches
from .utils import log, save_pickle_zst

_MASK64 = (1 << 64) - 1
_U1 = np.uint64(1)
_RANGE_RE = re.compile(r"^(?P<path>.*?)(?:\[(?P<a>\d+):(?P<b>\d+)\])?$")


def parse_column(column: str) -> tuple[str, int, int]:
    """Return (path, width, lo) for a pkl column name ``path`` or ``path[hi:lo]``."""
    m = _RANGE_RE.match(column)
    assert m is not None
    if m.group("a") is None:
        return column, 1, 0
    a, b = int(m.group("a")), int(m.group("b"))
    return m.group("path"), abs(a - b) + 1, min(a, b)


def _to_lanes(values: np.ndarray, width: int) -> list[np.ndarray]:
    """Split an object array of non-negative ints into little-endian uint64 lanes."""
    n = values.shape[0]
    if width <= 53:
        try:
            return [values.astype(np.uint64)]
        except (OverflowError, ValueError, TypeError):
            return [np.fromiter((int(v) & _MASK64 for v in values), np.uint64, n)]
    if width <= 64:
        return [np.fromiter((int(v) & _MASK64 for v in values), np.uint64, n)]
    nlanes = (width + 63) // 64
    return [np.fromiter(((int(v) >> (64 * L)) & _MASK64 for v in values),
                        np.uint64, n) for L in range(nlanes)]


def _bit(lanes: list[np.ndarray], k: int) -> np.ndarray:
    """Single-bit uint8 array for bit ``k`` across the (lane-split) values."""
    lane = lanes[k // 64]
    return ((lane >> np.uint64(k % 64)) & _U1).astype(np.uint8)


def _column_or_and(lanes: list[np.ndarray]) -> tuple[int, int]:
    """Per-bit OR- and AND-reduction over all cycles, as arbitrary-precision ints."""
    gor = gand = 0
    for L, lane in enumerate(lanes):
        gor |= int(np.bitwise_or.reduce(lane)) << (64 * L)
        gand |= int(np.bitwise_and.reduce(lane)) << (64 * L)
    return gor, gand


def _load_raw(cfg: Config, bench: str) -> pd.DataFrame:
    """Raw (multi-bit) func frame for one benchmark, union of configured modules."""
    frames = []
    for m in cfg.build.modules:
        p = _find(Path(cfg.build.source_db_root) / "func" / m, bench, _FUNC_SUFFIXES)
        frames.append(pd.read_pickle(p))
    raw = frames[0] if len(frames) == 1 else pd.concat(frames, axis=1)
    return raw.sort_index()


def build(cfg: Config) -> list[Path]:
    """Materialize ``<out_root>/func/<bench>_func.pkl.zst`` for every benchmark."""
    src_dir0 = Path(cfg.build.source_db_root) / "func" / cfg.build.modules[0]
    benches = discover_benches(str(src_dir0))
    if not benches:
        raise RuntimeError(f"no source func pkls under {src_dir0}")
    out_func = Path(cfg.build.out_root) / "func"
    out_func.mkdir(parents=True, exist_ok=True)
    log.info("build_db: modules=%s benches=%s -> %s",
             cfg.build.modules, benches, out_func)

    # ---- pass 1: canonical schema + global live-bit mask ------------------
    src_columns: list[str] | None = None
    meta: dict[str, tuple[str, int, int]] = {}  # column -> (path, width, lo)
    gor: dict[str, int] = {}
    gand: dict[str, int | None] = {}
    if cfg.build.drop_dead_bits:
        for b in benches:
            raw = _load_raw(cfg, b)
            cols = sorted(raw.columns)
            if src_columns is None:
                src_columns = cols
                for c in cols:
                    meta[c] = parse_column(c)
                    gor[c], gand[c] = 0, None
            elif cols != src_columns:
                raise ValueError(
                    f"benchmark {b!r} column set differs from {benches[0]!r} "
                    f"({len(cols)} vs {len(src_columns)}) - cannot build a canonical dataset")
            for c in src_columns:
                _, w, _ = meta[c]
                cor, cand = _column_or_and(_to_lanes(raw[c].to_numpy(), w))
                gor[c] |= cor
                gand[c] = cand if gand[c] is None else (gand[c] & cand)
            log.info("  pass1 scanned %-12s (%d cols)", b, len(src_columns))
            del raw
    else:
        raw = _load_raw(cfg, benches[0])
        src_columns = sorted(raw.columns)
        for c in src_columns:
            meta[c] = parse_column(c)
        del raw

    # live bits per column: 1 somewhere AND 0 somewhere (kept in canonical order)
    live_bits: dict[str, list[int]] = {}
    canonical: list[str] = []
    for c in src_columns:
        path, w, lo = meta[c]
        if cfg.build.drop_dead_bits:
            o, a = gor[c], gand[c] or 0
            ks = [k for k in range(w) if (o >> k) & 1 and not (a >> k) & 1]
        else:
            ks = list(range(w))
        live_bits[c] = ks
        for k in ks:
            canonical.append(path if w == 1 else f"{path}[{lo + k}]")
    if len(canonical) != len(set(canonical)):
        raise ValueError("expanded single-bit names are not unique")
    log.info("build_db: %d source cols -> %d live single-bit features",
             len(src_columns), len(canonical))

    # ---- pass 2: slice live bits and write per benchmark ------------------
    written: list[Path] = []
    for b in benches:
        raw = _load_raw(cfg, b)
        data = {}
        for c in src_columns:
            ks = live_bits[c]
            if not ks:
                continue
            path, w, lo = meta[c]
            lanes = _to_lanes(raw[c].to_numpy(), w)
            for k in ks:
                name = path if w == 1 else f"{path}[{lo + k}]"
                data[name] = _bit(lanes, k)
        df = pd.DataFrame(data, index=raw.index.astype("int64"))[canonical]
        out = out_func / f"{b}_func.pkl.zst"
        save_pickle_zst(df, out)
        written.append(out)
        log.info("  wrote %-12s %s -> %s", b, tuple(df.shape), out.name)
        del raw, data, df
    return written
