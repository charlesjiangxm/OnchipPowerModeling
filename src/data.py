"""Data loading, x/y pair matching, per-benchmark train/val/test split,
and non-overlapped window averaging.

The library accepts pickled pandas DataFrames:
  - x pkl: shape (rows, signal-features); columns are feature names.
  - y pkl: shape (rows, >=1 columns); one column is `y_label`.
Stems are matched by stripping `_func` from x file stems and `_pwr` from
y file stems, then comparing the remainder. Fatal on any unmatched pair,
schema mismatch, or row-count mismatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_path


log = logging.getLogger(__name__)

_X_SUFFIX = "_func"
_Y_SUFFIX = "_pwr"


@dataclass
class Benchmark:
    name: str
    x: pd.DataFrame  # (N, F)
    y: pd.Series     # (N,)


@dataclass
class SplitData:
    train_x: pd.DataFrame
    train_y: pd.Series
    val_x: pd.DataFrame
    val_y: pd.Series
    test_x: pd.DataFrame
    test_y: pd.Series


def _strip_suffix(stem: str, suffix: str) -> str:
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def _index_by_stem(paths: list[str | Path], suffix: str, kind: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for raw in paths:
        p = resolve_path(raw)
        if not p.is_file():
            raise FileNotFoundError(f"{kind} pkl not found: {p}")
        key = _strip_suffix(p.stem, suffix)
        if key in out:
            raise ValueError(
                f"Duplicate {kind} stem after stripping {suffix!r}: "
                f"{out[key]!r} and {p!r} both map to {key!r}"
            )
        out[key] = p
    return out


def _read_x(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path}: expected pandas DataFrame, got {type(df).__name__}")
    return df.astype(np.float64).reset_index(drop=True)


def _read_y(path: Path, y_label: str) -> pd.Series:
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path}: expected pandas DataFrame, got {type(df).__name__}")
    if y_label not in df.columns:
        raise KeyError(
            f"{path}: y_label={y_label!r} not in columns. Available: {list(df.columns)}"
        )
    return df[y_label].astype(np.float64).reset_index(drop=True)


def load_pairs(
    trainset_x_path: list[str],
    testset_x_path: list[str],
    y_path: list[str],
    y_label: str,
) -> tuple[list[Benchmark], list[Benchmark]]:
    """Load and stem-match (x, y) pairs. Returns (train_benchmarks, test_benchmarks)."""
    train_idx = _index_by_stem(trainset_x_path, _X_SUFFIX, "train x")
    test_idx = _index_by_stem(testset_x_path, _X_SUFFIX, "test x")
    y_idx = _index_by_stem(y_path, _Y_SUFFIX, "y")

    if overlap := set(train_idx) & set(test_idx):
        raise ValueError(
            f"The same stem(s) appear in both train and test x lists: {sorted(overlap)}"
        )

    needed = set(train_idx) | set(test_idx)
    missing_y = needed - set(y_idx)
    if missing_y:
        raise ValueError(
            f"No y pkl found for x stem(s): {sorted(missing_y)}. "
            f"Available y stems: {sorted(y_idx)}"
        )
    unused_y = set(y_idx) - needed
    if unused_y:
        log.warning("y pkl(s) with no matching x will be ignored: %s", sorted(unused_y))

    train = [_make_benchmark(s, train_idx[s], y_idx[s], y_label) for s in sorted(train_idx)]
    test = [_make_benchmark(s, test_idx[s], y_idx[s], y_label) for s in sorted(test_idx)]

    _assert_consistent_columns([b.x for b in train], "train")
    if test:
        _assert_consistent_columns([b.x for b in test], "test")
        train_cols = list(train[0].x.columns)
        test_cols = list(test[0].x.columns)
        if train_cols != test_cols:
            raise ValueError(
                f"Train and test x pkls have different columns "
                f"(train has {len(train_cols)}, test has {len(test_cols)} columns; "
                f"first mismatch: "
                f"train[0]={train_cols[0]!r} vs test[0]={test_cols[0]!r})"
            )

    return train, test


def _make_benchmark(name: str, x_path: Path, y_path: Path, y_label: str) -> Benchmark:
    x = _read_x(x_path)
    y = _read_y(y_path, y_label)
    if len(x) != len(y):
        raise ValueError(
            f"Row-count mismatch for benchmark {name!r}: "
            f"x={len(x)} ({x_path}) vs y={len(y)} ({y_path})"
        )
    log.info("  loaded benchmark %s: x=%s y=%d  (%s, %s)",
             name, x.shape, len(y), x_path.name, y_path.name)
    return Benchmark(name=name, x=x, y=y)


def _assert_consistent_columns(xs: list[pd.DataFrame], where: str) -> None:
    if not xs:
        return
    ref_cols = list(xs[0].columns)
    for df in xs[1:]:
        if list(df.columns) != ref_cols:
            diff_first = next(
                (c for c in df.columns if c not in ref_cols),
                ref_cols[0] if ref_cols else "<none>",
            )
            raise ValueError(
                f"{where} pkls have inconsistent feature columns "
                f"(first offender: {diff_first!r}). All {where} pkls must share "
                f"the same column schema."
            )


def split_per_benchmark(
    train_bms: list[Benchmark],
    test_bms: list[Benchmark],
    train_val_test_ratio: tuple[float, float, float],
    seed: int,
) -> SplitData:
    """Per-benchmark seeded contiguous block cut.

    - testset_x_path empty: cut one (val+test)*N_b block per benchmark at a
      seeded random start; first val_ratio/(val+test) of the block -> val,
      rest -> test.
    - testset_x_path non-empty: cut one (val+test)*N_b block per benchmark
      as val (the entire block). test comes from test_bms in full.
    """
    train_ratio, val_ratio, test_ratio = train_val_test_ratio
    cut_frac = val_ratio + test_ratio
    rng = np.random.default_rng(seed)

    use_external_test = bool(test_bms)
    val_frac_within_cut = (val_ratio / cut_frac) if cut_frac > 0 else 0.0

    train_xs, train_ys = [], []
    val_xs, val_ys = [], []
    internal_test_xs, internal_test_ys = [], []

    for bm in train_bms:
        n = len(bm.x)
        block = int(round(n * cut_frac))
        if block >= n:
            raise ValueError(
                f"benchmark {bm.name!r}: val+test ratio {cut_frac} consumes the "
                f"entire benchmark (block={block} >= n={n})"
            )
        if block == 0:
            train_xs.append(bm.x)
            train_ys.append(bm.y)
            continue

        start = int(rng.integers(0, n - block + 1))
        end = start + block

        train_part_x = pd.concat([bm.x.iloc[:start], bm.x.iloc[end:]], ignore_index=True)
        train_part_y = pd.concat([bm.y.iloc[:start], bm.y.iloc[end:]], ignore_index=True)
        train_xs.append(train_part_x)
        train_ys.append(train_part_y)

        if use_external_test:
            val_xs.append(bm.x.iloc[start:end].reset_index(drop=True))
            val_ys.append(bm.y.iloc[start:end].reset_index(drop=True))
        else:
            val_size = int(round(block * val_frac_within_cut))
            v_end = start + val_size
            val_xs.append(bm.x.iloc[start:v_end].reset_index(drop=True))
            val_ys.append(bm.y.iloc[start:v_end].reset_index(drop=True))
            internal_test_xs.append(bm.x.iloc[v_end:end].reset_index(drop=True))
            internal_test_ys.append(bm.y.iloc[v_end:end].reset_index(drop=True))

        log.info(
            "  split %s: n=%d  train=%d  val=%d  test=%d  cut@[%d,%d)",
            bm.name, n, n - block,
            len(val_xs[-1]),
            len(internal_test_xs[-1]) if internal_test_xs and not use_external_test else 0,
            start, end,
        )

    if use_external_test:
        test_xs = [bm.x for bm in test_bms]
        test_ys = [bm.y for bm in test_bms]
        for bm in test_bms:
            log.info("  external test %s: n=%d", bm.name, len(bm.x))
    else:
        test_xs = internal_test_xs
        test_ys = internal_test_ys

    return SplitData(
        train_x=pd.concat(train_xs, ignore_index=True) if train_xs else _empty_df(train_bms),
        train_y=pd.concat(train_ys, ignore_index=True) if train_ys else _empty_y(),
        val_x=pd.concat(val_xs, ignore_index=True) if val_xs else _empty_df(train_bms),
        val_y=pd.concat(val_ys, ignore_index=True) if val_ys else _empty_y(),
        test_x=pd.concat(test_xs, ignore_index=True) if test_xs else _empty_df(train_bms),
        test_y=pd.concat(test_ys, ignore_index=True) if test_ys else _empty_y(),
    )


def _empty_df(train_bms: list[Benchmark]) -> pd.DataFrame:
    return pd.DataFrame(columns=train_bms[0].x.columns) if train_bms else pd.DataFrame()


def _empty_y() -> pd.Series:
    return pd.Series([], dtype=np.float64)


def avg_window(x: pd.DataFrame, y: pd.Series, wsize: int) -> tuple[pd.DataFrame, pd.Series]:
    """Non-overlapped mean over `wsize` consecutive rows. Drops tail bin."""
    if wsize < 1:
        raise ValueError(f"avg_wsize must be >= 1, got {wsize}")
    if wsize == 1:
        return x.reset_index(drop=True), y.reset_index(drop=True)

    n = len(x)
    n_bins = n // wsize
    if n_bins == 0:
        raise ValueError(
            f"avg_wsize={wsize} larger than dataset rows={n}; tail-drop would empty it."
        )
    n_kept = n_bins * wsize

    x_arr = x.iloc[:n_kept].to_numpy(dtype=np.float64, copy=False)
    y_arr = y.iloc[:n_kept].to_numpy(dtype=np.float64, copy=False)
    x_avg = x_arr.reshape(n_bins, wsize, -1).mean(axis=1)
    y_avg = y_arr.reshape(n_bins, wsize).mean(axis=1)

    return (
        pd.DataFrame(x_avg, columns=x.columns),
        pd.Series(y_avg, name=y.name),
    )


def assert_finite(x: pd.DataFrame, where: str) -> None:
    """Fail loudly on NaN / inf before standardization."""
    arr = x.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(arr).all():
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        raise ValueError(
            f"{where}: {n_nan} NaN and {n_inf} inf cells detected. "
            f"Standardization would propagate these; clean the input pkls first."
        )
