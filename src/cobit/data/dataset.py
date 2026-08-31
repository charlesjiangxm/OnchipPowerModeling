"""Split-aware, column-restricted loading of the cached dataset.

The heavy artifacts stay sparse on disk; callers choose which global feature
columns to materialize:

- Stage 1 (proxy selection) loads the kept (ever-toggling) columns as sparse.
- Stage 2 (HPO / training) loads only the Q proxy columns, densified.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ..config import CobitConfig
from ..utils import load_json, log
from .discovery import discover, plan_benchmarks
from .registry import Registry
from .build import scope_col_range


@dataclasses.dataclass
class Split:
    """Row selections per benchmark for one of train/val/test."""

    benchmarks: list[str]
    row_slices: dict[str, slice]  # per-bench rows (into the bench's own rows)


@dataclasses.dataclass
class DatasetBundle:
    """Materialized matrices for one experiment split."""

    X_train: sparse.csr_matrix | np.ndarray
    y_train: np.ndarray
    X_val: sparse.csr_matrix | np.ndarray
    y_val: np.ndarray
    X_test: sparse.csr_matrix | np.ndarray
    y_test: np.ndarray
    col_ids: np.ndarray  # global feature ids of the columns of X_*
    feature_names: list[str]
    # bench -> slice into the concatenated rows of the corresponding X_*
    train_slices: dict[str, slice]
    val_slices: dict[str, slice]
    test_slices: dict[str, slice]
    target: str
    registry_hash: str


class DatasetCache:
    """Read-side view of the artifacts written by :mod:`cobit.data.build`."""

    def __init__(self, cfg: CobitConfig):
        self.cfg = cfg
        self.cache = Path(cfg.data.cache_dir)
        self.registry = Registry.load(self.cache / "nets.json")
        self.layout = discover(cfg.data.db_root)
        self.train_benches, self.test_benches = plan_benchmarks(
            self.layout, cfg.split.test_benchmarks
        )

    # -- manifests -----------------------------------------------------------
    def manifest(self, bench: str) -> dict:
        return load_json(self.cache / "features" / bench / "manifest.json")

    @property
    def _data_dtype(self):
        return np.uint8 if self.cfg.data.bit_expand else np.float64

    @property
    def _dense_dtype(self):
        return np.float32 if self.cfg.data.bit_expand else np.float64

    def n_rows(self, bench: str) -> int:
        return int(self.manifest(bench)["n_rows"])

    def labels(self, bench: str, rows: slice | None = None) -> pd.DataFrame:
        df = pd.read_parquet(self.cache / "features" / bench / "labels.parquet")
        return df.iloc[rows] if rows is not None else df

    def data_stamp(self, benches: list[str] | None = None) -> str:
        """Content stamp of the source pkls behind the given benchmarks.

        Folded into stage artifacts so re-dumped DBs (same schema, new data)
        invalidate proxies and per-Q results, not just the feature cache.
        """
        from ..utils import stable_hash

        benches = benches if benches is not None else (self.train_benches + self.test_benches)
        return stable_hash({b: self.manifest(b)["source_stamp"] for b in benches})

    # -- kept bits ------------------------------------------------------------
    def toggle_counts(self, benches: list[str]) -> np.ndarray:
        total = np.zeros(self.registry.n_features, dtype=np.uint64)
        for b in benches:
            counts_path = self.cache / "features" / b / "counts.npy"
            if counts_path.exists():
                total += np.load(counts_path).astype(np.uint64)
            else:  # caches built before counts.npy existed
                with np.load(self.cache / "bit_stats.npz") as z:
                    total += z[b].astype(np.uint64)
        return total

    def kept_ids(self) -> np.ndarray:
        """Global ids of bits that toggle in the TRAINING benchmarks."""
        counts = self.toggle_counts(self.train_benches)
        mask = counts >= self.cfg.data.min_toggle_count
        ids = np.flatnonzero(mask)
        log.info(
            "kept bits: %d / %d toggle >=%d times in training benchmarks",
            ids.size, self.registry.n_features, self.cfg.data.min_toggle_count,
        )
        return ids

    # -- feature loading --------------------------------------------------------
    def load_features(
        self,
        bench: str,
        col_ids: np.ndarray,
        rows: slice | None = None,
        dense: bool = False,
    ) -> sparse.csr_matrix | np.ndarray:
        """Load one benchmark's feature rows restricted to ``col_ids``."""
        manifest = self.manifest(bench)
        n_rows = manifest["n_rows"]
        rows = rows if rows is not None else slice(0, n_rows)
        start, stop, _ = rows.indices(n_rows)

        col_ids = np.asarray(col_ids, dtype=np.int64)
        # map global col ids into per-scope local ids once
        scope_plans = []
        for scope in self.layout.scopes:
            s_lo, s_hi = scope_col_range(self.registry, scope)
            in_scope = np.flatnonzero((col_ids >= s_lo) & (col_ids < s_hi))
            if in_scope.size:
                scope_plans.append((scope, s_lo, in_scope))

        out_blocks: list[sparse.csr_matrix] = []
        bdir = self.cache / "features" / bench
        for cm in manifest["chunks"]:
            c_start, c_stop = cm["start"], cm["stop"]
            if c_stop <= start or c_start >= stop:
                continue
            lo = max(start, c_start) - c_start
            hi = min(stop, c_stop) - c_start
            n_local = hi - lo
            block = sparse.csr_matrix((n_local, col_ids.size), dtype=self._data_dtype)
            parts: list[tuple[np.ndarray, sparse.csr_matrix]] = []
            for scope, s_lo, sel_positions in scope_plans:
                shard_path = bdir / scope / f"chunk_{cm['chunk']:05d}.npz"
                if not shard_path.exists():
                    continue  # zero-filled scope
                shard = sparse.load_npz(shard_path)
                local_cols = col_ids[sel_positions] - s_lo
                sub = shard[lo:hi].tocsc()[:, local_cols].tocsr()
                parts.append((sel_positions, sub))
            if parts:
                # scatter the scope sub-blocks into their output column positions
                rows_all, cols_all, data_all = [], [], []
                for sel_positions, sub in parts:
                    coo = sub.tocoo()
                    rows_all.append(coo.row)
                    cols_all.append(sel_positions[coo.col])
                    data_all.append(coo.data)
                block = sparse.coo_matrix(
                    (
                        np.concatenate(data_all),
                        (np.concatenate(rows_all), np.concatenate(cols_all)),
                    ),
                    shape=(n_local, col_ids.size),
                ).tocsr()
            out_blocks.append(block)

        X = sparse.vstack(out_blocks, format="csr") if out_blocks else sparse.csr_matrix(
            (0, col_ids.size), dtype=self._data_dtype
        )
        return np.asarray(X.todense(), dtype=self._dense_dtype) if dense else X

    # -- split assembly ---------------------------------------------------------
    def split_rows(self, bench: str) -> tuple[slice, slice]:
        """(train_rows, val_rows) for a training benchmark: contiguous tail val."""
        n = self.n_rows(bench)
        frac = self.cfg.split.val_fraction
        if frac == 0.0:  # explicit "no validation split"
            return slice(0, n), slice(n, n)
        n_val = int(round(n * frac))
        n_val = min(max(n_val, 1), n - 1) if n >= 2 else 0
        return slice(0, n - n_val), slice(n - n_val, n)

    def load_split(
        self, col_ids: np.ndarray, dense: bool = False
    ) -> DatasetBundle:
        target = self.cfg.data.target

        def _gather(benches: list[str], which: str):
            mats, ys, slices, cursor = [], [], {}, 0
            for b in benches:
                if which == "train":
                    rows, _ = self.split_rows(b)
                elif which == "val":
                    _, rows = self.split_rows(b)
                else:
                    rows = None
                X = self.load_features(b, col_ids, rows=rows, dense=dense)
                y = self.labels(b, rows=rows)[target].to_numpy(dtype=np.float64)
                if X.shape[0] != y.size:
                    raise RuntimeError(f"{b}: X rows {X.shape[0]} != y rows {y.size}")
                mats.append(X)
                ys.append(y)
                slices[b] = slice(cursor, cursor + y.size)
                cursor += y.size
            if dense:
                Xall = np.vstack(mats) if mats else np.zeros((0, col_ids.size), self._dense_dtype)
            else:
                Xall = (
                    sparse.vstack(mats, format="csr")
                    if mats
                    else sparse.csr_matrix((0, col_ids.size), dtype=self._data_dtype)
                )
            yall = np.concatenate(ys) if ys else np.zeros(0)
            return Xall, yall, slices

        X_train, y_train, tr_sl = _gather(self.train_benches, "train")
        X_val, y_val, va_sl = _gather(self.train_benches, "val")
        X_test, y_test, te_sl = _gather(self.test_benches, "test")

        col_ids = np.asarray(col_ids, dtype=np.int64)
        return DatasetBundle(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            col_ids=col_ids,
            feature_names=self.registry.feature_names(col_ids),
            train_slices=tr_sl, val_slices=va_sl, test_slices=te_sl,
            target=target,
            registry_hash=self.registry.content_hash,
        )

    def selection_matrix(
        self, kept: np.ndarray, max_rows: int, seed: int
    ) -> tuple[sparse.csc_matrix, np.ndarray]:
        """Sparse (rows x kept-bits) training matrix for Stage 1.

        Rows are subsampled to ``max_rows`` with contiguous seeded blocks
        allocated proportionally per benchmark (train rows only, val excluded).
        """
        rng = np.random.default_rng(seed)
        per_bench: list[tuple[str, slice]] = []
        lens = []
        for b in self.train_benches:
            tr, _ = self.split_rows(b)
            per_bench.append((b, tr))
            lens.append(tr.stop - tr.start)
        total = sum(lens)
        budget = min(max_rows, total)
        mats, ys = [], []
        for (b, tr), n in zip(per_bench, lens):
            take = max(1, int(round(budget * n / total)))
            take = min(take, n)
            start = tr.start + (
                int(rng.integers(0, n - take + 1)) if n > take else 0
            )
            rows = slice(start, start + take)
            mats.append(self.load_features(b, kept, rows=rows))
            ys.append(self.labels(b, rows=rows)[self.cfg.data.target].to_numpy(float))
        X = sparse.vstack(mats, format="csc")
        del mats
        X = X.astype(np.float64)
        y = np.concatenate(ys)
        log.info("selection matrix: %d x %d, nnz=%d", X.shape[0], X.shape[1], X.nnz)
        return X, y


def load_bundle(cfg: CobitConfig, col_ids: np.ndarray, dense: bool = False) -> DatasetBundle:
    return DatasetCache(cfg).load_split(col_ids, dense=dense)


# -- multicycle aggregation ------------------------------------------------------


def aggregate_windows(
    X: np.ndarray | sparse.spmatrix,
    y: np.ndarray,
    bench_slices: dict[str, slice],
    tau: int,
    mode: str = "sum",
) -> tuple[np.ndarray | sparse.spmatrix, np.ndarray, dict[str, slice]]:
    """Aggregate rows into non-overlapping tau-cycle windows per benchmark.

    Features: per-window toggle ``sum`` (default; hardware analog = small
    counters), ``mean`` (toggle rate) or ``any`` (binary). Labels: window mean.
    Tail cycles that do not fill a window are dropped.
    """
    x_parts, y_parts, out_slices, cursor = [], [], {}, 0
    for bench, sl in bench_slices.items():
        n = sl.stop - sl.start
        n_win = n // tau
        if n_win == 0:
            continue
        Xb = X[sl.start : sl.start + n_win * tau]
        yb = y[sl.start : sl.start + n_win * tau]
        if sparse.issparse(Xb):
            agg = sparse.kron(
                sparse.eye(n_win, format="csr"),
                sparse.csr_matrix(np.ones((1, tau))),
                format="csr",
            )
            Xw = agg @ Xb
            if mode == "mean":
                Xw = Xw / tau
            elif mode == "any":
                Xw = (Xw > 0).astype(np.uint8)
        else:
            Xw = Xb.reshape(n_win, tau, -1).sum(axis=1)
            if mode == "mean":
                Xw = Xw / tau
            elif mode == "any":
                Xw = (Xw > 0).astype(np.float32)
        x_parts.append(Xw)
        y_parts.append(yb.reshape(n_win, tau).mean(axis=1))
        out_slices[bench] = slice(cursor, cursor + n_win)
        cursor += n_win
    if not x_parts:
        empty_x = X[:0]
        return empty_x, y[:0], {}
    Xout = sparse.vstack(x_parts, format="csr") if sparse.issparse(X) else np.vstack(x_parts)
    return Xout, np.concatenate(y_parts), out_slices
