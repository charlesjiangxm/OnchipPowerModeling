"""Loaders for the MATERIALIZED single-bit dataset (produced by ``--build_db``).

Because every ``<bench>_func.pkl.zst`` on disk already holds single-bit uint8
columns in one canonical order, the feature matrix IS the frame -- no registry,
bitmask or bit-expansion happens here. The (small) training benchmarks are read
into a sparse CSR; the (large) test benchmarks are read once and sliced to the
selected proxy columns by name (:class:`Union`).

Window averaging (``data.window_size``): every benchmark is reduced to
non-overlapping ``window_size``-cycle rows *here*, before feature selection and
before the fit, so both stages see the same aggregated design matrix. A feature
becomes the bit's density in the window (a float in [0, 1]; still exactly 0/1 at
``window_size = 1``) and the label becomes the mean power over the window. The
trailing partial window of each benchmark is dropped, and the train/val cut is
taken on whole windows so no window straddles the split.

Row alignment: power traces carry one extra trailing cycle (pwr rows = func rows
+ 1), so the target is reindexed onto the func index -- dropping that surplus row
is mandatory or every label is off by one.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .config import Config
from .selection import ProxyResult, select_proxies
from .utils import log

_FUNC_SUFFIXES = ("_func.pkl.zst", "_func.pkl")
_PWR_SUFFIXES = ("_pwr.pkl.zst", "_pwr.pkl")


def _find(dir_path: str, bench: str, suffixes) -> Path:
    for suf in suffixes:
        p = Path(dir_path) / f"{bench}{suf}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no {suffixes[0]} for benchmark {bench!r} under {dir_path}")


def discover_benches(func_dir: str) -> list[str]:
    """Sorted benchmark names from ``<bench>_func.pkl[.zst]`` files under func_dir."""
    for suf in _FUNC_SUFFIXES:
        benches = sorted(p.name[: -len(suf)] for p in Path(func_dir).glob(f"*{suf}"))
        if benches:
            return benches
    return []


def load_aligned(func_dir: str, pwr_dir: str, bench: str, target: str):
    """Load one benchmark's single-bit func frame and target vector, row-aligned."""
    func = pd.read_pickle(_find(func_dir, bench, _FUNC_SUFFIXES)).sort_index()
    pwr = pd.read_pickle(_find(pwr_dir, bench, _PWR_SUFFIXES)).copy()
    if target not in pwr.columns:
        raise KeyError(f"{bench}: target {target!r} not in pwr columns {list(pwr.columns)}")
    pwr.index = pwr.index.astype("int64")
    y = pwr[target].reindex(func.index.astype("int64")).to_numpy(dtype=np.float64)
    if np.isnan(y).any():
        raise ValueError(f"{bench}: {int(np.isnan(y).sum())} func rows have no matching power label")
    return func, y


def _bits_of(func: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Dense per-cycle 0/1 uint8 matrix of a single-bit frame, canonical order."""
    return func[columns].to_numpy(dtype=np.uint8, copy=False)


def n_windows(n_rows: int, window: int) -> int:
    """Whole non-overlapping windows in ``n_rows`` cycles (partial tail dropped)."""
    return int(n_rows) // int(window)


def window_average(a: np.ndarray, window: int, dtype=None) -> np.ndarray:
    """Mean of every non-overlapping block of ``window`` rows along axis 0.

    Works for the 1-D target and the 2-D feature matrix alike; the trailing
    partial block is dropped. Splitting axis 0 is always expressible as a stride
    change, so the reshape is a view and no copy of ``a`` is materialized -- the
    per-cycle bit matrices are the largest arrays in the pipeline. Accumulating
    in ``dtype`` keeps 0/1 features exact (a window sum <= window). ``window = 1``
    returns the input unchanged (only cast, if ``dtype`` asks for it).
    """
    a = np.asarray(a)
    window = int(window)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window == 1:
        return a if dtype is None else a.astype(dtype, copy=False)
    n_win = n_windows(a.shape[0], window)
    out_dtype = dtype or (a.dtype if a.dtype.kind == "f" else np.float32)
    head = a[: n_win * window]
    return head.reshape((n_win, window) + a.shape[1:]).mean(axis=1, dtype=out_dtype)


@dataclasses.dataclass
class FlatBundle:
    columns: list[str]  # canonical single-bit feature names; index = global col id
    # n_train x n_features; 0/1 uint8 at window_size == 1, else float32 densities
    X_train: sparse.csr_matrix
    y_train: np.ndarray
    X_val: sparse.csr_matrix
    y_val: np.ndarray
    train_slices: dict[str, slice]
    val_slices: dict[str, slice]
    kept_ids: np.ndarray  # global ids passing the variance filter
    sel_rows: np.ndarray  # seeded train-row subsample used for MCP selection
    target: str
    train_benches: list[str]
    test_benches: list[str]
    func_dir: str
    pwr_dir: str
    window_size: int = 1  # cycles averaged into one row (1 -> per-cycle)

    def selection_matrix(self):
        """(CSC float64 over kept columns, y) for the seeded selection subsample."""
        X = self.X_train[self.sel_rows][:, self.kept_ids].tocsc().astype(np.float64)
        return X, self.y_train[self.sel_rows]

    def densify_train(self, col_ids: np.ndarray, dtype=np.float32) -> np.ndarray:
        return np.asarray(self.X_train[:, col_ids].todense(), dtype=dtype)

    def densify_val(self, col_ids: np.ndarray, dtype=np.float32) -> np.ndarray:
        return np.asarray(self.X_val[:, col_ids].todense(), dtype=dtype)

    def names_of(self, col_ids) -> list[str]:
        return [self.columns[int(i)] for i in col_ids]


def _split_rows(n: int, val_fraction: float) -> tuple[int, int]:
    n_val = int(n * val_fraction)
    return n - n_val, n_val


def load_split(cfg: Config, seed: int | None = None) -> FlatBundle:
    """Assemble the in-memory training/validation bundle from the single-bit dataset."""
    func_dir, pwr_dir = cfg.data.func_dir, cfg.data.pwr_dir
    target = cfg.data.target
    window = int(cfg.data.window_size)
    if window < 1:
        raise ValueError(f"data.window_size must be >= 1, got {window}")
    seed = cfg.runtime.seed if seed is None else seed
    test_benches = list(cfg.split.test_benchmarks)

    all_benches = discover_benches(func_dir)
    if not all_benches:
        raise RuntimeError(f"no <bench>_func.pkl[.zst] under {func_dir} (run --build_db first)")
    unknown = set(test_benches) - set(all_benches)
    if unknown:
        raise ValueError(f"test benchmarks not found in {func_dir}: {sorted(unknown)}")
    train_benches = [b for b in all_benches if b not in test_benches]
    if not train_benches:
        raise RuntimeError("no training benchmarks after removing the test set")
    log.info("split: train=%s  test=%s  window_size=%d cycle(s)",
             train_benches, test_benches, window)

    # canonical columns from the first training benchmark (identical across all)
    first = pd.read_pickle(_find(func_dir, train_benches[0], _FUNC_SUFFIXES))
    columns = list(first.columns)
    del first
    n_features = len(columns)

    Xtr_parts, ytr_parts, Xval_parts, yval_parts = [], [], [], []
    train_slices: dict[str, slice] = {}
    val_slices: dict[str, slice] = {}
    tr_off = val_off = 0
    # per-cycle #ones over the training cycles, accumulated before averaging so
    # that min_toggle_count keeps its meaning at any window_size
    col_ones = np.zeros(n_features, dtype=np.int64)
    n_train_cycles = 0
    for b in train_benches:
        func, y = load_aligned(func_dir, pwr_dir, b, target)
        bits = _bits_of(func, columns)
        del func
        n_cyc = bits.shape[0]
        n_win = n_windows(n_cyc, window)
        if n_win == 0:
            raise RuntimeError(
                f"{b}: {n_cyc} cycles is shorter than data.window_size={window} "
                f"- no complete window to average"
            )
        n_tr, n_val = _split_rows(n_win, cfg.split.val_fraction)
        col_ones += bits[: n_tr * window].sum(axis=0, dtype=np.int64)
        n_train_cycles += n_tr * window
        Xb = sparse.csr_matrix(window_average(bits, window, dtype=np.float32)
                               if window > 1 else bits)
        del bits
        yb = window_average(y, window)
        Xtr_parts.append(Xb[:n_tr])
        ytr_parts.append(yb[:n_tr])
        train_slices[b] = slice(tr_off, tr_off + n_tr)
        tr_off += n_tr
        if n_val:
            Xval_parts.append(Xb[n_tr:])
            yval_parts.append(yb[n_tr:])
            val_slices[b] = slice(val_off, val_off + n_val)
            val_off += n_val
        log.info("  %-12s cycles=%d -> rows=%d (train %d / val %d, %d cycles dropped)",
                 b, n_cyc, n_win, n_tr, n_val, n_cyc - n_win * window)

    X_train = sparse.vstack(Xtr_parts, format="csr")
    y_train = np.concatenate(ytr_parts)
    if Xval_parts:
        X_val = sparse.vstack(Xval_parts, format="csr")
        y_val = np.concatenate(yval_parts)
    else:
        X_val = sparse.csr_matrix((0, n_features), dtype=X_train.dtype)
        y_val = np.empty(0, dtype=np.float64)

    # kept bits: variance filter -- drop all-zero AND constant-1 columns, judged
    # on the per-cycle bits, so the column set is stable across window_size (only
    # the handful of tail cycles a window boundary drops can change a count)
    kept = (col_ones >= cfg.data.min_toggle_count) & (col_ones < n_train_cycles)
    n_train = X_train.shape[0]
    if window > 1 and n_train:
        # a bit with the same density in every window survives the count filter
        # yet is constant in the averaged matrix -- and so collinear with the
        # selector's intercept; drop it too
        col_max = np.asarray(X_train.max(axis=0).todense()).ravel()
        col_min = np.asarray(X_train.min(axis=0).todense()).ravel()
        flat = col_max <= col_min
        n_flat = int(np.count_nonzero(flat & kept))
        if n_flat:
            log.info("dropping %d bit(s) constant after %d-cycle averaging", n_flat, window)
        kept &= ~flat
    kept_ids = np.flatnonzero(kept).astype(np.int64)
    log.info("kept bits: %d / %d (0 < ones < %d train cycles, floor=%d)",
             kept_ids.size, n_features, n_train_cycles, cfg.data.min_toggle_count)

    max_rows = cfg.selection.max_rows
    if 0 < max_rows < n_train:
        rng = np.random.default_rng(seed)
        sel_rows = np.sort(rng.choice(n_train, size=max_rows, replace=False)).astype(np.int64)
    else:
        sel_rows = np.arange(n_train, dtype=np.int64)

    return FlatBundle(
        columns=columns, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        train_slices=train_slices, val_slices=val_slices, kept_ids=kept_ids, sel_rows=sel_rows,
        target=target, train_benches=train_benches, test_benches=test_benches,
        func_dir=func_dir, pwr_dir=pwr_dir, window_size=window,
    )


def mcp_select(cfg: Config, bundle: FlatBundle) -> dict[int, ProxyResult]:
    """LR-MCP proxy sets for every target Q, deterministic given the seed."""
    X, y = bundle.selection_matrix()
    log.info("selection matrix: %s over %d kept bits", X.shape, bundle.kept_ids.size)
    return select_proxies(cfg, X, y, bundle.kept_ids, bundle.names_of)


def featurize_test(cfg: Config, columns: list[str], col_ids: np.ndarray):
    """Dense ``n_test x len(col_ids)`` float32 matrix over the selected bits.

    Each test benchmark's single-bit frame is read once and the proxy columns are
    selected by name (robust to any column reordering). Columns are emitted in
    ``col_ids`` order to match the training matrix, and rows are averaged over
    the same ``data.window_size`` cycles as the training matrix.
    """
    col_ids = np.asarray(col_ids, dtype=np.int64)
    names = [columns[int(i)] for i in col_ids]
    window = int(cfg.data.window_size)
    X_parts, y_parts = [], []
    slices: dict[str, slice] = {}
    off = 0
    for b in cfg.split.test_benchmarks:
        func, y = load_aligned(cfg.data.func_dir, cfg.data.pwr_dir, b, cfg.data.target)
        Xb = func[names].to_numpy(dtype=np.float32, copy=False)
        del func
        n_cyc = Xb.shape[0]
        if n_windows(n_cyc, window) == 0:
            raise RuntimeError(
                f"{b}: {n_cyc} cycles is shorter than data.window_size={window} "
                f"- no complete window to average"
            )
        Xb = window_average(Xb, window, dtype=np.float32)
        y = window_average(y, window)
        n = Xb.shape[0]
        X_parts.append(Xb)
        y_parts.append(y)
        slices[b] = slice(off, off + n)
        off += n
        log.info("  test %-12s cycles=%d -> rows=%d over %d proxies",
                 b, n_cyc, n, col_ids.size)
    X_test = np.vstack(X_parts) if X_parts else np.zeros((0, col_ids.size), dtype=np.float32)
    y_test = np.concatenate(y_parts) if y_parts else np.empty(0, dtype=np.float64)
    return X_test, y_test, slices


class Union:
    """Dense train/val/test matrices over the union of all selected proxy ids."""

    def __init__(self, cfg: Config, bundle: FlatBundle, col_id_arrays: list[np.ndarray]):
        self.ids = np.unique(np.concatenate(col_id_arrays))
        self.y_train = bundle.y_train
        self.y_val = bundle.y_val
        self.train_slices = bundle.train_slices
        self.val_slices = bundle.val_slices
        self.Xtr = bundle.densify_train(self.ids)
        self.Xval = bundle.densify_val(self.ids) if bundle.y_val.size else None
        import time
        t0 = time.time()
        self.Xte, self.y_test, self.test_slices = featurize_test(cfg, bundle.columns, self.ids)
        log.info("union test featurized once (%s) in %.1fs", self.Xte.shape, time.time() - t0)

    def slice(self, col_ids: np.ndarray):
        pos = np.searchsorted(self.ids, col_ids)
        assert np.array_equal(self.ids[pos], col_ids), "proxy ids missing from union"
        Xval = self.Xval[:, pos] if self.Xval is not None else None
        return self.Xtr[:, pos], Xval, self.Xte[:, pos]
