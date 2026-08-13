"""Build the on-disk dataset cache from the raw pkl database.

Cache layout (under ``cfg.data.cache_dir``)::

    nets.json                          canonical registry
    bit_stats.npz                      per-benchmark per-bit toggle counts
    features/<bench>/manifest.json     rows, chunks, nnz, missing scopes, hashes
    features/<bench>/index.npy         aligned int64 time_ns
    features/<bench>/labels.parquet    aligned power labels (all columns)
    features/<bench>/<scope>/chunk_00000.npz   CSR, shape (chunk_len, scope_width)

Each scope owns a contiguous global column range (registry order), so a full
feature chunk is the hstack of its scope blocks in scope order; a missing
scope (allowed for test benchmarks only) contributes an all-zero block.
"""

from __future__ import annotations

import gc
import io
import ctypes

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

from ..config import CobitConfig
from ..utils import atomic_write_bytes, load_json, log, save_json
from .bitmask import expand_net_column
from .discovery import DbLayout, discover
from .registry import Registry, build_registry


def _atomic_np_save(path: Path, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, arr)
    atomic_write_bytes(path, buf.getvalue())


def scope_col_range(registry: Registry, scope: str) -> tuple[int, int]:
    nets = registry.nets_of_scope(scope)
    if not nets:
        return (0, 0)
    return nets[0].base_col, nets[-1].base_col + nets[-1].width


SUBSCOPE_PKL_THRESHOLD = 6 * 1024**3  # 6 GB — above this, load sub-scope pkls individually


def _find_subscope_pkls(layout: DbLayout, scope: str, bench: str) -> list[Path]:
    """Find sub-scope pkl files for a given scope.

    Each first-level module directory (e.g. aq_core/ifu/x_aq_ifu_icache/)
    contains a per-bench pkl with a subset of the merged scope pkl's columns.
    """
    if scope == "top":
        return []
    core = Path(layout.db_root) / "aq_core"
    scope_dir = core / scope
    if not scope_dir.is_dir():
        return []
    sub_pkls: list[Path] = []
    for sub_dir in sorted(scope_dir.iterdir()):
        if sub_dir.is_dir():
            pkl = sub_dir / f"{bench}_func.pkl"
            if pkl.exists():
                sub_pkls.append(pkl)
    return sub_pkls


def _build_scope_from_subscope_pkls(
    cfg: CobitConfig,
    registry: Registry,
    bench: str,
    scope: str,
    sub_pkls: list[Path],
    ref_index: pd.Index,
    chunks_meta: list[dict],
    counts: np.ndarray,
    bdir: Path,
) -> int:
    """Build scope features from individual sub-scope pkls.

    Loads each sub-scope pkl one at a time, accumulating per-chunk COO
    triplets, then builds CSR matrices.  This avoids the memory spike
    from deserialising a large merged scope pkl in one shot.
    """
    s_lo, s_hi = scope_col_range(registry, scope)
    scope_dir = bdir / scope
    scope_dir.mkdir(exist_ok=True)
    nnz_scope = 0
    bit_expand = cfg.data.bit_expand
    data_dtype = np.uint8 if bit_expand else np.float64
    nets = registry.nets_of_scope(scope)
    cols_expected = {n.column for n in nets}
    all_cols_seen: set[str] = set()

    chunk_rows: list[list[np.ndarray]] = [[] for _ in chunks_meta]
    chunk_cols: list[list[np.ndarray]] = [[] for _ in chunks_meta]
    chunk_data: list[list[np.ndarray]] = [[] for _ in chunks_meta]

    for sub_pkl in sub_pkls:
        df = pd.read_pickle(sub_pkl)
        df.index = df.index.astype(np.int64)
        inter = df.index.intersection(ref_index)
        if len(inter) != len(ref_index):
            raise RuntimeError(
                f"{bench}/{scope}: sub-scope {sub_pkl.parent.name} "
                f"index disagrees with reference index "
                f"({len(inter)}/{len(ref_index)} rows shared)"
            )
        keep_pos = df.index.get_indexer(ref_index)
        pkl_cols = set(df.columns)
        sub_nets = [n for n in nets if n.column in pkl_cols]
        all_cols_seen.update(n.column for n in sub_nets)
        for cm_idx, cm in enumerate(chunks_meta):
            pos = keep_pos[cm["start"]: cm["stop"]]
            for net in sub_nets:
                vals = df[net.column].to_numpy()[pos]
                if bit_expand:
                    expand_net_column(
                        vals, net.width, net.base_col - s_lo, 0,
                        chunk_rows[cm_idx], chunk_cols[cm_idx],
                        context=f"{bench}/{scope}/{net.column}",
                    )
                else:
                    nz = np.flatnonzero(vals != 0)
                    if nz.size == 0:
                        continue
                    chunk_rows[cm_idx].append(nz.astype(np.int64))
                    chunk_cols[cm_idx].append(
                        np.full(nz.size, net.base_col - s_lo, dtype=np.int64)
                    )
                    chunk_data[cm_idx].append(
                        np.asarray(vals[nz], dtype=np.float64)
                    )
        del df
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass

    if all_cols_seen != cols_expected:
        missing_count = len(cols_expected - all_cols_seen)
        missing_sample = sorted(cols_expected - all_cols_seen)[:5]
        log.warning(
            "features/%s/%s: %d/%d columns not in sub-scope pkls - zero-filled "
            "(missing e.g. %s)",
            bench, scope, missing_count, len(cols_expected), missing_sample,
        )

    for cm_idx, cm in enumerate(chunks_meta):
        n_local = cm["stop"] - cm["start"]
        if chunk_rows[cm_idx]:
            r = np.concatenate(chunk_rows[cm_idx])
            c = np.concatenate(chunk_cols[cm_idx])
            if bit_expand:
                data = np.ones(r.size, dtype=np.uint8)
            else:
                data = np.concatenate(chunk_data[cm_idx])
            mat = sparse.coo_matrix(
                (data, (r, c)), shape=(n_local, s_hi - s_lo)
            ).tocsr()
            np.add.at(counts, s_lo + c, 1)
            del r, c, data
        else:
            mat = sparse.csr_matrix((n_local, s_hi - s_lo), dtype=data_dtype)
        sparse.save_npz(scope_dir / f"chunk_{cm['chunk']:05d}.npz", mat)
        nnz_scope += int(mat.nnz)

    del chunk_rows, chunk_cols, chunk_data
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    log.info(
        "features/%s/%s built from %d sub-scope pkls (%d nnz)",
        bench, scope, len(sub_pkls), nnz_scope,
    )
    return nnz_scope


def _source_stamp(layout: DbLayout, bench: str) -> dict:
    stamp = {}
    for scope in layout.coverage[bench]:
        p = layout.func_pkl(scope, bench)
        st = p.stat()
        stamp[scope] = [st.st_size, int(st.st_mtime)]
    p = layout.pwr_pkl(bench)
    st = p.stat()
    stamp["__pwr__"] = [st.st_size, int(st.st_mtime)]
    return stamp


def build_dataset(cfg: CobitConfig, force: bool = False) -> tuple[DbLayout, Registry]:
    layout = discover(cfg.data.db_root)
    cache = Path(cfg.data.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    reg_path = cache / "nets.json"
    if reg_path.exists() and not force:
        registry = Registry.load(reg_path)
        fresh = build_registry(layout, bit_expand=cfg.data.bit_expand)
        if fresh.content_hash != registry.content_hash:
            raise RuntimeError(
                "cached registry does not match the database - rerun with --force "
                f"({reg_path})"
            )
        if fresh.bit_expand != registry.bit_expand:
            raise RuntimeError(
                "cached registry bit_expand mode does not match config - rerun with --force "
                f"({reg_path})"
            )
    else:
        registry = build_registry(layout, bit_expand=cfg.data.bit_expand)
        registry.save(reg_path)

    stats: dict[str, np.ndarray] = {}
    for bench in layout.benchmarks:
        counts = _build_benchmark(cfg, layout, registry, bench, force=force)
        stats[bench] = counts

    # bit_stats.npz: one uint64 count vector per benchmark (assembled from the
    # per-benchmark counts.npy files, written atomically so an interrupted
    # build never leaves a truncated archive behind)
    buf = io.BytesIO()
    np.savez_compressed(buf, **stats)
    atomic_write_bytes(cache / "bit_stats.npz", buf.getvalue())
    log.info("dataset cache complete: %d benchmarks under %s", len(stats), cache)
    return layout, registry


def _build_benchmark(
    cfg: CobitConfig, layout: DbLayout, registry: Registry, bench: str, force: bool
) -> np.ndarray:
    cache = Path(cfg.data.cache_dir)
    bdir = cache / "features" / bench
    manifest_path = bdir / "manifest.json"
    stamp = _source_stamp(layout, bench)

    counts_path = bdir / "counts.npy"
    if manifest_path.exists() and not force:
        manifest = load_json(manifest_path)
        if (
            manifest.get("registry_hash") == registry.content_hash
            and manifest.get("chunk_rows") == cfg.data.chunk_rows
            and manifest.get("source_stamp") == stamp
        ):
            # counts.npy is written together with the manifest, so a fresh
            # manifest implies it exists; guard anyway so a damaged cache
            # falls through to a rebuild instead of crashing every run
            if counts_path.exists():
                try:
                    log.info("features/%s up to date - skipped", bench)
                    return np.load(counts_path).astype(np.uint64)
                except (OSError, ValueError):
                    pass
        log.info("features/%s stale - rebuilding", bench)

    bdir.mkdir(parents=True, exist_ok=True)

    # --- labels first: they define the authoritative row index -------------
    pwr = pd.read_pickle(layout.pwr_pkl(bench))
    pwr.index = pwr.index.astype(np.int64)
    if not pwr.index.is_monotonic_increasing or pwr.index.has_duplicates:
        raise RuntimeError(f"pwr index of {bench} not monotonic/unique")

    # Reference func index from the first available scope.
    ref_index: pd.Index | None = None
    missing_scopes = layout.missing_scopes(bench)
    chunk_rows = int(cfg.data.chunk_rows)
    counts = np.zeros(registry.n_features, dtype=np.uint64)
    scope_nnz: dict[str, int] = {}
    chunks_meta: list[dict] = []

    for scope in layout.scopes:
        if scope in missing_scopes:
            continue
        merged_pkl = layout.func_pkl(scope, bench)
        sub_pkls = _find_subscope_pkls(layout, scope, bench)
        if (
            merged_pkl.stat().st_size > SUBSCOPE_PKL_THRESHOLD
            and sub_pkls
            and ref_index is not None
        ):
            scope_nnz[scope] = _build_scope_from_subscope_pkls(
                cfg, registry, bench, scope, sub_pkls,
                ref_index, chunks_meta, counts, bdir,
            )
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except (OSError, AttributeError):
                pass
            continue
        df = pd.read_pickle(layout.func_pkl(scope, bench))
        df.index = df.index.astype(np.int64)
        if ref_index is None:
            if not df.index.is_monotonic_increasing or df.index.has_duplicates:
                raise RuntimeError(f"func index of {bench}/{scope} not monotonic/unique")
            joined = df.index.intersection(pwr.index)
            dropped = len(df.index) - len(joined)
            if dropped:
                log.warning(
                    "%s: %d func rows have no power label - dropped", bench, dropped
                )
            ref_index = joined
            keep_pos = df.index.get_indexer(joined)
            n_rows = len(joined)
            n_chunks = max(1, -(-n_rows // chunk_rows))
            chunks_meta = [
                {"chunk": c, "start": c * chunk_rows,
                 "stop": min((c + 1) * chunk_rows, n_rows)}
                for c in range(n_chunks)
            ]
        else:
            # every scope of a benchmark must cover the reference cycle index
            inter = df.index.intersection(ref_index)
            if len(inter) != len(ref_index):
                raise RuntimeError(
                    f"{bench}/{scope}: func index disagrees with reference index "
                    f"({len(inter)}/{len(ref_index)} rows shared)"
                )
            keep_pos = df.index.get_indexer(ref_index)

        nets = registry.nets_of_scope(scope)
        cols_present = set(df.columns)
        cols_expected = {n.column for n in nets}
        if cols_present != cols_expected:
            missing = sorted(cols_expected - cols_present)[:5]
            extra = sorted(cols_present - cols_expected)[:5]
            raise RuntimeError(
                f"{bench}/{scope}: columns differ from registry "
                f"(missing e.g. {missing}, extra e.g. {extra})"
            )

        s_lo, s_hi = scope_col_range(registry, scope)
        scope_dir = bdir / scope
        scope_dir.mkdir(exist_ok=True)
        nnz_scope = 0
        # Extract each column inline per chunk rather than pre-materializing an
        # arrays dict: holding all columns as separate numpy arrays duplicates
        # ~n_rows*n_cols*8 bytes of pointers (~30 GB for coremark/vpu), which
        # alone can exceed node memory on large scopes.
        bit_expand = cfg.data.bit_expand
        data_dtype = np.uint8 if bit_expand else np.float64
        for cm in chunks_meta:
            pos = keep_pos[cm["start"]: cm["stop"]]
            rows_out: list[np.ndarray] = []
            cols_out: list[np.ndarray] = []
            data_vals: list[np.ndarray] = []
            for net in nets:
                vals = df[net.column].to_numpy()[pos]
                if bit_expand:
                    expand_net_column(
                        vals, net.width, net.base_col - s_lo, 0, rows_out, cols_out,
                        context=f"{bench}/{scope}/{net.column}",
                    )
                else:
                    nz = np.flatnonzero(vals != 0)
                    if nz.size == 0:
                        continue
                    rows_out.append(nz.astype(np.int64))
                    cols_out.append(
                        np.full(nz.size, net.base_col - s_lo, dtype=np.int64)
                    )
                    data_vals.append(np.asarray(vals[nz], dtype=np.float64))
            n_local = len(pos)
            if rows_out:
                # Free the per-net COO lists before building the CSR so the
                # list-of-arrays and the concatenated arrays do not coexist
                # (the dominant peak on wide scopes like vpu).
                r = np.concatenate(rows_out)
                del rows_out
                c = np.concatenate(cols_out)
                del cols_out
                if bit_expand:
                    data = np.ones(r.size, dtype=np.uint8)
                else:
                    data = np.concatenate(data_vals)
                    del data_vals
                mat = sparse.coo_matrix(
                    (data, (r, c)), shape=(n_local, s_hi - s_lo)
                ).tocsr()
                np.add.at(counts, s_lo + c, 1)
                del r, c, data
            else:
                mat = sparse.csr_matrix((n_local, s_hi - s_lo), dtype=data_dtype)
            sparse.save_npz(scope_dir / f"chunk_{cm['chunk']:05d}.npz", mat)
            nnz_scope += int(mat.nnz)
        scope_nnz[scope] = nnz_scope
        del df
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass

    assert ref_index is not None, f"{bench}: no scope pkl could be read"

    labels = pwr.loc[ref_index]
    lbuf = io.BytesIO()
    labels.to_parquet(lbuf)
    atomic_write_bytes(bdir / "labels.parquet", lbuf.getvalue())
    _atomic_np_save(bdir / "index.npy", ref_index.to_numpy(dtype=np.int64))
    _atomic_np_save(bdir / "counts.npy", counts)

    save_json(
        manifest_path,
        {
            "benchmark": bench,
            "n_rows": int(len(ref_index)),
            "chunk_rows": chunk_rows,
            "chunks": chunks_meta,
            "scopes": layout.coverage[bench],
            "missing_scopes": missing_scopes,
            "scope_nnz": scope_nnz,
            "registry_hash": registry.content_hash,
            "source_stamp": stamp,
            "label_columns": list(labels.columns),
        },
    )
    total_nnz = sum(scope_nnz.values())
    log.info(
        "features/%s: %d rows, %d nnz (density %.4f%%)%s",
        bench,
        len(ref_index),
        total_nnz,
        100.0 * total_nnz / max(1, len(ref_index) * registry.n_features),
        f", ZERO-FILLED scopes: {missing_scopes}" if missing_scopes else "",
    )
    return counts
