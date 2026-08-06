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

import io

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
        fresh = build_registry(layout)
        if fresh.content_hash != registry.content_hash:
            raise RuntimeError(
                "cached registry does not match the database - rerun with --force "
                f"({reg_path})"
            )
    else:
        registry = build_registry(layout)
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
        for cm in chunks_meta:
            pos = keep_pos[cm["start"]: cm["stop"]]
            rows_out: list[np.ndarray] = []
            cols_out: list[np.ndarray] = []
            for net in nets:
                vals = df[net.column].to_numpy()[pos]
                expand_net_column(
                    vals, net.width, net.base_col - s_lo, 0, rows_out, cols_out,
                    context=f"{bench}/{scope}/{net.column}",
                )
            n_local = len(pos)
            if rows_out:
                # Free the per-net COO lists before building the CSR so the
                # list-of-arrays and the concatenated arrays do not coexist
                # (the dominant peak on wide scopes like vpu).
                r = np.concatenate(rows_out)
                del rows_out
                c = np.concatenate(cols_out)
                del cols_out
                data = np.ones(r.size, dtype=np.uint8)
                mat = sparse.coo_matrix(
                    (data, (r, c)), shape=(n_local, s_hi - s_lo)
                ).tocsr()
                np.add.at(counts, s_lo + c, 1)
                del r, c, data
            else:
                mat = sparse.csr_matrix((n_local, s_hi - s_lo), dtype=np.uint8)
            sparse.save_npz(scope_dir / f"chunk_{cm['chunk']:05d}.npz", mat)
            nnz_scope += int(mat.nnz)
        scope_nnz[scope] = nnz_scope
        del df

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
