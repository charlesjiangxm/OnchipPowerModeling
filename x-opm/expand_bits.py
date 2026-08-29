#!/usr/bin/env python
"""Bit-expand the C906 func feature pickles into single-bit columns.

Interpreter: ``~/anaconda3/bin/python`` (matches the rest of x-opm; invoked by
absolute path, no ``conda activate``).

Each ``*_func.pkl`` in the source directory stores multi-bit RTL signals as a
single column holding the full integer bus value -- e.g. a column named
``x_aq_cp0_top/cp0_biu_lpmd_b[1:0]`` whose cell value ``3`` means ``2'b11``.
This script produces a *derived* dataset where every such multi-bit signal is
exploded into one ``uint8`` column per bit (``...lpmd_b[1]``, ``...lpmd_b[0]``),
while genuine single-bit signals (bare path names, no ``[a:b]`` suffix) pass
through unchanged.  Rows and the ``time_ns`` index are preserved 1:1.

The bit split reuses the tested pipeline utilities:
  - ``schema.parse_column``  -> ``(path, width, lo)``
  - ``transform.split_bits`` -> ``[(name, uint8_array)]`` (handles >64-bit buses)
  - ``io_utils.save_pickle`` -> atomic ``pd.to_pickle`` + ``os.replace``

Each input pickle is converted in its own OS process so the whole directory
converts in parallel (one process per file).

Usage (from the repo root)::

    ~/anaconda3/bin/python x-opm/expand_bits.py \
        [--src-dir c906_db_net_1cyc_20260729/aq_core/cp0] \
        [--dst-dir c906_db_net_1cyc_20260729/aq_core/cp0_bits] \
        [--workers N] [--cases csr coremark ...] [--overwrite]
"""

from __future__ import annotations

import argparse
import glob
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# Repo convention: make sibling modules importable regardless of CWD (run.py:23).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from io_utils import ensure_dir, save_pickle  # noqa: E402
from schema import parse_column  # noqa: E402
from transform import split_bits  # noqa: E402

log = logging.getLogger("expand_bits")

_SUFFIX = "_func.pkl.zst"


def discover_cases(src_dir: str, cases_filter: list[str] | None = None) -> list[str]:
    """Return sorted case stems for ``*_func.pkl`` directly under ``src_dir``.

    Non-recursive on purpose: picks exactly the ``<case>_func.pkl`` files and
    skips the hierarchy subdirectories (``x_aq_cp0_{iui,regs,special}/``).
    """
    paths = glob.glob(os.path.join(src_dir, "*" + _SUFFIX))
    stems = sorted(os.path.basename(p)[: -len(_SUFFIX)] for p in paths)
    if cases_filter:
        wanted = list(dict.fromkeys(cases_filter))  # de-dup, keep order
        present = set(stems)
        missing = [c for c in wanted if c not in present]
        if missing:
            raise SystemExit(
                f"--cases not found in {src_dir}: {missing} (available: {stems})"
            )
        stems = [c for c in stems if c in set(wanted)]
    return stems


def default_dst_dir(src_dir: str) -> str:
    """``.../aq_core/cp0`` -> sibling ``.../aq_core/cp0_bits``."""
    src = os.path.abspath(src_dir).rstrip(os.sep)
    return os.path.join(os.path.dirname(src), os.path.basename(src) + "_bits")


def transform_df(df: pd.DataFrame, msb_first: bool = True) -> pd.DataFrame:
    """Expand every column to single-bit ``uint8``; preserve index and order.

    Multi-bit ``<path>[HIGH:LOW]`` -> columns ``<path>[HIGH]``..``<path>[LOW]``;
    single-bit (bare) columns keep their name and value (normalized to uint8).
    Uses one preallocated 2-D buffer (avoids pandas block-consolidation copies on
    the big files) and an explicit column list (so any name collision fails loud
    instead of being silently collapsed as a dict would).
    """
    n = len(df)
    plans = [(col, *parse_column(col)) for col in df.columns]  # (col, path, width, lo)
    ncols = sum(width for _, _, width, _ in plans)
    buf = np.empty((n, ncols), dtype=np.uint8)
    names: list[str] = []
    j = 0
    for col, path, width, lo in plans:
        vals = df[col].to_numpy()  # object array of Python ints
        if width == 1:
            # Bare names are 1-bit RTL nets (values 0/1). split_bits would take
            # v & 1, so a value >1 would be silently truncated -- surface it.
            uniq = np.unique(vals)
            bad = [int(v) for v in uniq if int(v) not in (0, 1)]
            if bad:
                log.warning("scalar column %r has non-binary values %s", col, bad[:3])
        try:
            pairs = split_bits(vals, path, width, lo)  # LSB-first, uint8
        except Exception as exc:  # noqa: BLE001 - re-raise with the offending column
            raise ValueError(f"split_bits failed on column {col!r}") from exc
        if msb_first and width > 1:
            pairs = pairs[::-1]  # [HIGH]..[0] to mirror the [HIGH:LOW] name order
        for name, arr in pairs:
            names.append(name)
            buf[:, j] = arr
            j += 1
    assert j == ncols, f"filled {j} columns, expected {ncols}"
    if len(set(names)) != len(names):
        raise ValueError("duplicate expanded column name detected")
    out = pd.DataFrame(buf, index=df.index, columns=names, copy=False)
    out.index.name = df.index.name  # keep "time_ns"
    return out


def expand_one(src_path: str, dst_path: str, msb_first: bool, overwrite: bool) -> dict:
    """Top-level (picklable) worker: read -> transform -> atomic save -> stats."""
    t0 = time.time()
    case = os.path.basename(src_path)[: -len(_SUFFIX)]
    if os.path.exists(dst_path) and not overwrite:
        return {"case": case, "status": "skipped", "dst": dst_path}
    df = pd.read_pickle(src_path)
    in_shape = tuple(df.shape)
    n_multibit = sum(1 for c in df.columns if parse_column(c)[1] > 1)
    out = transform_df(df, msb_first=msb_first)
    out_shape = tuple(out.shape)
    del df
    save_pickle(out, dst_path)
    return {
        "case": case,
        "status": "ok",
        "in_shape": in_shape,
        "out_shape": out_shape,
        "n_multibit": n_multibit,
        "elapsed": round(time.time() - t0, 1),
        "dst": dst_path,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--src-dir",
        default="dataset/c906_db_net_1cyc_20260729/aq_core/cp0",
        help="directory holding the <case>_func.pkl.zst inputs",
    )
    p.add_argument(
        "--dst-dir",
        default=None,
        help="output directory (default: sibling <src>_bits, e.g. .../cp0_bits)",
    )
    p.add_argument("--workers", type=int, default=None, help="parallel processes (default: min(#files, cpus))")
    p.add_argument("--cases", nargs="+", default=None, help="subset of case stems to convert")
    p.add_argument("--overwrite", action="store_true", help="rewrite outputs that already exist")
    p.add_argument(
        "--lsb-first",
        action="store_true",
        help="emit bit columns [0]..[HIGH] instead of the default [HIGH]..[0]",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    src_dir = os.path.abspath(args.src_dir)
    if not os.path.isdir(src_dir):
        raise SystemExit(f"--src-dir not a directory: {src_dir}")
    dst_dir = os.path.abspath(args.dst_dir) if args.dst_dir else default_dst_dir(src_dir)
    ensure_dir(dst_dir)
    msb_first = not args.lsb_first

    cases = discover_cases(src_dir, args.cases)
    if not cases:
        raise SystemExit(f"no *{_SUFFIX} files found in {src_dir}")

    # Largest file first so the long pole (conv_softmax) starts at t=0.
    def _src(case: str) -> str:
        return os.path.join(src_dir, case + _SUFFIX)

    cases.sort(key=lambda c: os.path.getsize(_src(c)), reverse=True)
    workers = args.workers or min(len(cases), os.cpu_count() or 1)

    log.info(
        "expanding %d case(s) from %s -> %s with %d worker(s) (%s-first bit order)",
        len(cases), src_dir, dst_dir, workers, "msb" if msb_first else "lsb",
    )

    results: list[dict] = []
    failures: list[str] = []
    ctx = mp.get_context("fork")  # Linux: cheap, children inherit sys.path + imports
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        fut2case = {
            ex.submit(expand_one, _src(c), os.path.join(dst_dir, c + _SUFFIX), msb_first, args.overwrite): c
            for c in cases
        }
        for fut in as_completed(fut2case):
            case = fut2case[fut]
            try:
                r = fut.result()
            except Exception:  # noqa: BLE001 - isolate one file's failure
                log.error("FAILED %s\n%s", case, traceback.format_exc())
                failures.append(case)
                continue
            results.append(r)
            if r["status"] == "skipped":
                log.info("skip %s (exists; use --overwrite)", case)
            else:
                log.info(
                    "done %s: %s -> %s (%d multi-bit) in %ss",
                    case, r["in_shape"], r["out_shape"], r["n_multibit"], r["elapsed"],
                )

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    log.info(
        "finished: %d converted, %d skipped, %d failed -> %s",
        len(ok), len(skipped), len(failures), dst_dir,
    )
    if failures:
        log.error("failed cases: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
