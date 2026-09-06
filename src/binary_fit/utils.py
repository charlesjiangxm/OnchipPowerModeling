"""Shared helpers: logging, atomic file writes, hashing, JSON/CSV I/O."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("binary_fit")


def setup_logging(level: int = logging.INFO) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    logging.getLogger("binary_fit").setLevel(level)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write to a temp file in the same directory, then rename into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def save_pickle_zst(obj: Any, path: Path) -> None:
    """Atomically pickle ``obj`` to ``path`` with zstd compression.

    The atomic write targets a ``.tmp`` sibling with no ``.zst`` suffix, so
    pandas cannot infer the codec -- pass it explicitly.
    """
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        pd.to_pickle(obj, tmp, compression="zstd")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class _JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(path: Path, obj: Any, indent: int = 2) -> None:
    atomic_write_text(Path(path), json.dumps(obj, indent=indent, cls=_JSONEncoder))


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def stable_hash(obj: Any) -> str:
    """Deterministic sha256 of a JSON-serializable object (sorted keys)."""
    blob = json.dumps(obj, sort_keys=True, cls=_JSONEncoder).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# proxy / coefficient CSVs
# --------------------------------------------------------------------------- #
PROXY_CSV_HEADER = ["rank", "name", "col_id", "mcp_weight"]


def save_proxies_csv(path: Path, names, col_ids, weights) -> None:
    """Write the MCP-selected proxies, ranked by ``|mcp_weight|`` descending.

    This is the source of truth consumed by ``--fit``: the top-``q`` rows are the
    proxies used at ``-q q``, so the descending-|weight| order MUST hold. Columns
    are ``rank, name, col_id, mcp_weight`` where ``col_id`` is the proxy's global
    feature id (its position in the materialized single-bit column set).
    """
    import numpy as np

    names = list(names)
    col_ids = np.asarray(col_ids).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if not (len(names) == col_ids.size == weights.size):
        raise ValueError(
            f"proxy table length mismatch: names={len(names)} "
            f"col_ids={col_ids.size} weights={weights.size}"
        )
    order = sorted(range(len(names)), key=lambda j: -abs(weights[j]))
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(PROXY_CSV_HEADER)
    for rank, j in enumerate(order, start=1):
        writer.writerow([rank, names[j], int(col_ids[j]), f"{weights[j]:.8g}"])
    atomic_write_text(Path(path), buf.getvalue())


def load_proxies_csv(path: Path):
    """Read ``proxies.csv`` back into (names, col_ids, weights), rank order preserved."""
    import numpy as np

    names: list[str] = []
    col_ids: list[int] = []
    weights: list[float] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["name"])
            col_ids.append(int(row["col_id"]))
            weights.append(float(row["mcp_weight"]))
    return names, np.asarray(col_ids, dtype=np.int64), np.asarray(weights, dtype=float)


COEF_CSV_HEADER = ["rank", "name", "col_id", "value", "importance"]
RIDGE_COEF_CSV_HEADER = ["rank", "name", "col_id", "coef_std", "coef_watts"]


def save_coefficients_csv(path: Path, names, col_ids, values, importances) -> None:
    """Dump a per-feature coefficient/importance table to CSV, ranked by importance.

    Columns are ``rank, name, col_id, value, importance``: ``value`` is the
    Stage-1 LR-MCP linear coefficient of the proxy, ``importance`` is the trained
    model's feature importance (normalized to sum to 1; 0 for a feature the model
    never used). Rows sorted by ``importance`` desc, ties by ``|value|`` desc. The
    four inputs must be index-aligned. Shared by the tree and nn backends.
    """
    import numpy as np

    names = list(names)
    col_ids = np.asarray(col_ids).ravel()
    values = np.asarray(values, dtype=float).ravel()
    imp = np.asarray(importances, dtype=float).ravel()
    if not (len(names) == col_ids.size == values.size == imp.size):
        raise ValueError(
            f"coefficient table length mismatch: names={len(names)} "
            f"col_ids={col_ids.size} values={values.size} importances={imp.size}"
        )
    total = float(imp.sum())
    imp_norm = imp / total if total > 0 else imp
    order = sorted(range(len(names)), key=lambda j: (-imp_norm[j], -abs(values[j])))

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(COEF_CSV_HEADER)
    for rank, j in enumerate(order, start=1):
        writer.writerow([rank, names[j], int(col_ids[j]),
                         f"{values[j]:.8g}", f"{imp_norm[j]:.8g}"])
    atomic_write_text(Path(path), buf.getvalue())


def save_ridge_coefficients_csv(path: Path, names, col_ids, coef_std, coef_watts) -> None:
    """Dump the ridge backend's SIGNED coefficients, ranked by ``|coef_std|`` desc.

    A separate file rather than a column of ``coefficients.csv``: that table is
    shared by all three backends and its ``value`` column is documented as the
    Stage-1 LR-MCP weight of the proxy, so a fitted linear coefficient has
    nowhere to live there (and its ``importance`` column must stay non-negative).

    ``coef_std`` is the coefficient in standardized space -- dimensionless, and
    the one that is comparable across bits. ``coef_watts`` is the same fit in the
    target's own units: watts per unit of that feature, which at
    ``data.window_size = 1`` is watts per assertion of the bit. The matching
    ``intercept_watts`` is recorded in ``result.json`` under ``best``, because a
    single scalar in a per-feature table would have to be a fake row.
    """
    import numpy as np

    names = list(names)
    col_ids = np.asarray(col_ids).ravel()
    std = np.asarray(coef_std, dtype=float).ravel()
    watts = np.asarray(coef_watts, dtype=float).ravel()
    if not (len(names) == col_ids.size == std.size == watts.size):
        raise ValueError(
            f"ridge coefficient table length mismatch: names={len(names)} "
            f"col_ids={col_ids.size} coef_std={std.size} coef_watts={watts.size}"
        )
    order = sorted(range(len(names)), key=lambda j: -abs(std[j]))
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(RIDGE_COEF_CSV_HEADER)
    for rank, j in enumerate(order, start=1):
        writer.writerow([rank, names[j], int(col_ids[j]),
                         f"{std[j]:.8g}", f"{watts[j]:.8g}"])
    atomic_write_text(Path(path), buf.getvalue())
