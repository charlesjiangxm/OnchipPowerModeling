"""Shared helpers: logging, atomic file writes, hashing, JSON I/O."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("cobit")


def setup_logging(level: int = logging.INFO) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    logging.getLogger("cobit").setLevel(level)


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
