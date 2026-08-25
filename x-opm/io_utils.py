"""Small IO helpers: atomic writes for pickles / json / csv.

Atomic = write to a temp file in the same directory, then ``os.replace`` so a
reader never observes a half-written file and a crash never corrupts an existing
one (mirrors ``cobit/utils.py``).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pandas as pd


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _atomic_write(path: str, write_fn) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    os.close(fd)
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_pickle(obj: Any, path: str) -> None:
    # The atomic write goes through a ``.tmp`` file, so pandas can't infer the
    # compression from the final extension -- pass it explicitly when the target
    # is a ``.zst`` (leaving plain ``.pkl`` writers on raw pickle).
    comp = "zstd" if path.endswith(".zst") else "infer"
    _atomic_write(path, lambda tmp: pd.to_pickle(obj, tmp, compression=comp))


def save_json(obj: Any, path: str) -> None:
    def _w(tmp: str) -> None:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
    _atomic_write(path, _w)


def save_csv(df: pd.DataFrame, path: str) -> None:
    _atomic_write(path, lambda tmp: df.to_csv(tmp, index=False))


def read_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)
