"""Canonical net table and global bit-feature naming.

A *net* is one pkl column: ``x/y/z[hi:lo]`` or a bare single-bit name. Its
value at a cycle is an integer bitmask where mask bit ``k`` means RTL bit
``lo + k`` of the net toggled during that cycle.

The registry fixes a canonical net order (scope order, then lexicographic
path within scope) that is independent of pandas column order, and assigns
every net a ``base_col`` so that global feature id = ``base_col + k`` and
feature name = ``path[lo + k]`` (bare name for 1-bit nets).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pandas as pd

from ..utils import load_json, log, save_json, stable_hash
from .discovery import DbLayout

_RANGE_RE = re.compile(r"^(?P<path>.*?)(?:\[(?P<a>\d+):(?P<b>\d+)\])?$")


@dataclasses.dataclass(frozen=True)
class Net:
    scope: str
    column: str  # exact pkl column name
    path: str  # column name without the [hi:lo] suffix
    width: int
    lo: int
    base_col: int  # first global feature id of this net


def parse_column(column: str) -> tuple[str, int, int]:
    """Return (path, width, lo) for a pkl column name."""
    m = _RANGE_RE.match(column)
    assert m is not None
    if m.group("a") is None:
        return column, 1, 0
    a, b = int(m.group("a")), int(m.group("b"))
    return m.group("path"), abs(a - b) + 1, min(a, b)


class Registry:
    def __init__(self, nets: list[Net]):
        self.nets = nets
        self.n_features = nets[-1].base_col + nets[-1].width if nets else 0
        self._by_scope: dict[str, list[Net]] = {}
        for n in nets:
            self._by_scope.setdefault(n.scope, []).append(n)
        self.content_hash = stable_hash(
            [[n.scope, n.column, n.width, n.lo, n.base_col] for n in nets]
        )

    # -- lookups -------------------------------------------------------------
    def nets_of_scope(self, scope: str) -> list[Net]:
        return self._by_scope.get(scope, [])

    def feature_names(self, ids=None) -> list[str]:
        """Names for the given global feature ids (or all)."""
        import numpy as np

        starts = np.array([n.base_col for n in self.nets])
        if ids is None:
            ids = range(self.n_features)
        out = []
        for gid in ids:
            i = int(np.searchsorted(starts, gid, side="right") - 1)
            net = self.nets[i]
            k = gid - net.base_col
            assert 0 <= k < net.width, f"feature id {gid} out of range"
            # column == path exactly when there is no [hi:lo] range suffix;
            # scalar nets whose PATH contains instance-array brackets (e.g.
            # .../RAM_DIN_VEC[0]/ram_instance/PortAClk) keep their bare name
            out.append(net.path if net.column == net.path else f"{net.path}[{net.lo + k}]")
        return out

    # -- persistence -----------------------------------------------------------
    def save(self, path: Path) -> None:
        save_json(
            path,
            {
                "content_hash": self.content_hash,
                "n_features": self.n_features,
                "nets": [
                    dict(scope=n.scope, column=n.column, path=n.path,
                         width=n.width, lo=n.lo, base_col=n.base_col)
                    for n in self.nets
                ],
            },
        )

    @classmethod
    def load(cls, path: Path) -> "Registry":
        raw = load_json(path)
        reg = cls([Net(**d) for d in raw["nets"]])
        if reg.content_hash != raw["content_hash"]:
            raise RuntimeError(f"registry content hash mismatch in {path}")
        return reg


def build_registry(layout: DbLayout) -> Registry:
    """Read one benchmark pkl per scope (columns only) and canonicalize.

    Column sets are identical across benchmarks within a scope (verified),
    so any benchmark that has the scope works.
    """
    nets: list[Net] = []
    base = 0
    for scope in layout.scopes:
        bench = next((b for b in layout.benchmarks if scope in layout.coverage[b]), None)
        if bench is None:
            log.warning("scope %s has no benchmark pkl at all - excluded from registry", scope)
            continue
        df = pd.read_pickle(layout.func_pkl(scope, bench))
        for column in sorted(df.columns):
            path, width, lo = parse_column(column)
            nets.append(Net(scope=scope, column=column, path=path, width=width,
                            lo=lo, base_col=base))
            base += width
    reg = Registry(nets)
    log.info("registry: %d nets, %d bit features, hash %s",
             len(nets), reg.n_features, reg.content_hash)
    return reg
