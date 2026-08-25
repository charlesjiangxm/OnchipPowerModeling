"""Scan the toggle database: feature scopes, benchmarks, coverage.

The DB layout (verified against c906_db_net_1cyc_20260729):

- ``aq_core/<bench>_func.pkl``          top-level nets ("top" scope)
- ``aq_core/<module>/<bench>_func.pkl`` one pkl per first-level module,
  whose columns cover the module's ENTIRE subtree (sub-module directories
  such as ``aq_core/rtu/x_aq_rtu_int/`` hold redundant column subsets and
  are intentionally ignored)
- ``pwr/<bench>_pwr.pkl``               per-cycle power labels

Feature space = top scope + all module scopes; the union is duplicate-free.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from ..utils import log

TOP_SCOPE = "top"

# Pickles may be stored raw (``X.pkl``) or zstd-compressed (``X.pkl.zst``).
# pandas infers zstd from the ``.zst`` extension, so we only have to resolve the
# path; readers stay unchanged. Prefer the compressed sibling when it exists.
_FUNC_SUFFIXES = ("_func.pkl.zst", "_func.pkl")


def _pick(pkl: Path) -> Path:
    """``X.pkl`` -> ``X.pkl.zst`` if that compressed sibling exists, else ``X.pkl``."""
    z = pkl.with_name(pkl.name + ".zst")
    return z if z.exists() else pkl


def _bench_stem(name: str) -> str:
    """Strip a ``_func.pkl`` / ``_func.pkl.zst`` suffix to recover the benchmark name."""
    for suf in _FUNC_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


@dataclasses.dataclass
class DbLayout:
    db_root: Path
    scopes: list[str]  # ["top", <module names>...]
    benchmarks: list[str]  # all benchmarks with at least one func pkl + a pwr pkl
    coverage: dict[str, list[str]]  # bench -> scopes that HAVE a func pkl

    def func_pkl(self, scope: str, bench: str) -> Path:
        core = self.db_root / "aq_core"
        base = core if scope == TOP_SCOPE else core / scope
        return _pick(base / f"{bench}_func.pkl")

    def pwr_pkl(self, bench: str) -> Path:
        return _pick(self.db_root / "pwr" / f"{bench}_pwr.pkl")

    def missing_scopes(self, bench: str) -> list[str]:
        return [s for s in self.scopes if s not in self.coverage[bench]]

    def complete_benchmarks(self) -> list[str]:
        return [b for b in self.benchmarks if not self.missing_scopes(b)]


def discover(db_root: str | Path) -> DbLayout:
    db_root = Path(db_root)
    core = db_root / "aq_core"
    if not core.is_dir():
        raise FileNotFoundError(f"no aq_core/ under {db_root}")
    if not (db_root / "pwr").is_dir():
        raise FileNotFoundError(f"no pwr/ under {db_root}")

    modules = sorted(p.name for p in core.iterdir() if p.is_dir())
    scopes = [TOP_SCOPE] + modules

    benches: set[str] = set()
    coverage: dict[str, set[str]] = {}
    for scope in scopes:
        base = core if scope == TOP_SCOPE else core / scope
        for pkl in list(base.glob("*_func.pkl")) + list(base.glob("*_func.pkl.zst")):
            bench = _bench_stem(pkl.name)
            benches.add(bench)
            coverage.setdefault(bench, set()).add(scope)

    kept = []
    for bench in sorted(benches):
        if not _pick(db_root / "pwr" / f"{bench}_pwr.pkl").is_file():
            log.warning("benchmark %s has func pkls but no pwr pkl - skipped", bench)
            continue
        kept.append(bench)

    layout = DbLayout(
        db_root=db_root,
        scopes=scopes,
        benchmarks=kept,
        coverage={b: sorted(coverage[b]) for b in kept},
    )
    for bench in kept:
        miss = layout.missing_scopes(bench)
        if miss:
            log.warning("benchmark %s is missing scopes: %s", bench, ", ".join(miss))
    return layout


def plan_benchmarks(layout: DbLayout, test_benchmarks: list[str]) -> tuple[list[str], list[str]]:
    """Split discovered benchmarks into (train, test) per the missing-pkl policy.

    Training benchmarks must have full scope coverage (incomplete ones are
    dropped with a warning). Test benchmarks are never dropped: if scopes are
    missing, their bits are zero-filled at build time (loud warning).
    """
    unknown = [b for b in test_benchmarks if b not in layout.benchmarks]
    if unknown:
        raise ValueError(f"test benchmarks not found in DB: {unknown}")
    train = []
    for b in layout.benchmarks:
        if b in test_benchmarks:
            continue
        miss = layout.missing_scopes(b)
        if miss:
            log.warning(
                "dropping training benchmark %s (missing scopes: %s)", b, ", ".join(miss)
            )
            continue
        train.append(b)
    for b in test_benchmarks:
        miss = layout.missing_scopes(b)
        if miss:
            log.warning(
                "TEST benchmark %s is missing scopes %s - their bits will be "
                "ZERO-FILLED; power labels still include those modules, so "
                "interpret its metrics accordingly",
                b,
                ", ".join(miss),
            )
    if not train:
        raise RuntimeError("no complete training benchmarks left after policy filtering")
    return train, list(test_benchmarks)
