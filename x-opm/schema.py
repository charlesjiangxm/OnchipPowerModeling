"""Column parsing and A/B/C/D signal classification for x-opm.

A raw column name is a full RTL hierarchy path with an optional ``[msb:lsb]``
bit-range suffix, e.g. ``x_aq_cp0_top/cp0_dtu_wdata[63:0]``.  Leaf names collide
across hierarchy depth, so the *full path* is the unique feature identifier.

Classification follows ``doc/x-opm trainning procedure.md`` step 2, matching
keywords against the full (lowercased) path with priority **B > A > C > D**:

- type B (clock gating): path contains ``clk`` AND ``_en``
- type A (control):      path contains any of _en/_vld/_stall/_req/_busy/_idle
- type C (data bus):     path contains ``data``
- type D:                everything else (control & status payload)

Note on type B: the procedure literally says "contains clk and en", but a bare
``en`` substring mis-tags clocks such as ``.../fence_clk`` (the ``en`` comes from
"f-e-n-ce").  Requiring ``_en`` keeps every genuine ICG enable
(``global_en``, ``module_en``, ``regs_clk_en`` ... all contain ``_en``) while
dropping those false positives.  The token lists are configurable (see config.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# path with an optional trailing [a:b] range; single-index [k] (instance arrays)
# is intentionally NOT treated as a range -> width 1, name kept verbatim.
_RANGE_RE = re.compile(r"^(?P<path>.*?)(?:\[(?P<a>\d+):(?P<b>\d+)\])?$")


def parse_column(column: str) -> tuple[str, int, int]:
    """Return ``(path, width, lo)`` for a raw column name.

    ``path`` is the name with the ``[msb:lsb]`` suffix stripped (equal to
    ``column`` for a single-bit net).  ``width = abs(a-b)+1`` and ``lo = min(a,b)``
    (msb/lsb order agnostic).  A name with no ``[a:b]`` suffix -> ``(column, 1, 0)``.
    """
    m = _RANGE_RE.match(column)
    if m is None or m.group("a") is None:
        return column, 1, 0
    a, b = int(m.group("a")), int(m.group("b"))
    return m.group("path"), abs(a - b) + 1, min(a, b)


@dataclass(frozen=True)
class TypeRules:
    """Keyword tokens that drive classification (all matched on lowercased path)."""

    a_tokens: tuple[str, ...] = ("_en", "_vld", "_stall", "_req", "_busy", "_idle")
    b_clk_token: str = "clk"
    b_en_token: str = "_en"
    c_token: str = "data"
    invert_tokens: tuple[str, ...] = ("_stall", "_idle")


def classify(column: str, rules: TypeRules) -> tuple[str, str]:
    """Classify a column into ('A'|'B'|'C'|'D', human-readable rule string).

    Priority B > A > C > D.  The rule string records exactly which token(s)
    triggered the assignment, for the feature report.
    """
    lc = column.lower()
    if rules.b_clk_token in lc and rules.b_en_token in lc:
        return "B", f"contains '{rules.b_clk_token}' and '{rules.b_en_token}'"
    for tok in rules.a_tokens:
        if tok in lc:
            return "A", f"contains '{tok}'"
    if rules.c_token in lc:
        return "C", f"contains '{rules.c_token}'"
    return "D", "not type A/B/C"


def should_invert(column: str, rules: TypeRules) -> bool:
    """True for type-A signals negatively correlated with power (_stall / _idle)."""
    lc = column.lower()
    return any(tok in lc for tok in rules.invert_tokens)
