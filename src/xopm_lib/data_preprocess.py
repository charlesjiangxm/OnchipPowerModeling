"""X-OPM data preparation (steps 1-5): raw signal-state dumps -> scaled ML features.

Single-file implementation as mandated by ``doc/spec/x-opm-trainning-procedure.md``.
Turns per-cycle RTL signal-state dumps (``*_func.pkl.zst``) paired with per-cycle
power (``*_pwr.pkl.zst``) into scaled features written to ``dataset_processed/``,
split into train / test sets.

Two passes, both driven by a per-module YAML config:

  fit(cfg)        -- TRAIN cases only. Streams every training case through an
                     O(#columns) accumulator (never concatenates rows, so the
                     7-17 GB random-stimulus frames stay tractable), decides which
                     raw columns survive (drop constants + exact duplicates), and
                     records a per-feature transform plan into ``manifest.json``.
                     Keeping every decision train-only is what prevents leakage.
  apply_case(...) -- Every case (train and test). Loads + time-aligns one case,
                     applies the manifest's plan, writes one ``.pkl.zst`` per
                     (case, category) plus an aligned target, then frees the frame.

Categories (spec step 2, priority Control > Data > Config; clock-gating folded
into Control). Tokens are matched on the *leaf signal name* -- the substring after
the last '/' with any trailing '[...]' bit-range stripped -- not the full path, so
a token appearing only in a parent instance name does not misclassify a signal.
Control tokens match as *whole words*: a token is not counted when it is merely the
prefix of a longer word, so '_en' matches 'wr_en'/'wr_en0' but not 'entry'/'enable'.
Signals below any 'x_aq_spsram*' macro instance are dropped outright at load time:

  control : signal name contains whole-word _en/_vld/_stall/_busy/_idle or (clk and _en)
            -> bit-split into single bits; invert _stall/_idle; drop dead bits.
  data    : leaf *ends with* a 'data'/'din'/'dout' core (optional leading '_' + an
            optional suffix: raw, bank/bank0/bankx, or a numeric index) -- e.g. data,
            wdata, biu_lsu_rdata, mem_wdata, dcache_data_din_bank0, icache_data_32 --
            AND carries none of the access-qualifier words sel/dirty/read/rd/write/wr as
            a whole '_'-delimited field. So a control-qualified bus (data_idx, data_wen),
            a read/write port (read_data_7 -> 'read' field, wr_data_idx) or a coincidental
            'din' (rounding, pending) is NOT data, but rdata/wdata buses (rd/wr are the
            head of the core, not a field) ARE data.
            -> per-cycle XOR toggle -> Hamming distance (one scalar), scaled /W.
  config  : everything else
            -> raw integer scaled by 2^width-1.

CLI (interpreter must be ~/anaconda3/bin/python):

    python src/xopm_lib/data_preprocess.py --config config/xopm/cp0.yaml --stage all
    python src/xopm_lib/data_preprocess.py --config config/xopm/cp0.yaml --stage fit
    python src/xopm_lib/data_preprocess.py --config config/xopm/cp0.yaml --stage transform --cases coremark,conv_softmax
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("xopm.preprocess")

_MASK64 = (1 << 64) - 1
_U1 = np.uint64(1)
_STORE_DTYPE = np.float32          # scaled features live in [0,1]; float32 is ample
CATEGORIES = ("control", "data", "config")


# ===========================================================================
# Step 2 -- column parsing & 3-way classification (matched on leaf signal name)
# ===========================================================================

# path with an optional trailing [a:b] range; a single-index [k] (instance arrays)
# is intentionally NOT a range -> width 1, name kept verbatim.
_RANGE_RE = re.compile(r"^(?P<path>.*?)(?:\[(?P<a>\d+):(?P<b>\d+)\])?$")

# any trailing bracket ([a:b] or a single index [k]) -- stripped to get the leaf name.
_LEAF_RANGE_RE = re.compile(r"\[[^\]]*\]$")


def parse_column(column: str) -> tuple[str, int, int]:
    """Return ``(path, width, lo)`` for a raw column name.

    ``path`` is the name with any ``[msb:lsb]`` suffix stripped (== ``column`` for
    a 1-bit net). ``width = abs(a-b)+1`` and ``lo = min(a,b)`` (msb/lsb agnostic).
    A name with no ``[a:b]`` suffix -> ``(column, 1, 0)``.
    """
    m = _RANGE_RE.match(column)
    if m is None or m.group("a") is None:
        return column, 1, 0
    a, b = int(m.group("a")), int(m.group("b"))
    return m.group("path"), abs(a - b) + 1, min(a, b)


def signal_name(column: str) -> str:
    """Return the leaf *signal name* used for classification token matching.

    Per spec step 2: the substring after the last ``/`` with any trailing ``[...]``
    (a bit range ``[hi:lo]`` or a single index ``[k]``) stripped. Example:
    ``x_aq_lsu_top/.../stb_merge_data[49:0]`` -> ``stb_merge_data``. This is matching-
    only; ``parse_column`` still drives bit-split width/offset and feature naming, so
    a token in a *parent* instance name no longer forces a category.
    """
    leaf = column.rsplit("/", 1)[-1]
    return _LEAF_RANGE_RE.sub("", leaf)


# Signals below these macro instances are excluded outright (spec step 2: "Do not
# include signals below x_aq_spsram_* modules"). Matched on any '/'-delimited path
# segment, so a whole SRAM macro subtree (its A/CEN/Q/... pins) is dropped.
SPSRAM_EXCLUDE = ("x_aq_spsram",)


def is_excluded_column(column: str, prefixes: tuple[str, ...] = SPSRAM_EXCLUDE) -> bool:
    """True if any hierarchy segment of ``column`` starts with an excluded prefix."""
    return bool(prefixes) and any(seg.startswith(prefixes) for seg in column.split("/"))


def _contains_word(text: str, token: str) -> bool:
    """True if ``token`` occurs in ``text`` as a whole word (spec step 2).

    "Whole word" == the token is not immediately followed by another lowercase
    letter, so word-continuation is only broken by end-of-string, ``_`` or a digit.
    A token carrying its own leading ``_`` (``_en``) is thus anchored on the left by
    that ``_`` and on the right by a non-letter: ``_en`` matches ``wr_en``, ``wr_en0``
    and ``wr_en_x`` but NOT ``entry``/``enable``; ``_vld`` matches per-lane ``_vld0``
    but not ``vldbus``.
    """
    return re.search(re.escape(token) + r"(?![a-z])", text) is not None


def _has_token(text: str, token: str) -> bool:
    """True if ``token`` is a complete ``_``-delimited segment of ``text``.

    The data access-qualifier excludes (spec step 2: ``_*sel*_`` / ``_*rd*_`` / ...)
    fire only when the qualifier is a distinct hierarchical field, not merely a
    substring. So ``rd`` matches ``x_rd_data`` (segment ``rd``) and ``read`` matches
    ``read_data_7`` (leading segment ``read``), but ``rd`` does NOT match ``rdata`` /
    ``biu_lsu_rdata`` (the ``rd`` is the head of the ``rdata`` core, not its own field)
    and ``wr`` does NOT match ``wdata``. Start/end of the leaf count as segment
    boundaries, so a leading ``read_``/``write_`` still qualifies.
    """
    return token in text.split("_")


@dataclass(frozen=True)
class TypeRules:
    """Classification keyword tokens (all matched against the lowercased leaf name).

    Control tokens (``a_tokens`` + the ``clk``/``clk_en`` clock-gating pair) match as
    whole words via ``_contains_word`` -- ``_en`` matches ``wr_en``/``wr_en0`` but not
    ``entry``/``enable``, and the leading ``_`` keeps a bare ``en`` inside ``fence_clk``
    from matching. A non-control leaf is Data iff it matches an inclusion regex in
    ``data_patterns`` AND carries none of the ``data_exclude_words`` as a whole
    ``_``-delimited token (via ``_has_token``). The default ``data_patterns`` encode the
    spec's "ends with" suffix globs ``_*data*``/``_*rdata*``/``_*wdata*``/``_*din*``/
    ``_*dout*``: the leaf must *end* with a ``data``/``din``/``dout`` core optionally
    preceded by a prefix and followed by a limited suffix (``raw``, ``bank``/``bank0``/
    ``bankx``, or a numeric index like ``_10``/``data0``); the leading underscore is
    optional so a bare ``data``/``wdata`` bus still qualifies. So ``data``, ``wdata``,
    ``biu_lsu_rdata``, ``mem_wdata``, ``dcache_data_din_bank0`` and ``icache_data_32`` are
    Data, but ``data_idx``/``data_req``/``data_wen`` (a control qualifier after the core)
    and the ``din`` inside ``rounding``/``pending`` are not. ``data_exclude_words`` then
    veto an access-qualified bus only when the qualifier is its own field -- ``sel``/
    ``dirty`` (``alias_data_sel``, ``dcache_dirty_din``) and read/write ports
    (``read_data_7``, ``wr_data_idx``) fall through to Config, while ``rdata``/``wdata``
    (``rd``/``wr`` are the head of the data core, not a separate field) stay Data.
    ``exclude_segment_prefixes`` drops whole macro subtrees before classification.
    """

    a_tokens: tuple[str, ...] = ("_en", "_vld", "_stall", "_busy", "_idle")
    clk_token: str = "clk"
    clk_en_token: str = "_en"
    data_patterns: tuple[str, ...] = (
        r"(?:^|_).*?(?:data|din|dout)(?:_?(?:raw|bank[0-9x]*|[0-9]+))*$",)
    data_exclude_words: tuple[str, ...] = ("sel", "dirty", "read", "rd", "write", "wr")
    invert_tokens: tuple[str, ...] = ("_stall", "_idle")
    exclude_segment_prefixes: tuple[str, ...] = SPSRAM_EXCLUDE


def classify(column: str, rules: TypeRules) -> tuple[str, str]:
    """Classify a column into ``('control'|'data'|'config', rule_string)``.

    Tokens are matched on the leaf ``signal_name`` (after the last ``/``, bit-range
    stripped), lowercased. Control tokens match as whole words (see ``_contains_word``);
    Data is a leaf matching a ``rules.data_patterns`` inclusion regex (the "ends with"
    globs ``_*data*`` / ``_*rdata*`` / ``_*wdata*`` / ``_*din*`` / ``_*dout*``, with an
    optional leading underscore) that carries none of the ``rules.data_exclude_words``
    (sel/dirty/read/rd/write/wr) as a whole ``_``-delimited token (see ``_has_token``).
    So ``*rdata`` / ``wdata`` buses stay Data (``rd``/``wr`` are the head of the core, not
    a separate field), while ``read_data_7`` and ``dcache_dirty_din`` are vetoed to Config.
    Priority Control > Data > Config, so a leaf matching a control token *and* a data
    pattern is Control, and a data-pattern leaf carrying an exclude token falls through to
    Config. Clock-gating (``clk`` and ``_en``) is folded into Control. The rule string
    records the trigger.
    """
    lc = signal_name(column).lower()
    if _contains_word(lc, rules.clk_token) and _contains_word(lc, rules.clk_en_token):
        return "control", f"contains '{rules.clk_token}' and '{rules.clk_en_token}'"
    for tok in rules.a_tokens:
        if _contains_word(lc, tok):
            return "control", f"contains '{tok}'"
    matched = next((p for p in rules.data_patterns if re.search(p, lc)), None)
    if matched is not None:
        veto = next((w for w in rules.data_exclude_words if _has_token(lc, w)), None)
        if veto is None:
            return "data", f"leaf ends with data pattern {matched!r}"
        return "config", f"data pattern {matched!r} vetoed by token '{veto}'"
    return "config", "not control/data"


def should_invert(column: str, rules: TypeRules) -> bool:
    """True for control signals negatively correlated with power (_stall / _idle).

    Matched on the leaf ``signal_name`` as whole words (spec step 3: "signal name
    contains ..."), consistent with ``classify``.
    """
    lc = signal_name(column).lower()
    return any(_contains_word(lc, tok) for tok in rules.invert_tokens)


# ===========================================================================
# Step 3-4 -- vectorized per-signal transforms and scaling
# ===========================================================================

def to_lanes(values: np.ndarray, width: int) -> list[np.ndarray]:
    """Split an object array of non-negative ints into little-endian uint64 lanes.

    Integers < 2^53 round-trip exactly through the fast ``astype(uint64)`` path.
    For width 54..64 (and >64) the cells may be Python floats -- whose value >= 2^53
    is already lossy on disk -- so we go through exact ``int(v) & mask`` conversion,
    which also folds the ``==2^width`` dump artifact / near-2^64 rounding instead of
    silently overflowing uint64.
    """
    n = values.shape[0]
    if width <= 53:
        try:
            return [values.astype(np.uint64)]
        except (OverflowError, ValueError, TypeError):
            return [np.fromiter((int(v) & _MASK64 for v in values), np.uint64, n)]
    if width <= 64:
        return [np.fromiter((int(v) & _MASK64 for v in values), np.uint64, n)]
    nlanes = (width + 63) // 64
    return [np.fromiter(((int(v) >> (64 * L)) & _MASK64 for v in values),
                        np.uint64, n) for L in range(nlanes)]


def split_bits(values: np.ndarray, path: str, width: int, lo: int,
               invert: bool = False,
               live_bits: list[int] | None = None) -> list[tuple[str, np.ndarray]]:
    """Bit-split a (possibly multi-bit) signal into single-bit uint8 features.

    Bit ``k`` (0..width-1) is ``(intval >> k) & 1`` labelled ``path[lo+k]``; a
    width-1 signal keeps its original name (never ``name[0]``). ``invert`` flips
    each bit (for ``_stall``/``_idle``). ``live_bits`` (if given) restricts output
    to those bit indices (dead-bit dropping); ``None`` emits every bit.
    """
    lanes = to_lanes(values, width)
    keep = range(width) if live_bits is None else live_bits
    out: list[tuple[str, np.ndarray]] = []
    for k in keep:
        lane = lanes[k // 64]
        bit = ((lane >> np.uint64(k % 64)) & _U1).astype(np.uint8)
        if invert:
            bit = np.uint8(1) - bit
        name = path if width == 1 else f"{path}[{lo + k}]"
        out.append((name, bit))
    return out


def toggle_hamming(values: np.ndarray, width: int,
                   count_initial: bool = True) -> np.ndarray:
    """Hamming distance of the per-cycle toggle of a bus (spec step 3, data).

    ``toggle[t] = state[t] XOR state[t-1]`` with ``state[-1] = 0`` (per case, so
    no toggle carries across a benchmark boundary), then ``hd[t] = popcount``.
    ``count_initial=True`` counts cycle 0 as a toggle-from-reset.
    """
    n = values.shape[0]
    ham = np.zeros(n, dtype=np.int32)
    for lane in to_lanes(values, width):
        tog = np.empty_like(lane)
        tog[0] = lane[0] if count_initial else np.uint64(0)
        if n > 1:
            tog[1:] = lane[1:] ^ lane[:-1]
        ham += np.bitwise_count(tog).astype(np.int32)
    return ham


def scale_bits(bits: np.ndarray) -> np.ndarray:
    """Single-bit features: max value is 1, so scaling is a no-op (to float)."""
    return bits.astype(np.float64)


def scale_hamming(ham: np.ndarray, bus_width: int) -> np.ndarray:
    """Hamming in [0, W] -> divide by bus width W for a clean [0,1] fraction."""
    return ham.astype(np.float64) / float(bus_width)


def scale_raw_int(values: np.ndarray, width: int) -> np.ndarray:
    """Config raw integer scaled by its max representable value 2^width - 1.

    A few cells carry a dump artifact of exactly 2^width; the clip guarantees the
    [0,1] invariant regardless of float64 rounding of the denominator.
    """
    denom = float((1 << width) - 1)
    return np.clip(values.astype(np.float64) / denom, 0.0, 1.0)


# ===========================================================================
# Step 1 -- streaming raw statistics + constant / duplicate / dead-bit detection
# ===========================================================================

class RawStats:
    """Accumulate per-column stats over the TRAIN cases, case by case.

    Constant (zero-variance / bus-level invariant) is detected exactly via
    ``min == max``. Exact duplicates share a blake2b fingerprint of the width-
    masked uint64 value stream (so an int-vs-float storage of the same value
    still matches). For control columns we additionally OR/AND every value so
    per-bit dead (never-toggling) bits can be dropped after bit-splitting.
    """

    def __init__(self, columns: list[str], widths: dict[str, int],
                 control_cols: list[str], detect_duplicates: bool = True,
                 track_bits: bool = True):
        self.columns = columns
        self.widths = widths
        self.detect_duplicates = detect_duplicates
        self.control_cols = set(control_cols) if track_bits else set()
        self.min: dict[str, int] = {}
        self.max: dict[str, int] = {}
        self.sum: dict[str, float] = {c: 0.0 for c in columns}
        self.count: int = 0
        self._hash: dict[str, "hashlib._Hash"] = (
            {c: hashlib.blake2b(digest_size=16) for c in columns}
            if detect_duplicates else {})
        # OR/AND lane accumulators for dead-bit detection (control only).
        self.or_acc: dict[str, list[int]] = {}
        self.and_acc: dict[str, list[int]] = {}
        for c in self.control_cols:
            nl = (widths[c] + 63) // 64
            self.or_acc[c] = [0] * nl
            self.and_acc[c] = [_MASK64] * nl

    def update(self, df: pd.DataFrame) -> None:
        """Fold one aligned case DataFrame (columns already canonical) into stats."""
        self.count += df.shape[0]
        for col in self.columns:
            width = self.widths[col]
            v = df[col].to_numpy()
            lanes = to_lanes(v, width)
            if width <= 64:
                lane = lanes[0]
                self._merge(col, int(lane.min()), int(lane.max()))
                self.sum[col] += float(lane.astype(np.float64).sum())
            else:
                ints = [int(x) for x in v]
                self._merge(col, min(ints), max(ints))
                self.sum[col] += float(sum(ints))
            if self.detect_duplicates:
                for lane in lanes:
                    self._hash[col].update(lane.tobytes())
            if col in self.control_cols:
                oa, aa = self.or_acc[col], self.and_acc[col]
                for L, lane in enumerate(lanes):
                    if lane.size:
                        oa[L] |= int(np.bitwise_or.reduce(lane))
                        aa[L] &= int(np.bitwise_and.reduce(lane))

    def _merge(self, col: str, mn: int, mx: int) -> None:
        self.min[col] = mn if col not in self.min else min(self.min[col], mn)
        self.max[col] = mx if col not in self.max else max(self.max[col], mx)

    def finalize(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for col in self.columns:
            mn, mx = self.min[col], self.max[col]
            rec = {
                "raw_min": mn,
                "raw_max": mx,
                "raw_mean": self.sum[col] / self.count if self.count else 0.0,
                "constant": (mn == mx),
            }
            if self.detect_duplicates:
                rec["value_hash"] = self._hash[col].hexdigest()
            out[col] = rec
        return out

    def live_control_bits(self, col: str, width: int) -> list[int]:
        """Bits of a control bus that actually toggle across pooled train."""
        oa, aa = self.or_acc.get(col), self.and_acc.get(col)
        if oa is None:                       # tracking disabled -> keep all bits
            return list(range(width))
        return [k for k in range(width)
                if ((oa[k // 64] >> (k % 64)) & 1) != ((aa[k // 64] >> (k % 64)) & 1)]


def duplicate_groups(finalized: dict[str, dict], kept: list[str]) -> list[list[str]]:
    """Group kept columns sharing an identical value fingerprint (size >= 2).

    Members are ordered shortest-path-then-lexicographic, so the representative is
    ``members[0]`` (physical nets dumped at multiple hierarchy depths collapse to
    the shallowest name).
    """
    by_hash: dict[str, list[str]] = {}
    for col in kept:
        h = finalized[col].get("value_hash")
        if h is None:
            continue
        by_hash.setdefault(h, []).append(col)
    groups = [sorted(cols, key=lambda c: (len(c), c))
              for cols in by_hash.values() if len(cols) >= 2]
    groups.sort(key=lambda g: (len(g[0]), g[0]))
    return groups


# ===========================================================================
# Load a case and align features with the power target in time
# ===========================================================================

def load_features(func_pkl: str, canonical_cols: list[str] | None = None,
                  exclude_prefixes: tuple[str, ...] = ()) -> pd.DataFrame:
    """Read a ``*_func.pkl.zst`` (pandas auto-infers zstd), int64 ``time_ns`` index.

    ``exclude_prefixes`` (spec step 2) drops any column below an excluded macro
    subtree *before* the canonical-set check, so a case file still carrying, e.g.,
    ``x_aq_spsram*`` pins reconciles with the (already-filtered) canonical columns.
    When ``canonical_cols`` is given the remaining column *set* must match exactly and
    the frame is reindexed to that fixed order so nothing downstream depends on the
    stored order.
    """
    df = pd.read_pickle(func_pkl)
    df.index = df.index.astype(np.int64)
    df.index.name = "time_ns"
    if exclude_prefixes:
        drop = [c for c in df.columns if is_excluded_column(c, exclude_prefixes)]
        if drop:
            df = df.drop(columns=drop)
    if canonical_cols is not None:
        if set(df.columns) != set(canonical_cols):
            missing = set(canonical_cols) - set(df.columns)
            extra = set(df.columns) - set(canonical_cols)
            raise RuntimeError(
                f"{func_pkl}: column set differs from canonical "
                f"(missing {len(missing)}, extra {len(extra)})")
        df = df.reindex(columns=canonical_cols)
    return df


def align_case(func_pkl: str, pwr_pkl: str, target_col: str,
               canonical_cols: list[str] | None = None,
               exclude_prefixes: tuple[str, ...] = ()
               ) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(features, target)`` sharing an identical int64 ``time_ns`` index.

    The feature frame is kept intact in its native cycle order (toggles depend on
    consecutive rows being consecutive cycles); the power frame -- which has one
    extra trailing cycle on c906 and a float index -- is reindexed onto the feature
    index, dropping its surplus row. A leading offset (feature cycle with no power
    label) surfaces as a NaN and is raised, not silently dropped.
    """
    feats = load_features(func_pkl, canonical_cols, exclude_prefixes)
    idx = feats.index.to_numpy()
    if not (feats.index.is_monotonic_increasing and feats.index.is_unique):
        raise RuntimeError(f"{func_pkl}: feature index is not monotonic/unique")
    if len(idx) and not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
        log.warning("%s: feature index has gaps -> toggles span the gaps", func_pkl)

    pwr = pd.read_pickle(pwr_pkl)
    pwr.index = pwr.index.astype(np.int64)
    col = target_col
    if col not in pwr.columns:
        if pwr.shape[1] == 1:                # random-stimulus pwr is single-column
            col = pwr.columns[0]
        else:
            raise RuntimeError(f"{pwr_pkl}: target column {target_col!r} not found")
    y = pwr[col].reindex(feats.index)
    n_missing = int(y.isna().sum())
    if n_missing:
        raise RuntimeError(
            f"{pwr_pkl}: {n_missing} feature cycles have no power label "
            f"(feature index [{idx[0]}..{idx[-1]}] not covered by power index)")
    return feats, y.rename("target")


# ===========================================================================
# Atomic IO
# ===========================================================================

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
    # The atomic write targets a ``.tmp`` file, so pandas cannot infer compression
    # from the final extension -- pass it explicitly for ``.zst`` targets.
    comp = "zstd" if path.endswith(".zst") else "infer"
    _atomic_write(path, lambda tmp: pd.to_pickle(obj, tmp, compression=comp))


def save_json(obj: Any, path: str) -> None:
    def _w(tmp: str) -> None:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
    _atomic_write(path, _w)


def read_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


# ===========================================================================
# Config (per-module YAML) and case enumeration
# ===========================================================================

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_root(root: str | None) -> str | None:
    """Return an existing dataset directory for ``root`` (cwd-independent).

    Tolerates a bare dataset dir name by also trying ``<repo>/dataset/<root>``.
    """
    if not root:
        return root
    for cand in (root, os.path.join(_REPO_ROOT, root),
                 os.path.join(_REPO_ROOT, "dataset", root)):
        if os.path.isdir(cand):
            return cand
    return root


@dataclass(frozen=True)
class Case:
    name: str
    func_pkl: str
    pwr_pkl: str
    target_col: str
    is_test: bool


@dataclass(frozen=True)
class PreprocessConfig:
    db_root: str                       # c906 root, e.g. c906_db_net_1cyc_20260729
    module: str                        # cp0 / idu / ...
    target: str                        # power column, e.g. x_aq_core/Pc(x_aq_cp0_top)
    cases: list[str]                   # all c906 benchmark names
    test_benchmarks: list[str]         # subset held out as the test set
    output_dir: str = "dataset_processed"
    rand_root: str | None = None       # random-stimulus root (train augmentation)
    rand_case_name: str = "random"

    rules: TypeRules = field(default_factory=TypeRules)
    hamming_count_initial: bool = True
    detect_duplicates: bool = True
    dedup_identical: bool = True
    drop_dead_control_bits: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "db_root", _resolve_root(self.db_root))
        object.__setattr__(self, "rand_root", _resolve_root(self.rand_root))

    # ---------------------------------------------------------------- paths
    def func_pkl(self, case: str) -> str:
        return f"{self.db_root}/func/{self.module}/{case}_func.pkl.zst"

    def pwr_pkl(self, case: str) -> str:
        return f"{self.db_root}/pwr/{case}_pwr.pkl.zst"

    def rand_func_pkl(self) -> str:
        return f"{self.rand_root}/{self.module}_random_func.pkl.zst"

    def rand_pwr_pkl(self) -> str:
        return f"{self.rand_root}/{self.module}_random_pwr.pkl.zst"

    @property
    def manifest_path(self) -> str:
        return f"{self.output_dir}/{self.module}/manifest.json"

    def case_out_dir(self, case: Case) -> str:
        split = "testset" if case.is_test else "trainset"
        return f"{self.output_dir}/{split}/{self.module}"

    def iter_cases(self) -> list[Case]:
        """Every case for this module: c906 benchmarks + the random-stimulus case."""
        test = set(self.test_benchmarks)
        out = [Case(c, self.func_pkl(c), self.pwr_pkl(c), self.target, c in test)
               for c in self.cases]
        if self.rand_root and os.path.exists(self.rand_func_pkl()):
            out.append(Case(self.rand_case_name, self.rand_func_pkl(),
                            self.rand_pwr_pkl(), self.target, False))
        return out

    def train_cases(self) -> list[Case]:
        return [c for c in self.iter_cases() if not c.is_test]

    # --------------------------------------------------------------- loading
    @staticmethod
    def from_yaml(path: str) -> "PreprocessConfig":
        with open(path) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        rules_raw = raw.pop("rules", None)
        cfg = PreprocessConfig(**raw)
        if rules_raw:
            cfg = replace(cfg, rules=TypeRules(**{
                k: tuple(v) if isinstance(v, list) else v
                for k, v in rules_raw.items()}))
        return cfg


# ===========================================================================
# Per-feature transform plan
# ===========================================================================

def plan_feature(col: str, cfg: PreprocessConfig, stats: RawStats) -> dict:
    """Category / rule / width + transform + scale plan for one surviving column."""
    path, width, lo = parse_column(col)
    category, rule = classify(col, cfg.rules)
    invert = category == "control" and should_invert(col, cfg.rules)
    plan = {"source_signal": path, "category": category, "classification_rule": rule,
            "width": width, "lo": lo, "invert": invert}
    if category == "control":
        live = (stats.live_control_bits(col, width)
                if cfg.drop_dead_control_bits else list(range(width)))
        plan.update({"transform": "bit_split", "scale": "bits", "scale_divisor": 1,
                     "live_bits": live, "n_dead_bits": width - len(live),
                     "n_features": len(live)})
    elif category == "data":
        plan.update({"transform": "toggle_hd", "scale": "hd", "scale_divisor": width,
                     "n_features": 1})
    else:  # config
        plan.update({"transform": "raw_int", "scale": "raw_int",
                     "scale_divisor": (1 << width) - 1, "n_features": 1})
    return plan


def _raw_name(path: str, width: int, lo: int) -> str:
    return path if width == 1 else f"{path}[{lo + width - 1}:{lo}]"


def transform_signal(values: np.ndarray, plan: dict,
                     count_initial: bool) -> list[tuple[str, np.ndarray]]:
    """Return list of (feature_name, scaled float array) for one source signal."""
    path, width, lo = plan["source_signal"], plan["width"], plan["lo"]
    t = plan["transform"]
    if t == "bit_split":
        return [(name, scale_bits(bit)) for name, bit in
                split_bits(values, path, width, lo, plan["invert"], plan.get("live_bits"))]
    if t == "toggle_hd":
        ham = toggle_hamming(values, width, count_initial)
        return [(f"{path}_toggle_hd", scale_hamming(ham, width))]
    return [(_raw_name(path, width, lo), scale_raw_int(values, width))]


# ===========================================================================
# Fit stage (steps 1-2 + plan) -- train-only decisions -> manifest.json
# ===========================================================================

def fit(cfg: PreprocessConfig, force: bool = False) -> dict:
    train = cfg.train_cases()
    if not train:
        raise SystemExit("no training cases (check cfg.cases / test_benchmarks)")

    # 1. canonical column order + widths, from the SMALLEST train case (cheap read).
    #    Signals below excluded macro subtrees (spec step 2) are dropped up-front.
    excl = cfg.rules.exclude_segment_prefixes
    canon_case = min(train, key=lambda c: os.path.getsize(c.func_pkl))
    raw_cols = sorted(load_features(canon_case.func_pkl).columns)
    excluded = [c for c in raw_cols if is_excluded_column(c, excl)]
    excluded_set = set(excluded)
    canonical = [c for c in raw_cols if c not in excluded_set]
    widths = {c: parse_column(c)[1] for c in canonical}
    control_cols = [c for c in canonical if classify(c, cfg.rules)[0] == "control"]
    log.info("module=%s canonical columns=%d (control=%d) from %s; excluded %d below %s",
             cfg.module, len(canonical), len(control_cols), canon_case.name,
             len(excluded), list(excl))

    # 2. stream TRAIN cases -> raw stats + constancy + dup fingerprints + dead bits.
    stats = RawStats(canonical, widths, control_cols,
                     detect_duplicates=cfg.detect_duplicates,
                     track_bits=cfg.drop_dead_control_bits)
    for case in train:
        df = load_features(case.func_pkl, canonical, excl)
        stats.update(df)
        log.info("fit: folded %s (%d rows)", case.name, df.shape[0])
        del df
        gc.collect()
    finalized = stats.finalize()

    # 3. drop constant (zero-variance / invariant) columns.
    kept = [c for c in canonical if not finalized[c]["constant"]]
    dropped = {c: {"reason": "zero-variance (constant across train)",
                   "raw_min": finalized[c]["raw_min"],
                   "raw_max": finalized[c]["raw_max"], "width": widths[c]}
               for c in canonical if finalized[c]["constant"]}
    log.info("kept %d / %d signals (dropped %d constant)",
             len(kept), len(canonical), len(dropped))

    # 4. exact-duplicate detection / dedup.
    dup_groups = duplicate_groups(finalized, kept) if cfg.detect_duplicates else []
    if cfg.dedup_identical:
        deduped: set[str] = set()
        for group in dup_groups:
            rep = group[0]
            for member in group[1:]:
                dropped[member] = {"reason": f"duplicate of {rep}",
                                   "raw_min": finalized[member]["raw_min"],
                                   "raw_max": finalized[member]["raw_max"],
                                   "width": widths[member]}
                deduped.add(member)
        kept = [c for c in kept if c not in deduped]
    log.info("duplicate groups: %d (dedup=%s)", len(dup_groups), cfg.dedup_identical)

    # 5. plan features + attach raw stats.
    features: dict[str, dict] = {}
    for col in kept:
        plan = plan_feature(col, cfg, stats)
        plan.update({"raw_min": finalized[col]["raw_min"],
                     "raw_max": finalized[col]["raw_max"],
                     "raw_mean": finalized[col]["raw_mean"]})
        if plan["raw_max"] > (1 << plan["width"]) - 1:
            log.warning("%s: raw_max %d exceeds 2^%d-1",
                        col, plan["raw_max"], plan["width"])
        features[col] = plan

    by_cat = {cat: sum(1 for f in features.values() if f["category"] == cat)
              for cat in CATEGORIES}
    manifest = {
        "config": {
            "db_root": cfg.db_root, "module": cfg.module, "target": cfg.target,
            "train_cases": [c.name for c in train],
            "test_benchmarks": cfg.test_benchmarks,
            "hamming_count_initial": cfg.hamming_count_initial,
            "dedup_identical": cfg.dedup_identical,
            "drop_dead_control_bits": cfg.drop_dead_control_bits,
        },
        "n_raw_columns": len(canonical),
        "excluded_segment_prefixes": list(excl),
        "n_excluded_columns": len(excluded),
        "excluded_columns_sample": excluded[:10],
        "n_kept_signals": len(kept),
        "n_signals_by_category": by_cat,
        "n_final_features": sum(f["n_features"] for f in features.values()),
        "count_train_rows": stats.count,
        "canonical_columns": canonical,
        "features": features,
        "dropped": dropped,
        "duplicate_groups": dup_groups,
    }
    save_json(manifest, cfg.manifest_path)
    log.info("wrote manifest -> %s (%d final features across %s)",
             cfg.manifest_path, manifest["n_final_features"], by_cat)
    return manifest


# ===========================================================================
# Apply stage (steps 3-5) -- transform one case, write per-category pkls
# ===========================================================================

def apply_case(case: Case, manifest: dict, cfg: PreprocessConfig,
               force: bool = False) -> dict:
    out_dir = cfg.case_out_dir(case)
    target_path = os.path.join(out_dir, f"{case.name}_target.pkl.zst")
    if os.path.exists(target_path) and not force:
        log.info("apply: skip %s (exists; use --force to rebuild)", case.name)
        return {"case": case.name, "skipped": True}

    feats, y = align_case(case.func_pkl, case.pwr_pkl, case.target_col,
                          manifest["canonical_columns"],
                          cfg.rules.exclude_segment_prefixes)
    index = feats.index

    per_cat: dict[str, dict[str, np.ndarray]] = {c: {} for c in CATEGORIES}
    for col, plan in manifest["features"].items():
        vals = feats.pop(col).to_numpy()     # pop -> release each column's memory
        for name, arr in transform_signal(vals, plan, cfg.hamming_count_initial):
            per_cat[plan["category"]][name] = arr.astype(_STORE_DTYPE)
        del vals
    del feats
    gc.collect()

    ensure_dir(out_dir)
    counts = {}
    for cat in CATEGORIES:
        cols = per_cat[cat]
        counts[cat] = len(cols)
        if not cols:
            continue
        df = pd.DataFrame(cols, index=index)
        df.index.name = "time_ns"
        save_pickle(df, os.path.join(out_dir, f"{case.name}_{cat}.pkl.zst"))
        del df
    save_pickle(y.astype(np.float64), target_path)
    gc.collect()
    log.info("apply: %s -> %s  rows=%d counts=%s", case.name, out_dir, len(index), counts)
    return {"case": case.name, "skipped": False, "rows": int(len(index)),
            "feature_counts": counts}


# ===========================================================================
# Orchestration + CLI
# ===========================================================================

def _load_manifest(cfg: PreprocessConfig) -> dict:
    if not os.path.exists(cfg.manifest_path):
        raise SystemExit(f"manifest not found: {cfg.manifest_path} (run --stage fit first)")
    return read_json(cfg.manifest_path)


def _transform(cfg: PreprocessConfig, manifest: dict,
               names: list[str] | None, force: bool) -> None:
    by_name = {c.name: c for c in cfg.iter_cases()}
    selected = [by_name[n] for n in names] if names else list(by_name.values())
    # smallest file first: independent, resumable, giant random frames last.
    selected.sort(key=lambda c: os.path.getsize(c.func_pkl))
    for case in selected:
        apply_case(case, manifest, cfg, force=force)


def run(cfg: PreprocessConfig, stage: str = "all",
        cases: list[str] | None = None, force: bool = False) -> None:
    if stage in ("fit", "all"):
        manifest = fit(cfg, force=force)
    if stage in ("transform", "all"):
        manifest = manifest if stage == "all" else _load_manifest(cfg)
        _transform(cfg, manifest, cases, force)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S")


def main(argv: list[str] | None = None) -> None:
    _setup_logging()
    ap = argparse.ArgumentParser(prog="xopm-data-preprocess")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["fit", "transform", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cases", default=None, help="comma-separated subset for transform")
    args = ap.parse_args(argv)

    cfg = PreprocessConfig.from_yaml(args.config)
    log.info("stage=%s module=%s train=%d test=%d", args.stage, cfg.module,
             len(cfg.train_cases()), len(cfg.test_benchmarks))
    cases = args.cases.split(",") if args.cases else None
    run(cfg, stage=args.stage, cases=cases, force=args.force)
    log.info("done: %s", args.stage)


if __name__ == "__main__":
    main()
