"""Typed configuration tree loaded from YAML with dotted CLI overrides.

Every pipeline stage reads its knobs from one nested :class:`Config`.
``stage_hash()`` of the relevant subtree is stamped into HPO study names so a
resume after a config/proxy change starts fresh instead of replaying stale trials.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .utils import stable_hash


@dataclass
class BuildConfig:
    # Stage 0 (--build_db): where the raw per-cycle signal-state func pkls live
    # and where the materialized single-bit dataset is written.
    source_db_root: str = "dataset/c906_db_net_1cyc_20260729"
    modules: list[str] = field(default_factory=lambda: ["cp0"])  # union of module func/<m>/
    out_root: str = "dataset/c906_db_net_1cyc_singlebit_20260729"
    # Drop bits that are globally constant (never 1, or always 1) across ALL
    # benchmarks -- shrinks the dataset and removes uninformative features. The
    # drop is split-independent. False -> keep every bit (single pass).
    drop_dead_bits: bool = True


@dataclass
class DataConfig:
    target: str = "Pc(x_aq_core)"
    # Non-overlapping window (in cycles) averaged to ONE row before feature
    # selection and the fit: features become per-window bit densities in [0, 1]
    # and the label becomes the mean power over the window. Applied identically
    # to the train/val and test benchmarks; the trailing partial window of each
    # benchmark is dropped. 1 -> per-cycle rows (no aggregation).
    window_size: int = 32
    # Kept-bit floor: a column is used only if 0 < #ones and #ones < n_rows,
    # counted on the PER-CYCLE bits of the training rows (so the surviving column
    # set barely moves with window_size -- only the dropped tail cycles differ).
    # Drops all-zero AND constant-1
    # bits; the latter are collinear with the MCP intercept and carry no
    # information for raw signal STATE features.
    min_toggle_count: int = 1
    # Flat single-scope layout consumed by every stage after --build_db:
    # directories of <bench>_func.pkl.zst (single-bit uint8) and <bench>_pwr.pkl.zst.
    func_dir: str = ""
    pwr_dir: str = ""


@dataclass
class SplitConfig:
    # Benchmarks held out as the test set; everything else with a func pkl trains.
    test_benchmarks: list[str] = field(
        default_factory=lambda: ["conv_softmax", "coremark"]
    )
    # Contiguous tail fraction of each training benchmark used as validation.
    val_fraction: float = 0.2


@dataclass
class SelectionConfig:
    # "mcp" (LR-MCP via skglm, the prior-art selector) | "lasso" (sklearn) |
    # "fsr" (greedy forward selection / OMP: takes Q directly, so target_qs are
    # hit exactly and no alpha grid, gamma or tol is involved at all).
    selector: str = "mcp"
    # MCP concavity. LOWER gamma reaches MORE proxies at a converged tolerance:
    # gamma -> infinity IS Lasso (most shrinkage, fewest entrants) and gamma -> 1+
    # is hard thresholding. Measured on the aq_core single-bit matrix at tol=1e-6,
    # gamma 1.5/3/10/30/100/1000/1e6 -> Q 4559/4990/3459/3219/2174/1014/903.
    gamma: float = 3.0
    # Solver stopping tolerance, passed through to the penalized selectors.
    # MUST be set for skglm: its AndersonCD breaks on the ABSOLUTE test
    # `stop_crit <= tol`, and for a zero coefficient the MCP subdifferential
    # distance is max(0, |grad_j| - alpha). When the target's physical scale makes
    # alpha_max small (1.8e-3 for Pc(x_aq_core) in watts, only 18x skglm's default
    # tol=1e-4), that default exits after ~7 iterations at EVERY alpha with most
    # columns still violating their entry condition -- and reports convergence.
    # That is what pinned Q at 17 for every target_q. Note sklearn's Lasso scales
    # its own tol by ||y||^2/n, so the same number is stricter there.
    tol: float = 1e-6
    target_qs: list[int] = field(default_factory=lambda: [400])  # the max-q set is recorded
    q_tol: float = 0.10  # accept Q within +/-10% of target
    max_bisect: int = 12  # bisection refinements per target Q
    max_rows: int = 200_000  # row subsample cap for the selector fit
    grid_points: int = 30  # coarse alpha grid size
    grid_decades: float = 4.0  # alpha grid spans alpha_max * 10**-decades
    max_iter: int = 100  # solver iterations per alpha
    fit_intercept: bool = True
    # Collapse exact-duplicate columns inside a selected support, keeping the
    # strongest-weighted representative. 80.3% of the aq_core kept bits are exact
    # copies of another bit, so an undeduplicated Q overstates how many distinct
    # signals were found (top-1000 by |corr| holds only 259 distinct vectors).
    dedup_proxies: bool = True
    exact_q: bool = True  # truncate the |weight|-sorted support to exactly target_q
    # Re-solve the SELECTED alpha from a cold start before reporting its support.
    # The descending-alpha sweep warm-starts each fit from the previous one, which
    # makes the search cheap but lands MCP -- a nonconvex penalty -- on a badly
    # non-stationary point: on the aq_core matrix the warm path returned a support
    # with 138,516 excluded columns violating stationarity (max |grad|/alpha = 594)
    # where a cold solve at the same alpha has 16 (max 1.06). One extra fit per
    # target (~15-20 s) buys a support that is actually the penalized solution.
    resolve_cold: bool = True
    # Log the stationarity (KKT) residual of every penalized fit, so a support
    # that is not actually a solution can never ship silently again.
    kkt_report: bool = True


@dataclass
class HpoConfig:
    sampler: str = "nsga3"
    pruner: str = "hyperband"  # "hyperband" | "median" | "none"
    prune_mode: str = "truncate"  # "truncate" | "prune"
    n_trials: int = 256  # trials per (Q, R) study
    population_size: int = 32
    r_rgs: list[int] = field(default_factory=lambda: [20, 30, 40, 50, 60, 70, 80, 90])
    t_th: int = 800  # leaf-count budget for Best Trial selection
    # nn (MLP) HPO: number of Optuna trials and parallel workers.
    nn_n_trials: int = 40
    nn_n_jobs: int = 1


@dataclass
class RidgeConfig:
    # Stage-2 ridge backend (--model ridge). Its L2 strength is chosen by
    # RidgeCV's leave-one-out generalized CV *inside the fitting rows*, not by an
    # optuna study on the validation tail -- so these are not HPO sampler knobs
    # and deliberately do NOT live in HpoConfig. Two reasons: they would be
    # misfiled there, and hpo.study_stamp() hashes the whole `hpo` section into
    # every tree study name, so adding fields to it silently orphans the trials
    # already in an existing analysis/.../optuna.db.
    # Grid bounds are ROW-RELATIVE (multiplied by the fitting row count in
    # models.ridge_alphas): sklearn minimizes a SUM, and train-standardizing makes
    # diag(Z'Z) = n exactly, so a unit-variance direction is shrunk by
    # 1/(1 + alpha_rel). alpha_rel 1e2 -> shrink to 0.0099, 1e-6 -> 0.999999, so
    # the default grid spans "almost fully shrunk" to "effectively OLS" at ANY n.
    # A fixed ABSOLUTE grid cannot: topping out at 1e4 it shrinks to 0.013 at
    # n=135 but only to 0.95 at n=200_000.
    alpha_rel_max: float = 1e2  # top of the log grid, per fitting row
    grid_decades: float = 8.0  # spans alpha_rel_max * 10**-decades -> 1e-6 floor
    grid_points: int = 25
    # The grid floor is bounded away from zero on purpose. 80.3% of the aq_core
    # kept bits are exact copies of another bit (see the README), so X'X is
    # genuinely rank-deficient and the L2 term is what makes the solve well-posed:
    # measured on duplicated columns, alpha=0 returns max|coef| = 6.8e4 where
    # alpha=1e-8 returns 1.5.
    #
    # Seeded row cap for the ridge fit (0 -> every row). RidgeCV's GCV path takes
    # an SVD of the design matrix whose U factor is n_rows x Q float64: 200k x 100
    # is 160 MB, but the window_size=1 x Q=1000 corner would be 16 GB. A linear
    # model with Q <= 1000 is fully determined by a 200k-row sample, and at the
    # default window_size=32 the cap never fires on aq_core (measured: 6895
    # train + 1721 val = 8616 fitting rows).
    max_rows: int = 200_000
    fit_intercept: bool = True


@dataclass
class TrainConfig:
    nthread: int = 0  # 0 -> xgboost default (all cores)
    base_seed: int = 0


@dataclass
class EvalConfig:
    apet: list[float] = field(default_factory=lambda: [0.01, 0.03, 0.05])
    peak_window: int = 1000  # cycles per peak-detection window
    peak_sigma: float = 3.0
    multicycle_windows: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    mape_eps_frac: float = 1e-3  # cycles with y <= eps_frac*median(y) are MAPE-masked


@dataclass
class RuntimeConfig:
    seed: int = 0
    allow_tiny: bool = False  # relax degenerate-data guards (tiny synthetic tests)
    output_dir: str = "analysis/binary_fit"
    run_name: str = ""


@dataclass
class Config:
    build: BuildConfig = field(default_factory=BuildConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    hpo: HpoConfig = field(default_factory=HpoConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    config_path: str = ""  # set by from_yaml

    # -- construction ------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = _from_dict(cls, raw, ctx=str(path))
        cfg.config_path = str(path)
        for ov in overrides or []:
            _apply_override(cfg, ov)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not (0.0 <= self.split.val_fraction < 1.0):
            raise ValueError(
                f"split.val_fraction must be in [0, 1), got {self.split.val_fraction}"
            )
        w = self.data.window_size
        # reject rather than coerce: a float 32.5 would silently floor to 32
        if isinstance(w, bool) or not isinstance(w, int) or w < 1:
            raise ValueError(f"data.window_size must be an integer >= 1 cycle, got {w!r}")
        # A degenerate ridge grid otherwise surfaces from inside sklearn as
        # "Found array with 0 sample(s)", which says nothing about the real cause.
        # The type check is not pedantry: plain YAML needs a SIGNED exponent, so
        # `alpha_rel_max: 1.0e2` loads as the *string* "1.0e2" and would otherwise
        # only fail much later, as a TypeError from numpy.
        r = self.ridge
        grid = {"ridge.alpha_rel_max": r.alpha_rel_max, "ridge.grid_decades": r.grid_decades,
                "ridge.grid_points": r.grid_points, "ridge.max_rows": r.max_rows}
        for key, value in grid.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number, got {value!r} "
                                 f"(a YAML exponent needs a sign: 1.0e+4, not 1.0e4)")
        if not (r.alpha_rel_max > 0 and r.grid_decades >= 0 and r.grid_points >= 1
                and r.max_rows >= 0):
            raise ValueError(
                f"ridge grid must have alpha_rel_max > 0, grid_decades >= 0, "
                f"grid_points >= 1 and max_rows >= 0, got alpha_rel_max={r.alpha_rel_max!r} "
                f"grid_decades={r.grid_decades!r} grid_points={r.grid_points!r} "
                f"max_rows={r.max_rows!r}"
            )

    # -- serialization / hashing -------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash_of(self, *sections: str) -> str:
        d = self.to_dict()
        return stable_hash({s: d[s] for s in sections})

    def stage_hash(self, *sections: str) -> str:
        """Section hash plus the runtime knobs that change results."""
        d = self.to_dict()
        payload = {s: d[s] for s in sections}
        payload["__runtime__"] = {
            "seed": self.runtime.seed,
            "allow_tiny": self.runtime.allow_tiny,
        }
        return stable_hash(payload)


def _from_dict(dc_type: type, raw: dict, ctx: str = "") -> Any:
    """Recursively build a dataclass from a dict, rejecting unknown keys."""
    if not isinstance(raw, dict):
        raise TypeError(f"{ctx}: expected mapping for {dc_type.__name__}, got {type(raw).__name__}")
    fields = {f.name: f for f in dataclasses.fields(dc_type)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise KeyError(f"{ctx}: unknown config keys for {dc_type.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        target = _SECTION_TYPES.get((dc_type.__name__, name))
        if target is not None:
            kwargs[name] = _from_dict(target, value or {}, ctx=f"{ctx}:{name}")
        else:
            kwargs[name] = value
    return dc_type(**kwargs)


_SECTION_TYPES = {
    ("Config", "build"): BuildConfig,
    ("Config", "data"): DataConfig,
    ("Config", "split"): SplitConfig,
    ("Config", "selection"): SelectionConfig,
    ("Config", "hpo"): HpoConfig,
    ("Config", "ridge"): RidgeConfig,
    ("Config", "train"): TrainConfig,
    ("Config", "eval"): EvalConfig,
    ("Config", "runtime"): RuntimeConfig,
}


def _apply_override(cfg: Config, override: str) -> None:
    """Apply a 'section.key=value' override; value parsed with YAML semantics."""
    if "=" not in override:
        raise ValueError(f"override must look like section.key=value, got {override!r}")
    dotted, _, text = override.partition("=")
    parts = dotted.strip().split(".")
    obj: Any = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    key = parts[-1]
    if not hasattr(obj, key):
        raise KeyError(f"unknown config key {dotted!r}")
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        value = text
    setattr(obj, key, value)
