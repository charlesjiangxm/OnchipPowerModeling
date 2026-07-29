"""Typed configuration tree loaded from YAML with dotted CLI overrides.

Every pipeline stage reads its knobs from one nested :class:`CobitConfig`.
``config_hash()`` of the relevant subtree is stamped into stage artifacts so
reruns can be skipped when nothing they depend on changed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .utils import stable_hash


@dataclass
class DataConfig:
    db_root: str = "c906_db_net_1cyc_20260729_2rows"
    cache_dir: str = "output/cobit_cache"
    target: str = "Pc(x_aq_core)"
    chunk_rows: int = 65536
    min_toggle_count: int = 1  # kept-bit threshold on train toggle counts


@dataclass
class SplitConfig:
    # Benchmarks held out as the test set (never dropped; missing scopes are
    # zero-filled with a warning). Everything else with full coverage trains.
    test_benchmarks: list[str] = field(default_factory=lambda: ["conv_softmax"])
    # Contiguous tail fraction of each training benchmark used as validation.
    val_fraction: float = 0.2


@dataclass
class SelectionConfig:
    selector: str = "mcp"  # "mcp" (skglm) | "lasso" (sklearn fallback)
    gamma: float = 3.0  # MCP concavity (paper's gamma; unstated there)
    target_qs: list[int] = field(
        default_factory=lambda: [50, 100, 150, 200, 250, 300, 370, 400]
    )
    q_tol: float = 0.10  # accept Q within +/-10% of target
    max_bisect: int = 12  # bisection refinements per target Q
    max_rows: int = 200_000  # row subsample cap for the selector fit
    grid_points: int = 30  # coarse alpha grid size
    grid_decades: float = 4.0  # alpha grid spans alpha_max * 10**-decades
    max_iter: int = 100  # solver iterations per alpha
    fit_intercept: bool = True


@dataclass
class HpoConfig:
    sampler: str = "nsga3"  # candidate pair (CP) sampler
    pruner: str = "hyperband"  # "hyperband" | "median" | "none"
    prune_mode: str = "truncate"  # "truncate" | "prune"
    n_trials: int = 256  # trials per (Q, R) study
    population_size: int = 32
    r_rgs: list[int] = field(default_factory=lambda: [20, 30, 40, 50, 60, 70, 80, 90])
    t_th: int = 800  # leaf-count budget for Best Trial selection
    # Normalization reference point for hypervolume: (MAPE %, total leaves).
    hv_ref: list[float] = field(default_factory=lambda: [32.0, 3200.0])
    # Region of interest for the pair comparison (paper Sec. IV-B: [5%, 1000]).
    # Distinct from t_th, which is the Best-Trial leaf budget.
    roi_mape: float = 5.0
    roi_leaves: float = 1000.0
    # --- Algorithm 2 sampler-pruner pair comparison (optional stage) ---
    run_pair_comparison: bool = False
    pair_pop_sizes: list[int] = field(default_factory=lambda: [16, 32, 48])
    pair_n_trials: int = 512  # paper used 8192/pair on a cluster
    pair_q: int | None = None  # Q for the comparison (paper: 197); None -> first target_qs
    pair_r: int = 50  # R for the comparison


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
    trace_plot_cycles: int = 12000
    # MAPE mask: cycles with y <= eps_frac * median(y) are excluded (count reported).
    mape_eps_frac: float = 1e-3


@dataclass
class RuntimeConfig:
    seed: int = 0
    allow_tiny: bool = False  # relax degenerate-data guards for the 2-row sample
    output_dir: str = "output/cobit_runs"
    run_name: str = ""  # default: <config_stem>_<timestamp> chosen by the CLI


@dataclass
class CobitConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    hpo: HpoConfig = field(default_factory=HpoConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    config_path: str = ""  # set by from_yaml

    # -- construction ------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> "CobitConfig":
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

    # -- serialization / hashing -------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash_of(self, *sections: str) -> str:
        """Hash of one or more config sections (for artifact stamps)."""
        d = self.to_dict()
        return stable_hash({s: d[s] for s in sections})

    def stage_hash(self, *sections: str) -> str:
        """Section hash plus the runtime knobs that change results.

        runtime.seed drives Stage-1 subsampling and the HPO sampler;
        runtime.allow_tiny changes selection fallbacks. output_dir/run_name
        are deliberately excluded so relocating a run does not invalidate it.
        """
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
        ftype = fields[name].type
        # nested dataclass section
        target = _SECTION_TYPES.get((dc_type.__name__, name))
        if target is not None:
            kwargs[name] = _from_dict(target, value or {}, ctx=f"{ctx}:{name}")
        else:
            kwargs[name] = value
        del ftype
    return dc_type(**kwargs)


_SECTION_TYPES = {
    ("CobitConfig", "data"): DataConfig,
    ("CobitConfig", "split"): SplitConfig,
    ("CobitConfig", "selection"): SelectionConfig,
    ("CobitConfig", "hpo"): HpoConfig,
    ("CobitConfig", "train"): TrainConfig,
    ("CobitConfig", "eval"): EvalConfig,
    ("CobitConfig", "runtime"): RuntimeConfig,
}


def _apply_override(cfg: CobitConfig, override: str) -> None:
    """Apply a 'section.key=value' override; value parsed as a Python literal."""
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
    # YAML semantics so 'false'/'true'/'null'/lists parse the same way they
    # would in the config file (ast.literal_eval would keep 'false' a string)
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        value = text  # bare string
    setattr(obj, key, value)
