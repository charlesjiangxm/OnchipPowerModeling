#!/usr/bin/env python
"""Shared config for the learnability scripts so they stay module-consistent.

The 4 scripts (compute_series, compute_signal_stats, correlate_toggle_power,
plot_relevance) all target one aq_core submodule at a time. Everything they need is
derived from a single --module argument via the helpers below. Default module is
`vpu` to preserve the original hardcoded behavior.
"""
import argparse
import glob
import os

ROOT = "/ic/projects/A513software/jingbo.jiang/OnchipPowerModeling/dataset/c906_db_net_1cyc_20260729"
OUTROOT = "/ic/projects/A513software/jingbo.jiang/OnchipPowerModeling/out/x-opm"
SCALE = 1000.0  # W -> mW

# canonical benchmark set; a module dir may miss some (lsu/vpu lack conv_softmax) and
# also holds x_aq_*_func.pkl sub-block files which are NOT benchmarks.
BENCHMARKS = ["cache", "conv_softmax", "coremark", "csr", "debug", "exception",
              "interrupt", "ISA_FP", "ISA_INT", "ISA_LS", "ISA_THEAD", "MMU"]

# default per-module concurrency for the heavy pkl-loading steps (each pkl is 1-11 GB)
CASE_WORKERS = int(os.environ.get("CASE_WORKERS", "4"))


def module_dir(m):
    return os.path.join(ROOT, "aq_core", m)


def pwr_dir():
    return os.path.join(ROOT, "pwr")


def pwr_col(m):
    return f"x_aq_core/Pc(x_aq_{m}_top)"


def out_dir(m):
    return os.path.join(OUTROOT, f"{m}_activity")


def discover_cases(m):
    """Benchmark cases present for module m (excludes x_aq_* sub-block pkls)."""
    found = set()
    for fp in glob.glob(os.path.join(module_dir(m), "*_func.pkl.zst")):
        name = os.path.basename(fp)[:-len("_func.pkl.zst")]
        if name in BENCHMARKS:
            found.add(name)
    return [b for b in BENCHMARKS if b in found]


def parse(desc=""):
    """Return (module, cases, out). --cases overrides auto-discovery (comma list)."""
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--module", default="vpu",
                    help="aq_core submodule (default: vpu)")
    ap.add_argument("--cases", default=None,
                    help="comma-separated benchmark override (default: auto-discover)")
    a = ap.parse_args()
    cases = [c for c in a.cases.split(",") if c] if a.cases else discover_cases(a.module)
    out = out_dir(a.module)
    os.makedirs(out, exist_ok=True)
    return a.module, cases, out
