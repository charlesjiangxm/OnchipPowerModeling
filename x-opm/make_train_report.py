#!/usr/bin/env python
"""Assemble one type-set folder's ``report.md`` from the two backends' outputs.

Reads ``<folder>/cobit/`` and ``<folder>/rulefit/`` (each with metrics.json,
coefficient.csv, and the power-vs-time/scatter PNGs) and writes
``<folder>/report.md``: dataset name + dimensions, a model-comparison table
(train/val/test R2, MAPE, RMSE), embedded power-vs-time (true vs predicted, mW)
+ scatter plots, top-K coefficient names + values per model, and the parameters
used.

Usage:
    ~/anaconda3/bin/python x-opm/make_train_report.py --folder out/x-opm/results/typeAB [--top 25]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BACKENDS = ["cobit", "rulefit"]
SPLITS = ["train", "val", "test"]


def _load(folder: Path, backend: str):
    d = folder / backend
    mfile = d / "metrics.json"
    if not mfile.exists():
        return None
    metrics = json.loads(mfile.read_text())
    coef = None
    cfile = d / "coefficient.csv"
    if cfile.exists():
        coef = pd.read_csv(cfile)
    return metrics, coef


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    folder = Path(args.folder)
    typeset = folder.name

    loaded = {b: _load(folder, b) for b in BACKENDS}
    present = [b for b in BACKENDS if loaded[b] is not None]

    L = []
    L.append(f"# x-opm power model — feature set `{typeset}`\n")
    L.append("Comparison of two model families trained on the x-opm CP0 typed "
             "dataset, restricted to this feature-type subset.\n")

    # ---- dataset section (from whichever backend is present) ----
    ref = loaded[present[0]][0] if present else None
    win = int(ref.get("window", 1)) if ref else 1
    if ref:
        ds = ref["dataset"]
        unit = ds.get("row_unit", "cycle")
        L.append("## Dataset\n")
        L.append(f"- **Name:** {ds['name']}")
        L.append(f"- **Target:** `{ds['target']}`")
        if win > 1:
            L.append(f"- **Aggregation:** mean over non-overlapping **{win}-cycle "
                     f"windows** (features and target), per benchmark")
        L.append(f"- **Feature types used:** {typeset.replace('type','')} "
                 f"(counts: {ds['n_features_by_type']}) → **{ds['n_features']} features**")
        tb = ds.get("test_benchmark", ds.get("test_benchmarks"))
        if isinstance(tb, (list, tuple)):
            tb = ", ".join(map(str, tb))
        L.append(f"- **Rows ({unit}s):** train={ds['n_train_rows']:,}, "
                 f"val={ds['n_val_rows']:,}, test={ds['n_test_rows']:,} "
                 f"(test benchmark = `{tb}`, leave-benchmarks-out)\n")

    # ---- metrics table ----
    L.append("## Metrics (R² / MAPE% / RMSE)\n")
    header = "| model | split | R² | MAPE (%) | RMSE |"
    L.append(header)
    L.append("|---|---|---|---|---|")
    for b in present:
        m = loaded[b][0]["metrics"]
        for s in SPLITS:
            L.append(f"| {b} | {s} | {_fmt(m[s]['r2'])} | "
                     f"{_fmt(m[s]['mape'], 2)} | {_fmt(m[s]['rmse'], 6)} |")
        g = loaded[b][0].get("extra", {}).get("gbdt")
        if g and g.get("metrics"):
            for s in SPLITS:
                gm = g["metrics"][s]
                L.append(f"| {b}-gbdt | {s} | {_fmt(gm['r2'])} | "
                         f"{_fmt(gm['mape'], 2)} | {_fmt(gm['rmse'], 6)} |")
    L.append("")

    # ---- predictability diagnostic ----
    L.append("## Predictability note (target diagnostic)\n")
    if win > 1:
        L.append(f"This run averages both features and the target `Pc(x_aq_cp0_top)` "
                 f"over non-overlapping **{win}-cycle windows**. Per-cycle, this cp0 "
                 f"target is near-unlearnable (max feature corr ≈0.01, in-sample "
                 f"XGBoost train R²≈0.001, trace lag-1 autocorr −0.65). Window-"
                 f"averaging suppresses the cycle-alternating clock artifact and lifts "
                 f"predictability — total signal toggle activity vs cp0 power rises "
                 f"from ≈0.02 (per-cycle) to ≈0.29 at 100-cycle and ≈0.43 at 500-cycle "
                 f"windows; {win} cycles sits in that range, so expect a **modest "
                 f"positive R²**, well above the per-cycle ~0 (see `out/x-opm/results/`).\n")
        L.append("For a strongly predictable target, whole-core `Pc(x_aq_core)` "
                 "correlates ≈0.90 with activity at 500-cycle windows.\n")
    else:
        L.append("Per-cycle cp0 power `Pc(x_aq_cp0_top)` is near-unlearnable from "
                 "these signals (max feature corr ≈0.01, in-sample XGBoost train "
                 "R²≈0.001, trace lag-1 autocorr −0.65), so R² is ~0 across all "
                 "models/feature-sets — a property of the target, not the pipeline. "
                 "More predictable: window-average the target, or target whole-core "
                 "`Pc(x_aq_core)`.\n")

    # ---- per-backend detail ----
    for b in present:
        metrics, coef = loaded[b]
        extra = metrics.get("extra", {})
        L.append(f"## {b}\n")
        L.append(f"- elapsed: {metrics.get('elapsed_sec','?')} s")
        if b == "cobit":
            L.append(f"- backend: XGBoost (cobit/model.py) — no monotone constraint "
                     f"(matches cobit.pdf), base_score=0, Optuna trials="
                     f"{extra.get('n_trials')}, rounds={extra.get('num_rounds')}")
            L.append(f"- best_params: `{extra.get('best_params')}`")
            L.append("- **\"Coefficients\" = per-feature gain importance** "
                     "(a tree model has no linear coefficients).")
        else:
            fr = extra.get("fit_rows")
            fr_txt = f"{fr:,}" if fr else "?"
            L.append(f"- backend: RuleFit (third_party/rulefit) — positive coefficients "
                     f"(monotonicity via {extra.get('monotonicity','positive-coef')}); "
                     f"fitted intercept ({extra.get('intercept_note','leakage floor')}); "
                     f"max_rules={extra.get('max_rules')}; fit_rows={fr_txt}")
            L.append(f"- intercept={_fmt(extra.get('intercept'),6)}, "
                     f"terms kept={extra.get('n_terms_kept')} "
                     f"({extra.get('n_rule_terms')} rules + "
                     f"{extra.get('n_linear_terms')} linear)")
            pen = extra.get("penalty", "l1")
            if pen == "elasticnet":
                reg = (f"ElasticNetCV (l1_ratio={_fmt(extra.get('l1_ratio'), 3)}, "
                       f"alpha={_fmt(extra.get('alpha'), 6)})")
            else:
                reg = f"LassoCV (alpha={_fmt(extra.get('alpha'), 6)})"
            L.append(f"- regularizer: {reg}")
            g = extra.get("gbdt")
            if g:
                gm = g.get("metrics", {})
                L.append(f"- **GBDT head** on transformed features "
                         f"({g.get('features')}: {g.get('n_features')} feats = "
                         f"{g.get('n_linear_terms')} linear + "
                         f"{g.get('n_selected_rules')}/{g.get('n_rules_total')} rules; "
                         f"Optuna trials={g.get('n_trials')}, "
                         f"rounds={g.get('num_rounds')}) → see `{b}/gbdt/`")
                if gm:
                    L.append(f"  - GBDT R²: train={_fmt(gm.get('train', {}).get('r2'))}, "
                             f"val={_fmt(gm.get('val', {}).get('r2'))}, "
                             f"test={_fmt(gm.get('test', {}).get('r2'))}")

        # power-vs-time + scatter images
        xunit = f"{win}-cycle window index" if win > 1 else "cycle index"
        L.append(f"\n**Power vs time** (true & predicted power in mW vs {xunit}):\n")
        for s in SPLITS:
            L.append(f"![{b} power {s}]({b}/power_{s}.png)")
        L.append("\n**Predicted vs true:**\n")
        for s in SPLITS:
            L.append(f"![{b} scatter {s}]({b}/scatter_{s}.png)")
        if extra.get("gbdt"):
            L.append("\n**GBDT head — power vs time & scatter:**\n")
            for s in SPLITS:
                L.append(f"![{b}-gbdt power {s}]({b}/gbdt/power_{s}.png)")
            for s in SPLITS:
                L.append(f"![{b}-gbdt scatter {s}]({b}/gbdt/scatter_{s}.png)")

        # top coefficients
        L.append(f"\n**Top {args.top} coefficients** (`{b}/coefficient.csv`):\n")
        if coef is not None and len(coef):
            if b == "cobit":
                view = coef.head(args.top)[["feature", "gain", "weight", "cover"]]
            else:
                cols = [c for c in ["rule", "type", "coef", "support", "importance"]
                        if c in coef.columns]
                view = coef.head(args.top)[cols]
            L.append(view.to_markdown(index=False))
        else:
            L.append("_(no non-zero coefficients)_")
        L.append("")

    (folder / "report.md").write_text("\n".join(L) + "\n")
    print(f"wrote {folder / 'report.md'}  (backends: {present})")


if __name__ == "__main__":
    main()
