#!/usr/bin/env python
"""Regenerate the power-vs-time plots for EXISTING x-opm result folders without
retraining, then refresh each ``report.md``.

For every ``<base>/{typeAB,typeABC,typeABCD}/{cobit,rulefit}`` folder we rebuild
the (deterministic) dataset and recover the model's predictions:

  * cobit   -- reload ``model.xgb.json`` and multiply by the saved
               ``target_train_scale`` (exact; no retraining / Optuna).
  * rulefit -- re-fit the seeded RuleFit into a scratch dir (deterministic; the
               model was not persisted). ``coefficient.csv`` is NOT touched.

Only ``power_{split}.png`` are (re)written; ``metrics.json`` / ``coefficient.csv``
are left as-is. Stale ``residual_{split}.png`` are removed. Reconstructed R2 is
printed next to the recorded R2 as a sanity check. Finally ``make_train_report``
rebuilds each folder's ``report.md``.

Usage:
    ~/anaconda3/bin/python x-opm/regen_power_plots.py \
        --bases out/x-opm/results out/x-opm/results2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402  (x-opm/train.py)

TYPESETS = ["typeAB", "typeABC", "typeABCD"]
BACKENDS = ["cobit", "rulefit"]
REPO = Path(__file__).resolve().parents[1]


def _meta(folder: Path):
    m = json.loads((folder / "metrics.json").read_text())
    argv = json.loads((folder / "run_config.json").read_text())["argv"]
    return m, argv


def _preds_cobit(folder: Path, data, feature_names):
    Xtr, ytr, Xval, yval, Xte, yte = data
    m, _ = _meta(folder)
    yscale = float(m["extra"]["target_train_scale"])
    safe, _back = T._sanitize_names(feature_names)
    booster = xgb.Booster()
    booster.load_model(str(folder / "model.xgb.json"))
    pr = lambda X: booster.predict(xgb.DMatrix(X, feature_names=safe)) * yscale
    return {"train": pr(Xtr), "val": pr(Xval), "test": pr(Xte)}


def _preds_rulefit(data, feature_names, argv):
    ns = SimpleNamespace(
        seed=int(argv.get("seed", 0)),
        rulefit_max_rows=int(argv.get("rulefit_max_rows", 200_000)),
        max_rules=int(argv.get("max_rules", 500)),
    )
    with tempfile.TemporaryDirectory() as tmp:
        preds, _extra = T.run_rulefit(data, feature_names, ns, Path(tmp))
    return preds


def regen_base(base: Path):
    print(f"== {base}")
    for ts in TYPESETS:
        present = [b for b in BACKENDS if (base / ts / b / "metrics.json").exists()]
        if not present:
            print(f"  {ts}: no backends present, skip")
            continue
        # dataset config is shared across backends of a typeset -> build once.
        m0, argv0 = _meta(base / ts / present[0])
        types = list(m0["types"])
        window = int(m0.get("window") or argv0.get("window") or 1)
        val_fraction = float(argv0.get("val_fraction", 0.2))
        print(f"  {ts}: types={m0['types']} window={window} val_fraction={val_fraction}")
        Xtr, ytr, Xval, yval, Xte, yte, feat, slices = T.build_dataset(
            types, val_fraction, window)
        data = (Xtr, ytr, Xval, yval, Xte, yte)
        ys = {"train": ytr, "val": yval, "test": yte}

        for b in present:
            folder = base / ts / b
            m, argv = _meta(folder)
            if b == "cobit":
                preds = _preds_cobit(folder, data, feat)
            else:
                preds = _preds_rulefit(data, feat, argv)
            for s in ("train", "val", "test"):
                r2_rec = T.r2_score(ys[s], preds[s])
                r2_saved = m["metrics"][s]["r2"]
                flag = "" if abs(r2_rec - r2_saved) < 5e-3 else "  <-- MISMATCH"
                print(f"    {b:8s} {s:5s}: R2 recon={r2_rec:+.4f} "
                      f"recorded={r2_saved:+.4f}{flag}")
            T.make_plots(folder, b, m["types"],
                         {s: (ys[s], preds[s], slices[s]) for s in ys},
                         window=window)
            for s in ("train", "val", "test"):
                stale = folder / f"residual_{s}.png"
                if stale.exists():
                    stale.unlink()
            print(f"    {b:8s}: wrote power_*.png, removed residual_*.png")

        # rebuild report.md for this typeset folder
        r = subprocess.run(
            [sys.executable, str(REPO / "x-opm" / "make_train_report.py"),
             "--folder", str(base / ts)],
            capture_output=True, text=True)
        print(f"    report: {r.stdout.strip() or r.stderr.strip()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", required=True)
    args = ap.parse_args()
    for b in args.bases:
        p = Path(b)
        if not p.is_absolute():
            p = REPO / p
        regen_base(p)


if __name__ == "__main__":
    main()
