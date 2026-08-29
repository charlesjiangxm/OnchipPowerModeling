#!/usr/bin/env bash
# Run the six x-opm power-model experiments (cobit + rulefit) x (AB, ABC, ABCD)
# in parallel, then build one report.md per type-set folder.
#
#   bash x-opm/run_experiments.sh
#
# Env overrides: PY, OUTBASE, WINDOW, N_TRIALS, NUM_ROUNDS, RF_MAX_ROWS, MAX_RULES, COBIT_NTHREAD
set -u

cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
PY="${PY:-$HOME/anaconda3/bin/python}"
# OUTBASE may be relative to repo root or absolute.
OUTBASE="${OUTBASE:-out/x-opm/results}"
case "$OUTBASE" in /*) RESULTS="$OUTBASE" ;; *) RESULTS="$REPO/$OUTBASE" ;; esac
LOGS="$RESULTS/logs"
mkdir -p "$LOGS"

WINDOW="${WINDOW:-1}"
N_TRIALS="${N_TRIALS:-30}"
NUM_ROUNDS="${NUM_ROUNDS:-300}"
RF_MAX_ROWS="${RF_MAX_ROWS:-200000}"
MAX_RULES="${MAX_RULES:-500}"
COBIT_NTHREAD="${COBIT_NTHREAD:-20}"
echo "== OUTBASE=$RESULTS  WINDOW=$WINDOW =="

# type-set folder name -> feature-type string
declare -A TYPESETS=( [typeAB]=AB [typeABC]=ABC [typeABCD]=ABCD )

echo "== launching 6 experiments in parallel =="
declare -A PID2NAME
pids=()

for ts in "${!TYPESETS[@]}"; do
  types="${TYPESETS[$ts]}"

  # cobit backend (XGBoost, multi-threaded -> capped)
  out_cobit="$RESULTS/$ts/cobit"
  OMP_NUM_THREADS="$COBIT_NTHREAD" "$PY" "$REPO/x-opm/train.py" \
      --backend cobit --types "$types" --outdir "$out_cobit" --window "$WINDOW" \
      --n-trials "$N_TRIALS" --num-rounds "$NUM_ROUNDS" \
      --nthread "$COBIT_NTHREAD" \
      > "$LOGS/${ts}_cobit.log" 2>&1 &
  p=$!; pids+=("$p"); PID2NAME[$p]="${ts}/cobit"

  # rulefit backend (sklearn, largely single-threaded -> small OMP)
  out_rf="$RESULTS/$ts/rulefit"
  OMP_NUM_THREADS=8 "$PY" "$REPO/x-opm/train.py" \
      --backend rulefit --types "$types" --outdir "$out_rf" --window "$WINDOW" \
      --rulefit-max-rows "$RF_MAX_ROWS" --max-rules "$MAX_RULES" \
      > "$LOGS/${ts}_rulefit.log" 2>&1 &
  p=$!; pids+=("$p"); PID2NAME[$p]="${ts}/rulefit"
done

echo "PIDs: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do
  if wait "$p"; then
    echo "  OK   ${PID2NAME[$p]}"
  else
    echo "  FAIL ${PID2NAME[$p]} (see $LOGS/${PID2NAME[$p]/\//_}.log)"
    fail=1
  fi
done

echo "== building reports =="
for ts in "${!TYPESETS[@]}"; do
  "$PY" "$REPO/x-opm/make_train_report.py" --folder "$RESULTS/$ts" \
      >> "$LOGS/reports.log" 2>&1 && echo "  report: $ts" || echo "  report FAIL: $ts"
done

echo "== done (fail=$fail) =="
exit "$fail"
