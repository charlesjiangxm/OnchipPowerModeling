#!/usr/bin/env bash
# Run the learnability screen (4 steps) for every aq_core submodule, all modules in
# parallel. Each module runs its 4 steps sequentially; the 8 modules run concurrently.
#
#   bash x-opm/learnability/run_all_modules.sh
#
# Env overrides:
#   PY            python interpreter (default ~/anaconda3/bin/python)
#   MODULES       space-separated module list
#   CASES         comma-separated benchmark override (default: auto-discover per module)
#   CASE_WORKERS  per-module internal parallelism over benchmarks (default 4)
set -u

cd "$(dirname "$0")/../.." || exit 1
REPO="$(pwd)"
PY="${PY:-$HOME/anaconda3/bin/python}"
LDIR="$REPO/x-opm/learnability"
MODULES="${MODULES:-cp0 idu ifu iu lsu rtu vidu vpu}"
export CASE_WORKERS="${CASE_WORKERS:-4}"
CASES_ARG=""
[ -n "${CASES:-}" ] && CASES_ARG="--cases $CASES"

echo "== learnability run  modules=[$MODULES]  CASE_WORKERS=$CASE_WORKERS =="

run_module() {
    local m="$1"
    local out="$REPO/out/x-opm/${m}_activity"
    mkdir -p "$out"
    local log="$out/run.log"
    {
        echo "===== module $m  $(date) ====="
        for step in compute_series compute_signal_stats correlate_toggle_power plot_relevance; do
            echo "--- $step ---"
            "$PY" "$LDIR/$step.py" --module "$m" $CASES_ARG || {
                echo "!! $step FAILED for $m"; exit 1; }
        done
        echo "===== module $m DONE  $(date) ====="
    } >"$log" 2>&1
}

declare -A PID2MOD
pids=()
for m in $MODULES; do
    run_module "$m" &
    PID2MOD[$!]=$m
    pids+=($!)
done

echo "== launched ${#pids[@]} modules; waiting =="
rc=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then
        echo "  [OK]   ${PID2MOD[$pid]}"
    else
        echo "  [FAIL] ${PID2MOD[$pid]}  (see out/x-opm/${PID2MOD[$pid]}_activity/run.log)"
        rc=1
    fi
done
echo "== all modules finished (rc=$rc) =="
exit $rc
