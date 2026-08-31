#!/usr/bin/env bash
# Parallel driver for the X-OPM RuleFit pipeline. Sweeps several window sizes and,
# for each, trains all 8 modules concurrently (each capped to $THREADS cores so the
# total stays within a budget). Every (win_size, module) job runs in parallel and
# writes into analysis/x-opm/<ts>/win<w>/<module>; once all modules of a given
# win_size finish, its aq_core reconstruction + top-level report is produced.
#
#   PY=~/anaconda3/bin/python NTRIALS=30 THREADS=6 bash src/xopm_lib/run_all_parallel.sh
#
# 4 win_sizes x 8 modules = 32 jobs. Pick THREADS with your core budget in mind:
# 32 jobs x THREADS cores each = total cores. Override THREADS to change the budget.
set -uo pipefail

PY="${PY:-$HOME/anaconda3/bin/python}"
NTRIALS="${NTRIALS:-30}"
THREADS="${THREADS:-6}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

MODULES=(cp0 idu ifu iu lsu rtu vidu vpu)
WINS=(8 32 64 128)
TS="${OUTDIR:-analysis/x-opm/$(date +%Y-%m-%d-%H-%M)}"
rm -rf "$TS"; mkdir -p "$TS"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$TS/.mpl}"; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
       OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"

echo "[driver] out=$TS  trials=$NTRIALS  threads/job=$THREADS" \
     " wins=${WINS[*]}  modules=${#MODULES[@]}  jobs=$(( ${#WINS[@]} * ${#MODULES[@]} ))"

# Pre-warm the matplotlib font cache once so the parallel workers don't race on it.
"$PY" -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot" >/dev/null 2>&1 || true

declare -a PIDS JOBS
for w in "${WINS[@]}"; do
  WDIR="$TS/win${w}"; mkdir -p "$WDIR"
  for m in "${MODULES[@]}"; do
    "$PY" src/xopm_lib/model_regression.py --module "$m" --outdir "$WDIR" \
          --win-size "$w" --no-clean --no-reconstruct --n-trials "$NTRIALS" \
          > "$WDIR/${m}.log" 2>&1 &
    PIDS+=("$!"); JOBS+=("win${w}/${m}")
    echo "[driver] launched win${w}/${m} (pid $!)"
  done
done

FAIL=0
for i in "${!JOBS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[driver] OK   ${JOBS[$i]}"
  else
    echo "[driver] FAIL ${JOBS[$i]} (see $TS/${JOBS[$i]}.log)"; FAIL=1
  fi
done

echo "[driver] all modules finished; aggregating aq_core reconstruction + report per win"
for w in "${WINS[@]}"; do
  WDIR="$TS/win${w}"
  "$PY" src/xopm_lib/model_regression.py --outdir "$WDIR" --reconstruct-only \
        > "$WDIR/reconstruct.log" 2>&1 \
    && echo "[driver] reconstruct OK   win${w}" \
    || { echo "[driver] reconstruct FAIL win${w}"; FAIL=1; }
done
echo "[driver] DONE -> $TS  (fail=$FAIL)"
