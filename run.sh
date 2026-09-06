#!/bin/bash
set -euo pipefail

ROOT=/ic/projects/A513software/jingbo.jiang/OnchipPowerModeling
RUN="python $ROOT/src/binary_fit/run.py"
CFG=$ROOT/src/binary_fit/configs
WIN=4
QS=(1 5)

# Stage 0: build the single-bit dataset (config-agnostic; build sections are identical)
$RUN --build_db --config $CFG/cobit.yaml

# Stages 1+2 for one (method, model, Q); each logs to <outdir>/<method>.log
run_method() {
  local m=$1 model=$2 q=$3 out=$4
  $RUN --feature_select --config $CFG/$m.yaml --outdir "$out/$m" --window-size $WIN
  $RUN --fit            --config $CFG/$m.yaml --outdir "$out/$m" --window-size $WIN \
       --model $model -q $q
}

pids=()
for Q in "${QS[@]}"; do
  OUT=$ROOT/analysis/binary-fit/2026-09-04-18-${Q}proxy-4cyc
  mkdir -p "$OUT"
  run_method cobit tree  "$Q" "$OUT" > "$OUT/cobit.log" 2>&1 & pids+=($!)
  run_method nn    nn    "$Q" "$OUT" > "$OUT/nn.log"    2>&1 & pids+=($!)
  run_method ridge ridge "$Q" "$OUT" > "$OUT/ridge.log" 2>&1 & pids+=($!)
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc -eq 0 ] || echo "one or more runs failed - see $ROOT/analysis/binary-fit/2026-09-04-18-*proxy-4cyc/*.log" >&2
exit $rc
