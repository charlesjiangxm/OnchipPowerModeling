#!/usr/bin/env bash
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/configs"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

shopt -s nullglob
configs=( "$CONFIGS_DIR"/*.yaml )
total=${#configs[@]}
if [[ $total -eq 0 ]]; then
    echo "no configs found in $CONFIGS_DIR" >&2
    exit 1
fi

i=0
fails=0
for cfg in "${configs[@]}"; do
    i=$((i+1))
    name=$(basename "$cfg" .yaml)
    echo "[$(date '+%F %T')] ($i/$total) $name"
    if python "$SCRIPT_DIR/run_fit.py" --config "$cfg" >"$LOG_DIR/$name.log" 2>&1; then
        echo "  OK"
    else
        echo "  FAIL (see $LOG_DIR/$name.log)"
        fails=$((fails+1))
    fi
done

echo "done: $((total-fails))/$total succeeded, $fails failed"
exit $(( fails > 0 ? 1 : 0 ))
