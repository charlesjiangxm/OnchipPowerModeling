#!/bin/bash
set -e
cd /scratch/PI/eeweiz/jjiangan/OnchipPowerModelingNew
mkdir -p output/logs

echo "=== Round 1 (Scheme A): test=conv_softmax ==="
python -m cobit run --config cobit/configs/full_schemeA.yaml --run-name schemeA \
    > output/logs/schemeA.log 2>&1
echo "Round 1 complete. See output/cobit_runs/schemeA/metrics.json"

echo "=== Round 2 (Scheme B): test=conv_softmax,coremark ==="
python -m cobit run --config cobit/configs/full_schemeB.yaml --run-name schemeB \
    > output/logs/schemeB.log 2>&1
echo "Round 2 complete. See output/cobit_runs/schemeB/metrics.json"
