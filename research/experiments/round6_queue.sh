#!/usr/bin/env bash
# Round 6 - queue the A-side selection study and the time-budgeted MCMC B study
# after the FastGS rounds have released the GPU (one GPU job at a time keeps the
# timing numbers comparable).
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
OUT=${OUT:-outputs/experiments/round6}
mkdir -p "$OUT"
log="$OUT/run.log"
echo "start $(date -Is)" | tee -a "$log"

while pgrep -f "round2_b_variants|round3_recipe" >/dev/null 2>&1; do sleep 15; done

echo "=== round4 A-selection $(date -Is)" >>"$log"
RUN_GPU=1 bash research/experiments/round4_a_selection.sh hw/billboard >>"$log" 2>&1
echo "round4_hw_exit=$?" >>"$log"
RUN_GPU=1 bash research/experiments/round4_a_selection.sh mipnerf360/counter >>"$log" 2>&1
echo "round4_counter_exit=$?" >>"$log"

echo "=== round5 MCMC budget B $(date -Is)" >>"$log"
bash research/experiments/round5_mcmc.sh >>"$log" 2>&1
echo "round5_exit=$?" >>"$log"
echo "done $(date -Is)" >>"$log"
