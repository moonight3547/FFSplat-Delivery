#!/usr/bin/env bash
# Sequential campaign orchestrator for the 4-GPU host: one campaign at a time,
# each spread over NLANES cards, joined with `wait` (no pgrep races).
#   bash research/experiments/run_all.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-/opt/conda/bin/conda}
NLANES=${NLANES:-4}

run_campaign() {  # script outdir
  local script="$1" out="$2" pids=() l
  echo "=== $(date -Is) $script -> $out" >> outputs/campaign_all.log
  for l in $(seq 0 $((NLANES - 1))); do
    OUT="$out" PY="$PY" bash "$script" "$l" "$NLANES" >> "outputs/campaign_all_$l.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "=== $(date -Is) done $script" >> outputs/campaign_all.log
}

run_campaign research/experiments/campaign_k.sh outputs/experiments/campaignK
run_campaign research/experiments/campaign_l.sh outputs/experiments/campaignL
run_campaign research/experiments/campaign_i.sh outputs/experiments/campaignI
