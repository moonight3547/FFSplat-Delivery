#!/usr/bin/env bash
# Campaign D - full quality/time curves on the 4x4090 host (full-speed cards, so
# the wall clock here is directly comparable to the measured FastGS 30K budget).
# Recipe under test: no opacity reset + lambda_dssim 0.4 + position LR schedule
# matched to the iteration count.
#
#   bash research/experiments/campaign_d.sh <lane> <nlanes>
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}
OUT=${OUT:-outputs/experiments/campaignD}
LANE=${1:-0}
NLANES=${2:-4}
mkdir -p "$OUT"

HW=hw/billboard;   BHW=99.49
CO=mipnerf360/counter; BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29

# tag|case|iterations|budget
JOBS=(
  "d_hw_3k|$HW|3000|$BHW"
  "d_hw_4k|$HW|4000|$BHW"
  "d_hw_6k|$HW|6000|$BHW"
  "d_hw_8k|$HW|8000|$BHW"
  "d_hw_10k|$HW|10000|$BHW"
  "d_co_3k|$CO|3000|$BCO"
  "d_co_4k|$CO|4000|$BCO"
  "d_co_6k|$CO|6000|$BCO"
  "d_co_8k|$CO|8000|$BCO"
  "d_co_10k|$CO|10000|$BCO"
  "d_dl_4k|$DL|4000|$BDL"
  "d_dl_6k|$DL|6000|$BDL"
  "d_dl_8k|$DL|8000|$BDL"
  "d_dl_10k|$DL|10000|$BDL"
  "d_dl_12k|$DL|12000|$BDL"
)

i=0
for job in "${JOBS[@]}"; do
  i=$((i + 1))
  [ $(( (i - 1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case iters budget <<<"$job"
  [ -e "$OUT/$tag/summary.json" ] && continue
  echo "start $tag lane$LANE $(date -Is)" >>"$OUT/queue.log"
  $PY run --no-capture-output -n fastgs python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$LANE.json" \
    --case "$case" --work "outputs/O0-20261030/work/$case" \
    --init-ply "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" \
    --out "$OUT/$tag" --iterations "$iters" --snapshots "" --budget "$budget" \
    --opacity-reset-interval 0 --lambda-dssim 0.4 --position-lr-max-steps "$iters" \
    >"$OUT/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$OUT/queue.log"
done
echo "lane$LANE done $(date -Is)" >>"$OUT/queue.log"
