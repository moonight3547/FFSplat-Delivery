#!/usr/bin/env bash
# Campaign E - squeeze quality at the winning short-run operating point.
# Campaign D: noreset + lambda_dssim 0.4 + LR schedule matched to N iterations gives
#   DL3DV  4k/30.6s -> 76.3
#   hw     4k/26.8s -> 50.1
#   counter 6k/39.3s -> 49.7
# and a uniform 4k step budget is the best single setting (mean 58.2 vs 49.6 baseline).
# This campaign holds that operating point and changes one knob at a time.
#
#   bash research/experiments/campaign_e.sh <lane> <nlanes>
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}
OUT=${OUT:-outputs/experiments/campaignE}
LANE=${1:-0}
NLANES=${2:-4}
mkdir -p "$OUT"

HW=hw/billboard;        BHW=99.49
CO=mipnerf360/counter;  BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
BASE="--opacity-reset-interval 0 --lambda-dssim 0.4"

# tag|case|iterations|budget|extra
JOBS=(
  "e_hw_4k_d06|$HW|4000|$BHW|--lambda-dssim 0.6"
  "e_hw_4k_d08|$HW|4000|$BHW|--lambda-dssim 0.8"
  "e_hw_4k_sh500|$HW|4000|$BHW|--sh-up-every 500"
  "e_hw_4k_dens|$HW|4000|$BHW|--densify-from 100 --densify-interval 50"
  "e_hw_4k_mult10|$HW|4000|$BHW|--mult 1.0"
  "e_hw_5k|$HW|5000|$BHW|"
  "e_hw_4k_scale2x|$HW|4000|$BHW|--scaling-lr 0.01"
  "e_co_6k_d06|$CO|6000|$BCO|--lambda-dssim 0.6"
  "e_co_6k_sh500|$CO|6000|$BCO|--sh-up-every 500"
  "e_co_5k|$CO|5000|$BCO|"
  "e_co_6k_dens|$CO|6000|$BCO|--densify-from 100 --densify-interval 50"
  "e_dl_4k_d06|$DL|4000|$BDL|--lambda-dssim 0.6"
  "e_dl_4k_sh500|$DL|4000|$BDL|--sh-up-every 500"
  "e_dl_4k_dens|$DL|4000|$BDL|--densify-from 100 --densify-interval 50"
  "e_dl_3k|$DL|3000|$BDL|"
  "e_dl_5k|$DL|5000|$BDL|"
)

i=0
for job in "${JOBS[@]}"; do
  i=$((i + 1))
  [ $(( (i - 1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case iters budget extra <<<"$job"
  [ -e "$OUT/$tag/summary.json" ] && continue
  echo "start $tag lane$LANE $(date -Is)" >>"$OUT/queue.log"
  # shellcheck disable=SC2086
  $PY run --no-capture-output -n fastgs python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$LANE.json" \
    --case "$case" --work "outputs/O0-20261030/work/$case" \
    --init-ply "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" \
    --out "$OUT/$tag" --iterations "$iters" --snapshots "" --budget "$budget" \
    $BASE --position-lr-max-steps "$iters" $extra >"$OUT/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$OUT/queue.log"
done
echo "lane$LANE done $(date -Is)" >>"$OUT/queue.log"
