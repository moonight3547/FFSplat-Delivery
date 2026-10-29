#!/usr/bin/env bash
# Campaign F - cross-validate the winner knobs from campaign E at each case's
# operating point, plus one more magnitude for the scaling LR.
# Campaign E: hw 4k + scaling_lr 0.01 -> 50.69 (was 50.05); counter 6k + dssim 0.6 -> 49.77.
#
#   bash research/experiments/campaign_f.sh <lane> <nlanes>
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}
OUT=${OUT:-outputs/experiments/campaignF}
LANE=${1:-0}
NLANES=${2:-4}
mkdir -p "$OUT"

HW=hw/billboard;        BHW=99.49
CO=mipnerf360/counter;  BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29

# tag|case|iterations|budget|extra
JOBS=(
  "f_hw_4k_s2x|$HW|4000|$BHW|--scaling-lr 0.01"
  "f_hw_4k_s3x|$HW|4000|$BHW|--scaling-lr 0.015"
  "f_hw_4k_s2x_d06|$HW|4000|$BHW|--scaling-lr 0.01 --lambda-dssim 0.6"
  "f_hw_3k_s2x|$HW|3000|$BHW|--scaling-lr 0.01"
  "f_co_6k_s2x|$CO|6000|$BCO|--scaling-lr 0.01"
  "f_co_4k_s2x|$CO|4000|$BCO|--scaling-lr 0.01"
  "f_co_6k_s2x_d06|$CO|6000|$BCO|--scaling-lr 0.01 --lambda-dssim 0.6"
  "f_dl_4k_s2x|$DL|4000|$BDL|--scaling-lr 0.01"
  "f_dl_3k_s2x|$DL|3000|$BDL|--scaling-lr 0.01"
  "f_dl_4k_s2x_d06|$DL|4000|$BDL|--scaling-lr 0.01 --lambda-dssim 0.6"
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
    --opacity-reset-interval 0 --lambda-dssim 0.4 --position-lr-max-steps "$iters" \
    $extra >"$OUT/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$OUT/queue.log"
done
echo "lane$LANE done $(date -Is)" >>"$OUT/queue.log"
