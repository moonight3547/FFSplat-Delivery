#!/usr/bin/env bash
# Campaign B - wide module-B sweep on the second 4090 box (2 lanes, quality-first).
#
# This box is a shared/throttled vGPU (~4 TFLOPS vs ~50 on the dedicated host), so
# wall-clock numbers here are NOT comparable to the calibrated host. We only use
# it for *quality at a fixed iteration count*; every run is tagged `nb_` and the
# scoring must re-measure any winner on a fast host.
#
#   bash research/experiments/campaign_b.sh 0   # lane 0
#   bash research/experiments/campaign_b.sh 1   # lane 1
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}
OUT=${OUT:-outputs/experiments/campaignB}
LANE=${1:-0}
mkdir -p "$OUT"

HW=hw/billboard
CO=mipnerf360/counter
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f

# tag|case|iterations|extra args (space separated)
JOBS=(
  "nb_hw_dssim06|$HW|8000|--opacity-reset-interval 0 --lambda-dssim 0.6"
  "nb_hw_dssim08|$HW|8000|--opacity-reset-interval 0 --lambda-dssim 0.8"
  "nb_hw_dssim04_lr8k|$HW|8000|--opacity-reset-interval 0 --lambda-dssim 0.4 --position-lr-max-steps 8000"
  "nb_hw_dssim04_mult10|$HW|8000|--opacity-reset-interval 0 --lambda-dssim 0.4 --mult 1.0"
  "nb_hw_dssim04_dens|$HW|8000|--opacity-reset-interval 0 --lambda-dssim 0.4 --densify-from 100 --densify-interval 50"
  "nb_hw_dssim04_12k|$HW|12000|--opacity-reset-interval 0 --lambda-dssim 0.4"
  "nb_hw_dssim04_4k|$HW|4000|--opacity-reset-interval 0 --lambda-dssim 0.4"
  "nb_co_dssim04|$CO|8000|--opacity-reset-interval 0 --lambda-dssim 0.4"
  "nb_co_dssim06|$CO|8000|--opacity-reset-interval 0 --lambda-dssim 0.6"
  "nb_co_noreset|$CO|8000|--opacity-reset-interval 0"
  "nb_dl_dssim04|$DL|8000|--opacity-reset-interval 0 --lambda-dssim 0.4"
  "nb_dl_dssim06|$DL|8000|--opacity-reset-interval 0 --lambda-dssim 0.6"
  "nb_dl_noreset12k|$DL|12000|--opacity-reset-interval 0"
  "nb_hw_dssim04_sh500|$HW|8000|--opacity-reset-interval 0 --lambda-dssim 0.4 --sh-up-every 500"
  "nb_hw_dssim04_densint200|$HW|8000|--opacity-reset-interval 0 --lambda-dssim 0.4 --densify-interval 200"
  "nb_co_dssim04_12k|$CO|12000|--opacity-reset-interval 0 --lambda-dssim 0.4"
  "nb_dl_dssim04_12k|$DL|12000|--opacity-reset-interval 0 --lambda-dssim 0.4"
  "nb_hw_dssim05_nodensify|$HW|8000|--opacity-reset-interval 0 --lambda-dssim 0.5"
)

i=0
for job in "${JOBS[@]}"; do
  i=$((i + 1))
  [ $((i % 2)) -eq "$LANE" ] || continue
  IFS='|' read -r tag case iters extra <<<"$job"
  dir="$OUT/$tag"
  if [ -e "$dir/summary.json" ]; then continue; fi
  echo "start $tag lane$LANE $(date -Is)" >>"$OUT/queue.log"
  # shellcheck disable=SC2086
  $PY run --no-capture-output -n fastgs python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$LANE.json" \
    --case "$case" --work "outputs/O0-20261030/work/$case" \
    --init-ply "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" \
    --out "$dir" --iterations "$iters" --snapshots "" --budget 95.0 \
    $extra >"$OUT/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$OUT/queue.log"
done
echo "lane$LANE done $(date -Is)" >>"$OUT/queue.log"
