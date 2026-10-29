#!/usr/bin/env bash
# Campaign A - timing-accurate module-B sweeps on the dedicated (fast) 4090 host.
# Wall clock here is comparable to the measured FastGS random-30K budget, so these
# rows carry the proxy score; the throttled second box only supplies quality.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
OUT=${OUT:-outputs/experiments/campaignA}
mkdir -p "$OUT"

HW=hw/billboard
CO=mipnerf360/counter
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f

# tag|case|iterations|snapshots|budget|extra
JOBS=(
  "a_hw_dssim04_4k|$HW|4000|4000|99.49|--opacity-reset-interval 0 --lambda-dssim 0.4"
  "a_hw_dssim04_6k|$HW|6000|6000|99.49|--opacity-reset-interval 0 --lambda-dssim 0.4"
  "a_hw_dssim04_8k|$HW|8000|8000|99.49|--opacity-reset-interval 0 --lambda-dssim 0.4"
  "a_hw_dssim04_10k|$HW|10000|10000|99.49|--opacity-reset-interval 0 --lambda-dssim 0.4"
  "a_co_dssim04_8k|$CO|8000|8000|92.57|--opacity-reset-interval 0 --lambda-dssim 0.4"
  "a_co_dssim04_12k|$CO|12000|12000|92.57|--opacity-reset-interval 0 --lambda-dssim 0.4"
  "a_dl_dssim04_8k|$DL|8000|8000|94.29|--opacity-reset-interval 0 --lambda-dssim 0.4"
  "a_dl_dssim04_12k|$DL|12000|12000|94.29|--opacity-reset-interval 0 --lambda-dssim 0.4"
  "a_hw_dssim04_8k_lr8k|$HW|8000|8000|99.49|--opacity-reset-interval 0 --lambda-dssim 0.4 --position-lr-max-steps 8000"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r tag case iters snaps budget extra <<<"$job"
  dir="$OUT/$tag"
  if [ -e "$dir/summary.json" ]; then continue; fi
  echo "start $tag $(date -Is)" >>"$OUT/queue.log"
  # shellcheck disable=SC2086
  $PY run --no-capture-output -n ffsplat-enhance python -m research.run_sweep \
    --case "$case" --work "outputs/O0-20261030/work/$case" \
    --init-ply "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" \
    --out "$dir" --iterations "$iters" --snapshots "$snaps" --budget "$budget" \
    $extra >"$OUT/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$OUT/queue.log"
done
echo "campaignA done $(date -Is)" >>"$OUT/queue.log"
