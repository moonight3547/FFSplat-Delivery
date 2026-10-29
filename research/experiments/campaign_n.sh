#!/usr/bin/env bash
# Campaign N - the confirmed strict budget (FastGS random-init 30k ~95 s) recipe grid.
# Recipe: opacity-sharpened ReSplat init + FastGS continuation, no opacity reset,
# lambda_dssim 0.6, scaling_lr 0.01, position-LR schedule matched to the step count.
# This campaign pins the delivery number: one uniform setting over all three cases.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}
ENV=${ENV:-fastgs}
D=${D:-outputs/experiments/campaignN}
LANE=${1:-0}; NLANES=${2:-4}
mkdir -p "$D/ply"
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
BASE="--opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01"
sharp() { local t; t=$(echo "$1" | tr '/' '_'); local o="$D/ply/${t}_s.ply"
  [ -f "$o" ] || $PY run --no-capture-output -n "$ENV" python -m research.ply_ops --in "outputs/O0-20261030/results/$1/resplat/resplat_world.ply" --out "$o" --op sharpen --gain 2.0 >/dev/null 2>&1
  echo "$o"; }
one() { local tag="$1" case="$2" steps="$3" budget="$4"; shift 4
  local dir="$D/$tag"; [ -e "$dir/summary.json" ] && return; [ -e "$dir" ] && rm -rf "$dir"
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  local ply; ply=$(sharp "$case")
  $PY run --no-capture-output -n "$ENV" python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$LANE.json" --case "$case" \
    --work "outputs/O0-20261030/work/$case" --init-ply "$ply" \
    --out "$dir" --iterations "$steps" --snapshots "" --budget "$budget" \
    $BASE --position-lr-max-steps "$steps" "$@" >"$D/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"; }
JOBS=(
  "n_hw_4k|$HW|4000|$BHW|"
  "n_hw_3k|$HW|3000|$BHW|"
  "n_hw_5k|$HW|5000|$BHW|"
  "n_hw_4k_earlydens|$HW|4000|$BHW|--densify-until 1000"
  "n_co_4k|$CO|4000|$BCO|"
  "n_co_5k|$CO|5000|$BCO|"
  "n_co_6k|$CO|6000|$BCO|"
  "n_co_5k_earlydens|$CO|5000|$BCO|--densify-until 1250"
  "n_dl_3k|$DL|3000|$BDL|"
  "n_dl_4k|$DL|4000|$BDL|"
  "n_dl_5k|$DL|5000|$BDL|"
  "n_dl_4k_earlydens|$DL|4000|$BDL|--densify-until 1000"
)
i=0; for job in "${JOBS[@]}"; do i=$((i+1)); [ $(( (i-1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case steps budget extra <<<"$job"; one "$tag" "$case" "$steps" "$budget" $extra; done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
