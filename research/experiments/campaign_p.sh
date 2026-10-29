#!/usr/bin/env bash
# Campaign P - pin the delivery recipe. One uniform setting for all three cases.
#
# Everything here uses the confirmed strict budget (FastGS random-init 30k, ~95 s),
# a timing-valid full-speed host, and the same optimizer base:
#   no opacity reset, lambda_dssim 0.6, scaling_lr 0.01, matched position-LR schedule,
#   opacity-sharpened ReSplat init.
# The grid varies only the two knobs the archive says matter: step count and when
# structural (densify/prune) operations are allowed to run.
#
#   bash research/experiments/campaign_p.sh <lane> <nlanes>
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}
ENV=${ENV:-fastgs}
D=${D:-outputs/experiments/campaignP}
LANE=${1:-0}; NLANES=${2:-4}
mkdir -p "$D/ply"
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
BASE="--opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01"

sharp() {
  local t; t=$(echo "$1" | tr '/' '_'); local o="$D/ply/${t}_s.ply"
  [ -f "$o" ] || $PY run --no-capture-output -n "$ENV" python -m research.ply_ops \
    --in "outputs/O0-20261030/results/$1/resplat/resplat_world.ply" --out "$o" \
    --op sharpen --gain 2.0 >/dev/null 2>&1
  echo "$o"
}

one() {  # tag case steps budget extra...
  local tag="$1" case="$2" steps="$3" budget="$4"; shift 4
  local dir="$D/$tag"; [ -e "$dir/summary.json" ] && return; [ -e "$dir" ] && rm -rf "$dir"
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  local ply; ply=$(sharp "$case")
  $PY run --no-capture-output -n "$ENV" python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$LANE.json" --case "$case" \
    --work "outputs/O0-20261030/work/$case" --init-ply "$ply" \
    --out "$dir" --iterations "$steps" --snapshots "" --budget "$budget" \
    $BASE --position-lr-max-steps "$steps" "$@" >"$D/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"
  rm -rf "$dir/model" "$dir/eval" "$dir/snaps" 2>/dev/null
}

JOBS=()
for spec in "hw:$HW:$BHW" "co:$CO:$BCO" "dl:$DL:$BDL"; do
  IFS=':' read -r short case budget <<<"$spec"
  for steps in 3000 4000 5000 6000; do
    JOBS+=("p_${short}_${steps}_full|$case|$steps|$budget|")
    q=$((steps / 4))
    JOBS+=("p_${short}_${steps}_early|$case|$steps|$budget|--densify-until $q")
  done
done

i=0
for job in "${JOBS[@]}"; do
  i=$((i + 1)); [ $(( (i - 1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case steps budget extra <<<"$job"
  # shellcheck disable=SC2086
  one "$tag" "$case" "$steps" "$budget" $extra
done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
