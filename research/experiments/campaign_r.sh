#!/usr/bin/env bash
# Campaign R - variance + fine step grid around the delivery recipe.
# We need to know whether the +-0.5 point differences between neighbouring step
# counts are real. Same recipe as campaign P (sharpen init, noreset, dssim .6,
# scaling_lr .01, matched LR, structural ops in the first quarter), three seeds
# for the 5k point plus the 4.5k/5.5k neighbours.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}; ENV=${ENV:-fastgs}
D=${D:-outputs/experiments/campaignR}; LANE=${1:-0}; NLANES=${2:-4}
mkdir -p "$D/ply"
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
BASE="--opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01"
sharp(){ local t; t=$(echo "$1"|tr '/' '_'); local o="$D/ply/${t}_s.ply"; [ -f "$o" ] || $PY run --no-capture-output -n "$ENV" python -m research.ply_ops --in "outputs/O0-20261030/results/$1/resplat/resplat_world.ply" --out "$o" --op sharpen --gain 2.0 >/dev/null 2>&1; echo "$o"; }
one(){ local tag="$1" case="$2" steps="$3" budget="$4" seed="$5"; local dir="$D/$tag"
  [ -e "$dir/summary.json" ] && return; [ -e "$dir" ] && rm -rf "$dir"
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  local ply; ply=$(sharp "$case")
  $PY run --no-capture-output -n "$ENV" python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$LANE.json" --case "$case" \
    --work "outputs/O0-20261030/work/$case" --init-ply "$ply" \
    --out "$dir" --iterations "$steps" --snapshots "" --budget "$budget" \
    $BASE --position-lr-max-steps "$steps" --densify-until $((steps/4)) --seed "$seed" \
    >"$D/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"; rm -rf "$dir/model" "$dir/eval" 2>/dev/null; }
JOBS=()
for spec in "hw:$HW:$BHW" "co:$CO:$BCO" "dl:$DL:$BDL"; do
  IFS=':' read -r s case b <<<"$spec"
  JOBS+=("r_${s}_5k_s1|$case|5000|$b|20261015" "r_${s}_5k_s2|$case|5000|$b|1234" "r_${s}_5k_s3|$case|5000|$b|999")
  JOBS+=("r_${s}_45k|$case|4500|$b|20261015" "r_${s}_55k|$case|5500|$b|20261015")
done
i=0; for job in "${JOBS[@]}"; do i=$((i+1)); [ $(( (i-1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case steps budget seed <<<"$job"; one "$tag" "$case" "$steps" "$budget" "$seed"; done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
