#!/usr/bin/env bash
# Campaign T - robustness grid for the unseen scenes. The 13-scene set contains
# scene types the 3-case tuning never saw, so the two knobs with the largest
# measured effect (lambda_dssim, scaling_lr) are swept around the delivery value at
# the fixed 5k-step / early-structural schedule.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}; ENV=${ENV:-fastgs}
D=${D:-outputs/experiments/campaignT}; LANE=${1:-0}; NLANES=${2:-4}
mkdir -p "$D/ply"
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
sharp(){ local t; t=$(echo "$1"|tr '/' '_'); local o="$D/ply/${t}_s.ply"; [ -f "$o" ] || $PY run --no-capture-output -n "$ENV" python -m research.ply_ops --in "outputs/O0-20261030/results/$1/resplat/resplat_world.ply" --out "$o" --op sharpen --gain 2.0 >/dev/null 2>&1; echo "$o"; }
one(){ local tag="$1" case="$2" budget="$3"; shift 3; local dir="$D/$tag"
  [ -e "$dir/summary.json" ] && return; rm -rf "$dir"
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  local ply; ply=$(sharp "$case")
  $PY run --no-capture-output -n "$ENV" python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$LANE.json" --case "$case" \
    --work "outputs/O0-20261030/work/$case" --init-ply "$ply" --out "$dir" \
    --iterations 5000 --snapshots "" --budget "$budget" --opacity-reset-interval 0 \
    --position-lr-max-steps 5000 --densify-until 1250 "$@" >"$D/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"; rm -rf "$dir/model" "$dir/eval" 2>/dev/null; }
JOBS=()
for spec in "hw:$HW:$BHW" "co:$CO:$BCO" "dl:$DL:$BDL"; do
  IFS=':' read -r s case b <<<"$spec"
  JOBS+=("t_${s}_d04_s01|$case|$b|--lambda-dssim 0.4 --scaling-lr 0.01")
  JOBS+=("t_${s}_d08_s01|$case|$b|--lambda-dssim 0.8 --scaling-lr 0.01")
  JOBS+=("t_${s}_d06_s005|$case|$b|--lambda-dssim 0.6 --scaling-lr 0.005")
  JOBS+=("t_${s}_d06_s02|$case|$b|--lambda-dssim 0.6 --scaling-lr 0.02")
  JOBS+=("t_${s}_d06_s01_opa05|$case|$b|--lambda-dssim 0.6 --scaling-lr 0.01 --opacity-lr 0.05")
done
i=0; for job in "${JOBS[@]}"; do i=$((i+1)); [ $(( (i-1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case budget rest <<<"$job"; one "$tag" "$case" "$budget" $rest; done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
