#!/usr/bin/env bash
# Campaign X - final-recipe robustness: 5 seeds x 3 cases with the shipped recipe
# (raw ReSplat init, image cache, 5k steps, matched LR, no opacity reset,
#  lambda_dssim 0.6, scaling_lr 0.01, structural ops confined to the first quarter).
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}; ENV=${ENV:-fastgs}
D=${D:-outputs/experiments/campaignX}; LANE=${1:-0}; NLANES=${2:-4}
mkdir -p "$D"
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
one(){ local tag=$1 case=$2 budget=$3 seed=$4; local dir="$D/$tag"
  [ -e "$dir/summary.json" ] && return; rm -rf "$dir"
  local work="outputs/O0-20261030/work/$case" cache="$D/cache_$(echo $case | tr '/' '_')" init="outputs/O0-20261030/results/$case/resplat/resplat_world.ply"
  [ -f "$cache/manifest.json" ] || $PY run --no-capture-output -n "$ENV" python -m research.image_cache --work "$work" --out "$cache" >/dev/null 2>&1
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  $PY run --no-capture-output -n "$ENV" python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$LANE.json" --case "$case" --work "$work" \
    --init-ply "$init" --out "$dir" --iterations 5000 --snapshots "" --budget "$budget" \
    --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 \
    --position-lr-max-steps 5000 --densify-until 1250 --image-cache "$cache" --seed "$seed" \
    >"$D/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"; rm -rf "$dir/model" "$dir/eval" 2>/dev/null; }
JOBS=()
for spec in "hw:$HW:$BHW" "co:$CO:$BCO" "dl:$DL:$BDL"; do
  IFS=':' read -r s case b <<<"$spec"
  for seed in 20261015 1234 999 777 31337; do JOBS+=("x_${s}_s${seed}|$case|$b|$seed"); done
done
i=0; for job in "${JOBS[@]}"; do i=$((i+1)); [ $(( (i-1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case budget seed <<<"$job"; one "$tag" "$case" "$budget" "$seed"; done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
