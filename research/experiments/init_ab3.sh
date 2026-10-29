#!/usr/bin/env bash
# Clean 3-case A/B on the init transform: raw ReSplat PLY vs opacity-sharpened PLY,
# same delivery module-B recipe (5k, matched LR, no reset, dssim .6, scaling .01,
# structural ops in the first quarter). Campaign H suggested +1 for sharpening but
# a single counter run suggested -2.9, so this settles it with matched pairs.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}; ENV=${ENV:-fastgs}
D=${D:-outputs/experiments/initAB3}; LANE=${1:-0}; NLANES=${2:-4}
mkdir -p "$D/ply"
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
one(){ local tag=$1 case=$2 budget=$3 mode=$4; local dir="$D/$tag"
  [ -e "$dir/summary.json" ] && return; rm -rf "$dir"
  local ply
  if [ "$mode" = sharp ]; then
    ply="$D/ply/$(echo $case | tr '/' '_').ply"
    [ -f "$ply" ] || $PY run --no-capture-output -n "$ENV" python -m research.ply_ops \
      --in "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" --out "$ply" --op sharpen --gain 2.0 >/dev/null 2>&1
  else
    ply="outputs/O0-20261030/results/$case/resplat/resplat_world.ply"
  fi
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  $PY run --no-capture-output -n "$ENV" python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$LANE.json" --case "$case" \
    --work "outputs/O0-20261030/work/$case" --init-ply "$ply" --out "$dir" \
    --iterations 5000 --snapshots "" --budget "$budget" --opacity-reset-interval 0 \
    --lambda-dssim 0.6 --scaling-lr 0.01 --position-lr-max-steps 5000 --densify-until 1250 \
    >"$D/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"; rm -rf "$dir/model" "$dir/eval" 2>/dev/null; }
JOBS=()
for spec in "hw:$HW:$BHW" "co:$CO:$BCO" "dl:$DL:$BDL"; do
  IFS=':' read -r s case b <<<"$spec"
  JOBS+=("iab_${s}_raw|$case|$b|raw" "iab_${s}_sharp|$case|$b|sharp")
done
i=0; for job in "${JOBS[@]}"; do i=$((i+1)); [ $(( (i-1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case budget mode <<<"$job"; one "$tag" "$case" "$budget" "$mode"; done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
