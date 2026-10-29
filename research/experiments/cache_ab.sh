#!/usr/bin/env bash
# Measure E0: how much of module B's wall clock is PIL decoding of the prepared PNGs?
# Builds a uint8 cache per case and runs the delivery recipe with and without it.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}; ENV=${ENV:-fastgs}
D=${D:-outputs/experiments/cacheAB}; mkdir -p "$D"
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
one(){ local tag=$1 case=$2 budget=$3 cache=$4
  local dir="$D/$tag"; [ -e "$dir/summary.json" ] && return; rm -rf "$dir"
  local ply="outputs/O0-20261030/results/$case/resplat/resplat_world.ply"
  echo "start $tag $(date -Is)" >>"$D/queue.log"
  $PY run --no-capture-output -n "$ENV" python -m research.run_sweep \
    --profile competition/profiles/gpufree-0.json \
    --case "$case" --work "outputs/O0-20261030/work/$case" --init-ply "$ply" \
    --out "$dir" --iterations 5000 --snapshots "" --budget "$budget" \
    --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 \
    --position-lr-max-steps 5000 --densify-until 1250 $cache >"$D/$tag.log" 2>&1
  rm -rf "$dir/model" "$dir/eval" 2>/dev/null; }
for spec in "hw:$HW:$BHW" "co:$CO:$BCO" "dl:$DL:$BDL"; do
  IFS=':' read -r s case b <<<"$spec"
  cache="$D/cache_$s"
  [ -f "$cache/manifest.json" ] || $PY run --no-capture-output -n "$ENV" python -m research.image_cache \
    --work "outputs/O0-20261030/work/$case" --out "$cache" >>"$D/cache.log" 2>&1
  one "cab_${s}_base" "$case" "$b" ""
  one "cab_${s}_cache" "$case" "$b" "--image-cache $cache"
done
echo done >>"$D/queue.log"
