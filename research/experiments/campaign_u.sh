#!/usr/bin/env bash
# Campaign U - module-design comparison at the delivery operating point, with seeds.
# Three module-B designs are compared end-to-end under the confirmed strict budget:
#   full   : FastGS structural ops for the first quarter of a 5k run (delivery)
#   attr   : attribute-only refinement, no densify/clone/split/prune at all
#   late   : structural ops only after the first half (attribute-only warm start)
# Three seeds each, three cases, so a difference below the seed spread is not claimed.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}; ENV=${ENV:-fastgs}
D=${D:-outputs/experiments/campaignU}; LANE=${1:-0}; NLANES=${2:-4}
mkdir -p "$D/ply"
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
sharp(){ local t; t=$(echo "$1"|tr '/' '_'); local o="$D/ply/${t}_s.ply"; [ -f "$o" ] || $PY run --no-capture-output -n "$ENV" python -m research.ply_ops --in "outputs/O0-20261030/results/$1/resplat/resplat_world.ply" --out "$o" --op sharpen --gain 2.0 >/dev/null 2>&1; echo "$o"; }
one(){ local tag="$1" case="$2" budget="$3" seed="$4"; shift 4; local dir="$D/$tag"
  [ -e "$dir/summary.json" ] && return; rm -rf "$dir"
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  local ply; ply=$(sharp "$case")
  $PY run --no-capture-output -n "$ENV" python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$LANE.json" --case "$case" --work "outputs/O0-20261030/work/$case" \
    --init-ply "$ply" --out "$dir" --iterations 5000 --snapshots "" --budget "$budget" \
    --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 \
    --position-lr-max-steps 5000 --seed "$seed" "$@" >"$D/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"; rm -rf "$dir/model" "$dir/eval" 2>/dev/null; }
JOBS=()
for spec in "hw:$HW:$BHW" "co:$CO:$BCO" "dl:$DL:$BDL"; do
  IFS=':' read -r s case b <<<"$spec"
  for seed in 20261015 1234 999; do
    JOBS+=("u_${s}_full_$seed|$case|$b|$seed|--densify-until 1250")
    JOBS+=("u_${s}_attr_$seed|$case|$b|$seed|--no-densify")
    JOBS+=("u_${s}_late_$seed|$case|$b|$seed|--densify-from 2500")
  done
done
i=0; for job in "${JOBS[@]}"; do i=$((i+1)); [ $(( (i-1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case budget seed extra <<<"$job"; one "$tag" "$case" "$budget" "$seed" $extra; done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
