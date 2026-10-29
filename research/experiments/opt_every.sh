#!/usr/bin/env bash
# E8-lite: the optimizer step is 22% of module B (hw breakdown: render 30%, backward
# 42%, optimizer 22%). FastGS's own schedule already steps less often after 15k
# iterations, but our 5k runs never reach it. This tests the same idea inside the budget.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}; ENV=${ENV:-fastgs}; D=${D:-outputs/experiments/optEvery}; LANE=${1:-0}; NLANES=${2:-2}
mkdir -p "$D"
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
one(){ local tag=$1 case=$2 budget=$3 sched=$4; local dir="$D/$tag"
  [ -e "$dir/summary.json" ] && return; rm -rf "$dir"
  local work="outputs/O0-20261030/work/$case" cache="$D/cache_$(echo $case|tr '/' '_')"
  [ -f "$cache/manifest.json" ] || $PY run --no-capture-output -n "$ENV" python -m research.image_cache --work "$work" --out "$cache" >/dev/null 2>&1
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  $PY run --no-capture-output -n "$ENV" python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$LANE.json" --case "$case" --work "$work" \
    --init-ply "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" --out "$dir" \
    --iterations 5000 --snapshots "" --budget "$budget" --opacity-reset-interval 0 \
    --lambda-dssim 0.6 --scaling-lr 0.01 --position-lr-max-steps 5000 --densify-until 1250 \
    --image-cache "$cache" ${sched:+--opt-step-schedule "$sched"} >"$D/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"; rm -rf "$dir/model" "$dir/eval" 2>/dev/null; }
JOBS=("oe_hw_ctrl|$HW|$BHW|" "oe_hw_s2_2500|$HW|$BHW|2500:2" "oe_hw_s2_0|$HW|$BHW|0:2" \
      "oe_co_ctrl|$CO|$BCO|" "oe_co_s2_2500|$CO|$BCO|2500:2" "oe_co_s2_0|$CO|$BCO|0:2")
i=0; for job in "${JOBS[@]}"; do i=$((i+1)); [ $(( (i-1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case budget sched <<<"$job"; one "$tag" "$case" "$budget" "$sched"; done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
