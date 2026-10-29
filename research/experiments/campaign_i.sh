#!/usr/bin/env bash
# Campaign I - literature-guided module-B designs (not knob tuning).
#
# ForeSplat (arXiv 2605.22020) describes the standard predict-then-refine recipe used
# by AnySplat / InstantSplat / Splatt3R / MVSGaussian: the post-optimization *keeps the
# number of Gaussians fixed* - densification, cloning, splitting and pruning are all
# disabled and only the attributes move. FastGS does the opposite (aggressive
# densify+prune designed for training from scratch). This campaign tests that
# difference directly, on top of the two init treatments that survived campaign H
# (raw ReSplat PLY, and the opacity-sharpened PLY).
#
#   bash research/experiments/campaign_i.sh <lane> <nlanes>
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}
D=${D:-outputs/experiments/campaignI}
LANE=${1:-0}
NLANES=${2:-4}
mkdir -p "$D/ply"

HW=hw/billboard;        BHW=99.49
CO=mipnerf360/counter;  BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
OPTS="--opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01"

sharp_ply() {  # case -> path
  local tag src out
  tag=$(echo "$1" | tr '/' '_')
  src="outputs/O0-20261030/results/$1/resplat/resplat_world.ply"
  out="$D/ply/${tag}_sharp.ply"
  [ -f "$out" ] || $PY run --no-capture-output -n fastgs python -m research.ply_ops \
      --in "$src" --out "$out" --op sharpen --gain 2.0 >/dev/null 2>&1
  echo "$out"
}

run_one() {  # tag case iters budget plymode extra...
  local tag="$1" case="$2" iters="$3" budget="$4" plymode="$5"; shift 5
  local dir="$D/$tag" ply
  [ -e "$dir/summary.json" ] && return
  [ -e "$dir" ] && rm -rf "$dir"   # incomplete leftover from an interrupted attempt
  if [ "$plymode" = sharp ]; then ply=$(sharp_ply "$case"); else ply="outputs/O0-20261030/results/$case/resplat/resplat_world.ply"; fi
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  $PY run --no-capture-output -n fastgs python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$LANE.json" --case "$case" \
    --work "outputs/O0-20261030/work/$case" --init-ply "$ply" \
    --out "$dir" --iterations "$iters" --snapshots "" --budget "$budget" \
    $OPTS --position-lr-max-steps "$iters" "$@" >"$D/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"
}

# tag|case|iters|budget|ply|extra
JOBS=(
  # attribute-only refinement (ForeSplat / InstantSplat recipe) - no structural ops
  "i_hw_nodens_4k|$HW|4000|$BHW|raw|--no-densify"
  "i_co_nodens_5k|$CO|5000|$BCO|raw|--no-densify"
  "i_dl_nodens_4k|$DL|4000|$BDL|raw|--no-densify"
  "i_hw_sharp_nodens_4k|$HW|4000|$BHW|sharp|--no-densify"
  "i_co_sharp_nodens_5k|$CO|5000|$BCO|sharp|--no-densify"
  "i_dl_sharp_nodens_4k|$DL|4000|$BDL|sharp|--no-densify"
  # attribute-only with a longer horizon (it cannot grow capacity, so it may need steps)
  "i_hw_sharp_nodens_8k|$HW|8000|$BHW|sharp|--no-densify"
  "i_co_sharp_nodens_8k|$CO|8000|$BCO|sharp|--no-densify"
  "i_dl_sharp_nodens_8k|$DL|8000|$BDL|sharp|--no-densify"
  # attribute-only needs a different LR regime: raise the anchor LR
  "i_hw_sharp_nodens_4k_lr2|$HW|4000|$BHW|sharp|--no-densify --position-lr-init 0.00032"
  "i_co_sharp_nodens_5k_lr2|$CO|5000|$BCO|sharp|--no-densify --position-lr-init 0.00032"
  "i_dl_sharp_nodens_4k_lr2|$DL|4000|$BDL|sharp|--no-densify --position-lr-init 0.00032"
)

i=0
for job in "${JOBS[@]}"; do
  i=$((i + 1))
  [ $(( (i - 1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case iters budget ply extra <<<"$job"
  # shellcheck disable=SC2086
  run_one "$tag" "$case" "$iters" "$budget" "$ply" $extra
done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
