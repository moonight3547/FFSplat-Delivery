#!/usr/bin/env bash
# Campaign J - is FastGS even the right optimizer for module B?
#
# FastGS (arXiv 2511.04283) is built for training *from scratch* on SfM points in
# ~100 s; its densification/pruning is a from-scratch acceleration device. The task's
# own time budget, however, is defined by *classic 3DGS from scratch 30k*, which is
# ~5x slower per iteration. So the natural module-B candidates are:
#
#   classic-3DGS continuation  (the reference optimizer, densification incl.)
#   fastgs continuation        (our current entry)
#   attribute-only refinement  (ForeSplat/InstantSplat recipe, campaign I)
#
# This campaign runs the classic 3DGS continuation from the same ReSplat PLY at two
# step counts (5k ~= 100 s, 15k ~= 275 s on a 4090) so the comparison is by wall clock.
#
#   bash research/experiments/campaign_j.sh <lane> <nlanes>
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}
ENV=${ENV:-fastgs}
D=${D:-outputs/experiments/campaignJ}
LANE=${1:-0}
NLANES=${2:-4}
mkdir -p "$D"

HW=hw/billboard
CO=mipnerf360/counter
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f

run_one() {  # tag case iters
  local tag="$1" case="$2" iters="$3" dir work ply
  dir="$D/$tag"
  work="outputs/O0-20261030/work/$case"
  ply="outputs/O0-20261030/results/$case/resplat/resplat_world.ply"
  [ -f "$dir/metrics.json" ] && return
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" \
    python competition/upstream/baseline/module_b/train_from_ply.py \
    -s "$work" --eval -r 1 --sh_degree 3 --quiet -m "$dir" \
    --init_ply "$ply" --iterations "$iters" --position_lr_max_steps "$iters" \
    --densify_until_iter "$iters" >>"$D/$tag.log" 2>&1
  echo "train_exit=$?" >>"$D/queue.log"
  CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" \
    python competition/upstream/baseline/render_test.py -s "$work" -m "$dir" \
    --backend classic3dgs --iteration "$iters" --eval -r 1 --sh_degree 3 --quiet \
    >>"$D/$tag.log" 2>&1
  CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" \
    python competition/upstream/baseline/evaluate.py --model_path "$dir" --device cuda:0 \
    >>"$D/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"
}

JOBS=(
  "j_classic_hw_5k|$HW|5000"
  "j_classic_co_5k|$CO|5000"
  "j_classic_dl_5k|$DL|5000"
  "j_classic_hw_15k|$HW|15000"
  "j_classic_co_15k|$CO|15000"
  "j_classic_dl_15k|$DL|15000"
)

i=0
for job in "${JOBS[@]}"; do
  i=$((i + 1))
  [ $(( (i - 1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case iters <<<"$job"
  run_one "$tag" "$case" "$iters"
done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
