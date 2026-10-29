#!/usr/bin/env bash
# Campaign L - what changes if the time budget is the organizer's *classic 3DGS 30k*
# instead of the stricter FastGS-random-30k we have been using?
#
# The task text says: module B must be faster than classic 3DGS trained from scratch
# for 30000 iterations (T_budget = 30000 * T_iter_classic). The organizer measured
# 546.7 / 915.1 / 531.0 s for dl3dv / hw / counter on a 4090. We have been scoring
# against the 5x stricter FastGS-random-30k budget (~95 s) that the user confirmed
# earlier. This campaign maps the quality/time curve far beyond 100 s so the same
# runs can be scored under both readings, and re-measures one classic-3DGS 30k
# reference to check the organizer's number.
#
#   bash research/experiments/campaign_l.sh <lane> <nlanes>
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}
ENV=${ENV:-fastgs}
D=${D:-outputs/experiments/campaignL}
LANE=${1:-0}
NLANES=${2:-4}
mkdir -p "$D"

HW=hw/billboard;        BHW=99.49
CO=mipnerf360/counter;  BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
OPTS="--opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01"

sweep() {  # tag case iters budget
  local tag="$1" case="$2" iters="$3" budget="$4" dir="$D/$tag"
  [ -e "$dir/summary.json" ] && return
  [ -e "$dir" ] && rm -rf "$dir"   # incomplete leftover from an interrupted attempt
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  $PY run --no-capture-output -n "$ENV" python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$LANE.json" --case "$case" \
    --work "outputs/O0-20261030/work/$case" \
    --init-ply "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" \
    --out "$dir" --iterations "$iters" --snapshots "" --budget "$budget" \
    $OPTS --position-lr-max-steps "$iters" >"$D/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"
}

# classic-3DGS 30k from scratch: the organizer's own budget definition (one case as a check)
classic_ref() {  # tag case
  local tag="$1" case="$2" dir="$D/$tag"
  [ -f "$dir/module_b.json" ] && return
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" \
    python competition/upstream/baseline/module_b/train_from_ply.py \
    -s "outputs/O0-20261030/work/$case" --eval -r 1 --sh_degree 3 --quiet -m "$dir" \
    --init_random --iterations 30000 --position_lr_max_steps 30000 \
    --densify_until_iter 15000 >>"$D/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"
}

JOBS=(
  "l_hw_10k|sweep|$HW|10000|$BHW"
  "l_hw_20k|sweep|$HW|20000|$BHW"
  "l_hw_40k|sweep|$HW|40000|$BHW"
  "l_co_10k|sweep|$CO|10000|$BCO"
  "l_co_20k|sweep|$CO|20000|$BCO"
  "l_co_40k|sweep|$CO|40000|$BCO"
  "l_dl_10k|sweep|$DL|10000|$BDL"
  "l_dl_20k|sweep|$DL|20000|$BDL"
  "l_dl_40k|sweep|$DL|40000|$BDL"
  "l_classic_ref_hw|classic|$HW|30000|0"
  "l_classic_ref_dl|classic|$DL|30000|0"
)

i=0
for job in "${JOBS[@]}"; do
  i=$((i + 1))
  [ $(( (i - 1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag kind case iters budget <<<"$job"
  case "$kind" in
    sweep) sweep "$tag" "$case" "$iters" "$budget" ;;
    classic) classic_ref "$tag" "$case" ;;
  esac
done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
