#!/usr/bin/env bash
# Campaign V - module B as 3DGS-MCMC (Taming-3DGS style), not FastGS.
#
# FastGS is a from-scratch accelerator; its densify/prune heuristic was tuned for
# SfM initialisation and the organizer does not require it for module B. This
# campaign runs the standalone gsplat MCMC trainer (research/mcmc_budget.py) on the
# SAME ReSplat initialisation, with an explicit wall-clock budget, and scores the
# result through the official renderer + metric code.
#
#   bash research/experiments/campaign_mcmc.sh <lane> <nlanes>
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}; ENV=${ENV:-fastgs}
D=${D:-outputs/experiments/campaignV}; LANE=${1:-0}; NLANES=${2:-2}
mkdir -p "$D"
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29

one() {  # tag case budget init_mode
  local tag="$1" case="$2" budget="$3" initmode="$4"
  local dir="$D/$tag"; [ -f "$dir/metrics.json" ] && return; rm -rf "$dir"; mkdir -p "$dir"
  local init
  if [ "$initmode" = sharp ]; then
    init="$D/init_$(echo $case | tr '/' '_')_sharp.ply"
    [ -f "$init" ] || $PY run --no-capture-output -n "$ENV" python -m research.ply_ops \
      --in "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" --out "$init" \
      --op sharpen --gain 2.0 >/dev/null 2>&1
  else
    init="outputs/O0-20261030/results/$case/resplat/resplat_world.ply"
  fi
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" python -m research.mcmc_budget \
    --work "outputs/O0-20261030/work/$case" --init "$init" --out "$dir/model" \
    --budget-seconds "$budget" --budget-fraction 0.95 --device cuda:0 \
    --res-scales 0.5,0.75,1.0 --res-untils 0.4,0.7 --cap-mult 2.0 --refine-every 100 \
    >>"$D/$tag.log" 2>&1
  echo "train_exit=$?" >>"$D/queue.log"
  local ply iter
  ply=$(ls "$dir"/model/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | head -1)
  iter=$(echo "$ply" | sed 's/.*iteration_//; s|/.*||')
  if [ -n "$ply" ] && [ -f "$ply" ]; then
    CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" python -m research.render_eval \
      --source "outputs/O0-20261030/work/$case" --out "$dir/eval" --eval -r 1 --sh_degree 3 \
      --ply "$ply" --name "iter$iter" >>"$D/$tag.log" 2>&1
    CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" python competition/upstream/baseline/evaluate.py \
      --model_path "$dir/eval/iter$iter" --device cuda:0 >>"$D/$tag.log" 2>&1
    cp "$dir/eval/iter$iter/metrics.json" "$dir/metrics.json" 2>/dev/null
    cp "$dir/model/module_b.json" "$dir/module_b.json" 2>/dev/null
  fi
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"; rm -rf "$dir/model" "$dir/eval" 2>/dev/null
}
JOBS=(
  "v_hw_b40|$HW|40|sharp"
  "v_hw_b80|$HW|80|sharp"
  "v_co_b40|$CO|40|sharp"
  "v_co_b80|$CO|80|sharp"
  "v_dl_b40|$DL|40|sharp"
  "v_dl_b80|$DL|80|sharp"
  "v_co_b40_raw|$CO|40|raw"
  "v_hw_b40_raw|$HW|40|raw"
)
i=0; for job in "${JOBS[@]}"; do i=$((i+1)); [ $(( (i-1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case budget init <<<"$job"; one "$tag" "$case" "$budget" "$init"; done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
