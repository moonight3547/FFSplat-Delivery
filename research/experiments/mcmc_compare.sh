#!/usr/bin/env bash
# MCMC as module B - a real comparison against FastGS, not a knob tweak.
# research/mcmc_budget.py is a standalone gsplat MCMC (Taming-3DGS style) trainer on
# the same ReSplat init with an explicit wall-clock budget. This runs it at a few
# budgets and cap settings and scores through the official renderer + metrics.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}; ENV=${ENV:-fastgs}
D=${D:-outputs/experiments/mcmcCmp}; LANE=${1:-0}; NLANES=${2:-2}
mkdir -p "$D"
export TORCH_CUDA_ARCH_LIST=8.9
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
one(){ local tag=$1 case=$2 budget=$3 bench=$4; shift 4
  local dir="$D/$tag"; [ -f "$dir/metrics.json" ] && return; rm -rf "$dir"; mkdir -p "$dir"
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" python -m research.mcmc_budget \
    --work "outputs/O0-20261030/work/$case" \
    --init "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" \
    --out "$dir/model" --budget-seconds "$budget" --budget-fraction 0.9 --device cuda:0 \
    --res-scales 0.5,0.75,1.0 --res-untils 0.4,0.7 "$@" >>"$D/$tag.log" 2>&1
  local ply iter
  ply=$(ls "$dir"/model/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | head -1)
  iter=$(basename "$(dirname "$ply")" 2>/dev/null | sed 's/iteration_//')
  if [ -n "$ply" ]; then
    CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" python -m research.render_eval \
      --source "outputs/O0-20261030/work/$case" --out "$dir/eval" --eval -r 1 --sh_degree 3 \
      --ply "$ply" --name "iter$iter" >>"$D/$tag.log" 2>&1
    CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" python competition/upstream/baseline/evaluate.py \
      --model_path "$dir/eval/iter$iter" --device cuda:0 >>"$D/$tag.log" 2>&1
    cp "$dir/eval/iter$iter/metrics.json" "$dir/metrics.json" 2>/dev/null
    cp "$dir/model/module_b.json" "$dir/module_b.json" 2>/dev/null
  fi
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"
  rm -rf "$dir/model" "$dir/eval" 2>/dev/null; }
JOBS=(
  "m_hw_b40|$HW|40|$BHW|"
  "m_hw_b80|$HW|80|$BHW|"
  "m_hw_b40_cap3|$HW|40|$BHW|--cap-mult 3.0"
  "m_hw_b40_ref50|$HW|40|$BHW|--refine-every 50"
  "m_co_b40|$CO|40|$BCO|"
  "m_co_b80|$CO|80|$BCO|"
  "m_dl_b40|$DL|40|$BDL|"
  "m_dl_b80|$DL|80|$BDL|"
)
i=0; for job in "${JOBS[@]}"; do i=$((i+1)); [ $(( (i-1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case budget bench extra <<<"$job"; one "$tag" "$case" "$budget" "$bench" $extra; done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
