#!/usr/bin/env bash
# Campaign W - MCMC as module B, pushed much harder than the first attempt.
# The first try used the trainer defaults (cap x2, refine every 100, 40 s) and lost to
# FastGS. MCMC's quality is controlled by the primitive cap, the refine cadence and the
# horizon, so this grid explores those explicitly, per case, with the official metrics.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD TORCH_CUDA_ARCH_LIST=8.9
PY=${PY:-/opt/conda/bin/conda}; ENV=${ENV:-fastgs}
D=${D:-outputs/experiments/mcmcGrid}; LANE=${1:-0}; NLANES=${2:-4}
mkdir -p "$D"
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
one(){ local tag=$1 case=$2 budget=$3 bench=$4; shift 4
  local dir="$D/$tag"; [ -f "$dir/metrics.json" ] && return; rm -rf "$dir"; mkdir -p "$dir"
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" python -m research.mcmc_budget \
    --work "outputs/O0-20261030/work/$case" \
    --init "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" \
    --out "$dir/model" --budget-seconds "$budget" --budget-fraction 0.92 --device cuda:0 \
    --res-scales 0.5,0.75,1.0 --res-untils 0.35,0.65 "$@" >>"$D/$tag.log" 2>&1
  local ply iter
  ply=$(ls "$dir"/model/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | head -1)
  iter=$(basename "$(dirname "$ply" 2>/dev/null)" 2>/dev/null | sed 's/iteration_//')
  if [ -n "$ply" ]; then
    CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" python -m research.render_eval \
      --source "outputs/O0-20261030/work/$case" --out "$dir/eval" --eval -r 1 --sh_degree 3 \
      --ply "$ply" --name "iter$iter" >>"$D/$tag.log" 2>&1
    CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" python competition/upstream/baseline/evaluate.py \
      --model_path "$dir/eval/iter$iter" --device cuda:0 >>"$D/$tag.log" 2>&1
    cp "$dir/eval/iter$iter/metrics.json" "$dir/metrics.json" 2>/dev/null
    cp "$dir/model/module_b.json" "$dir/module_b.json" 2>/dev/null
  fi
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"; rm -rf "$dir/model" "$dir/eval" 2>/dev/null; }
JOBS=(
  "w_hw_b45_c2|$HW|45|$BHW|--cap-mult 2.0"
  "w_hw_b90_c2|$HW|90|$BHW|--cap-mult 2.0"
  "w_hw_b90_c3|$HW|90|$BHW|--cap-mult 3.0"
  "w_hw_b90_c4|$HW|90|$BHW|--cap-mult 4.0"
  "w_hw_b90_c3_r50|$HW|90|$BHW|--cap-mult 3.0 --refine-every 50"
  "w_hw_b90_c3_full|$HW|90|$BHW|--cap-mult 3.0 --res-scales 1.0 --res-untils ''"
  "w_co_b90_c3|$CO|90|$BCO|--cap-mult 3.0"
  "w_co_b90_c3_r50|$CO|90|$BCO|--cap-mult 3.0 --refine-every 50"
  "w_co_b60_c2|$CO|60|$BCO|--cap-mult 2.0"
  "w_dl_b90_c3|$DL|90|$BDL|--cap-mult 3.0"
  "w_dl_b90_c3_r50|$DL|90|$BDL|--cap-mult 3.0 --refine-every 50"
  "w_dl_b60_c2|$DL|60|$BDL|--cap-mult 2.0"
)
i=0; for job in "${JOBS[@]}"; do i=$((i+1)); [ $(( (i-1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case budget bench extra <<<"$job"; one "$tag" "$case" "$budget" "$bench" $extra; done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
