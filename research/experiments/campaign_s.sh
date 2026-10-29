#!/usr/bin/env bash
# Campaign S - module-B idea from the pre/post report: pose-guided geometric
# refinement (ICPR 2026). A feed-forward reconstruction can keep Gaussian
# attributes anchored slightly off the true camera geometry; here the last 25% of
# the run additionally optimizes a per-view camera translation (world-shift form,
# since the FastGS rasterizer does not differentiate camera matrices).
# Control is the same recipe without --pose-lr (campaign R / delivery check).
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}; ENV=${ENV:-fastgs}
D=${D:-outputs/experiments/campaignS}; LANE=${1:-0}; NLANES=${2:-4}
mkdir -p "$D/ply"
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
sharp(){ local t; t=$(echo "$1"|tr '/' '_'); local o="$D/ply/${t}_s.ply"; [ -f "$o" ] || $PY run --no-capture-output -n "$ENV" python -m research.ply_ops --in "outputs/O0-20261030/results/$1/resplat/resplat_world.ply" --out "$o" --op sharpen --gain 2.0 >/dev/null 2>&1; echo "$o"; }
one(){ local tag="$1" case="$2" budget="$3" lr="$4" frac="$5"; local dir="$D/$tag"
  [ -e "$dir/metrics.json" ] && return; rm -rf "$dir"; mkdir -p "$dir"
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  local ply; ply=$(sharp "$case")
  CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" python -m research.depth_reg \
    --source "outputs/O0-20261030/work/$case" --model_path "$dir/model" \
    --eval -r 1 --sh_degree 3 --quiet --init_ply "$ply" \
    --iterations 5000 --position_lr_max_steps 5000 --opacity_reset_interval 0 \
    --lambda_dssim 0.6 --scaling_lr 0.01 --densify_until_iter 1250 \
    --pose-lr "$lr" --pose-start-frac "$frac" --pose-max-shift 0.02 \
    --record "$dir/train_record.json" >"$D/$tag.log" 2>&1
  echo "train_exit=$? $tag" >>"$D/queue.log"
  local it ply_out
  it=$(python3 -c "import json;print(json.load(open('$dir/train_record.json'))['completed_iterations'])" 2>/dev/null || echo 5000)
  ply_out=$(python3 -c "import json;print(json.load(open('$dir/train_record.json'))['final_ply'])" 2>/dev/null || echo "")
  if [ -n "$ply_out" ] && [ -f "$ply_out" ]; then
    CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" python -m research.render_eval \
      --source "outputs/O0-20261030/work/$case" --out "$dir/eval" --eval -r 1 --sh_degree 3 \
      --ply "$ply_out" --name "iter$it" >>"$D/$tag.log" 2>&1
    CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" python competition/upstream/baseline/evaluate.py \
      --model_path "$dir/eval/iter$it" --device cuda:0 >>"$D/$tag.log" 2>&1
    cp "$dir/eval/iter$it/metrics.json" "$dir/metrics.json" 2>/dev/null
  fi
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"; rm -rf "$dir/model" "$dir/eval" 2>/dev/null; }
JOBS=(
  "s_hw_pose1e4|$HW|$BHW|1e-4|0.75"
  "s_co_pose1e4|$CO|$BCO|1e-4|0.75"
  "s_dl_pose1e4|$DL|$BDL|1e-4|0.75"
  "s_hw_pose5e5|$HW|$BHW|5e-5|0.5"
  "s_co_pose5e5|$CO|$BCO|5e-5|0.5"
  "s_dl_pose5e5|$DL|$BDL|5e-5|0.5"
)
i=0; for job in "${JOBS[@]}"; do i=$((i+1)); [ $(( (i-1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case budget lr frac <<<"$job"; one "$tag" "$case" "$budget" "$lr" "$frac"; done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
