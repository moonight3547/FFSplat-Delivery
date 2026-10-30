#!/usr/bin/env bash
# Module-B arms only (no A/ReSplat): the init PLY already exists from the preparation
# step, so each arm is a short training run plus the official metrics.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
FPY=${FPY:-/opt/conda/envs/fastgs/bin/python}
CASE=${CASE:?}; GPU=${GPU:-0}; SECS=${SECS:-40}; OUT=${OUT:-outputs/arms}
tag=$(echo "$CASE" | tr '/' '-'); d="$OUT/$tag"; mkdir -p "$d"
work=${WORK:?work dir}; init=${INIT:?init ply}
export CUDA_VISIBLE_DEVICES=$GPU
$FPY -m research.image_cache --work "$work" --out "$d/cache" >/dev/null 2>&1
run_arm(){  # name extra-args...
  local name=$1; shift; local dir="$d/$name"
  [ -f "$dir/metrics.json" ] && return; rm -rf "$dir"; mkdir -p "$dir"
  $FPY -m research.train_sweep --source "$work" --model_path "$dir/model" --eval -r 1 \
    --sh_degree 3 --quiet --init_ply "$init" --iterations 100000 --snapshots "" \
    --limit_seconds "$SECS" --limit_reserve 1.0 --image_cache "$d/cache" \
    --record "$dir/train_record.json" "$@" > "$dir/train.log" 2>&1
  local ply iter
  ply=$(ls "$dir"/model/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | head -1)
  iter=$(basename "$(dirname "$ply" 2>/dev/null)" 2>/dev/null | sed 's/iteration_//')
  if [ -n "$ply" ]; then
    $FPY -m research.render_eval --source "$work" --out "$dir/eval" --eval -r 1 --sh_degree 3 \
      --ply "$ply" --name "iter$iter" >> "$dir/train.log" 2>&1
    $FPY competition/upstream/baseline/evaluate.py --model_path "$dir/eval/iter$iter" --device cuda:0 >> "$dir/train.log" 2>&1
    cp "$dir/eval/iter$iter/metrics.json" "$dir/metrics.json" 2>/dev/null
  fi
  rm -rf "$dir/model/point_cloud" 2>/dev/null
}
run_arm fastgs --position_lr_max_steps 5000 --opacity_reset_interval 0 --lambda_dssim 0.6 \
  --scaling_lr 0.01 --densify_until_iter 1250
run_arm inject --position_lr_max_steps 5000 --opacity_reset_interval 0 --lambda_dssim 0.6 \
  --scaling_lr 0.01 --densify_until_iter 1250 --inject_holes 60000 --inject_at 500
echo "ARMS_DONE $CASE"
