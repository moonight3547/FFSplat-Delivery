#!/usr/bin/env bash
# One low-scoring scene, three module-B strategies, all with wall-clock snapshots:
#   fastgs  : the delivery recipe
#   mcmc    : 3DGS-MCMC (cap x3, refine every 50)
#   inject  : hole injection (Gaussian-Frosting / PUP-style: new low-opacity
#             Gaussians in EMPTY voxels at step 800, pruned later by opacity)
# Snapshots are scored afterwards, so the metric time is excluded from module-B time.
#
#   CASE=<group>/<scene> GPU=<index> bash research/experiments/curve_case.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}
RESPLAT=${RESPLAT:-ffsplat-enhance}; FASTGS=${FASTGS:-organizer-fastgs}
# The ReSplat env may live outside the conda root that $PY belongs to (copied node),
# so allow a direct interpreter path: RESPLAT_PY=/path/to/env/bin/python
if [ -n "${RESPLAT_PY:-}" ]; then
  # copied conda envs need their own CUDA headers/nvcc on the path for the JIT build
  _env_root="$(dirname "$(dirname "$RESPLAT_PY")")"
  export CUDA_HOME="$_env_root" PATH="$_env_root/bin:$PATH"
  export CPATH="$_env_root/targets/x86_64-linux/include"
  export LIBRARY_PATH="$_env_root/targets/x86_64-linux/lib"
  resplat() { "$RESPLAT_PY" "$@"; }
else
  resplat() { $PY run --no-capture-output -n "$RESPLAT" python "$@"; }
fi
CKPT=${CKPT:-/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth}
DATA=${DATA:-/workspace/organizer-assets/datasets}
OUT=${OUT:-outputs/curve}; SNAP=${SNAP:-6,15,30}; BUDGET=${BUDGET:-42}
GPU=${GPU:-0}
CASE=${CASE:?case required, e.g. mipnerf360/bicycle}
g=${CASE%%/*}; s=${CASE#*/}
case $g in dl3dv) dd=DL3DV;; hw) dd=hw;; *) dd=$g;; esac
src="$DATA/$dd/$s"; tag=$(echo "$CASE" | tr '/' '-'); d="$OUT/$tag"; mkdir -p "$d"
export CUDA_VISIBLE_DEVICES=$GPU
if [ ! -f "$d/resplat/resplat_world.ply" ]; then
  [ -f "$d/work/split.json" ] || resplat competition/upstream/baseline/module_a/preprocess.py \
    --scene_dir "$src" --group "$g" --out "$d/work" --num_context 16 --llffhold 8 --max_resolution 960 > "$d/a.log" 2>&1
  resplat -m research.resplat_variant --work "$d/work" \
    --out "$d/resplat" --checkpoint "$CKPT" --num_refine 2 --device cuda:0 > "$d/r.log" 2>&1
fi
[ -f "$d/resplat/resplat_world.ply" ] || { echo "RESPLAT_FAIL $CASE"; exit 1; }
$PY run --no-capture-output -n "$FASTGS" python -m research.image_cache --work "$d/work" --out "$d/cache" >/dev/null 2>&1

score_curve(){  # dir snapdir
  local dd="$1" sd="$2" args=""
  for p in $(ls "$sd"/*.ply 2>/dev/null | sort); do
    n=$(basename "$p" .ply | sed 's/snap_//; s/s$//')
    args="$args --ply $p --name $n"
  done
  [ -n "$args" ] && $PY run --no-capture-output -n "$FASTGS" python -m research.curve_metrics \
    --source "$d/work" --out "$dd/curve.json" $args > "$dd/metrics.log" 2>&1
  rm -rf "$sd"/*.ply 2>/dev/null   # snapshots are ~100 MB each; the disk is small
}
run_fastgs(){
  [ -f "$d/fastgs/curve.json" ] && return; rm -rf "$d/fastgs"; mkdir -p "$d/fastgs/snaps"
  $PY run --no-capture-output -n "$FASTGS" python -m research.train_sweep \
    --source "$d/work" --model_path "$d/fastgs/model" --eval -r 1 --sh_degree 3 --quiet \
    --init_ply "$d/resplat/resplat_world.ply" --iterations 100000 --snapshots "" \
    --snap_dir "$d/fastgs/snaps" --snap_seconds "$SNAP" --limit_seconds "$BUDGET" --limit_reserve 1.0 \
    --position_lr_max_steps 5000 --opacity_reset_interval 0 --lambda_dssim 0.6 --scaling_lr 0.01 \
    --densify_until_iter 1250 --image_cache "$d/cache" --record "$d/fastgs/train_record.json" > "$d/fastgs/train.log" 2>&1
  score_curve "$d/fastgs" "$d/fastgs/snaps"
}
run_mcmc(){
  [ -f "$d/mcmc/curve.json" ] && return; rm -rf "$d/mcmc"; mkdir -p "$d/mcmc"
  TORCH_CUDA_ARCH_LIST=8.9 $PY run --no-capture-output -n "$FASTGS" python -m research.mcmc_budget \
    --work "$d/work" --init "$d/resplat/resplat_world.ply" --out "$d/mcmc/model" \
    --budget-seconds "$BUDGET" --budget-fraction 0.98 --device cuda:0 \
    --res-scales 0.5,0.75,1.0 --res-untils 0.4,0.7 --cap-mult 3.0 --refine-every 50 \
    --snap-seconds "$SNAP" > "$d/mcmc/train.log" 2>&1
  score_curve "$d/mcmc" "$d/mcmc/model/snaps"
}
run_inject(){
  [ -f "$d/inject/curve.json" ] && return; rm -rf "$d/inject"; mkdir -p "$d/inject/snaps"
  $PY run --no-capture-output -n "$FASTGS" python -m research.train_sweep \
    --source "$d/work" --model_path "$d/inject/model" --eval -r 1 --sh_degree 3 --quiet \
    --init_ply "$d/resplat/resplat_world.ply" --iterations 100000 --snapshots "" \
    --snap_dir "$d/inject/snaps" --snap_seconds "$SNAP" --limit_seconds "$BUDGET" --limit_reserve 1.0 \
    --position_lr_max_steps 5000 --opacity_reset_interval 0 --lambda_dssim 0.6 --scaling_lr 0.01 \
    --densify_until_iter 1250 --image_cache "$d/cache" --inject_holes 60000 --inject_at 800 \
    --record "$d/inject/train_record.json" > "$d/inject/train.log" 2>&1
  score_curve "$d/inject" "$d/inject/snaps"
}
run_fastgs; run_mcmc; run_inject
echo "CASE_DONE $CASE $(date -Is)"
