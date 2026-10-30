#!/usr/bin/env bash
# LPIPS-vs-time curve on the low-scoring scenes: FastGS delivery vs 3DGS-MCMC.
# Both trainers dump PLYs at the same wall-clock seconds; module-B time excludes the
# metric computation (snapshots are scored afterwards by research.curve_metrics).
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
RESPLAT=${RESPLAT:-ffsplat-enhance}; FASTGS=${FASTGS:-organizer-fastgs}
CKPT=${CKPT:-/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth}
DATA=${DATA:-/workspace/organizer-assets/datasets}
OUT=${OUT:-outputs/curve}; mkdir -p "$OUT"
SNAP=${SNAP:-5,10,15,20,30,40}
BUDGET=${BUDGET:-42}
for spec in "dl3dv:0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce:94.29" \
            "dl3dv:0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e:94.29" \
            "dl3dv:3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3:94.29" \
            "mipnerf360:bicycle:100"; do
  g=${spec%%:*}; rest=${spec#*:}; s=${rest%%:*}; b=${rest##*:}
  case $g in dl3dv) dd=DL3DV;; *) dd=$g;; esac
  case_id="$g/$s"; tag=$(echo "$case_id" | tr '/' '-'); d="$OUT/$tag"
  mkdir -p "$d"
  if [ ! -f "$d/resplat/resplat_world.ply" ]; then
    [ -f "$d/work/split.json" ] || $PY run --no-capture-output -n "$RESPLAT" python competition/upstream/baseline/module_a/preprocess.py \
      --scene_dir "$DATA/$dd/$s" --group "$g" --out "$d/work" --num_context 16 --llffhold 8 --max_resolution 960 > "$d/a.log" 2>&1
    $PY run --no-capture-output -n "$RESPLAT" python -m research.resplat_variant --work "$d/work" \
      --out "$d/resplat" --checkpoint "$CKPT" --num_refine 2 --device cuda:0 > "$d/r.log" 2>&1
  fi
  [ -f "$d/resplat/resplat_world.ply" ] || { echo "resplat failed $case_id"; continue; }
  $PY run --no-capture-output -n "$FASTGS" python -m research.image_cache --work "$d/work" --out "$d/cache" >/dev/null 2>&1
  # FastGS delivery recipe with snapshots
  if [ ! -f "$d/fastgs/curve.json" ]; then
    rm -rf "$d/fastgs"; mkdir -p "$d/fastgs/snaps"
    $PY run --no-capture-output -n "$FASTGS" python -m research.train_sweep \
      --source "$d/work" --model_path "$d/fastgs/model" --eval -r 1 --sh_degree 3 --quiet \
      --init_ply "$d/resplat/resplat_world.ply" --iterations 100000 --snapshots "" \
      --snap_dir "$d/fastgs/snaps" --snap_seconds "$SNAP" --limit_seconds "$BUDGET" \
      --limit_reserve 1.0 --position_lr_max_steps 5000 --opacity_reset_interval 0 \
      --lambda_dssim 0.6 --scaling_lr 0.01 --densify_until_iter 1250 \
      --image_cache "$d/cache" --record "$d/fastgs/train_record.json" > "$d/fastgs/train.log" 2>&1
    plies=$(ls "$d"/fastgs/snaps/*.ply 2>/dev/null | sort)
    args=""; for p in $plies; do n=$(basename "$p" .ply | sed 's/snap_//; s/s$//'); args="$args --ply $p --name $n"; done
    $PY run --no-capture-output -n "$FASTGS" python -m research.curve_metrics --source "$d/work" \
      --out "$d/fastgs/curve.json" $args > "$d/fastgs/metrics.log" 2>&1
  fi
  # MCMC with snapshots
  if [ ! -f "$d/mcmc/curve.json" ]; then
    rm -rf "$d/mcmc"; mkdir -p "$d/mcmc"
    TORCH_CUDA_ARCH_LIST=8.9 $PY run --no-capture-output -n "$FASTGS" python -m research.mcmc_budget \
      --work "$d/work" --init "$d/resplat/resplat_world.ply" --out "$d/mcmc/model" \
      --budget-seconds "$BUDGET" --budget-fraction 0.98 --device cuda:0 \
      --res-scales 0.5,0.75,1.0 --res-untils 0.4,0.7 --cap-mult 3.0 --refine-every 50 \
      --snap-seconds "$SNAP" > "$d/mcmc/train.log" 2>&1
    plies=$(ls "$d"/mcmc/model/snaps/*.ply 2>/dev/null | sort)
    args=""; for p in $plies; do n=$(basename "$p" .ply | sed 's/snap_//; s/s$//'); args="$args --ply $p --name $n"; done
    $PY run --no-capture-output -n "$FASTGS" python -m research.curve_metrics --source "$d/work" \
      --out "$d/mcmc/curve.json" $args > "$d/mcmc/metrics.log" 2>&1
  fi
  echo "DONE $case_id"
  rm -rf "$d/work/images"
done
echo CURVE_ALL_DONE
