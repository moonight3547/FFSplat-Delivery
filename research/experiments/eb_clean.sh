#!/usr/bin/env bash
# E-B: does cleaning the exported ReSplat PLY help?
# For each hard scene: A -> ReSplat -> (a) 0-step render of raw vs cleaned PLY,
# (b) 5K delivery training on raw vs cleaned. Gate: 0-step LPIPS -0.02 or 5K +2 total.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
RESPLAT=${RESPLAT:-ffsplat-enhance}; FASTGS=${FASTGS:-organizer-fastgs}
CKPT=${CKPT:-/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth}
DATA=${DATA:-/workspace/organizer-assets/datasets}
OUT=${OUT:-outputs/eb}; mkdir -p "$OUT"
CASES=("dl3dv:0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce:94.29"
        "dl3dv:0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e:94.29"
        "dl3dv:3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3:94.29")
for spec in "${CASES[@]}"; do
  g=${spec%%:*}; rest=${spec#*:}; s=${rest%%:*}; bud=${rest##*:}
  case $g in dl3dv) dd=DL3DV;; *) dd=$g;; esac
  tag=$(echo "$g-$s" | cut -c1-20); d="$OUT/$tag"; mkdir -p "$d"
  if [ ! -f "$d/resplat/resplat_world.ply" ]; then
    [ -f "$d/work/split.json" ] || $PY run --no-capture-output -n "$RESPLAT" python competition/upstream/baseline/module_a/preprocess.py \
      --scene_dir "$DATA/$dd/$s" --group "$g" --out "$d/work" --num_context 16 --llffhold 8 --max_resolution 960 > "$d/a.log" 2>&1
    $PY run --no-capture-output -n "$RESPLAT" python -m research.resplat_variant --work "$d/work" \
      --out "$d/resplat" --checkpoint "$CKPT" --num_refine 2 --device cuda:0 > "$d/r.log" 2>&1
  fi
  [ -f "$d/resplat/resplat_world.ply" ] || { echo "PREP_FAIL $tag"; continue; }
  # cleaned copy: drop floater-like Gaussians (low opacity / extreme scale)
  [ -f "$d/clean.ply" ] || $PY run --no-capture-output -n "$FASTGS" python -m research.ply_ops \
    --in "$d/resplat/resplat_world.ply" --out "$d/clean.ply" --op prune > "$d/prune.log" 2>&1
  for arm in raw clean; do
    pl=$([ "$arm" = raw ] && echo "$d/resplat/resplat_world.ply" || echo "$d/clean.ply")
    [ -f "$pl" ] || continue
    # 0-step: render + official metrics, no training
    if [ ! -f "$d/$arm-0/metrics.json" ]; then
      rm -rf "$d/$arm-0"
      $PY run --no-capture-output -n "$FASTGS" python -m research.render_eval --source "$d/work" \
        --out "$d/$arm-0/eval" --eval -r 1 --sh_degree 3 --ply "$pl" --name step0 >> "$d/$arm.log" 2>&1
      $PY run --no-capture-output -n "$FASTGS" python competition/upstream/baseline/evaluate.py \
        --model_path "$d/$arm-0/eval/step0" --device cuda:0 >> "$d/$arm.log" 2>&1
      cp "$d/$arm-0/eval/step0/metrics.json" "$d/$arm-0/metrics.json" 2>/dev/null
      rm -rf "$d/$arm-0/eval"
    fi
    # 5K delivery training on that init
    if [ ! -f "$d/$arm-5k/summary.json" ]; then
      rm -rf "$d/$arm-5k"
      $PY run --no-capture-output -n "$FASTGS" python -m research.image_cache --work "$d/work" --out "$d/cache" >/dev/null 2>&1
      $PY run --no-capture-output -n "$FASTGS" python -m research.run_sweep --case "$g/$s" --work "$d/work" \
        --init-ply "$pl" --out "$d/$arm-5k" --iterations 5000 --snapshots "" --budget "$bud" \
        --position-lr-max-steps 5000 --densify-until 1250 --image-cache "$d/cache" \
        --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 --realloc-frac 0.15 --realloc-at 1250 \
        > "$d/$arm-5k.log" 2>&1
      rm -rf "$d/$arm-5k/model" "$d/$arm-5k/eval" 2>/dev/null
    fi
  done
  echo "EB_DONE $tag"
  rm -rf "$d/work/images"
done
python3 - "$OUT" <<'PY'
import json, glob, sys
for d in sorted(glob.glob(sys.argv[1] + "/*/")):
    tag = d.rstrip("/").split("/")[-1]
    row = [tag]
    for arm in ("raw", "clean"):
        try:
            m = json.load(open(d + arm + "-0/metrics.json"))["mean"]
            row.append("%s0step LPIPS %.4f" % (arm, m["lpips"]))
        except Exception:
            row.append("%s0step -" % arm)
        try:
            s = json.load(open(d + arm + "-5k/summary.json"))["results"][0]
            row.append("%s5k %.2f" % (arm, s["score"]["total_points"]))
        except Exception:
            row.append("%s5k -" % arm)
    print(" | ".join(row))
PY
