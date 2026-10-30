#!/usr/bin/env bash
# Optimization-friendliness diagnostic (ForeSplat idea, no extra training cost):
# measure the same init at 0 / 500 / 2000 / 5000 module-B steps.
#   large slope  -> optimization-limited -> invest in module B
#   bad at 0 step -> init-limited -> invest in module A / initialization
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
RESPLAT=${RESPLAT:-ffsplat-enhance}; FASTGS=${FASTGS:-organizer-fastgs}
CKPT=${CKPT:-/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth}
DATA=${DATA:-/workspace/organizer-assets/datasets}
OUT=${OUT:-outputs/ocurve}; mkdir -p "$OUT"
CASES=("dl3dv:0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce:94.29:hard"
        "dl3dv:0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e:94.29:hard"
        "dl3dv:3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3:94.29:hard"
        "hw:billboard:99.49:easy"
        "mipnerf360:counter:92.57:mid")
for spec in "${CASES[@]}"; do
  g=${spec%%:*}; rest=${spec#*:}; s=${rest%%:*}; rest2=${rest#*:}; bud=${rest2%%:*}; kind=${rest2##*:}
  case $g in dl3dv) dd=DL3DV;; *) dd=$g;; esac
  tag="$kind-$(echo "$s" | cut -c1-10)"; d="$OUT/$tag"; mkdir -p "$d"
  if [ ! -f "$d/resplat/resplat_world.ply" ]; then
    [ -f "$d/work/split.json" ] || $PY run --no-capture-output -n "$RESPLAT" python competition/upstream/baseline/module_a/preprocess.py \
      --scene_dir "$DATA/$dd/$s" --group "$g" --out "$d/work" --num_context 16 --llffhold 8 --max_resolution 960 > "$d/a.log" 2>&1
    $PY run --no-capture-output -n "$RESPLAT" python -m research.resplat_variant --work "$d/work" \
      --out "$d/resplat" --checkpoint "$CKPT" --num_refine 2 --device cuda:0 --fix-sh-basis > "$d/r.log" 2>&1
  fi
  [ -f "$d/resplat/resplat_world.ply" ] || { echo "PREP_FAIL $tag"; continue; }
  $PY run --no-capture-output -n "$FASTGS" python -m research.image_cache --work "$d/work" --out "$d/cache" >/dev/null 2>&1
  # 0-step point
  if [ ! -f "$d/s0/metrics.json" ]; then
    rm -rf "$d/s0"
    $PY run --no-capture-output -n "$FASTGS" python -m research.render_eval --source "$d/work" \
      --out "$d/s0/eval" --eval -r 1 --sh_degree 3 --ply "$d/resplat/resplat_world.ply" --name step0 > "$d/s0.log" 2>&1
    $PY run --no-capture-output -n "$FASTGS" python competition/upstream/baseline/evaluate.py \
      --model_path "$d/s0/eval/step0" --device cuda:0 >> "$d/s0.log" 2>&1
    cp "$d/s0/eval/step0/metrics.json" "$d/s0/metrics.json" 2>/dev/null
    rm -rf "$d/s0/eval"
  fi
  for steps in 500 2000 5000; do
    [ -f "$d/s$steps/summary.json" ] && continue
    rm -rf "$d/s$steps"
    $PY run --no-capture-output -n "$FASTGS" python -m research.run_sweep --case "$g/$s" --work "$d/work" \
      --init-ply "$d/resplat/resplat_world.ply" --out "$d/s$steps" --iterations "$steps" --snapshots "" \
      --budget "$bud" --position-lr-max-steps "$steps" --densify-until $((steps/4)) --image-cache "$d/cache" \
      --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 --realloc-frac 0.15 --realloc-at $((steps/4)) \
      > "$d/s$steps.log" 2>&1
    rm -rf "$d/s$steps/model" "$d/s$steps/eval" 2>/dev/null
  done
  echo "OCURVE_DONE $tag"
  rm -rf "$d/work/images"
done
python3 - "$OUT" <<'PY'
import json, glob, sys
def q(m):
    return ((min(max(m["psnr"]-15,0),25)/25*20)
            + (min(max(m["ssim"]-0.70,0),0.25)/0.25*20)
            + (min(max(0.25-m["lpips"],0),0.20)/0.20*40))
for d in sorted(glob.glob(sys.argv[1] + "/*/")):
    tag = d.rstrip("/").split("/")[-1]
    vals = []
    try:
        vals.append(("0", q(json.load(open(d + "s0/metrics.json"))["mean"])))
    except Exception:
        vals.append(("0", None))
    for st in (500, 2000, 5000):
        try:
            r = json.load(open(d + "s%d/summary.json" % st))["results"][0]
            vals.append((str(st), r["score"]["quality_points"]))
        except Exception:
            vals.append((str(st), None))
    print(tag, " ".join("%s=%s" % (k, "-" if v is None else "%.2f" % v) for k, v in vals))
PY
