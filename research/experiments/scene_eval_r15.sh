#!/usr/bin/env bash
# Prepare-one-evaluate-one loop: the preparation host has only a few GB of disk, so
# each scene is taken through A -> ReSplat -> both module-B arms -> metrics, its
# numbers are appended to a JSONL, and its artifacts are removed before the next scene.
# This yields the complete 13-scene baseline-vs-delivery table on a single GPU.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
RESPLAT=${RESPLAT:-ffsplat-enhance}; FASTGS=${FASTGS:-organizer-fastgs}
CKPT=${CKPT:-/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth}
DATA=${DATA:-/workspace/organizer-assets/datasets}
OUT=${OUT:-outputs/scene13r15}; mkdir -p "$OUT"
RES="$OUT/results.jsonl"; log="$OUT/run.log"
echo "start $(date -Is)" >>"$log"
DL="001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f 0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce 0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e 2dad2d298927ff9b2ed7daa6128244e81d331e8cf1abf6c8598f35276ad38b84 3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3 9a451604b8d2583d47b14662635d8f74710c419cc061705ac15b8cfc31ece446"
HW="billboard basketball car trashbin"; MP="bicycle bonsai counter"
CASES=(); for s in $DL; do CASES+=("dl3dv/$s"); done
for s in $HW; do CASES+=("hw/$s"); done; for s in $MP; do CASES+=("mipnerf360/$s"); done

for case in "${CASES[@]}"; do
  if [ -n "${ONLY:-}" ] && [ "${case#*/}" != "${ONLY}" ]; then continue; fi
  group=${case%%/*}; scene=${case#*/}; d="$OUT/work"
  case "$group" in dl3dv) dd=DL3DV ;; *) dd=$group ;; esac
  src="$DATA/$dd/$scene"
  if grep -q "\"$case\"" "$RES" 2>/dev/null; then echo "skip $case" >>"$log"; continue; fi
  [ -d "$src" ] || { echo "missing $src" >>"$log"; continue; }
  case "$case" in
    hw/billboard) budget=99.49 ;;
    mipnerf360/counter) budget=92.57 ;;
    dl3dv/*) budget=94.29 ;;
    *) budget=95 ;;
  esac
  echo "=== $case $(date -Is)" >>"$log"
  # stale copies would silently carry the previous scene's numbers if a run fails
  rm -f "$OUT/base_summary.json" "$OUT/del_summary.json"
  rm -rf "$d"
  $PY run --no-capture-output -n "$RESPLAT" python competition/upstream/baseline/module_a/preprocess.py \
    --scene_dir "$src" --group "$group" --out "$d" --num_context 16 --llffhold 8 --max_resolution 960 >>"$log" 2>&1
  [ -f "$d/split.json" ] || { echo "A failed $case" >>"$log"; rm -rf "$d"; continue; }
  $PY run --no-capture-output -n "$RESPLAT" python -m research.resplat_variant --work "$d" \
    --out "$OUT/resplat" --checkpoint "$CKPT" --num_refine 2 --device cuda:0 \
    --fix-sh-basis >>"$log" 2>&1
  init="$OUT/resplat/resplat_world.ply"
  [ -f "$init" ] || { echo "resplat failed $case" >>"$log"; rm -rf "$d"; continue; }
  $PY run --no-capture-output -n "$FASTGS" python -m research.image_cache --work "$d" --out "$OUT/cache" >>"$log" 2>&1
  # arm 1: organizer baseline module B (10k, default schedule)
  $PY run --no-capture-output -n "$FASTGS" python -m research.run_sweep --case "$case" --work "$d" \
    --init-ply "$init" --out "$OUT/base" --iterations 10000 --snapshots "" --budget "$budget" \
    --opacity-reset-interval 3000 --position-lr-max-steps 10000 --densify-until 15000 \
    --image-cache "$OUT/cache" >>"$log" 2>&1
  cp "$OUT/base/summary.json" "$OUT/base_summary.json" 2>/dev/null
  # arm 2: delivery recipe
  $PY run --no-capture-output -n "$FASTGS" python -m research.run_sweep --case "$case" --work "$d" \
    --init-ply "$init" --out "$OUT/del" --iterations 5000 --snapshots "" --budget "$budget" \
    --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 --position-lr-max-steps 5000 \
    --densify-until 1250 --image-cache "$OUT/cache" --realloc-frac 0.15 --realloc-at 1250 >>"$log" 2>&1
  cp "$OUT/del/summary.json" "$OUT/del_summary.json" 2>/dev/null
  python3 - "$case" "$OUT/base_summary.json" "$OUT/del_summary.json" >>"$RES" <<'PY'
import json, sys
case, base_p, del_p = sys.argv[1], sys.argv[2], sys.argv[3]
def pick(p):
    try:
        d = json.load(open(p)); r = d["results"][0]; s = r["score"]
        return dict(b=r["b_seconds_estimate"], psnr=r["psnr"], ssim=r["ssim"], lpips=r["lpips"],
                    q=s["quality_points"], t=s["time_points"], total=s["total_points"])
    except Exception:
        return None
print(json.dumps(dict(case=case, baseline=pick(base_p), delivery=pick(del_p))))
PY
  rm -rf "$d" "$OUT/resplat" "$OUT/cache" "$OUT/base" "$OUT/del" 2>/dev/null
  echo "done $case $(date -Is)" >>"$log"
done
echo "ALL_DONE $(date -Is)" >>"$log"
