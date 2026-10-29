#!/usr/bin/env bash
# Evaluate the delivery recipe on every organizer scene (the user's ask: only test the
# 13-scene set with the competitive configuration, not every experiment).
# Per scene: module A -> FastGS random-30k reference (the per-scene budget) ->
# ReSplat (one 16-view pass) -> opacity-sharpened init -> delivery module B (5k,
# structural ops only in the first quarter) -> official metrics.
# Heavy intermediates are deleted after each scene: the host disk is small.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
RESPLAT=${RESPLAT:-ffsplat-enhance}; FASTGS=${FASTGS:-organizer-fastgs}
CKPT=${CKPT:-/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth}
DATA=${DATA:-/workspace/organizer-assets/datasets}
OUT=${OUT:-outputs/eval13}
mkdir -p "$OUT"
log="$OUT/run.log"
echo "start $(date -Is)" | tee -a "$log"
DL="001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f 0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce 0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e 2dad2d298927ff9b2ed7daa6128244e81d331e8cf1abf6c8598f35276ad38b84 3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3 9a451604b8d2583d47b14662635d8f74710c419cc061705ac15b8cfc31ece446"
HW="billboard basketball car trashbin"; MP="bicycle bonsai counter"
CASES=(); for s in $DL; do CASES+=("dl3dv/$s"); done
for s in $HW; do CASES+=("hw/$s"); done; for s in $MP; do CASES+=("mipnerf360/$s"); done
for case in "${CASES[@]}"; do
  group=${case%%/*}; scene=${case#*/}; d="$OUT/$case"
  # on-disk dataset folder names differ from the case group only for DL3DV
  case "$group" in dl3dv) dirname=DL3DV ;; *) dirname=$group ;; esac
  src="$DATA/$dirname/$scene"
  [ -f "$d/metrics.json" ] && [ -f "$d/metrics_baseline.json" ] && { echo "skip $case" >>"$log"; continue; }
  [ -d "$src" ] || { echo "missing $src" >>"$log"; continue; }
  mkdir -p "$d"; work="$d/work"
  echo "=== $case $(date -Is)" >>"$log"
  if [ ! -f "$work/split.json" ]; then
    $PY run --no-capture-output -n "$RESPLAT" python competition/upstream/baseline/module_a/preprocess.py \
      --scene_dir "$src" --group "$group" --out "$work" --num_context 16 --llffhold 8 --max_resolution 960 >>"$log" 2>&1 \
      || { echo "A failed $case" >>"$log"; continue; }
  fi
  if [ ! -f "$d/ref/module_b.json" ]; then
    $PY run --no-capture-output -n "$FASTGS" python competition/upstream/baseline/module_b/train_from_ply_fastgs.py \
      -s "$work" --eval -r 1 --sh_degree 3 --quiet -m "$d/ref" --iterations 30000 \
      --position_lr_max_steps 30000 --densify_until_iter 15000 --init_random >>"$log" 2>&1 \
      || { echo "ref failed $case" >>"$log"; continue; }
  fi
  budget=$(python3 -c "import json;print(json.load(open('$d/ref/module_b.json'))['total_seconds'])")
  if [ ! -f "$d/resplat/resplat_world.ply" ]; then
    $PY run --no-capture-output -n "$RESPLAT" python -m research.resplat_variant --work "$work" \
      --out "$d/resplat" --checkpoint "$CKPT" --num_refine 2 --device cuda:0 >>"$log" 2>&1 \
      || { echo "resplat failed $case" >>"$log"; continue; }
  fi
  if [ ! -f "$d/metrics_baseline.json" ]; then
    # organizer baseline for this scene: FastGS continuation 10k, default recipe
    $PY run --no-capture-output -n "$FASTGS" python -m research.run_sweep --case "$case" --work "$work" \
      --init-ply "$d/resplat/resplat_world.ply" --out "$d/b_base" --iterations 10000 --snapshots "" \
      --budget "$budget" >>"$log" 2>&1
    cp "$d/b_base/summary.json" "$d/b_base_summary.json" 2>/dev/null
    cp "$d/b_base/eval/iter10000/metrics.json" "$d/metrics_baseline.json" 2>/dev/null
    rm -rf "$d/b_base/model" "$d/b_base/eval" 2>/dev/null
  fi
  # matched 3-case A/B (research/experiments/init_ab3.sh): raw init beats the
  # opacity-sharpened variant on the mean, so the delivery recipe uses it unchanged
  cp "$d/resplat/resplat_world.ply" "$d/init_sharp.ply" 2>/dev/null
  $PY run --no-capture-output -n "$FASTGS" python -m research.run_sweep --case "$case" --work "$work" \
    --init-ply "$d/init_sharp.ply" --out "$d/b" --iterations 5000 --snapshots "" --budget "$budget" \
    --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 \
    --position-lr-max-steps 5000 --densify-until 1250 >>"$log" 2>&1
  cp "$d/b/summary.json" "$d/b_summary.json" 2>/dev/null
  cp "$d/b/eval/iter05000/metrics.json" "$d/metrics.json" 2>/dev/null
  python3 - "$d" "$case" >>"$log" 2>&1 <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
s = json.loads((d / "b_summary.json").read_text())
r = s["results"][0]; sc = r["score"]
print("RESULT %-40s B %6.1f PSNR %.3f SSIM %.4f LPIPS %.4f q %6.2f t %5.2f = %5.2f" % (
    sys.argv[2], r["b_seconds_estimate"], r["psnr"], r["ssim"], r["lpips"],
    sc["quality_points"], sc["time_points"], sc["total_points"]))
PY
  # keep only JSON + the initial PLY: the host disk is ~4 GB and there are 13 scenes
  # Only task-owned intermediates are removable here: $src is the organizer's raw
  # dataset, which lives outside this experiment and is expensive to re-upload.
  if [ -f "$d/metrics.json" ] && [ -f "$d/b_summary.json" ]; then
    rm -rf "$d/b/model" "$d/b/eval" "$d/ref/point_cloud" "$d/ref/test_renders" "$d/ref/test_gt" \
           "$d/work/images" 2>/dev/null
  fi
  du -sh "$d" >>"$log" 2>/dev/null
done
echo "done $(date -Is)" >>"$log"; grep -a RESULT "$log" | tail -20
