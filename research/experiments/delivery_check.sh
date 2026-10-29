#!/usr/bin/env bash
# Serial, uncontended measurement of the delivery recipe.
# Run this on a machine where nothing else uses the GPU: the score's TIME term is
# 20 points and is linear in B seconds, so concurrent jobs distort it by up to 45%.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}; ENV=${ENV:-organizer-fastgs}
D=${D:-outputs/experiments/delivery}
mkdir -p "$D"
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
for spec in "$HW|$BHW" "$CO|$BCO" "$DL|$BDL"; do
  case=${spec%|*}; budget=${spec#*|}; tag=$(echo "$case" | tr '/' '-')
  dir="$D/$tag"
  [ -e "$dir/summary.json" ] && { echo "skip $tag"; continue; }
  rm -rf "$dir"
  ply="$D/ply_$(echo $case | tr '/' '_').ply"
  mkdir -p "$D"
  [ -f "$ply" ] || $PY run --no-capture-output -n "$ENV" python -m research.ply_ops \
    --in "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" --out "$ply" \
    --op sharpen --gain 2.0 >/dev/null 2>&1
  echo "=== delivery $case $(date -Is)"
  $PY run --no-capture-output -n "$ENV" python -m research.run_sweep \
    --case "$case" --work "outputs/O0-20261030/work/$case" --init-ply "$ply" \
    --out "$dir" --iterations 5000 --snapshots "" --budget "$budget" \
    --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 \
    --position-lr-max-steps 5000 --densify-until 1250 >"$D/$tag.log" 2>&1
  rm -rf "$dir/model" "$dir/eval" 2>/dev/null
done
python3 - "$D" <<'PY'
import json, glob, sys, statistics
rows = []
for p in sorted(glob.glob(sys.argv[1] + "/*/summary.json")):
    d = json.load(open(p))
    for r in d["results"]:
        s = r["score"]
        rows.append((d["case"], r["b_seconds_estimate"], r["psnr"], r["ssim"], r["lpips"],
                     s["quality_points"], s["time_points"], s["total_points"]))
for r in rows:
    print("%-30s B %6.2fs PSNR %.3f SSIM %.4f LPIPS %.4f q %6.2f t %5.2f = %5.2f" % r)
if rows:
    print("MEAN total = %.2f" % statistics.mean(r[-1] for r in rows))
PY
