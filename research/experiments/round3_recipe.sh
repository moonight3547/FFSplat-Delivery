#!/usr/bin/env bash
# Round 3 - apply the winning short-continuation recipe to the remaining cases
# and cross-combine the two levers that helped (no opacity reset, denser growth).
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
OUT=${OUT:-outputs/experiments/round3}
mkdir -p "$OUT"
log="$OUT/run.log"
echo "start $(date -Is)" | tee -a "$log"

while pgrep -f round2_b_variants >/dev/null 2>&1; do sleep 15; done

sweep() {  # tag, case, budget, iterations, snapshots, extra args...
  local tag="$1" case="$2" budget="$3" iters="$4" snaps="$5"
  shift 5
  local dir="$OUT/$tag"
  echo "=== $tag $(date -Is) :: $*" >>"$log"
  if [ -e "$dir" ]; then echo "skip existing $dir" >>"$log"; return; fi
  $PY run --no-capture-output -n ffsplat-enhance python -m research.run_sweep \
    --case "$case" --work "outputs/O0-20261030/work/$case" \
    --init-ply "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" \
    --out "$dir" --iterations "$iters" --snapshots "$snaps" --budget "$budget" \
    "$@" >>"$log" 2>&1
  echo "sweep_exit=$?" >>"$log"
  python3 - "$dir/summary.json" <<'PY' 2>/dev/null | tee -a "$log"
import json, sys
d = json.load(open(sys.argv[1]))
print("  " + sys.argv[1].split("/")[-2])
for r in d["results"]:
    s = r["score"]
    print("  it %6d N %7d B %6.2fs PSNR %7.3f SSIM %.4f LPIPS %.4f | q %5.2f t %5.2f = %5.2f" % (
        r["iteration"], r["num_gaussians"], r["b_seconds_estimate"], r["psnr"], r["ssim"],
        r["lpips"], s["quality_points"], s["time_points"], s["total_points"]))
PY
}

# no-reset on the other two cases (hw already measured: 45.17 at 4k)
sweep counter-noreset mipnerf360/counter 92.57 8000 4000,8000 --opacity-reset-interval 0
sweep dl3dv-noreset "dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f" \
  94.29 12000 4000,8000,12000 --opacity-reset-interval 0
# combine the two levers that helped on hw
sweep hw-dense-noreset hw/billboard 99.49 8000 4000,8000 \
  --opacity-reset-interval 0 --densify-count-thresh 2 --max-gaussians 900000
echo "done $(date -Is)" >>"$log"
