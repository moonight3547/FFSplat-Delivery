#!/usr/bin/env bash
# Round 2 - module B lever probe on the baseline (refine=2) initialization.
# Every variant is compared at equal *measured* module-B seconds; the proxy
# score always uses the per-case FastGS random-init 30K budget.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
OUT=${OUT:-outputs/experiments/round2}
mkdir -p "$OUT"
log="$OUT/run.log"
echo "start $(date -Is)" | tee -a "$log"

# wait for the previous round to release the GPU
while pgrep -f round1_refine >/dev/null 2>&1; do sleep 10; done

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

# a) do the periodic opacity resets help or hurt a short continuation?
sweep hw-noreset hw/billboard 99.49 12000 4000,8000,12000 --opacity-reset-interval 0
# b) coarse-to-fine resolution: same seconds, more optimizer steps
sweep hw-resramp hw/billboard 99.49 24000 8000,16000,24000 \
  --res-scales 0.5,0.75,1.0 --res-until 0.6,0.85
# c) optimize the scored perceptual metric directly (LPIPS is 40/100 of quality)
sweep hw-lpips hw/billboard 99.49 12000 4000,8000,12000 \
  --lpips-weight 0.2 --lpips-every 1 --lpips-scale 0.5
# d) same perceptual term on the scene with the worst LPIPS
sweep counter-lpips mipnerf360/counter 92.57 12000 4000,8000,12000 \
  --lpips-weight 0.2 --lpips-every 1 --lpips-scale 0.5
# e) coarse-to-fine on that scene
sweep counter-resramp mipnerf360/counter 92.57 24000 8000,16000,24000 \
  --res-scales 0.5,0.75,1.0 --res-until 0.6,0.85
# f) aggressive densification: keep detail instead of pruning it away
sweep hw-dense hw/billboard 99.49 12000 4000,8000,12000 \
  --densify-count-thresh 2 --max-gaussians 900000
sweep counter-dense mipnerf360/counter 92.57 12000 4000,8000,12000 \
  --densify-count-thresh 2 --max-gaussians 900000
echo "done $(date -Is)" >>"$log"
