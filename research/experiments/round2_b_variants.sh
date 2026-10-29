#!/usr/bin/env bash
# Round 2 - module B lever probe on the baseline (refine=2) initialization.
# Variants are only ever compared at equal *measured* module-B seconds; the
# proxy score uses the per-case FastGS random-init 30K budget.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
OUT=${OUT:-outputs/experiments/round2}
mkdir -p "$OUT"
log="$OUT/run.log"
echo "start $(date -Is)" | tee -a "$log"

sweep() {  # tag, case, budget, iterations, snapshots, extra args...
  local tag="$1" case="$2" budget="$3" iters="$4" snaps="$5"
  shift 5
  local dir="$OUT/$tag"
  echo "=== $tag $(date -Is) :: $*" >>"$log"
  [ -e "$dir" ] && { echo "skip existing $dir" >>"$log"; return; }
  $PY run --no-capture-output -n ffsplat-enhance python -m research.run_sweep \
    --case "$case" --work "outputs/O0-20261030/work/$case" \
    --init-ply "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" \
    --out "$dir" --iterations "$iters" --snapshots "$snaps" --budget "$budget" \
    "$@" >>"$log" 2>&1
  echo "sweep_exit=$?" >>"$log"
  python3 - "$dir/summary.json" <<'PY' 2>/dev/null | tee -a "$log"
import json, sys
d = json.load(open(sys.argv[1]))
for r in d["results"]:
    s = r["score"]
    print("it %6d N %7d B %6.2fs PSNR %7.3f SSIM %.4f LPIPS %.4f | q %5.2f t %5.2f = %5.2f" % (
        r["iteration"], r["num_gaussians"], r["b_seconds_estimate"], r["psnr"], r["ssim"],
        r["lpips"], s["quality_points"], s["time_points"], s["total_points"]))
PY
}

# a) does the periodic opacity reset help or hurt a short continuation?
sweep hw-noreset hw/billboard 99.49 12000 4000,8000,12000 --opacity_reset_interval 0
# b) coarse-to-fine resolution: same seconds, more optimizer steps
sweep hw-resramp hw/billboard 99.49 24000 8000,16000,24000 \
  --res_scales 0.5,0.75,1.0 --res_until 0.6,0.85
echo "done $(date -Is)" >>"$log"
