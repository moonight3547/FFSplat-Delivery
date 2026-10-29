#!/usr/bin/env bash
# Round 1 - module A / ReSplat lever probe.
# Question: does running more recurrent refinement passes improve the
# feed-forward initialization that module B has to start from?
# Everything is measured with the official metrics on the frozen held-out split.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
CKPT=${CKPT:-/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth}
OUT=${OUT:-outputs/experiments/round1}
mkdir -p "$OUT"
log="$OUT/run.log"
echo "start $(date -Is)" | tee -a "$log"

run() {  # case, num_refine, budget
  local case="$1" n="$2" budget="$3" tag
  tag=$(echo "$case" | tr '/' '-')
  echo "=== $case refine=$n $(date -Is)" >>"$log"
  if [ ! -f "$OUT/$tag-r$n/resplat_world.ply" ]; then
    $PY run --no-capture-output -n ffsplat-enhance python -m research.resplat_variant \
      --work "outputs/O0-20261030/work/$case" --out "$OUT/$tag-r$n" \
      --checkpoint "$CKPT" --num_refine "$n" --device cuda:0 >>"$log" 2>&1
    echo "resplat_exit=$?" >>"$log"
  fi
  $PY run --no-capture-output -n ffsplat-enhance python -m research.score_ply \
    --work "outputs/O0-20261030/work/$case" --out "$OUT/$tag-r$n-score" \
    --ply "$OUT/$tag-r$n/resplat_world.ply" --name "r$n" --budget "$budget" 2>&1 \
    | grep -E '^\{' | tee -a "$log"
}

for n in 4 6; do
  run "hw/billboard" "$n" 99.49
  run "mipnerf360/counter" "$n" 92.57
done
echo "done $(date -Is)" >>"$log"
