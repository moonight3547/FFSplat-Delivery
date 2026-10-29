#!/usr/bin/env bash
# Module-A test under the organizer's constraint: pick at most 16 context views and
# run ReSplat ONCE (no multi-subset ensembles). Three deterministic policies are
# compared end-to-end with the delivery module-B recipe, so the number that matters
# is the final score, not the zero-step initialization quality.
#
# Runs on the only host with the ReSplat environment (H1).
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
RESPLAT=${RESPLAT:-ffsplat-enhance}; FASTGS=${FASTGS:-organizer-fastgs}
CKPT=${CKPT:-/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth}
D=${D:-outputs/experiments/selection}
mkdir -p "$D"
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29

run_case() {  # case budget
  local case="$1" budget="$2" tag src policy dir work ply
  tag=$(echo "$case" | tr '/' '-'); src="outputs/O0-20261030/work/$case"
  for policy in uniform fps coverage; do
    dir="$D/$tag-$policy"
    [ -f "$dir/metrics.json" ] && continue
    rm -rf "$dir"; mkdir -p "$dir"
    echo "=== $case $policy $(date -Is)" >>"$D/queue.log"
    $PY run --no-capture-output -n "$RESPLAT" python -m research.select_views \
      --work "$src" --out "$dir/sel" --policy "$policy" >>"$dir/log.txt" 2>&1
    [ -f "$dir/sel/context16.json" ] || { echo "select failed $policy" >>"$D/queue.log"; continue; }
    # stage a work dir that only differs in split.json["context"]
    work="$dir/work"; rm -rf "$work"; mkdir -p "$work"
    ln -s "$(cd "$src" && pwd)/images" "$work/images"
    ln -s "$(cd "$src" && pwd)/sparse" "$work/sparse"
    python3 - "$src/split.json" "$dir/sel/context16.json" "$work/split.json" <<'PY'
import json, sys
split = json.load(open(sys.argv[1])); sel = json.load(open(sys.argv[2]))
split["context"] = sel["context"]
json.dump(split, open(sys.argv[3], "w"))
PY
    $PY run --no-capture-output -n "$RESPLAT" python -m research.resplat_variant \
      --work "$work" --out "$dir/resplat" --checkpoint "$CKPT" --num_refine 2 \
      --device cuda:0 >>"$dir/log.txt" 2>&1
    ply="$dir/resplat/resplat_world.ply"
    [ -f "$ply" ] || { echo "resplat failed $policy" >>"$D/queue.log"; continue; }
    $PY run --no-capture-output -n "$FASTGS" python -m research.ply_ops \
      --in "$ply" --out "$dir/init_sharp.ply" --op sharpen --gain 2.0 >>"$dir/log.txt" 2>&1
    $PY run --no-capture-output -n "$FASTGS" python -m research.run_sweep \
      --case "$case" --work "$work" --init-ply "$dir/init_sharp.ply" \
      --out "$dir/b" --iterations 5000 --snapshots "" --budget "$budget" \
      --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 \
      --position-lr-max-steps 5000 --densify-until 1250 >>"$dir/log.txt" 2>&1
    cp "$dir/b/eval/iter05000/metrics.json" "$dir/metrics.json" 2>/dev/null
    cp "$dir/b/summary.json" "$dir/b_summary.json" 2>/dev/null
    rm -rf "$dir/b/model" "$dir/b/eval" 2>/dev/null
    echo "done $case $policy" >>"$D/queue.log"
  done
}
run_case "$HW" "$BHW"
run_case "$CO" "$BCO"
run_case "$DL" "$BDL"
echo all-done >>"$D/queue.log"
