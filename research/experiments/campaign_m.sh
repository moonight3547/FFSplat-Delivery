#!/usr/bin/env bash
# Campaign M - module A idea: two-subset feed-forward ensemble.
#
# Campaign H showed the ReSplat initialization is worth ~28 points, far more than any
# module-B design choice (2 points). So the highest-leverage remaining experiment is
# on the *initialization* itself. ReSplat consumes exactly 16 context views; the
# organizer picks one uniform-16 set. Here we run it twice on two different 16-view
# subsets of the same train split, merge the two Gaussian sets, and give the merged
# init to module B. ReSplat time is not part of module B's budget, so this is "free"
# quality if it works.
#
# Runs on the host that has the resplat environment (H1):
#   bash research/experiments/campaign_m.sh [case_filter]
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
FASTGS=${FASTGS:-organizer-fastgs}
RESPLAT=${RESPLAT:-ffsplat-enhance}
CKPT=${CKPT:-/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth}
D=${D:-outputs/experiments/campaignM}
FILTER=${1:-}
mkdir -p "$D"

HW=hw/billboard;        BHW=99.49
CO=mipnerf360/counter;  BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29

stage() {  # src_work dst_work offset
  local src="$1" dst="$2" offset="$3"
  rm -rf "$dst"; mkdir -p "$dst"
  ln -s "$(cd "$src" && pwd)/images" "$dst/images"
  ln -s "$(cd "$src" && pwd)/sparse" "$dst/sparse"
  python3 - "$src/split.json" "$dst/split.json" "$offset" <<'PY'
import json, sys
src, dst, offset = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = json.load(open(src))
train = s["train"]
n = 16
# set A: organizer's uniform-16 over the train order; set B: same spacing, offset
idx = [int(round(k * (len(train) - 1) / (n - 1))) for k in range(n)]
idx_b = [(i + offset) % len(train) for i in idx]
s["context"] = [train[i] for i in sorted(set(idx_b))]
s["context"] = s["context"][:n]
json.dump(s, open(dst, "w"))
print("staged context", len(s["context"]))
PY
}


run_case() {  # case budget
  local case="$1" budget="$2" tag workA workB out
  tag=$(echo "$case" | tr '/' '-')
  [ -n "$FILTER" ] && [[ "$case" != *"$FILTER"* ]] && return
  local src="outputs/O0-20261030/work/$case"
  local init="outputs/O0-20261030/results/$case/resplat/resplat_world.ply"
  out="$D/$tag"
  mkdir -p "$out"
  [ -f "$out/merged.ply" ] || {
    stage "$src" "$out/workB" 7
    $PY run --no-capture-output -n "$RESPLAT" python -m research.resplat_variant \
      --work "$out/workB" --out "$out/resplat_b" --checkpoint "$CKPT" \
      --num_refine 2 --device cuda:0 >>"$out/log.txt" 2>&1
    $PY run --no-capture-output -n "$FASTGS" python -m research.ply_ops \
      --in "$init" --in "$out/resplat_b/resplat_world.ply" --out "$out/merged.ply" \
      --op merge >>"$out/log.txt" 2>&1
  }
  for steps in 2500 3500; do
    local bdir="$out/merged_${steps}"
    [ -e "$bdir/summary.json" ] && continue
    echo "=== $case merged ${steps} $(date -Is)" >>"$D/queue.log"
    $PY run --no-capture-output -n "$FASTGS" python -m research.run_sweep \
      --case "$case" --work "$src" --init-ply "$out/merged.ply" \
      --out "$bdir" --iterations "$steps" --snapshots "" --budget "$budget" \
      --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 \
      --position-lr-max-steps "$steps" >>"$out/log.txt" 2>&1
    echo "exit=$? $case merged $steps $(date -Is)" >>"$D/queue.log"
  done
}

run_case "$HW" "$BHW"
run_case "$CO" "$BCO"
run_case "$DL" "$BDL"
echo "done $(date -Is)" >>"$D/queue.log"
