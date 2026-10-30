#!/usr/bin/env bash
# N1 texture-aware voxel-growth experiment (task-7).  One driver, 3 lanes, no overlap.
#
#   bash run.sh <cells.tsv> <LANE> <NLANES>          # parallel sweep, one lane per GPU
#   bash run.sh serial.tsv 0 1                       # serial anchors (single card)
#
# Per cell: (optionally) build the transformed init PLY with research.ply_voxgrow.py
# (cached under $D/ply), then run the unchanged delivery recipe of module B:
#   --iterations 5000 --snapshots "" --position-lr-max-steps 5000
#   --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01
#   --densify-from 500 --densify-until 1250 --realloc-frac 0.15 --realloc-at 1250
#   --image-cache <work>/image_cache --init-ply <transformed PLY>
# Resumable: a cell with summary.json is skipped.  Logs/queue under $D.
set -u
CELLS=${1:?cells tsv}
[ "${CELLS#/}" = "$CELLS" ] && CELLS="$PWD/$CELLS"   # must survive the cd below
LANE=${2:-0}; NLANES=${3:-3}
GPU_LANE=${GPU_LANE:-$LANE}   # physical GPU/profile lane; round-robin lane stays $LANE
REPO=/root/projects/FFSplat-Hackathon   # GPU lane = $LANE -> competition/profiles/gpufree-$LANE.json
D=${D:-/root/gpufree-data/outputs/texvox-20261030}
PYENV=/opt/conda/envs/fastgs/bin/python
export PYTHONPATH=$REPO
mkdir -p "$D/logs" "$D/cells" "$D/ply"
cd "$REPO" || exit 1

transform() {  # $1 tag, $2 init ply, $3 transform args  -> echoes transformed ply path
  local tag=$1 init=$2 args=$3 out="$D/ply/$1.ply"
  if [ -s "$out" ]; then echo "$out"; return 0; fi
  # shellcheck disable=SC2086
  $PYENV -m research.ply_voxgrow --in "$init" --out "$out" $args \
      --summary-out "${out%.ply}.json" --seed 0 >>"$D/logs/transform.log" 2>&1
  if [ ! -s "$out" ]; then echo "[transform-failed] $tag" >&2; return 1; fi
  echo "$out"
}

one() {  # tag case work init budget arm args
  local tag=$1 case=$2 work=$3 init=$4 budget=$5 arm=$6 args=$7
  local dir="$D/cells/$tag"
  # never delete a cell dir: a restart must not be able to wipe results.  A dir with
  # summary.json is done; a dir without one is a crashed/partial cell -> skip and log it.
  if [ -f "$dir/summary.json" ]; then echo "skip $tag (done)"; return 0; fi
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "RESUME-SKIP $tag (partial cell dir kept: $dir)" | tee -a "$D/queue.log"; return 0
  fi
  local plypath=$init
  if [ "$arm" != "control" ]; then
    plypath=$(transform "$tag" "$init" "$args") || { echo "TRANSFORM_FAIL $tag" >>"$D/queue.log"; return 1; }
  fi
  echo "start $tag lane$LANE $(date -Is) ply=$plypath" >>"$D/queue.log"
  /opt/conda/bin/conda run --no-capture-output -n fastgs python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$GPU_LANE.json" --case "$case" --work "$work" \
    --init-ply "$plypath" --out "$dir" --iterations 5000 --snapshots "" \
    --budget "$budget" --position-lr-max-steps 5000 --opacity-reset-interval 0 \
    --lambda-dssim 0.6 --scaling-lr 0.01 --densify-from 500 --densify-until 1250 \
    --realloc-frac 0.15 --realloc-at 1250 --image-cache "$work/image_cache" \
    >"$D/logs/$tag.log" 2>&1
  local rc=$?
  echo "exit=$rc $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  rm -rf "$dir/model"          # keep summary.json + eval metrics, drop the 130MB model dir
  return $rc
}

i=0
while IFS=$'\t' read -r tag case work init budget arm args; do
  [ -z "${tag:-}" ] && continue
  i=$((i + 1))
  [ $(((i - 1) % NLANES)) -eq "$LANE" ] || continue
  one "$tag" "$case" "$work" "$init" "$budget" "$arm" "$args" || true
done <"$CELLS"
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
