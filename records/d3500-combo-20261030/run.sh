#!/usr/bin/env bash
# Task-20: D3500 (500->3500, 29 native events) x M1 x A-side scene cache, N1 GPU2 quiet box.
# B = scene_load + init + snapshot train_seconds + 0.5 (PLY export/eval outside B).
set -u
REPO=/root/projects/FFSplat-Hackathon
OUT=/root/gpufree-data/outputs/d3500-combo-20261030
PY=/opt/conda/envs/fastgs/bin/python
cd "$REPO"

# 1) six-cell serial matrix (resume-safe; scene bundles reused by version hash)
$PY /tmp/stagedloss/d3500_combo.py       # -> $OUT/matrix/<scene>/<arm>/sweep
# 2) reduce to rows: deltas vs d3500 and vs task-19 BigD2500 best
$PY /tmp/stagedloss/d3500_analyze.py     # -> $OUT/results.json

# Per-arm recipe flags:
#  d3500        : --iterations 5000 --densify-from 500 --densify-until 3500
#                 --densification-interval 100 --grad-abs-thresh 0.0008
#                 --realloc-frac 0.15 --realloc-at 1250 --lambda-dssim 0.6
#                 --scaling-lr 0.01 --opacity-reset-interval 0
#                 --position-lr-max-steps 5000 --image-cache <work>/image_cache
#                 --no-quick-test --seed 20261015
#  d3500_cache  : + --scene-cache /root/gpufree-data/outputs/b-eng-20261030/scene_cache/<scene>
#  d3500_m1     : + --mid-prune-at 1250 --mid-prune-keep 0.6
#  d3500_m1_cache: both (bicycle only)
#  scenes: bicycle (4 arms) and dl3dv/0b5635 (d3500, d3500_cache)
