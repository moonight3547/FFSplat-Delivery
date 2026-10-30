#!/usr/bin/env bash
# Task-19: BigD2500 x M1 x A-side scene tensor cache, serial quiet box on N1.
# B = scene_load + init + snapshot train_seconds + 0.5 (PLY export/eval excluded).
set -u
REPO=/root/projects/FFSplat-Hackathon
OUT=/root/gpufree-data/outputs/bigd-combo-20261030
PY=/opt/conda/envs/fastgs/bin/python
cd "$REPO"

# 0) verify/reuse the A-side scene bundles (hash-versioned; A-side build only)
$PY -m research.scene_cache verify --work outputs/prep/mipnerf360/bicycle/work \
    --cache /root/gpufree-data/outputs/b-eng-20261030/scene_cache/bicycle
$PY -m research.scene_cache verify \
    --work outputs/curve/dl3dv-0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce/work \
    --cache /root/gpufree-data/outputs/b-eng-20261030/scene_cache/0b5635

# 1) the 8-cell serial matrix (resume-safe: summary.json is authoritative)
$PY /tmp/stagedloss/bigd_combo.py        # -> $OUT/matrix/<scene>/<arm>/sweep

# 2) reduce to rows with deltas vs bigd and vs delivery
$PY /tmp/stagedloss/bigd_analyze.py      # -> $OUT/results.json

# Per-arm core command (arm flags appended):
#  delivery:   --densify-until 1250                       (default native threshold 0.0012)
#  bigd:       --densify-until 2500 --grad-abs-thresh 0.0008
#  bigd_m1:    bigd + --mid-prune-at 1250 --mid-prune-keep 0.6
#  *_cache:    + --scene-cache /root/gpufree-data/outputs/b-eng-20261030/scene_cache/<scene>
#  common:     --iterations 5000 --snapshots "" --position-lr-max-steps 5000
#              --densify-from 500 --densification-interval 100 --opacity-reset-interval 0
#              --lambda-dssim 0.6 --scaling-lr 0.01 --realloc-frac 0.15 --realloc-at 1250
#              --image-cache <work>/image_cache --no-quick-test --seed 20261015
#              --profile competition/profiles/gpufree-2.json
