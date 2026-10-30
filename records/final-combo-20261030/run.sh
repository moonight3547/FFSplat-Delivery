#!/usr/bin/env bash
# Task-21: final combination VGG100/200 x M1 x scene cache x D2500/D3500. N1 GPU2 quiet box.
# B = scene_load + init + snapshot train_seconds + 0.5 (PLY export/eval outside B).
set -u
REPO=/root/projects/FFSplat-Hackathon
OUT=/root/gpufree-data/outputs/final-combo-20261030
PY=/opt/conda/envs/fastgs/bin/python
cd "$REPO"

# 1) eight-cell serial matrix (resume-safe; scene bundles reused by version hash)
$PY /tmp/stagedloss/final_combo.py       # -> $OUT/matrix/<scene>/<arm>/sweep
# 2) reduce to rows: ranking + LPIPS points + delta vs each arm's no-VGG reference
$PY /tmp/stagedloss/final_analyze.py     # -> $OUT/results.json

# Arm grid (all with S=5000, LR horizon 5000, densify_from 500, interval 100,
# native grad_abs_thresh 0.0008, realloc .15@1250, image cache, seed 20261015):
#  bicycle: d2500|d3500 x v100|v200  + M1 keep .6@1250 + scene cache
#  0b5635 : d2500|d3500 x v100|v200  + scene cache (no M1)
# VGG tail: full-res (--lpips-scale 1.0) weight 0.1 inside the same 5000 steps:
#  v100 = --lpips-start 4901 --lpips-end 5000
#  v200 = --lpips-start 4801 --lpips-end 5000
