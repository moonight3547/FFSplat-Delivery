#!/usr/bin/env bash
# Task-23: framework optimization on N1. All paths default off; delivery untouched.
set -u
REPO=/root/projects/FFSplat-Hackathon
OUT=/root/gpufree-data/outputs/framework-opt-20261030
PY=/opt/conda/envs/fastgs/bin/python
cd "$REPO"

# 1) GT VGG-LPIPS feature cache: identity + A/B (D3500 + VGG200 + cache + M1)
$PY /tmp/stagedloss/gtcache_identity.py     # -> $OUT/gtcache_identity.json
$PY /tmp/stagedloss/gtcache_ab.py           # -> $OUT/gtcache_ab/{bicycle,0b5635}/{ctrl,gtcache}
# 2) official Faster-GS CUDA Adam under the D3500 recipe (opt-in env; build once in task-22)
$PY /tmp/stagedloss/gtcache_adam_d3500.py   # -> $OUT/adam_d3500/{bicycle,0b5635}
# identity/smoke: $PY /tmp/stagedloss/fsadm_smoke.py (see records/d4000-13-20261030/)
# 3) per-cell recipe flags (D3500 variant):
#   --iterations 5000 --position-lr-max-steps 5000 --densify-from 500 --densify-interval 100
#   --densify-until 3500 --grad-abs-thresh 0.0008 --opacity-reset-interval 0 --lambda-dssim .6
#   --scaling-lr .01 --realloc-frac .15 --realloc-at 1250 --no-quick-test --seed 20261015
#   --lpips-weight .1 --lpips-scale 1.0 --lpips-start 4801 --lpips-end 5000
#   --mid-prune-at 1250 --mid-prune-keep <gate keep> --scene-cache <bundle>
#   [--gt-feature-cache --gt-feature-cache-views 40]  [--lpips-timing]
