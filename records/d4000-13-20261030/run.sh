#!/usr/bin/env bash
# Task-22: N1 full-13, 5K-D4000 + VGG200 + scene cache + deterministic M1 gate.
# B = scene_load + init + snapshot train_seconds + 0.5 (PLY export/eval excluded).
set -u
REPO=/root/projects/FFSplat-Hackathon
OUT=/root/gpufree-data/outputs/d4000-13-20261030
PY=/opt/conda/envs/fastgs/bin/python
cd "$REPO"

# 1) 13-case serial run: per case gate (research.prune_gate tau .48 keep .6) then run_sweep
$PY /tmp/stagedloss/d4000_13.py            # -> $OUT/matrix/<tag>/sweep + gate.json
$PY /tmp/stagedloss/d4000_agg.py           # -> $OUT/results.jsonl
# 2) same-node 0b5635 repeat (variance estimate)
$PY /tmp/stagedloss/d4000_repeat.py        # -> $OUT/repeat/0b5635
# 3) official Faster-GS CUDA Adam pilot (opt-in env, default OFF in delivery)
#    build once: cd /tmp/fastergs_export/FasterGSCudaBackend && python setup.py build_ext --inplace
$PY /tmp/stagedloss/fsadm_smoke.py         # numeric/state identity smoke
$PY /tmp/stagedloss/d4000_official_adam_cell.py   # bicycle compatibility cell
$PY /tmp/stagedloss/d4000_official_adam_more.py   # 0b5635 + 3e4dbe compatibility cells
# common per-case recipe flags (all 13):
#   --iterations 5000 --snapshots "" --position-lr-max-steps 5000 --densify-from 500
#   --densify-interval 100 --densify-until 4000 --grad-abs-thresh 0.0008
#   --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01
#   --realloc-frac 0.15 --realloc-at 1250 --no-quick-test --seed 20261015
#   --lpips-weight 0.1 --lpips-scale 1.0 --lpips-start 4801 --lpips-end 5000
#   --mid-prune-at 1250 --mid-prune-keep <gate keep> --scene-cache <bundle>
#   profile competition/profiles/gpufree-2.json (N1 GPU2)
