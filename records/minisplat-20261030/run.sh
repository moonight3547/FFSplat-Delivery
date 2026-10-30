#!/usr/bin/env bash
# Mini-Splatting density-control experiment (task-10).  Run on N1 from repo root.
# Everything writes under /root/gpufree-data (N1 root disk ~99% full).
#
#   bash records/minisplat-20261030/run.sh
#
# SERIAL, one GPU, quiet box: B time is the measured mechanism, so a competing
# sweep voids the experiment.  Lane 2 = GPU2 (lane hints in cells.txt).  The
# driver resume-skips any cell whose <out>/<scene>/<arm>/sweep/summary.json exists;
# there is no rm -rf of an out-root anywhere in this script.
set -u
REPO=/root/projects/FFSplat-Hackathon
ROOT=/root/gpufree-data/outputs/minisplat-20261030
PY=/opt/conda/envs/fastgs/bin/python
cd "$REPO"

# ---- M2 (post-5000 importance prune + 400-step refine) + M1 (mid-training prune)
#      + the paired control cells.  Variants and prune parameters live in the
#      MINISPLAT report §2; arms are registered in
#      records/staged-loss-20261030/run_staged.py (m2_50/70/90, m1_1250_70).
$PY records/staged-loss-20261030/run_staged.py --mode serial --allow-busy \
  --out-root "$ROOT/matrix" \
  --cells-file records/minisplat-20261030/cells.txt \
  --stage2-iters 400 --standard-iters 1000 --mcmc-budget 0 \
  --stage3-iters 100 --stage3-weight 0.1 \
  --report

# ---- standalone scorer/pruner (same code the trainer uses), for diagnostics:
# $PY -m research.ply_importance --work outputs/prep/mipnerf360/bicycle/work \
#     --ply <delivery ply> --out-ply /tmp/bicycle_keep70.ply --keep 0.7 \
#     --views 24 --record /tmp/bicycle_prune.json
