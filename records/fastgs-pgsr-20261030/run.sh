#!/usr/bin/env bash
# FastGS+PGSR hybrid (task-12): staged gate -> A/B -> depth-cost probe.
# Run on N1 from the repo root; serial, quiet box (B is the measurement).
set -u
REPO=/root/projects/FFSplat-Hackathon
ROOT=/root/gpufree-data/outputs/fastgs-pgsr-20261030
PY=/opt/conda/envs/fastgs/bin/python
cd "$REPO"
mkdir -p "$ROOT"

# ---- 0. gate: do the priors change the Gaussian shape statistics? (1500-step
#      bicycle runs, identical schedule except the weight; shape_stats in the
#      JSON records).  Re-run with the loop in the report §2 / /tmp/pgsr_gate.sh.
#      A positive gate must show anisotropy / normal-cos moving; ours showed
#      normal-cos is identically 0 (ReSplat PLY normals are all zero) and the
#      init is already planar, so only the quality A/B remained.

# ---- 1. A/B on the delivery recipe (serial, one GPU, quiet box)
$PY records/staged-loss-20261030/run_staged.py --mode serial --allow-busy \
  --out-root "$ROOT/matrix" --cells-file /tmp/pgsr_cells.txt \
  --stage2-iters 400 --standard-iters 1000 --mcmc-budget 0 \
  --stage3-iters 100 --stage3-weight 0.1

# ---- 2. gsplat depth-cost probe (RGB vs RGB+ED from the same Gaussian tensors)
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n fastgs python -m research.depth_cost_probe \
  --work outputs/prep/mipnerf360/bicycle/work \
  --ply "$ROOT/../stagedloss-20261030/selfcheck/mipnerf360-bicycle/ctrl/sweep/model/point_cloud/iteration_5000/point_cloud.ply" \
  --views 3 --reps 15 --record "$ROOT/depth_probe.json"
