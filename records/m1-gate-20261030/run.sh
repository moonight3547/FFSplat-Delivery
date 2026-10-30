#!/usr/bin/env bash
# Task-13: M1 gate calibration + A/B.  N1, serial quiet box (anything timed).
set -u
REPO=/root/projects/FFSplat-Hackathon
OUT=/root/gpufree-data/outputs/m1-gate-20261030
PY=/opt/conda/envs/fastgs/bin/python
cd "$REPO"

# ---- 1. calibration: 8 fixed TRAIN views from the init PLY, mean VGG-LPIPS.
#      ~8-16 s/scene; run once per case (serial, GPU2). The 13-case table is in
#      records/m1-gate-20261030/train_lpips_table.json.
#      python -m research.prune_gate --work <work> --ply <resplat_world.ply> \
#          --tau 0.48 --keep 0.6 --record "$OUT/gate/<tag>.json"

# ---- 2. A/B end-to-end: gate->decision->module B (delivery recipe) ----------
cat > /tmp/m1gate_ab_cells.txt <<CELLS
mipnerf360/bicycle ctrl 2
mipnerf360/bicycle m1_1250_60 2
mipnerf360/counter ctrl 2
mipnerf360/counter m1_1250_60 2
CELLS
$PY records/staged-loss-20261030/run_staged.py --mode serial --allow-busy \
  --out-root "$OUT/ab" --cells-file /tmp/m1gate_ab_cells.txt \
  --stage2-iters 400 --standard-iters 1000 --mcmc-budget 0 \
  --stage3-iters 100 --stage3-weight 0.1

# ---- 3. deploy (one scene): gate runs pre-B, keep=1.0 => delivery unchanged --
# python run_pipeline.py --scene <scene dir> --output <out> --mid-prune auto
