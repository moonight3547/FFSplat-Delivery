#!/usr/bin/env bash
# VGG100 vs VGG200 on the 5K-D3500 base recipe — H1, task-17.
# Serial, quiet-box gated; one shared ctrl per case; uniform arms for all cases.
set -u
cd /workspace/organizer-projects/FFSplat-Hackathon || exit 1
export PYTHONPATH=$PWD
mkdir -p outputs/vggab-d3500-20261030/logs
python3 outputs/vggab-d3500-20261030/run_vggab.py \
  --out-root outputs/vggab-d3500-20261030 \
  --recipe outputs/vggab-d3500-20261030/recipe.json \
  --arms ctrl,vgg100,vgg200
# density audit (asserts 29 native events @ grad_abs_thresh 0.0008 for every cell)
python3 outputs/vggab-d3500-20261030/audit_density.py
