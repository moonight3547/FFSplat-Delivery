#!/usr/bin/env bash
# FINAL-13 validation: 5K-D3500 + VGG200 + scene cache + deterministic M1 gate.
# PREPARED ONLY — launch on the Lead's go, on a quiet H1, serial.
set -u
cd /workspace/organizer-projects/FFSplat-Hackathon || exit 1
export PYTHONPATH=$PWD
git log --oneline -1
python3 scripts/run_exact13_gate.py \
  --out-root outputs/final13-d3500-v200-cache-20261030 \
  --recipe bigd3500 --vgg-tail 200 --scene-cache \
  --tau 0.48 --keep 0.6
