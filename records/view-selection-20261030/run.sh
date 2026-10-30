#!/usr/bin/env bash
# Co-visibility 16-view selection experiment (task-11, H1, single GPU, serial).
# All commands run on H1 from /workspace/organizer-projects/FFSplat-Hackathon.
#   $C run -n organizer-fastgs  : CPU-only steps (selection, image_cache) + module B/metrics
#   $C run -n ffsplat-enhance   : ReSplat
# A/ReSplat are NOT in the scored TIME; module B is unchanged (delivery recipe).
set -eu
C=/usr/local/miniconda3/bin/conda
REPO=/workspace/organizer-projects/FFSplat-Hackathon
CKPT=/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth

# 0) (CPU) co-visibility context selection: PLY-point visibility -> Jaccard kernel ->
#    16-view subsets (uniform control / covis-maxmin / covis-cover / covis-fps).
cd "$REPO"
for pair in \
  "mipnerf360-bicycle|outputs/cand/mipnerf360-bicycle/work|outputs/clean13-20261030/resplat/mipnerf360-bicycle/resplat_world.ply" \
  "mipnerf360-counter|outputs/cand/mipnerf360-counter/work|outputs/clean13-20261030/resplat/mipnerf360-counter/resplat_world.ply" \
  "dl3dv-001dccbc|outputs/multi/dl3dv-001dccbc/work|outputs/clean13-20261030/resplat/dl3dv-001dccbc/resplat_world.ply"; do
  IFS='|' read -r scene work ply <<< "$pair"
  $C run --no-capture-output -n organizer-fastgs python /root/select_context.py \
    --work "$work" --ply "$ply" \
    --out outputs/viewsel-20261030/selection/$scene.json --scene "$scene"
done

# 1) (CPU) build candidate work dirs: copy split.json, rewrite ONLY split["context"]
#    (16 train-only names), symlink images/sparse/module_a/image_cache.
python3 /root/viewsel_prepare.py

# 2) Serial, quiet-gated GPU stage: ReSplat for 12 candidates -> context self-render
#    PSNR triage -> delivery e2e for the best candidate per scene if it beats uniform
#    by >= +0.3 dB. Gating: gpu<50MiB, loadavg1<4, no research/colmap processes.
python3 /root/viewsel_run.py
