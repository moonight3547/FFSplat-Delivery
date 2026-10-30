#!/usr/bin/env bash
# Staged post-5000 post-pass matrix (task-8).  Run on N1 from the repo root:
#   bash records/staged-loss-20261030/run.sh
# All outputs go under /root/gpufree-data (N1 root disk is ~99% full).
#
# Sequencing is part of the method: the selfcheck runs ALONE first, then the
# timed matrix.  On this host a competing sweep inflates module-B by ~2.5x
# (measured: ctrl B 27.3 s -> 90.4 s when three training procs + another job
# shared the box), which makes the TIME term meaningless.  The 2 lanes below
# were verified contention-free (ctrl B 27.14 s vs 27.29 s single-card anchor).
set -u
REPO=/root/projects/FFSplat-Hackathon
ROOT=/root/gpufree-data/outputs/stagedloss-20261030
PY=/opt/conda/envs/fastgs/bin/python
cd "$REPO"

# ---- 0. GATE: stages off must reproduce the delivery cell.  N1 serial anchor:
#      q 10.05 / B 27.29 / total 25.89; H1 clean13: total 25.31 / q 9.76 /
#      B 28.5.  Measured here: q 9.890 / B 27.28 / total 25.73 (delta reported
#      in STAGED-LOSS-20261030.md).
# NOTE: never delete an out-root here.  The driver already RESUME-SKIPs any cell
# whose <out>/<scene>/<arm>/sweep/summary.json exists, and a failed cell records
# status=fail and moves on; there is no launcher retry loop.  Deleting the matrix
# dir is what silently destroyed completed cells on 2026-10-31 (see report §7).
printf 'mipnerf360/bicycle ctrl 0\n' > /tmp/selfcheck_cells.txt
$PY records/staged-loss-20261030/run_staged.py --mode serial \
  --out-root "$ROOT/selfcheck" --cells-file /tmp/selfcheck_cells.txt --report

# ---- 1. compact arm matrix (13 cells, 2 lanes).  Arms:
#        vgg            = delivery + 100 VGG-LPIPS steps (stage2=0)   [headline]
#        absgrad        = +500 net-zero AbsGrad relocation steps
#        absgrad_vgg    = absgrad then +100 VGG-LPIPS steps
#        mcmc           = warm-start research.mcmc_budget from the phase-1 PLY
#        mcmc_vgg       = mcmc then +100 VGG-LPIPS steps
#        standard       = plain 1-cam L1+DSSIM continuation (ablation)
#      Phase-3 LPIPS uses the existing --lpips_every 1 / --lpips_scale 0.5 path
#      with weight 0.1.
$PY records/staged-loss-20261030/run_staged.py --mode parallel --lanes 2 \
  --out-root "$ROOT/matrix" \
  --cells-file records/staged-loss-20261030/matrix_cells.txt \
  --stage2-iters 500 \
  --standard-iters 1000 \
  --mcmc-budget 0 \
  --stage3-iters 100 --stage3-weight "${STAGE3_WEIGHT:-0.1}" \
  --report

# ---- 2. step-count sweeps (user "500 步是不是太少"): VGG steps first, then
#      AbsGrad.  Same out-root as the first sweep, so RESUME-SKIP reuses any
#      completed cell (vgg_200/vgg_400 were measured by the launched sweep2).
$PY records/staged-loss-20261030/run_staged.py --mode serial \
  --out-root "$ROOT/sweep2" \
  --cells-file records/staged-loss-20261030/vgg_sweep_cells.txt \
  --stage2-iters 500 --standard-iters 1000 --mcmc-budget 0 \
  --stage3-iters 100 --stage3-weight 0.1 --report

# user grid (vgg_500/vgg_1000, absgrad_1000/2000/3000/5000; 3000+5000 skipped
# by Lead decision once absgrad proved monotonically diverging)
$PY records/staged-loss-20261030/run_staged.py --mode serial \
  --out-root "$ROOT/sweep2" \
  --cells-file records/staged-loss-20261030/sweep3_cells.txt \
  --stage2-iters 500 --standard-iters 1000 --mcmc-budget 0 \
  --stage3-iters 100 --stage3-weight 0.1 --report
# ---- 2. optional single-card repeats if any delta lands inside run-to-run
#      noise (bicycle ctrl q is reproducible to ~0.1; B to ~0.05 s):
# WINNER=vgg
# printf 'mipnerf360/bicycle ctrl 0\nmipnerf360/bicycle %s 0\n' "$WINNER" \
#   > /tmp/serial_cells.txt
# $PY records/staged-loss-20261030/run_staged.py --mode serial \
#   --out-root "$ROOT/serial" --cells-file /tmp/serial_cells.txt \
#   --stage2-iters 500 --standard-iters 1000 --mcmc-budget 0 \
#   --stage3-iters 100 --stage3-weight 0.1 --report
