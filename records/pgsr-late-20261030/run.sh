#!/usr/bin/env bash
# Task-15: PGSR late-window (4000-5000) low-frequency depth regularization.
# Serial quiet box on N1 (B/TIME is the measurement).
set -u
REPO=/root/projects/FFSplat-Hackathon
OUT=/root/gpufree-data/outputs/pgsr-late-20261030
PY=/opt/conda/envs/fastgs/bin/python
cd "$REPO"
mkdir -p "$OUT"

# arms: ctrl | pgsr_sv10 (1 extra gsplat RGB+ED pass every 10 steps in 4000-5000,
#       single-view normal-depth consistency, weight 0.2) | pgsr_sv5
cat > /tmp/pgsr_late_cells.txt <<CELLS
dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce ctrl 2
dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce pgsr_sv10 2
dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce pgsr_sv5 2
dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e ctrl 2
dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e pgsr_sv10 2
dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e pgsr_sv5 2
CELLS
$PY records/staged-loss-20261030/run_staged.py --mode serial --allow-busy \
  --out-root "$OUT/matrix" --cells-file /tmp/pgsr_late_cells.txt \
  --stage2-iters 400 --standard-iters 1000 --mcmc-budget 0 \
  --stage3-iters 100 --stage3-weight 0.1
