#!/usr/bin/env bash
# Resume protocol for task-9 (bicycle-only grow-only densification, SERIAL only).
#
#   bash resume.sh diag   [LANE]   # alpha diagnostic + 3D hole cloud (~60 s)
#   bash resume.sh serial [LANE]   # the 4 serial cells             (~5-6 min)
#   bash resume.sh all    [LANE]   # diag then serial
#
# 2026-10-31 02:15: **# GPU2 / shared box** -- staged-loss runs its step-sweep on GPU0
# (serial), so this experiment is pinned to GPU2 and GPU1 is kept empty as a buffer.
# Because the box is shared, B/TIME here is CONTENDED: the verdict is quality-first and
# a clean Δtotal comes from a separate quiet-box re-anchor.
#
# Modes pass a lane-scoped idle gate: the target GPU (default 2) must be under 1500 MiB
# with no compute apps, and no diagnostic/training job of ours may already be running.
# (A whole-node check would refuse forever while staged-loss owns GPU0.)
set -u
MODE=${1:-all}
LANE=${2:-2}                     # GPU2 by default: task-8 owns GPU0, GPU1 is the buffer
D=/root/gpufree-data/outputs/bicycle-grow-20261030
R=/root/projects/FFSplat-Hackathon
PY=/opt/conda/envs/fastgs/bin/python
mkdir -p "$D/logs" "$D/cells" "$D/ply"
cd "$R" || exit 1
export PYTHONPATH=$R

idle_gate() {
  local fail=0 procs apps mem
  # 1) no diagnostic/training job OF OURS already running (avoid duplicate lanes)
  procs=$(pgrep -af "init_error_diag" | grep -v "resume.sh" | grep -v pgrep | head -3)
  if [ -n "$procs" ]; then
    echo "REFUSE: our diagnostic already running:" >&2; echo "$procs" >&2; fail=1
  fi
  # 2) the TARGET GPU only (GPU0 belongs to task-8 on this shared box)
  mem=$(nvidia-smi --id="$LANE" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "${mem:-0}" -ge 1500 ]; then
    echo "REFUSE: GPU${LANE} has ${mem} MiB allocated" >&2; fail=1
  fi
  apps=$(nvidia-smi --id="$LANE" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -3)
  if [ -n "$apps" ]; then
    echo "REFUSE: GPU${LANE} compute apps alive:" >&2; echo "$apps" >&2; fail=1
  fi
  if [ "$fail" -eq 0 ]; then
    echo "IDLE-GATE PASS lane=${LANE} (# GPU2 / shared box) $(date -Is)" | tee -a "$D/queue.log"
  fi
  return "$fail"
}

diag() {
  idle_gate || exit 3
  CUDA_VISIBLE_DEVICES="$LANE" $PY -m research.init_error_diag \
    --source outputs/prep/mipnerf360/bicycle/work \
    --ply outputs/prep/mipnerf360/bicycle/resplat/resplat_world.ply \
    --views 6 --out "$D/init_error_diag_alpha.json" \
    --out-holes "$D/holes_bicycle.npy" > "$D/logs/diag_alpha.log" 2>&1
  local rc=$?
  echo "diag rc=$rc $(date -Is)" | tee -a "$D/queue.log"
  tail -30 "$D/logs/diag_alpha.log"
  return $rc
}

serial() {
  idle_gate || exit 3
  [ -s "$D/holes_bicycle.npy" ] || { echo "missing $D/holes_bicycle.npy (run: bash resume.sh diag)" >&2; exit 4; }
  GPU_LANE="$LANE" D="$D" bash "$D/run.sh" "$D/serial.tsv" 0 1
}

case "$MODE" in
  diag) diag ;;
  serial) serial ;;
  all) diag && serial ;;
  *) echo "usage: bash resume.sh {diag|serial|all} [lane]" >&2; exit 2 ;;
esac
echo "mode=$MODE done $(date -Is)" | tee -a "$D/queue.log"
