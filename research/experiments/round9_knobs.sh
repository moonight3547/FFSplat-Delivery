#!/usr/bin/env bash
# Round 9 - module-B knob sweep on the second 4090 host (2 GPUs in parallel).
# Each variant changes one knob from the winning "no opacity reset" recipe and
# runs the same 8000 iterations on hw/billboard, so the rows are comparable.
#
#   bash research/experiments/round9_knobs.sh            # both lanes in parallel
#   LANE=0 bash research/experiments/round9_knobs.sh     # one lane only
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}
OUT=${OUT:-outputs/experiments/round9}
mkdir -p "$OUT"

var() {  # gpu, tag, extra args...
  local gpu="$1" tag="$2"; shift 2
  local dir="$OUT/$tag" log="$OUT/$tag.log"
  if [ -e "$dir" ]; then echo "skip $tag" >>"$OUT/queue.log"; return; fi
  echo "start $tag gpu$gpu $(date -Is)" >>"$OUT/queue.log"
  $PY run --no-capture-output -n fastgs python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$gpu.json" \
    --case hw/billboard --work outputs/O0-20261030/work/hw/billboard \
    --init-ply outputs/O0-20261030/results/hw/billboard/resplat/resplat_world.ply \
    --out "$dir" --iterations 8000 --snapshots 4000,8000 --budget 99.49 \
    --opacity-reset-interval 0 "$@" >"$log" 2>&1
  echo "exit=$? $tag" >>"$OUT/queue.log"
}

lane() {
  case "$1" in
    0)
      var 0 kw_dssim04 "--lambda-dssim 0.4"
      var 0 kw_mult10 "--mult 1.0"
      var 0 kw_dens_early "--densify-from 100 --densify-interval 50"
      var 0 kw_lr8k "--position-lr-max-steps 8000"
      ;;
    1)
      var 1 kw_lr_init2x "--position-lr-init 0.00032"
      var 1 kw_scale_lr "--scaling-lr 0.01"
      var 1 kw_dssim01 "--lambda-dssim 0.1"
      var 1 kw_lr8k_init2x "--position-lr-max-steps 8000 --position-lr-init 0.00032"
      ;;
  esac
}

if [ "${LANE:-all}" = all ]; then
  lane 0 & p0=$!
  lane 1 & p1=$!
  wait $p0 $p1
  echo "all lanes done $(date -Is)" >>"$OUT/queue.log"
else
  lane "$LANE"
fi
