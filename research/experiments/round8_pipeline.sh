#!/usr/bin/env bash
# Round 8 - end-to-end validation of the single-scene competition entry
# (run_pipeline.py), on a scene the research sweeps did not use for tuning.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
OUT=${OUT:-outputs/pipeline}
mkdir -p "$OUT"
log="$OUT/run.log"
echo "start $(date -Is)" | tee -a "$log"

while pgrep -f "round[0-9]_|round5_mcmc" >/dev/null 2>&1; do sleep 15; done

DATA=/workspace/organizer-assets/datasets
run() {  # tag, scene, group, extra...
  local tag="$1" scene="$2" group="$3"; shift 3
  if [ -e "$OUT/$tag" ]; then echo "skip existing $OUT/$tag" >>"$log"; return; fi
  echo "=== pipeline $tag $(date -Is) :: $*" >>"$log"
  $PY run --no-capture-output -n ffsplat-enhance python run_pipeline.py \
    --scene "$scene" --group "$group" --output "$OUT/$tag" "$@" >>"$log" 2>&1
  echo "pipeline_exit=$?" >>"$log"
}

# counter: reuse the measured budget (the expensive 30K re-measurement is the
# other code path and is exercised by the next run)
run counter "$DATA/mipnerf360/counter" mipnerf360 --budget-seconds 92.57
# hw: full self-calibrating path, i.e. exactly what the grader would run
run hw "$DATA/hw/billboard" hw
echo "done $(date -Is)" >>"$log"
