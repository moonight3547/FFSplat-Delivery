#!/usr/bin/env bash
# Round 5 - standalone wall-clock-budgeted gsplat MCMC module B.
#
# Trains 30 s and 60 s module-B variants of research/mcmc_budget.py on the two
# short-continuation focus cases, then scores every exported PLY with exactly
# the official held-out renderer/metrics through research/score_ply.py.
#
# The trainer budget is module-B wall-clock: its own allocation is
# budget * FRACTION (default 0.9), including setup and PLY export.
# The score_ply --budget value is the per-case FastGS random-init 30K
# reference, used only for the proxy TIME points.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
ENV=${ENV:-ffsplat-enhance}
PYTHON=${PYTHON:-python3}
OUT=${OUT:-outputs/experiments/round5}
FRACTION=${FRACTION:-0.9}
SEED=${SEED:-20261015}
DEVICE=${DEVICE:-cuda:0}
CAP_MULT=${CAP_MULT:-2.0}
REFINE_EVERY=${REFINE_EVERY:-100}
SH_DEGREE=${SH_DEGREE:-3}
RES_SCALES=${RES_SCALES:-0.5,0.75,1.0}
RES_UNTILS=${RES_UNTILS:-0.4,0.7}
# FastGS random-init 30K reference seconds for the proxy TIME points.
REF_HW=${REF_HW:-99.49}
REF_COUNTER=${REF_COUNTER:-92.57}
mkdir -p "$OUT"
log="$OUT/run.log"
echo "start $(date -Is)" | tee -a "$log"

run_one() {  # case, reference_budget, trainer_budget
  local case="$1" ref_budget="$2" budget="$3" tag dir steps total_s train_exit score_exit
  tag=$(echo "$case" | tr '/' '-')
  dir="$OUT/$tag-b$budget"
  echo "=== $case B=${budget}s (allocation $FRACTION) $(date -Is)" >>"$log"

  if [ -f "$dir/module_b.json" ]; then
    echo "reuse existing $dir/module_b.json" >>"$log"
  elif [ -e "$dir" ]; then
    echo "skip non-empty partial trainer dir $dir" >>"$log"
    return
  else
    $PY run --no-capture-output -n "$ENV" python -m research.mcmc_budget \
      --work "outputs/O0-20261030/work/$case" \
      --init "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" \
      --out "$dir" --budget-seconds "$budget" --budget-fraction "$FRACTION" \
      --res-scales "$RES_SCALES" --res-untils "$RES_UNTILS" \
      --cap-mult "$CAP_MULT" --refine-every "$REFINE_EVERY" --sh-degree "$SH_DEGREE" \
      --seed "$SEED" --device "$DEVICE" >>"$log" 2>&1
    train_exit=$?
    echo "train_exit=$train_exit" >>"$log"
    if [ "$train_exit" -ne 0 ] || [ ! -f "$dir/module_b.json" ]; then
      echo "trainer failed for $tag-b$budget" | tee -a "$log"
      return
    fi
  fi

  read -r steps total_s allocation < <("$PYTHON" - "$dir/module_b.json" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1]))
print(record["steps"], record["total_seconds"], record["allocation_seconds"])
PY
)
  if [ -z "${steps:-}" ] || [ -z "${total_s:-}" ] || [ -z "${allocation:-}" ]; then
    echo "could not read $dir/module_b.json" | tee -a "$log"
    return
  fi
  echo "  B measured ${total_s}s, ${steps} steps, allocation ${allocation}s" >>"$log"

  if [ -f "$dir/score/score_summary.json" ]; then
    echo "reuse existing $dir/score/score_summary.json" >>"$log"
  else
    $PY run --no-capture-output -n "$ENV" python -m research.score_ply \
      --work "outputs/O0-20261030/work/$case" --out "$dir/score" \
      --ply "$dir/point_cloud/iteration_$steps/point_cloud.ply" --name "b$budget" \
      --budget "$ref_budget" --b-seconds "$total_s" >>"$log" 2>&1
    score_exit=$?
    echo "score_exit=$score_exit" >>"$log"
    if [ "$score_exit" -ne 0 ]; then
      echo "scoring failed for $tag-b$budget" | tee -a "$log"
      return
    fi
  fi
  "$PYTHON" - "$dir/score/score_summary.json" "$dir/module_b.json" <<'PY' | tee -a "$log"
import json
import os
import sys
d = json.load(open(sys.argv[1]))
m = json.load(open(sys.argv[2]))
r = d["rows"][0]
s = r["score"]
print("  %-28s B %6.2fs steps %6d N %7d | PSNR %7.3f SSIM %.4f LPIPS %.4f | q %5.2f t %5.2f = %5.2f"
      % (os.path.basename(os.path.dirname(os.path.dirname(sys.argv[1]))), m["total_seconds"],
         m["steps"], m["num_final_gaussians"], r["psnr"], r["ssim"], r["lpips"],
         s["quality_points"], s["time_points"], s["total_points"]))
PY
}

for case in "hw/billboard" "mipnerf360/counter"; do
  if [ "$case" = "hw/billboard" ]; then ref=$REF_HW; else ref=$REF_COUNTER; fi
  for budget in 30 60; do
    run_one "$case" "$ref" "$budget"
  done
done

echo "--- round5 summary ---" | tee -a "$log"
"$PYTHON" - "$OUT" <<'PY' | tee -a "$log"
import glob
import json
import os
import sys
rows = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*-b*", "score", "score_summary.json"))):
    d = json.load(open(path))
    model_path = os.path.join(os.path.dirname(os.path.dirname(path)), "module_b.json")
    m = json.load(open(model_path))
    r = d["rows"][0]
    s = r["score"]
    rows.append((os.path.basename(os.path.dirname(os.path.dirname(path))), m["total_seconds"],
                 m["steps"], m["num_final_gaussians"], r["psnr"], r["ssim"], r["lpips"],
                 s["quality_points"], s["time_points"], s["total_points"]))
if not rows:
    print("no scored outputs yet")
for row in rows:
    print("%-28s B %6.2fs steps %6d N %7d | PSNR %7.3f SSIM %.4f LPIPS %.4f | q %5.2f t %5.2f = %5.2f"
          % row)
PY
echo "done $(date -Is)" >>"$log"
