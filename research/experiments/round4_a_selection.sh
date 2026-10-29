#!/usr/bin/env bash
# Round 4 - module A lever: compare 16-view context policies.
# The selector is CPU-only and writes <case>/<policy>/context16.json.  The
# prepared scene dir is never modified: for ReSplat, stage a copy and replace
# only split.json["context"].  The GPU stage is off by default and is documented
# at the bottom; set RUN_GPU=1 to execute it on the GPU host.
#
#   bash research/experiments/round4_a_selection.sh hw/billboard
#   RUN_GPU=1 bash research/experiments/round4_a_selection.sh hw/billboard
#
# The input work dir is the existing module A output, e.g.
#   outputs/O0-20261030/work/hw/billboard
# If it is missing, run module A first (GPU host, resplat env):
#   $PY run -n resplat python competition/upstream/baseline/module_a/preprocess.py \
#     --scene_dir <dataset-scene> --group <group> --out "$WORK" \
#     --num_context 16 --llffhold 8 --max_resolution 960
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
ENV=${ENV:-ffsplat-enhance}
CASE=${1:-${CASE:-hw/billboard}}
WORK=${WORK:-outputs/O0-20261030/work/$CASE}
ROOT=${ROOT:-outputs/experiments/round4}
OUT=${OUT:-$ROOT/$CASE}
CKPT=${CKPT:-/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth}
RUN_GPU=${RUN_GPU:-0}
mkdir -p "$OUT"
log="$OUT/run.log"
echo "start $(date +%Y-%m-%dT%H:%M:%S%z) case=$CASE work=$WORK" | tee -a "$log"

if [ ! -f "$WORK/split.json" ]; then
  echo "missing prepared work dir: $WORK/split.json" | tee -a "$log" >&2
  exit 2
fi

# ---- module A: select 16 context views for each policy ----------------------
for policy in uniform fps coverage; do
  dir="$OUT/$policy"
  mkdir -p "$dir"
  echo "=== select $policy $(date +%Y-%m-%dT%H:%M:%S%z)" >>"$log"
  if [ -f "$dir/context16.json" ]; then
    echo "skip existing $dir/context16.json" | tee -a "$log"
    continue
  fi
  $PY run --no-capture-output -n "$ENV" python -m research.select_views \
    --work "$WORK" --out "$dir" --policy "$policy" >>"$log" 2>&1
  rc=$?
  tail -n 1 "$log"
  echo "select_exit=$rc" >>"$log"
  if [ "$rc" -ne 0 ]; then
    echo "selector failed for $policy (rc=$rc); see $log" >&2
    tail -n 20 "$log" >&2
    exit "$rc"
  fi
done

python3 - "$OUT" <<'PY' | tee -a "$log"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for policy in ("uniform", "fps", "coverage"):
    path = root / policy / "context16.json"
    if not path.exists():
        continue
    record = json.loads(path.read_text(encoding="utf-8"))
    scores = record["scores"]
    print(f"  {policy:8s} min_sep={scores['min_separation']:.4f} "
          f"mean_nn={scores['mean_nearest_separation']:.4f} "
          f"context={' '.join(record['context'])}")
PY

# ---- module A copy + ReSplat + init scoring (GPU only) ----------------------
stage_work() {  # <context16.json> <destination work dir>
  local record="$1" destination="$2"
  if [ -e "$destination" ]; then
    python3 - "$destination" "$record" <<'PY'
import json
import pathlib
import sys

destination, record_path = map(pathlib.Path, sys.argv[1:])
split = json.loads((destination / "split.json").read_text(encoding="utf-8"))
record = json.loads(record_path.read_text(encoding="utf-8"))
if split.get("context") != record.get("context"):
    raise SystemExit(1)
PY
    if [ $? -ne 0 ]; then
      echo "existing staged work does not match $record: $destination" >&2
      return 1
    fi
    echo "reuse staged work $destination" >>"$log"
    return 0
  fi
  mkdir -p "$(dirname "$destination")"
  if ! cp -a "$WORK" "$destination"; then
    echo "failed to copy $WORK -> $destination" | tee -a "$log" >&2
    return 1
  fi
  python3 - "$destination" "$record" <<'PY'
import json
import pathlib
import sys

destination, record_path = map(pathlib.Path, sys.argv[1:])
record = json.loads(record_path.read_text(encoding="utf-8"))
split = json.loads((destination / "split.json").read_text(encoding="utf-8"))
context = record.get("context")
train = split.get("train", [])
if not isinstance(context, list) or len(context) != 16 or len(set(context)) != 16:
    raise SystemExit("selection must contain 16 unique names")
if not set(context) <= set(train):
    raise SystemExit("selection contains a view outside split.json['train']")
split["context"] = list(context)
(destination / "split.json").write_text(
    json.dumps(split, indent=2) + "\n", encoding="utf-8")
(destination / "module_a_selection.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[round4] staged {destination} context={len(context)}")
PY
}

if [ "$RUN_GPU" = "1" ]; then
  if [ ! -f "$CKPT" ]; then
    echo "missing ReSplat checkpoint: $CKPT" | tee -a "$log" >&2
    exit 2
  fi
  for policy in uniform fps coverage; do
    dir="$OUT/$policy"
    record="$dir/context16.json"
    echo "=== ReSplat $policy $(date +%Y-%m-%dT%H:%M:%S%z)" >>"$log"
    [ -f "$record" ] || { echo "missing $record" >&2; exit 2; }
    stage_work "$record" "$dir/work" >>"$log" 2>&1 || exit $?

    if [ ! -f "$dir/resplat/resplat_world.ply" ]; then
      if ! $PY run --no-capture-output -n "$ENV" python -m research.resplat_variant \
          --work "$dir/work" --out "$dir/resplat" --checkpoint "$CKPT" \
          --num_refine 2 --device cuda:0 >>"$log" 2>&1; then
        echo "ReSplat failed for $policy; see $log" | tee -a "$log" >&2
        exit 1
      fi
      echo "resplat_exit=0" >>"$log"
    fi
    if [ ! -f "$dir/score-init/score_summary.json" ]; then
      if ! $PY run --no-capture-output -n "$ENV" python -m research.score_ply \
          --work "$dir/work" --out "$dir/score-init" \
          --ply "$dir/resplat/resplat_world.ply" --name "$policy" >>"$log" 2>&1; then
        echo "init scoring failed for $policy; see $log" | tee -a "$log" >&2
        exit 1
      fi
      echo "score_exit=0" >>"$log"
    fi
  done

  python3 - "$OUT" <<'PY' | tee -a "$log"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for policy in ("uniform", "fps", "coverage"):
    path = root / policy / "score-init" / "score_summary.json"
    if not path.exists():
        continue
    for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
        score = row["score"]
        print(f"  {policy:8s} PSNR {row['psnr']:.3f} SSIM {row['ssim']:.4f} "
              f"LPIPS {row['lpips']:.4f} quality {score['quality_points']:.2f}")
PY
else
  cat <<EOF
[round4] selector outputs are ready under $OUT
[round4] GPU stage NOT executed.  On the GPU host run:
  cd $PWD
  RUN_GPU=1 bash research/experiments/round4_a_selection.sh $CASE
This stages $OUT/<policy>/work (a copy; the scene dir is untouched), runs
ReSplat with num_refine=2, and scores the feed-forward initialization (B=0)
with the official PSNR/SSIM/VGG-LPIPS evaluator.  The log is $log.
EOF
fi
echo "done $(date +%Y-%m-%dT%H:%M:%S%z)" >>"$log"
