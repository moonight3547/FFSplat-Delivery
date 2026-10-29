#!/usr/bin/env bash
# Baseline-vs-recipe on all 13 organizer scenes (runs on the calibrated host:
# it is the only machine with module A + ReSplat + COLMAP).
#
# Per scene, once:
#   1. module A (organizer preprocess)            -> work/
#   2. reference FastGS random 30k                -> measured budget
#   3. ReSplat (num_refine=2, frozen weights)     -> world-space PLY
#   4. module B, organizer recipe (10k, defaults) -> baseline metrics
#   5. module B, our recipe (4k, matched LR, noreset, dssim .6, scaling_lr .01)
#   6. keep every JSON, drop the heavy model dirs (reproducible from the PLY)
#
# Resumable: a scene with both reports is skipped. Safe to relaunch after the
# pod drops.
#   bash research/experiments/baseline13.sh [max_scenes]
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
RESPLAT=${RESPLAT:-ffsplat-enhance}
FASTGS=${FASTGS:-organizer-fastgs}
CKPT=${CKPT:-/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth}
DATA=${DATA:-/workspace/organizer-assets/datasets}
OUT=${OUT:-outputs/baseline13}
LIMIT=${1:-99}
mkdir -p "$OUT"
log="$OUT/run.log"
echo "start $(date -Is)" | tee -a "$log"

DL="001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f 0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce 0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e 2dad2d298927ff9b2ed7daa6128244e81d331e8cf1abf6c8598f35276ad38b84 3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3 9a451604b8d2583d47b14662635d8f74710c419cc061705ac15b8cfc31ece446"
HW="billboard basketball car trashbin"
MP="bicycle bonsai counter"

CASES=()
for s in $DL; do CASES+=("dl3dv/$s"); done
for s in $HW; do CASES+=("hw/$s"); done
for s in $MP; do CASES+=("mipnerf360/$s"); done

done_count=0
for case in "${CASES[@]}"; do
  group=${case%%/*}; scene=${case#*/}
  src="$DATA/$group/$scene"
  d="$OUT/$case"
  if [ -f "$d/recipe/metrics.json" ] && [ -f "$d/baseline/metrics.json" ]; then
    echo "skip $case (done)" >>"$log"; continue
  fi
  [ "$done_count" -ge "$LIMIT" ] && { echo "limit reached" >>"$log"; break; }
  if [ ! -d "$src" ]; then echo "missing scene $src" >>"$log"; continue; fi
  echo "=== $case $(date -Is)" >>"$log"
  work="$d/work"; mkdir -p "$d"
  if [ ! -f "$work/split.json" ]; then
    $PY run --no-capture-output -n "$RESPLAT" python competition/upstream/baseline/module_a/preprocess.py \
      --scene_dir "$src" --group "$group" --out "$work" --num_context 16 --llffhold 8 \
      --max_resolution 960 >>"$log" 2>&1 || { echo "A failed $case" >>"$log"; continue; }
  fi
  if [ ! -f "$d/reference/module_b.json" ]; then
    $PY run --no-capture-output -n "$FASTGS" python competition/upstream/baseline/module_b/train_from_ply_fastgs.py \
      -s "$work" --eval -r 1 --sh_degree 3 --quiet -m "$d/reference" \
      --iterations 30000 --position_lr_max_steps 30000 --densify_until_iter 15000 --init_random \
      >>"$log" 2>&1 || { echo "ref failed $case" >>"$log"; continue; }
  fi
  if [ ! -f "$d/resplat/resplat_world.ply" ]; then
    $PY run --no-capture-output -n "$RESPLAT" python -m research.resplat_variant \
      --work "$work" --out "$d/resplat" --checkpoint "$CKPT" --num_refine 2 --device cuda:0 \
      >>"$log" 2>&1 || { echo "resplat failed $case" >>"$log"; continue; }
  fi
  for variant in baseline recipe; do
    [ -f "$d/$variant/metrics.json" ] && continue
    if [ "$variant" = baseline ]; then
      bargs="--iterations 10000 --position_lr_max_steps 30000"
    else
      bargs="--iterations 4000 --position_lr_max_steps 4000 --opacity_reset_interval 0 --lambda_dssim 0.6 --scaling_lr 0.01"
    fi
    # shellcheck disable=SC2086
    $PY run --no-capture-output -n "$FASTGS" python -m research.train_sweep \
      --source "$work" --model_path "$d/$variant" --eval -r 1 --sh_degree 3 --quiet \
      --init_ply "$d/resplat/resplat_world.ply" $bargs --record "$d/$variant/train_record.json" \
      >>"$log" 2>&1 || { echo "B-$variant failed $case" >>"$log"; continue; }
    it=$(python3 -c "import json;print(json.load(open('$d/$variant/train_record.json'))['completed_iterations'])")
    $PY run --no-capture-output -n "$FASTGS" python competition/upstream/baseline/render_test.py \
      -s "$work" -m "$d/$variant" --backend fastgs --iteration "$it" --eval -r 1 --sh_degree 3 --quiet \
      >>"$log" 2>&1
    $PY run --no-capture-output -n "$RESPLAT" python competition/upstream/baseline/evaluate.py \
      --model_path "$d/$variant" --device cuda:0 >>"$log" 2>&1
    rm -rf "$d/$variant/point_cloud" "$d/$variant/test_renders" "$d/$variant/test_gt"
  done
  # free the raw scene once prepared (dataset archive stays on the controller)
  rm -rf "$src"
  done_count=$((done_count+1))
  python3 - "$d" <<'PY' | tee -a "$log"
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
row = {"case": str(d).split("baseline13/")[-1]}
for v in ("baseline", "recipe"):
    m = d / v / "metrics.json"
    r = d / v / "train_record.json"
    if m.exists():
        row[v] = {"mean": json.loads(m.read_text())["mean"],
                  "b_seconds": json.loads(r.read_text())["total_seconds"] if r.exists() else None}
print(json.dumps(row))
PY
done
echo "done $(date -Is)" >>"$log"
