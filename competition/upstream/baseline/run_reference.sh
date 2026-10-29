#!/usr/bin/env bash
# OPTIONAL: classic-3DGS-from-scratch reference (the time bar players must beat).
# Trains vanilla 3DGS for 30k iterations from random frustum points on the
# same work dirs / splits produced by Module A, and reports wall-clock time.
#
# Usage:
#   bash run_reference.sh [groups...]
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ITERATIONS=${REF_ITERATIONS:-30000}

source /home/guojunfu/anaconda3/etc/profile.d/conda.sh

SCENE_GROUPS=("$@")
if [ ${#SCENE_GROUPS[@]} -eq 0 ]; then
  SCENE_GROUPS=($(python -c "import json,sys;print(' '.join(json.load(open('$HERE/scenes.json')).keys()))"))
fi

for GRP in "${SCENE_GROUPS[@]}"; do
  SCENE=$(python -c "import json;print(json.load(open('$HERE/scenes.json'))['$GRP']['scene'])")
  WORK="$HERE/work/$GRP/$SCENE"
  RES="$HERE/results/$GRP/$SCENE/reference_3dgs"
  if [ ! -f "$WORK/split.json" ]; then
    echo "[$GRP/$SCENE] work dir missing ($WORK) - run run_train.sh first; skipping"
    continue
  fi
  echo ""
  echo "=== [$GRP] $SCENE : classic 3DGS reference (30k, random init) ==="
  conda activate fastgs
  export PYTHONPATH="$HERE/3dgs:$HERE/3dgs/submodules/diff-gaussian-rasterization:$HERE/3dgs/submodules/simple-knn:${PYTHONPATH:-}"
  python "$HERE/module_b/train_from_ply.py" \
      -s "$WORK" \
      -m "$RES" \
      --eval \
      -r 1 \
      --sh_degree 3 \
      --iterations "$ITERATIONS" \
      --position_lr_max_steps "$ITERATIONS" \
      --densify_until_iter 15000 \
      --init_random \
      --quiet
  conda deactivate
done
python "$HERE/summarize.py" --work "$HERE/work" --results "$HERE/results" --out "$HERE/results"
