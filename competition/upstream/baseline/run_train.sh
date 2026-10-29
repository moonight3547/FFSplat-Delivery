#!/usr/bin/env bash
# One-click end-to-end baseline training:
#   Module A (pose acquisition + 16-view selection)
#     -> ReSplat (16-view feed-forward reconstruction)
#     -> Module B (classic 3DGS continuation, 10k steps, all train views)
#
# Usage:
#   bash run_train.sh                 # all groups in scenes.json
#   bash run_train.sh dl3dv hw        # subset of groups
#
# Outputs:
#   work/<group>/<scene>/            Module A work dir (COLMAP-format scene)
#   results/<group>/<scene>/resplat/  world PLY + stage timing
#   results/<group>/<scene>/module_b/ trained model + stage timing
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
COMP="$(cd "$HERE/.." && pwd)"
DATASETS="$COMP/datasets"
NUM_CONTEXT=${NUM_CONTEXT:-16}
ITERATIONS=${ITERATIONS:-10000}


source /home/guojunfu/anaconda3/etc/profile.d/conda.sh


SCENE_GROUPS=("$@")
if [ ${#SCENE_GROUPS[@]} -eq 0 ]; then
  SCENE_GROUPS=($(python -c "import json,sys;print(' '.join(json.load(open('$HERE/scenes.json')).keys()))"))
fi

for GRP in "${SCENE_GROUPS[@]}"; do
  SCENE=$(python -c "import json;print(json.load(open('$HERE/scenes.json'))['$GRP']['scene'])")
  SRC_REL=$(python -c "import json;print(json.load(open('$HERE/scenes.json'))['$GRP']['source'])")
  SCENE_DIR="$COMP/$SRC_REL/$SCENE"
  WORK="$HERE/work/$GRP/$SCENE"
  RES="$HERE/results/$GRP/$SCENE"
  echo ""
  echo "==================================================================="
  echo "  [$GRP] $SCENE"
  echo "==================================================================="

  if [ ! -d "$SCENE_DIR" ]; then
    echo "scene dir not found: $SCENE_DIR (dataset not downloaded?) - skipping"
    continue
  fi

  # ---------------- Module A (resplat env) ----------------
  conda activate resplat
  python "$HERE/module_a/preprocess.py" \
      --scene_dir "$SCENE_DIR" \
      --group "$GRP" \
      --out "$WORK" \
      --num_context "$NUM_CONTEXT"

  # ---------------- ReSplat (resplat env) ----------------
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python "$HERE/run_resplat.py" \
      --work "$WORK" \
      --out "$RES/resplat"

  # ---------------- Module B: FastGS (fastgs env) ----------------
  conda deactivate
  conda activate fastgs
  export PYTHONPATH="$HERE/fastgs:${PYTHONPATH:-}"
  python "$HERE/module_b/train_from_ply_fastgs.py" \
      -s "$WORK" \
      -m "$RES/module_b" \
      --eval \
      -r 1 \
      --sh_degree 3 \
      --iterations "$ITERATIONS" \
      --init_ply "$RES/resplat/resplat_world.ply" \
      --quiet
  conda deactivate
done

echo ""
echo "Training finished. Run 'bash run_eval.sh' to compute metrics."
