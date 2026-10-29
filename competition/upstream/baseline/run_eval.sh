#!/usr/bin/env bash
# One-click evaluation:
#   1. render test views with the trained Module-B model (fastgs env)
#   2. PSNR / SSIM / LPIPS via ReSplat's official metric code (resplat env)
#   3. summary table (metrics + per-stage timing)
#
# Usage:
#   bash run_eval.sh                 # all groups in scenes.json
#   bash run_eval.sh dl3dv hw        # subset of groups
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ITERATIONS=${ITERATIONS:-10000}


source /home/guojunfu/anaconda3/etc/profile.d/conda.sh


SCENE_GROUPS=("$@")
if [ ${#SCENE_GROUPS[@]} -eq 0 ]; then
  SCENE_GROUPS=($(python -c "import json,sys;print(' '.join(json.load(open('$HERE/scenes.json')).keys()))"))
fi

for GRP in "${SCENE_GROUPS[@]}"; do
  SCENE=$(python -c "import json;print(json.load(open('$HERE/scenes.json'))['$GRP']['scene'])")
  WORK="$HERE/work/$GRP/$SCENE"
  RES="$HERE/results/$GRP/$SCENE"
  MODEL="$RES/module_b"
  if [ ! -f "$MODEL/point_cloud/iteration_$ITERATIONS/point_cloud.ply" ]; then
    echo "[$GRP/$SCENE] trained model missing ($MODEL) - skipping"
    continue
  fi
  echo ""
  echo "=== [$GRP] $SCENE : render + eval ==="

  # 1. render test views (fastgs env; render_test.py resolves its own backend path)
  conda activate fastgs
  python "$HERE/render_test.py" \
      -s "$WORK" \
      -m "$MODEL" \
      --eval -r 1 \
      --backend fastgs \
      --iteration "$ITERATIONS" \
      --quiet
  conda deactivate

  # 2. metrics (resplat env)
  conda activate resplat
  python "$HERE/evaluate.py" --model_path "$MODEL"
  conda deactivate
done

# 3. summary
python "$HERE/summarize.py" --work "$HERE/work" --results "$HERE/results" --out "$HERE/results"
