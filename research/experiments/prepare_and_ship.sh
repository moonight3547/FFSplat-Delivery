#!/usr/bin/env bash
# Preparation host (the only machine with module A + ReSplat): for every organizer
# scene that is not yet on the evaluation node, run A -> FastGS random-30k reference
# (the per-scene budget) -> one ReSplat pass, then ship work + PLY + budget to the
# evaluation node and append the manifest entry. Resumable: anything already shipped
# is skipped.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
RESPLAT=${RESPLAT:-ffsplat-enhance}; FASTGS=${FASTGS:-organizer-fastgs}
CKPT=${CKPT:-/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth}
DATA=${DATA:-/workspace/organizer-assets/datasets}
OUT=${OUT:-outputs/prep}
NODE=${NODE:-root@183.147.142.40}
NODE_PORT=${NODE_PORT:-30130}
ENVF=${ENVF:-/root/prep_manifest.json}
mkdir -p "$OUT"
log="$OUT/run.log"
DL="001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f 0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce 0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e 2dad2d298927ff9b2ed7daa6128244e81d331e8cf1abf6c8598f35276ad38b84 3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3 9a451604b8d2583d47b14662635d8f74710c419cc061705ac15b8cfc31ece446"
HW="billboard basketball car trashbin"; MP="bicycle bonsai counter"
CASES=(); for s in $DL; do CASES+=("dl3dv/$s"); done
for s in $HW; do CASES+=("hw/$s"); done; for s in $MP; do CASES+=("mipnerf360/$s"); done
for case in "${CASES[@]}"; do
  group=${case%%/*}; scene=${case#*/}; d="$OUT/$case"
  case "$group" in dl3dv) dd=DL3DV ;; *) dd=$group ;; esac
  src="$DATA/$dd/$scene"; work="$d/work"
  [ -f "$d/shipped" ] && continue
  [ -d "$src" ] || { echo "missing $src" >>"$log"; continue; }
  mkdir -p "$d"; echo "=== $case $(date -Is)" >>"$log"
  [ -f "$work/split.json" ] || $PY run --no-capture-output -n "$RESPLAT" python competition/upstream/baseline/module_a/preprocess.py \
    --scene_dir "$src" --group "$group" --out "$work" --num_context 16 --llffhold 8 --max_resolution 960 >>"$log" 2>&1
  [ -f "$work/split.json" ] || { echo "A failed $case" >>"$log"; continue; }
  [ -f "$d/ref/module_b.json" ] || $PY run --no-capture-output -n "$FASTGS" python competition/upstream/baseline/module_b/train_from_ply_fastgs.py \
    -s "$work" --eval -r 1 --sh_degree 3 --quiet -m "$d/ref" --iterations 30000 \
    --position_lr_max_steps 30000 --densify_until_iter 15000 --init_random >>"$log" 2>&1
  budget=$(python3 -c "import json;print(json.load(open('$d/ref/module_b.json'))['total_seconds'])" 2>/dev/null || echo "")
  [ -n "$budget" ] || { echo "ref failed $case" >>"$log"; continue; }
  [ -f "$d/resplat/resplat_world.ply" ] || $PY run --no-capture-output -n "$RESPLAT" python -m research.resplat_variant \
    --work "$work" --out "$d/resplat" --checkpoint "$CKPT" --num_refine 2 --device cuda:0 >>"$log" 2>&1
  [ -f "$d/resplat/resplat_world.ply" ] || { echo "resplat failed $case" >>"$log"; continue; }
  # ship work + PLY to the evaluation node
  tar cf - -C "$d" work resplat/resplat_world.ply 2>/dev/null | \
    ssh -i ~/.ssh/ton1 -o StrictHostKeyChecking=no -p "$NODE_PORT" "$NODE" \
    "mkdir -p /root/projects/FFSplat-Hackathon/$d && tar xf - -C /root/projects/FFSplat-Hackathon/$d" 2>/dev/null
  python3 - "$d" "$case" "$budget" >>"$log" <<'PY'
import json, pathlib, sys
d, case, budget = pathlib.Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
entry = dict(case=case, work=str(d / "work"),
             init=str(d / "resplat/resplat_world.ply"), budget_seconds=budget)
print("MANIFEST", json.dumps(entry))
PY
  touch "$d/shipped"
  # keep only the manifest line and the measured budget: the preparation host has a
  # small disk and the artifacts already live on the evaluation node
  python3 -c "import json,sys; print(json.dumps(dict(case=sys.argv[2], work=sys.argv[1]+'/work', init=sys.argv[1]+'/resplat/resplat_world.ply', budget_seconds=float(sys.argv[3]))))" "$d" "$case" "$budget" >>"$OUT/manifest.jsonl"
  rm -rf "$d/ref/point_cloud" "$d/ref/test_renders" "$d/ref/test_gt" "$d/work/images" "$d/resplat" 2>/dev/null
done
echo "PREP_DONE $(date -Is)" >>"$log"
