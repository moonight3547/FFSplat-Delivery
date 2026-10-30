#!/usr/bin/env bash
# One scene, three 16-view policies, end-to-end (ReSplat + delivery module B + metrics).
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
RESPLAT=${RESPLAT:-ffsplat-enhance}; FASTGS=${FASTGS:-organizer-fastgs}
CKPT=${CKPT:-/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth}
DATA=${DATA:-/workspace/organizer-assets/datasets}
OUT=${OUT:-outputs/sel1}; mkdir -p "$OUT"
CASE=${CASE:?}; BUDGET=${BUDGET:-95}
g=${CASE%%/*}; s=${CASE#*/}
case $g in dl3dv) dd=DL3DV;; *) dd=$g;; esac
base="$OUT/$(echo $CASE | tr '/' '-')"; mkdir -p "$base"
[ -f "$base/work/split.json" ] || $PY run --no-capture-output -n "$RESPLAT" python competition/upstream/baseline/module_a/preprocess.py \
  --scene_dir "$DATA/$dd/$s" --group "$g" --out "$base/work" --num_context 16 --llffhold 8 --max_resolution 960 > "$base/a.log" 2>&1
for policy in uniform fps coverage; do
  d="$base/$policy"; [ -f "$d/summary.json" ] && continue; rm -rf "$d"; mkdir -p "$d/sel"
  $PY run --no-capture-output -n "$RESPLAT" python -m research.select_views --work "$base/work" --out "$d/sel" --policy "$policy" > "$d/sel.log" 2>&1
  [ -f "$d/sel/context16.json" ] || { echo "SELECT_FAIL $policy"; continue; }
  w="$d/work"; mkdir -p "$w"
  ln -sfn "$(cd "$base/work" && pwd)/images" "$w/images"; ln -sfn "$(cd "$base/work" && pwd)/sparse" "$w/sparse"
  python3 -c "import json,sys;s=json.load(open(sys.argv[1]));s['context']=json.load(open(sys.argv[2]))['context'];json.dump(s,open(sys.argv[3],'w'))" "$base/work/split.json" "$d/sel/context16.json" "$w/split.json"
  $PY run --no-capture-output -n "$RESPLAT" python -m research.resplat_variant --work "$w" --out "$d/resplat" --checkpoint "$CKPT" --num_refine 2 --device cuda:0 > "$d/r.log" 2>&1
  [ -f "$d/resplat/resplat_world.ply" ] || { echo "RESPLAT_FAIL $policy"; continue; }
  $PY run --no-capture-output -n "$FASTGS" python -m research.image_cache --work "$w" --out "$d/cache" >/dev/null 2>&1
  $PY run --no-capture-output -n "$FASTGS" python -m research.run_sweep --case "$CASE" --work "$w" \
    --init-ply "$d/resplat/resplat_world.ply" --out "$d/b" --iterations 5000 --snapshots "" \
    --budget "$BUDGET" --position-lr-max-steps 5000 --densify-until 1250 --image-cache "$d/cache" \
    --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 > "$d/b.log" 2>&1
  cp "$d/b/summary.json" "$d/summary.json" 2>/dev/null
  rm -rf "$d/b/model" "$d/b/eval" "$d/resplat" "$d/cache" 2>/dev/null
  echo "POLICY_DONE $policy"
done
echo ALL_POLICIES_DONE
