#!/usr/bin/env bash
# Module-A test on the scenes that module B cannot fix. Bicycle scores ~10 quality
# points with both the baseline and the delivery recipe, and doubling module B's
# steps only adds ~1 point, so the initialization is the limit. Here the three
# deterministic 16-view policies are compared end-to-end on that scene.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/usr/local/miniconda3/bin/conda}
RESPLAT=${RESPLAT:-ffsplat-enhance}; FASTGS=${FASTGS:-organizer-fastgs}
CKPT=${CKPT:-/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth}
D=${D:-outputs/experiments/selHard}; mkdir -p "$D"
CASE=${CASE:-mipnerf360/bicycle}; BUDGET=${BUDGET:-100.0}
src="outputs/O0-20261030/work/$CASE"
[ -d "$src" ] || src="outputs/prep/$CASE/work"
echo "source: $src"
for policy in uniform fps coverage; do
  dir="$D/$(echo $CASE|tr '/' '-')-$policy"; [ -f "$dir/metrics.json" ] && continue
  rm -rf "$dir"; mkdir -p "$dir"
  echo "=== $policy $(date -Is)"
  $PY run --no-capture-output -n "$RESPLAT" python -m research.select_views --work "$src" --out "$dir/sel" --policy "$policy" >"$dir/log.txt" 2>&1
  [ -f "$dir/sel/context16.json" ] || { echo "select failed"; continue; }
  work="$dir/work"; rm -rf "$work"; mkdir -p "$work"
  ln -sfn "$(cd "$src" && pwd)/images" "$work/images"; ln -sfn "$(cd "$src" && pwd)/sparse" "$work/sparse"
  python3 -c "import json,sys; s=json.load(open(sys.argv[1])); s['context']=json.load(open(sys.argv[2]))['context']; json.dump(s, open(sys.argv[3],'w'))" "$src/split.json" "$dir/sel/context16.json" "$work/split.json"
  $PY run --no-capture-output -n "$RESPLAT" python -m research.resplat_variant --work "$work" --out "$dir/resplat" --checkpoint "$CKPT" --num_refine 2 --device cuda:0 >>"$dir/log.txt" 2>&1
  [ -f "$dir/resplat/resplat_world.ply" ] || { echo "resplat failed"; continue; }
  $PY run --no-capture-output -n "$FASTGS" python -m research.image_cache --work "$work" --out "$dir/cache" >/dev/null 2>&1
  $PY run --no-capture-output -n "$FASTGS" python -m research.run_sweep --case "$CASE" --work "$work" \
    --init-ply "$dir/resplat/resplat_world.ply" --out "$dir/b" --iterations 5000 --snapshots "" \
    --budget "$BUDGET" --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 \
    --position-lr-max-steps 5000 --densify-until 1250 --image-cache "$dir/cache" >>"$dir/log.txt" 2>&1
  cp "$dir/b/summary.json" "$dir/summary.json" 2>/dev/null; rm -rf "$dir/b/model" "$dir/b/eval"
  python3 -c "
import json,sys
d=json.load(open('$dir/summary.json'))
r=d['results'][0]; s=r['score']
print('%-10s PSNR %.3f SSIM %.4f LPIPS %.4f q%6.2f = %5.2f'%('$policy',r['psnr'],r['ssim'],r['lpips'],s['quality_points'],s['total_points']))
" 2>/dev/null
done
