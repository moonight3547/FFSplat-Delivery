#!/usr/bin/env bash
# Campaign H - compare module-B *approaches*, not knobs.
#
# All variants use the same optimizer settings (4k steps, matched LR schedule, no
# opacity reset, lambda_dssim 0.6, scaling_lr 0.01) so the only difference is the
# module design being tested:
#
#   H0  fastgs      : FastGS continuation from the ReSplat PLY        (current entry)
#   H1  random      : no ReSplat init at all, random frustum points    (how much is the init worth?)
#   H2  densify2    : split every Gaussian x2 before training          (capacity from the init)
#   H3  prune       : drop floaters (opacity<0.01 or max scale>0.5)    (clean the init)
#   H4  prune+dens2 : clean, then split                                (both)
#   H5  sharpen     : push opacities toward 0/1, then train            (floater suppression w/o loss)
#
#   bash research/experiments/campaign_h.sh <lane> <nlanes>
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}
D=${D:-outputs/experiments/campaignH}
LANE=${1:-0}
NLANES=${2:-4}
mkdir -p "$D/ply"

HW=hw/billboard;        BHW=99.49
CO=mipnerf360/counter;  BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29

OPTS="--opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01"

prep() {  # name case -> echoes ply path
  local variant="$1" case="$2" tag src
  tag=$(echo "$case" | tr '/' '_')
  src="outputs/O0-20261030/results/$case/resplat/resplat_world.ply"
  case "$variant" in
    densify2)      out="$D/ply/${tag}_dens2.ply";  [ -f "$out" ] || $PY run --no-capture-output -n fastgs python -m research.ply_ops --in "$src" --out "$out" --op densify --factor 2 >/dev/null 2>&1; echo "$out" ;;
    prune)         out="$D/ply/${tag}_prune.ply";  [ -f "$out" ] || $PY run --no-capture-output -n fastgs python -m research.ply_ops --in "$src" --out "$out" --op prune >/dev/null 2>&1; echo "$out" ;;
    sharpen)       out="$D/ply/${tag}_sharp.ply";  [ -f "$out" ] || $PY run --no-capture-output -n fastgs python -m research.ply_ops --in "$src" --out "$out" --op sharpen --gain 2.0 >/dev/null 2>&1; echo "$out" ;;
    prune+dens2)   out="$D/ply/${tag}_dens2.ply";  [ -f "$out" ] || $PY run --no-capture-output -n fastgs python -m research.ply_ops --in "$src" --out "$out" --op densify --factor 2 >/dev/null 2>&1; echo "$out" ;;
    *)             echo "$src" ;;
  esac
}

run_one() {  # variant case iters budget
  local variant="$1" case="$2" iters="$3" budget="$4" tag dir ply extra
  tag="$variant-$(echo "$case" | tr '/' '-')"
  dir="$D/$tag"
  [ -e "$dir/summary.json" ] && return
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  if [ "$variant" = "random" ]; then
    $PY run --no-capture-output -n fastgs python -m research.run_sweep \
      --profile "competition/profiles/gpufree-$LANE.json" --case "$case" \
      --work "outputs/O0-20261030/work/$case" --init-random \
      --out "$dir" --iterations "$iters" --snapshots "" --budget "$budget" \
      $OPTS --position_lr_max_steps "$iters" >"$D/$tag.log" 2>&1
  else
    ply=$(prep "$variant" "$case")
    $PY run --no-capture-output -n fastgs python -m research.run_sweep \
      --profile "competition/profiles/gpufree-$LANE.json" --case "$case" \
      --work "outputs/O0-20261030/work/$case" --init-ply "$ply" \
      --out "$dir" --iterations "$iters" --snapshots "" --budget "$budget" \
      $OPTS --position_lr_max_steps "$iters" >"$D/$tag.log" 2>&1
  fi
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"
}

VARIANTS=(fastgs random densify2 prune sharpen)
CASES=("$HW:4000:$BHW" "$CO:5000:$BCO" "$DL:4000:$BDL")

# build the job list: variant-major so the four lanes spread different ideas
JOBS=()
for v in "${VARIANTS[@]}"; do
  for entry in "${CASES[@]}"; do
    IFS=':' read -r case iters budget <<<"$entry"
    JOBS+=("$v|$case|$iters|$budget")
  done
done

i=0
for job in "${JOBS[@]}"; do
  i=$((i + 1))
  [ $(( (i - 1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r v case iters budget <<<"$job"
  run_one "$v" "$case" "$iters" "$budget"
done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
