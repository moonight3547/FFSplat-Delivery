#!/usr/bin/env bash
# Campaign K - hybrid structural schedule for module B.
#
# Campaign H/I isolated two opposite designs:
#   FastGS VCD/VCP densify+prune everywhere  -> best on hw/counter
#   attribute-only refinement (ForeSplat)    -> best on the large DL3DV scene
# The structural decision does not have to be global in time. This campaign turns
# densification on/off along the *same* 4-5k step run and keeps the ReSplat PLY
# (optionally opacity-sharpened) as the only init.
#
#   bash research/experiments/campaign_k.sh <lane> <nlanes>
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}
D=${D:-outputs/experiments/campaignK}
LANE=${1:-0}
NLANES=${2:-4}
mkdir -p "$D/ply"

HW=hw/billboard;        BHW=99.49
CO=mipnerf360/counter;  BCO=92.57
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f; BDL=94.29
OPTS="--opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01"

sharp_ply() {
  local tag src out
  tag=$(echo "$1" | tr '/' '_')
  src="outputs/O0-20261030/results/$1/resplat/resplat_world.ply"
  out="$D/ply/${tag}_sharp.ply"
  [ -f "$out" ] || $PY run --no-capture-output -n fastgs python -m research.ply_ops \
      --in "$src" --out "$out" --op sharpen --gain 2.0 >/dev/null 2>&1
  echo "$out"
}

run_one() {  # tag case iters budget ply extra...
  local tag="$1" case="$2" iters="$3" budget="$4" plymode="$5"; shift 5
  local dir="$D/$tag" ply
  [ -e "$dir/summary.json" ] && return
  if [ "$plymode" = sharp ]; then ply=$(sharp_ply "$case"); else ply="outputs/O0-20261030/results/$case/resplat/resplat_world.ply"; fi
  echo "start $tag lane$LANE $(date -Is)" >>"$D/queue.log"
  $PY run --no-capture-output -n fastgs python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$LANE.json" --case "$case" \
    --work "outputs/O0-20261030/work/$case" --init-ply "$ply" \
    --out "$dir" --iterations "$iters" --snapshots "" --budget "$budget" \
    $OPTS --position-lr-max-steps "$iters" "$@" >"$D/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$D/queue.log"
}

# tag|case|iters|budget|ply|extra
JOBS=(
  # structural ops only in the second half (attribute-only warm start)
  "k_hw_half_4k|$HW|4000|$BHW|raw|--densify-from 2000"
  "k_co_half_5k|$CO|5000|$BCO|raw|--densify-from 2500"
  "k_dl_half_4k|$DL|4000|$BDL|raw|--densify-from 2000"
  # structural ops only in the first quarter, then attribute-only polish
  "k_hw_early_4k|$HW|4000|$BHW|raw|--densify-until 1000"
  "k_co_early_5k|$CO|5000|$BCO|raw|--densify-until 1250"
  "k_dl_early_4k|$DL|4000|$BDL|raw|--densify-until 1000"
  # same two schedules on the sharpened init
  "k_hw_half_4k_sharp|$HW|4000|$BHW|sharp|--densify-from 2000"
  "k_co_half_5k_sharp|$CO|5000|$BCO|sharp|--densify-from 2500"
  "k_dl_half_4k_sharp|$DL|4000|$BDL|sharp|--densify-from 2000"
  # no structural ops at all, but with the sharpened init and full horizon
  "k_dl_nodens_4k_sharp|$DL|4000|$BDL|sharp|--no-densify"
  "k_hw_nodens_5k_sharp|$HW|5000|$BHW|sharp|--no-densify"
  "k_co_nodens_6k_sharp|$CO|6000|$BCO|sharp|--no-densify"
)

i=0
for job in "${JOBS[@]}"; do
  i=$((i + 1))
  [ $(( (i - 1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case iters budget ply extra <<<"$job"
  # shellcheck disable=SC2086
  run_one "$tag" "$case" "$iters" "$budget" "$ply" $extra
done
echo "lane$LANE done $(date -Is)" >>"$D/queue.log"
