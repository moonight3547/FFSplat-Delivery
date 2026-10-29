#!/usr/bin/env bash
# Campaign C - the LR schedule matched to the real run length.
# Campaign B showed that for a short continuation the default 30k LR schedule is
# still near its start, so the position LR never decays: setting
# --position-lr-max-steps = <iterations> gave hw 35.4 vs 32.5 quality points at the
# same 8k steps. This campaign finds the iteration sweet spot per case and
# validates the combination (noreset + lambda_dssim 0.4 + matched LR schedule).
#
#   bash research/experiments/campaign_c.sh <lane> <nlanes>
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}
OUT=${OUT:-outputs/experiments/campaignC}
LANE=${1:-0}
NLANES=${2:-2}
mkdir -p "$OUT"

HW=hw/billboard
CO=mipnerf360/counter
DL=dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f

# tag|case|iterations
JOBS=(
  "c_hw_lrm_3k|$HW|3000"
  "c_hw_lrm_4k|$HW|4000"
  "c_hw_lrm_6k|$HW|6000"
  "c_hw_lrm_8k|$HW|8000"
  "c_co_lrm_4k|$CO|4000"
  "c_co_lrm_8k|$CO|8000"
  "c_dl_lrm_4k|$DL|4000"
  "c_dl_lrm_8k|$DL|8000"
  "c_hw_lrm_8k_d06|$HW|8000|dssim"
  "c_co_lrm_4k_d06|$CO|4000|dssim"
  "c_dl_lrm_8k_d06|$DL|8000|dssim"
  "c_hw_lrm_8k_mult10|$HW|8000|mult"
  "c_hw_lrm_6k_d06|$HW|6000|dssim"
  "c_co_lrm_8k_d06|$CO|8000|dssim"
)

i=0
for job in "${JOBS[@]}"; do
  i=$((i + 1))
  [ $(( (i - 1) % NLANES )) -eq "$LANE" ] || continue
  IFS='|' read -r tag case iters variant <<<"$job"
  [ -e "$OUT/$tag/summary.json" ] && continue
  extra="--opacity-reset-interval 0 --lambda-dssim 0.4 --position-lr-max-steps $iters"
  case "${variant:-}" in
    dssim) extra="--opacity-reset-interval 0 --lambda-dssim 0.6 --position-lr-max-steps $iters" ;;
    mult)  extra="--opacity-reset-interval 0 --lambda-dssim 0.4 --position-lr-max-steps $iters --mult 1.0" ;;
  esac
  echo "start $tag lane$LANE $(date -Is)" >>"$OUT/queue.log"
  # shellcheck disable=SC2086
  $PY run --no-capture-output -n fastgs python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$((LANE % 2)).json" \
    --case "$case" --work "outputs/O0-20261030/work/$case" \
    --init-ply "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" \
    --out "$OUT/$tag" --iterations "$iters" --snapshots "" --budget 95.0 \
    $extra >"$OUT/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$OUT/queue.log"
done
echo "lane$LANE done $(date -Is)" >>"$OUT/queue.log"
