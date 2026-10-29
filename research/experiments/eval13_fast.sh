#!/usr/bin/env bash
# Fast 13-scene evaluation on a dedicated multi-GPU node.
#
# Only module B + the official metrics run here: module A / ReSplat are done once on
# the preparation host and their artifacts (prepared work dir, world PLY, measured
# budget) are shipped in. A full round is therefore 13 x (B ~30 s + render + metrics)
# spread over the lanes, which fits in ~<10 min on 3 cards.
#
# Inputs (per scene, from a manifest JSON):
#   case            case id, e.g. "hw/billboard"
#   work            prepared scene dir (module A output)
#   init            ReSplat world PLY
#   budget_seconds  FastGS random-init 30k seconds measured on the same GPU class
#
#   bash research/experiments/eval13_fast.sh <lane> <nlanes> <manifest> <outdir> [recipe args...]
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}; ENV=${ENV:-fastgs}
LANE=${1:?lane}; NLANES=${2:?nlanes}; MANIFEST=${3:?manifest}; OUT=${4:?outdir}
shift 4
RECIPE=${RECIPE:-"--opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01"}
STEPS=${STEPS:-5000}
DENSIFY_UNTIL=${DENSIFY_UNTIL:-1250}
mkdir -p "$OUT"
python3 - "$MANIFEST" "$OUT/scenes.tsv" <<'PY'
import json, sys
scenes = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as fh:
    for s in scenes:
        fh.write("\t".join([s["case"], s["work"], s["init"], str(s["budget_seconds"])]) + "\n")
PY
i=0
while IFS=$'\t' read -r case work init budget; do
  i=$((i + 1))
  [ $(( (i - 1) % NLANES )) -eq "$LANE" ] || continue
  tag=$(echo "$case" | tr '/' '-')
  dir="$OUT/$tag"
  [ -f "$dir/metrics.json" ] && continue
  rm -rf "$dir"
  echo "start $tag lane$LANE $(date -Is)" >>"$OUT/queue.log"
  CUDA_VISIBLE_DEVICES=$LANE $PY run --no-capture-output -n "$ENV" python -m research.run_sweep \
    --profile "competition/profiles/gpufree-$LANE.json" --case "$case" --work "$work" \
    --init-ply "$init" --out "$dir" --iterations "$STEPS" --snapshots "" \
    --budget "$budget" --position-lr-max-steps "$STEPS" --densify-until "$DENSIFY_UNTIL" \
    $RECIPE "$@" >>"$OUT/$tag.log" 2>&1
  echo "exit=$? $tag $(date -Is)" >>"$OUT/queue.log"
  rm -rf "$dir/model" "$dir/eval" 2>/dev/null
done < "$OUT/scenes.tsv"
python3 - "$OUT" <<'PY'
import glob, json, statistics, sys
rows = []
for p in sorted(glob.glob(sys.argv[1] + "/*/summary.json")):
    d = json.load(open(p))
    for r in d["results"]:
        s = r["score"]
        rows.append((d["case"], r["b_seconds_estimate"], r["psnr"], r["ssim"], r["lpips"],
                     s["quality_points"], s["time_points"], s["total_points"], s["valid"]))
for r in rows:
    print("%-44s B %6.2fs PSNR %.3f SSIM %.4f LPIPS %.4f q %6.2f t %5.2f = %5.2f %s" %
          (r[0][:44], r[1], r[2], r[3], r[4], r[5], r[6], r[7], "" if r[8] else "INVALID"))
if rows:
    q = statistics.mean(r[5] for r in rows)
    tot = statistics.mean(r[7] for r in rows)
    print("SCENES %d  mean_quality %.2f  mean_total %.2f" % (len(rows), q, tot))
PY
