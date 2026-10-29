#!/usr/bin/env bash
# Evaluation node daemon: rebuild the manifest from whatever scenes have arrived,
# then run both arms (organizer baseline and the delivery recipe) on the missing ones.
# Idempotent: a scene+arm with metrics already present is skipped, so this can be
# started once and left running while the preparation host ships scenes.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PY=${PY:-/opt/conda/bin/conda}
NODE_OUT=${NODE_OUT:-outputs/rounds}
mkdir -p "$NODE_OUT"
while true; do
  python3 - "${NODE_OUT}/manifest.json" <<'PY'
import glob, json, os, sys
scenes = []
for w in sorted(glob.glob("outputs/prep/*/*/work") + glob.glob("outputs/O0-20261030/work/*/*")):
    if not os.path.exists(os.path.join(w, "split.json")):
        continue
    case = "/".join(w.replace("/work", "").split("/")[-2:])
    init = w.replace("/work", "/resplat/resplat_world.ply")
    if not os.path.exists(init):
        init = "outputs/O0-20261030/results/%s/resplat/resplat_world.ply" % case
    if not os.path.exists(init):
        continue
    budget = {"hw/billboard": 99.49, "mipnerf360/counter": 92.57}.get(case, 100.0)
    if case.startswith("dl3dv/"):
        budget = 94.29
    scenes.append(dict(case=case, work=w, init=init, budget_seconds=budget))
json.dump(scenes, open(sys.argv[1], "w"), indent=2)
print("[watch] scenes:", len(scenes))
PY
  for l in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$l STEPS=5000 DENSIFY_UNTIL=1250 RECIPE="--opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01" \
      bash research/experiments/eval13_fast.sh $l 3 "$NODE_OUT/manifest.json" "$NODE_OUT/delivery" > "$NODE_OUT/log_delivery_$l.txt" 2>&1 &
  done
  wait
  for l in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$l STEPS=10000 DENSIFY_UNTIL=15000 LR_MAX_STEPS=30000 RECIPE="--opacity-reset-interval 3000" \
      bash research/experiments/eval13_fast.sh $l 3 "$NODE_OUT/manifest.json" "$NODE_OUT/baseline" > "$NODE_OUT/log_baseline_$l.txt" 2>&1 &
  done
  wait
  sleep 300
done
