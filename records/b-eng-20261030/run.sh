#!/usr/bin/env bash
# Task-18 module-B engineering audit: breakdown, microbenches, prototype A/B, A-side scene cache.
# N1, serial quiet box for anything timed. Raw output only under /root/gpufree-data.
set -u
REPO=/root/projects/FFSplat-Hackathon
OUT=/root/gpufree-data/outputs/b-eng-20261030
PY=/opt/conda/envs/fastgs/bin/python
cd "$REPO"

# 1) time breakdown (Part 1) -- sync-instrumented shares
#    $PY /tmp/stagedloss/beng_breakdown.py        # bicycle + 0b5635 -> breakdown/*/record.json
# 2) microbenches (Part 2/3) -- no-op .cuda, list copy, realloc parse, metric_map, render reuse
#    $PY /tmp/stagedloss/beng_micro.py            # -> micro/microbench.json
# 3) isolated A/B of the changed work: test cameras inside B
#    $PY /tmp/stagedloss/beng_camonly2.py        # -> camonly/camonly_isolated2.json
# 4) prototype A/B (noise-limited): ctrl / train_cameras_only / lean_step / both
#    $PY /tmp/stagedloss/beng_ab.py              # -> ab/ab.jsonl

# 5) A-side scene tensor cache (user-approved data residency): build once per prepared dir
$PY -m research.scene_cache build --work outputs/prep/mipnerf360/bicycle/work \
    --out "$OUT/scene_cache/bicycle"
$PY -m research.scene_cache verify --work outputs/prep/mipnerf360/bicycle/work \
    --cache "$OUT/scene_cache/bicycle"
$PY -m research.scene_cache bench --work outputs/prep/mipnerf360/bicycle/work \
    --cache "$OUT/scene_cache/bicycle" --reps 3

# 6) B-side cache hit in the real training loop (B scene-load bytes: cache vs normal)
#    $PY /tmp/stagedloss/beng_cache_hits.py       # -> cache_hits/cache_hits.json
