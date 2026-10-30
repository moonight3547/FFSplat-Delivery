#!/usr/bin/env python3
"""Build the cell list for the N1 texture-voxel-growth experiment (task-7).

Each row is one module-B training run:
  tag <TAB> case <TAB> work <TAB> init_ply <TAB> budget <TAB> arm <TAB> transform_args

`control` uses the untouched ReSplat init PLY.  Every other arm first writes a
transformed PLY with `python -m research.ply_voxgrow ... <transform_args>` into
$D/ply/<tag>.ply (cached: re-running a finished cell costs nothing).

Primary arm = net-flat (prune the lowest-opacity 16.7% so the +20% growth restores the
original total count -> zero module-B TIME penalty).  `td_plus20` is the same growth
without the net-flat prune (total +16.1%), included to price the extra primitives.
"""
import json
import sys
from pathlib import Path

MANIFEST = "/root/projects/FFSplat-Hackathon/records/evalnode-manifest-13.json"
SCENES = {
    "mip-bicycle": "mipnerf360/bicycle",
    "dl-0f622ea": "dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e",
    "hw-billboard": "hw/billboard",
    "mip-counter": "mipnerf360/counter",
}
ARMS = {
    # arm name: (transform args or None for control)
    "control": None,
    "td_flat": "--op grow_texdist --net-flat --voxels 32",
    "tdm_flat": "--op grow_texdist_merge --net-flat --voxels 32",
    "do_flat": "--op grow_distonly --net-flat --voxels 32",
    "td_plus20": "--op grow_texdist --voxels 32",
    "td_flat64": "--op grow_texdist --net-flat --voxels 64",
    "prune_only": "--op prune --net-flat",
}
MATRIX = [(s, a) for a in ("control", "td_flat", "tdm_flat", "do_flat") for s in SCENES]
MATRIX += [("mip-bicycle", "td_plus20"), ("dl-0f622ea", "td_plus20"),
           ("mip-bicycle", "td_flat64"), ("mip-bicycle", "prune_only"),
           ("dl-0f622ea", "prune_only")]
# serial anchors: single card, other two GPUs idle -> clean B/TIME
SERIAL = [(s, a) for s in ("mip-bicycle", "dl-0f622ea")
          for a in ("control", "tdm_flat", "td_plus20")]


def main(kind="main"):
    man = {e["case"]: e for e in json.load(open(MANIFEST))}
    rows, seen = [], set()
    matrix = MATRIX if kind == "main" else SERIAL
    for scene, arm in matrix:
        case = SCENES[scene]
        e = man[case]
        tag = f"{scene}__{arm}" + ("" if kind == "main" else "__serial")
        if tag in seen:
            continue
        seen.add(tag)
        rows.append("\t".join([tag, case, e["work"], e["init"], str(e["budget_seconds"]),
                               arm, ARMS[arm] or "-"]))
    print("\n".join(rows))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
