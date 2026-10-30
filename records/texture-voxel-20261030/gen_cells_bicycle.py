#!/usr/bin/env python3
"""Cell list for the bicycle-only grow-only densification sweep (task-9).

One row per module-B run:
  tag <TAB> case <TAB> work <TAB> init_ply <TAB> budget <TAB> arm <TAB> transform_args

All growth arms are GROW-ONLY (`--no-prune`), 64^3 voxels, and use the distance-only
weight (`--op grow_distonly`) -- the texture term was shown to add nothing in task-7.
The transform PLY is cached under $D/ply/<tag>.ply, so re-running a finished cell is free.

G5 `faronly` and G6 `trajectory` from the task text are intentionally NOT in the matrix:
the train-view diagnostic (research/init_error_diag.py) shows the init error share in the
periphery is *below* the Gaussian share (scene-centre: 7.3% error vs 8.7% Gaussians beyond
0.5R; camera-trajectory: 33.3% vs 41.5% beyond 1.5x median), i.e. the growth's far prior
has no target.  `inneronly` is added instead as the sign test.
"""
import json
import sys

MANIFEST = "/root/projects/FFSplat-Hackathon/records/evalnode-manifest-13.json"
CASE = "mipnerf360/bicycle"
SCENE = "mip-bicycle"
CAM = "/root/projects/FFSplat-Hackathon/outputs/prep/mipnerf360/bicycle/work/sparse/0/images.txt"
ARMS = {
    "control": None,
    # G1: user's original idea, but grow-only + discriminative gate (count <= 1 at 64^3)
    "g1_hard05": "--op grow_distonly --no-prune --grow-frac 0.2 --voxels 64 "
                 "--under-mode le1 --dist-mode hard --far-frac 0.5 --far-weight 2",
    # G2: continuous (d/R)^2 ramp instead of the hard 0.5R step
    "g2_ramp2": "--op grow_distonly --no-prune --grow-frac 0.2 --voxels 64 "
                "--under-mode le1 --dist-mode ramp2 --far-frac 0.5 --far-weight 2",
    # G3: 10th-percentile gate + lower threshold 0.35R
    "g3_p10_035": "--op grow_distonly --no-prune --grow-frac 0.2 --voxels 64 "
                  "--under-mode p10 --dist-mode hard --far-frac 0.35 --far-weight 2",
    # G4: +40% budget with the ramp weight
    "g4_grow40": "--op grow_distonly --no-prune --grow-frac 0.4 --voxels 64 "
                 "--under-mode le1 --dist-mode ramp2 --far-frac 0.5 --far-weight 2",
    # G9: hole-targeted growth -- weight by the 3D density of train-view alpha<0.5 hole
    # points (research.init_error_diag --out-holes), not by distance
    "g9_holes": "--op grow_distonly --no-prune --grow-frac 0.2 --voxels 64 "
                "--under-mode none --dist-mode holes "
                "--hole-points /root/gpufree-data/outputs/bicycle-grow-20261030/holes_bicycle.npy",
    # G10: error-density growth -- weight per 64^3 voxel = accumulated TRAIN-view rendered
    # squared error (unprojected with z-buffer/EDT depth).  This is the mechanism implied
    # by the alpha diagnostic: the init covers everything, it is just inaccurate, so grow
    # where the model is demonstrably wrong (train views only, uniform rule).
    "g10_errgrow": "--op grow_distonly --no-prune --grow-frac 0.2 --voxels 64 "
                   "--under-mode none --dist-mode errdensity "
                   "--error-points /root/gpufree-data/outputs/bicycle-grow-20261030/error_points_bicycle.npy",
    # G8: sign test -- grow only in the INNER region (diagnostic says the error is there)
    "g8_inner": "--op grow_distonly --no-prune --grow-frac 0.2 --voxels 64 "
                "--under-mode none --dist-mode inneronly --far-frac 0.25 --far-weight 2",
}
# serial-only matrix on the shared box (GPU2; staged-loss owns GPU0).
# g9_holes is REMOVED: the alpha diagnostic found 0.0% pixels with alpha<0.5 in all six
# train views (mean alpha 0.9964) -> there is no hole cloud and no hole target at all.
# Its slot goes to g8_inner, the sign test implied by the diagnostic (error is inner).
SERIAL = ["control", "g1_hard05", "g2_ramp2", "g8_inner"]
G10 = ["control", "g10_errgrow"]
MATRIX = SERIAL + ["g3_p10_035", "g4_grow40", "g9_holes", "g10_errgrow"]


def main(kind="main"):
    man = {e["case"]: e for e in json.load(open(MANIFEST))}
    e = man[CASE]
    arms = {"main": MATRIX, "serial": SERIAL, "g10": G10}.get(kind)
    if arms is None:
        raise SystemExit(f"unknown cells kind {kind}")
    rows = []
    for arm in arms:
        tag = f"{SCENE}__{arm}" + ("" if kind == "main" else "__serial")
        rows.append("\t".join([tag, CASE, e["work"], e["init"], str(e["budget_seconds"]),
                               arm, ARMS[arm] or "-"]))
    print("\n".join(rows))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
