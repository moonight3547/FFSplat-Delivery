#!/usr/bin/env python3
"""Aggregate the staged-loss matrix (task-8) into the report tables.

Reads results.jsonl (one row per cell/stage, written by run_staged.py) and
prints:
  * per-cell table: n_gaussians, stage1/2/3 seconds, B, PSNR/SSIM/LPIPS,
    quality / TIME / total proxy points
  * per-scene paired delta vs the cell's own phase-1 row (ctrl)
  * 4-scene means per arm/stage, and quality gained per extra second of B
The stage label is normalised here because early driver builds dropped the
snapshot list from summary.json (iterations still order the stages).
"""
import json
import sys
from collections import defaultdict

BUDGET = {}


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalise(rows):
    """Assign stage1/stage2/stage3 within each (case, arm, out_dir) group."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["arm"], row.get("out_dir"))].append(row)
    for key, grp in groups.items():
        if all(r.get("stage") in ("stage1", "stage2", "stage3") for r in grp):
            continue
        ordered = sorted(grp, key=lambda r: r.get("iteration") or 0)
        for idx, row in enumerate(ordered):
            row["stage"] = ("stage1", "stage2", "stage3")[idx]
    return rows


def main(path):
    rows = normalise(load(path))
    cases = sorted({r["case"] for r in rows})
    arms = [a for a in ("absgrad", "absgrad_vgg", "mcmc", "mcmc_vgg",
                        "standard", "ctrl") if any(r["arm"] == a for r in rows)]

    print("=== per cell (serial N1 measurement; B = module-B seconds incl. every stage) ===")
    head = ("%-22s %-12s %-6s %8s %6s %6s %6s %8s %7s %7s %7s %7s %7s"
            % ("case", "arm", "stage", "N", "s1", "s2", "s3",
               "B", "psnr", "ssim", "lpips", "q", "total"))
    print(head)
    for case in cases:
        for arm in arms:
            for row in sorted([r for r in rows if r["case"] == case and r["arm"] == arm],
                              key=lambda r: r.get("iteration") or 0):
                sc = row.get("score") or {}
                print("%-22s %-12s %-6s %8d %6.1f %6.1f %6.1f %8.2f %7.3f %7.4f "
                      "%7.4f %7.2f %7.2f" % (
                          case.split("/")[-1][:22], arm, row["stage"],
                          row.get("num_gaussians") or 0,
                          row.get("stage1_seconds") or 0.0,
                          row.get("stage2_seconds") or 0.0,
                          row.get("stage3_seconds") or 0.0,
                          row.get("b_seconds") or 0.0, row.get("psnr") or 0.0,
                          row.get("ssim") or 0.0, row.get("lpips") or 0.0,
                          sc.get("quality_points") or 0.0,
                          sc.get("total_points") or 0.0))

    print("\n=== paired delta vs the same cell's phase-1 (ctrl) row ===")
    print("%-22s %-20s %9s %9s %9s %8s %8s"
          % ("case", "arm/stage", "ctrl_total", "arm_total", "d_total", "d_q", "d_B"))
    deltas = defaultdict(list)
    q_deltas = defaultdict(list)
    b_deltas = defaultdict(list)
    extra_seconds = defaultdict(list)
    for case in cases:
        for arm in arms:
            grp = [r for r in rows if r["case"] == case and r["arm"] == arm]
            ctrl = next((r for r in grp if r["stage"] == "stage1"), None)
            if not ctrl:
                continue
            for row in [r for r in grp if r["stage"] != "stage1"]:
                ct = (ctrl.get("score") or {}).get("total_points") or 0.0
                at = (row.get("score") or {}).get("total_points") or 0.0
                cq = (ctrl.get("score") or {}).get("quality_points") or 0.0
                aq = (row.get("score") or {}).get("quality_points") or 0.0
                db = (row.get("b_seconds") or 0.0) - (ctrl.get("b_seconds") or 0.0)
                print("%-22s %-20s %9.3f %9.3f %+9.3f %+8.3f %+8.2f" % (
                    case.split("/")[-1][:22], arm + "/" + row["stage"],
                    ct, at, at - ct, aq - cq, db))
                key = arm + "/" + row["stage"]
                deltas[key].append(at - ct)
                q_deltas[key].append(aq - cq)
                b_deltas[key].append(db)
                extra_seconds[key].append(db)

    print("\n=== means over the scenes present (paired) ===")
    print("%-22s %5s %11s %11s %11s %12s" % (
        "arm/stage", "n", "d_total", "d_quality", "d_B", "d_q / extra_s"))
    for key in sorted(deltas):
        n = len(deltas[key])
        dt = sum(deltas[key]) / n
        dq = sum(q_deltas[key]) / n
        db = sum(b_deltas[key]) / n
        ratio = (dq / db) if abs(db) > 1e-9 else float("nan")
        print("%-22s %5d %+11.3f %+11.3f %+11.2f %12.3f" % (
            key, n, dt, dq, db, ratio))

    print("\n=== extra B + score breakdown per arm (means) ===")
    for key in sorted(deltas):
        print("%-22s mean_B_ctrl=%.2f mean_d_B=%+.2f -> TIME %.2f points at risk"
              % (key, sum(b_deltas[key]) / len(b_deltas[key]),
                 sum(b_deltas[key]) / len(b_deltas[key]),
                 -20.0 * (sum(b_deltas[key]) / len(b_deltas[key])) / (0.9 * 94.29)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")
