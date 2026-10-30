#!/usr/bin/env python3
"""Summarise the N1 texture-voxel-growth cells into results.jsonl + a markdown table.

  python summarize.py [--cells DIR] [--out results.jsonl]

Reads, per cell:
  <cells>/<tag>/summary.json      run_sweep output (psnr/ssim/lpips, b_seconds_estimate, score)
  <ply-dir>/<tag>.json            ply_voxgrow transform summary (counts, prune/candidate stats)
and prints the per-arm 4-scene means plus the per-scene/per-arm table.

The proxy score is the organizer's linear /100: quality = 20*clip((psnr-15)/25) +
20*clip((ssim-0.70)/0.25) + 40*clip((0.25-lpips)/0.20); TIME = 20*min(1,(1-B/budget)/0.9);
total = quality + TIME.  B is module-B only (scene load + init + train), as in run_sweep.
"""
import argparse
import json
from pathlib import Path

SCENES = ["mip-bicycle", "dl-0f622ea", "hw-billboard", "mip-counter"]
ARMS = ["control", "td_flat", "tdm_flat", "do_flat"]


def load_cell(cells, plydir, tag):
    s = cells / tag / "summary.json"
    if not s.is_file():
        return None
    summary = json.loads(s.read_text())
    r = summary["results"][-1]
    parts = tag.split("__")
    row = dict(tag=tag, case=summary["case"], arm=parts[1],
               scene=parts[0], serial=len(parts) > 2,
               n_gaussians=r["num_gaussians"],
               train_seconds=r["train_seconds"], b_seconds=r["b_seconds_estimate"],
               budget=summary["budget_seconds"], psnr=r["psnr"], ssim=r["ssim"],
               lpips=r["lpips"], quality=r["score"]["quality_points"],
               time_points=r["score"]["time_points"], total=r["score"]["total_points"],
               valid=r["score"]["valid"])
    p = plydir / f"{tag}.json"
    if p.is_file():
        t = json.loads(p.read_text())
        for k, v in (t.get("landing") or {}).items():
            if isinstance(v, (int, float)):
                row[f"l_{k}"] = v
        for k in ("under_mode", "under_voxels", "occupied_voxels", "children_per_voxel_max",
                  "children_per_voxel_mean", "far_threshold", "prune_mode", "n_cameras"):
            if k in t:
                row[f"t_{k}"] = t[k]
        for k in ("n_in", "n_pruned", "pruned_frac", "n_after_prune", "n_out",
                  "net_growth", "pruned_opacity_ceiling", "far_threshold",
                  "far_gaussian_frac", "mean_occupancy", "under_voxels", "occupied_voxels",
                  "voxels", "merged", "seconds"):
            if k in t:
                row[f"t_{k}"] = t[k]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/gpufree-data/outputs/texvox-20261030")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    cells, plydir = root / "cells", root / "ply"
    rows = []
    for d in sorted(cells.iterdir()) if cells.is_dir() else []:
        if not d.is_dir():
            continue
        r = load_cell(cells, plydir, d.name)
        if r:
            rows.append(r)
    out = Path(args.out or root / "results.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(rows)} cells -> {out}\n")

    def fmt(rs):
        if not rs:
            return "  (none)"
        m = lambda k: sum(r[k] for r in rs if r[k] is not None) / len(rs)
        return (f"q {m('quality'):6.2f}  B {m('b_seconds'):6.1f}s  "
                f"TIME {m('time_points'):5.2f}  total {m('total'):6.2f}  n {m('n_gaussians')/1e3:6.1f}k")

    par = [r for r in rows if not r["serial"]]
    scenes = sorted({r["scene"] for r in par}) or SCENES
    arms = (["control"] if any(r["arm"] == "control" for r in par) else []) + \
        [a for a in sorted({r["arm"] for r in par}) if a != "control"]
    print("per-arm means over all scenes in the sweep (parallel runs; B inflated vs serial)")
    for arm in arms:
        rs = [r for r in par if r["arm"] == arm and r["scene"] in scenes]
        print(f"  {arm:12s} n={len(rs)}  {fmt(rs)}")

    serial = [r for r in rows if r["serial"]]
    if serial:
        print("\nSERIAL anchors (single card, other GPUs idle -> clean B/TIME)")
        for scene in scenes:
            ctl = next((r for r in serial if r["scene"] == scene and r["arm"] == "control"), None)
            for r in [x for x in serial if x["scene"] == scene]:
                d = ""
                if ctl and r is not ctl:
                    d = (f"  dq {r['quality']-ctl['quality']:+.2f}  dB {r['b_seconds']-ctl['b_seconds']:+.1f}s"
                         f"  dTIME {r['time_points']-ctl['time_points']:+.2f}"
                         f"  dtotal {r['total']-ctl['total']:+.2f}")
                print(f"  {scene:14s} {r['arm']:10s} q {r['quality']:6.2f} B {r['b_seconds']:6.1f}s "
                      f"TIME {r['time_points']:5.2f} total {r['total']:6.2f}{d}")

    print("\nper-scene table (parallel unless tag ends in __serial)")
    hdr = (f"{'scene':14s} {'arm':10s} {'n_in':>8s} {'pruned':>7s} {'n_out':>8s} "
           f"{'n_train':>8s} {'c>0.5R':>6s} {'c>0.75R':>7s} {'PSNR':>6s} {'SSIM':>6s} {'LPIPS':>6s} {'q':>6s} {'B':>6s} {'TIME':>5s} {'tot':>6s}")
    print(hdr)
    for scene in scenes:
        for arm in sorted({r["arm"] for r in rows if r["scene"] == scene}):
            r = next((x for x in rows if x["scene"] == scene and x["arm"] == arm), None)
            if not r:
                continue
            n_in = r.get("t_n_in", "")
            pr = f"{r['t_pruned_frac']*100:.1f}%" if "t_pruned_frac" in r else ""
            n_out = r.get("t_n_out", "")
            c5 = f"{r['l_share_children_gt0.5R']:.2f}" if 'l_share_children_gt0.5R' in r else ""
            c75 = f"{r['l_share_children_gt0.75R']:.2f}" if 'l_share_children_gt0.75R' in r else ""
            print(f"{scene:14s} {arm:10s} {str(n_in):>8s} {pr:>7s} {str(n_out):>8s} "
                  f"{r['n_gaussians']:>8d} {c5:>6s} {c75:>7s} "
                  f"{r['psnr']:6.2f} {r['ssim']:6.3f} {r['lpips']:6.3f} {r['quality']:6.2f} "
                  f"{r['b_seconds']:6.1f} {r['time_points']:5.2f} {r['total']:6.2f}")


if __name__ == "__main__":
    main()
