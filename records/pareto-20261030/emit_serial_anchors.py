#!/usr/bin/env python3
"""Emit the raw serial-anchor JSONL requested by the Lead/analyst.

  python3 emit_serial_anchors.py --serial <pareto-20261030-serial/results.jsonl> \
      --concurrent <pareto-20261030/results.jsonl> \
      --out serial-anchors.jsonl

One line per serial measurement:
  {"case", "S", "b_seconds", "b_definition", "lane": "serial", "reused_out",
   "b_concurrent", "inflation_ratio", "train_seconds", "psnr", "ssim", "lpips",
   "quality_points", "time_points", "total_points", "summary_json"}
b_definition is run_sweep's b_seconds_estimate = scene_load + init + snap_train + 0.5
(same definition as the 13-case reference table; PLY export excluded).
"""
import argparse
import json

B_DEF = "scene_load+init+train+0.5"


def load(path):
    rows = {}
    if not path:
        return rows
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[(r["case"], r["S"])] = r
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--serial", required=True)
    p.add_argument("--concurrent", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ser = load(args.serial)
    con = load(args.concurrent)
    n = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for (case, S) in sorted(ser):
            r = ser[(case, S)]
            c = con.get((case, S)) or {}
            bs = r.get("b_seconds_estimate")
            bc = c.get("b_seconds_estimate")
            fh.write(json.dumps(dict(
                case=case, S=S,
                b_seconds=bs, b_definition=B_DEF, lane="serial",
                reused_out=r.get("out_dir"),
                b_concurrent=bc,
                inflation_ratio=(round(bc / bs, 4) if bs and bc else None),
                train_seconds=r.get("train_seconds"),
                scene_load_seconds=r.get("scene_load_seconds"),
                init_seconds=r.get("init_seconds"),
                psnr=r.get("psnr"), ssim=r.get("ssim"), lpips=r.get("lpips"),
                quality_points=r.get("quality_points"),
                time_points=r.get("time_points"),
                total_points=r.get("total_points"),
                summary_json=r.get("summary_json"),
            )) + "\n")
            n += 1
    print("wrote %d anchors -> %s" % (n, args.out))


if __name__ == "__main__":
    main()
