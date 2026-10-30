#!/usr/bin/env python3
"""Build records/minisplat-20261030/results.jsonl from the raw cell files."""
import glob
import json
import os
import sys

BUDGET = {"mipnerf360/bicycle": 95.0, "mipnerf360/counter": 92.57,
          "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce": 94.29}


def q(psnr, ssim, lpips):
    return (20.0 * max(0.0, min(1.0, (psnr - 15.0) / 25.0))
            + 20.0 * max(0.0, min(1.0, (ssim - 0.70) / 0.25))
            + 40.0 * max(0.0, min(1.0, (0.25 - lpips) / 0.20)))


def tp(b, budget):
    return 20.0 * max(0.0, min(1.0, (1.0 - b / budget) / 0.9))


def load(path):
    summary = json.load(open(os.path.join(path, "summary.json")))
    record = json.load(open(os.path.join(path, "train_record.json")))
    return summary, record


def rows_for(case, arm, path, ctrl_row=None):
    summary, record = load(path)
    budget = BUDGET.get(case, 94.29)
    out = []
    snaps = {s["iteration"]: s for s in record.get("snapshots", [])}
    for res in summary["results"]:
        b = res["b_seconds_estimate"]
        snap = snaps.get(res["iteration"], {})
        prune = snap.get("prune")
        qq = q(res["psnr"], res["ssim"], res["lpips"])
        tt = tp(b, budget)
        row = dict(case=case, arm=arm, iteration=res["iteration"],
                   stage=snap.get("stage") or ("stage1" if res["iteration"] == 5000 else "run"),
                   num_gaussians=res["num_gaussians"], b_seconds=b,
                   psnr=res["psnr"], ssim=res["ssim"], lpips=res["lpips"],
                   quality_points=round(qq, 4), time_points=round(tt, 4),
                   total_points=round(qq + tt, 4), budget_seconds=budget,
                   stage1_seconds=record.get("stage1_seconds"),
                   stage2_seconds=record.get("stage2_seconds"),
                   mid_prune_at=record.get("mid_prune_at"),
                   prune_keep=record.get("prune_keep"),
                   mid_prune_keep=record.get("mid_prune_keep"),
                   prune=prune, prune_events=record.get("prune_events"),
                   summary_json=os.path.join(path, "summary.json"),
                   clean=True)
        if ctrl_row is not None:
            row["ctrl_b_seconds"] = ctrl_row["b_seconds"]
            row["ctrl_quality_points"] = ctrl_row["quality_points"]
            row["d_quality"] = round(qq - ctrl_row["quality_points"], 4)
            row["d_b"] = round(b - ctrl_row["b_seconds"], 3)
            row["d_time"] = round(tt - tp(ctrl_row["b_seconds"], budget), 4)
            row["d_total"] = round(row["d_quality"] + row["d_time"], 4)
        out.append(row)
    return out


def main(root):
    cells = {}
    for d in glob.glob(os.path.join(root, "*", "*")):
        if os.path.isdir(os.path.join(d, "sweep")):
            tag = os.path.basename(os.path.dirname(d))
            arm = os.path.basename(d)
            cells[(tag, arm)] = d
    rows = []
    for (tag, arm), path in sorted(cells.items()):
        case = json.load(open(os.path.join(path, "sweep", "summary.json")))["case"]
        if arm == "ctrl":
            rows.extend(rows_for(case, arm, os.path.join(path, "sweep")))
        elif arm.startswith("m2_"):
            own = rows_for(case, arm, os.path.join(path, "sweep"))
            ctrl = own[0]
            for r in own:
                r.update(ctrl_b_seconds=ctrl["b_seconds"],
                         ctrl_quality_points=ctrl["quality_points"],
                         d_quality=round(r["quality_points"] - ctrl["quality_points"], 4),
                         d_b=round(r["b_seconds"] - ctrl["b_seconds"], 3),
                         d_time=round(r["time_points"] - ctrl["time_points"], 4),
                         d_total=round((r["quality_points"] - ctrl["quality_points"])
                                       + (r["time_points"] - ctrl["time_points"]), 4))
            rows.extend(own)
        elif arm.startswith("m1_"):
            own = rows_for(case, arm, os.path.join(path, "sweep"))
            ctrl_path = cells.get((tag, "ctrl"))
            if ctrl_path:
                ctrl = rows_for(case, "ctrl", os.path.join(ctrl_path, "sweep"))[-1]
                for r in own:
                    r.update(ctrl_b_seconds=ctrl["b_seconds"],
                             ctrl_quality_points=ctrl["quality_points"],
                             d_quality=round(r["quality_points"] - ctrl["quality_points"], 4),
                             d_b=round(r["b_seconds"] - ctrl["b_seconds"], 3),
                             d_time=round(r["time_points"] - ctrl["time_points"], 4),
                             d_total=round((r["quality_points"] - ctrl["quality_points"])
                                           + (r["time_points"] - ctrl["time_points"]), 4))
            rows.extend(own)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("wrote %d rows -> %s" % (len(rows), out))
    for r in rows:
        print("%-20s %-12s it=%-5s N=%-8d B=%-6.2f q=%7.3f dq=%s dB=%s dtotal=%s"
              % (r["case"].split("/")[-1][:20], r["arm"], r["iteration"],
                 r["num_gaussians"], r["b_seconds"], r["quality_points"],
                 r.get("d_quality"), r.get("d_b"), r.get("d_total")))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raw/matrix")
