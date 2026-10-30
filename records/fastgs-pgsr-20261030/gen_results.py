#!/usr/bin/env python3
"""Build fastgs-pgsr-20261030/results.jsonl from the A/B matrix + gate records."""
import glob, json, os, sys
BUDGET = {"mipnerf360/bicycle": 95.0, "mipnerf360/counter": 92.57}

def q(psnr, ssim, lpips):
    return (20*max(0,min(1,(psnr-15)/25)) + 20*max(0,min(1,(ssim-0.70)/0.25))
            + 40*max(0,min(1,(0.25-lpips)/0.20)))

def tp(b, budget):
    return 20*max(0,min(1,(1-b/budget)/0.9))

def main(root):
    rows = []
    ctrls = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "*", "sweep", "summary.json"))):
        case_dir = f.split(os.sep)[-4]
        arm = f.split(os.sep)[-3]
        case = json.load(open(f))["case"]
        budget = BUDGET.get(case, 94.29)
        d = json.load(open(f))
        rec = json.load(open(os.path.join(os.path.dirname(f), "train_record.json")))
        for res in d["results"]:
            b = res["b_seconds_estimate"]
            qq, tt = q(res["psnr"], res["ssim"], res["lpips"]), tp(b, budget)
            row = dict(case=case, arm=arm, iteration=res["iteration"],
                       num_gaussians=res["num_gaussians"], b_seconds=b,
                       psnr=res["psnr"], ssim=res["ssim"], lpips=res["lpips"],
                       quality_points=round(qq, 4), time_points=round(tt, 4),
                       total_points=round(qq + tt, 4), budget_seconds=budget,
                       stage1_seconds=rec.get("stage1_seconds"),
                       pgsr=rec.get("pgsr"), summary_json=f, clean=True)
            if arm == "ctrl":
                ctrls[case] = row
            rows.append(row)
    for row in rows:
        ctrl = ctrls.get(row["case"])
        if ctrl and row["arm"] != "ctrl":
            row["ctrl_b_seconds"] = ctrl["b_seconds"]
            row["ctrl_quality_points"] = ctrl["quality_points"]
            row["d_quality"] = round(row["quality_points"] - ctrl["quality_points"], 4)
            row["d_b"] = round(row["b_seconds"] - ctrl["b_seconds"], 3)
            row["d_time"] = round(row["time_points"] - ctrl["time_points"], 4)
            row["d_total"] = round(row["d_quality"] + row["d_time"], 4)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")
    with open(out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print("wrote %d rows -> %s" % (len(rows), out))
    for row in rows:
        print("%-20s %-12s N=%-8d B=%-6.2f q=%7.3f dq=%s dB=%s dtotal=%s"
              % (row["case"].split("/")[-1][:20], row["arm"], row["num_gaussians"],
                 row["b_seconds"], row["quality_points"], row.get("d_quality"),
                 row.get("d_b"), row.get("d_total")))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raw/matrix")
