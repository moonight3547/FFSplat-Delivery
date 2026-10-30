import json
from pathlib import Path
OUT = Path("/root/gpufree-data/outputs/d3500-combo-20261030/matrix")
BUDGET = {"mipnerf360/bicycle": 95.0,
          "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce": 94.29}
PREV_BEST = {"mipnerf360/bicycle": 28.127,
             "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce": 36.615}  # task-19 best (bigd_m1_cache / bigd_cache)

def q_of(psnr, ssim, lpips):
    return (20*max(0,min(1,(psnr-15)/25)) + 20*max(0,min(1,(ssim-0.70)/0.25))
            + 40*max(0,min(1,(0.25-lpips)/0.20)))

def tp(b, budget):
    return 20*max(0,min(1,(1-b/budget)/0.9))

rows = []
for case_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
    for arm_dir in sorted(p for p in case_dir.iterdir() if p.is_dir()):
        summ = arm_dir / "sweep" / "summary.json"; rec_p = arm_dir / "sweep" / "train_record.json"
        if not summ.exists() or not rec_p.exists(): continue
        d = json.load(open(summ)); rec = json.load(open(rec_p))
        res = d["results"][-1]; case = d["case"]; budget = BUDGET[case]
        b = res["b_seconds_estimate"]; q = q_of(res["psnr"], res["ssim"], res["lpips"])
        snaps = rec.get("snapshots") or []
        dev = rec.get("densification_events") or []
        rows.append(dict(case=case, scene=case_dir.name, arm=arm_dir.name,
                         n=res["num_gaussians"], psnr=res["psnr"], ssim=res["ssim"],
                         lpips=res["lpips"], q=round(q, 5), b=round(b, 4),
                         time=round(tp(b, budget), 5), total=round(q + tp(b, budget), 5),
                         budget=budget,
                         scene_load=round(rec["scene_load_seconds"], 4),
                         init=round(rec["init_seconds"], 4),
                         snap_train=round(snaps[0]["train_seconds"], 3) if snaps else None,
                         record_train=round(rec["train_seconds"], 3),
                         densify_events=len(dev),
                         densify_thresholds=sorted({e["actual_native_parameters"]["grad_abs_thresh"]
                                                    for e in dev}),
                         prune_events=rec.get("prune_events") or [],
                         scene_cache=bool((rec.get("eng") or {}).get("scene_cache"))))
rows.sort(key=lambda r: (r["scene"], r["arm"]))
for scene in BUDGET:
    grp = [x for x in rows if x["case"] == scene]
    base = next(x for x in grp if x["arm"] == "d3500")
    for x in grp:
        x["d_q_vs_d3500"] = round(x["q"] - base["q"], 4)
        x["d_b_vs_d3500"] = round(x["b"] - base["b"], 3)
        x["d_time_vs_d3500"] = round(x["time"] - base["time"], 4)
        x["d_total_vs_d3500"] = round(x["total"] - base["total"], 4)
        x["prev_task19_best_total"] = PREV_BEST[scene]
        x["d_total_vs_task19_best"] = round(x["total"] - PREV_BEST[scene], 4)
json.dump(rows, open("/root/gpufree-data/outputs/d3500-combo-20261030/results.json", "w"), indent=2)
for x in rows:
    print("%-8s %-15s N=%-8d dev=%-2d thr=%-6s q=%7.3f B=%6.2f TIME=%6.3f total=%7.3f "
          "load=%5.2f init=%4.2f snap=%6.2f cache=%-1s | vs d3500 dq=%+.3f dB=%+.2f dT=%+.3f | vs task19 %+.3f"
          % (x["scene"], x["arm"], x["n"], x["densify_events"],
             ",".join(str(t) for t in x["densify_thresholds"]), x["q"], x["b"],
             x["time"], x["total"], x["scene_load"], x["init"], x["snap_train"] or 0,
             int(x["scene_cache"]), x["d_q_vs_d3500"], x["d_b_vs_d3500"],
             x["d_total_vs_d3500"], x["d_total_vs_task19_best"]))
