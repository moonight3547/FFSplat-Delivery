import json
from pathlib import Path
OUT = Path("/root/gpufree-data/outputs/bigd-combo-20261030/matrix")
BUDGET = {"mipnerf360/bicycle": 95.0,
          "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce": 94.29}

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
        b = res["b_seconds_estimate"]
        snaps = rec.get("snapshots") or []
        snap_train = snaps[0].get("train_seconds") if snaps else None
        dev = rec.get("densification_events") or []
        prune = rec.get("prune_events") or []
        rows.append(dict(case=case, scene=case_dir.name, arm=arm_dir.name,
                         n=res["num_gaussians"], psnr=res["psnr"], ssim=res["ssim"],
                         lpips=res["lpips"], q=round(res["score"]["quality_points"], 5),
                         b=round(b, 4), time=round(res["score"]["time_points"], 5),
                         total=round(res["score"]["total_points"], 5), budget=budget,
                         scene_load=round(rec.get("scene_load_seconds", 0.0), 4),
                         init=round(rec.get("init_seconds", 0.0), 4),
                         record_train=round(rec.get("train_seconds", 0.0), 3),
                         snap_train=round(snap_train, 3) if snap_train else None,
                         densify_events=len(dev),
                         densify_thresholds=sorted({(e.get("actual_native_parameters") or {}).get("grad_abs_thresh") for e in dev}),
                         prune_events=[{k: e.get(k) for k in ("n_before", "n_after", "keep_requested", "seconds")} for e in prune],
                         scene_cache=bool((rec.get("eng") or {}).get("scene_cache"))))
rows.sort(key=lambda r: (r["scene"], r["arm"]))
json.dump(rows, open("/root/gpufree-data/outputs/bigd-combo-20261030/results.json", "w"), indent=2)
for tag in sorted({r["scene"] for r in rows}):
    group = [r for r in rows if r["scene"] == tag]
    base_bigd = next((r for r in group if r["arm"] == "bigd"), None)
    base_del = next((r for r in group if r["arm"] == "delivery"), None)
    for r in group:
        dq = round(r["q"] - base_bigd["q"], 3) if base_bigd else None
        db = round(r["b"] - base_bigd["b"], 2) if base_bigd else None
        dt = round(r["total"] - base_bigd["total"], 3) if base_bigd else None
        dq_del = round(r["q"] - base_del["q"], 3) if base_del else None
        dt_del = round(r["total"] - base_del["total"], 3) if base_del else None
        print("%-8s %-15s N=%-8d dev=%-2d thr=%-6s q=%7.3f B=%6.2f TIME=%6.3f total=%7.3f "
              "load=%5.2f init=%4.2f snap_train=%6.2f cache=%-1s | vs bigd dq=%s dB=%s dT=%s | vs deliv dq=%s dT=%s"
              % (r["scene"], r["arm"], r["n"], r["densify_events"],
                 ",".join(str(x) for x in r["densify_thresholds"]), r["q"], r["b"],
                 r["time"], r["total"], r["scene_load"], r["init"], r["snap_train"] or 0,
                 int(r["scene_cache"]), dq, db, dt, dq_del, dt_del))
