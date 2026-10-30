import json
from pathlib import Path
OUT = Path("/root/gpufree-data/outputs/d4000-13-20261030")
MAT = OUT / "matrix"
CLEAN = {  # records/clean13-20261030/results.jsonl matched baseline (delivery, no gate)
 "hw/billboard": 53.67290926435828, "hw/basketball": 67.62046305098875,
 "hw/car": 48.53559196559025, "hw/trashbin": 64.97650419764423,
 "mipnerf360/bicycle": 25.31011456233717, "mipnerf360/bonsai": 63.91858646481592,
 "mipnerf360/counter": 55.17158984097854,
 "dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f": 78.94231991986493,
 "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce": 31.10574259810206,
 "dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e": 31.552958455583433,
 "dl3dv/2dad2d298927ff9b2ed7daa6128244e81d331e8cf1abf6c8598f35276ad38b84": 49.75791395079112,
 "dl3dv/3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3": 35.03899809710881,
 "dl3dv/9a451604b8d2583d47b14662635d8f74710c419cc061705ac15b8cfc31ece446": 59.37744700699919,
}
def q_of(psnr, ssim, lpips):
    return (20*max(0,min(1,(psnr-15)/25)) + 20*max(0,min(1,(ssim-0.70)/0.25))
            + 40*max(0,min(1,(0.25-lpips)/0.20)))
def tp(b, budget):
    return 20*max(0,min(1,(1-b/budget)/0.9)) if b < budget else 0.0
rows = []
for cell in sorted(p for p in MAT.iterdir() if p.is_dir()):
    summ = cell/"sweep"/"summary.json"; recp = cell/"sweep"/"train_record.json"
    if not summ.exists() or not recp.exists(): continue
    s = json.load(open(summ)); r = json.load(open(recp))
    res = s["results"][-1]; case = s["case"]
    cj = json.load(open(cell/"cell.json")) if (cell/"cell.json").exists() else {}
    g = json.load(open(cell/"gate.json")) if (cell/"gate.json").exists() else {}
    snaps = r.get("snapshots") or []
    b = res["b_seconds_estimate"]; q = q_of(res["psnr"], res["ssim"], res["lpips"])
    budget = float(cj.get("budget", 95.0))
    rows.append(dict(family="d4000-13-20261030", case=case, tag=cell.name,
                     gate_train_lpips=g.get("train_lpips"), gate_keep=cj.get("keep", 1.0),
                     n=res["num_gaussians"], psnr=res["psnr"], ssim=res["ssim"],
                     lpips=res["lpips"], q=round(q,5),
                     b=round(b,4), time=round(tp(b,budget),5), total=round(q+tp(b,budget),5),
                     budget=budget,
                     scene_load=round(r["scene_load_seconds"],4),
                     init=round(r["init_seconds"],4),
                     snap_train=round(snaps[0]["train_seconds"],3) if snaps else None,
                     densify_events=len(r.get("densification_events") or []),
                     densify_thresholds=sorted({e["actual_native_parameters"]["grad_abs_thresh"]
                                                for e in (r.get("densification_events") or [])}),
                     prune_events=r.get("prune_events") or [],
                     scene_cache=bool((r.get("eng") or {}).get("scene_cache")),
                     cache_build_seconds=(cj.get("cache") or {}).get("build_seconds_a_side"),
                     clean13_total=CLEAN.get(case),
                     d_total_vs_clean13=round(res["score"]["total_points"]-CLEAN[case],4)
                     if case in CLEAN else None))
with open(OUT/"results.jsonl","w") as fh:
    for x in rows: fh.write(json.dumps(x)+"\n")
if rows:
    mean = sum(x["total"] for x in rows)/len(rows)
    mean_d = sum(x["d_total_vs_clean13"] for x in rows if x["d_total_vs_clean13"] is not None)/len([x for x in rows if x["d_total_vs_clean13"] is not None])
    print("n=%d/%d mean_total=%.3f mean_delta_vs_clean13=%+.3f" % (len(rows), len(CLEAN), mean, mean_d))
print(json.dumps(sorted(rows, key=lambda x: -x["d_total_vs_clean13"] or 0)[:3], indent=0)[:500])
