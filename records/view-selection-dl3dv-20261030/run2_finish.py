#!/usr/bin/env python3
"""Finish task-14 after H1 is free: (a) triage the extreme-baseline-v2 candidates,
(b) final held-out ranking, (c) 6 delivery e2e runs (uniform control + best-by-held-out
per bad case), serial and quiet-gated. Reuses helpers from /root/run2.py.
"""
import json, os, sys, time
sys.path.insert(0, "/root")
import run2 as R  # helpers: B, LOGS, wait_quiet, resplat_triage, delivery_e2e, say, load

B = R.B
BAD = R.BAD

man = json.load(open(os.path.join(B, "candidates.json")))
scenes = {s["scene"]: s for s in man["scenes"]}
v2 = []
v2file = os.path.join(B, "results-stage1-v2.json")
if os.path.isfile(v2file):
    v2 = json.load(open(v2file)); R.say("RESUME v2 rows=%d" % len(v2))
else:
    if not R.wait_quiet("phaseA-v2-start"):
        raise SystemExit(1)
    for scene in BAD:
        sel = json.load(open(os.path.join(B, "selection-v2", scene + ".json")))
        sc = scenes[scene]
        c = dict(scene=scene, method="extreme-baseline-v2", case=sc["case"], budget=sc["budget"],
                 work=os.path.relpath(os.path.join(B, "work", scene, "extreme-baseline-v2"), R.REPO),
                 control_total=sc["control_total"], control_q=sc["control_q"],
                 control_ctx_psnr=sc["control_ctx_psnr"], n_context=16,
                 heldout_names=sel["heldout_names"],
                 context_names=sel["methods"]["extreme-baseline"]["names"], sel_stats={})
        v2.append(R.resplat_triage(c, "%s-extreme-baseline-v2" % scene))
        json.dump(v2, open(v2file, "w"), indent=2)

rows = json.load(open(os.path.join(B, "results-stage1.json"))) + v2

# ---- final held-out ranking (LPIPS asc, PSNR desc; adoption needs both beats) ----
by = {}
for r in rows:
    if r.get("holdout"):
        by.setdefault(r["scene"], {})[r["method"]] = r
ranked = {}
for scene, m in by.items():
    u = m["uniform"]; uh = u["holdout"]
    table = []
    for meth, r in m.items():
        if meth == "uniform":
            continue
        h = r["holdout"]
        beats = bool(h["lpips"] < uh["lpips"] and h["psnr"] > uh["psnr"])
        table.append(dict(method=meth, holdout_psnr=h["psnr"], holdout_ssim=h["ssim"],
                          holdout_lpips=h["lpips"], ctx_psnr=r.get("ctx_psnr_mean"),
                          num_gaussians=r.get("num_gaussians"),
                          beats_uniform_on_heldout=beats))
    table.sort(key=lambda d: (d["holdout_lpips"], -d["holdout_psnr"]))
    table.append(dict(method="uniform", holdout_psnr=uh["psnr"], holdout_ssim=uh["ssim"],
                      holdout_lpips=uh["lpips"], ctx_psnr=u.get("ctx_psnr_mean"),
                      num_gaussians=u.get("num_gaussians"), beats_uniform_on_heldout=False))
    best = table[0]
    ranked[scene] = dict(uniform_psnr=uh["psnr"], uniform_lpips=uh["lpips"],
                         best_method=best["method"],
                         best_beats_uniform_on_heldout=best["beats_uniform_on_heldout"],
                         table=table)
    R.say("RANK %s uniform_hl_lpips=%.4f best=%s beats=%s" %
          (scene, uh["lpips"], best["method"], best["beats_uniform_on_heldout"]))
json.dump(ranked, open(os.path.join(B, "heldout-metrics.json"), "w"), indent=2)

# ---- Phase C: e2e = uniform control + best-by-held-out per bad case ----
e2efile = os.path.join(B, "results-e2e.json")
e2e = json.load(open(e2efile)) if os.path.isfile(e2efile) else []
done = {(r["scene"], r["method"]) for r in e2e}
for scene in BAD:
    best = ranked[scene]["best_method"]
    for meth in ["uniform"] + ([best] if best and best != "uniform" else []):
        if (scene, meth) in done:
            continue
        c = None
        if meth == "extreme-baseline-v2":
            c = [x for x in v2 if x["scene"] == scene][0]
        else:
            c = [x for x in man["candidates"] if x["scene"] == scene and x["method"] == meth][0]
        if not R.wait_quiet("e2e-%s-%s" % (scene, meth)):
            raise SystemExit(1)
        e2e.append(R.delivery_e2e(c, "%s-%s" % (scene, meth)))
        json.dump(e2e, open(e2efile, "w"), indent=2)

# ---- summary ----
bad_deltas = [r["delta_q_vs_control"] for r in e2e
              if r.get("delta_q_vs_control") is not None and r["method"] != "uniform"]
summary = dict(ranked=ranked, stage1_rows=len(rows), v2_rows=len(v2), e2e=e2e)
summary["projected_13case_gain_bad3"] = sum(bad_deltas) / 13.0 if bad_deltas else None
json.dump(summary, open(os.path.join(B, "results.json"), "w"), indent=2)
with open(os.path.join(B, "results.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
    for r in e2e:
        f.write(json.dumps(dict(r, stage="e2e")) + "\n")
R.say("FINISH rows=%d e2e=%d projected_gain=%s" %
      (len(rows), len(e2e), summary["projected_13case_gain_bad3"]))
