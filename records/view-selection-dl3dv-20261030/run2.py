#!/usr/bin/env python3
"""DL3DV bad-case view-selection study: held-out-train triage + delivery e2e (H1, serial).

Phase A: for 4 scenes x candidates — rewrite context, ReSplat, ctx self-render PSNR,
         held-out 8-train-view render (fastgs) + official PSNR/SSIM/VGG-LPIPS (evaluate.py).
Phase B: 12-frame variant of the best-ranked method on the 3 bad cases.
Phase C: delivery e2e (S=5000 recipe) for uniform control + best-by-held-out per bad case.
Ranking uses the HELD-OUT train metrics (task-11 lesson: ctx self-render is misleading).
"""
import json, os, subprocess, time

REPO = "/workspace/organizer-projects/FFSplat-Hackathon"
B = os.path.join(REPO, "outputs/viewsel-dl3dv-20261030")
CKPT = "/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth"
CONDA = "/usr/local/miniconda3/bin/conda"
LOGS = os.path.join(B, "logs"); os.makedirs(LOGS, exist_ok=True)
LOG = os.path.join(LOGS, "run.log")
BAD = ["dl3dv-0b5635d5", "dl3dv-0f622eaa", "dl3dv-3e4dbe41"]

def say(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
    print(line, flush=True); open(LOG, "a").write(line + "\n")

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=REPO)

def state():
    try: u = int(sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1").stdout.strip())
    except Exception: u = 99999
    try: load1 = float(open("/proc/loadavg").read().split()[0])
    except Exception: load1 = 99.0
    others = sh("ps -eo args= | grep -E '^python -m research\\.(run_sweep|resplat_variant|train_sweep|render_eval)' | grep -v grep").stdout.strip()
    colmap = sh("pgrep -x colmap").stdout.strip()
    return u, load1, others, colmap

def wait_quiet(tag, max_min=25):
    t0 = time.time(); n = 0; waited = 0
    while True:
        u, load1, others, colmap = state()
        ok = (u < 50) and (load1 < 4.0) and (not others) and (not colmap)
        n = n + 1 if ok else 0
        if n >= 2:
            say("QUIET(%s) gpu=%dMiB load1=%.2f waited=%ds" % (tag, u, load1, waited)); return True
        if waited % 120 == 0: say("WAIT-QUIET(%s) gpu=%dMiB load1=%.2f" % (tag, u, load1))
        if waited >= max_min * 60: say("QUIET-TIMEOUT(%s)" % tag); return False
        time.sleep(20); waited += 20

def run(cmd, logpath):
    with open(logpath, "w") as f:
        return subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=REPO).returncode

def load(p):
    try: return json.load(open(p))
    except Exception: return None

def heldout_metrics(scene, method, ply, work, views):
    hd = os.path.join(B, "holdout", scene, method, "init")
    rc = run("%s run --no-capture-output -n organizer-fastgs python research/holdout_render.py "
             "--work %s --ply %s --views %s --out %s --quiet"
             % (CONDA, work, ply, ",".join(views), hd), os.path.join(LOGS, "holdout-%s-%s.log" % (scene, method)))
    m = None
    if rc == 0:
        run("%s run --no-capture-output -n ffsplat-enhance python competition/upstream/baseline/evaluate.py "
            "--model_path %s --device cuda:0" % (CONDA, hd), os.path.join(LOGS, "eval-%s-%s.log" % (scene, method)))
        m = load(os.path.join(hd, "metrics.json"))
    return rc, (m or {}).get("mean"), (m or {}).get("per_view")

def resplat_triage(c, tag):
    rdir = os.path.join(B, "resplat", c["scene"], c["method"])
    rc = run("%s run --no-capture-output -n ffsplat-enhance python -m research.resplat_variant "
             "--work %s --out %s --checkpoint %s --num_refine 2 --device cuda:0 --fix-sh-basis"
             % (CONDA, c["work"], rdir, CKPT), os.path.join(LOGS, "resplat-%s.log" % tag))
    row = dict(c, resplat_rc=rc, resplat_out=os.path.relpath(rdir, REPO))
    rj = load(os.path.join(rdir, "resplat.json"))
    if rj: row.update(num_gaussians=rj.get("num_gaussians"), resplat_seconds=rj.get("total_seconds"))
    ply = os.path.join(rdir, "resplat_world.ply")
    cp = os.path.join(LOGS, "ctxpsnr-%s.json" % tag)
    if os.path.isfile(ply):
        run("%s run --no-capture-output -n organizer-fastgs python %s %s %s > %s"
            % (CONDA, os.path.join(B, "ctx_psnr.py"), rdir, c["work"], cp),
            os.path.join(LOGS, "ctxpsnr-%s.err" % tag))
        cj = load(cp) or {}
        row.update(ctx_psnr_mean=cj.get("ctx_psnr_mean"), ctx_render_n=cj.get("n_render"),
                   ctx_matched_n=cj.get("n_matched"))
        rc2, mean, per = heldout_metrics(c["scene"], c["method"], ply, c["work"], c["heldout_names"])
        row.update(holdout_rc=rc2, holdout=mean, holdout_per_view=per)
        say("TRIAGE %s gauss=%s ctx=%s holdout_psnr=%s holdout_lpips=%s" %
            (tag, row.get("num_gaussians"), row.get("ctx_psnr_mean"),
             (mean or {}).get("psnr"), (mean or {}).get("lpips")))
    else:
        say("RESPLAT-FAIL %s rc=%d" % (tag, rc))
    return row

def delivery_e2e(c, tag):
    outdir = os.path.join(B, "runs", c["scene"], c["method"])
    rdir = os.path.join(B, "resplat", c["scene"], c["method"])
    cmd = ("%s run --no-capture-output -n organizer-fastgs python -m research.run_sweep "
           "--case '%s' --work %s --init-ply %s/resplat_world.ply --out %s "
           "--iterations 5000 --snapshots '' --budget %s --position-lr-max-steps 5000 "
           "--opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 "
           "--densify-from 500 --densify-until 1250 --realloc-frac 0.15 --realloc-at 1250 "
           "--image-cache %s/image_cache"
           % (CONDA, c["case"], c["work"], rdir, outdir, c["budget"], c["work"]))
    rc = run(cmd, os.path.join(LOGS, "e2e-%s.log" % tag))
    row = dict(scene=c["scene"], method=c["method"], case=c["case"], e2e_rc=rc,
               control_total=c["control_total"], control_q=c["control_q"])
    sj = load(os.path.join(outdir, "summary.json"))
    if sj:
        s = sj["results"][0]; sc = s["score"]
        row.update(psnr=s["psnr"], ssim=s["ssim"], lpips=s["lpips"],
                   quality_points=sc["quality_points"], b_seconds=s["b_seconds_estimate"],
                   time_points=sc["time_points"], total_points=sc["total_points"],
                   delta_q_vs_control=sc["quality_points"] - c["control_q"],
                   delta_total_vs_control=sc["total_points"] - c["control_total"])
    say("E2E %s rc=%d q=%s dq=%s B=%s total=%s dt=%s" %
        (tag, rc, row.get("quality_points"), row.get("delta_q_vs_control"), row.get("b_seconds"),
         row.get("total_points"), row.get("delta_total_vs_control")))
    return row

def main():
    man = json.load(open(os.path.join(B, "candidates.json")))
    say("=== DL3DV view-selection start: %d candidates over %d scenes" %
        (len(man["candidates"]), len(man["scenes"])))
    if not wait_quiet("phaseA-start"):
        raise SystemExit(1)
    rows = []
    for c in man["candidates"]:
        tag = "%s-%s" % (c["scene"], c["method"])
        rows.append(resplat_triage(c, tag))
        json.dump(rows, open(os.path.join(B, "results-stage1.json"), "w"), indent=2)
        with open(os.path.join(B, "results.jsonl"), "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")

    # ranking on held-out metrics
    by = {}
    for r in rows:
        if r.get("holdout"): by.setdefault(r["scene"], {})[r["method"]] = r
    ranked = {}
    for scene, m in by.items():
        u = m.get("uniform")
        cands = []
        for meth, r in m.items():
            if meth == "uniform": continue
            h = r["holdout"]
            beats = bool(u and h["lpips"] < u["holdout"]["lpips"] and h["psnr"] > u["holdout"]["psnr"])
            cands.append((h["lpips"], -h["psnr"], meth, beats))
        cands.sort()
        best = cands[0] if cands else None
        ranked[scene] = dict(uniform_lpips=u["holdout"]["lpips"] if u else None,
                             uniform_psnr=u["holdout"]["psnr"] if u else None,
                             best_method=best[2] if best else None,
                             best_beats_uniform_on_heldout=best[3] if best else None,
                             table=[dict(method=meth, holdout_psnr=m["holdout"]["psnr"],
                                         holdout_lpips=m["holdout"]["lpips"],
                                         holdout_ssim=m["holdout"]["ssim"],
                                         ctx_psnr=m.get("ctx_psnr_mean"),
                                         num_gaussians=m.get("num_gaussians"),
                                         beats_uniform_on_heldout=beats)
                                    for _, _, meth, beats in sorted(cands)]
                             + [dict(method="uniform", holdout_psnr=u["holdout"]["psnr"],
                                     holdout_lpips=u["holdout"]["lpips"],
                                     holdout_ssim=u["holdout"]["ssim"],
                                     ctx_psnr=u.get("ctx_psnr_mean"),
                                     num_gaussians=u.get("num_gaussians"),
                                     beats_uniform_on_heldout=False)])
        say("RANK %s uniform_hl_lpips=%.4f best=%s beats=%s" %
            (scene, ranked[scene]["uniform_lpips"] or -1, ranked[scene]["best_method"],
             ranked[scene]["best_beats_uniform_on_heldout"]))
    json.dump(ranked, open(os.path.join(B, "heldout-metrics.json"), "w"), indent=2)

    # Phase B: 12-frame variant — NOT RUN. The pinned upstream wrapper
    # (competition/upstream/baseline/run_resplat.py) validates
    # ``len(context) == min(16, len(train))``, i.e. exactly 16 context views; a 12-frame
    # variant would require modifying protected upstream code, which this task forbids.
    # Documented as untested in the report.
    say("SKIP-12 upstream run_resplat.py pins exactly 16 context views; 12-frame variant not run")
    twelve = []

    # Phase C: e2e = uniform control + best-by-held-out per bad case
    e2e = []
    for scene in BAD:
        sc = [s for s in man["scenes"] if s["scene"] == scene][0]
        best = ranked[scene]["best_method"]
        for meth in ["uniform"] + ([best] if best and best != "uniform" else []):
            c = [x for x in man["candidates"] if x["scene"] == scene and x["method"] == meth][0]
            if not wait_quiet("e2e-%s-%s" % (scene, meth)):
                raise SystemExit(1)
            e2e.append(delivery_e2e(c, "%s-%s" % (scene, meth)))
            json.dump(e2e, open(os.path.join(B, "results-e2e.json"), "w"), indent=2)
    # summary
    summary = dict(ranked=ranked, stage1_rows=len(rows), twelve=twelve, e2e=e2e)
    bad_deltas = [r["delta_q_vs_control"] for r in e2e if r.get("delta_q_vs_control") is not None]
    if bad_deltas:
        summary["projected_13case_gain_bad3"] = sum(bad_deltas) / 13.0
    json.dump(summary, open(os.path.join(B, "results.json"), "w"), indent=2)
    say("ALL_DONE rows=%d twelve=%d e2e=%d" % (len(rows), len(twelve), len(e2e)))

if __name__ == "__main__":
    main()
