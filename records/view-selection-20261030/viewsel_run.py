#!/usr/bin/env python3
"""Serial, quiet-gated runner for the co-visibility 16-view selection experiment.

Stages (all serial, GPU gated on gpu<50MiB + loadavg1<4 + no research/colmap procs):
  1) ReSplat for 3 scenes x 4 context selections (uniform control + 3 covis variants)
  2) context self-render PSNR triage (my ctx_psnr.py: renders vs GT for the 16 context views)
  3) full delivery e2e (S=5000) only for the best candidate per scene if it beats uniform by
     >= +0.3 dB; compares against clean13 controls.
"""
import json, os, subprocess, time

REPO = "/workspace/organizer-projects/FFSplat-Hackathon"
B = os.path.join(REPO, "outputs/viewsel-20261030")
CKPT = "/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth"
CONDA = "/usr/local/miniconda3/bin/conda"
LOGS = os.path.join(B, "logs"); os.makedirs(LOGS, exist_ok=True)
LOG = os.path.join(LOGS, "run.log")

def say(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=REPO)

def state():
    try:
        u = int(sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1").stdout.strip())
    except Exception:
        u = 99999
    try:
        load1 = float(open("/proc/loadavg").read().split()[0])
    except Exception:
        load1 = 99.0
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
        if waited % 120 == 0:
            say("WAIT-QUIET(%s) gpu=%dMiB load1=%.2f busy=%s" % (tag, u, load1, (others or colmap or "")[:100]))
        if waited >= max_min * 60:
            say("QUIET-TIMEOUT(%s)" % tag); return False
        time.sleep(20); waited += 20

def run(cmd, logpath):
    with open(logpath, "w") as f:
        r = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
    return r.returncode

def main():
    man = json.load(open(os.path.join(B, "candidates.json")))
    rows = []
    for c in man["candidates"]:
        tag = "%s-%s" % (c["scene"], c["method"])
        rdir = os.path.join(B, "resplat", c["scene"], c["method"])
        if not wait_quiet("resplat-" + tag):
            break
        rc = run("%s run --no-capture-output -n ffsplat-enhance python -m research.resplat_variant "
                 "--work %s --out %s --checkpoint %s --num_refine 2 --device cuda:0 --fix-sh-basis"
                 % (CONDA, c["work"], rdir, CKPT), os.path.join(LOGS, "resplat-%s.log" % tag))
        row = dict(scene=c["scene"], method=c["method"], case=c["case"], budget=c["budget"],
                   work=c["work"], resplat_rc=rc, resplat_out=os.path.relpath(rdir, REPO),
                   control_ctx_psnr=c["control_ctx_psnr"], control_total=c["control_total"],
                   control_q=c["control_q"], covis_stats=c["covis_stats"],
                   context_names=c["context_names"])
        rj = os.path.join(rdir, "resplat.json")
        if os.path.isfile(rj):
            d = json.load(open(rj))
            row.update(num_gaussians=d.get("num_gaussians"), resplat_seconds=d.get("total_seconds"))
        # context self-render PSNR (triage metric; renders vs GT for the 16 context views)
        cp = os.path.join(LOGS, "ctxpsnr-%s.json" % tag)
        rc = run("%s run --no-capture-output -n organizer-fastgs python %s %s %s > %s"
                 % (CONDA, os.path.join(B, "ctx_psnr.py"), rdir, c["work"], cp),
                 os.path.join(LOGS, "ctxpsnr-%s.err" % tag))
        cj = json.load(open(cp)) if os.path.isfile(cp) else {}
        row.update(ctx_psnr_mean=cj.get("ctx_psnr_mean"), ctx_render_n=cj.get("n_render"),
                   ctx_matched_n=cj.get("n_matched"))
        if row.get("ctx_psnr_mean") is not None:
            row["delta_ctx_vs_clean13_control"] = row["ctx_psnr_mean"] - c["control_ctx_psnr"]
        say("RESPLAT %s rc=%d gauss=%s ctx_psnr=%s d(vs clean13 control)=%s" %
            (tag, rc, row.get("num_gaussians"), row.get("ctx_psnr_mean"),
             row.get("delta_ctx_vs_clean13_control")))
        rows.append(row)
        json.dump(rows, open(os.path.join(B, "results-stage1.json"), "w"), indent=2)

    # ---- triage vs the same-session uniform control ----
    by_scene = {}
    for r in rows:
        by_scene.setdefault(r["scene"], {})[r["method"]] = r
    e2e = []
    for scene, m in by_scene.items():
        if "uniform" not in m:
            continue
        ctrl = m["uniform"].get("ctx_psnr_mean")
        if ctrl is None:
            continue
        for meth, r in m.items():
            if meth == "uniform" or r.get("ctx_psnr_mean") is None:
                continue
            r["delta_ctx_vs_uniform"] = r["ctx_psnr_mean"] - ctrl
        cands = [(r.get("delta_ctx_vs_uniform", -9), meth) for meth, r in m.items() if meth != "uniform"]
        cands.sort(reverse=True)
        if cands and cands[0][0] >= 0.30:
            meth = cands[0][1]
            e2e.append(dict(scene=scene, method=meth, delta_ctx_vs_uniform=cands[0][0],
                            row=m[meth], ctrl=ctrl))
            say("TRIAGE-PASS %s %s delta_ctx=%+.3f dB" % (scene, meth, cands[0][0]))
        else:
            say("TRIAGE-FAIL %s best_delta_ctx=%+.3f dB (no e2e)" % (scene, cands[0][0] if cands else -9))

    # ---- e2e for winners ----
    e2e_rows = []
    for e in e2e:
        r = e["row"]; tag = "%s-%s" % (e["scene"], r["method"])
        outdir = os.path.join(B, "runs", e["scene"], r["method"])
        if not wait_quiet("e2e-" + tag):
            break
        cmd = ("%s run --no-capture-output -n organizer-fastgs python -m research.run_sweep "
               "--case '%s' --work %s --init-ply %s/resplat_world.ply --out %s "
               "--iterations 5000 --snapshots '' --budget %s "
               "--position-lr-max-steps 5000 --opacity-reset-interval 0 --lambda-dssim 0.6 "
               "--scaling-lr 0.01 --densify-from 500 --densify-until 1250 "
               "--realloc-frac 0.15 --realloc-at 1250 --image-cache %s/image_cache"
               % (CONDA, r["case"], r["work"], os.path.join(B, "resplat", e["scene"], r["method"]),
                  outdir, r["budget"], r["work"]))
        rc = run(cmd, os.path.join(LOGS, "e2e-%s.log" % tag))
        sj = os.path.join(outdir, "summary.json")
        row = dict(scene=e["scene"], method=r["method"], case=r["case"], e2e_rc=rc,
                   delta_ctx_vs_uniform=e["delta_ctx_vs_uniform"],
                   ctx_psnr_mean=r.get("ctx_psnr_mean"), num_gaussians=r.get("num_gaussians"),
                   control_total=r["control_total"], control_q=r["control_q"])
        if os.path.isfile(sj):
            s = json.load(open(sj))["results"][0]; sc = s["score"]
            row.update(psnr=s["psnr"], ssim=s["ssim"], lpips=s["lpips"],
                       quality_points=sc["quality_points"], b_seconds=s["b_seconds_estimate"],
                       time_points=sc["time_points"], total_points=sc["total_points"])
            row["delta_total_vs_control"] = row["total_points"] - r["control_total"]
            row["delta_q_vs_control"] = row["quality_points"] - r["control_q"]
        say("E2E %s rc=%d total=%s d(total)=%s d(q)=%s" %
            (tag, rc, row.get("total_points"), row.get("delta_total_vs_control"), row.get("delta_q_vs_control")))
        e2e_rows.append(row)
        json.dump(e2e_rows, open(os.path.join(B, "results-e2e.json"), "w"), indent=2)

    json.dump({"stage1": rows, "e2e": e2e_rows}, open(os.path.join(B, "results.json"), "w"), indent=2)
    with open(os.path.join(B, "results.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        for r in e2e_rows:
            f.write(json.dumps(dict(r, stage="e2e")) + "\n")
    say("ALL_DONE rows=%d e2e=%d" % (len(rows), len(e2e_rows)))

if __name__ == "__main__":
    main()
