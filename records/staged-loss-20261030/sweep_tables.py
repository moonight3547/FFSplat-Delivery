#!/usr/bin/env python3
"""Render the §10 sweep tables from the on-node cell dirs (run on the Mac after
copying the sweep2/*/sweep/summary.json + train_record.json + the LPIPS points).

Durable per-step costs measured on N1 (used for TIME because the sweep window is
CPU/shared-box contended; see STAGED-LOSS-20261030.md §10):
  VGG phase-3 step     : 1.426 s / 100 steps = 14.26 ms/step
  absgrad phase-2 step: 2.42  s / 500 steps =  4.84 ms/step
"""
import json
import sys

VGG_MS = 14.26
ABSGRAD_MS = 4.84
BUDGET = 95.0


def q_points(psnr, ssim, lpips):
    p = 20.0 * max(0.0, min(1.0, (psnr - 15.0) / 25.0))
    s = 20.0 * max(0.0, min(1.0, (ssim - 0.70) / 0.25))
    l = 40.0 * max(0.0, min(1.0, (0.25 - lpips) / 0.20))
    return p + s + l, p, s, l


def time_delta(extra_seconds):
    return -20.0 * extra_seconds / (0.9 * BUDGET)


def main(paths):
    rows = []
    for path in paths:
        data = json.load(open(path))
        results = data["results"]
        base = results[0]
        end = results[-1]
        rec = json.load(open(str(path).replace("summary.json", "train_record.json")))
        arm = path.strip("/").split("/")[-3]
        steps = end["iteration"] - base["iteration"]
        if arm.startswith("vgg"):
            extra_s = rec.get("stage3_seconds") or steps * VGG_MS / 1000.0
            kind = "vgg"
        else:
            extra_s = rec.get("stage2_seconds") or steps * ABSGRAD_MS / 1000.0
            kind = "absgrad"
        qb, pb, sb, lb = q_points(base["psnr"], base["ssim"], base["lpips"])
        qe, pe, se, le = q_points(end["psnr"], end["ssim"], end["lpips"])
        below_cliff = end["lpips"] < 0.25
        dt = time_delta(extra_s)
        dq = qe - qb
        rows.append(dict(
            arm=arm, steps=steps, kind=kind,
            base=(base["psnr"], base["ssim"], base["lpips"], qb),
            end=(end["psnr"], end["ssim"], end["lpips"], qe, below_cliff),
            dq=dq, extra_s=extra_s, dt=dt, dtotal=dq + dt,
            n=end["num_gaussians"],
            measured_s=rec.get("stage3_seconds") if kind == "vgg" else rec.get("stage2_seconds")))
    print("| arm | steps | iter%d PSNR/SSIM/LPIPS (q) | iter%d PSNR/SSIM/LPIPS (q) | Δq | stage s | ΔTIME | Δtotal | LPIPS<0.25 |"
          % (5000, 5000 + rows[0]["steps"]) if rows else "no rows")
    for r in sorted(rows, key=lambda r: (r["kind"], r["steps"])):
        b = r["base"]
        e = r["end"]
        print("| %s | %d | %.3f/%.4f/%.4f (%.3f) | %.3f/%.4f/%.4f (%.3f) | %+.3f | %.2f | %+.2f | %+.2f | %s |"
              % (r["arm"], r["steps"], b[0], b[1], b[2], b[3], e[0], e[1], e[2], e[3],
                 r["dq"], r["extra_s"], r["dt"], r["dtotal"],
                 "YES" if e[4] else "no"))
    print()
    for r in sorted(rows, key=lambda r: (r["kind"], r["steps"])):
        print("%-20s steps=%5d  measured_stage_s=%.2f  gaussians=%d  dq=%+.3f dTIME=%+.2f dtotal=%+.2f"
              % (r["arm"], r["steps"], r["measured_s"] or 0.0, r["n"], r["dq"], r["dt"], r["dtotal"]))
    # break-even bars (TIME points the arm must earn back quality-wise)
    print("\nbreak-even bars: dq > -dTIME")
    for r in sorted(rows, key=lambda r: (r["kind"], r["steps"])):
        print("  %-20s needs dq > %+.2f (observed %+.3f -> %s)"
              % (r["arm"], -r["dt"], r["dq"], "PASS" if r["dq"] > -r["dt"] else "FAIL"))


if __name__ == "__main__":
    main(sys.argv[1:])
