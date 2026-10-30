#!/usr/bin/env python3
"""N1 steps-Pareto report generator (concurrent grid + serial anchors).

  python3 n1_report.py --parallel results.jsonl --serial serial/results.jsonl \
      --md N1-SWEEP-SUMMARY.md

Scoring (authoritative proxy):
  quality = 20*clip((PSNR-15)/25) + 20*clip((SSIM-0.70)/0.25) + 40*clip((0.25-LPIPS)/0.20)
  TIME    = 20*min(1, (1-B/budget)/0.9)
  total   = quality + TIME
Serial-corrected estimate: for each scene the concurrent->serial inflation factor
f = B_par(S=5000) / B_ser(S=5000) is applied to every concurrent B of that scene
(flagged "est"; measured serial cells are used verbatim where available).
"""
import argparse
import json
from collections import defaultdict


def clip(x):
    return max(0.0, min(1.0, x))


def quality(r):
    return (20.0 * clip((r["psnr"] - 15.0) / 25.0)
            + 20.0 * clip((r["ssim"] - 0.70) / 0.25)
            + 40.0 * clip((0.25 - r["lpips"]) / 0.20))


def time_pts(b, budget):
    if b is None or budget is None or b >= budget:
        return 0.0
    return 20.0 * min(1.0, (1.0 - b / budget) / 0.9)


def total(r, b=None):
    return quality(r) + time_pts(b if b is not None else r["b_seconds_estimate"],
                                 r["budget_seconds"])


def load(path):
    rows = {}
    if not path:
        return rows
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[(r["case"], r["S"])] = r
    return rows


def f2(x):
    return "-" if x is None else "%.2f" % x


def f3(x):
    return "-" if x is None else "%.3f" % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", required=True)
    ap.add_argument("--serial", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    par = load(args.parallel)
    ser = load(args.serial)
    cases = sorted({c for c, _ in par})
    steps = sorted({s for _, s in par})
    short = {c: (c.split("/")[0] + "/" + c.split("/")[1][:10]) for c in cases}

    # ---- run-to-run repeatability (psnr/ssim/lpips, same cell, serial vs concurrent)
    reps = []
    for (c, S), r in ser.items():
        p = par.get((c, S))
        if p:
            reps.append((short[c], S, r["psnr"] - p["psnr"], r["ssim"] - p["ssim"],
                         r["lpips"] - p["lpips"]))

    # ---- B inflation factor per scene from the S=5000 anchor
    infl = {}
    for c in cases:
        bp = par.get((c, 5000), {}).get("b_seconds_estimate")
        bs = ser.get((c, 5000), {}).get("b_seconds_estimate")
        if bp and bs:
            infl[c] = bp / bs

    def b_clean(c, S):
        """Measured serial B if available, else concurrent B / per-scene inflation."""
        s = ser.get((c, S))
        if s:
            return s["b_seconds_estimate"]
        p = par.get((c, S))
        if p and c in infl:
            return p["b_seconds_estimate"] / infl[c]
        return None

    md = []
    w = md.append
    w("# N1 steps-Pareto sweep — raw results\n")
    w("Grid: 13 scenes x S in {%s}; delivery recipe with step-matched schedule "
      "(position-lr %s, densify max(20,S/10)->max(40,S/4), realloc 0.15 @ max(20,S/4), "
      "opacity-reset 0, lambda_dssim 0.6, scaling_lr 0.01, uint8 image cache)."
      % (", ".join(map(str, steps)), "S"))
    w("Runs: %d concurrent cells (3 lanes) + %d serial anchor cells (1 lane). "
      "B = scene_load + init + snap_train + 0.5 (same definition as the 13-case "
      "reference table; PLY export and post-train quick test excluded).\n"
      % (len(par), len(ser)))

    # ============================ 1. headline
    del_par = [total(par[(c, 5000)]) for c in cases if (c, 5000) in par]
    del_ser = [total(ser[(c, 5000)]) for c in cases if (c, 5000) in ser]
    w("## 1. Policy comparison (13-scene mean of total proxy)\n")
    w("| policy | mean total | B source | n |")
    w("|---|---|---|---|")
    if del_par:
        w("| (a) fixed S=5000 = delivery | **%.3f** | concurrent | %d |"
          % (sum(del_par) / len(del_par), len(del_par)))
    if del_ser:
        w("| (a) fixed S=5000 = delivery | **%.3f** | serial (clean) | %d |"
          % (sum(del_ser) / len(del_ser), len(del_ser)))

    # uniform means, two ways
    uni_par, uni_est = {}, {}
    for S in steps:
        vp = [total(par[(c, S)]) for c in cases if (c, S) in par]
        ve = [total(par[(c, S)], b_clean(c, S)) for c in cases if (c, S) in par]
        if vp:
            uni_par[S] = sum(vp) / len(vp)
        if ve:
            uni_est[S] = sum(ve) / len(ve)
    bS = max(uni_par, key=uni_par.get)
    bSe = max(uni_est, key=uni_est.get)
    w("| (b) best uniform S = %d | %.3f | concurrent | 13 |" % (bS, uni_par[bS]))
    if uni_est:
        w("| (b') best uniform S = %d | %.3f | serial-corrected (est) | 13 |"
          % (bSe, uni_est[bSe]))

    # oracle
    orc_par = []
    orc_est = []
    argmax_par, argmax_est = {}, {}
    for c in cases:
        rs = [par[(c, S)] for S in steps if (c, S) in par]
        if not rs:
            continue
        bestp = max(rs, key=lambda r: total(r))
        argmax_par[c] = bestp["S"]
        orc_par.append(total(bestp))
        beste = max(rs, key=lambda r: total(r, b_clean(c, r["S"])))
        argmax_est[c] = beste["S"]
        orc_est.append(total(beste, b_clean(c, beste["S"])))
    if orc_par:
        w("| (c) per-scene oracle (argmax) | %.3f | concurrent | 13 |"
          % (sum(orc_par) / len(orc_par)))
    if orc_est:
        w("| (c') per-scene oracle (argmax) | %.3f | serial-corrected (est) | 13 |"
          % (sum(orc_est) / len(orc_est)))
    w("")

    # ============================ 2. per-scene table
    w("## 2. Per-scene table (S ascending; q / B / t / total)\n")
    w("Concurrent rows are the raw 3-lane measurement; `B*`/`total*` use the "
      "serial-corrected B (measured anchor where it exists, else scaled by that "
      "scene's S=5000 anchor).\n")
    w("| scene | S | q | B conc | t | total | B* | total* |")
    w("|---|---|---|---|---|---|---|---|")
    for c in cases:
        for S in steps:
            r = par.get((c, S))
            if not r:
                continue
            bc = b_clean(c, S)
            mark = " **[serial]**" if (c, S) in ser else ""
            w("| %s%s | %d | %.2f | %s | %.2f | %.3f | %s | %s |"
              % (short[c], mark, S, quality(r), f2(r["b_seconds_estimate"]),
                 time_pts(r["b_seconds_estimate"], r["budget_seconds"]), total(r),
                 f2(bc), f3(total(r, bc))))
    w("")

    # ============================ 3. argmax + gains
    w("## 3. Per-scene argmax and gain over fixed S=5000\n")
    w("| scene | argmax (conc) | total@argmax | total@5000 | gain | argmax (serial-"
      "corrected) | total@argmax* | total@5000* | gain* |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c in cases:
        if (c, 5000) not in par:
            continue
        r0 = par[(c, 5000)]
        ap = argmax_par[c]
        rp = par[(c, ap)]
        ae = argmax_est.get(c, ap)
        re_ = par[(c, ae)]
        w("| %s | %d | %.3f | %.3f | %+.3f | %d | %.3f | %.3f | %+.3f |"
          % (short[c], ap, total(rp), total(r0), total(rp) - total(r0),
             ae, total(re_, b_clean(c, ae)), total(r0, b_clean(c, 5000)),
             total(re_, b_clean(c, ae)) - total(r0, b_clean(c, 5000))))
    w("")

    # ============================ 4. serial anchors raw
    if ser:
        w("## 4. Serial anchors (single card, other two GPUs idle)\n")
        w("| scene | S | B serial | B concurrent | inflation | q ser | total ser | "
          "psnr ser |")
        w("|---|---|---|---|---|---|---|---|")
        for (c, S) in sorted(ser):
            r = ser[(c, S)]
            p = par.get((c, S), {})
            w("| %s | %d | %.2f | %.2f | x%.3f | %.2f | %.3f | %.3f |"
              % (short[c], S, r["b_seconds_estimate"],
                 p.get("b_seconds_estimate", float("nan")),
                 (p.get("b_seconds_estimate", 0) / r["b_seconds_estimate"]),
                 quality(r), total(r), r["psnr"]))
        w("")
        w("Run-to-run repeatability (serial - concurrent, same config): "
          "dPSNR %.3f..%.3f, dSSIM %.4f..%.4f, dLPIPS %.4f..%.4f (n=%d).\n"
          % (min(x[2] for x in reps), max(x[2] for x in reps),
             min(x[3] for x in reps), max(x[3] for x in reps),
             min(x[4] for x in reps), max(x[4] for x in reps), len(reps)))

    # ============================ 5. B(S) decomposition
    w("## 5. B(S) decomposition and per-step rate\n")
    w("| scene | S | load | init | train(record) | snap_train | B | B* | "
      "iters/s | gaussians |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for c in cases:
        for S in steps:
            r = par.get((c, S))
            if not r:
                continue
            w("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |"
              % (short[c], S, f2(r.get("scene_load_seconds")),
                 f2(r.get("init_seconds")), f2(r.get("train_seconds")),
                 f2(r.get("b_seconds_estimate", 0) - (r.get("scene_load_seconds") or 0)
                    - (r.get("init_seconds") or 0) - 0.5),
                 f2(r["b_seconds_estimate"]), f2(b_clean(c, S)),
                 f2(r.get("iterations_per_second")), r.get("num_final_gaussians")))
    w("")

    text = "\n".join(md)
    print(text)
    if args.md:
        open(args.md, "w", encoding="utf-8").write(text + "\n")
        print("\nwrote", args.md)


if __name__ == "__main__":
    main()
