#!/usr/bin/env python3
"""Mini-Splatting report tables (run on the Mac against raw/ copies).

Per-scene control = the same out-root `ctrl` cell (delivery recipe, S=5000).
M2 arms are post-5000 prune(+refine) cells: their own iter05000 row is the paired
control (same run), so deltas are strict inside-run.
M1 arms are single 5000-step runs with a mid-training prune; they are compared
against the control cell of the same scene.
"""
import glob
import json
import os
import sys

BUDGET = {"mipnerf360/bicycle": 95.0, "mipnerf360/counter": 92.57}
TIME_K = 20.0


def q(psnr, ssim, lpips):
    p = 20.0 * max(0.0, min(1.0, (psnr - 15.0) / 25.0))
    s = 20.0 * max(0.0, min(1.0, (ssim - 0.70) / 0.25))
    l = 40.0 * max(0.0, min(1.0, (0.25 - lpips) / 0.20))
    return p + s + l


def time_pts(b, budget):
    return TIME_K * max(0.0, min(1.0, (1.0 - b / budget) / 0.9))


def load_cell(cell_dir):
    summary = json.load(open(os.path.join(cell_dir, "sweep", "summary.json")))
    record = json.load(open(os.path.join(cell_dir, "sweep", "train_record.json")))
    case = summary["case"]
    budget = BUDGET.get(case, 94.29)
    rows = []
    for res in summary["results"]:
        b = res["b_seconds_estimate"]
        rows.append(dict(iteration=res["iteration"], n=res["num_gaussians"],
                         b=b, psnr=res["psnr"], ssim=res["ssim"],
                         lpips=res["lpips"], q=q(res["psnr"], res["ssim"], res["lpips"]),
                         t=time_pts(b, budget), n_round=res["num_gaussians"],
                         total=q(res["psnr"], res["ssim"], res["lpips"]) + time_pts(b, budget)))
    return dict(case=case, budget=budget, rows=rows, record=record,
                stage1_s=record.get("stage1_seconds"),
                stage2_s=record.get("stage2_seconds"),
                prune_events=record.get("prune_events") or [],
                stage2_trainer=record.get("stage2_trainer"))


def fmt_row(r, extra=""):
    return ("%6d %8.3f %6.4f %6.4f %7.3f %8.2f %6.2f %7.3f %s"
            % (r["n"], r["psnr"], r["ssim"], r["lpips"], r["q"], r["b"],
               r["t"], r["total"], extra))


def main(root):
    cells = {}
    for d in glob.glob(os.path.join(root, "*", "*")):
        if not os.path.isdir(os.path.join(d, "sweep")):
            continue
        tag = os.path.basename(os.path.dirname(d))
        arm = os.path.basename(d)
        cells[(tag, arm)] = load_cell(d)
    cases = sorted({t for t, _ in cells})
    print("=== M2 importance-prune post-pass (paired: same-run iter05000 -> stage end) ===")
    print("%-22s %-8s %6s %8s %6s %6s %7s %8s %6s %7s | %8s %7s %7s %8s %8s"
          % ("case", "arm", "N", "psnr", "ssim", "lpips", "q", "B", "TIME", "total",
             "dN", "dB", "dTIME", "dq", "dtotal"))
    for tag in cases:
        ctrl = cells.get((tag, "ctrl"))
        for arm in ("m2_50", "m2_70", "m2_90", "m2_95"):
            cell = cells.get((tag, arm))
            if not cell:
                continue
            base = cell["rows"][0]
            end = cell["rows"][-1]
            prune = (cell["record"].get("prune_events") or []) + \
                    ([end.get("prune")] if isinstance(end, dict) and end.get("prune") else [])
            # stage snapshot carries the prune stats inside the record snapshots
            pe = None
            for snap in cell["record"].get("snapshots", []):
                if snap.get("prune"):
                    pe = snap["prune"]
            n_before = pe["n_before"] if pe else base["n"]
            n_after = pe["n_after"] if pe else end["n"]
            dn = (n_after - n_before) / n_before * 100.0
            db = end["b"] - base["b"]
            dt = -(TIME_K * db) / (0.9 * cell["budget"])
            dq = end["q"] - base["q"]
            print("%-22s %-8s %6d %8.3f %6.4f %6.4f %7.3f %8.2f %6.2f %7.3f | %+7.1f%% %+7.2f %+7.2f %+8.3f %+8.3f"
                  % (tag.split("/")[-1][:22], arm, end["n"], end["psnr"], end["ssim"],
                     end["lpips"], end["q"], end["b"], end["t"], end["total"],
                     dn, db, dt, dq, dq + dt))
            print("    before: N=%d q=%.3f B=%.2f | after: N=%d q=%.3f B=%.2f | "
                  "prune_s=%.3f refine_s=%.2f"
                  % (n_before, base["q"], base["b"], n_after, end["q"], end["b"],
                     pe["seconds"] if pe else -1, cell["stage2_s"] or -1))
    print("\n=== M1 mid-training prune (single 5000-step run, vs same-scene ctrl) ===")
    print("%-22s %-12s %6s %8s %6s %6s %7s %8s %6s %7s | %8s %7s %7s %8s %8s"
          % ("case", "arm", "N", "psnr", "ssim", "lpips", "q", "B", "TIME", "total",
             "dN", "dB", "dTIME", "dq", "dtotal"))
    for tag in cases:
        ctrl = cells.get((tag, "ctrl"))
        if not ctrl:
            continue
        c = ctrl["rows"][-1]
        for arm in ("m1_1250_90", "m1_1250_80", "m1_1250_70", "m1_1250_60", "m1_1250_50", "m1_1250_40", "m1_2000_70", "m1_1000_70"):
            cell = cells.get((tag, arm))
            if not cell:
                continue
            r = cell["rows"][-1]
            pe = (cell["record"].get("prune_events") or [None])[0]
            dn = ((r["n"] - c["n"]) / c["n"] * 100.0)
            db = r["b"] - c["b"]
            dt = -(TIME_K * db) / (0.9 * cell["budget"])
            dq = r["q"] - c["q"]
            print("%-22s %-12s %6d %8.3f %6.4f %6.4f %7.3f %8.2f %6.2f %7.3f | %+7.1f%% %+7.2f %+7.2f %+8.3f %+8.3f"
                  % (tag.split("/")[-1][:22], arm, r["n"], r["psnr"], r["ssim"],
                     r["lpips"], r["q"], r["b"], r["t"], r["total"],
                     dn, db, dt, dq, dq + dt))
            print("    ctrl: N=%d q=%.3f B=%.2f | arm: N=%d q=%.3f B=%.2f | "
                  "prune at iter %s: %d -> %d in %.2fs (mid_prune_seconds)"
                  % (c["n"], c["q"], c["b"], r["n"], r["q"], r["b"],
                     (pe or {}).get("label", "?"),
                     (pe or {}).get("n_before", 0), (pe or {}).get("n_after", 0),
                     (pe or {}).get("seconds", -1)))
    print("\n=== B-vs-count trade-off (M1 vs ctrl) ===")
    for tag in cases:
        ctrl = cells.get((tag, "ctrl"))
        if not ctrl:
            continue
        c = ctrl["rows"][-1]
        print("%-22s ctrl N=%d B=%.2f (%.2f us/Gauss)"
              % (tag.split("/")[-1][:22], c["n"], c["b"], 1e6 * c["b"] / c["n"]))
        for arm in ("m1_1250_90", "m1_1250_80", "m1_1250_70", "m1_1250_60", "m1_1250_50", "m1_1250_40", "m1_2000_70"):
            cell = cells.get((tag, arm))
            if cell:
                r = cell["rows"][-1]
                print("%-22s %-12s N=%d B=%.2f (%.2f us/Gauss) dq=%+.3f dtotal=%+.3f"
                      % ("", arm, r["n"], r["b"], 1e6 * r["b"] / r["n"],
                         r["q"] - c["q"], r["q"] + r["t"] - (c["q"] + c["t"])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raw/matrix")
