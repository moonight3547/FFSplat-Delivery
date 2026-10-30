#!/usr/bin/env python3
"""Exact 13-case delivery score WITH the M1 train-LPIPS gate (H1 measurement).

For every clean-13 case this script
  1. runs ``research.prune_gate`` on the case's ReSplat init PLY (8 fixed TRAIN
     views, VGG-LPIPS) -> keep 0.6 when train-LPIPS > tau, else 1.0;
  2. runs the delivery recipe (S=5000, matched LR schedule, structural ops in the
     first quarter, single-shot realloc 15 % @1250) through ``research.run_sweep``,
     passing ``--mid_prune_at 1250 --mid_prune_keep <gate keep>`` when the gate
     fires (keep == 1.0 is passed as well: the trainer treats it as a no-op);
  3. reads ``<out>/summary.json`` and aggregates the 13-case mean.

It is strictly serial and waits for a quiet GPU before starting (the module-B
time is the scored quantity, so a contended box would void the measurement).

Usage (on the GPU host, from the repo root):
    python scripts/run_exact13_gate.py --out-root outputs/exact13-gate-20261030 \
        --quiet-wait 3600
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "competition/profiles/compshare-20261029.json"

# tag, case, work dir, init PLY, budget (seconds) -- from records/clean13-20261030
CASES = [
    ("dl3dv-001dccbc", "dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f",
     "outputs/multi/dl3dv-001dccbc/work", 94.29),
    ("dl3dv-0b5635d5", "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce",
     "outputs/multi/dl3dv-0b5635d5/work", 94.29),
    ("dl3dv-0f622eaa", "dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e",
     "outputs/multi/dl3dv-0f622eaa/work", 94.29),
    ("dl3dv-2dad2d29", "dl3dv/2dad2d298927ff9b2ed7daa6128244e81d331e8cf1abf6c8598f35276ad38b84",
     "outputs/multi/dl3dv-2dad2d29/work", 94.29),
    ("dl3dv-3e4dbe41", "dl3dv/3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3",
     "outputs/multi/dl3dv-3e4dbe41/work", 94.29),
    ("dl3dv-9a451604", "dl3dv/9a451604b8d2583d47b14662635d8f74710c419cc061705ac15b8cfc31ece446",
     "outputs/multi/dl3dv-9a451604/work", 94.29),
    ("mipnerf360-bicycle", "mipnerf360/bicycle",
     "outputs/cand/mipnerf360-bicycle/work", 95.0),
    ("mipnerf360-bonsai", "mipnerf360/bonsai",
     "outputs/cand/mipnerf360-bonsai/work", 95.0),
    ("mipnerf360-counter", "mipnerf360/counter",
     "outputs/cand/mipnerf360-counter/work", 92.57),
    ("hw-billboard", "hw/billboard",
     "outputs/clean13-20261030/A/hw/billboard/work", 99.49),
    ("hw-basketball", "hw/basketball",
     "outputs/clean13-20261030/A/hw/basketball/work", 95.0),
    ("hw-car", "hw/car",
     "outputs/clean13-20261030/A/hw/car/work", 95.0),
    ("hw-trashbin", "hw/trashbin",
     "outputs/clean13-20261030/A/hw/trashbin/work", 95.0),
]

CLEAN_TOTALS = {  # records/clean13-20261030/results.jsonl (no gate)
    "dl3dv-001dccbc": 78.94231991986493, "dl3dv-0b5635d5": 31.10574259810206,
    "dl3dv-0f622eaa": 31.552958455583433, "dl3dv-2dad2d29": 49.75791395079112,
    "dl3dv-3e4dbe41": 35.03899809710881, "dl3dv-9a451604": 59.37744700699919,
    "mipnerf360-bicycle": 25.31011456233717, "mipnerf360-bonsai": 63.91858646481592,
    "mipnerf360-counter": 55.17158984097854, "hw-billboard": 53.67290926435828,
    "hw-basketball": 67.62046305098875, "hw-car": 48.53559196559025,
    "hw-trashbin": 64.97650419764423,
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def conda_cmd(profile, runtime, *argv):
    return [profile["conda"], "run", "--no-capture-output", "-n",
            profile["environments"][runtime], "python", *map(str, argv)]


def run(argv, log):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env.update(read(PROFILE).get("environment", {}))
    line = "+ " + " ".join(map(str, argv))
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()
    subprocess.run(argv, cwd=REPO, env=env, check=True,
                   stdout=log, stderr=subprocess.STDOUT)


def gpu_busy():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        used = [int(v) for v in out.split() if v.strip()]
    except Exception:
        return True
    if any(u > 1200 for u in used):
        return True
    ps = subprocess.run(["pgrep", "-af",
                         "resplat_variant|research.run_sweep|train_sweep|render_eval"],
                        capture_output=True, text=True)
    foreign = [l for l in ps.stdout.splitlines()
               if "run_exact13_gate" not in l and "pgrep" not in l]
    return bool(foreign)


def wait_quiet(max_wait, log):
    t0 = time.time()
    while gpu_busy():
        if time.time() - t0 > max_wait:
            raise SystemExit("QUIET-WAIT TIMED OUT: GPU still busy")
        time.sleep(20)
    print("[quiet] GPU idle, starting measurement", flush=True)
    log.write("[quiet] GPU idle\n")
    log.flush()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="outputs/exact13-gate-20261030")
    ap.add_argument("--tau", type=float, default=0.48)
    ap.add_argument("--keep", type=float, default=0.6)
    ap.add_argument("--quiet-wait", type=int, default=3600)
    ap.add_argument("--only", default=None, help="comma-separated tags subset")
    args = ap.parse_args(argv)

    profile = read(PROFILE)
    out_root = REPO / args.out_root
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    results_path = out_root / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["tag"])

    only = set(args.only.split(",")) if args.only else None
    todo = [c for c in CASES if (only is None or c[0] in only) and c[0] not in done]
    print(f"[plan] {len(todo)} case(s) to run, {len(done)} already done", flush=True)

    for tag, case, work, budget in todo:
        init = f"outputs/clean13-20261030/resplat/{tag}/resplat_world.ply"
        if not (REPO / init).exists():
            print(f"[skip] {tag}: init PLY missing ({init})", flush=True)
            continue
        cell = out_root / "cells" / tag
        cell.mkdir(parents=True, exist_ok=True)
        log = open(out_root / "logs" / f"{tag}.log", "a", encoding="utf-8")
        try:
            wait_quiet(args.quiet_wait, log)
            gate_rec = cell / "gate.json"
            run(conda_cmd(profile, "fastgs", "-m", "research.prune_gate",
                          "--work", work, "--ply", init,
                          "--tau", args.tau, "--keep", args.keep,
                          "--record", str(gate_rec)), log)
            gate = read(gate_rec)
            keep = float(gate["keep"])
            print(f"[gate] {tag}: train_lpips={gate['train_lpips']} keep={keep}", flush=True)

            argv_sweep = ["-m", "research.run_sweep", "--profile", str(PROFILE),
                          "--case", case, "--work", work, "--init-ply", init,
                          "--out", str(cell / "sweep"), "--budget", str(budget),
                          "--iterations", "5000", "--position-lr-max-steps", "5000",
                          "--opacity-reset-interval", "0", "--lambda-dssim", "0.6",
                          "--scaling-lr", "0.01", "--densify-from", "500",
                          "--densify-until", "1250", "--realloc-frac", "0.15",
                          "--realloc-at", "1250", "--image-cache", f"{work}/image_cache",
                          "--snapshots", "", "--seed", "20261015",
                          "--mid-prune-at", "1250", "--mid-prune-keep", str(keep)]
            run(conda_cmd(profile, "fastgs", *argv_sweep), log)

            summary = read(cell / "sweep" / "summary.json")
            row = summary["results"][-1]
            score = row["score"]
            rec = dict(tag=tag, case=case, budget=budget,
                       gate=gate, keep=keep,
                       num_gaussians=row.get("num_gaussians"),
                       psnr=row.get("psnr"), ssim=row.get("ssim"),
                       lpips=row.get("lpips"),
                       quality_points=score["quality_points"],
                       b_seconds=row.get("b_seconds_estimate"),
                       time_points=score["time_points"],
                       total_points=score["total_points"],
                       clean13_total=CLEAN_TOTALS.get(tag))
            with results_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[case] {tag}: total={rec['total_points']:.3f} "
                  f"(clean13 {rec['clean13_total']:.3f}) q={rec['quality_points']:.2f} "
                  f"B={rec['b_seconds']:.2f}", flush=True)
        finally:
            log.close()

    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    if rows:
        mean_total = sum(r["total_points"] for r in rows) / len(rows)
        mean_clean = sum(r["clean13_total"] for r in rows) / len(rows)
        payload = dict(n=len(rows), mean_total=mean_total, mean_clean13=mean_clean,
                       delta=mean_total - mean_clean,
                       per_case={r["tag"]: r for r in rows})
        (out_root / "aggregate.json").write_text(json.dumps(payload, indent=2))
        print(f"\n[RESULT] n={len(rows)} mean_total={mean_total:.3f} "
              f"(clean13 {mean_clean:.3f}, delta {mean_total - mean_clean:+.3f})",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
