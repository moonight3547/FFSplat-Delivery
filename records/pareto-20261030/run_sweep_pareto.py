#!/usr/bin/env python3
"""Steps-Pareto sweep driver for N1 (13 scenes x 6 step counts = 78 cells).

Every cell runs the *delivery recipe* with a step-matched schedule, only the
number of module-B optimization steps S varies:

  research.run_sweep --case C --work W --init-ply P --out <root>/<tag>/s<S>
    --iterations S --snapshots "" --budget B
    --position-lr-max-steps S --opacity-reset-interval 0 --lambda-dssim 0.6
    --scaling-lr 0.01
    --densify-from max(20,S//10) --densify-until max(40,S//4)
    --realloc-frac 0.15 --realloc-at max(20,S//4)
    --image-cache W/image_cache

At S=5000 this is exactly the measured delivery recipe (densify 500->1250,
realloc at 1250, position_lr_max_steps 5000).

Modes:
  parallel : 3 lanes, one per GPU profile competition/profiles/gpufree-{0,1,2}.json
  serial   : one lane (GPU 0), other GPUs expected idle -> clean B seconds
  report   : print the Pareto table from existing summary.json files only

Resumable: a cell whose summary.json parses is skipped. One retry per cell.
Failures are recorded (never silently dropped) in <root>/failures.jsonl.
Per-cell metadata (lane, wall seconds, cache) in <root>/cells/*.json.
results.jsonl is rebuilt from all summary.json under <root> at the end.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path("/root/projects/FFSplat-Hackathon")
CONDA = "/opt/conda/bin/conda"
DEFAULT_MANIFEST = REPO / "records/evalnode-manifest-13.json"
STEPS = [500, 1000, 2000, 3000, 5000, 8000]
PRINT_LOCK = threading.Lock()
GPU_LOCK = threading.Lock()  # serialises nothing important; keeps log writes tidy


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(root, msg):
    line = "%s %s" % (now(), msg)
    with PRINT_LOCK:
        print(line, flush=True)
        with open(Path(root) / "queue.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- tags / cells
def build_tags(entries):
    """Short, unique, filesystem-safe tag per case."""
    tags, seen = {}, {}
    for entry in entries:
        case = entry["case"]
        kind, _, name = case.partition("/")
        if kind == "dl3dv":
            base = "dl3dv-" + name[:12]
        elif kind == "mipnerf360":
            base = "mip-" + name
        elif kind == "hw":
            base = "hw-" + name
        else:
            base = re.sub(r"[^A-Za-z0-9._-]+", "-", case)[:40]
        if base in seen:  # collision -> add digest
            base = base + "-" + hashlib.md5(case.encode()).hexdigest()[:6]
        seen[base] = case
        tags[case] = base
    return tags


def resolve(root, path):
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)


def recipe_iters(S):
    return {
        "densify_from": max(20, S // 10),
        "densify_until": max(40, S // 4),
        "realloc_at": max(20, S // 4),
        "position_lr_max_steps": S,
    }


def cell_paths(root, tag, S):
    return Path(root) / tag / ("s%d" % S)


def cell_meta_path(root, tag, S):
    return Path(root) / "cells" / ("%s_s%d.json" % (tag, S))


def summary_ok(path):
    try:
        data = read_json(path)
    except Exception:
        return None
    results = data.get("results") or []
    if not results:
        return None
    return data


def build_cell_list(entries, steps, cells_file):
    if cells_file:
        cells = []
        for raw in Path(cells_file).read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            case, S = parts[0], int(parts[1])
            cells.append((case, S))
        return cells
    cells = []
    for S in steps:  # S-major: lanes start on three different scenes, same S
        for entry in entries:
            cells.append((entry["case"], S))
    return cells


# ------------------------------------------------------------------ execution
def run_command(cmd, logfile, timeout):
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "ab") as fh:
        fh.write(("\n+ " + " ".join(map(str, cmd)) + "\n").encode())
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=fh,
                                stderr=subprocess.STDOUT, start_new_session=True,
                                env=dict(os.environ, PYTHONPATH=str(REPO)))
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            proc.wait()
            return 124


def gpu_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    used = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            used.append(int(line))
    return used


def cache_ready(cache):
    return (cache / "manifest.json").exists() and (cache / "images_u8.npy").exists()


def ensure_image_cache(root, entry, tag, S, timeout, dry_run=False,
                       cache_wait=0.0, build_cache=False):
    """Return usable cache dir, or None if it must run without the cache.

    The cache is built by the h1-prep task (`<work>/image_cache`); by default we
    never write it (avoid two writers on the same npy) and instead poll until it
    appears, then run without it and flag the cell in results.jsonl.
    """
    work = resolve(root, entry["work"])
    cache = work / "image_cache"
    if cache_ready(cache):
        return cache
    if dry_run:
        return cache if build_cache else None
    announced = False
    deadline = time.time() + cache_wait
    while True:
        if cache_ready(cache):
            return cache
        if build_cache:
            log(root, "CACHE_BUILD %s (missing %s/manifest.json)" % (entry["case"], cache))
            cmd = [CONDA, "run", "--no-capture-output", "-n", "fastgs", "python", "-m",
                   "research.image_cache", "--work", str(work), "--out", str(cache)]
            rc = run_command(cmd, Path(root) / "logs" / ("cache_%s.log" % tag), timeout)
            if rc == 0 and cache_ready(cache):
                return cache
            log(root, "CACHE_FAIL %s rc=%s -> running WITHOUT image cache"
                % (entry["case"], rc))
            return None
        if time.time() >= deadline:
            log(root, "CACHE_MISSING %s -> running WITHOUT image cache "
                      "(B will be flagged cached=false)" % entry["case"])
            return None
        if not announced:
            log(root, "CACHE_WAIT %s: %s absent, polling up to %.0fs"
                % (entry["case"], cache, cache_wait))
            announced = True
        time.sleep(10)


def run_cell(root, entry, S, lane, timeout, dry_run=False, cache_wait=0.0,
             build_cache=False, no_quick_test=False):
    """Run one cell. Returns a meta dict, or "skip"/"dry"."""
    tag = entry["tag"]
    out = cell_paths(root, tag, S)
    summ = out / "summary.json"
    data = summary_ok(summ)
    if data is not None:
        prev_work = data.get("work")
        same = (prev_work is not None
                and str(resolve(root, prev_work)) == str(resolve(root, entry["work"])))
        if same:
            log(root, "SKIP lane%d %s s%d (summary.json exists)" % (lane, entry["case"], S))
            return "skip"
        log(root, "STALE lane%d %s s%d (summary work %r != manifest %r) -> re-run"
            % (lane, entry["case"], S, prev_work, entry["work"]))
    if out.exists() and any(out.iterdir()):  # crashed earlier, no usable summary.json
        failed = out.with_name(out.name + ".failed-%d" % int(time.time()))
        log(root, "MOVE_ASIDE lane%d %s s%d -> %s" % (lane, entry["case"], S, failed))
        shutil.move(str(out), str(failed))
    cache = ensure_image_cache(root, entry, tag, S, timeout, dry_run,
                               cache_wait, build_cache)
    it = recipe_iters(S)
    cmd = [CONDA, "run", "--no-capture-output", "-n", "fastgs", "python", "-m",
           "research.run_sweep",
           "--profile", "competition/profiles/gpufree-%d.json" % lane,
           "--case", entry["case"],
           "--work", str(resolve(root, entry["work"])),
           "--init-ply", str(resolve(root, entry["init"])),
           "--out", str(out),
           "--iterations", str(S),
           "--snapshots", "",
           "--budget", str(entry["budget_seconds"]),
           "--position-lr-max-steps", str(it["position_lr_max_steps"]),
           "--opacity-reset-interval", "0",
           "--lambda-dssim", "0.6",
           "--scaling-lr", "0.01",
           "--densify-from", str(it["densify_from"]),
           "--densify-until", str(it["densify_until"]),
           "--realloc-frac", "0.15",
           "--realloc-at", str(it["realloc_at"])]
    if cache is not None:
        cmd += ["--image-cache", str(cache)]
    if no_quick_test:
        # quick test is a diagnostic rendered AFTER module-B timing is taken; it does
        # not affect B, but costs ~5-15s/cell. Serial re-timing runs skip it.
        cmd += ["--no-quick-test"]
    meta = dict(case=entry["case"], S=S, lane=lane, out=str(out),
                cmd=[str(c) for c in cmd], cache=str(cache) if cache else None,
                cache_missing=(cache is None), started=now())
    write_json(cell_meta_path(root, tag, S), meta)
    if dry_run:
        print("DRY-RUN lane%d %s s%d: %s" % (lane, entry["case"], S, " ".join(cmd)))
        return "dry"
    log(root, "START lane%d %s s%d -> %s" % (lane, entry["case"], S, out))
    t0 = time.time()
    rc = run_command(cmd, Path(root) / "logs" / ("%s_s%d.log" % (tag, S)), timeout)
    meta["wall_seconds"] = round(time.time() - t0, 1)
    meta["finished"] = now()
    meta["rc"] = rc
    meta["status"] = "ok" if (rc == 0 and summary_ok(summ) is not None) else "fail"
    if meta["status"] == "fail" and out.exists():
        failed = out.with_name(out.name + ".failed-%d" % int(time.time()))
        if not failed.exists():
            try:
                shutil.move(str(out), str(failed))
            except Exception:
                pass
    write_json(cell_meta_path(root, tag, S), meta)
    log(root, "END lane%d %s s%d rc=%d wall=%.1fs status=%s"
        % (lane, entry["case"], S, rc, meta["wall_seconds"], meta["status"]))
    return meta


def record_failure(root, entry, S, lane, attempt, error):
    row = dict(case=entry["case"], S=S, lane=lane, attempt=attempt,
               error=str(error)[-4000:], ts=now())
    line = json.dumps(row)
    with PRINT_LOCK:
        with open(Path(root) / "failures.jsonl", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    log(root, "FAIL lane%d %s s%d attempt%d: %s"
        % (lane, entry["case"], S, attempt, str(error).splitlines()[-1][:200]))


def lane_worker(root, queue, entries_by_case, timeout, dry_run, cache_wait,
                build_cache, no_quick_test=False):
    for entry, S in queue:
        lane = entry["lane"]
        try:
            outcome = run_cell(root, entry, S, lane, timeout, dry_run,
                               cache_wait, build_cache, no_quick_test)
            if outcome in ("skip", "dry"):
                continue
            if isinstance(outcome, dict) and outcome.get("status") == "ok":
                continue
            # first attempt failed -> retry once
            log(root, "RETRY lane%d %s s%d" % (lane, entry["case"], S))
            outcome = run_cell(root, entry, S, lane, timeout, dry_run,
                               cache_wait, build_cache, no_quick_test)
            if not (isinstance(outcome, dict) and outcome.get("status") == "ok"):
                record_failure(root, entry, S, lane, 2,
                               "retry failed: %s" % (outcome,))
        except Exception as exc:  # never let one cell kill a lane
            record_failure(root, entry, S, lane, 2, repr(exc))


# ------------------------------------------------------------------- results
def rebuild_results(root, tags):
    rows = []
    root = Path(root)
    if not root.exists():
        return rows
    for tag in sorted(tags.values()):
        for cell_dir in sorted((root / tag).glob("s*")):
            if not re.fullmatch(r"s\d+", cell_dir.name):
                continue
            S = int(cell_dir.name[1:])
            data = summary_ok(cell_dir / "summary.json")
            if data is None:
                continue
            case = next((c for c, t in tags.items() if t == tag), None)
            if case is None:
                continue
            result = data["results"][-1]
            score = result.get("score") or {}
            record = data.get("train_record") or {}
            b = result.get("b_seconds_estimate")
            load_s = record.get("scene_load_seconds")
            init_s = record.get("init_seconds")
            train_s = record.get("train_seconds")
            total_s = record.get("total_seconds")
            export_s = None
            if None not in (total_s, load_s, init_s, train_s):
                export_s = round(total_s - (load_s + init_s + train_s), 3)
            meta = {}
            meta_path = cell_meta_path(root, tag, S)
            if meta_path.exists():
                try:
                    meta = read_json(meta_path)
                except Exception:
                    meta = {}
            rows.append(dict(
                case=case, S=S, tag=tag, lane=meta.get("lane"),
                b_seconds_estimate=b,
                psnr=result.get("psnr"), ssim=result.get("ssim"),
                lpips=result.get("lpips"),
                quality_points=score.get("quality_points"),
                time_points=score.get("time_points"),
                total_points=score.get("total_points"),
                valid=score.get("valid"),
                budget_seconds=data.get("budget_seconds"),
                final_gaussians=result.get("num_gaussians"),
                num_init_gaussians=record.get("num_init_gaussians"),
                num_final_gaussians=record.get("num_final_gaussians"),
                num_train_views=record.get("num_train_views"),
                num_test_views=record.get("num_test_views"),
                scene_load_seconds=load_s,
                init_seconds=init_s,
                train_seconds=train_s,
                export_seconds_estimate=export_s,
                total_seconds=total_s,
                iterations_per_second=record.get("iterations_per_second"),
                # diagnostic rendered on the scene's TEST cameras after module-B timing
                # (research/train_sweep.py::quick_test_psnr). Recorded because prior
                # runs recorded it; NOT a train-only signal -- see report caveat.
                quick_test_psnr=record.get("quick_test_psnr"),
                settings=record.get("settings"),
                wall_seconds=meta.get("wall_seconds"),
                cached=bool(meta.get("cache")),
                cache_missing=bool(meta.get("cache_missing")),
                train_record=str(cell_dir / "train_record.json"),
                out_dir=str(cell_dir),
                summary_json=str(cell_dir / "summary.json"),
            ))
    rows.sort(key=lambda r: (r["case"], r["S"]))
    out = Path(root) / "results.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return rows


def report(rows, root, title=""):
    if not rows:
        print("(no rows)")
        return
    cases = sorted({r["case"] for r in rows})
    steps = sorted({r["S"] for r in rows})
    print("\n=== %s (%d rows) %s ===" % (title or "Pareto", len(rows), root))
    print("%-46s %5s %8s %8s %8s %8s %7s %7s %7s" % (
        "case", "S", "q", "B(s)", "t", "total", "psnr", "ssim", "lpips"))
    best = {}
    for case in cases:
        for S in steps:
            row = next((r for r in rows if r["case"] == case and r["S"] == S), None)
            if not row:
                continue
            print("%-46s %5d %8.2f %8.2f %8.2f %8.3f %7.3f %7.4f %7.4f" % (
                case[:46], S, row["quality_points"], row["b_seconds_estimate"],
                row["time_points"] or 0.0, row["total_points"], row["psnr"],
                row["ssim"], row["lpips"]))
            if row["total_points"] is not None:
                if case not in best or row["total_points"] > best[case]["total_points"]:
                    best[case] = row
        print("-" * 120)
    print("\nper-scene argmax(total):")
    for case in cases:
        r = best.get(case)
        if r:
            print("  %-46s S=%5d total=%.3f (q=%.2f B=%.2f t=%.2f)"
                  % (case, r["S"], r["total_points"], r["quality_points"],
                     r["b_seconds_estimate"], r["time_points"] or 0.0))
    print("\nuniform-S mean total over %d scenes:" % len(cases))
    means = {}
    for S in steps:
        vals = [r["total_points"] for r in rows
                if r["S"] == S and r["total_points"] is not None]
        if vals:
            means[S] = sum(vals) / len(vals)
            print("  S=%5d  mean=%.3f  (n=%d)" % (S, means[S], len(vals)))
    if means:
        bestS = max(means, key=means.get)
        print("  best uniform S = %d (mean %.3f)" % (bestS, means[bestS]))
    oracle = [best[c]["total_points"] for c in cases if c in best]
    if oracle:
        print("  per-scene oracle mean = %.3f" % (sum(oracle) / len(oracle)))


# ---------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["parallel", "serial", "report"], default="parallel")
    p.add_argument("--out-root", required=True)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--steps", default=",".join(str(s) for s in STEPS))
    p.add_argument("--cases", default="", help="comma separated case filter (substring match)")
    p.add_argument("--cells-file", default="", help="optional 'case S' whitespace lines")
    p.add_argument("--timeout", type=float, default=2400,
                   help="per-cell wall timeout seconds (default 2400)")
    p.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    p.add_argument("--allow-busy", action="store_true",
                   help="start even if GPUs already have compute processes")
    p.add_argument("--cache-wait", type=float, default=900,
                   help="seconds to poll for <work>/image_cache before running "
                        "without it (default 900)")
    p.add_argument("--no-quick-test", action="store_true", dest="no_quick_test",
                   help="skip the post-train diagnostic quick test (does not affect B; "
                        "use for serial re-timing runs only, concurrent cells keep it)")
    p.add_argument("--build-cache", action="store_true",
                   help="build <work>/image_cache if missing (off by default: the "
                        "h1-prep task owns cache building)")
    p.add_argument("--report", action="store_true", help="print report after the run")
    args = p.parse_args(argv)

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    if not Path(args.manifest).exists():
        raise SystemExit("manifest not found: %s" % args.manifest)
    entries = read_json(args.manifest)
    tags = build_tags(entries)
    if args.cases:
        wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
        entries = [e for e in entries if any(w in e["case"] for w in wanted)]
        if not entries:
            raise SystemExit("case filter matched nothing")
    for entry in entries:
        for key in ("case", "work", "init", "budget_seconds"):
            if key not in entry:
                raise SystemExit("manifest entry missing %r: %r" % (key, entry))
        entry["tag"] = tags[entry["case"]]

    if args.mode == "report":
        rows = rebuild_results(root, tags)
        report(rows, root, "report")
        return 0

    cells = build_cell_list(entries, steps, args.cells_file)
    if args.mode == "serial":
        lanes = 1
    else:
        lanes = 3
    queues = [[] for _ in range(lanes)]
    for idx, (case, S) in enumerate(cells):
        entry = next(e for e in entries if e["case"] == case)
        entry["lane"] = idx % lanes
        queues[idx % lanes].append((entry, S))

    used = gpu_used_mib()
    log(root, "PREFLIGHT mode=%s lanes=%d cells=%d gpus_used_mib=%s manifest=%s"
        % (args.mode, lanes, len(cells), used, args.manifest))
    if used and not args.allow_busy:
        busy = [m for m in used if m > 1500]
        if busy:
            if args.mode == "serial":
                raise SystemExit(
                    "serial timing requested but GPUs busy (used_mib=%r); "
                    "use --allow-busy to override" % used)
            log(root, "WARNING: other compute processes present %r" % used)
    for idx, queue in enumerate(queues):
        log(root, "LANE%d %d cells: %s" % (
            idx, len(queue), ", ".join("%s/s%d" % (e["tag"], S) for e, S in queue)))

    if args.dry_run:
        for q in queues:
            for entry, S in q:
                run_cell(root, entry, S, entry["lane"], args.timeout, dry_run=True,
                         cache_wait=0.0, build_cache=args.build_cache)
        return 0

    threads = [threading.Thread(target=lane_worker, name="lane%d" % i,
                                args=(root, q, entries, args.timeout, False,
                                      args.cache_wait, args.build_cache,
                                      args.no_quick_test))
               for i, q in enumerate(queues)]
    t0 = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    log(root, "ALL_DONE wall=%.1fs" % (time.time() - t0))

    rows = rebuild_results(root, tags)
    failures = 0
    if (root / "failures.jsonl").exists():
        failures = len((root / "failures.jsonl").read_text(encoding="utf-8").splitlines())
    no_cache = sum(1 for r in rows if r.get("cache_missing"))
    log(root, "RESULTS rows=%d failures=%d cache_missing=%d -> %s"
        % (len(rows), failures, no_cache, root / "results.jsonl"))
    if args.report:
        report(rows, root, args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
