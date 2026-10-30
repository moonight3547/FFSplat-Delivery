#!/usr/bin/env python3
"""Staged post-5000 post-pass driver for N1 (task-8).

Phase 1 = delivery recipe (`research.run_sweep`, 5000 steps, FastGS 1-cam
training and the measured delivery schedule).  Phase 2 = an EFFICIENT post-pass
(same loss, extra steps) and phase 3 = +100 VGG-LPIPS steps.

Arms
  absgrad       phase 2 = `--stage2_trainer absgrad`, K steps in-process
                (net-zero relocation on the 2D-projected gradient norm).
  absgrad_vgg   absgrad + 100 VGG-LPIPS steps (phase 3, same process).
  mcmc          chain: phase-1 run_sweep -> `research.mcmc_budget` warm-started
                from the phase-1 PLY (3DGS-MCMC relocation) -> eval.
  mcmc_vgg      mcmc + a phase-3 run_sweep warm-started from the MCMC PLY
                (`--iterations 0 --stage3_iters 100`), +100 VGG-LPIPS steps.
  standard      1-cam L1+DSSIM continuation (the user's original proposal, kept
                as an ablation only), K_s steps in-process.
  standard1000  standard with 1000 steps (2-scene ablation).
  ctrl          phase-1 only (used for the serial ctrl re-run).

Every cell also yields its own phase-1 row (iter 5000) so the per-scene
comparison is paired on the same node, inputs and run.  B is module-B training
time: phase-1 recipe B plus every post-stage (chained stages add the extra
process setup honestly).

Modes
  parallel : 3 lanes, one per GPU profile competition/profiles/gpufree-{0,1,2}.json
  serial   : one lane (GPU 0) for clean B seconds
  report   : print the aggregate tables from existing summaries only

Resumable: a cell whose cell.json says ok is skipped.
"""
import argparse
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
SCENES = (
    "mipnerf360/bicycle",
    "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce",
    "dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e",
    "dl3dv/3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3",
)
# arities: steps/budget are filled from CLI
DEFAULT_ARMS = {
    "absgrad": dict(kind="train", trainer="absgrad", stage2_iters=800, stage3=0),
    "absgrad_vgg": dict(kind="train", trainer="absgrad", stage2_iters=800, stage3=100),
    "mcmc": dict(kind="mcmc", budget=11.0, stage3=0),
    "mcmc_vgg": dict(kind="mcmc", budget=11.0, stage3=100),
    "standard": dict(kind="train", trainer="standard", stage2_iters=1000, stage3=0),
    "standard1000": dict(kind="train", trainer="standard", stage2_iters=1000, stage3=0),
    # absgrad step-count / LR sweep (bicycle only): lr_from_ratio 0.1 makes the
    # stage-2 start LR 10x the phase-1 final LR (7.95e-5 on bicycle) and it still
    # decays to 0.1x of that = 7.95e-6.
    "absgrad_1000": dict(kind="train", trainer="absgrad", stage2_iters=1000, stage3=0),
    "absgrad_2000": dict(kind="train", trainer="absgrad", stage2_iters=2000, stage3=0),
    "absgrad_3000": dict(kind="train", trainer="absgrad", stage2_iters=3000, stage3=0),
    "absgrad_5000": dict(kind="train", trainer="absgrad", stage2_iters=5000, stage3=0),
    "absgrad_2000_lr10": dict(kind="train", trainer="absgrad", stage2_iters=2000,
                              stage3=0, lr_from_ratio=0.1),
    "absgrad_5000_lr10": dict(kind="train", trainer="absgrad", stage2_iters=5000,
                              stage3=0, lr_from_ratio=0.1),
    # ---- Mini-Splatting density control (task-10).  m2_*: delivery then prune to
    # <keep> by train-view importance + 400 standard refine steps.
    "m2_50": dict(kind="train", trainer="prune", stage2_iters=400, stage3=0,
                  prune_keep=0.5, prune_views=24),
    "m2_70": dict(kind="train", trainer="prune", stage2_iters=400, stage3=0,
                  prune_keep=0.7, prune_views=24),
    "m2_90": dict(kind="train", trainer="prune", stage2_iters=400, stage3=0,
                  prune_keep=0.9, prune_views=24),
    # m1_*: one 5000-step run with a Mini-Splatting prune in the middle, so the
    # remaining steps train a smaller model (the B-time lever).
    "m1_1250_80": dict(kind="train", trainer="standard", stage2_iters=0, stage3=0,
                       mid_prune_at=1250, mid_prune_keep=0.8, mid_prune_views=24),
    "m1_1250_70": dict(kind="train", trainer="standard", stage2_iters=0, stage3=0,
                       mid_prune_at=1250, mid_prune_keep=0.7, mid_prune_views=24),
    "m1_1250_80": dict(kind="train", trainer="standard", stage2_iters=0, stage3=0,
                       mid_prune_at=1250, mid_prune_keep=0.8, mid_prune_views=24),
    "m1_1250_60": dict(kind="train", trainer="standard", stage2_iters=0, stage3=0,
                       mid_prune_at=1250, mid_prune_keep=0.6, mid_prune_views=24),
    "m1_1250_50": dict(kind="train", trainer="standard", stage2_iters=0, stage3=0,
                       mid_prune_at=1250, mid_prune_keep=0.5, mid_prune_views=24),
    "m1_1250_40": dict(kind="train", trainer="standard", stage2_iters=0, stage3=0,
                       mid_prune_at=1250, mid_prune_keep=0.4, mid_prune_views=24),
    "m1_1250_90": dict(kind="train", trainer="standard", stage2_iters=0, stage3=0,
                       mid_prune_at=1250, mid_prune_keep=0.9, mid_prune_views=24),
    "m1_2000_70": dict(kind="train", trainer="standard", stage2_iters=0, stage3=0,
                       mid_prune_at=2000, mid_prune_keep=0.7, mid_prune_views=24),
    "ctrl": dict(kind="train", trainer="standard", stage2_iters=0, stage3=0),
    # +100 VGG-LPIPS steps directly on top of the delivery recipe (no phase 2):
    # the Lead's priority-2 arm because the MCMC pilot was already negative.
    "vgg": dict(kind="train", trainer="standard", stage2_iters=0, stage3=100),
    # VGG-stage step-count sweep (bicycle): the LPIPS trajectory 0.2989->0.2892 in
    # 100 steps may cross the 0.25 scoring cliff within a few hundred steps.
    "vgg_200": dict(kind="train", trainer="standard", stage2_iters=0, stage3=200),
    "vgg_400": dict(kind="train", trainer="standard", stage2_iters=0, stage3=400),
    "vgg_500": dict(kind="train", trainer="standard", stage2_iters=0, stage3=500),
    "vgg_800": dict(kind="train", trainer="standard", stage2_iters=0, stage3=800),
    "vgg_1000": dict(kind="train", trainer="standard", stage2_iters=0, stage3=1000),
}
# Auto MCMC wall budgets (seconds).  3DGS-MCMC only starts relocating after one
# warm-up epoch (steps > num_train_views), so the budget is calibrated per scene
# to just clear that gate; measured on N1 at ~27 ms/step incl. image decode.
MCMC_BUDGET_AUTO = {
    "mip-bicycle": 13.0,          # 169 views
    "dl3dv-0b5635d53b3b": 19.0,   # 333 views
    "dl3dv-0f622eaabc6d": 18.0,   # 283 views
    "dl3dv-3e4dbe4135ad": 17.0,   # 268 views
}
PRINT_LOCK = threading.Lock()


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


def proxy_scores(psnr, ssim, lpips, b_seconds, budget_seconds):
    """Organizer's linear /100 proxy (identical to research/run_sweep.py)."""
    out = {}
    out["psnr_points"] = 20.0 * max(0.0, min(1.0, (psnr - 15.0) / 25.0))
    out["ssim_points"] = 20.0 * max(0.0, min(1.0, (ssim - 0.70) / 0.25))
    out["lpips_points"] = 40.0 * max(0.0, min(1.0, (0.25 - lpips) / 0.20))
    if b_seconds is None or budget_seconds is None:
        out["time_points"] = None
    elif b_seconds >= budget_seconds:
        out["time_points"] = 0.0
    else:
        out["time_points"] = 20.0 * min(1.0, (1.0 - b_seconds / budget_seconds) / 0.9)
    out["quality_points"] = (out["psnr_points"] + out["ssim_points"]
                             + out["lpips_points"])
    out["total_points"] = out["quality_points"] + (out["time_points"] or 0.0)
    out["valid"] = (b_seconds < budget_seconds) if b_seconds and budget_seconds else None
    return out


def run_command(cmd, logfile, timeout, env_extra=None):
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "ab") as fh:
        fh.write(("\n+ " + " ".join(map(str, cmd)) + "\n").encode())
        fh.flush()
        env = dict(os.environ, PYTHONPATH=str(REPO))
        if env_extra:
            env.update(env_extra)
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=fh,
                                stderr=subprocess.STDOUT, start_new_session=True,
                                env=env)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            proc.wait()
            return 124


def tag_for(case):
    kind, _, name = case.partition("/")
    if kind == "dl3dv":
        return "dl3dv-" + name[:12]
    return kind + "-" + name


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)


def gpu_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    return [int(x) for x in out.split() if x.isdigit()]


# ---------------------------------------------------------------- run_sweep
def run_sweep(cell_dir, entry, lane, stage_flags, timeout, log_name,
              iterations=5000, no_quick_test=True):
    cell_dir = Path(cell_dir)
    cmd = [CONDA, "run", "--no-capture-output", "-n", "fastgs", "python", "-m",
           "research.run_sweep",
           "--profile", "competition/profiles/gpufree-%d.json" % lane,
           "--case", entry["case"],
           "--work", str(resolve(entry["work"])),
           "--init-ply", str(resolve(entry["init"])),
           "--out", str(cell_dir),
           "--iterations", str(iterations),
           "--snapshots", "",
           "--budget", str(entry["budget_seconds"]),
           "--position-lr-max-steps", "5000",
           "--opacity-reset-interval", "0",
           "--lambda-dssim", "0.6",
           "--scaling-lr", "0.01",
           "--densify-from", "500",
           "--densify-until", "1250",
           "--realloc-frac", "0.15",
           "--realloc-at", "1250"]
    cache = resolve(entry["work"]) / "image_cache"
    if (cache / "manifest.json").exists() and (cache / "images_u8.npy").exists():
        cmd += ["--image-cache", str(cache)]
    for flag, value in stage_flags.items():
        if value is None:
            continue
        cmd += [flag, str(value)]
    if no_quick_test:
        cmd += ["--no-quick-test"]
    rc = run_command(cmd, cell_dir.parent / "logs" / (log_name + ".log"), timeout)
    return rc, cmd


# ------------------------------------------------------------------- eval
def eval_ply(cell_dir, entry, ply, name, timeout, lane=0):
    """Render `ply` on the official test cameras and score it with evaluate.py.

    The subprocesses do not go through run_sweep's profile, so pin them to the
    lane GPU explicitly (otherwise every lane evaluates on cuda:0).
    """
    cell_dir = Path(cell_dir)
    env = {"CUDA_VISIBLE_DEVICES": str(lane)}
    ev = cell_dir / "eval"
    cmd = [CONDA, "run", "--no-capture-output", "-n", "fastgs", "python", "-m",
           "research.render_eval",
           "--source=" + str(resolve(entry["work"])), "--out=" + str(ev),
           "--eval", "-r", "1", "--sh_degree", "3",
           "--ply", str(ply), "--name", name]
    rc = run_command(cmd, cell_dir / "logs" / ("render_%s.log" % name), timeout,
                     env_extra=env)
    if rc != 0:
        return None
    target = ev / name
    cmd = [CONDA, "run", "--no-capture-output", "-n", "fastgs", "python",
           str(REPO / "competition/upstream/baseline/evaluate.py"),
           "--model_path", str(target), "--device", "cuda:0"]
    rc = run_command(cmd, cell_dir / "logs" / ("evaluate_%s.log" % name), timeout,
                     env_extra=env)
    if rc != 0:
        return None
    try:
        return read_json(target / "metrics.json")["mean"]
    except Exception:
        return None


# --------------------------------------------------------------- cell types
def stage_rows_from_summary(cell_dir, arm):
    """Turn a run_sweep summary into (ctrl, stage2, stage3) rows."""
    summary = read_json(Path(cell_dir) / "summary.json")
    # summary.json drops the snapshot list; the full record on disk keeps it
    record_file = Path(cell_dir) / "train_record.json"
    if record_file.exists():
        record = read_json(record_file)
    else:
        record = summary["train_record"]
    snaps = {s["iteration"]: s for s in record.get("snapshots", [])}
    rows = []
    for res in summary["results"]:
        it = res["iteration"]
        snap = snaps.get(it, {})
        stage = snap.get("stage") or ("stage1" if it == 5000 else "unknown")
        rows.append(dict(
            arm=arm, stage=stage, iteration=it,
            num_gaussians=res["num_gaussians"],
            train_seconds=res["train_seconds"],
            b_seconds=res["b_seconds_estimate"],
            psnr=res["psnr"], ssim=res["ssim"], lpips=res["lpips"],
            score=res["score"], snap=snap,
            scene_load_seconds=record.get("scene_load_seconds"),
            init_seconds=record.get("init_seconds"),
            stage1_seconds=record.get("stage1_seconds"),
            stage2_seconds=record.get("stage2_seconds"),
            stage3_seconds=record.get("stage3_seconds"),
            stage1_iterations=record.get("stage1_iterations"),
            stage2_iterations=record.get("stage2_iterations"),
            stage3_iterations=record.get("stage3_iterations"),
            train_record=str(Path(cell_dir) / "train_record.json"),
            summary_json=str(Path(cell_dir) / "summary.json"),
        ))
    return rows, record


def cell_train(root, entry, arm, spec, lane, timeout, stage3_iters=None,
               stage3_weight=0.1):
    cell = Path(root) / entry["tag"] / arm
    # RESUME-SKIP: a completed run_sweep (summary.json) is authoritative, even if
    # the driver died before writing cell.json/rows.json.  Never delete the
    # out-root from a launcher; this is what makes restarts safe.
    if (cell / "sweep" / "summary.json").exists():
        meta = read_json(cell / "cell.json") if (cell / "cell.json").exists() else {}
        if not (cell / "rows.json").exists():
            try:
                rows, record = stage_rows_from_summary(cell / "sweep", arm)
                for row in rows:
                    row["case"] = entry["case"]
                    row["budget_seconds"] = entry["budget_seconds"]
                write_json(cell / "rows.json", rows)
                meta.update(kind="train", arm=arm, lane=lane, case=entry["case"],
                            status="ok", resumed=True, record=record)
                write_json(cell / "cell.json", meta)
            except Exception as exc:  # noqa: BLE001
                log(root, "RESUME_FAIL lane%d %s %s: %r"
                    % (lane, entry["tag"], arm, exc))
        log(root, "SKIP lane%d %s %s (summary.json exists)" % (lane, entry["tag"], arm))
        return meta
    if cell.exists():
        stale = cell.with_name(cell.name + ".stale-%d" % int(time.time()))
        shutil.move(str(cell), str(stale))
    cell.mkdir(parents=True, exist_ok=True)
    # the CLI override only re-sizes arms that already carry a VGG phase, so
    # absgrad != absgrad_vgg and ctrl stays a pure phase-1 cell
    # named vgg_<n> arms keep their own step count; the generic vgg arm follows the CLI
    if arm == "vgg":
        s3 = stage3_iters if stage3_iters is not None else spec["stage3"]
    else:
        s3 = spec["stage3"]
    if arm == "ctrl":
        s3 = 0
    flags = {
        "--stage2-iters": spec["stage2_iters"],
        "--stage2-trainer": spec["trainer"],
        "--stage2-cams": 1,
        "--stage3-iters": s3 or None,
        "--stage3-lpips-weight": (stage3_weight if s3 else None),
        "--stage2-lr-from-ratio": spec.get("lr_from_ratio"),
    }
    if spec.get("prune_keep") is not None:
        flags["--prune-keep"] = spec["prune_keep"]
        flags["--prune-views"] = spec.get("prune_views", 24)
    if spec.get("mid_prune_at"):
        flags["--mid-prune-at"] = spec["mid_prune_at"]
        flags["--mid-prune-keep"] = spec.get("mid_prune_keep", 1.0)
        flags["--mid-prune-views"] = spec.get("mid_prune_views", 24)
    # run_sweep refuses a non-empty --out dir, so the trainer output lives in
    # its own subdir (cell.json/rows.json stay in the cell dir)
    sweep = cell / "sweep"
    meta = dict(kind="train", arm=arm, lane=lane, case=entry["case"],
                spec=spec, stage3_iters=s3, stage3_weight=stage3_weight,
                status="running", started=now())
    write_json(cell / "cell.json", meta)
    log(root, "START lane%d %s %s (trainer=%s stage2=%s stage3=%s)"
        % (lane, entry["tag"], arm, spec["trainer"], spec["stage2_iters"], s3))
    t0 = time.time()
    rc, cmd = run_sweep(sweep, entry, lane, flags, timeout, "sweep")
    meta["cmd"] = [str(c) for c in cmd]
    meta["sweep_rc"] = rc
    meta["wall_seconds"] = round(time.time() - t0, 1)
    rows = []
    if rc == 0 and (sweep / "summary.json").exists():
        try:
            rows, record = stage_rows_from_summary(sweep, arm)
            meta["record"] = record
            meta["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            meta["status"] = "fail"
            meta["error"] = repr(exc)
    else:
        meta["status"] = "fail"
    meta["finished"] = now()
    write_json(cell / "cell.json", meta)
    for row in rows:
        row["case"] = entry["case"]
        row["budget_seconds"] = entry["budget_seconds"]
    write_json(cell / "rows.json", rows)
    log(root, "END lane%d %s %s rc=%s wall=%.1fs rows=%d status=%s"
        % (lane, entry["tag"], arm, rc, meta["wall_seconds"], len(rows),
           meta["status"]))
    return meta


def cell_mcmc(root, entry, arm, spec, lane, timeout, stage3_iters=None,
              stage3_weight=0.1, stage3_lr_ratio=0.01, no_prefix=False):
    """phase1 (run_sweep) -> research.mcmc_budget warm start -> optional phase3."""
    cell = Path(root) / entry["tag"] / arm
    if (cell / "cell.json").exists():
        meta = read_json(cell / "cell.json")
        if meta.get("status") == "ok" and (cell / "rows.json").exists():
            log(root, "SKIP lane%d %s %s (ok)" % (lane, entry["tag"], arm))
            return meta
    if cell.exists():
        stale = cell.with_name(cell.name + ".stale-%d" % int(time.time()))
        shutil.move(str(cell), str(stale))
    cell.mkdir(parents=True, exist_ok=True)
    s3 = spec["stage3"] if stage3_iters is None else stage3_iters
    budget = spec["budget"] if spec["budget"] > 0 else MCMC_BUDGET_AUTO.get(
        entry["tag"], 16.0)
    meta = dict(kind="mcmc", arm=arm, lane=lane, case=entry["case"],
                spec=spec, mcmc_budget=budget, stage3_iters=s3,
                status="running", started=now())
    write_json(cell / "cell.json", meta)
    log(root, "START lane%d %s %s (mcmc budget=%ss stage3=%s)"
        % (lane, entry["tag"], arm, budget, s3))
    t0 = time.time()

    # ---- phase 1: delivery recipe (also yields the paired ctrl row)
    p1 = cell / "phase1"
    rc1, cmd1 = run_sweep(p1, entry, lane, {}, timeout, "phase1")
    meta["phase1_cmd"] = [str(c) for c in cmd1]
    if rc1 != 0 or not (p1 / "summary.json").exists():
        meta["status"] = "fail"
        meta["finished"] = now()
        write_json(cell / "cell.json", meta)
        log(root, "FAIL lane%d %s %s phase1 rc=%s" % (lane, entry["tag"], arm, rc1))
        return meta
    try:
        rows, p1_record = stage_rows_from_summary(p1, arm)
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "fail"
        meta["error"] = "phase1 parse: %r" % (exc,)
        write_json(cell / "cell.json", meta)
        return meta
    ctrl_row = rows[-1] if rows else None
    if ctrl_row is None:
        meta["status"] = "fail"
        meta["error"] = "phase1 produced no rows"
        write_json(cell / "cell.json", meta)
        return meta
    b1 = ctrl_row["b_seconds"]
    ply1 = Path(p1_record["final_ply"])
    if not ply1.is_absolute():
        ply1 = REPO / ply1

    # ---- phase 2: 3DGS-MCMC warm start from the phase-1 PLY
    mcmc_out = cell / "mcmc"
    cmd2 = [CONDA, "run", "--no-capture-output", "-n", "fastgs", "python", "-m",
            "research.mcmc_budget",
            "--work", str(resolve(entry["work"])),
            "--init", str(ply1),
            "--out", str(mcmc_out),
            "--budget-seconds", str(budget),
            "--budget-fraction", "0.9",
            "--res-scales", "1.0",
            "--cap-mult", "2.0",
            "--refine-every", "100",
            "--sh-degree", "3",
            "--seed", "20261015"]
    rc2 = run_command(cmd2, cell / "logs" / "mcmc.log", timeout,
                      env_extra={"CUDA_VISIBLE_DEVICES": str(lane)})
    meta["mcmc_cmd"] = [str(c) for c in cmd2]
    meta["mcmc_rc"] = rc2
    if rc2 == 0 and (mcmc_out / "module_b.json").exists():
        mb = read_json(mcmc_out / "module_b.json")
        meta["mcmc"] = {k: mb.get(k) for k in (
            "steps", "total_seconds", "setup_seconds", "train_seconds",
            "export_seconds", "num_init_gaussians", "num_final_gaussians",
            "stop_reason")}
        b2 = float(mb["total_seconds"])
        # align B with the FastGS convention used by research.run_sweep:
        # scene_load + init + train + 0.5 (the PLY export is the flat 0.5, exactly
        # as for the delivery run), so do not charge MCMC's own export again.
        b2_aligned = (float(mb.get("setup_seconds", 0.0))
                      + float(mb.get("train_seconds", 0.0)) + 0.5)
        ply2 = mcmc_out / mb["ply"]
        # score the stage-2 model (needed even when phase 3 also runs)
        name2 = "iter%05d" % int(mb.get("steps", 0))
        metrics2 = eval_ply(cell, entry, ply2, name2, timeout, lane=lane)
        row2 = None
        if metrics2 is not None:
            b_total = b1 + b2_aligned
            sc = proxy_scores(metrics2["psnr"], metrics2["ssim"],
                              metrics2["lpips"], b_total,
                              entry["budget_seconds"])
            row2 = dict(arm=arm, stage="stage2", iteration=int(mb.get("steps", 0)),
                        num_gaussians=int(mb.get("num_final_gaussians", 0)),
                        train_seconds=round(ctrl_row["train_seconds"]
                                            + float(mb.get("setup_seconds", 0.0))
                                            + float(mb.get("train_seconds", 0.0)), 3),
                        b_seconds=round(b_total, 2),
                        psnr=metrics2["psnr"], ssim=metrics2["ssim"],
                        lpips=metrics2["lpips"], score=sc, snap=None,
                        stage1_seconds=ctrl_row["stage1_seconds"],
                        stage2_seconds=round(b2_aligned, 3), stage3_seconds=0.0,
                        mcmc_steps=int(mb.get("steps", 0)),
                        mcmc_setup_seconds=mb.get("setup_seconds"),
                        mcmc_train_seconds=mb.get("train_seconds"),
                        mcmc_export_seconds=mb.get("export_seconds"),
                        mcmc_total_seconds=b2, mcmc_b_aligned=round(b2_aligned, 3),
                        evaluation=metrics2,
                        summary_json=str(p1 / "summary.json"))
        else:
            meta["error"] = "mcmc eval failed"
            row2 = None
        # ---- phase 3: +100 VGG-LPIPS steps warm-started from the MCMC PLY
        row3 = None
        if s3 and row2 is not None:
            p3 = cell / "phase3"
            flags3 = {
                "--stage3-iters": s3,
                "--stage3-lpips-weight": stage3_weight,
                "--stage2-lr-from-ratio": stage3_lr_ratio,
                "--stage2-lr-final": None,
            }
            rc3, cmd3 = run_sweep(p3, entry, lane, flags3, timeout, "phase3",
                                  iterations=0)
            meta["phase3_cmd"] = [str(c) for c in cmd3]
            meta["phase3_rc"] = rc3
            if rc3 == 0 and (p3 / "summary.json").exists():
                try:
                    rows3, rec3 = stage_rows_from_summary(p3, arm)
                except Exception:  # noqa: BLE001
                    rows3, rec3 = [], None
                if rows3:
                    r3 = rows3[-1]
                    b3 = r3["b_seconds"]
                    b_total = b1 + b2_aligned + b3
                    sc = proxy_scores(r3["psnr"], r3["ssim"], r3["lpips"],
                                      b_total, entry["budget_seconds"])
                    row3 = dict(r3, stage="stage3", arm=arm,
                                b_seconds=round(b_total, 2),
                                stage2_seconds=round(b2_aligned, 3),
                                stage3_seconds=r3["stage3_seconds"],
                                mcmc_steps=int(mb.get("steps", 0)),
                                score=sc)
        meta["rows"] = [ctrl_row, row2, row3]
        meta["status"] = "ok" if row2 is not None else "fail"
    else:
        meta["status"] = "fail"
    meta["wall_seconds"] = round(time.time() - t0, 1)
    meta["finished"] = now()
    write_json(cell / "cell.json", meta)
    rows = [r for r in (meta.get("rows") or []) if r is not None]
    write_json(cell / "rows.json", rows)
    log(root, "END lane%d %s %s wall=%.1fs rows=%d status=%s"
        % (lane, entry["tag"], arm, meta["wall_seconds"], len(rows),
           meta["status"]))
    return meta


# ------------------------------------------------------------------ results
def collect_rows(root, tags):
    rows = []
    for case, tag in tags.items():
        for cell_dir in sorted((Path(root) / tag).glob("*")):
            rows_file = cell_dir / "rows.json"
            if not rows_file.exists():
                continue
            meta_file = cell_dir / "cell.json"
            meta = read_json(meta_file) if meta_file.exists() else {}
            if meta.get("status") != "ok":
                continue
            for row in read_json(rows_file):
                row["case"] = case
                row["budget_seconds"] = meta.get("budget", row.get("budget_seconds"))
                row["lane"] = meta.get("lane")
                row["out_dir"] = str(cell_dir)
                rows.append(row)
    rows.sort(key=lambda r: (r["case"], r["arm"], r["iteration"]))
    return rows


def report(rows):
    if not rows:
        print("(no rows)")
        return
    cases = sorted({r["case"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    print("\n=== per cell (paired on the same run; B includes every stage) ===")
    print("%-24s %-13s %-6s %6s %6s %6s %6s %7s %7s %7s %7s %8s" % (
        "case", "arm", "stage", "N", "s1", "s2", "s3", "B", "psnr", "ssim",
        "lpips", "total"))
    for case in cases:
        for arm in arms:
            for r in [x for x in rows if x["case"] == case and x["arm"] == arm]:
                score = r.get("score") or {}
                print("%-24s %-13s %-6s %6d %6.1f %6.1f %6.1f %7.2f %7.3f %7.4f "
                      "%7.4f %8.3f" % (
                          case.split("/")[-1][:24], arm, r["stage"][-6:],
                          r["num_gaussians"], r.get("stage1_seconds") or 0.0,
                          r.get("stage2_seconds") or 0.0,
                          r.get("stage3_seconds") or 0.0,
                          r["b_seconds"] or 0.0, r["psnr"], r["ssim"], r["lpips"],
                          score.get("total_points") or 0.0))
    print("\n=== per-scene delta vs paired ctrl (total points) ===")
    print("%-24s %-28s %8s %8s %8s %8s" % (
        "case", "arm/stage", "ctrl", "arm", "d_total", "d_B"))
    for case in cases:
        ctrl = next((r for r in rows if r["case"] == case and r["stage"] == "stage1"),
                    None)
        if not ctrl:
            continue
        for arm in arms:
            for r in [x for x in rows if x["case"] == case and x["arm"] == arm
                      and x["stage"] != "stage1"]:
                d_total = ((r["score"] or {}).get("total_points") or 0.0) - \
                          ((ctrl["score"] or {}).get("total_points") or 0.0)
                d_b = (r["b_seconds"] or 0.0) - (ctrl["b_seconds"] or 0.0)
                print("%-24s %-28s %8.3f %8.3f %+8.3f %+8.2f" % (
                    case.split("/")[-1][:24], arm + "/" + r["stage"],
                    (ctrl["score"] or {}).get("total_points") or 0.0,
                    (r["score"] or {}).get("total_points") or 0.0, d_total, d_b))
    print("\n=== arm means over scenes that have a paired ctrl ===")
    for arm in arms:
        for stage in ("stage1", "stage2", "stage3"):
            vals = []
            qps = []
            for case in cases:
                ctrl = next((r for r in rows if r["case"] == case
                             and r["stage"] == "stage1"), None)
                r = next((x for x in rows if x["case"] == case and x["arm"] == arm
                          and x["stage"] == stage), None)
                if not ctrl or not r:
                    continue
                vals.append(((r["score"] or {}).get("total_points") or 0.0)
                            - ((ctrl["score"] or {}).get("total_points") or 0.0))
                qps.append(((r["score"] or {}).get("quality_points") or 0.0)
                           - ((ctrl["score"] or {}).get("quality_points") or 0.0))
            if vals:
                print("  %-13s %-6s n=%d  mean_d_total=%+.3f  mean_d_quality=%+.3f"
                      % (arm, stage, len(vals), sum(vals) / len(vals),
                         sum(qps) / len(qps)))


# ---------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["parallel", "serial", "report"], default="parallel")
    p.add_argument("--lanes", type=int, default=0,
                   help="lane count (0 = 1 for serial, 3 for parallel); this host "
                        "inflates phase-1 B when >=3 lanes compete for power")
    p.add_argument("--out-root", required=True)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--cases", default="", help="comma separated case filter")
    p.add_argument("--cells-file", default="",
                   help="optional '<case> <arm> [lane]' lines (overrides the matrix)")
    p.add_argument("--arms", default="absgrad,absgrad_vgg,mcmc,mcmc_vgg",
                   help="comma separated arms for the default matrix")
    p.add_argument("--stage2-iters", type=int, default=800, dest="stage2_iters",
                   help="absgrad phase-2 step count")
    p.add_argument("--standard-iters", type=int, default=1000, dest="standard_iters",
                   help="1-cam standard phase-2 step count (ablation arm only)")
    p.add_argument("--mcmc-budget", type=float, default=0.0,
                   help="research.mcmc_budget --budget-seconds; 0 = per-scene auto "
                        "(MCMC_BUDGET_AUTO, calibrated to clear the warm-up epoch)")
    p.add_argument("--stage3-iters", type=int, default=100, dest="stage3_iters")
    p.add_argument("--stage3-weight", type=float, default=0.1, dest="stage3_weight")
    p.add_argument("--timeout", type=float, default=2400)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--report", action="store_true")
    args = p.parse_args(argv)

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    entries = read_json(args.manifest)
    if args.cases:
        wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
        entries = [e for e in entries if any(w in e["case"] for w in wanted)]
    entries = [e for e in entries if e["case"] in SCENES] if not args.cells_file else entries
    if not entries:
        raise SystemExit("no matching manifest entries")
    for e in entries:
        e["tag"] = tag_for(e["case"])
    tags = {e["case"]: e["tag"] for e in entries}

    arms = {}

    def spec_for(arm):
        """Arm spec, so a --cells-file may name any arm from DEFAULT_ARMS."""
        if arm in arms:
            return arms[arm]
        if arm not in DEFAULT_ARMS:
            raise SystemExit("unknown arm %r (have %s)"
                             % (arm, sorted(DEFAULT_ARMS)))
        spec = dict(DEFAULT_ARMS[arm])
        if spec["kind"] == "train" and spec["trainer"] in ("absgrad", "standard"):
            if arm in ("standard", "standard1000"):
                spec["stage2_iters"] = args.standard_iters
            elif arm in ("ctrl", "vgg") or arm.startswith("vgg_"):
                spec["stage2_iters"] = 0
            elif arm == "absgrad":
                spec["stage2_iters"] = args.stage2_iters
            # named absgrad_<n>[_lr10] arms keep their own step count
        if spec["kind"] == "mcmc":
            spec["budget"] = args.mcmc_budget
        arms[arm] = spec
        return spec

    for _name in args.arms.split(","):
        if _name.strip():
            spec_for(_name.strip())

    if args.mode == "report":
        report(collect_rows(root, tags))
        return 0

    # ---- cell queue
    cells = []
    if args.cells_file:
        for raw in Path(args.cells_file).read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            case, arm = parts[0], parts[1]
            lane = int(parts[2]) if len(parts) > 2 else None
            spec_for(arm)  # register the arm even if --arms did not list it
            cells.append((case, arm, lane))
    else:
        for case in [e["case"] for e in entries]:
            for arm in arms:
                cells.append((case, arm, None))
    lanes = 1 if args.mode == "serial" else 3
    if args.lanes > 0:
        lanes = args.lanes
    # one queue per node GPU profile (gpufree-0/1/2 -> CUDA_VISIBLE_DEVICES 0/1/2);
    # a cells file may pin a cell to any of them (e.g. lane 2 = the free GPU)
    queues = [[] for _ in range(3)]
    for idx, (case, arm, lane_hint) in enumerate(cells):
        lane = lane_hint if lane_hint is not None else idx % lanes
        if not 0 <= lane < 3:
            raise SystemExit("lane %r out of range (profiles 0..2)" % (lane,))
        entry = next(e for e in entries if e["case"] == case)
        queues[lane].append((entry, arm, lane))
    used = gpu_used_mib()
    log(root, "PREFLIGHT mode=%s lanes=%d cells=%d gpus_used_mib=%s"
        % (args.mode, lanes, len(cells), used))
    if used and not args.allow_busy and args.mode == "serial":
        raise SystemExit("serial run requested but GPUs busy: %r" % used)
    if args.dry_run:
        for q in queues:
            for entry, arm, lane in q:
                print("DRY lane%d %s %s %r" % (lane, entry["tag"], arm, arms[arm]))
        return 0

    def worker(queue):
        for entry, arm, lane in queue:
            spec = spec_for(arm)
            try:
                if spec["kind"] == "mcmc":
                    cell_mcmc(root, entry, arm, spec, lane, args.timeout,
                              stage3_iters=args.stage3_iters,
                              stage3_weight=args.stage3_weight)
                else:
                    cell_train(root, entry, arm, spec, lane, args.timeout,
                               stage3_iters=args.stage3_iters,
                               stage3_weight=args.stage3_weight)
            except Exception as exc:  # noqa: BLE001
                log(root, "FAIL lane%d %s %s: %r"
                    % (lane, entry["tag"], arm, exc))

    threads = [threading.Thread(target=worker, name="lane%d" % i,
                                args=(q,)) for i, q in enumerate(queues) if q]
    t0 = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    log(root, "ALL_DONE wall=%.1fs" % (time.time() - t0))
    rows = collect_rows(root, tags)
    (root / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    log(root, "RESULTS rows=%d -> %s" % (len(rows), root / "results.jsonl"))
    if args.report:
        report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
