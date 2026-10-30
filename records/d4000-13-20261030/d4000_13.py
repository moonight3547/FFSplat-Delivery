"""N1 full-13: 5K-D4000 + VGG200 + scene cache + deterministic M1 gate (task-22)."""
import json, os, shutil, subprocess, time
from pathlib import Path
REPO = Path("/root/projects/FFSplat-Hackathon")
OUT = Path("/root/gpufree-data/outputs/d4000-13-20261030"); OUT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT = OUT / "scene_cache"; CACHE_ROOT.mkdir(parents=True, exist_ok=True)
OLD_CACHE = Path("/root/gpufree-data/outputs/b-eng-20261030/scene_cache")
manifest = json.load(open(REPO / "records/evalnode-manifest-13.json"))
PY = "/opt/conda/envs/fastgs/bin/python"; CONDA = "/opt/conda/bin/conda"

BASE = ["--iterations", "5000", "--snapshots", "", "--profile",
        "competition/profiles/gpufree-2.json", "--seed", "20261015",
        "--position-lr-max-steps", "5000", "--densify-from", "500",
        "--densify-interval", "100", "--opacity-reset-interval", "0",
        "--lambda-dssim", "0.6", "--scaling-lr", "0.01",
        "--realloc-frac", "0.15", "--realloc-at", "1250", "--no-quick-test"]
D4000 = ["--densify-until", "4000", "--grad-abs-thresh", "0.0008"]
VGG200 = ["--lpips-weight", "0.1", "--lpips-scale", "1.0",
          "--lpips-start", "4801", "--lpips-end", "5000"]

import sys; sys.path.insert(0, str(REPO))
from research.scene_cache import version_of  # noqa: E402

def tag_of(case):
    return case.split("/")[-1][:18]

def ensure_cache(case, work):
    tag = tag_of(case)
    # reuse the two bundles built in the B-engineering audit when the hash matches
    for cand in (CACHE_ROOT / tag, OLD_CACHE / tag):
        if (cand / "meta.json").exists():
            import numpy as np
            try:
                ver = str(np.load(cand / "cams.npz", allow_pickle=False)["version"])
            except Exception:  # noqa: BLE001
                continue
            if ver == version_of(work):
                return cand, dict(reused=True, rebuilt=False, cache=str(cand), version=ver)
    dst = CACHE_ROOT / tag
    t0 = time.time()
    r = subprocess.run([PY, "-m", "research.scene_cache", "build", "--work", str(work),
                        "--out", str(dst)], cwd=str(REPO), capture_output=True, text=True)
    print("CACHE", tag, "rc", r.returncode, "%.1fs" % (time.time() - t0), flush=True)
    if r.returncode != 0:
        print((r.stderr or "")[-800:], flush=True)
        return None, dict(rebuilt=False, error=(r.stderr or "")[-500:])
    return dst, dict(reused=False, rebuilt=True,
                     build_seconds_a_side=round(time.time() - t0, 2), cache=str(dst))

rows = []
for e in manifest:
    case = e["case"]; tag = tag_of(case)
    work = REPO / e["work"]; init = REPO / e["init"]; budget = float(e["budget_seconds"])
    cell = OUT / "matrix" / tag
    sweep = cell / "sweep"
    if (sweep / "summary.json").exists():
        print("SKIP", tag, flush=True)
        continue
    cell.mkdir(parents=True, exist_ok=True)
    cache, cache_info = ensure_cache(case, work)
    gate_rec = cell / "gate.json"
    t0 = time.time()
    if not gate_rec.exists():
        g = subprocess.run([CONDA, "run", "--no-capture-output", "-n", "fastgs", "python",
                            "-m", "research.prune_gate", "--work", str(work),
                            "--ply", str(init), "--tau", "0.48", "--keep", "0.6",
                            "--record", str(gate_rec)],
                           cwd=str(REPO), capture_output=True, text=True)
        print("GATE", tag, "rc", g.returncode, "%.1fs" % (time.time() - t0), flush=True)
        if g.returncode != 0:
            print((g.stderr or "")[-600:], flush=True)
    gate = json.load(open(gate_rec)) if gate_rec.exists() else {}
    train_lpips = gate.get("train_lpips"); keep = float(gate.get("keep", 1.0))
    flags = list(BASE) + D4000 + VGG200
    flags += ["--mid-prune-at", "1250", "--mid-prune-keep", str(keep)]
    if cache is not None:
        flags += ["--scene-cache", str(cache)]
    argv = [CONDA, "run", "--no-capture-output", "-n", "fastgs", "python", "-m",
            "research.run_sweep", "--case", case, "--work", str(work),
            "--init-ply", str(init), "--out", str(sweep),
            "--budget", str(budget), "--image-cache", str(work / "image_cache"),
            *flags]
    t0 = time.time()
    with open(cell / "sweep.log", "w") as fh:
        rc = subprocess.run(argv, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT).returncode
    print("CELL", tag, "gate_lpips", train_lpips, "keep", keep, "cache", bool(cache),
          "rc", rc, "%.1fs" % (time.time() - t0), flush=True)
    (cell / "cell.json").write_text(json.dumps(dict(
        case=case, tag=tag, budget=budget, gate=train_lpips, keep=keep,
        cache=cache_info, rc=rc, seconds=round(time.time() - t0, 1)), indent=2))

json.dump(dict(cases=[e["case"] for e in manifest]), open(OUT / "matrix.json", "w"), indent=2)
print("D4000_13_DONE", flush=True)
