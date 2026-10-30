import json, os, subprocess, sys, time
from pathlib import Path
REPO = Path("/root/projects/FFSplat-Hackathon")
OUT = Path("/root/gpufree-data/outputs/bigd-combo-20261030")
OUT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT = Path("/root/gpufree-data/outputs/b-eng-20261030/scene_cache")
manifest = {e["case"]: e for e in json.load(open(REPO / "records/evalnode-manifest-13.json"))}
PY = "/opt/conda/envs/fastgs/bin/python"
CONDA = "/opt/conda/bin/conda"
BUDGET = {"mipnerf360/bicycle": 95.0,
          "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce": 94.29}
CASES = {
    "bicycle": "mipnerf360/bicycle",
    "0b5635": "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce",
}

# ---------- 1) verify (or rebuild) the A-side scene bundles; build time is A-side only
cache_info = {}
for tag, case in CASES.items():
    work = REPO / manifest[case]["work"]
    cache = CACHE_ROOT / tag
    r = subprocess.run([PY, "-m", "research.scene_cache", "verify", "--work", str(work),
                        "--cache", str(cache)], cwd=str(REPO), capture_output=True, text=True)
    ok = r.returncode == 0 and '"identical": true' in (r.stdout or "").replace(" ", "")
    info = dict(verified=ok, verify_stdout=(r.stdout or "")[-400:], rebuilt=False)
    if not ok:
        t0 = time.time()
        b = subprocess.run([PY, "-m", "research.scene_cache", "build", "--work", str(work),
                            "--out", str(cache)], cwd=str(REPO), capture_output=True, text=True)
        info["rebuilt"] = True
        info["build_seconds_a_side"] = round(time.time() - t0, 2)
        info["build_stdout"] = (b.stdout or "")[-600:]
        v = subprocess.run([PY, "-m", "research.scene_cache", "verify", "--work", str(work),
                            "--cache", str(cache)], cwd=str(REPO), capture_output=True, text=True)
        info["reverify_ok"] = '"identical": true' in (v.stdout or "").replace(" ", "")
    cache_info[tag] = info
    print("CACHE", tag, json.dumps(info), flush=True)

BASE = ["--iterations", "5000", "--snapshots", "", "--profile",
        "competition/profiles/gpufree-2.json", "--seed", "20261015",
        "--position-lr-max-steps", "5000", "--densify-from", "500",
        "--densify-interval", "100", "--opacity-reset-interval", "0",
        "--lambda-dssim", "0.6", "--scaling-lr", "0.01",
        "--realloc-frac", "0.15", "--realloc-at", "1250", "--no-quick-test"]
DELIVERY = ["--densify-until", "1250"]
BIGD = ["--densify-until", "2500", "--grad-abs-thresh", "0.0008"]

cells = []
def add(scene, arm, recipe, extra):
    cells.append(dict(scene=scene, arm=arm, recipe=recipe, extra=extra))

for arm, recipe, extra in (
        ("delivery", DELIVERY, []),
        ("bigd", BIGD, []),
        ("bigd_cache", BIGD, ["--scene-cache", str(CACHE_ROOT / "bicycle")]),
        ("bigd_m1", BIGD, ["--mid-prune-at", "1250", "--mid-prune-keep", "0.6"]),
        ("bigd_m1_cache", BIGD, ["--mid-prune-at", "1250", "--mid-prune-keep", "0.6",
                                 "--scene-cache", str(CACHE_ROOT / "bicycle")])):
    add("bicycle", arm, recipe, extra)
for arm, recipe, extra in (
        ("delivery", DELIVERY, []),
        ("bigd", BIGD, []),
        ("bigd_cache", BIGD, ["--scene-cache", str(CACHE_ROOT / "0b5635")])):
    add("0b5635", arm, recipe, extra)

# ---------- 2) serial training matrix (resume-skip on summary.json)
def run_cell(cell):
    tag, arm = cell["scene"], cell["arm"]
    case = CASES[tag]
    e = manifest[case]
    work = REPO / e["work"]
    init = REPO / e["init"]
    cell_dir = OUT / "matrix" / tag / arm
    sweep = cell_dir / "sweep"
    if (sweep / "summary.json").exists():
        print("SKIP", tag, arm, flush=True); return
    cell_dir.mkdir(parents=True, exist_ok=True)
    argv = [CONDA, "run", "--no-capture-output", "-n", "fastgs", "python", "-m",
            "research.run_sweep", "--case", case, "--work", str(work),
            "--init-ply", str(init), "--out", str(sweep),
            "--budget", str(BUDGET[case]), "--image-cache", str(work / "image_cache"),
            *BASE, *cell["recipe"], *cell["extra"]]
    t0 = time.time()
    with open(cell_dir / "sweep.log", "w") as fh:
        rc = subprocess.run(argv, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT).returncode
    print("CELL", tag, arm, "rc", rc, "%.1fs" % (time.time() - t0), flush=True)
    (cell_dir / "cell.json").write_text(json.dumps(dict(rc=rc, seconds=round(time.time() - t0, 1),
                                                        argv=[str(a) for a in argv]), indent=2))

for cell in cells:
    run_cell(cell)
json.dump(dict(cache=cache_info, cells=cells), open(OUT / "matrix.json", "w"), indent=2)
print("BIGD_COMBO_DONE", flush=True)
