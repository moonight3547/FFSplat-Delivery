import json, os, subprocess, time
from pathlib import Path
REPO = Path("/root/projects/FFSplat-Hackathon")
OUT = Path("/root/gpufree-data/outputs/d3500-combo-20261030"); OUT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT = Path("/root/gpufree-data/outputs/b-eng-20261030/scene_cache")
manifest = {e["case"]: e for e in json.load(open(REPO / "records/evalnode-manifest-13.json"))}
PY = "/opt/conda/envs/fastgs/bin/python"; CONDA = "/opt/conda/bin/conda"
BUDGET = {"mipnerf360/bicycle": 95.0,
          "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce": 94.29}
CASES = {"bicycle": "mipnerf360/bicycle",
         "0b5635": "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce"}
import sys
sys.path.insert(0, str(REPO))
from research.scene_cache import version_of  # noqa: E402

cache_info = {}
for tag, case in CASES.items():
    work = REPO / manifest[case]["work"]; cache = CACHE_ROOT / tag
    info = dict(reused=False, rebuilt=False)
    try:
        import numpy as np
        ver = str(np.load(cache / "cams.npz", allow_pickle=False)["version"])
        current = version_of(work)
        info["version"] = ver
        info["hash_match"] = (ver == current)
        if not info["hash_match"]:
            t0 = time.time()
            r = subprocess.run([PY, "-m", "research.scene_cache", "build", "--work", str(work),
                                "--out", str(cache)], cwd=str(REPO), capture_output=True, text=True)
            info.update(rebuilt=(r.returncode == 0),
                        build_seconds_a_side=round(time.time() - t0, 2),
                        build_err=(r.stderr or "")[-300:] if r.returncode else None)
        info["reused"] = (not info["rebuilt"])
    except Exception as exc:  # noqa: BLE001
        info["error"] = repr(exc)
    cache_info[tag] = info
    print("CACHE", tag, json.dumps(info), flush=True)

BASE = ["--iterations", "5000", "--snapshots", "", "--profile",
        "competition/profiles/gpufree-2.json", "--seed", "20261015",
        "--position-lr-max-steps", "5000", "--densify-from", "500",
        "--densify-interval", "100", "--opacity-reset-interval", "0",
        "--lambda-dssim", "0.6", "--scaling-lr", "0.01",
        "--realloc-frac", "0.15", "--realloc-at", "1250", "--no-quick-test"]
D3500 = ["--densify-until", "3500", "--grad-abs-thresh", "0.0008"]

cells = []
for arm, extra in (("d3500", []),
                   ("d3500_cache", ["--scene-cache", str(CACHE_ROOT / "bicycle")]),
                   ("d3500_m1", ["--mid-prune-at", "1250", "--mid-prune-keep", "0.6"]),
                   ("d3500_m1_cache", ["--mid-prune-at", "1250", "--mid-prune-keep", "0.6",
                                       "--scene-cache", str(CACHE_ROOT / "bicycle")])):
    cells.append(dict(scene="bicycle", arm=arm, extra=extra))
for arm, extra in (("d3500", []),
                   ("d3500_cache", ["--scene-cache", str(CACHE_ROOT / "0b5635")])):
    cells.append(dict(scene="0b5635", arm=arm, extra=extra))

def run_cell(cell):
    tag, arm = cell["scene"], cell["arm"]; case = CASES[tag]; e = manifest[case]
    work = REPO / e["work"]; init = REPO / e["init"]
    cell_dir = OUT / "matrix" / tag / arm; sweep = cell_dir / "sweep"
    if (sweep / "summary.json").exists():
        print("SKIP", tag, arm, flush=True); return
    cell_dir.mkdir(parents=True, exist_ok=True)
    argv = [CONDA, "run", "--no-capture-output", "-n", "fastgs", "python", "-m",
            "research.run_sweep", "--case", case, "--work", str(work),
            "--init-ply", str(init), "--out", str(sweep),
            "--budget", str(BUDGET[case]), "--image-cache", str(work / "image_cache"),
            *BASE, *D3500, *cell["extra"]]
    t0 = time.time()
    with open(cell_dir / "sweep.log", "w") as fh:
        rc = subprocess.run(argv, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT).returncode
    print("CELL", tag, arm, "rc", rc, "%.1fs" % (time.time() - t0), flush=True)
    (cell_dir / "cell.json").write_text(json.dumps(dict(rc=rc, seconds=round(time.time() - t0, 1)),
                                                    indent=2))

for cell in cells:
    run_cell(cell)
json.dump(dict(cache=cache_info, cells=cells), open(OUT / "matrix.json", "w"), indent=2)
print("D3500_COMBO_DONE", flush=True)
