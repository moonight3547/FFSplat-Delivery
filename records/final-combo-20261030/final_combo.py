import json, os, subprocess, time
from pathlib import Path
REPO = Path("/root/projects/FFSplat-Hackathon")
OUT = Path("/root/gpufree-data/outputs/final-combo-20261030"); OUT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT = Path("/root/gpufree-data/outputs/b-eng-20261030/scene_cache")
manifest = {e["case"]: e for e in json.load(open(REPO / "records/evalnode-manifest-13.json"))}
PY = "/opt/conda/envs/fastgs/bin/python"; CONDA = "/opt/conda/bin/conda"
BUDGET = {"mipnerf360/bicycle": 95.0,
          "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce": 94.29}
CASES = {"bicycle": "mipnerf360/bicycle",
         "0b5635": "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce"}
import sys; sys.path.insert(0, str(REPO))
from research.scene_cache import version_of  # noqa: E402

# cache hash check (reuse; rebuild only on a true hash mismatch)
cache_info = {}
for tag, case in CASES.items():
    work = REPO / manifest[case]["work"]; cache = CACHE_ROOT / tag
    info = dict(reused=False, rebuilt=False)
    try:
        import numpy as np
        ver = str(np.load(cache / "cams.npz", allow_pickle=False)["version"])
        info.update(version=ver, hash_match=(ver == version_of(work)))
        if not info["hash_match"]:
            t0 = time.time()
            r = subprocess.run([PY, "-m", "research.scene_cache", "build", "--work", str(work),
                                "--out", str(cache)], cwd=str(REPO), capture_output=True, text=True)
            info.update(rebuilt=(r.returncode == 0),
                        build_seconds_a_side=round(time.time() - t0, 2))
        info["reused"] = not info["rebuilt"]
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
D = {"d2500": ["--densify-until", "2500", "--grad-abs-thresh", "0.0008"],
     "d3500": ["--densify-until", "3500", "--grad-abs-thresh", "0.0008"]}
V = {"v100": ["--lpips-weight", "0.1", "--lpips-scale", "1.0",
              "--lpips-start", "4901", "--lpips-end", "5000"],
     "v200": ["--lpips-weight", "0.1", "--lpips-scale", "1.0",
              "--lpips-start", "4801", "--lpips-end", "5000"]}

cells = []
for d in ("d2500", "d3500"):
    for v in ("v100", "v200"):
        cells.append(dict(scene="bicycle", arm=f"{d}_m1_cache_{v}",
                          flags=D[d] + V[v] + ["--mid-prune-at", "1250",
                                               "--mid-prune-keep", "0.6",
                                               "--scene-cache", str(CACHE_ROOT / "bicycle")]))
for d in ("d2500", "d3500"):
    for v in ("v100", "v200"):
        cells.append(dict(scene="0b5635", arm=f"{d}_cache_{v}",
                          flags=D[d] + V[v] + ["--scene-cache", str(CACHE_ROOT / "0b5635")]))

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
            *BASE, *cell["flags"]]
    t0 = time.time()
    with open(cell_dir / "sweep.log", "w") as fh:
        rc = subprocess.run(argv, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT).returncode
    print("CELL", tag, arm, "rc", rc, "%.1fs" % (time.time() - t0), flush=True)
    (cell_dir / "cell.json").write_text(json.dumps(dict(rc=rc, seconds=round(time.time() - t0, 1)),
                                                    indent=2))

for cell in cells:
    run_cell(cell)
json.dump(dict(cache=cache_info, cells=cells), open(OUT / "matrix.json", "w"), indent=2)
print("FINAL_COMBO_DONE", flush=True)
