import json, subprocess, time
from pathlib import Path
REPO = Path("/root/projects/FFSplat-Hackathon")
OUT = Path("/root/gpufree-data/outputs/d4000-13-20261030/repeat"); OUT.mkdir(parents=True, exist_ok=True)
manifest = {e["case"]: e for e in json.load(open(REPO / "records/evalnode-manifest-13.json"))}
PY = "/opt/conda/envs/fastgs/bin/python"; CONDA = "/opt/conda/bin/conda"
CACHE = Path("/root/gpufree-data/outputs/d4000-13-20261030/scene_cache/0b5635d53b3b33294d")
BASE = ["--iterations","5000","--snapshots","","--profile","competition/profiles/gpufree-2.json",
        "--seed","20261015","--position-lr-max-steps","5000","--densify-from","500",
        "--densify-interval","100","--opacity-reset-interval","0","--lambda-dssim","0.6",
        "--scaling-lr","0.01","--realloc-frac","0.15","--realloc-at","1250","--no-quick-test"]
D4000 = ["--densify-until","4000","--grad-abs-thresh","0.0008"]
VGG200 = ["--lpips-weight","0.1","--lpips-scale","1.0","--lpips-start","4801","--lpips-end","5000"]
case = "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce"
e = manifest[case]; work = REPO / e["work"]; init = REPO / e["init"]
sweep = OUT / "0b5635"/ "sweep"
if not (sweep/"summary.json").exists():
    sweep.parent.mkdir(parents=True, exist_ok=True)
    argv = [CONDA,"run","--no-capture-output","-n","fastgs","python","-m","research.run_sweep",
            "--case",case,"--work",str(work),"--init-ply",str(init),"--out",str(sweep),
            "--budget", str(e["budget_seconds"]), "--image-cache", str(work/"image_cache"),
            *BASE, *D4000, *VGG200, "--mid-prune-at","1250","--mid-prune-keep","1.0",
            "--scene-cache", str(CACHE)]
    t0=time.time()
    with open(sweep.parent/"sweep.log","w") as fh:
        rc = subprocess.run(argv, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT).returncode
    print("REPEAT rc", rc, "%.1fs" % (time.time()-t0), flush=True)
print("REPEAT_DONE")
