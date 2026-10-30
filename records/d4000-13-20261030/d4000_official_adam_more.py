import json, os, subprocess, time
from pathlib import Path
REPO = Path("/root/projects/FFSplat-Hackathon")
OUT = Path("/root/gpufree-data/outputs/d4000-13-20261030/official_adam"); OUT.mkdir(parents=True, exist_ok=True)
manifest = {e["case"]: e for e in json.load(open(REPO / "records/evalnode-manifest-13.json"))}
OLD = Path("/root/gpufree-data/outputs/b-eng-20261030/scene_cache")
NEW = Path("/root/gpufree-data/outputs/d4000-13-20261030/scene_cache")
CASES = [("dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce", "0b5635", OLD / "0b5635", "1.0"),
         ("dl3dv/3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3", "3e4dbe", NEW / "3e4dbe4135ad3ba14c", "1.0")]
for case, tag, cache, keep in CASES:
    e = manifest[case]; sweep = OUT / tag / "sweep"
    if (sweep / "summary.json").exists():
        print("SKIP", tag, flush=True); continue
    sweep.parent.mkdir(parents=True, exist_ok=True)
    work = REPO / e["work"]
    args = ["/opt/conda/bin/conda","run","--no-capture-output","-n","fastgs","python","-m",
            "research.run_sweep","--case",case,"--work",str(work),"--init-ply",str(REPO/e["init"]),
            "--out",str(sweep),"--budget",str(e["budget_seconds"]),
            "--image-cache",str(work/"image_cache"),
            "--iterations","5000","--snapshots","","--profile","competition/profiles/gpufree-2.json",
            "--seed","20261015","--position-lr-max-steps","5000","--densify-from","500",
            "--densify-interval","100","--opacity-reset-interval","0","--lambda-dssim","0.6",
            "--scaling-lr","0.01","--realloc-frac","0.15","--realloc-at","1250","--no-quick-test",
            "--densify-until","4000","--grad-abs-thresh","0.0008",
            "--lpips-weight","0.1","--lpips-scale","1.0","--lpips-start","4801","--lpips-end","5000",
            "--mid-prune-at","1250","--mid-prune-keep",keep,"--scene-cache",str(cache)]
    env = dict(os.environ, FFSPLAT_OFFICIAL_ADAM="1")
    t0 = time.time()
    with open(sweep.parent / "sweep.log", "w") as fh:
        rc = subprocess.run(args, cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    print("OA", tag, "rc", rc, "%.1fs" % (time.time() - t0), flush=True)
print("OA_MORE_DONE")
