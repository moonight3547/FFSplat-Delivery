import json, os, subprocess, time
from pathlib import Path
REPO = Path("/root/projects/FFSplat-Hackathon")
OUT = Path("/root/gpufree-data/outputs/framework-opt-20261030/gtcache_ab"); OUT.mkdir(parents=True, exist_ok=True)
OLD = Path("/root/gpufree-data/outputs/b-eng-20261030/scene_cache")
manifest = {e["case"]: e for e in json.load(open(REPO / "records/evalnode-manifest-13.json"))}
CASES = [("mipnerf360/bicycle", "bicycle", OLD / "bicycle", "0.6"),
         ("dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce", "0b5635",
          OLD / "0b5635", "1.0")]
BASE = ["--iterations","5000","--snapshots","","--profile","competition/profiles/gpufree-2.json",
        "--seed","20261015","--position-lr-max-steps","5000","--densify-from","500",
        "--densify-interval","100","--opacity-reset-interval","0","--lambda-dssim","0.6",
        "--scaling-lr","0.01","--realloc-frac","0.15","--realloc-at","1250","--no-quiet-test" if False else "--no-quick-test",
        "--densify-until","3500","--grad-abs-thresh","0.0008",
        "--lpips-weight","0.1","--lpips-scale","1.0","--lpips-start","4801","--lpips-end","5000",
        "--lpips-timing"]
for case, tag, cache, keep in CASES:
    e = manifest[case]; work = REPO / e["work"]
    for arm, extra in (("ctrl", []),
                       ("gtcache", ["--gt-feature-cache", "--gt-feature-cache-views", "40"])):
        sweep = OUT / tag / arm / "sweep"
        if (sweep / "summary.json").exists():
            print("SKIP", tag, arm, flush=True); continue
        sweep.parent.mkdir(parents=True, exist_ok=True)
        argv = ["/opt/conda/bin/conda","run","--no-capture-output","-n","fastgs","python","-m",
                "research.run_sweep","--case",case,"--work",str(work),"--init-ply",str(REPO/e["init"]),
                "--out",str(sweep),"--budget",str(e["budget_seconds"]),
                "--image-cache",str(work/"image_cache"),
                *BASE, "--mid-prune-at","1250","--mid-prune-keep",keep,
                "--scene-cache",str(cache), *extra]
        t0 = time.time()
        with open(sweep.parent / "sweep.log", "w") as fh:
            rc = subprocess.run(argv, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT).returncode
        print("CELL", tag, arm, "rc", rc, "%.1fs" % (time.time() - t0), flush=True)
print("GTCACHE_AB_DONE")
