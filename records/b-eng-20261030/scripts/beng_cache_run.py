import json, os, subprocess, time
from pathlib import Path
REPO = Path("/root/projects/FFSplat-Hackathon")
OUT = Path("/root/gpufree-data/outputs/b-eng-20261030/scene_cache"); OUT.mkdir(parents=True, exist_ok=True)
manifest = {e["case"]: e for e in json.load(open(REPO / "records/evalnode-manifest-13.json"))}
env = dict(os.environ, CUDA_VISIBLE_DEVICES="2", PYTHONPATH=str(REPO))
PY = "/opt/conda/envs/fastgs/bin/python"
CASES = ["mipnerf360/bicycle",
         "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce"]
res = {}
for case in CASES:
    e = manifest[case]; tag = "bicycle" if "bicycle" in case else "0b5635"
    work = REPO / e["work"]; cache = OUT / tag
    t0 = time.time()
    r = subprocess.run([PY, "-m", "research.scene_cache", "build", "--work", str(work),
                        "--out", str(cache)], cwd=str(REPO), env=env, capture_output=True, text=True)
    build = json.loads(r.stdout[r.stdout.index("{"):]) if r.returncode == 0 else {"error": r.stderr[-800:]}
    build["wall_seconds"] = round(time.time() - t0, 2)
    r2 = subprocess.run([PY, "-m", "research.scene_cache", "verify", "--work", str(work),
                         "--cache", str(cache)], cwd=str(REPO), env=env, capture_output=True, text=True)
    verify = json.loads(r2.stdout[r2.stdout.index("{"):]) if r2.returncode == 0 else {"error": r2.stderr[-800:]}
    r3 = subprocess.run([PY, "-m", "research.scene_cache", "bench", "--work", str(work),
                         "--cache", str(cache), "--reps", "3"], cwd=str(REPO), env=env,
                        capture_output=True, text=True)
    bench = json.loads(r3.stdout[r3.stdout.index("{"):]) if r3.returncode == 0 else {"error": r3.stderr[-800:]}
    res[tag] = dict(build=build, verify=verify, bench=bench)
    print(json.dumps({tag: res[tag]}), flush=True)
json.dump(res, open(OUT / "results.json", "w"), indent=2)
print("CACHE_RUN_DONE")
