import json, os, subprocess, time
from pathlib import Path
REPO = Path("/root/projects/FFSplat-Hackathon")
OUT = Path("/root/gpufree-data/outputs/b-eng-20261030/breakdown")
OUT.mkdir(parents=True, exist_ok=True)
manifest = {e["case"]: e for e in json.load(open(REPO / "records/evalnode-manifest-13.json"))}
env = dict(os.environ, CUDA_VISIBLE_DEVICES="2", PYTHONPATH=str(REPO))
CASES = ["mipnerf360/bicycle",
         "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce"]
for case in CASES:
    e = manifest[case]
    tag = case.replace("/", "_")[:40]
    out = OUT / tag
    if (out / "record.json").exists():
        print("SKIP", tag); continue
    work = REPO / e["work"]
    argv = ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "fastgs", "python",
            "-m", "research.train_sweep",
            "--source", str(work), "--model_path", str(out / "model"), "--eval", "-r", "1",
            "--sh_degree", "3", "--quiet", "--init_ply", str(REPO / e["init"]),
            "--iterations", "5000", "--snapshots", "", "--record", str(out / "record.json"),
            "--seed", "20261015", "--position_lr_max_steps", "5000",
            "--densification_interval", "100", "--densify_from_iter", "500",
            "--densify_until_iter", "1250", "--opacity_reset_interval", "0",
            "--lambda_dssim", "0.6", "--scaling_lr", "0.01", "--realloc_frac", "0.15",
            "--realloc_at", "1250", "--image_cache", str(work / "image_cache"),
            "--time_breakdown", "--no_save_final", "--no_quick_test"]
    t0 = time.time()
    r = subprocess.run(argv, cwd=str(REPO), env=env, capture_output=True, text=True)
    print("BREAKDOWN", tag, "rc", r.returncode, "%.1fs" % (time.time() - t0), flush=True)
    if r.returncode != 0:
        print((r.stderr or "")[-1500:], flush=True)
    else:
        rec = json.load(open(out / "record.json"))
        print(json.dumps({k: rec.get(k) for k in ("train_seconds", "scene_load_seconds",
                                                  "init_seconds", "timing_breakdown")}), flush=True)
print("BREAKDOWN_DONE")
