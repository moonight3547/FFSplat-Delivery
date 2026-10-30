import hashlib, json, os, subprocess, time
from pathlib import Path
REPO = Path("/root/projects/FFSplat-Hackathon")
OUT = Path("/root/gpufree-data/outputs/b-eng-20261030/ab")
OUT.mkdir(parents=True, exist_ok=True)
manifest = {e["case"]: e for e in json.load(open(REPO / "records/evalnode-manifest-13.json"))}
env = dict(os.environ, CUDA_VISIBLE_DEVICES="2", PYTHONPATH=str(REPO))
ARMS = {
    "ctrl": [],
    "camonly": ["--train_cameras_only"],
    "lean": ["--lean_step"],
    "both": ["--train_cameras_only", "--lean_step"],
}
CASES = ["mipnerf360/bicycle",
         "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce"]
rows = []
for case in CASES:
    e = manifest[case]
    tag = "bicycle" if "bicycle" in case else "0b5635"
    for arm, flags in ARMS.items():
        out = OUT / ("%s__%s" % (tag, arm))
        rec_path = out / "record.json"
        if rec_path.exists():
            print("SKIP", tag, arm, flush=True)
        else:
            out.mkdir(parents=True, exist_ok=True)
            argv = ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "fastgs",
                    "python", "-m", "research.train_sweep",
                    "--source", str(REPO / e["work"]), "--model_path", str(out / "model"),
                    "--eval", "-r", "1", "--sh_degree", "3", "--quiet",
                    "--init_ply", str(REPO / e["init"]), "--iterations", "5000",
                    "--snapshots", "", "--record", str(rec_path), "--seed", "20261015",
                    "--position_lr_max_steps", "5000", "--densification_interval", "100",
                    "--densify_from_iter", "500", "--densify_until_iter", "1250",
                    "--opacity_reset_interval", "0", "--lambda_dssim", "0.6",
                    "--scaling_lr", "0.01", "--realloc_frac", "0.15", "--realloc_at", "1250",
                    "--image_cache", str(REPO / e["work"] / "image_cache"),
                    "--no_quick_test"] + flags
            t0 = time.time()
            r = subprocess.run(argv, cwd=str(REPO), env=env, capture_output=True, text=True)
            print("RUN", tag, arm, "rc", r.returncode, "%.1fs" % (time.time() - t0), flush=True)
            if r.returncode != 0:
                print((r.stderr or "")[-1200:], flush=True); continue
        rec = json.load(open(rec_path))
        final_ply = Path(rec["final_ply"])
        sha = hashlib.sha256(final_ply.read_bytes()).hexdigest() if final_ply.exists() else None
        rows.append(dict(case=case, tag=tag, arm=arm,
                         scene_load_seconds=rec["scene_load_seconds"],
                         init_seconds=rec["init_seconds"],
                         train_seconds=rec["train_seconds"],
                         total_seconds=rec["total_seconds"],
                         num_final_gaussians=rec["num_final_gaussians"],
                         num_test_views=rec["num_test_views"],
                         ply_sha256=sha, eng=rec.get("eng")))
        print(json.dumps(rows[-1]), flush=True)
with open(OUT / "ab.jsonl", "w") as fh:
    for row in rows:
        fh.write(json.dumps(row) + "\n")
print("BENG_AB_DONE", OUT / "ab.jsonl")
