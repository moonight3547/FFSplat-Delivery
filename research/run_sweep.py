"""One-shot research experiment driver: train -> render snapshots -> official metrics.

Runs entirely on the Linux GPU host. It shells out to the two pinned Conda
environments from a profile JSON (fastgs for training/rendering, resplat for the
official PSNR/SSIM/VGG-LPIPS implementation) so no metric is reimplemented here.

  python -m research.run_sweep --case hw/billboard \
      --work outputs/O0-20261030/work/hw/billboard \
      --init-ply outputs/O0-20261030/results/hw/billboard/resplat/resplat_world.ply \
      --out outputs/sweeps/hw-baseline-sched --iterations 12000 \
      --snapshots 1000,2000,4000,8000,12000

Outputs ``<out>/summary.json``: per-snapshot metrics, B-time estimate and the
organizer proxy score, plus the training record.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = REPO / "competition/profiles/compshare-20261029.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def conda_python(profile, runtime, *argv):
    env = profile["environments"][runtime]
    return [profile["conda"], "run", "--no-capture-output", "-n", env,
            "python", *map(str, argv)]


def run(argv, cwd=REPO, profile=None):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO)
    # use the *selected* profile, not the default: it carries CUDA_VISIBLE_DEVICES
    environment.update((profile or read(DEFAULT_PROFILE)).get("environment", {}))
    print("+ " + " ".join(map(str, argv)), flush=True)
    subprocess.run(argv, cwd=cwd, env=environment, check=True)


def proxy_scores(psnr, ssim, lpips, b_seconds, budget_seconds):
    """Organizer's linear /100 proxy (ProblemDesc 4). Time is module-B only."""
    out = {}
    out["psnr_points"] = 20.0 * max(0.0, min(1.0, (psnr - 15.0) / 25.0))
    out["ssim_points"] = 20.0 * max(0.0, min(1.0, (ssim - 0.70) / 0.25))
    out["lpips_points"] = 40.0 * max(0.0, min(1.0, (0.25 - lpips) / 0.20))
    if b_seconds is None or budget_seconds is None:
        out["time_points"] = None
    elif b_seconds >= budget_seconds:
        out["time_points"] = 0.0
    else:
        out["time_points"] = 20.0 * min(1.0, (1.0 - b_seconds / budget_seconds) / 0.9)
    parts = [out["psnr_points"], out["ssim_points"], out["lpips_points"]]
    out["quality_points"] = sum(parts)
    out["total_points"] = out["quality_points"] + (out["time_points"] or 0.0)
    out["budget_ratio"] = b_seconds / budget_seconds if b_seconds and budget_seconds else None
    out["valid"] = (b_seconds < budget_seconds) if b_seconds and budget_seconds else None
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--work", required=True, help="prepared scene dir (module_a output)")
    parser.add_argument("--init-ply", default=None, dest="init_ply")
    parser.add_argument("--init-random", action="store_true", dest="init_random")
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=12000)
    parser.add_argument("--snapshots", type=str, default="")
    parser.add_argument("--budget", type=float, default=None,
                        help="reference FastGS random 30K seconds for this case")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--mult", type=float, default=None)
    parser.add_argument("--position-lr-max-steps", type=int, default=None, dest="position_lr_max_steps")
    parser.add_argument("--position-lr-init", type=float, default=None, dest="position_lr_init")
    parser.add_argument("--position-lr-final", type=float, default=None, dest="position_lr_final")
    parser.add_argument("--densify-from", type=int, default=None, dest="densify_from_iter")
    parser.add_argument("--densify-until", type=int, default=None, dest="densify_until_iter")
    parser.add_argument("--densify-interval", type=int, default=None, dest="densification_interval")
    parser.add_argument("--opacity-reset-interval", type=int, default=None,
                        dest="opacity_reset_interval")
    parser.add_argument("--sh-up-every", type=int, default=None, dest="sh_up_every")
    parser.add_argument("--lambda-dssim", type=float, default=None, dest="lambda_dssim")
    parser.add_argument("--scaling-lr", type=float, default=None, dest="scaling_lr")
    parser.add_argument("--opacity-lr", type=float, default=None, dest="opacity_lr")
    parser.add_argument("--res-scales", default=None, dest="res_scales")
    parser.add_argument("--res-until", default=None, dest="res_until")
    parser.add_argument("--lpips-weight", type=float, default=None, dest="lpips_weight")
    parser.add_argument("--lpips-every", type=int, default=None, dest="lpips_every")
    parser.add_argument("--lpips-scale", type=float, default=None, dest="lpips_scale")
    parser.add_argument("--densify-count-thresh", type=int, default=None,
                        dest="densify_count_thresh")
    parser.add_argument("--max-gaussians", type=int, default=None, dest="max_gaussians")
    parser.add_argument("--inject-holes", type=int, default=0, dest="inject_holes",
                        help="add N low-opacity Gaussians in empty voxels (hole injection)")
    parser.add_argument("--inject-at", type=int, default=800, dest="inject_at")
    parser.add_argument("--opt-step-schedule", default="", dest="opt_step_schedule",
                        help="optimizer step-frequency schedule, e.g. '2500:2'")
    parser.add_argument("--realloc-frac", type=float, default=0.0, dest="realloc_frac")
    parser.add_argument("--realloc-at", type=int, default=1250, dest="realloc_at")
    parser.add_argument("--sh-degree", type=int, default=3, dest="sh_degree")
    parser.add_argument("--fused-adam", action="store_true", dest="fused_adam")
    parser.add_argument("--legacy-sync", action="store_true", dest="legacy_sync")
    parser.add_argument("--freeze-geometry", action="store_true", dest="freeze_geometry")
    parser.add_argument("--no-quick-test", action="store_true", dest="no_quick_test")
    parser.add_argument("--image-cache", default=None, dest="image_cache",
                        help="pre-decoded uint8 image cache dir (module-A artifact)")
    parser.add_argument("--no-densify", action="store_true", dest="no_densify")
    args = parser.parse_args(argv)

    profile = read(args.profile)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)
    snaps_dir = out / "snaps"
    model_dir = out / "model"
    eval_root = out / "eval"

    snapshots = [s for s in args.snapshots.split(",") if s.strip()]
    if str(args.iterations) not in snapshots:
        snapshots.append(str(args.iterations))

    train_argv = [f"--source={args.work}", f"--model_path={model_dir}",
                  "--eval", "-r", "1", "--sh_degree", str(args.sh_degree), "--quiet"]
    if args.init_random:
        train_argv.append("--init_random")
    else:
        train_argv += ["--init_ply", args.init_ply]
    train_argv += [
                  "--iterations", str(args.iterations),
                  "--snapshots", ",".join(snapshots),
                  "--snap_dir", str(snaps_dir),
                  "--record", str(out / "train_record.json"),
                  "--seed", str(args.seed)]
    optional = {
        "--position_lr_max_steps": args.position_lr_max_steps,
        "--position_lr_init": args.position_lr_init,
        "--position_lr_final": args.position_lr_final,
        "--densify_from_iter": args.densify_from_iter,
        "--densify_until_iter": args.densify_until_iter,
        "--densification_interval": args.densification_interval,
        "--opacity_reset_interval": args.opacity_reset_interval,
        "--sh_up_every": args.sh_up_every,
        "--mult": args.mult,
        "--lambda_dssim": args.lambda_dssim,
        "--scaling_lr": args.scaling_lr,
        "--opacity_lr": args.opacity_lr,
        "--res_scales": args.res_scales,
        "--res_until": args.res_until,
        "--lpips_weight": args.lpips_weight,
        "--lpips_every": args.lpips_every,
        "--lpips_scale": args.lpips_scale,
        "--densify_count_thresh": args.densify_count_thresh,
        "--max_gaussians": args.max_gaussians,
        "--image_cache": args.image_cache,
        "--realloc_frac": args.realloc_frac,
        "--realloc_at": args.realloc_at,
        "--opt_step_schedule": args.opt_step_schedule,
        "--inject_holes": args.inject_holes,
        "--inject_at": args.inject_at,
    }
    for flag, value in optional.items():
        if value is not None:
            train_argv.extend([flag, str(value)])
    if args.no_densify:
        train_argv.append("--no_densify")
    if args.fused_adam:
        train_argv.append("--fused_adam")
    if args.legacy_sync:
        train_argv.append("--legacy_sync")
    if args.freeze_geometry:
        train_argv.append("--freeze_geometry")
    if args.no_quick_test:
        train_argv.append("--no_quick_test")

    run(conda_python(profile, "fastgs", "-m", "research.train_sweep", *train_argv), profile=profile)
    record = read(out / "train_record.json")

    renders, names = [], []
    for snap in record["snapshots"]:
        name = f"iter{snap['iteration']:05d}"
        renders.extend(["--ply", snap["ply"]])
        names.extend(["--name", name])
    run(conda_python(profile, "fastgs", "-m", "research.render_eval",
                     f"--source={args.work}", f"--out={eval_root}", "--eval", "-r", "1",
                     "--sh_degree", str(args.sh_degree), *renders, *names), profile=profile)

    results = []
    for snap in record["snapshots"]:
        name = f"iter{snap['iteration']:05d}"
        target = eval_root / name
        run(conda_python(profile, "resplat", str(REPO / "competition/upstream/baseline/evaluate.py"),
                         "--model_path", target, "--device", profile["device"]), profile=profile)
        metrics = read(target / "metrics.json")["mean"]
        b_seconds = (record["scene_load_seconds"] + record["init_seconds"]
                     + snap["train_seconds"] + 0.5)
        row = dict(iteration=snap["iteration"], num_gaussians=snap["num_gaussians"],
                   train_seconds=round(snap["train_seconds"], 2),
                   b_seconds_estimate=round(b_seconds, 2),
                   psnr=metrics["psnr"], ssim=metrics["ssim"], lpips=metrics["lpips"],
                   score=proxy_scores(metrics["psnr"], metrics["ssim"], metrics["lpips"],
                                      b_seconds, args.budget))
        results.append(row)
        print(json.dumps(row), flush=True)

    summary = dict(case=args.case, work=str(args.work), init_ply=str(args.init_ply),
                   budget_seconds=args.budget, train_record={k: v for k, v in record.items()
                                                             if k != "snapshots"},
                   results=results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(written=str(out / "summary.json")), indent=2))


if __name__ == "__main__":
    main()
