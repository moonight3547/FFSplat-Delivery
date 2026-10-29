"""Score arbitrary Gaussian PLYs on the held-out views with the official metrics.

Useful for module-A / ReSplat ablations where the interesting number is the
quality of the feed-forward initialization itself (before any module-B step).

  python -m research.score_ply --work work/<case> --out <fresh dir> \
      --budget 94.3 --b-seconds 19.2 \
      --ply init.ply --name refine2 --ply refine4.ply --name refine4
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


def proxy(psnr, ssim, lpips, b_seconds=None, budget=None):
    points = dict(
        psnr=20.0 * max(0.0, min(1.0, (psnr - 15.0) / 25.0)),
        ssim=20.0 * max(0.0, min(1.0, (ssim - 0.70) / 0.25)),
        lpips=40.0 * max(0.0, min(1.0, (0.25 - lpips) / 0.20)))
    quality = sum(points.values())
    time_points = None
    if b_seconds is not None and budget:
        time_points = 0.0 if b_seconds >= budget else 20.0 * min(1.0, (1 - b_seconds / budget) / 0.9)
    return dict(quality_points=quality, time_points=time_points,
                total_points=quality + (time_points or 0.0),
                components=points)


def run(argv):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env.update(read(DEFAULT_PROFILE).get("environment", {}))
    print("+ " + " ".join(map(str, argv)), flush=True)
    subprocess.run(argv, cwd=REPO, env=env, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ply", action="append", default=[])
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--b-seconds", type=float, default=None, dest="b_seconds")
    parser.add_argument("--extra-json", default=None, help="merge extra fields into the summary")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    args = parser.parse_args(argv)
    if len(args.ply) != len(args.name) or not args.ply:
        raise SystemExit("--ply/--name pairs required")

    profile = read(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # The held-out split only exists when the reader runs with --eval (llffhold=8),
    # exactly like the organizer's render stage.
    render_args = [f"--source={args.work}", f"--out={out}", "--eval", "-r", "1", "--sh_degree", "3"]
    for ply, name in zip(args.ply, args.name):
        render_args.extend(["--ply", ply, "--name", name])
    run([profile["conda"], "run", "--no-capture-output", "-n", profile["environments"]["fastgs"],
         "python", "-m", "research.render_eval", *render_args])

    rows = []
    for name in args.name:
        target = out / name
        run([profile["conda"], "run", "--no-capture-output", "-n", profile["environments"]["resplat"],
             "python", str(REPO / "competition/upstream/baseline/evaluate.py"),
             "--model_path", target, "--device", profile["device"]])
        mean = read(target / "metrics.json")["mean"]
        row = dict(name=name, psnr=mean["psnr"], ssim=mean["ssim"], lpips=mean["lpips"],
                   score=proxy(mean["psnr"], mean["ssim"], mean["lpips"], args.b_seconds, args.budget))
        rows.append(row)
        print(json.dumps(row), flush=True)

    summary = dict(work=str(args.work), budget_seconds=args.budget,
                   b_seconds=args.b_seconds, rows=rows)
    if args.extra_json:
        summary.update(read(args.extra_json))
    (out / "score_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(written=str(out / "score_summary.json"))))


if __name__ == "__main__":
    main()
