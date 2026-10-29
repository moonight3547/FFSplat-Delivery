"""Four-cell organizer A/B ablation. Defaults to a CPU-only command preview."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CELLS = (("O0", "uniform16", "official-fastgs", None),
         ("A", "ranked16", "official-fastgs", "O0"),
         ("B", "uniform16", "mcmc-ctf", "O0"),
         ("AB", "ranked16", "mcmc-ctf", "A"))


def build_suite(args):
    commands = []
    for cell, a_policy, b_policy, control in CELLS:
        command = [sys.executable, "-m", "competition.run", "--spec", str(args.spec.resolve()),
            "--data", str(args.data.resolve()), "--checkpoint", str(args.checkpoint.resolve()),
            "--work", str(args.work.resolve()/cell), "--profile", str(args.profile.resolve()),
            "--a-policy", a_policy, "--policy", b_policy, "--b-settings", str(args.b_settings.resolve())]
        if control: command += ["--from-control", str(args.work.resolve()/control)]
        if cell == "A": command.append("--reuse-geometry-only")
        if args.groups: command += ["--groups", *args.groups]
        if getattr(args, "assets", None): command += ["--assets", str(args.assets.resolve())]
        commands.append(dict(cell=cell, argv=command))
    return commands


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("spec", "data", "checkpoint", "work"): p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--profile", type=Path, default=ROOT/"profiles/linux-cuda.json")
    p.add_argument("--b-settings", type=Path, default=ROOT/"policies/mcmc-ctf.json")
    p.add_argument("--groups", nargs="+", choices=["dl3dv", "hw", "mipnerf360"])
    p.add_argument("--execute", action="store_true")
    p.add_argument("--assets", type=Path, help="Registered asset manifest, required for execution")
    args = p.parse_args(argv)
    from .contracts import read_json
    from .budget import validate_settings
    validate_settings(read_json(args.b_settings))
    commands = build_suite(args)
    if not args.execute:
        print(json.dumps(dict(cells=commands, timing="Composed E2E for reused stages; no /100 formula"), indent=2))
        return
    if not args.assets: p.error("Execution requires --assets")
    if args.work.exists() and any(args.work.iterdir()):
        raise FileExistsError("Use a fresh suite directory; preserve completed/partial evidence")
    for row in commands:
        subprocess.run(row["argv"], cwd=ROOT.parent, check=True)


if __name__ == "__main__":
    main()
