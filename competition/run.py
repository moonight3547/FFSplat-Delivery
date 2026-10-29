"""Portable organizer launcher. Run with --describe before installing GPU runtimes."""
import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time

from .contracts import read_json, sha256, write_json
from .pairing import freeze_inputs, reuse_control, verify_inputs
from .runtime import bind_python

ROOT = Path(__file__).resolve().parent


def build_commands(args, spec, profile):
    upstream = args.upstream.resolve()
    baseline = upstream / "baseline"
    envs = profile["environments"]
    conda = profile["conda"]
    def python(env, script, *flags):
        return [conda, "run", "--no-capture-output", "-n", envs[env], "python", str(script), *map(str, flags)]
    result = []
    for case in spec["cases"]:
        if args.groups and case["group"] not in args.groups:
            continue
        work = args.work.resolve() / "work" / case["id"]
        out = args.work.resolve() / "results" / case["id"]
        scene = args.data.resolve() / case["source"] / case["scene"]
        common = ["-s", work, "--eval", "-r", 1, "--sh_degree", 3, "--quiet"]
        commands = {
            "prepare": python("resplat", baseline / "module_a/preprocess.py", "--scene_dir", scene,
                "--group", case["group"], "--out", work, "--num_context", 16, "--llffhold", 8,
                "--max_resolution", 960),
            "resplat": [conda, "run", "--no-capture-output", "-n", envs["resplat"],
                "python", "-m", "competition.resplat", "--work", str(work), "--out", str(out/"resplat"),
                "--checkpoint", str(args.checkpoint.resolve()), "--num_refine", "2", "--device", profile["device"],
                "--upstream", str(upstream)],
            "reference": python("fastgs", baseline / "module_b/train_from_ply_fastgs.py", *common,
                "-m", out / "reference_fastgs", "--iterations", 30000, "--position_lr_max_steps", 30000,
                "--densify_until_iter", 15000, "--init_random"),
            "train": python("fastgs", baseline / "module_b/train_from_ply_fastgs.py", *common,
                "-m", out / "module_b", "--iterations", 10000, "--init_ply", out / "resplat/resplat_world.ply"),
            "render": python("fastgs", baseline / "render_test.py", *common, "-m", out / "module_b",
                "--backend", "fastgs", "--iteration", 10000),
            "metrics": python("resplat", baseline / "evaluate.py", "--model_path", out / "module_b",
                "--device", profile["device"]),
        }
        if getattr(args, "a_policy", "uniform16") == "ranked16":
            commands["resplat"] = [conda, "run", "--no-capture-output", "-n", envs["resplat"],
                "python", "-m", "competition.rank_resplat", "--work", str(work), "--out", str(out/"resplat"),
                "--checkpoint", str(args.checkpoint.resolve()), "--upstream", str(upstream),
                "--device", profile["device"]]
        if args.policy == "mcmc-ctf":
            commands["train"] = [conda, "run", "--no-capture-output", "-n", envs["mcmc"],
                "python", "-m", "competition.mcmc", "--work", str(work),
                "--initial", str(out/"resplat/resplat_world.npz"), "--out", str(out/"module_b"),
                "--reference", str(out/"reference_fastgs/module_b.json"),
                "--resplat-record", str(out/"resplat/resplat.json"),
                "--settings", str(getattr(args, "b_settings", ROOT/"policies/mcmc-ctf.json").resolve())]
            commands["render"][-1] = "-1"  # Actual adaptive iteration; never pretend it ran10K.
        stages = ["prepare", "reference", "resplat", "train", "render", "metrics"] if args.stage == "all" else [args.stage]
        if getattr(args, "from_control", None):
            skipped = ("prepare", "reference") if getattr(args, "reuse_geometry_only", False) else ("prepare", "reference", "resplat")
            stages = [s for s in stages if s not in skipped]
        result.extend(dict(case=case["id"], stage=stage, work=str(work), results=str(out),
                           source=str(scene), argv=commands[stage]) for stage in stages)
    if not result:
        raise ValueError("No matching organizer cases")
    return result


def check_upstream(path, commit):
    manifest = read_json(ROOT / "upstream-snapshot.json")
    if manifest["commit"] != commit:
        raise ValueError("Organizer snapshot commit mismatch")
    files = {p.relative_to(path).as_posix(): sha256(p) for p in path.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if files != manifest["files"]:
        raise ValueError("Organizer snapshot bytes changed; restore the pinned snapshot")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="Directory containing DL3DV/, hw/, mipnerf360/")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--assets", type=Path, help="Verified manual-download manifest; required for execution")
    p.add_argument("--profile", type=Path, default=ROOT / "profiles/linux-cuda.json")
    p.add_argument("--upstream", type=Path, default=ROOT / "upstream")
    p.add_argument("--groups", nargs="+", choices=["dl3dv", "hw", "mipnerf360"])
    p.add_argument("--stage", default="all", choices=["all", "prepare", "reference", "resplat", "train", "render", "metrics"])
    p.add_argument("--describe", action="store_true", help="CPU-only, print command plan without launching anything")
    p.add_argument("--policy", choices=["official-fastgs", "mcmc-ctf"], default="official-fastgs")
    p.add_argument("--a-policy", choices=["uniform16", "ranked16"], default="uniform16")
    p.add_argument("--b-settings", type=Path, default=ROOT/"policies/mcmc-ctf.json")
    p.add_argument("--from-control", type=Path, help="Reuse measured preprocessing/reference and, by default, exact initial world Gaussians")
    p.add_argument("--reuse-geometry-only", action="store_true", help="For A ablation: clone uniform control inputs, rerank and rerun ReSplat")
    args = p.parse_args(argv)
    spec, profile = read_json(args.spec), read_json(args.profile)
    if args.policy == "mcmc-ctf":
        from .budget import validate_settings
        validate_settings(read_json(args.b_settings))
    lock = read_json(ROOT / "upstream-lock.json")
    if args.reuse_geometry_only and (not args.from_control or args.a_policy != "ranked16"):
        raise ValueError("Geometry-only reuse requires a uniform control and ranked16 A")
    if args.from_control and args.stage != "all":
        raise ValueError("Paired reuse requires --stage all and a fresh output")
    if spec.get("protocol") != "organizer-20261028" or spec["upstream"]["commit"] != lock["commit"]:
        raise ValueError("Organizer protocol/version mismatch")
    commands = build_commands(args, spec, profile)
    if args.describe:
        print(json.dumps({"benchmark_id": spec["benchmark_id"], "commands": commands}, indent=2))
        return
    if not args.assets:
        p.error("Execution requires --assets from python -m competition.assets")
    if platform.system().lower() != profile["execution_os"] or profile["execution_os"] != "linux":
        raise RuntimeError("Execute this CUDA profile on Linux; macOS/Windows may inspect and control over SSH")
    check_upstream(args.upstream, lock["commit"])
    from .assets import verify
    assets = verify(args.assets, spec, args.data, args.checkpoint)
    if not args.checkpoint.is_file() or args.checkpoint.name != lock["checkpoint_name"]:
        raise ValueError("The organizer 16-view checkpoint is required; legacy 8-view weights are incompatible")
    if not all(Path(c["source"]).is_dir() for c in commands):
        raise FileNotFoundError("Organizer source data missing; refusing to skip requested cases")
    hardware = dict(host=socket.gethostname(), gpu_inventory=subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True).strip())
    source_paths = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix in (".py", ".json")
                    and not p.is_relative_to(ROOT/"upstream")]
    identity = dict(benchmark=spec, profile=profile, hardware=hardware, policy=args.policy, a_policy=args.a_policy,
                    resplat_seed=20261015,
                    b_settings=read_json(args.b_settings) if args.policy == "mcmc-ctf" else None,
                    checkpoint_sha256=assets["checkpoint"]["sha256"],
                    assets_sha256=sha256(args.assets),
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip(),
                    source_sha256={str(p.relative_to(ROOT.parent)): sha256(p) for p in source_paths})
    if args.from_control:
        control = args.from_control.resolve()
        previous = read_json(control/"run_identity.json")
        if (previous.get("policy") != "official-fastgs" or previous["benchmark"] != spec or
                previous["profile"] != profile or previous.get("hardware") != hardware or
                previous["checkpoint_sha256"] != identity["checkpoint_sha256"] or
                previous.get("assets_sha256") != identity["assets_sha256"] or
                previous.get("source_sha256") != identity["source_sha256"]):
            raise ValueError("Paired control protocol/runtime/checkpoint mismatch")
        required_a = "uniform16" if args.reuse_geometry_only else args.a_policy
        if previous.get("a_policy", "uniform16") != required_a:
            raise ValueError("Paired control A policy mismatch")
        identity["paired_control"] = dict(path=str(control), identity_sha256=sha256(control/"run_identity.json"),
            reuse="geometry" if args.reuse_geometry_only else "initial_gaussians",
            timing="Measured reused stages plus newly measured stages; composed E2E")
    identity_path = args.work / "run_identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Existing work directory belongs to a different experiment")
    write_json(identity_path, identity)
    if args.from_control:
        reuse_control(control, args.work, sorted({c["case"] for c in commands}), args.reuse_geometry_only)
    needed = set()
    for command in commands:
        needed.add("fastgs" if command["stage"] in ("reference", "render") else
                   ("mcmc" if command["stage"] == "train" and args.policy == "mcmc-ctf" else
                    ("fastgs" if command["stage"] == "train" else "resplat")))
    environment = os.environ.copy()
    environment.update(profile.get("environment", {}))
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Extensions are installed into the selected Conda environment. Prepending
    # unbuilt upstream source packages would shadow their compiled installations.
    environment["PYTHONPATH"] = str(ROOT.parent)
    for runtime in sorted(needed):
        subprocess.run(bind_python([profile["conda"], "run", "--no-capture-output", "-n",
            profile["environments"][runtime], "python", "-m", "competition.doctor", "--runtime", runtime],
            ), cwd=ROOT.parent, env=environment, check=True)
    for item in commands:
        work, out = Path(item["work"]), Path(item["results"])
        stage = item["stage"]
        log = args.work / "logs" / item["case"] / (stage + ".log")
        done = log.with_suffix(".json")
        if done.exists() or log.exists():
            raise FileExistsError(f"Preserve prior/partial stage artifacts: {log}; select a fresh work directory")
        if stage != "prepare":
            verify_inputs(work)
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        print(json.dumps({"case": item["case"], "stage": stage}), flush=True)
        with log.open("w") as stream:
            subprocess.run(bind_python(item["argv"]), cwd=ROOT.parent, env=environment, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        write_json(done, dict(case=item["case"], stage=stage, subprocess_wall_seconds=time.perf_counter()-start,
                            argv=item["argv"], status="complete"))
        if stage == "resplat":
            write_json(out/"resplat/export_identity.json", dict(
                npz_sha256=sha256(out/"resplat/resplat_world.npz"),
                ply_sha256=sha256(out/"resplat/resplat_world.ply")))
        if stage == "prepare" or (stage == "resplat" and args.a_policy == "ranked16"):
            previous_geometry = read_json(work/"input_identity.json").get("geometry_sha256") if stage == "resplat" else None
            frozen = freeze_inputs(work, spec["benchmark_id"])
            if previous_geometry and frozen["geometry_sha256"] != previous_geometry:
                raise ValueError("A selection changed inputs beyond context membership")


if __name__ == "__main__":
    main()
