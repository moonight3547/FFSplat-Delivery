"""Independent organizer benchmark preparation, execution and strict reporting."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

DEFAULT_SPEC = Path(__file__).parents[1] / "benchmarks/organizer-examples3-v1.json"


def apply_paths(args):
    """Resolve portable workspace bindings relative to their JSON file.

    Explicit CLI paths take precedence. Never stores host paths in the benchmark.
    """
    if not getattr(args, "paths", None):
        return
    config = read(args.paths)
    allowed = {"data", "checkpoint", "assets", "profile"}
    if not set(config) <= allowed or not {"data", "checkpoint", "assets"} <= set(config):
        raise ValueError("Path bindings require data/checkpoint/assets and optional profile only")
    for key, value in config.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Path bindings must contain nonempty path strings")
        if getattr(args, key) is None:
            setattr(args, key, (args.paths.resolve().parent / value).resolve())


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_spec(path=DEFAULT_SPEC):
    spec = read(path)
    if spec.get("protocol") != "organizer-20261028" or spec["context_views"] != 16 or spec["llffhold"] != 8:
        raise ValueError("Not a supported organizer protocol")
    cases = spec["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and unique")
    for c in cases:
        for key in ("source", "scene", "group"):
            if Path(c[key]).name != c[key] or c[key] in (".", ".."):
                raise ValueError("Scene source must be a plain directory name")
        if c["id"] != c["group"] + "/" + c["scene"]:
            raise ValueError("Invalid case identity")
    return spec


def inventory(spec, data):
    """Freeze source bytes without transforming them into the legacy dataset."""
    cases = []
    for case in spec["cases"]:
        scene = Path(data) / case["source"] / case["scene"]
        if not scene.is_dir():
            raise FileNotFoundError(f"Organizer scene missing: {scene}")
        files = {p.relative_to(scene).as_posix(): dict(bytes=p.stat().st_size, sha256=digest(p))
                 for p in sorted(scene.rglob("*")) if p.is_file()}
        if not any(Path(n).suffix.lower() in (".png", ".jpg", ".jpeg") for n in files):
            raise ValueError(f"No images for {case['id']}")
        cases.append(dict(id=case["id"], files=files))
    identity = dict(benchmark_id=spec["benchmark_id"], cases=cases)
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def positive(value, name, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"Missing/invalid {name}")
    return float(value)


def geometry_digest(frozen):
    geometry = dict(files={k: v for k, v in frozen["files"].items() if k != "split.json"},
                    split={k: v for k, v in frozen["split"].items() if k != "context"})
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def summarize(spec, work, groups=None):
    """Missing stages never become zero, and partial runs never get a valid total."""
    root = Path(work)
    identity = read(root / "run_identity.json")
    if identity["benchmark"] != spec:
        raise ValueError("Run identity differs from requested benchmark")
    rows = []
    for case in spec["cases"]:
        if groups and case["group"] not in groups: continue
        folder = root / "results" / case["id"]
        row = dict(id=case["id"], group=case["group"], complete=False, valid=None)
        try:
            a = read(root / "work" / case["id"] / "module_a.json")
            r = read(folder / "resplat/resplat.json")
            b = read(folder / "module_b/module_b.json")
            ref = read(folder / "reference_fastgs/module_b.json")
            metrics = read(folder / "module_b/metrics.json")
            split = read(root / "work" / case["id"] / "split.json")
            frozen = read(root / "work" / case["id"] / "input_identity.json")
            if frozen["split"] != split:
                raise ValueError("Prepared split differs from frozen identity")
            names = split["images"]
            if (names != sorted(set(names)) or split.get("llffhold") != 8 or split["test"] != names[::8] or
                    split["train"] != [n for i, n in enumerate(names) if i % 8] or
                    len(split["context"]) != min(16, len(split["train"])) or len(set(split["context"])) != len(split["context"])):
                raise ValueError("Organizer split contract changed")
            expected_files = {"split.json", "sparse/0/cameras.txt", "sparse/0/images.txt"} | {"images/"+n for n in names}
            if set(frozen["files"]) != expected_files or frozen.get("geometry_sha256") != geometry_digest(frozen):
                raise ValueError("Incomplete or changed geometry identity")
            scene_work = root / "work" / case["id"]
            for name, expected in frozen["files"].items():
                path = scene_work / name
                if not path.resolve().is_relative_to(scene_work.resolve()) or digest(path) != expected:
                    raise ValueError("Prepared input bytes differ from frozen identity")
            exported = read(folder/"resplat/export_identity.json")
            for kind in ("npz", "ply"):
                if digest(folder/f"resplat/resplat_world.{kind}") != exported[kind+"_sha256"]:
                    raise ValueError("Initial Gaussians changed")
            if set(split["train"]) & set(split["test"]) or not set(split["context"]) <= set(split["train"]):
                raise ValueError("Reconstruction/evaluation leakage")
            if metrics["num_views"] != len(split["test"]) or sorted(v["name"] for v in metrics["per_view"]) != sorted(split["test"]):
                raise ValueError("Metrics do not cover exactly the frozen held-out views")
            if ref.get("trainer") != "fastgs" or ref["iterations"] != 30000 or not str(ref["init_mode"]).startswith("random:"):
                raise ValueError("Reference must be FastGS random-init full30000")
            times = {k: positive(d.get("total_seconds"), k, zero=True) for k, d in (("A", a), ("ReSplat", r), ("B", b))}
            budget = positive(ref.get("total_seconds"), "reference30K")
            mean = {k: positive(metrics["mean"].get(k), k, zero=True) for k in ("psnr", "ssim", "lpips")}
            wall = {k: positive(read(root/"logs"/case["id"]/(stage+".json"))["subprocess_wall_seconds"], stage)
                    for k, stage in (("A", "prepare"), ("ReSplat", "resplat"), ("B", "train"))}
            total = sum(times.values())
            reference_wall = positive(read(root/"logs"/case["id"]/"reference.json")["subprocess_wall_seconds"], "reference process")
            row.update(complete=True, valid=total < budget, metrics=mean, stage_seconds=times,
                       a_selection_seconds=positive(a.get("selection_seconds", 0), "A selection", zero=True),
                       b_iterations=b.get("iterations"), b_allocation_met=b.get("allocation_met"),
                       end_to_end_seconds=total, subprocess_wall_seconds=wall,
                       subprocess_end_to_end_seconds=sum(wall.values()),
                       reference_subprocess_seconds=reference_wall, subprocess_valid=sum(wall.values()) < reference_wall,
                       reference_trainer="fastgs", reference_init_mode=ref["init_mode"],
                       reference30k_seconds=budget, speedup=budget/total if total else None,
                       official_score_100=None, input_identity_sha256=digest(root/"work"/case["id"]/"input_identity.json"))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            row["error"] = str(exc)
        rows.append(row)
    if not rows: raise ValueError("No matching organizer cases")
    complete = all(r["complete"] for r in rows)
    return dict(benchmark_id=spec["benchmark_id"], protocol=spec["protocol"],
        policy=identity.get("policy", "unknown"), a_policy=identity.get("a_policy", "uniform16"),
        scope="full benchmark" if len(rows) == len(spec["cases"]) else "explicit group subset",
        subprocess_timing_scope="prepare + resplat (including ranked A selection, if enabled) + B",
        paired_control=identity.get("paired_control"),
        complete=complete, valid=all(r["valid"] is True for r in rows) if complete else None,
        official_score_100=None, score_status="New codebase supplies metrics and E2E timing, no confirmed /100 formula",
        mean_metrics={k: statistics.mean(r["metrics"][k] for r in rows) for k in ("psnr", "ssim", "lpips")} if complete else None,
        mean_end_to_end_seconds=statistics.mean(r["end_to_end_seconds"] for r in rows) if complete else None,
        mean_stage_seconds={k: statistics.mean(r["stage_seconds"][k] for r in rows)
                            for k in ("A", "ReSplat", "B")} if complete else None,
        rows=rows)


def compare_suite(spec, root, groups=None):
    """Independently reject unpaired cells before reporting factorial deltas."""
    root = Path(root)
    cells = {name: summarize(spec, root/name, groups) for name in ("O0", "A", "B", "AB")}
    if not all(r["complete"] for r in cells.values()):
        return dict(complete=False, official_score_100=None, cells=cells, comparison=None)
    identities = {name: read(root/name/"run_identity.json") for name in cells}
    for name, (a, b) in {"O0": ("uniform16", "official-fastgs"), "A": ("ranked16", "official-fastgs"),
                          "B": ("uniform16", "mcmc-ctf"), "AB": ("ranked16", "mcmc-ctf")}.items():
        if (cells[name]["a_policy"], cells[name]["policy"]) != (a, b):
            raise ValueError("Suite has incorrect A/B factor assignments")
        for key in ("hardware", "profile", "checkpoint_sha256", "source_sha256", "assets_sha256"):
            if not identities[name].get(key) or identities[name][key] != identities["O0"].get(key):
                raise ValueError(f"Unpaired suite {key}")
    if not identities["B"].get("b_settings") or identities["B"]["b_settings"] != identities["AB"].get("b_settings"):
        raise ValueError("B settings differ across factor cells")
    for target, control, mode in (("A", "O0", "geometry"), ("B", "O0", "initial_gaussians"), ("AB", "A", "initial_gaussians")):
        pair = identities[target].get("paired_control", {})
        if pair.get("identity_sha256") != digest(root/control/"run_identity.json") or pair.get("reuse") != mode:
            raise ValueError("Missing or changed control lineage")
    per_case = []
    for index, row in enumerate(cells["O0"]["rows"]):
        case = row["id"]
        inputs = {c: read(root/c/"work"/case/"input_identity.json") for c in cells}
        geometry = inputs["O0"].get("geometry_sha256")
        if not geometry or any(x.get("geometry_sha256") != geometry for x in inputs.values()):
            raise ValueError("Suite geometry or holdout changed")
        reference = digest(root/"O0/results"/case/"reference_fastgs/module_b.json")
        if any(digest(root/c/"results"/case/"reference_fastgs/module_b.json") != reference for c in cells):
            raise ValueError("Suite reference timing changed")
        for left, right in (("O0", "B"), ("A", "AB")):
            for suffix in ("resplat_world.npz", "resplat_world.ply", "resplat.json"):
                if digest(root/left/"results"/case/"resplat"/suffix) != digest(root/right/"results"/case/"resplat"/suffix):
                    raise ValueError("B cells did not reuse exact initialization/timing")
            for suffix in ("split.json", "module_a.json"):
                if digest(root/left/"work"/case/suffix) != digest(root/right/"work"/case/suffix):
                    raise ValueError("B cells have changed A inputs/timing")
        values = {c: cells[c]["rows"][index] for c in cells}
        per_case.append(dict(id=case, cells=values, deltas_vs_O0={c: {
            **{k: values[c]["metrics"][k]-row["metrics"][k] for k in ("psnr", "ssim", "lpips")},
            "end_to_end_seconds": values[c]["end_to_end_seconds"]-row["end_to_end_seconds"]}
            for c in ("A", "B", "AB")}))
    means = {c: cells[c]["mean_metrics"] for c in cells}
    return dict(complete=True, official_score_100=None, cells=cells, comparison=per_case,
        interaction={k: means["AB"][k]-means["A"][k]-means["B"][k]+means["O0"][k] for k in means["O0"]},
        interpretation="Raw metric deltas; PSNR/SSIM higher is better, LPIPS and time lower is better. Eligibility remains per case.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    p.add_argument("--action", choices=["describe", "inventory", "run", "summarize", "suite-plan", "suite", "compare"], default="describe")
    p.add_argument("--enhance", type=Path, default=Path(__file__).parents[1])
    p.add_argument("--data", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--assets", type=Path, help="Enhance competition.assets manifest; required for run/suite")
    p.add_argument("--paths", type=Path, help="Host path bindings JSON; relative to its file, CLI paths override it")
    p.add_argument("--work", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--profile", type=Path)
    p.add_argument("--policy", choices=["official-fastgs", "mcmc-ctf"], default="official-fastgs")
    p.add_argument("--a-policy", choices=["uniform16", "ranked16"], default="uniform16")
    p.add_argument("--b-settings", type=Path)
    p.add_argument("--from-control", type=Path)
    p.add_argument("--reuse-geometry-only", action="store_true")
    p.add_argument("--groups", nargs="+", choices=["dl3dv", "hw", "mipnerf360"])
    p.add_argument("--stage", default="all", choices=["all", "prepare", "reference", "resplat", "train", "render", "metrics"])
    args = p.parse_args()
    apply_paths(args)
    spec = load_spec(args.spec)
    if args.action == "inventory":
        if args.data is None: p.error("--data required")
        result = inventory(spec, args.data)
    elif args.action in ("summarize", "compare"):
        if args.work is None: p.error("--work required")
        result = (compare_suite if args.action == "compare" else summarize)(spec, args.work, args.groups)
    elif args.action == "describe" and args.data is None:
        result = spec
    else:
        if any(x is None for x in (args.data, args.checkpoint, args.work)):
            p.error("--data, --checkpoint and --work required for an execution plan/run")
        suite = args.action in ("suite-plan", "suite")
        if suite and (args.from_control or args.reuse_geometry_only or args.stage != "all" or
                      args.policy != "official-fastgs" or args.a_policy != "uniform16"):
            p.error("Suite defines its own four policies and reuse; use run for individual cells")
        command = [sys.executable, "-m", "competition.suite" if suite else "competition.run", "--spec", str(args.spec.resolve()),
            "--data", str(args.data.resolve()), "--checkpoint", str(args.checkpoint.resolve()),
            "--work", str(args.work.resolve())]
        if not suite: command += ["--stage", args.stage, "--policy", args.policy, "--a-policy", args.a_policy]
        if args.action == "suite": command.append("--execute")
        if args.action == "describe": command.append("--describe")
        if args.profile: command.extend(["--profile", str(args.profile.resolve())])
        if args.assets: command.extend(["--assets", str(args.assets.resolve())])
        if args.b_settings: command.extend(["--b-settings", str(args.b_settings.resolve())])
        if args.from_control: command.extend(["--from-control", str(args.from_control.resolve())])
        if args.reuse_geometry_only: command.append("--reuse-geometry-only")
        if args.groups: command.extend(["--groups", *args.groups])
        subprocess.run(command, cwd=args.enhance, check=True)
        return
    value = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists(): raise FileExistsError("Preserve prior evidence; select a new output path")
        args.output.write_text(value, encoding="utf-8")
    print(value)


if __name__ == "__main__":
    main()
