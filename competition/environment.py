"""Plan isolated organizer Conda runtimes; --execute builds on Linux only.

No data or model weights are downloaded. Build extensions in a separate staging
directory, preserving organizer source. Existing named environments are refused.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

from .contracts import read_json, sha256, write_json
from .run import check_upstream
from .runtime import bind_python

ROOT = Path(__file__).resolve().parent
FASTGS_COMMIT = "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f"
SETTINGS_FIELDS = {"image_height", "image_width", "tanfovx", "tanfovy", "bg", "scale_modifier",
                   "viewmatrix", "projmatrix", "sh_degree", "campos", "mult", "prefiltered", "debug", "get_flag", "metric_map"}


def audit_fastgs(source):
    """Check the explicit candidate's Python interface without importing CUDA."""
    wrapper = source/"diff_gaussian_rasterization_fastgs/__init__.py"
    tree = ast.parse(wrapper.read_text(encoding="utf-8"))
    settings = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GaussianRasterizationSettings")
    fields = {n.target.id for n in settings.body if isinstance(n, ast.AnnAssign)}
    rasterizer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GaussianRasterizer")
    forward = next(n for n in rasterizer.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    if not SETTINGS_FIELDS <= fields or not {"means3D", "means2D", "dc", "shs", "opacities"} <= {a.arg for a in forward.args.args}:
        raise ValueError("FastGS source does not match organizer Python interface")
    return dict(wrapper_sha256=sha256(wrapper), interface="Python signature checked; CUDA behavior unverified",
                provider="explicit pinned FastGS candidate, not organizer-supplied source", commit=FASTGS_COMMIT)


def plan(args):
    profile = read_json(args.profile)
    env = profile["environments"][args.runtime]
    conda = profile["conda"]
    run = [conda, "run", "--no-capture-output", "-n", env]
    pip = [*run, "python", "-m", "pip", "install"]
    cuda = "12.4" if args.runtime == "fastgs" else "12.8"
    torch, vision = ("2.4.0", "0.19.0") if args.runtime == "fastgs" else ("2.7.0", "0.22.0")
    stage = args.work.resolve()/args.runtime
    constraints = stage/"constraints.txt"
    constrained = [*pip, "-c", str(constraints)]
    commands = [[conda, "create", "-y", "-n", env, "python=3.12", "pip"],
                [*pip, "torch=="+torch, "torchvision=="+vision, "--index-url", "https://download.pytorch.org/whl/cu"+cuda.replace(".", "")],
                [*constrained, "setuptools==75.8.0", "wheel==0.45.1", "ninja==1.11.1.3"]]
    sources = []
    if args.runtime == "resplat":
        commands.append([*constrained, "-r", str(ROOT/"upstream/resplat/requirements.txt")])
        sources.append(ROOT/"upstream/resplat/src/model/encoder/pointops")
    else:
        commands.append([*constrained, "numpy==1.26.4", "scipy==1.15.2", "Pillow==11.1.0",
                         "plyfile==1.1", "tqdm==4.67.1"])
    if args.runtime in ("resplat", "mcmc"):
        commands.append([*constrained, "--no-build-isolation", "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3"])
    provider = None
    if args.runtime == "fastgs":
        if args.fastgs_provider != "pinned-fastgs":
            raise ValueError("Organizer omits FastGS extension source. Explicitly select --fastgs-provider pinned-fastgs for compatibility validation")
        source = ROOT.parent/"third_party/FastGS/submodules/diff-gaussian-rasterization_fastgs"
        provider = audit_fastgs(source)
        sources += [ROOT/"upstream/baseline/3dgs/submodules"/name for name in
                    ("diff-gaussian-rasterization", "simple-knn", "fused-ssim")]
        sources.append(source)
    for source in sources:
        commands.append([*constrained, "--no-build-isolation", "--no-deps", str(stage/"sources"/source.name)])
    commands += [[*run, "python", "-m", "pip", "check"],
                 [*run, "python", "-m", "competition.doctor", "--runtime", args.runtime]]
    return dict(runtime=args.runtime, environment=env, required_cuda_toolkit=cuda,
                constraints=f"torch=={torch}\ntorchvision=={vision}\nnumpy==1.26.4\n",
                sources=[str(p) for p in sources], fastgs_provider=provider, commands=commands)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", choices=["resplat", "fastgs", "mcmc"], required=True)
    p.add_argument("--profile", type=Path, default=ROOT/"profiles/organizer-fresh.json")
    p.add_argument("--work", type=Path, required=True, help="Fresh staging/log directory; outside competition/")
    p.add_argument("--cuda-home", type=Path, help="Installed matching toolkit, required with --execute")
    p.add_argument("--fastgs-provider", choices=["pinned-fastgs"])
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    result = plan(args)
    if not args.execute:
        print(json.dumps(result, indent=2))
        return
    if platform.system() != "Linux" or args.cuda_home is None:
        p.error("--execute requires Linux and --cuda-home pointing to the matching toolkit")
    if args.work.resolve().is_relative_to(ROOT):
        p.error("Keep build products outside competition/ and its source identity")
    check_upstream(ROOT/"upstream", read_json(ROOT/"upstream-lock.json")["commit"])
    if args.runtime == "fastgs":
        check_upstream(ROOT.parent/"third_party/FastGS", FASTGS_COMMIT)
    nvcc = subprocess.check_output([str(args.cuda_home/"bin/nvcc"), "--version"], text=True)
    if "release "+result["required_cuda_toolkit"]+"," not in nvcc:
        raise ValueError("nvcc must match the selected PyTorch CUDA wheel")
    profile = read_json(args.profile)
    existing = json.loads(subprocess.check_output([profile["conda"], "env", "list", "--json"], text=True))["envs"]
    if any(Path(e).name == result["environment"] for e in existing):
        raise FileExistsError("Preserve existing environment; select a fresh environment name in a copied profile")
    stage = args.work.resolve()/args.runtime
    stage.mkdir(parents=True, exist_ok=False)
    (stage/"constraints.txt").write_text(result["constraints"], encoding="utf-8")
    source_hashes = {}
    for name in result["sources"]:
        source = Path(name)
        target = stage/"sources"/source.name
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", "build", "dist", "*.egg-info", "*.so", "__pycache__"))
        source_hashes[source.name] = {p.relative_to(target).as_posix(): sha256(p) for p in sorted(target.rglob("*")) if p.is_file()}
    result.update(source_sha256=source_hashes, nvcc=nvcc, status="building")
    write_json(stage/"build.json", result)
    environment = os.environ.copy()
    environment.update(CUDA_HOME=str(args.cuda_home.resolve()),
                       PATH=str(args.cuda_home.resolve()/"bin")+os.pathsep+environment.get("PATH", ""),
                       TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="4", PYTHONPATH=str(ROOT.parent))
    try:
        for index, command in enumerate(result["commands"]):
            print(json.dumps(dict(step=index, argv=command)), flush=True)
            with (stage/f"step-{index:02}.log").open("w") as log:
                subprocess.run(bind_python(command), cwd=ROOT.parent, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        frozen = subprocess.check_output(bind_python([profile["conda"], "run", "-n", result["environment"], "python", "-m", "pip", "freeze"]), text=True)
        (stage/"pip-freeze.txt").write_text(frozen, encoding="utf-8")
        result["status"] = "imports_checked_gpu_render_acceptance_pending"
    except Exception:
        result["status"] = "failed_preserve_logs"
        raise
    finally:
        write_json(stage/"build.json", result)


if __name__ == "__main__":
    main()
