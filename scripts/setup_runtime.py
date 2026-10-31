#!/usr/bin/env python3
"""Create two fresh, isolated CUDA runtimes; existing shared environments stay intact.

Default: install. --plan only prints commands and checks without creating files.
Use --root on a large disk. A failed installation is resumable only with its own
matching manifest and --resume; an unrelated existing prefix is always refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO / "requirements/runtime-lock.json"
GIB = 1024 ** 3


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def conda_executable(requested=None):
    choices = [requested, os.environ.get("CONDA_EXE"), shutil.which("conda"),
               "/opt/conda/bin/conda", str(Path.home() / "miniforge3/bin/conda"),
               str(Path.home() / "miniconda3/bin/conda")]
    for candidate in choices:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    raise RuntimeError("Conda not found. Install Miniforge (https://github.com/conda-forge/miniforge), "
                       "then pass --conda /absolute/path/bin/conda. No existing environment is changed.")


def conda_python(conda, prefix, *args):
    # Explicit executable also prevents an inherited environment PATH selecting
    # another Python after conda activation.
    return [conda, "run", "--no-capture-output", "-p", str(prefix),
            str(Path(prefix) / "bin/python"), *map(str, args)]


def existing_parent(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def check_space(reservations, buffer_bytes=2 * GIB):
    """Sum reservations per filesystem, including simultaneous prefix/cache use."""
    devices = {}
    for path, size in reservations:
        parent = existing_parent(path)
        device = parent.stat().st_dev
        item = devices.setdefault(device, {"path": str(parent), "required": buffer_bytes, "allocations": []})
        item["required"] += size
        item["allocations"].append({"path": str(path), "bytes": size})
    for item in devices.values():
        item["free"] = shutil.disk_usage(item["path"]).free
        item["sufficient"] = item["free"] >= item["required"]
    return list(devices.values())


def toolkit_version(home):
    nvcc = Path(home) / "bin/nvcc"
    if not nvcc.is_file():
        return None
    value = subprocess.check_output([str(nvcc), "--version"], text=True)
    match = re.search(r"release (\d+\.\d+),", value)
    return match.group(1) if match else None


def runtime_header_paths(prefix, cuda_home):
    # Minimal nvcc installations do not include cublas/cusparse/cusolver
    # headers. The matching Torch NVIDIA wheels already provide those headers;
    # expose them to the compiler without installing a second library runtime.
    base = Path(prefix) / "lib/python3.12/site-packages/nvidia"
    return os.pathsep.join(map(str, [Path(cuda_home) / "include", *sorted(base.glob("*/include"))]))


def find_toolkit(version, explicit=None):
    if explicit:
        home = Path(explicit).resolve()
        if toolkit_version(home) != version:
            raise ValueError(f"{home}/bin/nvcc must be CUDA {version}")
        return str(home)
    for home in (f"/usr/local/cuda-{version}", "/usr/local/cuda", os.environ.get("CUDA_HOME")):
        if home and toolkit_version(home) == version:
            return str(Path(home).resolve())
    return None


def system_tools():
    needed = {name: shutil.which(name) for name in ("git", "gcc", "g++", "make", "cmake", "colmap", "nvidia-smi")}
    return {"executables": needed, "missing": [name for name, path in needed.items() if path is None],
            "debian_install": "sudo apt-get update && sudo apt-get install -y build-essential git cmake ninja-build colmap libgl1 libglib2.0-0",
            "driver_policy": "Existing NVIDIA driver is checked, never installed or replaced."}


def source_identity():
    paths = [Path(__file__), LOCK_PATH, *sorted((REPO / "requirements").glob("*.txt")),
             REPO / "resplat/requirements.txt"]
    for source in [REPO / "resplat/src/model/encoder/pointops", REPO / "baseline/3dgs/submodules"]:
        paths.extend(p for p in sorted(source.rglob("*")) if p.is_file()
                     and not any(part in (".git", "build", "__pycache__") for part in p.parts)
                     and p.suffix not in (".so", ".pyc"))
    return {str(p.relative_to(REPO)): digest(p) for p in paths}


def build_plan(args):
    lock = read(LOCK_PATH)
    root = args.root.expanduser().resolve()
    prefixes = {runtime: (getattr(args, runtime + "_prefix") or root / "envs" / runtime).expanduser().resolve()
                for runtime in ("resplat", "fastgs")}
    if prefixes["resplat"] == prefixes["fastgs"]:
        raise ValueError("ReSplat and FastGS require two distinct prefixes")
    for prefix in prefixes.values():
        if prefix == Path(sys.prefix).resolve() or prefix in (Path("/"), Path.home(), REPO):
            raise ValueError(f"Refusing unsafe/shared prefix: {prefix}")
    conda = conda_executable(args.conda)
    toolkits = {"resplat": find_toolkit("12.8", args.cuda128_home),
                "fastgs": find_toolkit("12.4", args.cuda124_home)}
    profile = {"schema_version": 1, "profile": "delivery-isolated-prefixes", "execution_os": "linux",
               "device": "cuda:0", "conda": conda, "environments": {k: str(v) for k, v in prefixes.items()},
               "environment": {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "MAX_JOBS": str(args.jobs),
                               "NVIDIA_TF32_OVERRIDE": "0", "QT_QPA_PLATFORM": "offscreen"}}
    profile["official_adam_manifest"] = str(root / "official-adam-manifest.json")
    profile["environment_by_runtime"] = {}
    for runtime in ("resplat", "fastgs"):
        home = toolkits[runtime] or str(prefixes[runtime])
        profile["environment_by_runtime"][runtime] = {
            "CUDA_HOME": home, "PATH": str(Path(home) / "bin") + os.pathsep + os.environ.get("PATH", ""),
            "CPATH": str(Path(home) / "include")}
    # Conservative installed sizes plus one serial pip/build working set.
    # This is a reservation estimate, not a measured package-size guarantee.
    reservations = [(prefixes["resplat"], 10 * GIB), (prefixes["fastgs"], 6 * GIB), (root, 4 * GIB)]
    return {"root": str(root), "prefixes": {k: str(v) for k, v in prefixes.items()},
            "conda": conda, "toolkits": toolkits, "lock": lock, "source_sha256": source_identity(),
            "profile": profile, "system": system_tools(), "space": check_space(reservations),
            "profile_path": str(root / "profile.json"), "acceptance_scope": "Install/import/CUDA checks; three-dataset training/evaluation is separate."}


class Installer:
    def __init__(self, args, plan):
        self.args, self.plan = args, plan
        self.root = Path(plan["root"])
        self.manifest_path = self.root / "install-manifest.json"
        identity = {k: plan[k] for k in ("prefixes", "conda", "lock", "source_sha256")}
        self.identity = identity
        if self.manifest_path.exists():
            self.manifest = read(self.manifest_path)
            prior = json.loads(json.dumps(self.manifest["identity"]))
            current = json.loads(json.dumps(identity))
            old_installer = prior["source_sha256"].pop("scripts/setup_runtime.py")
            new_installer = current["source_sha256"].pop("scripts/setup_runtime.py")
            if not args.resume or prior != current:
                raise FileExistsError("Existing installer root: use --resume only for the identical installer/lock/prefixes, or a fresh --root")
            if old_installer != new_installer:
                self.manifest.setdefault("installer_revisions", []).append({
                    "original_sha256": old_installer, "resumed_sha256": new_installer,
                    "unix": time.time(), "scope": "Installer-only repair; lock, dependencies, vendored build sources and prefixes unchanged"})
            self.manifest["status"] = "building"
        else:
            for prefix in plan["prefixes"].values():
                if Path(prefix).exists():
                    raise FileExistsError(f"Existing prefix is protected: {prefix}")
            if self.root.exists() and any(self.root.iterdir()):
                raise FileExistsError(f"Installer root must be empty: {self.root}")
            self.manifest = {"identity": identity, "status": "building", "steps": {}, "started_unix": time.time(),
                             "system": plan["system"], "space_before": plan["space"], "toolkits": {}}
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("logs", "tmp", "conda-pkgs", "build", "sources", "torch-extensions"):
            (self.root / name).mkdir(exist_ok=True)
        self.env = os.environ.copy()
        # New caches only. Remove overlays from other environments.
        for key in ("PYTHONHOME", "PYTHONPATH", "PIP_TARGET", "PIP_PREFIX", "CUDA_HOME", "CUDA_PATH"):
            self.env.pop(key, None)
        self.env.update(PYTHONNOUSERSITE="1", CONDA_PKGS_DIRS=str(self.root / "conda-pkgs"),
                        PIP_NO_CACHE_DIR="1", PIP_DISABLE_PIP_VERSION_CHECK="1", TMPDIR=str(self.root / "tmp"),
                        TORCH_EXTENSIONS_DIR=str(self.root / "torch-extensions"),
                        TORCH_CUDA_ARCH_LIST=args.arch, MAX_JOBS=str(args.jobs), NVIDIA_TF32_OVERRIDE="0",
                        QT_QPA_PLATFORM="offscreen", BUILD_NO_CUDA="0")
        write(self.manifest_path, self.manifest)

    def run(self, name, argv, *, cwd=None, env=None, retries=1):
        previous = self.manifest["steps"].get(name)
        if previous and previous["status"] == "complete":
            if previous["argv"] != list(map(str, argv)):
                raise RuntimeError(f"Completed step command changed: {name}; use a fresh installer root")
            return
        for location in (self.root, *map(Path, self.plan["prefixes"].values())):
            if shutil.disk_usage(existing_parent(location)).free < 2 * GIB:
                raise RuntimeError(f"Filesystem buffer below 2GiB before {name}: {location}; preserve partial install")
        log = self.root / "logs" / f"{name}.log"
        record = {"argv": list(map(str, argv)), "cwd": str(cwd or REPO), "status": "running", "started_unix": time.time()}
        self.manifest["steps"][name] = record
        write(self.manifest_path, self.manifest)
        print(json.dumps({"step": name, "argv": record["argv"], "log": str(log)}), flush=True)
        try:
            with log.open("a") as stream:
                record["attempts"] = []
                for attempt in range(1, retries + 1):
                    row = {"attempt": attempt, "started_unix": time.time()}
                    record["attempts"].append(row)
                    try:
                        subprocess.run(record["argv"], cwd=cwd or REPO, env=env or self.env,
                                       stdout=stream, stderr=subprocess.STDOUT, check=True)
                        row["status"] = "complete"
                        break
                    except subprocess.CalledProcessError:
                        row["status"] = "failed"
                        if attempt == retries:
                            raise
                        time.sleep(attempt)
        except BaseException:
            record["status"] = "failed_preserve_evidence"
            self.manifest["status"] = "failed_preserve_evidence"
            raise
        else:
            record["status"] = "complete"
        finally:
            record["finished_unix"] = time.time()
            write(self.manifest_path, self.manifest)

    def source(self, name):
        spec = self.plan["lock"]["sources"][name]
        destination = self.root / "sources" / name
        if not destination.exists():
            self.run(f"{name}-init", ["git", "init", str(destination)])
            self.run(f"{name}-remote", ["git", "remote", "add", "origin", spec["url"]], cwd=destination)
        if not (destination / ".git").is_dir():
            raise RuntimeError(f"Preserve incomplete source directory and select a fresh root: {destination}")
        # Fetch the pin directly: no unrelated repository history/assets. TLS
        # interruptions remain in the per-attempt log and are retried finitely.
        self.run(f"{name}-fetch", ["git", "-c", "http.version=HTTP/1.1", "fetch", "--depth", "1", "origin", spec["commit"]],
                 cwd=destination, retries=3)
        if f"{name}-clone" in self.manifest["steps"] and self.manifest["steps"][f"{name}-clone"]["status"] != "complete":
            self.manifest["steps"][f"{name}-clone"]["superseded_by"] = f"{name}-fetch"
        self.run(f"{name}-checkout", ["git", "checkout", "--detach", spec["commit"]], cwd=destination)
        self.run(f"{name}-submodules", ["git", "-c", "http.version=HTTP/1.1", "submodule", "update", "--init", "--recursive", "--depth", "1"],
                 cwd=destination, retries=3)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination, text=True).strip()
        if head != spec["commit"]:
            raise RuntimeError(f"Source commit mismatch: {name}")
        return destination

    def staged(self, source, runtime, name):
        destination = self.root / "build" / runtime / name
        if not destination.exists():
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", "build", "dist", "*.egg-info", "*.so", "__pycache__"))
        # Record the source tree before pip adds its build outputs.
        key = f"{runtime}/{name}"
        if key not in self.manifest.setdefault("build_sources", {}):
            self.manifest["build_sources"][key] = {
                "source": str(source), "files": {p.relative_to(destination).as_posix(): digest(p)
                            for p in sorted(destination.rglob("*")) if p.is_file()}}
        write(self.manifest_path, self.manifest)
        return destination

    def install_runtime(self, runtime):
        prefix = Path(self.plan["prefixes"][runtime])
        conda, lock = self.plan["conda"], self.plan["lock"]
        spec = lock["runtimes"][runtime]
        self.run(runtime + "-create", [conda, "create", "-y", "--override-channels", "-c", lock["conda_channel"],
                                      "-p", prefix, "python=" + lock["python"], "pip=25.0.1"])
        home = self.plan["toolkits"][runtime]
        if home is None:
            cuda = lock["cuda_components"][spec["cuda"]]
            self.run(runtime + "-cuda-toolkit", [conda, "install", "-y", "--override-channels", "-c", cuda["channel"],
                     "-c", lock["conda_channel"], "-p", prefix, *cuda["packages"]])
            home = str(prefix)
        if toolkit_version(home) != spec["cuda"]:
            raise RuntimeError(f"{runtime}: nvcc version does not match {spec['cuda']}")
        self.manifest["toolkits"][runtime] = {"home": home, "version": toolkit_version(home)}
        environment = self.env.copy()
        environment.update(CUDA_HOME=home, PATH=str(Path(home) / "bin") + os.pathsep + environment.get("PATH", ""))
        self.manifest.setdefault("build_environment", {})[runtime] = {
            key: environment.get(key) for key in ("CUDA_HOME", "TORCH_CUDA_ARCH_LIST", "MAX_JOBS", "NVIDIA_TF32_OVERRIDE")}
        python = conda_python(conda, prefix)
        constraint = self.root / f"{runtime}-constraints.txt"
        constraint.write_text(f"torch=={spec['torch']}\ntorchvision=={spec['torchvision']}\nnumpy==1.26.4\n")
        pip = [*python, "-m", "pip", "install", "--no-cache-dir", "-c", str(constraint)]
        self.run(runtime + "-torch", [*pip, "torch==" + spec["torch"], "torchvision==" + spec["torchvision"],
                                     "--index-url", spec["index_url"]], env=environment)
        self.run(runtime + "-build-tools", [*pip, "-r", REPO / "requirements/build.txt"], env=environment)
        self.run(runtime + "-requirements", [*pip, "-r", REPO / f"requirements/{runtime}.txt"], env=environment)
        environment["CPATH"] = runtime_header_paths(prefix, home)
        self.plan["profile"]["environment_by_runtime"][runtime]["CPATH"] = environment["CPATH"]
        self.manifest["build_environment"][runtime]["CPATH"] = environment["CPATH"]
        if runtime == "resplat":
            sources = [("pointops", REPO / "resplat/src/model/encoder/pointops"), ("gsplat", self.source("gsplat"))]
        else:
            fastgs = self.source("fastgs")
            official = self.source("official_adam")
            sources = [(name, REPO / "baseline/3dgs/submodules" / name)
                       for name in ("diff-gaussian-rasterization", "simple-knn", "fused-ssim")]
            sources += [("diff-gaussian-rasterization_fastgs", fastgs / "submodules/diff-gaussian-rasterization_fastgs"),
                        ("FasterGSCudaBackend", official / "FasterGSCudaBackend")]
        for name, source in sources:
            if not (source / "setup.py").is_file():
                raise FileNotFoundError(f"Required pinned build source missing: {source}")
            staged = self.staged(source, runtime, name)
            self.run(runtime + "-build-" + name, [*pip, "--no-build-isolation", "--no-deps", staged], env=environment)
        self.run(runtime + "-pip-check", [*python, "-m", "pip", "check"], env=environment)
        # Run doctor as a module without leaking a PYTHONPATH overlay into the
        # installed environment; cwd is the checked-out delivery repository.
        self.run(runtime + "-doctor", [*python, "-m", "competition.doctor", "--runtime", runtime], env=environment)
        self.run(runtime + "-verify", [*python, str(Path(__file__).resolve()), "--verify-runtime", runtime,
                 "--expected-prefix", prefix, "--verification-output", self.root / f"{runtime}-verification.json"], env=environment)
        for name, argv in (("pip-freeze", [*python, "-m", "pip", "freeze", "--all"]),
                           ("conda-explicit", [conda, "list", "-p", str(prefix), "--explicit"])):
            output = subprocess.check_output(argv, env=environment, text=True)
            (self.root / f"{runtime}-{name}.txt").write_text(output)

    def execute(self):
        for runtime in ("resplat", "fastgs"):
            self.install_runtime(runtime)
        write(self.root / "profile.json", self.plan["profile"])
        self.manifest.update(status="installed_imports_and_cuda_checked", finished_unix=time.time(),
                             profile_path=str(self.root / "profile.json"),
                             acceptance_scope=self.plan["acceptance_scope"])
        write(self.manifest_path, self.manifest)
        print(json.dumps({"status": self.manifest["status"], "profile": str(self.root / "profile.json"),
                          "manifest": str(self.manifest_path)}), flush=True)


def verify_runtime(runtime, prefix, output):
    import importlib
    import torch
    lock = read(LOCK_PATH)
    expected = lock["runtimes"][runtime]
    if Path(sys.prefix).resolve() != prefix.resolve() or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Python did not bind to the requested new prefix")
    if torch.__version__.split("+")[0] != expected["torch"] or torch.version.cuda != expected["cuda"]:
        raise RuntimeError("Torch/CUDA version differs from the installation lock")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the new environment")
    names = (["torch", "torchvision", "gsplat", "gsplat.csrc", "pointops._C", "lpips", "skimage", "cv2"]
             if runtime == "resplat" else ["torch", "torchvision", "diff_gaussian_rasterization_fastgs._C",
                 "diff_gaussian_rasterization._C", "simple_knn._C", "fused_ssim_cuda", "FasterGSCudaBackend._C",
                 "FasterGSCudaBackend.torch_bindings.adam", "lpips"])
    modules = {}
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(prefix.resolve()):
            raise RuntimeError(f"Module escaped the new prefix: {name}: {path}")
        modules[name] = {"file": str(path), "sha256": digest(path), "version": getattr(module, "__version__", None)}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = torch.tensor([1., 2., 3.], device="cuda", requires_grad=True)
    (x.square().sum()).backward()
    torch.cuda.synchronize()
    assert torch.equal(x.grad, torch.tensor([2., 4., 6.], device="cuda"))
    write(output, {"runtime": runtime, "python": sys.version, "prefix": str(prefix), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "modules": modules, "gpu": torch.cuda.get_device_name(0),
                   "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
                   "scope": "Imports plus Torch CUDA forward/backward; renderer and full-dataset acceptance are separate."})
    if runtime == "fastgs":
        source = output.parent / "sources/official_adam"
        source_class = source / "FasterGSCudaBackend/FasterGSCudaBackend/torch_bindings/adam.py"
        source_cuda = source / "FasterGSCudaBackend/FasterGSCudaBackend/adam/src/adam.cu"
        installed_class = Path(modules["FasterGSCudaBackend.torch_bindings.adam"]["file"])
        extension = Path(modules["FasterGSCudaBackend._C"]["file"])
        if digest(source_class) != digest(installed_class):
            raise RuntimeError("Installed official Adam Python class differs from pinned source")
        write(output.parent / "official-adam-manifest.json", {
            "official_source": str(source), "official_commit": lock["sources"]["official_adam"]["commit"],
            "isolated_site": str(extension.parents[1]), "Torch": torch.__version__, "CUDA": torch.version.cuda,
            "ABI": bool(torch._C._GLIBCXX_USE_CXX11_ABI), "Python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "class_file": str(installed_class), "class_SHA": digest(installed_class),
            "extension_file": str(extension), "extension_SHA": digest(extension), "CUDA_source_SHA": digest(source_cuda),
            "build_flags": {"cxx": ["-std=c++17", "-O3"], "nvcc": ["-std=c++17", "-O3", "-use_fast_math"],
                            "TORCH_CUDA_ARCH_LIST": os.environ["TORCH_CUDA_ARCH_LIST"], "MAX_JOBS": os.environ["MAX_JOBS"]},
            "scope": "Official optimizer component installed; full training quality/timing acceptance is separate."})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "outputs/runtime")
    parser.add_argument("--conda")
    parser.add_argument("--resplat-prefix", type=Path)
    parser.add_argument("--fastgs-prefix", type=Path)
    parser.add_argument("--cuda128-home", type=Path)
    parser.add_argument("--cuda124-home", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--arch", default="8.9", help="TORCH_CUDA_ARCH_LIST; use the target GPU capability")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--install-system-deps", action="store_true")
    parser.add_argument("--verify-runtime", choices=["resplat", "fastgs"], help=argparse.SUPPRESS)
    parser.add_argument("--expected-prefix", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--verification-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.verify_runtime:
        verify_runtime(args.verify_runtime, args.expected_prefix, args.verification_output)
        return
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    plan = build_plan(args)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if platform.system() != "Linux":
        parser.error("Installation requires Linux/CUDA; --plan is available on the control machine")
    if plan["system"]["missing"]:
        missing = set(plan["system"]["missing"])
        if "nvidia-smi" in missing:
            raise RuntimeError("NVIDIA driver is missing. Provision a CUDA GPU machine; this installer never changes drivers.")
        if not args.install_system_deps:
            raise RuntimeError("Missing tools: " + ", ".join(sorted(missing)) + ". Run: " + plan["system"]["debian_install"])
        if os.geteuid() != 0 or not shutil.which("apt-get"):
            raise RuntimeError("--install-system-deps needs root and apt-get; otherwise run the printed install command")
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "build-essential", "git", "cmake", "ninja-build", "colmap", "libgl1", "libglib2.0-0"], check=True)
        plan["system"] = system_tools()
        if plan["system"]["missing"]:
            raise RuntimeError("Required tools remain missing: " + str(plan["system"]["missing"]))
    if not args.resume and not all(row["sufficient"] for row in plan["space"]):
        raise RuntimeError("Insufficient reserved installation space (includes 2GiB filesystem buffer): " + json.dumps(plan["space"]))
    Installer(args, plan).execute()


if __name__ == "__main__":
    main()
