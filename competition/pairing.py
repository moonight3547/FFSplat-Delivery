"""Immutable control reuse and path-independent prepared scene identities."""
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
from .contracts import read_json, sha256, validate_split, validate_reference, write_json


def freeze_inputs(work, benchmark_id):
    work = Path(work)
    split = validate_split(read_json(work/"split.json"))
    files = ["split.json", "sparse/0/cameras.txt", "sparse/0/images.txt"] + ["images/"+n for n in split["images"]]
    frozen = dict(benchmark_id=benchmark_id, files={n: sha256(work/n) for n in files}, split=split)
    # A may change context only. Geometry, image bytes and held-out membership stay fixed.
    geometry = dict(files={k: v for k, v in frozen["files"].items() if k != "split.json"},
                    split={k: v for k, v in split.items() if k != "context"})
    frozen["geometry_sha256"] = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
    write_json(work/"input_identity.json", frozen)
    return frozen


def verify_inputs(work):
    work = Path(work)
    frozen = read_json(work/"input_identity.json")
    if validate_split(read_json(work/"split.json")) != frozen["split"]:
        raise ValueError("Control split changed")
    expected_files = {"split.json", "sparse/0/cameras.txt", "sparse/0/images.txt"} | {
        "images/"+n for n in frozen["split"]["images"]}
    if set(frozen["files"]) != expected_files:
        raise ValueError("Incomplete prepared input identity")
    for name, expected in frozen["files"].items():
        if not (work/name).resolve().is_relative_to(work.resolve()) or sha256(work/name) != expected:
            raise ValueError("Control prepared input changed")
    return frozen


def clone_work(source, destination):
    """Copy mutable metadata; hardlink image bytes only, with a cross-device copy fallback."""
    source, destination = Path(source), Path(destination)
    verify_inputs(source)
    if destination.exists():
        raise FileExistsError(f"Refuse to overwrite {destination}")
    def copy_file(src, dst):
        if Path(src).is_relative_to(source/"images"):
            try:
                os.link(src, dst)
                return dst
            except OSError as exc:
                if exc.errno not in (errno.EXDEV, errno.EPERM, errno.ENOTSUP): raise
        return shutil.copy2(src, dst)
    shutil.copytree(source, destination, copy_function=copy_file)


def reuse_control(control, destination, cases, geometry_only=False):
    control, destination = Path(control), Path(destination)
    for case in cases:
        source_work, source_result = control/"work"/case, control/"results"/case
        verify_inputs(source_work)
        required = [source_result/"module_b/metrics.json", source_result/"resplat/resplat_world.npz",
                    source_result/"reference_fastgs/module_b.json"]
        if not all(p.is_file() for p in required): raise FileNotFoundError("Paired control is incomplete")
        validate_reference(read_json(source_result/"reference_fastgs/module_b.json"))
        exported = read_json(source_result/"resplat/export_identity.json")
        for kind in ("npz", "ply"):
            if sha256(source_result/f"resplat/resplat_world.{kind}") != exported[kind+"_sha256"]:
                raise ValueError("Control initial Gaussians changed")
        clone_work(source_work, destination/"work"/case)
        reused = ("reference_fastgs",) if geometry_only else ("reference_fastgs", "resplat")
        for stage in reused:
            target = destination/"results"/case/stage
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): raise FileExistsError(str(target))
            target.symlink_to((source_result/stage).resolve(), target_is_directory=True)
        for stage in (("prepare", "reference") if geometry_only else ("prepare", "reference", "resplat")):
            target = destination/"logs"/case/(stage+".json")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): raise FileExistsError(str(target))
            shutil.copy2(control/"logs"/case/(stage+".json"), target)
