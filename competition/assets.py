"""Register or recheck manually downloaded organizer assets without loading weights.

Paths are supplied on each host; the saved manifest contains portable relative names
and byte hashes only. This checks file integrity, not CUDA or reconstruction quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from .contracts import read_json, sha256

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def reject_placeholder(path, allow_empty=False):
    with path.open("rb") as stream:
        head = stream.read(512).lstrip().lower()
    if (not head and not allow_empty) or head.startswith((b"version https://git-lfs.github.com/spec/", b"<!doctype html", b"<html")):
        raise ValueError(f"Empty file, LFS pointer or login page: {path}")


def checkpoint_record(path, lock):
    path = Path(path)
    if path.name != lock["checkpoint_name"] or not path.is_file():
        raise ValueError("Supply the organizer 16-view checkpoint with its original filename")
    reject_placeholder(path)
    # Never unpickle third-party weights during preparation. Check the archive and
    # every ZIP CRC (streamed); legacy pickle files are left to the model loader.
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not any(n.endswith("/data.pkl") or n == "data.pkl" for n in names):
                raise ValueError("ZIP is not a PyTorch checkpoint")
            if not any("/data/" in n or n.startswith("data/") for n in names):
                raise ValueError("Checkpoint has no tensor storage")
            if archive.testzip() is not None:
                raise ValueError("Checkpoint ZIP CRC check failed")
        kind = "torch_zip_crc_checked"
    else:
        with path.open("rb") as stream:
            if stream.read(1) != b"\x80":
                raise ValueError("Unrecognized or truncated checkpoint container")
        kind = "legacy_pickle_not_deserialized"
    digest = sha256(path)
    expected = lock.get("checkpoint_sha256")
    if expected and expected != digest:
        raise ValueError("Checkpoint differs from published SHA256")
    return dict(name=path.name, bytes=path.stat().st_size, sha256=digest,
                container=kind, published_sha256_verified=bool(expected))


def collect(spec, data, checkpoint, lock=None):
    from PIL import Image
    lock = lock or read_json(ROOT/"upstream-lock.json")
    if spec.get("protocol") != "organizer-20261028" or spec["upstream"]["commit"] != lock["commit"]:
        raise ValueError("Organizer protocol/upstream mismatch")
    rows = []
    for case in spec["cases"]:
        for key in ("source", "scene"):
            part = case[key]
            if not part or part in (".", "..") or any(c in part for c in "/\\:"):
                raise ValueError("Scene paths must be plain directory names")
        if case["id"] != case["group"] + "/" + case["scene"]:
            raise ValueError("Invalid case identity")
        scene = Path(data)/case["source"]/case["scene"]
        if not scene.is_dir():
            raise FileNotFoundError(f"Extract the selected organizer scene first: {scene}")
        files, images = {}, []
        for path in sorted(scene.rglob("*"), key=lambda p: p.relative_to(scene).as_posix()):
            if not path.is_file():
                continue
            name = path.relative_to(scene).as_posix()
            reject_placeholder(path, allow_empty=path.suffix.lower() not in IMAGE_EXTENSIONS)
            files[name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                with Image.open(path) as image:
                    image.load()  # Detect truncated downloads, not only valid headers.
                    images.append(dict(name=name, width=image.width, height=image.height))
        if len({Path(i["name"]).name for i in images}) < 2:
            raise ValueError(f"Insufficient images for train/held-out split: {case['id']}")
        pose = "colmap_required"
        for ext in ("bin", "txt"):
            if all(f"sparse/0/{stem}.{ext}" in files for stem in ("cameras", "images")):
                for stem in ("cameras", "images"):
                    reject_placeholder(scene/f"sparse/0/{stem}.{ext}")
                pose = f"colmap.{ext}"
                break
        if pose == "colmap_required" and "transforms.json" in files:
            frames = read_json(scene/"transforms.json").get("frames")
            if not isinstance(frames, list) or not frames:
                raise ValueError(f"Empty or malformed transforms.json: {scene}")
            pose = "transforms.json"
        if pose == "colmap_required" and not any(
                (scene/n).is_dir() for n in ("images", "images_4", "images_2", "images_8")):
            raise ValueError(f"Organizer COLMAP fallback needs an images directory: {scene}")
        rows.append(dict(id=case["id"], files=files, images=images, pose_route=pose))
    checkpoint_info = checkpoint_record(checkpoint, lock)
    expected_checkpoint = spec.get("checkpoint", {}).get("sha256")
    if expected_checkpoint and checkpoint_info["sha256"] != expected_checkpoint:
        raise ValueError("Checkpoint differs from the benchmark's pinned asset snapshot")
    return dict(schema_version=1, benchmark_id=spec["benchmark_id"], protocol=spec["protocol"],
                spec_sha256=identity(spec), upstream_commit=lock["commit"], cases=rows,
                checkpoint=checkpoint_info,
                scope="Byte inventory and image decoding only; camera parsing, model loading and CUDA acceptance pending")


def verify(manifest, spec, data, checkpoint):
    current = collect(spec, data, checkpoint)
    if current != read_json(manifest):
        raise ValueError("Assets/spec changed since registration; preserve old manifest and register a new one")
    return current


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path, help="New manifest; existing evidence is never overwritten")
    group.add_argument("--verify", type=Path, help="Rehash/decode files against a previously saved manifest")
    args = p.parse_args(argv)
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    spec = read_json(args.spec)
    report = verify(args.verify, spec, args.data, args.checkpoint) if args.verify else collect(spec, args.data, args.checkpoint)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status="verified" if args.verify else "registered", cases=len(report["cases"]),
                         checkpoint_sha256=report["checkpoint"]["sha256"], gpu_accepted=False), indent=2))


if __name__ == "__main__":
    main()
