"""CPU source/camera audit using selected functions from the pinned organizer.

Does not preprocess/resize pixels, run COLMAP or create a measured split. Metadata
and image headers are inspected for integrity only, not to choose a policy.
"""
import argparse
import ast
import collections
import json
import os
from pathlib import Path
import struct

import numpy as np
from PIL import Image
from .contracts import read_json, sha256, write_json
from .run import check_upstream

ROOT = Path(__file__).resolve().parent


def organizer_cpu_functions(upstream):
    namespace = dict(np=np, collections=collections, struct=struct, os=os, json=json, Image=Image,
                     IMAGE_EXTS={".jpg", ".jpeg", ".png"})
    sources = {
        upstream/"resplat/scripts/infer_colmap.py": {
            "CameraModel", "ColmapCamera", "BaseImage", "CAMERA_MODELS", "CAMERA_MODEL_IDS", "ColmapImage",
            "qvec2rotmat", "read_next_bytes", "read_extrinsics_binary", "read_intrinsics_binary",
            "read_extrinsics_text", "read_intrinsics_text", "compute_target_shape"},
        upstream/"baseline/module_a/preprocess.py": {"load_colmap_poses", "load_transforms_json", "natural_image_list"},
    }
    for path, names in sources.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes, found = [], set()
        for node in tree.body:
            name = getattr(node, "name", None)
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
            if name in names:
                nodes.append(node); found.add(name)
        if found != names:
            raise ValueError(f"Pinned CPU interface changed: {names-found}")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, {p.relative_to(upstream).as_posix(): sha256(p) for p in sources}


def audit_case(case, data, functions):
    scene = Path(data)/case["source"]/case["scene"]
    all_images = {p.relative_to(scene).as_posix() for p in scene.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    pose = functions["load_colmap_poses"](str(scene))
    expected_names, camera_count = None, None
    if pose:
        ext = pose["source"].split(".")[1]
        reader = functions["read_intrinsics_"+("binary" if ext == "bin" else "text")]
        camera_count = len(reader(str(scene/f"sparse/0/cameras.{ext}")))
        if camera_count != 1:
            raise ValueError("Organizer source uses one camera's intrinsics; audit multiple cameras before execution")
        expected_names = set(pose["poses"])
    else:
        pose = functions["load_transforms_json"](str(scene))
        if pose:
            frames = read_json(scene/"transforms.json")["frames"]
            expected_names = {Path(f["file_path"]).name for f in frames}
            if len(expected_names) != len(frames):
                raise ValueError("Duplicate transform frame names")
    if pose:
        entries = functions["natural_image_list"](str(scene), pose)
        if {n for n, _ in entries} != expected_names:
            raise ValueError(f"Pose/image membership mismatch: {case['id']}")
        for rotation, translation in pose["poses"].values():
            if not np.isfinite(rotation).all() or not np.isfinite(translation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
                raise ValueError("Nonfinite or nonrigid camera transform")
        if not all(np.isfinite(pose[k]) and pose[k] > 0 for k in ("W", "H", "fx", "fy")):
            raise ValueError("Invalid camera intrinsics")
        route = pose["source"]
    else:
        frames_file = scene/"frames_info.txt"
        if frames_file.is_file():
            keep = []
            for line in frames_file.read_text().splitlines():
                fields = line.split()
                if line.startswith("#") or not fields: continue
                if len(fields) < 15: raise ValueError("Malformed frames_info row")
                keep.append(fields[1]+".jpg")
            if len(set(keep)) != len(keep): raise ValueError("Duplicate frames_info timestamps")
            entries = [(n, str(scene/"images"/n)) for n in sorted(keep)]
            if not all(Path(p).is_file() for _, p in entries): raise ValueError("frames_info images missing")
            route = "frames_info_to_COLMAP_required"
        else:
            entries = functions["natural_image_list"](str(scene), None)
            route = "COLMAP_required"
    if not entries: raise ValueError("No source images resolved")
    dimensions = collections.Counter()
    for _, path in entries:
        with Image.open(path) as image: dimensions[image.size] += 1
    if len(dimensions) != 1: raise ValueError("Mixed source dimensions need explicit organizer review")
    (w, h), = dimensions
    target = functions["compute_target_shape"](h, w, 768 if h > w else 960, None)
    n = len(entries); test = len(range(0, n, 8)); train = n-test
    selected = {Path(p).relative_to(scene).as_posix() for _, p in entries}
    return dict(id=case["id"], group=case["group"], source_images=len(all_images), resolved_views=n,
                excluded_images=sorted(all_images-selected), pose_route=route, camera_count=camera_count,
                source_width_height=[w, h], processed_width_height=list(target[::-1]),
                predicted_split=dict(train=train, test=test, context=min(16, train)),
                split_status="deterministic metadata prediction, not a measured prepared split" if pose else "provisional: COLMAP registration can change membership",
                camera_source_width_height=[pose["W"], pose["H"]] if pose else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    upstream = ROOT/"upstream"; lock = read_json(ROOT/"upstream-lock.json")
    check_upstream(upstream, lock["commit"])
    functions, hashes = organizer_cpu_functions(upstream)
    spec = read_json(args.spec)
    report = dict(benchmark_id=spec["benchmark_id"], upstream_commit=lock["commit"], upstream_files=hashes,
                  scope="CPU metadata/header audit; no image resize, GPU run or scored split", cases=[audit_case(c, args.data, functions) for c in spec["cases"]])
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
