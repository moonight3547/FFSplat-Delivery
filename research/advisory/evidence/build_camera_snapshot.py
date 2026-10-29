"""Rebuild the diagnostic snapshot from local assets and immutable metadata exports.

Run from the repository root. Reads camera metadata only; never loads image RGB.
Prepared split evidence takes precedence. Missing splits are explicitly reconstructed.
"""
from __future__ import annotations

import hashlib
import json
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from research.select_views import load_scene, qvec2rotmat, _pinhole


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "research/advisory/evidence"
DATA = ROOT.parent / "datasets"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_colmap_binary(directory):
    """Read poses/intrinsics, seeking past feature observations without loading them."""
    cameras = {}
    model_names = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4),
                   2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
                   4: ("OPENCV", 8), 6: ("FULL_OPENCV", 12)}
    with (directory / "cameras.bin").open("rb") as stream:
        count, = struct.unpack("<Q", stream.read(8))
        for _ in range(count):
            cid, model, w, h = struct.unpack("<iiQQ", stream.read(24))
            name, n = model_names[model]
            params = struct.unpack("<" + "d" * n, stream.read(8 * n))
            fx, fy, cx, cy = _pinhole(name, params)
            cameras[cid] = dict(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
    rows = []
    with (directory / "images.bin").open("rb") as stream:
        count, = struct.unpack("<Q", stream.read(8))
        for _ in range(count):
            record = struct.unpack("<idddddddi", stream.read(64))
            name = bytearray()
            while True:
                char = stream.read(1)
                if not char:
                    raise ValueError("Truncated COLMAP image name")
                if char == b"\0":
                    break
                name.extend(char)
            nobs, = struct.unpack("<Q", stream.read(8))
            stream.seek(24 * nobs, 1)
            rotation = qvec2rotmat(record[1:5]).T
            center = -rotation @ np.array(record[5:8])
            rows.append(dict(name=name.decode(), center=center.tolist(),
                             R_c2w=rotation.tolist(), **cameras[record[8]]))
    return rows


def reconstructed_split(views):
    names = sorted(v["name"] for v in views)
    train = [n for i, n in enumerate(names) if i % 8 != 0]
    return dict(images=names, train=train,
                test=[n for i, n in enumerate(names) if i % 8 == 0],
                context=[train[i] for i in np.linspace(0, len(train)-1, 16, dtype=int)],
                llffhold=8)


def prepared(export, case):
    candidates = [p for p in export["files"] if p.endswith(case + "/work/split.json")
                  or p.endswith("/work/" + case + "/split.json")]
    if not candidates:
        return None
    prefix = candidates[0][:-len("split.json")]
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        for name in ["split.json", "sparse/0/cameras.txt", "sparse/0/images.txt"]:
            dest = work / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(export["files"][prefix + name]["text"])
        split, by_name = load_scene(work)
        views = [dict(name=v.name, center=v.center.tolist(), R_c2w=v.R_c2w.tolist(),
                      **{k: getattr(v.camera, k) for k in
                         ["width", "height", "fx", "fy", "cx", "cy"]})
                 for v in by_name.values()]
    return split, views, dict(kind="prepared_COLMAP_text", node=export["node"],
                             path=prefix, split_status="frozen",
                             files={n: export["files"][prefix+n]["sha256"] for n in
                                    ["split.json", "sparse/0/cameras.txt", "sparse/0/images.txt"]})


def main():
    h1 = json.loads((EVIDENCE / "20261030-camera-h1-metadata.json").read_text())
    n1 = json.loads((EVIDENCE / "20261030-camera-n1-metadata.json").read_text())
    scores = {}
    metric_archive = "20261030-0640-h1.json"
    for path, entry in json.loads((EVIDENCE / metric_archive).read_text())["files"].items():
        if path.endswith("b_summary.json"):
            data = entry["data"]
            result = data["results"][0]
            scores[data["case"]] = dict(lpips=result["lpips"], psnr=result["psnr"],
                                       ssim=result["ssim"], source=f"{metric_archive}::{path}",
                                       recipe="H1 eval13 5K candidate; historical snapshot")
    archive = "20261030-0648-evalnode-rounds.json"
    for path, entry in json.loads((EVIDENCE / archive).read_text())["files"].items():
        data = entry.get("data", {})
        if path.endswith("mipnerf360-bicycle/summary.json") and "round-new/" in path:
            result = data["results"][0]
            scores[data["case"]] = dict(lpips=result["lpips"], psnr=result["psnr"],
                                       ssim=result["ssim"], source=f"{archive}::{path}",
                                       recipe="N1 round-new 5K candidate; different run/node")
    rows = []
    for spec in json.loads((ROOT / "benchmarks/organizer-provided13-v1.json").read_text())["cases"]:
        case = spec["id"]
        pre = prepared(n1, case) or prepared(h1, case)
        if pre:
            split, views, source = pre
        elif spec["group"] == "dl3dv":
            path = DATA / spec["source"] / spec["scene"] / "transforms.json"
            data = json.loads(path.read_text())
            views = []
            for frame in data["frames"]:
                matrix = np.array(frame["transform_matrix"], dtype=float)
                matrix[:3, 1:3] *= -1  # OpenGL camera to OpenCV camera convention.
                name = Path(frame["file_path"]).name
                views.append(dict(name=name, center=matrix[:3, 3].tolist(),
                                  R_c2w=matrix[:3, :3].tolist(), width=data["w"], height=data["h"],
                                  fx=data["fl_x"], fy=data["fl_y"], cx=data["cx"], cy=data["cy"]))
            split = reconstructed_split(views)
            source = dict(kind="source_transforms_OpenGL_to_OpenCV", path=str(path),
                          sha256=digest(path), split_status="reconstructed",
                          note="All metadata frames; llffhold=8 and uniform16 reconstructed. No guarantee corrupted/unavailable RGB frames survived the original preparation.")
        elif spec["group"] == "hw":
            record = next(x for x in h1["binary_scenes"] if x["case"] == case)
            views = []
            for image in record["images"]:
                cam = record["cameras"][str(image["camera_id"])]
                assert cam["model"] == 1, cam
                fx, fy, cx, cy = cam["params"]
                rotation = qvec2rotmat(image["qvec"]).T
                center = -rotation @ np.array(image["tvec"])
                views.append(dict(name=image["name"], center=center.tolist(), R_c2w=rotation.tolist(),
                                  width=cam["width"], height=cam["height"], fx=fx, fy=fy, cx=cx, cy=cy))
            split = reconstructed_split(views)
            source = dict(kind="existing_undistorted_COLMAP_binary", node="H1", path=record["path"],
                          files=record["files"], split_status="reconstructed",
                          note="Original prepared split was cleaned. Reconstructed sorted registered-image llffhold8/uniform16 is diagnostic, not original context proof.")
        else:
            path = DATA / spec["source"] / spec["scene"] / "sparse/0"
            views = read_colmap_binary(path)
            split = reconstructed_split(views)
            source = dict(kind="source_COLMAP_binary", path=str(path),
                          files={n: digest(path/n) for n in ["cameras.bin", "images.bin"]},
                          split_status="reconstructed",
                          note="Source registered cameras; llffhold8/uniform16 reconstructed, not original context proof.")
        by_name = {v["name"]: v for v in views}
        assert len(by_name) == len(views)
        assert set(split["images"]) == set(by_name)
        rows.append(dict(case=case, source=source, split=split,
                         views=[by_name[n] for n in split["images"]],
                         metrics=scores.get(case, dict(lpips=None, source="No completed candidate metric in the captured archives", recipe="unknown"))))
    output = dict(schema_version=1, captured_at=datetime.now(timezone.utc).isoformat(),
                  purpose="camera-only diagnostic; no RGB, no training, no test-guided selection",
                  notes="Mixed historical result archives identify zero-score cases only; not a comparable 13-case leaderboard. Filename order is a capture-order proxy, not verified timestamps.",
                  cases=rows)
    path = EVIDENCE / "20261030-camera-snapshot.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(path)
    for row in rows:
        print(row["case"][:24], len(row["views"]), row["source"]["split_status"], row["metrics"]["lpips"])


if __name__ == "__main__":
    main()
