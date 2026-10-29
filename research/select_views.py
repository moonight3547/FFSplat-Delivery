"""Module A (round 4): deterministic 16-view context selection.

Pipeline position: after pose preprocessing / llffhold split creation and before
ReSplat.  The organizer baseline takes 16 train views at ``np.linspace`` indices;
this module keeps that baseline as ``--policy uniform`` and adds two geometry
policies that only use the COLMAP poses and the frozen train/test split:

  uniform   organizer baseline: 16 evenly spaced indices in sorted train order.
  fps       farthest-point sampling on camera centres, normalised by the median
            pairwise camera distance and started from the camera closest to the
            train centroid.
  coverage  greedy maximisation of weighted frustum coverage of a voxel grid
            over a scene proxy built from the camera cloud; each voxel is worth
            1/sqrt(number of candidate views that can see it), so views that
            reveal rare parts of the proxy are preferred.  Only the central
            50% x 50% of each image counts, so one wide-FOV view cannot claim
            the whole proxy and leave the remaining slots redundant.

Only ``split.json`` and ``sparse/0/{cameras,images}.txt`` are read.  The selector
is pure numpy / CPU, deterministic (all ties break on the lower train index),
and never writes into the prepared scene directory.  Context names are emitted
in sorted train order, exactly like the organizer baseline, so ReSplat's pivot
view (``context[8]``) keeps the same convention for every policy; the greedy
order is recorded separately in ``scores.selection_order``.

COLMAP convention: ``images.txt`` stores world-to-camera poses,
``X_cam = R(q) X_world + t``, hence ``camera_center = -R(q)^T t`` and
``R_c2w = R(q)^T``.  The synthetic ``--self-test`` round-trips this through a
generated cameras/images pair before testing selection determinism.

Output: ``<out>/context16.json`` with exactly
``{"policy": ..., "context": [16 names], "scores": {...}}``.

Usage (CPU, reproducible on the control machine):

  python -m research.select_views --work work/<case> --out /tmp/sel --policy fps
  python -m research.select_views --work work/<case> --out /tmp/sel --policy coverage --dry-run
  python -m research.select_views --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONTEXT_SIZE = 16
# Fixed grid axis on the longest side of the proxy box; other sides are shorter.
COVERAGE_GRID = 32
# Proxy box = camera-centre bbox inflated by 0.5 * scene_scale on every axis.
COVERAGE_MARGIN = 0.5
# Fraction of image width/height counted as reliable view coverage (0.5 -> 25%
# of the image area).  The periphery is lower-resolution and more distorted.
COVERAGE_CENTRAL_FRACTION = 0.5
POLICIES = ("uniform", "fps", "coverage")
CAMERA_MODELS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "FULL_OPENCV": 12,
}


# ---------------------------------------------------------------------------
# COLMAP text parsing (no pycolmap)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class View:
    name: str
    camera: Camera
    R_w2c: np.ndarray
    t_w2c: np.ndarray
    center: np.ndarray
    R_c2w: np.ndarray


def qvec2rotmat(qvec):
    """COLMAP world-to-camera quaternion (w, x, y, z) -> rotation matrix."""
    q = np.asarray(qvec, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Quaternion must have 4 values, got {q.shape}")
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Degenerate COLMAP quaternion")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rotmat2qvec(rotation):
    """Inverse of :func:`qvec2rotmat`; used by the synthetic round-trip test."""
    R = np.asarray(rotation, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"Rotation must be 3x3, got {R.shape}")
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s,
             (R[2, 1] - R[1, 2]) / s,
             (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s,
             0.25 * s,
             (R[0, 1] + R[1, 0]) / s,
             (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s,
             (R[0, 1] + R[1, 0]) / s,
             0.25 * s,
             (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s,
             (R[0, 2] + R[2, 0]) / s,
             (R[1, 2] + R[2, 1]) / s,
             0.25 * s]
    qvec = np.asarray(q, dtype=np.float64)
    qvec /= np.linalg.norm(qvec)
    return -qvec if qvec[0] < 0 else qvec


def _pinhole(model, params):
    if model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = params[:4]
    else:  # SIMPLE_RADIAL / RADIAL
        fx = fy = params[0]
        cx, cy = params[1:3]
    return float(fx), float(fy), float(cx), float(cy)


def read_cameras(path):
    """Parse a COLMAP text ``cameras.txt`` into {camera_id: Camera}."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing COLMAP cameras file: {path}")
    cameras = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"{path}:{lineno}: expected CAMERA_ID MODEL W H PARAMS")
        model = parts[1]
        if model not in CAMERA_MODELS:
            raise ValueError(f"{path}:{lineno}: unsupported camera model {model!r}; "
                             f"known: {', '.join(sorted(CAMERA_MODELS))}")
        expected = CAMERA_MODELS[model]
        if len(parts) - 4 != expected:
            raise ValueError(f"{path}:{lineno}: {model} needs {expected} params, "
                             f"got {len(parts) - 4}")
        try:
            camera_id = int(parts[0])
            width, height = int(parts[2]), int(parts[3])
            params = np.asarray([float(x) for x in parts[4:]], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: non-numeric camera row") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"{path}:{lineno}: invalid camera size {width}x{height}")
        if not np.isfinite(params).all():
            raise ValueError(f"{path}:{lineno}: non-finite camera parameters")
        fx, fy, cx, cy = _pinhole(model, params)
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError(f"{path}:{lineno}: non-positive focal length")
        if camera_id in cameras:
            raise ValueError(f"{path}:{lineno}: duplicate camera id {camera_id}")
        cameras[camera_id] = Camera(camera_id, model, width, height, fx, fy, cx, cy)
    if not cameras:
        raise ValueError(f"No cameras in {path}")
    return cameras


def read_images(path, cameras):
    """Parse COLMAP text ``images.txt``.

    COLMAP writes one 2D-points line after every image header (the line may be
    empty).  We consume that line unconditionally, but reject the common broken
    form where the points line is absent and the next header gets swallowed.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing COLMAP images file: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    views = []
    seen_ids = set()
    seen_names = set()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) != 10:
            raise ValueError(f"{path}:{i + 1}: expected 10 image-header fields, "
                             f"got {len(parts)}")
        try:
            image_id = int(parts[0])
            camera_id = int(parts[8])
            qvec = np.asarray([float(x) for x in parts[1:5]], dtype=np.float64)
            tvec = np.asarray([float(x) for x in parts[5:8]], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"{path}:{i + 1}: non-numeric image header") from exc
        name = parts[9]
        if camera_id not in cameras:
            raise ValueError(f"{path}:{i + 1}: image {name!r} references missing "
                             f"camera id {camera_id}")
        if image_id in seen_ids:
            raise ValueError(f"{path}:{i + 1}: duplicate image id {image_id}")
        if name in seen_names:
            raise ValueError(f"{path}:{i + 1}: duplicate image name {name!r}")
        if not np.isfinite(qvec).all() or not np.isfinite(tvec).all():
            raise ValueError(f"{path}:{i + 1}: non-finite pose for {name!r}")
        seen_ids.add(image_id)
        seen_names.add(name)

        R_w2c = qvec2rotmat(qvec)
        center = -R_w2c.T @ tvec
        R_c2w = R_w2c.T
        views.append(View(name, cameras[camera_id], R_w2c, tvec, center, R_c2w))

        i += 1
        if i >= len(lines):
            raise ValueError(f"{path}: image {name!r} has no 2D-points line")
        points = lines[i].strip()
        if points and not points.startswith("#"):
            point_parts = points.split()
            if len(point_parts) == 10 and point_parts[0].lstrip("+-").isdigit():
                raise ValueError(f"{path}:{i + 1}: expected a 2D-points line for "
                                 f"{name!r}, found another image header")
        i += 1
    if not views:
        raise ValueError(f"No images in {path}")
    return views


def load_split(path):
    """Read and validate the organizer split (images/train/test/resolution)."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing split file: {path}")
    try:
        split = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(split, dict):
        raise ValueError(f"{path}: split.json must contain an object")
    for key in ("images", "train", "test"):
        value = split.get(key)
        if not isinstance(value, list) or not all(isinstance(n, str) for n in value):
            raise ValueError(f"{path}: {key!r} must be a list of image names")
        if len(value) != len(set(value)):
            raise ValueError(f"{path}: {key!r} contains duplicates")
    images = split["images"]
    if images != sorted(images):
        raise ValueError(f"{path}: 'images' must be sorted")
    if any(Path(name).name != name or name in (".", "..") for name in images):
        raise ValueError(f"{path}: image names must be plain filenames")
    train, test = set(split["train"]), set(split["test"])
    if not train <= set(images) or not test <= set(images):
        raise ValueError(f"{path}: train/test contains an image outside 'images'")
    if train & test:
        raise ValueError(f"{path}: train and test overlap")
    if train | test != set(images):
        raise ValueError(f"{path}: train/test do not partition 'images'")
    if len(train) < CONTEXT_SIZE:
        raise ValueError(f"{path}: need at least {CONTEXT_SIZE} train views, "
                         f"found {len(train)}")
    resolution = split.get("resolution")
    if resolution is not None:
        if (not isinstance(resolution, list) or len(resolution) != 2
                or any(type(x) is not int or x <= 0 for x in resolution)):
            raise ValueError(f"{path}: 'resolution' must be [height, width]")
    return split


def load_scene(work):
    """Read the split and all poses of a prepared scene work dir."""
    work = Path(work)
    if not work.is_dir():
        raise ValueError(f"Prepared scene work dir not found: {work}")
    split = load_split(work / "split.json")
    cameras = read_cameras(work / "sparse/0/cameras.txt")
    views = read_images(work / "sparse/0/images.txt", cameras)
    by_name = {view.name: view for view in views}
    missing = [name for name in split["images"] if name not in by_name]
    extra = sorted(set(by_name) - set(split["images"]))
    if missing:
        raise ValueError(f"images.txt is missing {len(missing)} split image(s), "
                         f"first: {missing[0]!r}")
    if extra:
        raise ValueError(f"images.txt contains {len(extra)} image(s) outside the "
                         f"split, first: {extra[0]!r}")
    return split, by_name


# ---------------------------------------------------------------------------
# Selection policies
# ---------------------------------------------------------------------------


def _pairwise(centers):
    diff = centers[:, None, :] - centers[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _scene_scale(distance):
    positive = distance[distance > 1e-12]
    if positive.size == 0:
        raise ValueError("All camera centres coincide; cannot define a scene scale")
    scale = float(np.median(positive))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Non-finite or non-positive scene scale")
    return scale


def _uniform_order(n_train, k=CONTEXT_SIZE):
    """Exactly the organizer baseline: linspace over sorted train indices."""
    indices = np.linspace(0, n_train - 1, k, dtype=int)
    unique = sorted({int(i) for i in indices})
    if len(unique) != k:
        raise ValueError(f"linspace baseline produced {len(unique)} views, expected {k}")
    return unique


def _fps_order(centers, distance, scale, k=CONTEXT_SIZE):
    """Farthest-point sampling in centroid-normalised camera-centre space."""
    centroid = centers.mean(axis=0)
    start = int(np.argmin(np.linalg.norm(centers - centroid, axis=1)))
    chosen = [start]
    available = np.ones(len(centers), dtype=bool)
    available[start] = False
    min_distance = distance[start] / scale
    scores = [float(np.linalg.norm(centers[start] - centroid) / scale)]
    for _ in range(k - 1):
        candidate = int(np.argmax(np.where(available, min_distance, -np.inf)))
        chosen.append(candidate)
        scores.append(float(min_distance[candidate]))
        available[candidate] = False
        min_distance = np.minimum(min_distance, distance[candidate] / scale)
    return chosen, scores, start


def _coverage_order(views, centers, scale, k=CONTEXT_SIZE):
    """Greedy weighted frustum coverage over a voxelised scene proxy.

    The proxy box is the camera-centre bounding box, inflated by 0.5 * scene
    scale per axis and shifted by half a scale along the mean optical axis so
    forward-facing captures include the region in front of the rig.  A voxel is
    covered by a view when its centre projects inside the central 50% x 50% of
    that view's pinhole image (distortion ignored); the crop keeps a wide-FOV
    view from claiming the whole proxy.  Voxel weight is
    1/sqrt(#candidate views seeing it), so a candidate's marginal gain rewards
    views that expose rare geometry instead of re-covering the same volume.
    """
    if len(views) < k:
        raise ValueError(f"Need {k} candidates, got {len(views)}")
    lo = centers.min(axis=0)
    hi = centers.max(axis=0)
    mean_forward = np.mean([view.R_c2w[:, 2] for view in views], axis=0)
    offset = COVERAGE_MARGIN * scale * mean_forward
    centre = 0.5 * (lo + hi) + offset
    half = 0.5 * (hi - lo) + COVERAGE_MARGIN * scale
    lo = centre - half
    side = 2.0 * half
    longest = float(side.max())
    if longest <= 0.0:
        raise ValueError("Degenerate scene proxy box")
    cell = longest / COVERAGE_GRID
    dims = np.clip(np.ceil(side / cell).astype(int), 1, COVERAGE_GRID)
    axes = [lo[axis] + (np.arange(dims[axis]) + 0.5) * cell for axis in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    visible = []
    for view in views:
        # Row-vector form of R_c2w^T (X - C): (X - C) @ R_c2w.
        cam = (grid - view.center) @ view.R_c2w
        z = cam[:, 2]
        in_front = z > 1e-9
        safe_z = np.where(in_front, z, 1.0)
        u = view.camera.fx * cam[:, 0] / safe_z + view.camera.cx
        v = view.camera.fy * cam[:, 1] / safe_z + view.camera.cy
        half_width = 0.5 * COVERAGE_CENTRAL_FRACTION * view.camera.width
        half_height = 0.5 * COVERAGE_CENTRAL_FRACTION * view.camera.height
        inside = (in_front
                  & (np.abs(u - view.camera.cx) <= half_width)
                  & (np.abs(v - view.camera.cy) <= half_height))
        visible.append(np.flatnonzero(inside))

    frequency = np.zeros(len(grid), dtype=np.int64)
    for indices in visible:
        frequency[indices] += 1
    seen = frequency > 0
    if not seen.any():
        raise ValueError("No proxy voxel is visible from any candidate view")
    weights = np.zeros(len(grid), dtype=np.float64)
    weights[seen] = 1.0 / np.sqrt(frequency[seen])

    covered = np.zeros(len(grid), dtype=bool)
    available = np.ones(len(views), dtype=bool)
    order, scores, new_voxels, visible_voxels = [], [], [], []
    for _ in range(k):
        gains = np.full(len(views), -np.inf)
        for candidate in np.flatnonzero(available):
            indices = visible[candidate]
            gains[candidate] = weights[indices[~covered[indices]]].sum()
        pick = int(np.argmax(gains))
        if not math.isfinite(float(gains[pick])):
            raise ValueError("Greedy coverage failed to find an available view")
        indices = visible[pick]
        newly = indices[~covered[indices]]
        order.append(pick)
        scores.append(float(weights[newly].sum()))
        new_voxels.append(int(newly.size))
        visible_voxels.append(int(indices.size))
        covered[newly] = True
        available[pick] = False

    total_weight = float(weights.sum())
    coverage = {
        "grid_dims": [int(x) for x in dims],
        "cell_size": _num(cell),
        "bounds": [[_num(x) for x in lo], [_num(x) for x in (lo + side)]],
        "proxy_margin_scale": COVERAGE_MARGIN,
        "central_crop_fraction": COVERAGE_CENTRAL_FRACTION,
        "voxels": int(len(grid)),
        "visible_voxels": int(seen.sum()),
        "covered_voxels": int(covered.sum()),
        "unweighted_fraction": _num(covered.sum() / max(1, seen.sum())),
        "weighted_fraction": _num(weights[covered].sum() / total_weight),
        "voxel_weight": "1/sqrt(number of candidate views whose frustum contains the voxel)",
    }
    return (order, scores, new_voxels, visible_voxels, coverage,
            float(weights[covered].sum()))


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------


def _num(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite score: {value!r}")
    return 0.0 if value == 0.0 else round(value, 12)


def select(work, policy):
    """Compute the deterministic selection record for one policy."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; choose from {', '.join(POLICIES)}")
    split, by_name = load_scene(work)
    train = sorted(split["train"])
    views = [by_name[name] for name in train]
    centers = np.stack([view.center for view in views])
    distance = _pairwise(centers)
    scale = _scene_scale(distance)

    coverage = None
    new_voxels = visible_voxels = None
    if policy == "uniform":
        order = _uniform_order(len(train))
        scores = [None] * len(order)
        start = None
        objective = {"name": "organizer_linspace_baseline", "value": None}
    elif policy == "fps":
        order, scores, start = _fps_order(centers, distance, scale)
        objective = {"name": "maximin_camera_center_distance_normalized", "value": None}
    else:
        (order, scores, new_voxels, visible_voxels, coverage,
         coverage_total) = _coverage_order(views, centers, scale)
        start = None
        objective = {"name": "weighted_frustum_coverage", "value": _num(coverage_total)}

    selected = sorted(order)
    if len(selected) != CONTEXT_SIZE or len(set(selected)) != CONTEXT_SIZE:
        raise ValueError("Selection did not produce 16 unique train views")
    if not set(selected) <= set(range(len(train))):
        raise ValueError("Selection contains a non-train index")
    context = [train[index] for index in selected]

    subset = distance[np.ix_(selected, selected)] / scale
    np.fill_diagonal(subset, np.inf)
    nearest = subset.min(axis=1)
    min_separation = float(nearest.min())
    mean_nearest = float(nearest.mean())
    if policy == "fps":
        objective["value"] = _num(min_separation)

    step_of = {index: step for step, index in enumerate(order)}
    score_of = {index: scores[step] for step, index in enumerate(order)}
    extra_of = {}
    if coverage is not None:
        for step, index in enumerate(order):
            extra_of[index] = {"new_voxels": new_voxels[step],
                               "visible_voxels": visible_voxels[step]}
    nearest_of = {index: float(value) for index, value in zip(selected, nearest)}

    view_rows = []
    for index, name in enumerate(train):
        if index not in step_of:
            continue
        row = {
            "name": name,
            "train_index": index,
            "selection_step": step_of[index],
            "score": None if score_of[index] is None else _num(score_of[index]),
            "min_separation": _num(nearest_of[index]),
        }
        row.update(extra_of.get(index, {}))
        view_rows.append(row)

    parameters = {
        "uniform": {"rule": "np.linspace(0, len(train)-1, 16, dtype=int)"},
        "fps": {
            "start": "camera centre closest to the train centroid",
            "normalization": "median pairwise camera-centre distance",
            "tie_break": "lowest train index",
        },
        "coverage": {
            "grid_axis": COVERAGE_GRID,
            "projection": "pinhole, central 50%x50% crop (distortion ignored)",
            "tie_break": "lowest train index",
        },
    }[policy]

    record = {
        "policy": policy,
        "context": context,
        "scores": {
            "policy": policy,
            "num_images": len(split["images"]),
            "num_train": len(train),
            "num_test": len(split["test"]),
            "context_size": CONTEXT_SIZE,
            "scene_scale": _num(scale),
            "objective": objective,
            "min_separation": _num(min_separation),
            "mean_nearest_separation": _num(mean_nearest),
            "selection_order": [train[index] for index in order],
            "selection_indices": [int(index) for index in order],
            "parameters": parameters,
            "views": view_rows,
        },
    }
    if policy == "fps":
        record["scores"]["start_view"] = train[start]
    if coverage is not None:
        record["scores"]["coverage"] = coverage
    return record


def _summary(record, out_path=None):
    scores = record["scores"]
    extra = ""
    if record["policy"] == "coverage":
        extra = f" coverage={scores['coverage']['weighted_fraction']:.4f}"
    elif record["policy"] == "fps":
        extra = f" start={scores['start_view']}"
    target = str(out_path) if out_path is not None else "(dry-run: not written)"
    return (f"[select_views] policy={record['policy']} "
            f"train={scores['num_train']}/{scores['num_images']} "
            f"selected={len(record['context'])} scale={scores['scene_scale']:.4f} "
            f"min_sep={scores['min_separation']:.4f}"
            f"{extra} -> {target}")


# ---------------------------------------------------------------------------
# Output / CLI
# ---------------------------------------------------------------------------


def _ensure_outside_work(work, out):
    work = Path(work).resolve()
    out = Path(out).resolve()
    if out == work or work in out.parents:
        raise ValueError(f"--out must be outside the prepared scene dir: {out}")


def write_record(out_dir, record):
    """Write exactly ``<out>/context16.json`` atomically."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "context16.json"
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


# ---------------------------------------------------------------------------
# Self-test: synthetic COLMAP writer + convention/determinism checks
# ---------------------------------------------------------------------------


def _look_at(center, target, up=(0.0, 0.0, 1.0)):
    """Camera-to-world rotation with +Z forward and +Y down (OpenCV/COLMAP)."""
    z = np.asarray(target, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def _write_synthetic_scene(root, count=40):
    root = Path(root)
    (root / "sparse/0").mkdir(parents=True, exist_ok=True)
    names = [f"{index:03d}.png" for index in range(count)]
    poses = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        center = np.array([5.0 * math.cos(angle), 5.0 * math.sin(angle),
                           0.6 * math.sin(2.0 * angle)])
        R_c2w = _look_at(center, np.array([0.0, 0.0, 0.2]))
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ center
        poses.append((R_w2c, t_w2c, center, R_c2w))

    cameras_path = root / "sparse/0/cameras.txt"
    cameras_path.write_text(
        "# synthetic pinhole\n1 PINHOLE 640 480 520 510 320 240\n", encoding="utf-8")
    rows = []
    for index, (R_w2c, t_w2c, _, _) in enumerate(poses):
        q = rotmat2qvec(R_w2c)
        # Exercise both blank and populated 2D-points lines.
        points = "" if index % 3 else "10 20 0 30 40 0"
        rows.append(
            f"{index + 1} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g} "
            f"{t_w2c[0]:.12g} {t_w2c[1]:.12g} {t_w2c[2]:.12g} 1 {names[index]}\n"
            f"{points}\n")
    (root / "sparse/0/images.txt").write_text("".join(rows), encoding="utf-8")

    test = names[::8]
    train = [name for index, name in enumerate(names) if index % 8]
    context = [train[index] for index in _uniform_order(len(train))]
    split = {
        "images": names,
        "train": train,
        "test": test,
        "context": context,
        "llffhold": 8,
        "resolution": [480, 640],
    }
    (root / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return names, poses


def _self_test():
    checks = 0
    try:
        with tempfile.TemporaryDirectory(prefix="select_views_selftest_", dir="/tmp") as tmp:
            scene = Path(tmp) / "scene"
            names, poses = _write_synthetic_scene(scene)
            split = load_split(scene / "split.json")
            cameras = read_cameras(scene / "sparse/0/cameras.txt")
            views = read_images(scene / "sparse/0/images.txt", cameras)
            if len(views) != len(names):
                raise AssertionError("parser dropped views")
            if set(view.name for view in views) != set(names):
                raise AssertionError("parsed names differ")
            if len(cameras) != 1:
                raise AssertionError("camera count")

            center_error = rotation_error = quaternion_error = 0.0
            world = np.array([0.31, -0.27, 0.53])
            for view, (R_w2c, t_w2c, center, R_c2w) in zip(views, poses):
                center_error = max(center_error, float(np.linalg.norm(view.center - center)))
                rotation_error = max(rotation_error,
                                     float(np.abs(view.R_c2w - R_c2w).max()))
                quaternion_error = max(
                    quaternion_error,
                    float(np.abs(qvec2rotmat(rotmat2qvec(R_w2c)) - R_w2c).max()))
                cam_point = R_w2c @ world + t_w2c
                recovered = view.R_c2w @ cam_point + view.center
                if not np.allclose(recovered, world, atol=1e-9):
                    raise AssertionError("camera centre/rotation round-trip failed")
                if not np.allclose(view.R_w2c, R_w2c, atol=1e-9):
                    raise AssertionError("world-to-camera rotation mismatch")
                if not np.allclose(view.R_c2w @ view.R_w2c, np.eye(3), atol=1e-9):
                    raise AssertionError("c2w/w2c inverse mismatch")
            if center_error > 1e-9 or rotation_error > 1e-9 or quaternion_error > 1e-9:
                raise AssertionError("COLMAP round-trip error too large")
            checks += 1
            print(f"[self-test] PASS colmap round-trip "
                  f"(center {center_error:.2e}, rotation {rotation_error:.2e}, "
                  f"quat {quaternion_error:.2e})")

            records = {}
            for policy in POLICIES:
                first = select(scene, policy)
                second = select(scene, policy)
                if json.dumps(first, sort_keys=True) != json.dumps(second, sort_keys=True):
                    raise AssertionError(f"{policy} is not deterministic")
                context = first["context"]
                if len(context) != 16 or len(set(context)) != 16:
                    raise AssertionError(f"{policy} did not select 16 unique views")
                if not set(context) <= set(split["train"]):
                    raise AssertionError(f"{policy} selected a non-train view")
                if set(context) & set(split["test"]):
                    raise AssertionError(f"{policy} selected a held-out view")
                if context != sorted(context) or context != [n for n in split["train"]
                                                              if n in set(context)]:
                    raise AssertionError(f"{policy} context is not in train order")
                records[policy] = first
                checks += 1
                print(f"[self-test] PASS {policy} determinism + 16 unique train views")

            expected = [split["train"][index] for index in _uniform_order(len(split["train"]))]
            if records["uniform"]["context"] != expected:
                raise AssertionError("uniform policy differs from organizer baseline")
            checks += 1
            print("[self-test] PASS uniform matches organizer linspace16 baseline")

            coverage = records["coverage"]["scores"]["coverage"]
            if not (0.0 <= coverage["unweighted_fraction"] <= 1.0
                    and 0.0 <= coverage["weighted_fraction"] <= 1.0):
                raise AssertionError("coverage fractions out of range")
            if max(coverage["grid_dims"]) > COVERAGE_GRID:
                raise AssertionError("coverage grid exceeds cap")
            checks += 1
            print(f"[self-test] PASS coverage grid {coverage['grid_dims']} "
                  f"fraction={coverage['unweighted_fraction']:.3f}")

            before = {path.relative_to(scene): path.read_bytes()
                      for path in scene.rglob("*") if path.is_file()}
            out = Path(tmp) / "out"
            written = write_record(out, records["coverage"])
            if json.loads(written.read_text(encoding="utf-8")) != records["coverage"]:
                raise AssertionError("written record differs")
            after = {path.relative_to(scene): path.read_bytes()
                     for path in scene.rglob("*") if path.is_file()}
            if before != after:
                raise AssertionError("selector changed the prepared scene dir")
            for bad in (scene, scene / "nested"):
                try:
                    _ensure_outside_work(scene, bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError("writer accepted a path inside the scene dir")
            checks += 1
            print("[self-test] PASS writer only emits <out>/context16.json; scene untouched")

        print(f"[self-test] PASS all {checks} checks")
        return 0
    except Exception as exc:  # noqa: BLE001 - self-test must report FAIL, not crash
        print(f"[self-test] FAIL: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", help="prepared scene work dir (split.json + sparse/0)")
    parser.add_argument("--out", help="output dir; writes <out>/context16.json")
    parser.add_argument("--policy", choices=POLICIES, default="uniform")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and score on CPU without writing any file")
    parser.add_argument("--self-test", action="store_true",
                        help="synthetic COLMAP round-trip + determinism checks")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.work:
        parser.error("--work is required unless --self-test is used")
    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run is used")

    try:
        if args.out:
            _ensure_outside_work(args.work, args.out)
        record = select(args.work, args.policy)
        out_path = None
        if not args.dry_run:
            out_path = write_record(args.out, record)
        print(_summary(record, out_path))
        if args.dry_run:
            print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError) as exc:
        print(f"select_views: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
