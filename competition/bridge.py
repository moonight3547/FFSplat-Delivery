"""Organizer world NPZ / PINHOLE cameras / classic PLY conversion, no model edits."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .contracts import read_json, validate_split


def read_training_views(work):
    work = Path(work)
    split = validate_split(read_json(work / "split.json"))
    cameras = [s.split() for s in (work / "sparse/0/cameras.txt").read_text().splitlines()
               if s.strip() and not s.startswith("#")]
    if len(cameras) != 1 or cameras[0][1] != "PINHOLE":
        raise ValueError("Expected organizer single PINHOLE camera")
    _, _, w, h, fx, fy, cx, cy = cameras[0]
    w, h = int(w), int(h)
    if [h, w] != split["resolution"]:
        raise ValueError("Camera/split dimensions differ")
    K = np.array([[float(fx), 0, float(cx)], [0, float(fy), float(cy)], [0, 0, 1]], dtype=np.float32)
    # Keep blank correspondence lines: each non-comment image header has one following line.
    lines = iter((work / "sparse/0/images.txt").read_text().splitlines())
    poses = {}
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 10 or parts[8] != cameras[0][0]:
            raise ValueError("Unexpected organizer image header")
        q = np.array(list(map(float, parts[1:5])))
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        pose[:3, 3] = list(map(float, parts[5:8]))
        poses[parts[9]] = np.linalg.inv(pose).astype(np.float32)
        next(lines, None)
    if set(poses) != set(split["images"]):
        raise ValueError("Camera names differ from frozen split")
    return split, [dict(name=n, path=str(work/"images"/n), c2w=poses[n], K=K.copy(),
                       height=h, width=w) for n in split["train"]]


def load_world(path):
    with np.load(path, allow_pickle=False) as data:
        result = {k: np.asarray(data[k], dtype=np.float32) for k in
                  ("means", "scales", "rotations_xyzw", "harmonics", "opacities")}
    n = len(result["means"])
    shapes = dict(means=(n, 3), scales=(n, 3), rotations_xyzw=(n, 4), harmonics=(n, 3, 16), opacities=(n,))
    if n == 0 or any(result[k].shape != v for k, v in shapes.items()):
        raise ValueError("Expected nonempty organizer SH3 world Gaussians")
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError("Nonfinite world Gaussians")
    if np.any(result["scales"] <= 0) or np.any((result["opacities"] < 0) | (result["opacities"] > 1)):
        raise ValueError("Invalid scales or opacity")
    if not np.allclose(np.linalg.norm(result["rotations_xyzw"], axis=1), 1, atol=1e-4):
        raise ValueError("Nonunit rotations")
    return result


def parameter_arrays(world, center, scale):
    """No second pivot/SH rotation: organizer export is already world-space."""
    opacity = world["opacities"].clip(1e-6, 1-1e-6)
    sh = world["harmonics"].transpose(0, 2, 1)
    return [np.asarray((world["means"]-center)*scale, dtype=np.float32),
            np.log(world["scales"]*scale), world["rotations_xyzw"][:, [3, 0, 1, 2]],
            sh[:, 0].copy(), sh[:, 1:].copy(), np.log(opacity/(1-opacity))]


def export_ply(path, arrays, center, scale):
    """Classic channel-major SH and wxyz quaternion layout consumed by FastGS."""
    from plyfile import PlyData, PlyElement
    means, logs, quat, dc, rest, opacity = arrays
    n = len(means)
    names = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(45)] + ["opacity"]
    names += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    values = np.concatenate((means/scale+center, np.zeros_like(means), dc,
        rest.transpose(0, 2, 1).reshape(n, 45), opacity.reshape(n, 1), logs-np.log(scale), quat), axis=1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite PLY export")
    records = np.empty(n, dtype=[(name, "f4") for name in names])
    for i, name in enumerate(names):
        records[name] = values[:, i]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(records, "vertex")]).write(path)
