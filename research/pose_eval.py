"""Pose-set metrics for the hw pose-replacement experiment (2026-10-30).

Pure numpy (scipy only for the optional sparse refine).  Two jobs:

1. Read a module-A work dir / a raw COLMAP sparse model (txt *or* binary, so it
   works without pycolmap) into a ``PoseSet``.
2. Compare two pose sets after a similarity (Umeyama) alignment: registered
   coverage, per-camera rotation error and translation error relative to the
   scene scale.  This is the "stability matrix" of the report: run the same
   pipeline N times, compare each pair.

CLI
---
    python -m research.pose_eval stability --root outputs/pose-exp-20261030/p0/A \
        --out outputs/pose-exp-20261030/p0/stability.json
    python -m research.pose_eval compare --a <workA> --b <workB>

Conventions: a pose is ``(R_wc, t_wc)`` with ``x_cam = R_wc x_world + t_wc``
(the COLMAP / module-A convention).  The camera centre is ``C = -R_wc^T t_wc``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

# ---------------------------------------------------------------------------
# rotation helpers
# ---------------------------------------------------------------------------


def qvec2rotmat(qvec):
    """COLMAP quaternion order w, x, y, z -> 3x3 rotation matrix."""
    w, x, y, z = [float(v) for v in qvec]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def quat_xyzw_to_rotmat(q):
    """Hamilton quaternion (x, y, z, w) -> rotation matrix (AR frames_info order)."""
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def rotmat2qvec(R):
    """3x3 rotation matrix -> COLMAP quaternion order w, x, y, z."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q


def rot_angle_deg(Ra, Rb):
    """Geodesic angle between two rotation matrices, in degrees."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, c)))))


def mean_rotation(Rs):
    """Chordal L2 mean of rotation matrices (projected to SO(3))."""
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# ---------------------------------------------------------------------------
# pose containers / IO
# ---------------------------------------------------------------------------


class PoseSet:
    """A named set of camera poses (+ optional intrinsics) in one world frame."""

    def __init__(self, name="", poses=None, intrinsics=None, source=None, meta=None):
        self.name = name
        self.poses = dict(poses or {})  # name -> (R_wc 3x3, t_wc (3,))
        self.intrinsics = intrinsics or {}
        self.source = source
        self.meta = dict(meta or {})

    # -- basic access ------------------------------------------------------
    def names(self):
        return list(self.poses.keys())

    def rot(self, n):
        return self.poses[n][0]

    def trans(self, n):
        return self.poses[n][1]

    def centers(self, names=None):
        names = list(self.poses.keys()) if names is None else names
        return np.stack([-self.poses[n][0].T @ self.poses[n][1] for n in names])

    def center(self, n):
        return -self.poses[n][0].T @ self.poses[n][1]

    def c2w(self, n):
        R, t = self.poses[n]
        M = np.eye(4)
        M[:3, :3] = R.T
        M[:3, 3] = -R.T @ t
        return M

    def scene_scale(self):
        C = self.centers()
        if len(C) < 2:
            return 1.0
        # median distance to the centroid: stable against one far-away frame
        return float(np.median(np.linalg.norm(C - C.mean(0), axis=1)))

    def subset(self, names, name=""):
        names = [n for n in names if n in self.poses]
        return PoseSet(name or self.name, {n: self.poses[n] for n in names},
                       dict(self.intrinsics), self.source, dict(self.meta))


def read_cameras_txt(path):
    cams = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            cam_id = int(p[0])
            cams[cam_id] = {
                "model": p[1],
                "width": int(p[2]),
                "height": int(p[3]),
                "params": np.array([float(v) for v in p[4:]]),
            }
    return cams


def read_images_txt(path):
    """Return dict name -> (R_wc, t_wc) from a COLMAP images.txt."""
    poses = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 10:
                continue  # second line of the (image, POINTS2D) pair
            q = [float(v) for v in p[1:5]]
            t = np.array([float(v) for v in p[5:8]])
            poses[p[9]] = (qvec2rotmat(q), t)
    return poses


# COLMAP camera model id -> name (only the ones we can meet here)
_CAM_MODEL_IDS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}


def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            nparams = 4 if model_id in (0, 2, 4) else (
                5 if model_id in (1, 3, 5) else 8)
            # model-dependent param counts
            nparams = {
                0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12,
            }.get(model_id, 4)
            params = np.array(struct.unpack("<" + "d" * nparams, f.read(8 * nparams)))
            cams[cam_id] = {
                "model": _CAM_MODEL_IDS.get(model_id, str(model_id)),
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cams


def read_images_bin(path):
    poses = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            image_id, qw, qx, qy, qz, tx, ty, tz, cam_id = struct.unpack(
                "<i4d3di", f.read(64))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00" or c == b"":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * npts, os.SEEK_CUR)  # (x, y, point3D_id)
            poses[name.decode()] = (
                qvec2rotmat([qw, qx, qy, qz]), np.array([tx, ty, tz]))
    return poses


def read_colmap_model(model_dir):
    """Read a COLMAP sparse model from either txt or bin (whichever exists)."""
    poses, cams = None, None
    if os.path.exists(os.path.join(model_dir, "images.txt")):
        poses = read_images_txt(os.path.join(model_dir, "images.txt"))
        cams = read_cameras_txt(os.path.join(model_dir, "cameras.txt"))
    elif os.path.exists(os.path.join(model_dir, "images.bin")):
        poses = read_images_bin(os.path.join(model_dir, "images.bin"))
        cams = read_cameras_bin(os.path.join(model_dir, "cameras.bin"))
    else:
        raise FileNotFoundError(f"no COLMAP model in {model_dir}")
    intr = {}
    if cams:
        cam = cams[sorted(cams)[0]]
        intr = {"model": cam["model"], "width": cam["width"],
                "height": cam["height"], "params": cam["params"].tolist()}
        if cam["model"] in ("PINHOLE", "OPENCV", "SIMPLE_PINHOLE", "RADIAL",
                            "SIMPLE_RADIAL"):
            p = cam["params"]
            if cam["model"] == "SIMPLE_PINHOLE":
                fx = fy = p[0]; cx, cy = p[1], p[2]
            else:
                fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            intr.update(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))
    return PoseSet(os.path.basename(os.path.normpath(model_dir)), poses, intr,
                   source=model_dir)


def read_work_dir(work, name=None):
    """PoseSet from a module-A work dir or a raw COLMAP sparse model dir."""
    sparse = os.path.join(work, "sparse/0")
    if not (os.path.exists(os.path.join(sparse, "images.txt"))
            or os.path.exists(os.path.join(sparse, "images.bin"))):
        # also accept a raw COLMAP sparse model dir (cameras/images files directly)
        has_model = any(os.path.exists(os.path.join(work, f)) for f in
                        ("images.txt", "images.bin"))
        if not has_model:
            raise FileNotFoundError(f"no COLMAP model in {work} (looked for "
                                    f"sparse/0 and the dir itself)")
        sparse = work
    ps = read_colmap_model(sparse)
    ps.name = name or os.path.basename(os.path.normpath(work))
    ps.meta["work"] = os.path.abspath(work)
    split_p = os.path.join(work, "split.json")
    if os.path.exists(split_p):
        with open(split_p) as f:
            ps.meta["split"] = json.load(f)
    mod_p = os.path.join(work, "module_a.json")
    if os.path.exists(mod_p):
        with open(mod_p) as f:
            ps.meta["module_a"] = json.load(f)
    return ps


# ---------------------------------------------------------------------------
# similarity alignment / comparison
# ---------------------------------------------------------------------------


def umeyama(src, dst, with_scale=True):
    """Least-squares similarity mapping src -> dst (rows are points).

    Returns (s, R, t) such that dst ~= s * (R @ src.T).T + t.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError(f"shape mismatch {src.shape} vs {dst.shape}")
    n = src.shape[0]
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    S = src - mu_s
    D = dst - mu_d
    cov = D.T @ S / n
    U, sv, Vt = np.linalg.svd(cov)
    S_fix = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_fix[2, 2] = -1.0
    R = U @ S_fix @ Vt
    var_s = (S ** 2).sum() / n
    s = float((sv * np.diag(S_fix)).sum() / var_s) if with_scale else 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def relative_pose_disagreement(a, b, names):
    """Gauge-invariant agreement: relative rotations and baseline directions.

    For every camera pair (i, j) compare ``R_a_i R_a_j^T`` with ``R_b_i R_b_j^T``
    (rotation of j in i's frame) and the direction of ``C_a_i - C_a_j`` with
    ``C_b_i - C_b_j``.  These do not depend on the (arbitrary) world frame, so
    they are the right stability numbers for a circular camera orbit where the
    Umeyama rotation is ill-conditioned about the orbit axis.
    """
    names = [n for n in names if n in a.poses and n in b.poses]
    n = len(names)
    if n < 3:
        return {}
    Ra = [a.rot(x) for x in names]
    Rb = [b.rot(x) for x in names]
    Ca = a.centers(names)
    Cb = b.centers(names)
    rr, rt = [], []
    for i in range(n):
        for j in range(i + 1, n):
            rr.append(rot_angle_deg(Ra[i] @ Ra[j].T, Rb[i] @ Rb[j].T))
            # relative translation expressed in camera i's frame: this is the
            # epipolar (translation-direction) agreement and is invariant to the
            # arbitrary world frame, unlike a raw baseline comparison.
            da = Ra[i] @ (Ca[j] - Ca[i])
            db = Rb[i] @ (Cb[j] - Cb[i])
            na, nb = np.linalg.norm(da), np.linalg.norm(db)
            if na > 1e-12 and nb > 1e-12:
                c = float(np.clip((da @ db) / (na * nb), -1.0, 1.0))
                rt.append(float(math.degrees(math.acos(c))))
    rr = np.array(rr)
    rt = np.array(rt)
    return {
        "rel_rot": {"median_deg": float(np.median(rr)),
                    "p90_deg": float(np.percentile(rr, 90)),
                    "max_deg": float(rr.max()), "n_pairs": int(len(rr))},
        "rel_trans_cam": {"median_deg": float(np.median(rt)),
                          "p90_deg": float(np.percentile(rt, 90)),
                          "max_deg": float(rt.max()), "n_pairs": int(len(rt))},
    }


def compare_pose_sets(a, b, names=None):
    """Compare two pose sets on their common images.

    Reports rotation error statistics (after aligning the world frames with the
    Umeyama rotation of the camera centres) and translation error statistics in
    units of the scene scale (median centre radius of ``a``).  Also reports the
    per-set registration counts so a run that silently drops images is visible.
    """
    common = [n for n in a.names() if n in b.poses]
    if names is not None:
        common = [n for n in common if n in names]
    out = {
        "a": a.name, "b": b.name,
        "n_a": len(a.poses), "n_b": len(b.poses), "n_common": len(common),
        "only_a": sorted(set(a.names()) - set(b.names())),
        "only_b": sorted(set(b.names()) - set(a.names())),
    }
    if len(common) < 3:
        out["error"] = "not enough common images"
        return out
    Ca = a.centers(common)
    Cb = b.centers(common)
    scale = a.scene_scale()
    s, R, t = umeyama(Cb, Ca)
    Cb_al = s * (R @ Cb.T).T + t
    d = np.linalg.norm(Cb_al - Ca, axis=1) / max(scale, 1e-12)
    out["align"] = {"scale": s, "rotation_deg": rot_angle_deg(np.eye(3), R)}
    out["trans"] = {
        "mean_rel": float(d.mean()), "median_rel": float(np.median(d)),
        "p90_rel": float(np.percentile(d, 90)), "max_rel": float(d.max()),
    }
    # poses are stored w2c: a world-frame rotation R maps R_wc_b -> R_wc_b @ R^T
    errs = np.array([rot_angle_deg(a.rot(n), b.rot(n) @ R.T) for n in common])
    out["rot"] = {
        "mean_deg": float(errs.mean()), "median_deg": float(np.median(errs)),
        "p90_deg": float(np.percentile(errs, 90)), "max_deg": float(errs.max()),
        "note": "Umeyama-gauge camera-to-world rotation error; ill-conditioned "
                "for a near-planar orbit (see rel_rot/rel_baseline)",
    }
    # Gauge fixed by the *rotations* (not the centres): this is the meaningful
    # "same trajectory, same orientation" alignment.  For a near-circular orbit
    # the centre-only Umeyama above can absorb a constant world-origin shift as a
    # spurious rotation, so both numbers are reported.
    # NOTE the stored poses are w2c, but the gauge is a world rotation acting on
    # c2w: R_cw_a = R_g @ R_cw_b  <=>  R_wc_a = R_wc_b @ R_g^T.
    Rb_cw = np.stack([b.rot(n).T for n in common])
    Ra_cw = np.stack([a.rot(n).T for n in common])
    R_g = mean_rotation(np.stack([Ra_cw[i] @ Rb_cw[i].T for i in range(len(common))]))
    pred = R_g @ Rb_cw
    rot_g = np.array([rot_angle_deg(Ra_cw[i], pred[i]) for i in range(len(common))])
    src = (R_g @ Cb.T).T
    mu_s, mu_d = src.mean(0), Ca.mean(0)
    S, D = src - mu_s, Ca - mu_d
    s_g = float((S * D).sum() / max((S * S).sum(), 1e-12))
    t_g = mu_d - s_g * mu_s
    # residual of Ca ~= s_g*src + t_g  <=>  s_g*S - D  (S, D are mean-centred;
    # adding t_g again here would double-count the mean, a bug that once made a
    # perfectly aligned pair look like a 29%-of-scale disagreement)
    dg = np.linalg.norm(s_g * S - D, axis=1) / max(scale, 1e-12)
    out["rot_gauge_align"] = {
        "scale": s_g,
        "rot_median_deg": float(np.median(rot_g)),
        "rot_p90_deg": float(np.percentile(rot_g, 90)),
        "rot_max_deg": float(rot_g.max()),
        "trans_median_rel": float(np.median(dg)),
        "trans_max_rel": float(dg.max()),
    }
    out.update(relative_pose_disagreement(a, b, common))
    out["scene_scale_a"] = float(scale)
    return out


def stability_analysis(runs_by_scene, out_path=None):
    """Pairwise disagreement matrix for repeats of the *same* pipeline.

    ``runs_by_scene``: {scene: {"r1": PoseSet, "r2": PoseSet, ...}} (or a list of
    PoseSets per scene).
    """
    report = {}
    for scene, runs in runs_by_scene.items():
        if isinstance(runs, dict):
            runs = [runs[k] for k in sorted(runs)]
        rows = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                rows.append({"pair": [runs[i].name, runs[j].name],
                             **compare_pose_sets(runs[i], runs[j])})
        report[scene] = {
            "n_runs": len(runs),
            "registered": {r.name: len(r.poses) for r in runs},
            "pairs": rows,
            "summary": _summary(rows),
        }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
    return report


def _summary(pairs):
    if not pairs:
        return {}
    def agg(key, sub):
        vals = [p[key][sub] for p in pairs if key in p and sub in p[key]]
        return {"median": float(np.median(vals)), "max": float(np.max(vals))} if vals else {}
    n_common = [p.get("n_common", 0) for p in pairs]
    return {
        "rot_median_deg": agg("rot", "median_deg"), "rot_max_deg": agg("rot", "max_deg"),
        "trans_median_rel": agg("trans", "median_rel"),
        "trans_max_rel": agg("trans", "max_rel"),
        "rel_rot_median_deg": agg("rel_rot", "median_deg"),
        "rel_rot_max_deg": agg("rel_rot", "max_deg"),
        "rel_trans_cam_median_deg": agg("rel_trans_cam", "median_deg"),
        "rel_trans_cam_max_deg": agg("rel_trans_cam", "max_deg"),
        "rotgauge_rot_median_deg": agg("rot_gauge_align", "rot_median_deg"),
        "rotgauge_trans_median_rel": agg("rot_gauge_align", "trans_median_rel"),
        "n_common_min": int(min(n_common)) if n_common else 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _discover_runs(root):
    """{scene: {run: PoseSet}} for a tree <root>/<scene>/<run>/sparse/0."""
    out = {}
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    for scene in sorted(os.listdir(root)):
        sdir = os.path.join(root, scene)
        if not os.path.isdir(sdir):
            continue
        runs = {}
        for run in sorted(os.listdir(sdir)):
            rdir = os.path.join(sdir, run)
            if os.path.isdir(os.path.join(rdir, "sparse/0")) or os.path.isdir(
                    os.path.join(rdir, "images")) and os.path.exists(
                        os.path.join(rdir, "split.json")):
                if os.path.exists(os.path.join(rdir, "sparse/0/images.txt")) or \
                        os.path.exists(os.path.join(rdir, "sparse/0/images.bin")):
                    runs[run] = read_work_dir(rdir, name=run)
        if runs:
            out[scene] = runs
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("stability")
    p1.add_argument("--root", required=True)
    p1.add_argument("--out", default=None)
    p2 = sub.add_parser("compare")
    p2.add_argument("--a", required=True)
    p2.add_argument("--b", required=True)
    p3 = sub.add_parser("show")
    p3.add_argument("--work", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "stability":
        runs = _discover_runs(args.root)
        if not runs:
            print(f"no work dirs found under {args.root}", file=sys.stderr)
            return 1
        report = stability_analysis(runs, args.out)
        for scene, rep in report.items():
            print(f"### {scene}  registered={rep['registered']}")
            for p in rep["pairs"]:
                if "rot" not in p:
                    print(f"  {p['pair']}: {p.get('error')}")
                    continue
                rr = p.get("rel_rot", {})
                rt = p.get("rel_trans_cam", {})
                rg = p.get("rot_gauge_align", {})
                print(f"  {p['pair']}: common={p['n_common']}"
                      f" | GAUGE-INVARIANT relrot {rr.get('median_deg', float('nan')):.3f}/"
                      f"{rr.get('max_deg', float('nan')):.2f} deg,"
                      f" reltrans {rt.get('median_deg', float('nan')):.3f}/"
                      f"{rt.get('max_deg', float('nan')):.2f} deg"
                      f" | rot-gauge align rot {rg.get('rot_median_deg', float('nan')):.3f} deg"
                      f" trans {100*rg.get('trans_median_rel', float('nan')):.2f}%"
                      f" | centre-Umeyama rot {p['rot']['median_deg']:.2f} deg"
                      f" trans {100*p['trans']['median_rel']:.2f}%")
            print(f"  summary: {json.dumps(rep['summary'])}")
        return 0

    if args.cmd == "compare":
        a = read_work_dir(args.a)
        b = read_work_dir(args.b)
        print(json.dumps(compare_pose_sets(a, b), indent=2))
        return 0

    if args.cmd == "show":
        ps = read_work_dir(args.work)
        print(f"{ps.name}: {len(ps.poses)} images, intrinsics={ps.intrinsics}")
        print(f"scene_scale={ps.scene_scale():.4f}")
        C = ps.centers()
        print("centre extent (m):", (C.max(0) - C.min(0)))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
