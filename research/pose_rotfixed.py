"""Rotation-known pose pipeline for hw AR-captured scenes (2026-10-30).

Why
---
The organizer hw scenes ship ``frames_info.txt`` with *trustworthy* camera
rotations (validated against an independent COLMAP reconstruction: 0.54 deg
agreement) but invalid camera positions.  The current module-A pipeline ignores
the dump and re-runs incremental COLMAP SfM from scratch, which is (a) slow
(~130 s/scene on 20 CPU cores - SIFT is CPU-only on H1), and (b) non-deterministic
(mapper random seed / BA schedule / sub-model choice differ run to run, so the
downstream 3DGS score swings by tens of proxy points).

This file exploits the known rotations: only camera translations, the common
scale and the 3D points are unknown.

Modes
-----
``--mode rotonly``
    Cheap milestone / falsification test.  Take an existing COLMAP model, replace
    only each camera's rotation with the gauge-aligned AR rotation (camera
    centres kept), write a module-A-compatible work dir.  Isolates "how much of
    the COLMAP pose error is rotation?".

``--mode transavg``  (the replacement candidate)
    Rotations come from ``frames_info.txt``; translations are recovered from the
    SIFT matches in a COLMAP ``database.db``:
      1. per-pair relative translation direction, 2-point RANSAC with the known
         relative rotation (``x_j^T [t]_x R_ij x_i = 0`` is linear in ``t``);
      2. global translation averaging: ``min sum_ij ||(I - d d^T)(C_i - C_j)||^2``
         (linear + IRLS/Huber, one gauge camera fixed, arbitrary scale);
      3. track building (union-find over inlier matches) + linear DLT
         triangulation with a minimum-ray-angle test;
      4. alternating known-rotation bundle adjustment (camera translations and
         points only; rotations pinned) on the reprojection error;
      5. write the work dir by calling the *unmodified* upstream
         ``module_a/preprocess.py::write_work_dir`` so the format is identical.

    The output world frame is the AR world frame - arbitrary but consistent,
    which is all ReSplat / FastGS need.  Everything after ``database.db`` is
    deterministic (fixed per-pair RNG seeds, fixed ordering), unlike the COLMAP
    mapper.  The database itself can be produced once per scene and reused.

``--mode check``
    Compare the AR rotations against an independent COLMAP model: finds the
    quaternion convention, reports the gauge-fit residual, tests the AR positions
    against COLMAP camera centres and compares intrinsics.

Usage (on H1, resplat env; ``PYTHONPATH=<repo>``)
-------------------------------------------------
    python -m research.pose_rotfixed check --scene_dir DATA/hw/car \
        --model outputs/pose-exp-20261030/p0/tmp/car/r1/sparse/0
    python -m research.pose_rotfixed transavg --scene_dir DATA/hw/car \
        --db outputs/pose-exp-20261030/p0/tmp/car/r1/database.db \
        --out outputs/pose-exp-20261030/C/car/transavg
    python -m research.pose_rotfixed rotonly --scene_dir DATA/hw/car \
        --src_work outputs/pose-exp-20261030/p0/A/car/r1 \
        --out outputs/pose-exp-20261030/C/car/rotonly
    python -m research.pose_rotfixed self-check      # synthetic, CPU, no data

No file under ``competition/upstream`` is modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sqlite3
import sys
import time
import traceback

import numpy as np

try:  # normal use: `python -m research.pose_rotfixed`
    from research.pose_eval import (
        PoseSet, qvec2rotmat, quat_xyzw_to_rotmat, mean_rotation,
        read_colmap_model, read_work_dir, rot_angle_deg, rotmat2qvec, umeyama,
    )
except ImportError:  # direct script use
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pose_eval import (  # type: ignore
        PoseSet, qvec2rotmat, quat_xyzw_to_rotmat, mean_rotation,
        read_colmap_model, read_work_dir, rot_angle_deg, rotmat2qvec, umeyama,
    )

# AR camera axes are OpenGL-style (camera looks down -Z, +Y up); converting to
# the OpenCV camera axes used by COLMAP / 3DGS flips Y and Z (det = +1).
D_GL2CV = np.diag([1.0, -1.0, -1.0])

# Candidate readings of (rotX, rotY, rotZ, rotW): each maps the raw quaternion
# matrix Rq to a camera-to-world rotation in OpenCV camera axes.
# MEASURED on hw/car vs an independent COLMAP model (`--mode check`):
#   w2c_cv 0.60 deg mean / 1.22 max   <-- the dump stores a world->camera
#   c2w_cv 9.15 deg mean                  rotation in OpenCV axes, NOT the
#   c2w_gl 80.8 deg mean                  ARKit camera-to-world transform
#   w2c_gl 80.8 deg mean
ROT_CONVENTIONS = {
    "c2w_gl": lambda Rq: Rq @ D_GL2CV,     # ARKit ARCamera.transform (GL axes)
    "c2w_cv": lambda Rq: Rq,               # camera-to-world, OpenCV axes
    "w2c_gl": lambda Rq: Rq.T @ D_GL2CV,   # world-to-camera, GL axes
    "w2c_cv": lambda Rq: Rq.T,             # world-to-camera, OpenCV axes
}
DEFAULT_CONVENTION = "w2c_cv"


# ---------------------------------------------------------------------------
# frames_info.txt
# ---------------------------------------------------------------------------


def parse_frames_info(path):
    """Parse frames_info.txt -> list of dicts (file order)."""
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 15:
                continue
            frames.append({
                "frame": int(p[0]),
                "timestamp": p[1],
                "name": p[1] + ".jpg",
                "focal": (float(p[2]), float(p[3])),
                "principal": (float(p[4]), float(p[5])),
                "declared_size": (int(p[6]), int(p[7])),
                "pos": np.array([float(p[8]), float(p[9]), float(p[10])]),
                "quat_xyzw": np.array([float(p[11]), float(p[12]),
                                       float(p[13]), float(p[14])]),
            })
    return frames


def _jpeg_size(path):
    """(W, H) of a baseline JPEG without PIL (SOF marker scan); None on failure."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA,
                      0xCB, 0xCD, 0xCE, 0xCF):
            h = (data[i + 5] << 8) | data[i + 6]
            w = (data[i + 7] << 8) | data[i + 8]
            return int(w), int(h)
        if marker in (0xD8, 0xD9):
            i += 2
            continue
        seglen = (data[i + 2] << 8) | data[i + 3]
        i += 2 + seglen
    return None


def detect_image_size(scene_dir, frames):
    """Actual (W, H) of the frames_info jpgs (declared 1080x1440 is not always true)."""
    for fr in frames:
        p = os.path.join(scene_dir, "images", fr["name"])
        if os.path.isfile(p):
            sz = _jpeg_size(p)
            if sz:
                return sz
    return frames[0]["declared_size"]


def load_ar_cameras(scene_dir, convention=DEFAULT_CONVENTION, image_size=None):
    """Intrinsics + camera-to-world rotations from frames_info.txt.

    Returns ``(names, K, R_cw, positions, meta)`` with K in *actual image pixel*
    units and R_cw[i] a camera-to-world rotation in OpenCV camera axes.
    """
    frames = parse_frames_info(os.path.join(scene_dir, "frames_info.txt"))
    if not frames:
        raise RuntimeError(f"no usable frames in {scene_dir}/frames_info.txt")
    W_act, H_act = image_size or detect_image_size(scene_dir, frames)
    W_dec, H_dec = frames[0]["declared_size"]
    sx, sy = W_act / float(W_dec), H_act / float(H_dec)
    fx = frames[0]["focal"][0] * sx
    fy = frames[0]["focal"][1] * sy
    px, py = frames[0]["principal"]
    # principalX/Y are 0 for every frame while the focal lengths are real (they
    # match COLMAP to <1%) -> they are offsets from the image centre, not absolute
    # coordinates (a literal (0,0) principal point would be the top-left corner).
    cx = (W_dec / 2.0 + px) * sx
    cy = (H_dec / 2.0 + py) * sy
    conv = ROT_CONVENTIONS[convention]
    names = [fr["name"] for fr in frames]
    R_cw = np.stack([conv(quat_xyzw_to_rotmat(fr["quat_xyzw"])) for fr in frames])
    pos = np.stack([fr["pos"] for fr in frames])
    meta = {
        "scene_dir": os.path.abspath(scene_dir),
        "convention": convention,
        "declared_size": [W_dec, H_dec],
        "actual_size": [W_act, H_act],
        "intrinsics": {"fx": float(fx), "fy": float(fy), "cx": float(cx),
                       "cy": float(cy), "width": int(W_act), "height": int(H_act)},
        "principal_source": "declared zero treated as offset from image centre",
        "n_frames": len(frames),
    }
    K = {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
         "width": int(W_act), "height": int(H_act)}
    return names, K, R_cw, pos, meta


# ---------------------------------------------------------------------------
# gauge alignment against an independent COLMAP reconstruction
# ---------------------------------------------------------------------------


def fit_world_gauge(R_ar_cw, R_col_cw):
    """Best R_g with ``R_col_cw ~= R_g @ R_ar_cw``.  Returns (R_g, residuals_deg)."""
    R_ar_cw = np.asarray(R_ar_cw, dtype=np.float64)
    R_col_cw = np.asarray(R_col_cw, dtype=np.float64)
    Ms = np.stack([R_col_cw[i] @ R_ar_cw[i].T for i in range(len(R_ar_cw))])
    R_g = mean_rotation(Ms)
    pred = R_g @ R_ar_cw
    res = np.array([rot_angle_deg(R_col_cw[i], pred[i])
                    for i in range(len(R_ar_cw))])
    return R_g, res


def check_conventions(scene_dir, model_dir, image_size=None):
    """Report which AR rotation reading agrees with an independent COLMAP model."""
    frames = parse_frames_info(os.path.join(scene_dir, "frames_info.txt"))
    if not frames:
        raise RuntimeError(f"no frames in {scene_dir}/frames_info.txt")
    W_act, H_act = image_size or detect_image_size(scene_dir, frames)
    names, K, R_cw_cv, pos, meta = load_ar_cameras(
        scene_dir, convention="c2w_cv", image_size=(W_act, H_act))
    Rq = {fr["name"]: quat_xyzw_to_rotmat(fr["quat_xyzw"]) for fr in frames}
    col = read_colmap_model(model_dir)
    common = [n for n in names if n in col.poses]
    if len(common) < 8:
        raise RuntimeError(f"only {len(common)} common images between frames_info "
                           f"and {model_dir}")
    # col.rot(n) is w2c -> transpose for c2w
    R_col_cw = np.stack([col.rot(n).T for n in common])
    out = {"model": os.path.abspath(model_dir), "n_common": len(common),
           "actual_size": list(meta["actual_size"]),
           "declared_size": list(meta["declared_size"]),
           "colmap_intrinsics": col.intrinsics, "conventions": {}}
    raw = {
        "c2w_cv": np.stack([Rq[n] for n in common]),
        "c2w_gl": np.stack([Rq[n] @ D_GL2CV for n in common]),
        "w2c_cv": np.stack([Rq[n].T for n in common]),
        "w2c_gl": np.stack([Rq[n].T @ D_GL2CV for n in common]),
    }
    for cname, R_raw in raw.items():
        Rg, res = fit_world_gauge(R_raw, R_col_cw)
        out["conventions"][cname] = {
            "residual_mean_deg": float(res.mean()),
            "residual_median_deg": float(np.median(res)),
            "residual_max_deg": float(res.max()),
            "gauge_rotation_deg": rot_angle_deg(np.eye(3), Rg),
        }
    best = min(out["conventions"],
               key=lambda k: out["conventions"][k]["residual_mean_deg"])
    out["best"] = best

    # translation check: AR positions vs COLMAP camera centres after a similarity
    C_ar = np.stack([pos[names.index(n)] for n in common])
    C_col = np.stack([col.center(n) for n in common])
    s, R, t = umeyama(C_ar, C_col)
    C_fit = s * (C_ar @ R.T) + t
    scale = float(np.median(np.linalg.norm(C_col - C_col.mean(0), axis=1)))
    d = np.linalg.norm(C_fit - C_col, axis=1) / max(scale, 1e-12)
    out["ar_positions_vs_colmap_centres"] = {
        "umeyama_scale": float(s),
        "rot_deg": rot_angle_deg(np.eye(3), R),
        "residual_median_rel": float(np.median(d)),
        "residual_path_length_rel": float(np.median(
            np.linalg.norm(np.diff(C_fit, axis=0), axis=1)) /
            max(np.median(np.linalg.norm(np.diff(C_col, axis=0), axis=1)), 1e-12)),
        "residual_max_rel": float(d.max()),
        "note": "large => AR positions unusable as translations; only the "
                "rotations are kept (matches the organizer README)",
    }

    # intrinsics: AR focal (scaled to the actual image) vs COLMAP's estimate
    ci = col.intrinsics
    if "fx" in ci:
        out["intrinsics_ar_scaled"] = {k: K[k] for k in ("fx", "fy", "cx", "cy")}
        out["intrinsic_delta_px"] = {
            "fx": float(K["fx"] - ci["fx"]), "fy": float(K["fy"] - ci["fy"]),
            "cx": float(K["cx"] - ci["cx"]), "cy": float(K["cy"] - ci["cy"]),
            "fx_rel": float((K["fx"] - ci["fx"]) / ci["fx"]),
            "fy_rel": float((K["fy"] - ci["fy"]) / ci["fy"]),
        }
        params = ci.get("params", [])
        out["colmap_distortion"] = [float(v) for v in params[4:]] if len(params) > 4 else []
    return out


# ---------------------------------------------------------------------------
# COLMAP database
# ---------------------------------------------------------------------------


def load_db(db_path, names):
    """Read images/keypoints/matches for ``names`` from a COLMAP database.

    Returns ``(name2id, keypoints, matches)``:
      * name2id:  {name: image_id}
      * keypoints: {image_id: (N,2) float64}
      * matches:  list of (id1, id2, (M,2) int64 index pairs), id1 < id2
    """
    conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT image_id, name FROM images").fetchall()
        name2id = {str(n): int(i) for i, n in rows}
        kps = {}
        for n in names:
            iid = name2id.get(n)
            if iid is None:
                continue
            r = conn.execute(
                "SELECT rows, cols, data FROM keypoints WHERE image_id = ?",
                (iid,)).fetchone()
            if r is None:
                continue
            nrow, ncol, blob = r
            if blob is None:
                continue
            arr = np.frombuffer(blob, dtype=np.float32).reshape(int(nrow), int(ncol))
            kps[iid] = arr[:, :2].astype(np.float64)
        want = set(kps)
        matches = []
        for pid, nrow, ncol, blob in conn.execute(
                "SELECT pair_id, rows, cols, data FROM matches"):
            i1 = int(pid) // 2147483647
            i2 = int(pid) % 2147483647
            if i1 not in want or i2 not in want or blob is None:
                continue
            arr = np.frombuffer(blob, dtype=np.uint32).reshape(int(nrow), int(ncol))
            if arr.shape[0] == 0:
                continue
            matches.append((min(i1, i2), max(i1, i2), arr.astype(np.int64)))
        return name2id, kps, matches
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def invert_K(K):
    return np.array([[1.0 / K["fx"], 0.0, -K["cx"] / K["fx"]],
                     [0.0, 1.0 / K["fy"], -K["cy"] / K["fy"]],
                     [0.0, 0.0, 1.0]])


def bearings_from_pixels(uv, K):
    """Pixel coords (N,2) -> normalized bearings (N,3) with z = 1."""
    x = (uv[:, 0] - K["cx"]) / K["fx"]
    y = (uv[:, 1] - K["cy"]) / K["fy"]
    return np.stack([x, y, np.ones_like(x)], axis=1)


def project_points(X, R_wc, C, K):
    """World points -> pixels under w2c (R_wc, -R_wc C); also returns depths."""
    Xc = (X - C) @ R_wc.T
    z = Xc[:, 2]
    u = K["fx"] * Xc[:, 0] / z + K["cx"]
    v = K["fy"] * Xc[:, 1] / z + K["cy"]
    return np.stack([u, v], axis=1), z


def triangulate_dlt(Ps, xs):
    """Linear DLT.  Ps: list of (3,4) projection matrices, xs: list of (N,3) bearings."""
    M = np.zeros((4, 4))
    for P, x in zip(Ps, xs):
        r1 = x[:, 0:1] * P[2][None, :] - P[0][None, :]
        r2 = x[:, 1:2] * P[2][None, :] - P[1][None, :]
        M += r1.T @ r1 + r2.T @ r2
    try:
        _w, V = np.linalg.eigh(M)
    except np.linalg.LinAlgError:
        return None
    X = V[:, 0]
    if abs(X[3]) < 1e-12 or not np.isfinite(X).all():
        return None
    return (X[:3] / X[3]).reshape(1, 3)


def estimate_pair_translation(b_i, b_j, R_ij, f_eff, seed=0, n_iters=96,
                              thresh_px=2.0, min_inliers=25):
    """Relative translation direction t with ``X_j = R_ij X_i + t``.

    2-point RANSAC on the linear-in-t epipolar constraint; deterministic given
    ``seed``.  Returns None if the pair is unusable.
    """
    N = len(b_i)
    if N < max(min_inliers, 8):
        return None
    y = b_i @ R_ij.T
    A = np.cross(y, b_j)          # rows: t . (R_ij x_i x x_j) = 0
    if not np.isfinite(A).all():
        return None
    thresh = thresh_px / max(f_eff, 1e-9)
    rng = np.random.default_rng((int(seed) * 2654435761 + int(N)) % (2 ** 32))
    idx = rng.integers(0, N, size=(n_iters, 2))
    same = idx[:, 1] == idx[:, 0]
    idx[same, 1] = (idx[same, 1] + 1) % N
    T = np.cross(A[idx[:, 0]], A[idx[:, 1]])
    nz = np.linalg.norm(T, axis=1)
    good = nz > 1e-12
    if not good.any():
        return None
    T = T[good] / nz[good, None]
    E = np.abs(A @ T.T)                        # (N, n_hyp)
    counts = (E < thresh).sum(axis=0)
    best = int(np.argmax(counts))
    t = T[best]
    mask = E[:, best] < thresh
    for _ in range(2):
        if mask.sum() < 4:
            return None
        try:
            _u, _s, Vt = np.linalg.svd(A[mask], full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        t = Vt[-1]
        n = np.linalg.norm(t)
        if n < 1e-12:
            return None
        t = t / n
        mask = np.abs(A @ t) < thresh
    if mask.sum() < min_inliers:
        return None
    t = _orient_translation(t, b_i[mask], b_j[mask], R_ij)
    if t is None:
        return None
    mask = np.abs(A @ t) < thresh
    return {"t": t, "inliers": mask, "n_inliers": int(mask.sum()),
            "inlier_ratio": float(mask.mean()),
            "median_epipolar_px": float(np.median(np.abs(A @ t)[mask]) * f_eff)}


def _orient_translation(t, b_i, b_j, R_ij, max_pts=60):
    """Flip t so that the inlier correspondences triangulate in front of both cams.

    IMPORTANT: with two views the constraint is one scalar equation per match, so
    the sign of t is only fixed by cheirality.  Each match must be triangulated
    *individually* here - passing all matches to the multi-view DLT would average
    them into one meaningless point (this exact bug would silently flip ~half of
    the pairs and made translation averaging return a reflected solution).
    """
    n = len(b_i)
    if n == 0:
        return t
    sel = np.linspace(0, n - 1, min(n, max_pts)).astype(int)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])

    def count_positive(tt):
        P2 = np.hstack([R_ij, tt.reshape(3, 1)])
        pos = tot = 0
        for k in sel:
            X = triangulate_dlt([P1, P2], [b_i[k:k + 1], b_j[k:k + 1]])
            if X is None:
                continue
            tot += 1
            if X[0, 2] > 0 and float((R_ij @ X[0] + tt)[2]) > 0:
                pos += 1
        return pos, tot

    p1, n1 = count_positive(t)
    p2, n2 = count_positive(-t)
    if n1 == 0 and n2 == 0:
        return t
    return -t if p2 > p1 else t


# ---------------------------------------------------------------------------
# translation averaging
# ---------------------------------------------------------------------------


def translation_averaging(dirs, n_cams, iterations=8):
    """Global camera centres from pairwise unit baselines.

    ``dirs``: list of (i, j, d, weight) meaning ``C_i - C_j`` is parallel to the
    unit vector ``d``.  Gauge: the first active camera is fixed at the origin;
    the overall scale is arbitrary (unit RMS afterwards).

    Returns ``(C, info)``: C (n_cams, 3); cameras outside the largest connected
    component are left at zero and listed in ``info['inactive_cameras']``.
    """
    active = sorted({i for i, j, d, w in dirs} | {j for i, j, d, w in dirs})
    if len(active) < 4:
        raise RuntimeError(f"translation-averaging graph too small ({len(active)} cams)")
    root = active[0]
    var = [c for c in active if c != root]
    col = {c: k for k, c in enumerate(var)}
    nv = 3 * len(var)
    pairs = [(i, j, d) for (i, j, d, w) in dirs]
    w0 = np.array([w for (_i, _j, _d, w) in dirs], dtype=np.float64)

    def solve(weights):
        A = np.zeros((3 * len(pairs), nv))
        for r, (i, j, d) in enumerate(pairs):
            P = np.eye(3) - np.outer(d, d)
            P *= math.sqrt(max(weights[r], 1e-8))
            if i in col:
                A[3 * r:3 * r + 3, 3 * col[i]:3 * col[i] + 3] += P
            if j in col:
                A[3 * r:3 * r + 3, 3 * col[j]:3 * col[j] + 3] -= P
        _u, sv, Vt = np.linalg.svd(A, full_matrices=False)
        c = Vt[-1]
        C = np.zeros((n_cams, 3))
        for cc, k in col.items():
            C[cc] = c[3 * k:3 * k + 3]
        rms = math.sqrt(float((C[active] ** 2).sum()) / len(active))
        if rms < 1e-12:
            raise RuntimeError("translation averaging degenerate (zero scale)")
        return C / rms, float(sv[-1])

    weights = w0.copy()
    info = {"n_active_cameras": len(active), "n_pairs": len(pairs)}
    C = None
    for it in range(iterations):
        C, smallest_sv = solve(weights)
        # A direction-only system is invariant to a global sign flip (C -> -C is
        # still parallel to every d).  Resolve the mirror ambiguity with the
        # physical requirement C_i - C_j = s_ij d_ij with s_ij > 0.
        votes = sum(1 if float(d @ (C[i] - C[j])) > 0 else -1
                    for (i, j, d) in pairs)
        if votes < 0:
            C = -C
        res = np.zeros(len(pairs))
        for r, (i, j, d) in enumerate(pairs):
            base = C[i] - C[j]
            nb = float(np.linalg.norm(base))
            res[r] = 0.0 if nb < 1e-12 else float(
                np.linalg.norm((np.eye(3) - np.outer(d, d)) @ base) / nb)
        info["residual_median_deg"] = float(
            math.degrees(math.asin(min(1.0, float(np.median(res))))))
        info["residual_p90_deg"] = float(
            math.degrees(math.asin(min(1.0, float(np.percentile(res, 90))))))
        info["smallest_singular_value"] = smallest_sv
        scale = max(float(np.median(res)), 1e-6)
        w_new = np.array([
            w * (1.0 if r <= 2.0 * scale else 2.0 * scale / max(r, 1e-12))
            for (r, w) in zip(res, weights)])
        if it > 0 and np.allclose(w_new, weights, rtol=1e-3, atol=1e-9):
            break
        weights = w_new
    info["iterations"] = it + 1
    info["inactive_cameras"] = sorted(set(range(n_cams)) - set(active))
    return C, info


# ---------------------------------------------------------------------------
# tracks + known-rotation BA
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, a):
        p = self.parent.setdefault(a, a)
        while p != self.parent.get(p, p):
            self.parent[p] = self.parent.get(self.parent[p], self.parent[p])
            p = self.parent[p]
        return p

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_tracks(pair_inliers):
    """Union-find over (image_id, kp_idx) using the RANSAC inlier matches."""
    uf = UnionFind()
    for (i1, i2, idx, mask) in pair_inliers:
        for (a, b) in idx[mask]:
            uf.union((i1, int(a)), (i2, int(b)))
    roots = {}
    for node in list(uf.parent.keys()):
        roots.setdefault(uf.find(node), []).append(node)
    return [v for v in roots.values() if len(v) >= 2]


def _reproj_err(X, obs, R_wc, C, K):
    """Per-observation (img_idx, kp_idx, uv, pixel_error) for one world point."""
    out = []
    for (i, kidx, uv) in obs:
        Xc = R_wc[i] @ (X - C[i])
        if Xc[2] <= 1e-9:
            out.append((i, kidx, uv, float("inf")))
            continue
        u = K["fx"] * Xc[0] / Xc[2] + K["cx"]
        v = K["fy"] * Xc[1] / Xc[2] + K["cy"]
        out.append((i, kidx, uv, math.hypot(u - uv[0], v - uv[1])))
    return out


def triangulate_tracks(tracks, uv_of, R_wc, C, K, id2idx, min_angle_deg=1.0,
                       max_reproj_px=6.0):
    """Triangulate union-find tracks; returns (track_obs, points, stats).

    ``track_obs[k] = [(img_idx, kp_idx, (u,v)), ...]`` and ``points[k]`` is the
    world point.  Tracks with a too-small maximum ray angle, a point behind a
    camera or a reprojection error above ``max_reproj_px`` are dropped.
    """
    track_obs, points = [], []
    stats = {"tracks_input": len(tracks), "tracks_ok": 0,
             "rejected_angle": 0, "rejected_geom": 0, "rejected_reproj": 0}
    Kinv = invert_K(K)
    for tr in tracks:
        obs = []
        for (iid, kidx) in tr:
            img_idx = id2idx.get(iid)
            if img_idx is None:
                continue
            uv = uv_of(img_idx, kidx)
            obs.append((img_idx, kidx, uv))
        if len(obs) < 2 or len({o[0] for o in obs}) < 2:
            continue
        Ps, xs = [], []
        for (i, _k, uv) in obs:
            x = Kinv @ np.array([uv[0], uv[1], 1.0])
            Ps.append(np.hstack([R_wc[i], (-R_wc[i] @ C[i]).reshape(3, 1)]))
            xs.append(x.reshape(1, 3))
        X = triangulate_dlt(Ps, xs)
        if X is None:
            stats["rejected_geom"] += 1
            continue
        X = X[0]
        # minimum triangulation angle between the extreme viewing rays
        rays = []
        for (i, _k, _uv) in obs:
            r = X - C[i]
            n = float(np.linalg.norm(r))
            if n > 1e-12:
                rays.append(r / n)
        if len(rays) >= 2:
            cos_min = float(np.min(np.stack(rays) @ np.stack(rays).T))
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cos_min))))
            if ang < min_angle_deg:
                stats["rejected_angle"] += 1
                continue
        errs = _reproj_err(X, obs, R_wc, C, K)
        if any((not np.isfinite(e[3])) or e[3] > max_reproj_px for e in errs):
            stats["rejected_reproj"] += 1
            continue
        track_obs.append(obs)
        points.append(X)
    stats["tracks_ok"] = len(points)
    return track_obs, np.stack(points) if points else np.zeros((0, 3)), stats


def refine_known_rotation(track_obs, points, R_wc, C, K, iterations=6,
                          huber_px=2.0, prune_px=8.0, verbose=False,
                          refine_rotation_from=None):
    """Alternating BA: re-triangulate points, then GN on camera parameters.

    ``refine_rotation_from``: if not None, rotations are held fixed for the first
    ``refine_rotation_from`` iterations (known-rotation BA, i.e. using the AR
    rotations as a seed) and then released, so a joint rotations+translations BA
    can reach COLMAP-level self-consistency.  Camera 0 stays fixed in both the
    rotation and the translation for the gauge.

    Returns (track_obs, points, R_wc, C, hist).
    """
    C = np.array(C, dtype=np.float64, copy=True)
    R_wc = np.array(R_wc, dtype=np.float64, copy=True)
    points = np.array(points, dtype=np.float64, copy=True)
    n_cams = len(C)
    seed_R = R_wc.copy()
    Kinv = invert_K(K)
    hist = []
    for it in range(iterations):
        # ---- points: re-triangulate with the current centres ----
        new_pts = np.zeros_like(points)
        keep = np.zeros(len(track_obs), dtype=bool)
        Ps_cache = [np.hstack([R_wc[i], (-R_wc[i] @ C[i]).reshape(3, 1)])
                    for i in range(n_cams)]
        for k, obs in enumerate(track_obs):
            Ps = [Ps_cache[i] for (i, _k, _uv) in obs]
            xs = [(Kinv @ np.array([uv[0], uv[1], 1.0])).reshape(1, 3)
                  for (_i, _k, uv) in obs]
            X = triangulate_dlt(Ps, xs)
            if X is None:
                continue
            new_pts[k] = X[0]
            keep[k] = True
        # ---- cameras: GN on reprojection error (3 or 6 params per camera) ----
        free_rot = (refine_rotation_from is not None and it >= refine_rotation_from)
        C_new = C.copy()
        R_new = R_wc.copy()
        n_updated = 0
        for cam in range(1, n_cams):  # camera 0 is the gauge in both modes
            obs_idx = [k for k, obs in enumerate(track_obs) if keep[k]
                       and any(i == cam for (i, _k, _uv) in obs)]
            if not obs_idx:
                continue
            X = new_pts[obs_idx]
            uv = np.stack([np.array([e[2] for e in track_obs[k] if e[0] == cam][0])
                           for k in obs_idx])
            Xc = (X - C[cam]) @ R_wc[cam].T
            z = Xc[:, 2]
            if np.any(z <= 1e-9):
                continue
            u = K["fx"] * Xc[:, 0] / z + K["cx"]
            v = K["fy"] * Xc[:, 1] / z + K["cy"]
            r = np.stack([u - uv[:, 0], v - uv[:, 1]], axis=1)
            rn = np.linalg.norm(r, axis=1)
            w = np.where(rn <= huber_px, 1.0, huber_px / np.maximum(rn, 1e-9))
            Jp = np.zeros((len(X), 2, 3))
            Jp[:, 0, 0] = K["fx"] / z
            Jp[:, 0, 2] = -K["fx"] * Xc[:, 0] / z ** 2
            Jp[:, 1, 1] = K["fy"] / z
            Jp[:, 1, 2] = -K["fy"] * Xc[:, 1] / z ** 2
            Jt = (-Jp @ R_wc[cam]).reshape(-1, 3)   # d(res)/dC_cam
            if free_rot:
                # left perturbation R' = exp(delta) R  =>  dXc/ddelta = -[Xc]_x
                skew = np.zeros((len(X), 3, 3))
                skew[:, 0, 1] = -Xc[:, 2]; skew[:, 0, 2] = Xc[:, 1]
                skew[:, 1, 0] = Xc[:, 2];  skew[:, 1, 2] = -Xc[:, 0]
                skew[:, 2, 0] = -Xc[:, 1]; skew[:, 2, 1] = Xc[:, 0]
                Jr = (-Jp @ skew).reshape(-1, 3)     # d(res)/d rot delta
                J = np.concatenate([Jr, Jt], axis=1)  # (2N, 6)
                lam = 1e-6 * np.eye(6)
            else:
                J = Jt
                lam = 1e-6 * np.eye(3)
            wf = np.repeat(w, 2)
            H = (J * wf[:, None]).T @ J + lam
            g = (J * wf[:, None]).T @ r.reshape(-1)
            try:
                delta = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                continue
            if free_rot:
                delta_r = np.clip(delta[:3], -0.05, 0.05)   # <=~3 deg per iter
                delta_r = delta_r * 0.5                     # damping
                th = float(np.linalg.norm(delta_r))
                if th > 1e-12:
                    ax = delta_r / th
                    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]],
                                   [-ax[1], ax[0], 0]])
                    dR = np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)
                else:
                    dR = np.eye(3)
                R_new[cam] = dR @ R_new[cam]
                U, _sv, Vt = np.linalg.svd(R_new[cam])
                R_new[cam] = U @ Vt
            C_new[cam] = C[cam] + np.clip(delta[-3:], -1e6, 1e6)
            n_updated += 1
        # ---- prune bad observations, then rebuild the track list ----
        limit = prune_px if it < iterations - 1 else 2.0 * prune_px
        pruned_obs, pruned_pts = [], []
        for k, obs in enumerate(track_obs):
            if not keep[k]:
                continue
            X = new_pts[k]
            good = [(i, kidx, uv)
                    for (i, kidx, uv, e) in _reproj_err(X, obs, R_new, C_new, K)
                    if np.isfinite(e) and e <= limit]
            if len(good) >= 2 and len({g[0] for g in good}) >= 2:
                pruned_obs.append(good)
                pruned_pts.append(X)
        track_obs = pruned_obs
        points = np.stack(pruned_pts) if pruned_pts else np.zeros((0, 3))
        C = C_new
        R_wc = R_new
        if free_rot:
            dev = [rot_angle_deg(seed_R[i], R_wc[i]) for i in range(n_cams)]
            hist.append({"iter": it, "n_points": len(points),
                         "n_cameras_updated": n_updated, "rotations_free": True,
                         "rot_dev_from_seed_median_deg": float(np.median(dev)),
                         "rot_dev_from_seed_max_deg": float(np.max(dev))})
        else:
            hist.append({"iter": it, "n_points": len(points),
                         "n_cameras_updated": n_updated, "rotations_free": False})
        if verbose:
            print(f"    refine iter {it}: points={len(points)} "
                  f"cameras={n_updated}", flush=True)
        if len(points) < 100:
            break
    return track_obs, points, R_wc, C, hist


def reprojection_stats(track_obs, points, R_wc, C, K):
    errs = []
    for k, obs in enumerate(track_obs):
        for (i, kidx, uv, e) in _reproj_err(points[k], obs, R_wc, C, K):
            if np.isfinite(e):
                errs.append(e)
    errs = np.array(errs) if errs else np.array([np.nan])
    return {"median_px": float(np.nanmedian(errs)),
            "p90_px": float(np.nanpercentile(errs, 90)),
            "mean_px": float(np.nanmean(errs)), "n_observations": int(len(errs))}


# ---------------------------------------------------------------------------
# work-dir writing
# ---------------------------------------------------------------------------


def _load_upstream_preprocess(repo_root):
    path = os.path.join(repo_root, "competition/upstream/baseline/module_a/preprocess.py")
    spec = importlib.util.spec_from_file_location("module_a_preprocess", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)     # imports torch (resplat env) at module load
    return mod


def write_images_txt(path, names, R_wc, C, camera_id=1):
    """COLMAP images.txt with the same layout as upstream write_work_dir."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for idx, n in enumerate(names):
            R = np.asarray(R_wc[idx], dtype=np.float64)
            t = -R @ np.asarray(C[idx], dtype=np.float64)
            q = rotmat2qvec(R)
            stem = os.path.splitext(n)[0]
            f.write(f"{idx + 1} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g} "
                    f"{t[0]:.12g} {t[1]:.12g} {t[2]:.12g} {camera_id} {stem}.png\n\n")


def build_pose_dict(names, R_wc, C, K, scene_dir, source):
    poses = {}
    for i, n in enumerate(names):
        R = np.asarray(R_wc[i], dtype=np.float64)
        poses[n] = (R, -R @ np.asarray(C[i], dtype=np.float64))
    return {"model": "PINHOLE", "W": int(K["width"]), "H": int(K["height"]),
            "fx": float(K["fx"]), "fy": float(K["fy"]),
            "cx": float(K["cx"]), "cy": float(K["cy"]),
            "poses": poses, "source": source,
            "images_dir": os.path.join(scene_dir, "images")}


def write_work_dir(scene_dir, out_dir, pose_dict, args, repo_root, provenance):
    """Write a module-A work dir via the unmodified upstream writer."""
    up = _load_upstream_preprocess(repo_root)
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    split, resize_time = up.write_work_dir(
        out_dir, scene_dir, pose_dict, args.num_context, args.llffhold,
        args.max_resolution)
    info = {
        "group": "hw",
        "scene_dir": os.path.abspath(scene_dir),
        "pose_source": pose_dict.get("source"),
        "pose_estimation_seconds": provenance.get("pose_estimation_seconds", 0.0),
        "resize_seconds": resize_time,
        "total_seconds": provenance.get("pose_estimation_seconds",
                                        time.time() - t0) + resize_time,
        "num_images": len(pose_dict["poses"]),
        "num_train": len(split["train"]),
        "num_test": len(split["test"]),
        "num_context": args.num_context,
        "resolution": split["resolution"],
        "candidate": provenance,
    }
    with open(os.path.join(out_dir, "module_a.json"), "w") as f:
        json.dump(info, f, indent=2)
    return split, info


# ---------------------------------------------------------------------------
# mode: transavg
# ---------------------------------------------------------------------------


def run_transavg(args):
    t_start = time.time()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene_dir = os.path.abspath(args.scene_dir)
    names, K, R_cw, pos, meta = load_ar_cameras(
        scene_dir, convention=args.convention, image_size=args.image_size)
    R_wc = np.stack([R.T for R in R_cw])
    name2idx = {n: i for i, n in enumerate(names)}
    print(f"[rotfixed] {len(names)} AR frames; K={ {k: round(v, 4) for k, v in K.items()} }",
          flush=True)

    # ---- database ----
    t_db = time.time()
    name2id, kps, matches = load_db(args.db, names)
    id2idx = {i: name2idx[n] for n, i in name2id.items()}
    id2idx_inv = {v: k for k, v in id2idx.items()}

    def uv_of(img_idx, kidx):
        return kps[id2idx_inv[img_idx]][kidx]
    print(f"[rotfixed] db: {len(name2id)} images, {len(matches)} match pairs "
          f"({time.time() - t_db:.1f}s)", flush=True)
    missing = [n for n in names if n not in name2id]
    if missing:
        print(f"[rotfixed] WARNING {len(missing)} AR frames missing from db: "
              f"{missing[:5]}", flush=True)

    # ---- per-pair known-rotation translation ----
    t_pair = time.time()
    f_eff = 0.5 * (K["fx"] + K["fy"])
    pair_dirs, pair_inliers, pair_stats = [], [], []
    for (i1, i2, idx_pairs) in matches:
        if i1 not in id2idx or i2 not in id2idx:
            continue
        a, b = id2idx[i1], id2idx[i2]
        b_i = bearings_from_pixels(kps[i1][idx_pairs[:, 0]], K)
        b_j = bearings_from_pixels(kps[i2][idx_pairs[:, 1]], K)
        b_i /= np.linalg.norm(b_i, axis=1, keepdims=True)
        b_j /= np.linalg.norm(b_j, axis=1, keepdims=True)
        R_ij = R_wc[b] @ R_wc[a].T          # cam a -> cam b
        est = estimate_pair_translation(
            b_i, b_j, R_ij, f_eff, seed=(i1 * 131 + i2) % 100000,
            thresh_px=args.pair_thresh_px, min_inliers=args.min_pair_inliers)
        if est is None:
            pair_stats.append({"pair": [int(i1), int(i2)], "used": False,
                               "n_matches": int(len(idx_pairs))})
            continue
        # t is the relative translation in cam b coords: C_a - C_b = R_wc[b]^T t
        d_world = R_wc[b].T @ est["t"]
        d_world /= max(float(np.linalg.norm(d_world)), 1e-12)
        pair_dirs.append((a, b, d_world, math.sqrt(est["n_inliers"])))
        pair_inliers.append((i1, i2, idx_pairs, est["inliers"]))
        pair_stats.append({"pair": [int(i1), int(i2)], "used": True,
                           "n_matches": int(len(idx_pairs)),
                           "n_inliers": est["n_inliers"],
                           "inlier_ratio": est["inlier_ratio"],
                           "median_epipolar_px": est["median_epipolar_px"]})
    t_pair = time.time() - t_pair
    if len(pair_dirs) < 10:
        raise RuntimeError(f"only {len(pair_dirs)} usable pairs")
    ratios = np.array([p["inlier_ratio"] for p in pair_stats if p["used"]])
    print(f"[rotfixed] pair translations: {len(pair_dirs)} usable / "
          f"{len(pair_stats)} pairs ({t_pair:.1f}s); inlier-ratio median "
          f"{np.median(ratios):.2f}; epipolar median "
          f"{np.median([p['median_epipolar_px'] for p in pair_stats if p['used']]):.2f}px",
          flush=True)

    # ---- translation averaging ----
    t_ta = time.time()
    C, ta_info = translation_averaging(pair_dirs, len(names),
                                       iterations=args.ta_iterations)
    t_ta = time.time() - t_ta
    print(f"[rotfixed] translation averaging ({t_ta:.1f}s): "
          f"{ta_info['residual_median_deg']:.3f} deg median / "
          f"{ta_info['residual_p90_deg']:.3f} deg p90; "
          f"{ta_info['n_active_cameras']}/{len(names)} cameras", flush=True)
    if not np.isfinite(C).all():
        raise RuntimeError("translation averaging produced NaNs")

    # ---- tracks + triangulation ----
    t_tri = time.time()
    tracks = build_tracks(pair_inliers)
    track_obs, points, tri_stats = triangulate_tracks(
        tracks, uv_of, R_wc, C, K, id2idx, min_angle_deg=args.min_tri_angle)
    t_tri = time.time() - t_tri
    print(f"[rotfixed] triangulation ({t_tri:.1f}s): {tri_stats}", flush=True)
    if len(points) < 200:
        raise RuntimeError(f"only {len(points)} points triangulated")

    # ---- known-rotation BA ----
    t_ba = time.time()
    refine_from = (None if args.ba_rot_free_from is None
                   else max(0, int(args.ba_rot_free_from)))
    track_obs, points, R_wc_ref, C, hist = refine_known_rotation(
        track_obs, points, R_wc, C, K, iterations=args.ba_iterations,
        verbose=args.verbose, refine_rotation_from=refine_from)
    rot_dev_from_ar = [rot_angle_deg(R_wc[i], R_wc_ref[i]) for i in range(len(names))]
    t_ba = time.time() - t_ba
    reproj = reprojection_stats(track_obs, points, R_wc_ref, C, K)
    print(f"[rotfixed] known-rotation BA ({t_ba:.1f}s): {len(points)} points; "
          f"reprojection median {reproj['median_px']:.2f}px / "
          f"p90 {reproj['p90_px']:.2f}px", flush=True)

    # ---- write the work dir ----
    used = [i for i in range(len(names)) if i not in set(ta_info["inactive_cameras"])]
    names_used = [names[i] for i in used]
    R_used = [R_wc_ref[i] for i in used]
    C_used = [C[i] for i in used]
    pose_dict = build_pose_dict(names_used, R_used, C_used, K, scene_dir,
                                "rotfixed_transavg")
    provenance = {
        "pipeline": "rotfixed_transavg_v1",
        "db": os.path.abspath(args.db),
        "convention": args.convention,
        "intrinsics": {k: (int(v) if isinstance(v, int) else float(v))
                       for k, v in K.items()},
        "declared_size": meta["declared_size"],
        "actual_size": meta["actual_size"],
        "n_frames_frames_info": len(names),
        "n_registered": len(names_used),
        "unregistered": [n for n in names if n not in set(names_used)],
        "pair_stats": {"n_pairs_usable": len(pair_dirs),
                       "n_pairs_total": len(pair_stats),
                       "inlier_ratio_median": float(np.median(ratios)),
                       "epipolar_median_px": float(np.median(
                           [p["median_epipolar_px"] for p in pair_stats if p["used"]]))},
        "translation_averaging": ta_info,
        "triangulation": tri_stats,
        "reprojection": reproj,
        "ba_rot_free_from": refine_from,
        "rot_dev_from_ar_median_deg": float(np.median(rot_dev_from_ar)),
        "rot_dev_from_ar_max_deg": float(np.max(rot_dev_from_ar)),
        "pair_seconds": t_pair, "ta_seconds": t_ta,
        "triangulation_seconds": t_tri, "ba_seconds": t_ba,
        "ba_history": hist,
        "pose_estimation_seconds": time.time() - t_start,
    }
    split, info = write_work_dir(scene_dir, args.out, pose_dict, args, repo_root,
                                 provenance)
    ps = PoseSet("rotfixed", {n: (R_used[i], -R_used[i] @ C_used[i])
                              for i, n in enumerate(names_used)}, K, "rotfixed")
    ps.meta["provenance"] = provenance
    return ps, info


# ---------------------------------------------------------------------------
# mode: rotonly
# ---------------------------------------------------------------------------


def run_rotonly(args):
    """Replace only the rotations of an existing work dir with gauge-aligned AR ones."""
    t_start = time.time()
    scene_dir = os.path.abspath(args.scene_dir)
    src = read_work_dir(args.src_work, name="src")
    names, K, R_cw, pos, meta = load_ar_cameras(
        scene_dir, convention=args.convention, image_size=args.image_size)
    # AR names are "<timestamp>.jpg" while a module-A work dir stores "<stem>.png":
    # join on the extension-free stem.
    ar_idx = {os.path.splitext(n)[0]: i for i, n in enumerate(names)}
    src_stems = {n: os.path.splitext(n)[0] for n in src.names()}
    common = [n for n in src.names() if src_stems[n] in ar_idx]
    if len(common) < 8:
        raise RuntimeError(f"{len(common)} common images between AR and work dir")
    R_ar = np.stack([R_cw[ar_idx[src_stems[n]]] for n in common])
    R_col_cw = np.stack([src.rot(n).T for n in common])
    R_g, res = fit_world_gauge(R_ar, R_col_cw)
    print(f"[rotonly] gauge fit over {len(common)} images: mean {res.mean():.3f} deg, "
          f"median {np.median(res):.3f} deg, max {res.max():.3f} deg", flush=True)

    # output = a copy of the source work dir with only sparse/0/images.txt rewritten
    os.makedirs(args.out, exist_ok=True)
    if not os.path.exists(os.path.join(args.out, "images")):
        shutil.copytree(os.path.join(args.src_work, "images"),
                        os.path.join(args.out, "images"))
    os.makedirs(os.path.join(args.out, "sparse/0"), exist_ok=True)
    shutil.copy(os.path.join(args.src_work, "sparse/0/cameras.txt"),
                os.path.join(args.out, "sparse/0/cameras.txt"))
    if os.path.exists(os.path.join(args.src_work, "split.json")):
        shutil.copy(os.path.join(args.src_work, "split.json"),
                    os.path.join(args.out, "split.json"))

    names_out = src.names()
    R_new, C_keep = [], []
    for n in names_out:
        stem = os.path.splitext(n)[0]
        if stem not in ar_idx:
            raise RuntimeError(f"no AR rotation for {n}")
        R_new.append((R_g @ R_cw[ar_idx[stem]]).T)   # w2c rotation, COLMAP world frame
        C_keep.append(src.center(n))              # keep the COLMAP camera centre
    C_keep = np.stack(C_keep)
    write_images_txt(os.path.join(args.out, "sparse/0/images.txt"), names_out,
                     R_new, C_keep)
    prov = {
        "pipeline": "rotfixed_rotonly_v1",
        "src_work": os.path.abspath(args.src_work),
        "convention": args.convention,
        "gauge_fit": {"n": len(common), "mean_deg": float(res.mean()),
                      "median_deg": float(np.median(res)),
                      "max_deg": float(res.max())},
        "note": "COLMAP translations kept; only rotations replaced, camera "
                "centres preserved",
        "pose_estimation_seconds": time.time() - t_start,
    }
    with open(os.path.join(args.out, "module_a.json"), "w") as f:
        json.dump({"group": "hw", "scene_dir": scene_dir,
                   "pose_source": "rotfixed_rotonly",
                   "num_images": len(names_out), "candidate": prov,
                   "pose_estimation_seconds": time.time() - t_start}, f, indent=2)
    ps = PoseSet("rotonly", {n: (R_new[i], -R_new[i] @ C_keep[i])
                             for i, n in enumerate(names_out)}, None, "rotonly")
    return ps, prov


# ---------------------------------------------------------------------------
# mode: check
# ---------------------------------------------------------------------------


def run_check(args):
    out = check_conventions(args.scene_dir, args.model, image_size=args.image_size)
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# synthetic self-check (CPU, no data)
# ---------------------------------------------------------------------------


def self_check():
    rng = np.random.default_rng(7)
    n_cam, n_pts = 24, 400
    C = np.zeros((n_cam, 3))
    R_wc = np.zeros((n_cam, 3, 3))
    for i in range(n_cam):
        th = 2 * math.pi * i / n_cam
        el = 0.35 * math.sin(3 * th)
        r = 4.0 + 0.6 * math.sin(2 * th)
        C[i] = [r * math.cos(th), 0.8 * el, r * math.sin(th)]
        z = -C[i] / np.linalg.norm(C[i])
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R_wc[i] = np.stack([x, y, z], axis=1).T
    X = rng.normal(size=(n_pts, 3)) * 1.2
    K = {"fx": 700.0, "fy": 700.0, "cx": 360.0, "cy": 480.0,
         "width": 720, "height": 960}
    obs = {}
    for i in range(n_cam):
        uv, z = project_points(X, R_wc[i], C[i], K)
        ok = (z > 0.2) & (uv[:, 0] > 0) & (uv[:, 0] < K["width"]) & \
             (uv[:, 1] > 0) & (uv[:, 1] < K["height"])
        for k in np.where(ok)[0]:
            obs[(i, int(k))] = uv[k] + rng.normal(scale=0.3, size=2)
    pair_dirs, pair_inliers = [], []
    for a in range(n_cam):
        for b in range(a + 1, n_cam):
            ks = [k for k in range(n_pts) if (a, k) in obs and (b, k) in obs]
            if len(ks) < 12:
                continue
            uv_pairs = np.array([(obs[(a, k)], obs[(b, k)]) for k in ks])
            b_i = bearings_from_pixels(uv_pairs[:, 0], K)
            b_j = bearings_from_pixels(uv_pairs[:, 1], K)
            b_i /= np.linalg.norm(b_i, axis=1, keepdims=True)
            b_j /= np.linalg.norm(b_j, axis=1, keepdims=True)
            R_ij = R_wc[b] @ R_wc[a].T
            est = estimate_pair_translation(b_i, b_j, R_ij, 700.0,
                                            seed=a * 97 + b, min_inliers=8)
            if est is None:
                continue
            d = R_wc[b].T @ est["t"]
            d /= np.linalg.norm(d)
            pair_dirs.append((a, b, d, math.sqrt(est["n_inliers"])))
    C_est, ta = translation_averaging(pair_dirs, n_cam, iterations=6)
    s, R, t = umeyama(C_est, C)
    C_fit = s * (C_est @ R.T) + t
    scale = np.median(np.linalg.norm(C - C.mean(0), axis=1))
    err = np.linalg.norm(C_fit - C, axis=1) / scale
    print(f"self-check: TA {ta['residual_median_deg']:.3f} deg median; recovered "
          f"centres: median {100*np.median(err):.2f}% / max {100*err.max():.2f}% "
          f"of scene scale")
    ok = np.median(err) < 0.02 and err.max() < 0.15
    # triangulation + refinement accuracy on the synthetic cloud
    tracks = []
    for k in range(n_pts):
        tr = [(i, k) for i in range(n_cam) if (i, k) in obs]
        if len(tr) >= 3:
            tracks.append(tr)
    id2idx = {i: i for i in range(n_cam)}
    uv_of = lambda img_idx, kidx: obs[(img_idx, kidx)]  # noqa: E731
    track_obs, points, stats = triangulate_tracks(
        tracks, uv_of, R_wc, C_est, K, id2idx, min_angle_deg=1.0)
    # the reconstruction is only known up to a similarity: align it to the truth
    # before measuring point errors (the same s, R, t as the cameras)
    s_fit, R_fit, t_fit = umeyama(C_est, C)

    def aligned_pts(P):
        return s_fit * (P @ R_fit.T) + t_fit

    if len(points) > 0:
        gt = X[np.array([tr[0][1] for tr in track_obs])]
        d_pts = np.linalg.norm(aligned_pts(points) - gt, axis=1)
        print(f"self-check: triangulated {stats['tracks_ok']} points, median error "
              f"{np.median(d_pts):.4f} (aligned)")
        ok = ok and np.median(d_pts) < 0.02
    else:
        ok = False
    # known-rotation BA must not make the camera centres worse
    track_obs2, pts2, R_ba, C_ba, hist = refine_known_rotation(
        track_obs, points, R_wc, C_est, K, iterations=4)
    s2, R2, t2 = umeyama(C_ba, C)
    C_ba_fit = s2 * (C_ba @ R2.T) + t2
    err2 = np.linalg.norm(C_ba_fit - C, axis=1) / scale
    ok = ok and np.median(err2) < 0.02
    if len(pts2) > 0:
        gt2 = X[np.array([tr[0][1] for tr in track_obs2])]
        d2 = np.linalg.norm(s2 * (pts2 @ R2.T) + t2 - gt2, axis=1)
        print(f"self-check: after known-rotation BA {len(pts2)} points; centres "
              f"median {100*np.median(err2):.2f}%, point median {np.median(d2):.4f}")
        ok = ok and np.median(d2) < 0.02
    # S3 path: release the rotations after 2 iterations; the synthetic AR seed is
    # the exact truth here, so the free BA must not diverge from it
    track_obs3, pts3, R_free, C_free, hist3 = refine_known_rotation(
        track_obs, points, R_wc, C_est, K, iterations=4, refine_rotation_from=2,
        verbose=False)
    s3, R3, t3 = umeyama(C_free, C)
    err3 = np.linalg.norm(s3 * (C_free @ R3.T) + t3 - C, axis=1) / scale
    rot_dev = max(rot_angle_deg(R_wc[i], R_free[i]) for i in range(n_cam))
    print(f"self-check: rotation-free BA from iter2: centres median "
          f"{100*np.median(err3):.2f}%, max rot deviation from seed {rot_dev:.3f} deg")
    ok = ok and np.median(err3) < 0.02
    print("SELF-CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description="rotation-known pose pipeline for hw")
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--scene_dir", required=True)
        p.add_argument("--convention", default=DEFAULT_CONVENTION,
                       choices=sorted(ROT_CONVENTIONS))
        p.add_argument("--image_size", type=int, nargs=2, default=None,
                       help="actual (W H) of the jpgs; default: read from disk")
        p.add_argument("--verbose", action="store_true")

    p_chk = sub.add_parser("check")
    common(p_chk)
    p_chk.add_argument("--model", required=True, help="COLMAP sparse model dir")
    p_chk.add_argument("--out", default=None)

    p_rot = sub.add_parser("rotonly")
    common(p_rot)
    p_rot.add_argument("--src_work", required=True, help="existing module-A work dir")
    p_rot.add_argument("--out", required=True)

    p_ta = sub.add_parser("transavg")
    common(p_ta)
    p_ta.add_argument("--db", required=True, help="COLMAP database.db with SIFT matches")
    p_ta.add_argument("--out", required=True)
    p_ta.add_argument("--num_context", type=int, default=16)
    p_ta.add_argument("--llffhold", type=int, default=8)
    p_ta.add_argument("--max_resolution", type=int, default=960)
    p_ta.add_argument("--pair_thresh_px", type=float, default=2.0)
    p_ta.add_argument("--min_pair_inliers", type=int, default=25)
    p_ta.add_argument("--ta_iterations", type=int, default=8)
    p_ta.add_argument("--min_tri_angle", type=float, default=1.0)
    p_ta.add_argument("--ba_iterations", type=int, default=6)
    p_ta.add_argument("--ba-rot-free-from", dest="ba_rot_free_from", type=int,
                      default=None,
                      help="iteration from which camera rotations are optimised "
                           "too (AR rotations are the seed); default: never")

    sub.add_parser("self-check")

    args = ap.parse_args(argv)
    try:
        if args.mode == "check":
            run_check(args)
            return 0
        if args.mode == "rotonly":
            run_rotonly(args)
            return 0
        if args.mode == "transavg":
            run_transavg(args)
            return 0
        if args.mode == "self-check":
            return self_check()
    except Exception:
        traceback.print_exc()
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
