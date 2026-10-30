"""Grow the ReSplat init PLY along texture x distance importance (new module).

This is the 2026-10-31 "texture-aware voxel" experiment.  ``research/ply_ops.py`` is
NOT modified; only its ``load`` / ``save`` / ``sigmoid`` / ``logit`` helpers are reused.

Pipeline applied to one ReSplat world-space PLY *before* module B:

1. **opacity prune** -- two modes:

   * default: drop Gaussians whose ``sigmoid(opacity)`` is below ``--prune-min-opacity``
     (0.01, the floater threshold already used by ``ply_ops.op_prune``);
   * ``--net-flat``: drop the ``N - round(N/(1+grow_frac))`` lowest-opacity primitives
     (16.7% at +20%), so that after the growth the **total count equals the original**.
     This is the arm with **zero module-B TIME penalty**; it lands at opacity
     ~0.05-0.06 on the measured scenes (recorded in the summary JSON).

   ``grow_texdist_merge`` first merges the attributes of every pruned primitive into its
   **nearest surviving neighbour** instead of discarding them:

     * opacity  : ``logit(clip(a_s + sum(a_p), 0, 0.999))``
     * colour/SH: opacity-weighted mean ``(a_s*c_s + sum(a_p*c_p)) / (a_s + sum(a_p))``
     * scale    : per-axis ``max`` of the merged group
     * position/rotation keep the surviving neighbour's values

2. **+grow-frac occupancy top-up** -- add ``round(grow_frac * N_after_prune)`` new
   Gaussians (default +20%) by SPLITTING primitives that live in *under-Gaussian*
   voxels.  Definition used (documented in the report):

     ``count_v`` = surviving Gaussians in voxel ``v`` of a ``--voxels``^3 grid laid over
     the 0.5-99.5% quantile bounding box.  An **occupied** voxel is under-Gaussian when
     ``count_v < mean occupancy of the occupied voxels``; its *deficit* is
     ``ceil(mean) - count_v``, i.e. the number of primitives needed to bring it up to the
     scene's average local density.  Growth is distributed over these voxels
     proportionally to their importance, capped by the deficit (with the cap doubled /
     quadrupled / ... when the deficit alone cannot hold the +20% budget).

3. **voxel importance** -- the ``pi3_sample_points.py`` structure

     ``importance = conf^p * (minimum_weight + texture_weight * texture^gamma)
                    * (1 + center_weight * center) * (1 + support_weight * support)``

   adapted to what a PLY contains:

     * no confidence term and no multi-view support term (not measurable from a PLY);
     * ``texture`` = mean inverse Gaussian scale per voxel (small primitive == high
       frequency detail) -- the proxy already validated by ``ply_ops.op_texresample``;
       mapped to [0, 1] by its empirical percentile rank over occupied voxels so the
       pi3 default ``texture_weight=4.0, texture_gamma=1.25, minimum_weight=0.05``
       transfer directly;
     * pi3's weak *centre* prior is REPLACED by the requested distance-from-scene-centre
       prior: ``d > far_frac * scene_radius`` (default 0.5 * R) gets ``far_weight``
       (default 2x).  ``scene_radius`` = half the diagonal of the same 0.5-99.5%
       bounding box (the bounding-sphere radius of the robust scene box).

Variants (``--op``):

  ``grow_texdist``        texture * distance importance
  ``grow_texdist_merge``  same, but pruned attributes merged into nearest survivor first
  ``grow_distonly``       distance prior only
  ``grow_texonly``        texture only

Example (run in the ``fastgs`` conda env, PYTHONPATH=repo root):

  python -m research.ply_voxgrow --in resplat_world.ply --out grown.ply \
      --op grow_texdist --voxels 128 --grow-frac 0.2 --far-frac 0.5 --far-weight 2.0 \
      --summary-out grown.json
"""
import argparse
import json
import time

import numpy as np

from research.ply_ops import load, logit, save, sigmoid

SCALE_COLS = ("scale_0", "scale_1", "scale_2")
COLOR_COLS = ("f_dc_", "f_rest_")
OPS = ("grow_texdist", "grow_texdist_merge", "grow_distonly", "grow_texonly", "prune")


# ----------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------
def scale_mean(data):
    return np.stack([np.exp(data[c]) for c in SCALE_COLS], axis=1).mean(axis=1)


def scene_frame(xyz, clip=0.005):
    """Robust scene box: 0.5-99.5% quantile bounds, their centre and bounding radius."""
    lo = np.quantile(xyz, clip, axis=0)
    hi = np.quantile(xyz, 1.0 - clip, axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo) / 2.0)   # bounding-sphere radius of the box
    return center, radius, lo, hi


def voxel_index(xyz, lo, hi, voxels):
    cell = np.maximum(hi - lo, 1e-9) / float(voxels)
    idx = np.clip(((xyz - lo) / cell).astype(np.int64), 0, voxels - 1)
    keys = (idx[:, 0] * voxels + idx[:, 1]) * voxels + idx[:, 2]
    return keys


def unique_voxels(keys, n_vox):
    """Return (unique_keys, inverse) where inverse maps each point -> 0..n_unique-1."""
    uk, inv = np.unique(keys, return_inverse=True)
    return uk, inv, len(uk)


# ----------------------------------------------------------------------------------
# nearest-neighbour merge of pruned primitives (grow_texdist_merge)
# ----------------------------------------------------------------------------------
def nearest_survivor(xyz_s, xyz_p):
    """Index into xyz_s of the nearest survivor for every pruned point."""
    try:
        from scipy.spatial import cKDTree

        return cKDTree(xyz_s).query(xyz_p, k=1)[1]
    except Exception:
        # dependency-free fallback: voxel hash of the survivors + 27-cell search that
        # expands its radius until a cell is occupied (the PLY has no empty region there)
        size = float(np.median(np.linalg.norm(xyz_s - xyz_s.mean(0), axis=1)) or 0.0)
        if not size:
            size = 1.0
        size /= 8.0
        lo = np.minimum(xyz_s.min(0), xyz_p.min(0)).astype(np.float64)
        cells = np.floor((xyz_s - lo) / size).astype(np.int64)
        table = {}
        for i, c in enumerate(map(tuple, cells)):
            table.setdefault(c, []).append(i)
        out = np.empty(len(xyz_p), dtype=np.int64)
        pc = np.floor((xyz_p - lo) / size).astype(np.int64)
        for j, (q, cell) in enumerate(zip(xyz_p, pc)):
            best, best_d = -1, np.inf
            for r in range(0, 6):
                found = False
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        for dz in range(-r, r + 1):
                            if max(abs(dx), abs(dy), abs(dz)) != r:
                                continue
                            for cand in table.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()):
                                d = float(np.sum((xyz_s[cand] - q) ** 2))
                                if d < best_d:
                                    best_d, best = d, cand
                                    found = True
                if found and best >= 0:
                    break
            if best < 0:
                best = int(np.argmin(np.sum((xyz_s - q) ** 2, axis=1)))
            out[j] = best
        return out


def merge_pruned_into_survivors(data, keep):
    """Merge every pruned primitive into its nearest surviving neighbour.

    Returns the survivor-only attributes with the merged contributions baked in.
    """
    pruned = ~keep
    n_s, n_p = int(keep.sum()), int(pruned.sum())
    out = {k: np.asarray(v[keep]).copy() for k, v in data.items()}
    if n_p == 0 or n_s == 0:
        return out, dict(merged=0)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    a = sigmoid(data["opacity"]).astype(np.float64)
    idx = nearest_survivor(xyz[keep], xyz[pruned])
    a_s, a_p = a[keep], a[pruned]
    # grouped (reduceat) accumulation: np.add.at is unbuffered and slow on 1e5+ rows
    order = np.argsort(idx, kind="stable")
    idx_sorted = idx[order]
    a_sorted = a_p[order]
    uniq, starts = np.unique(idx_sorted, return_index=True)
    w_sorted = np.add.reduceat(a_sorted, starts)
    w = np.zeros(n_s, dtype=np.float64)
    w[uniq] = w_sorted
    den = np.maximum(a_s + w, 1e-12)
    for c in (c for c in data if c.startswith(COLOR_COLS)):
        num = np.zeros(n_s, dtype=np.float64)
        num[uniq] = np.add.reduceat(
            a_sorted * data[c][pruned][order].astype(np.float64), starts)
        merged = (a_s * data[c][keep].astype(np.float64) + num) / den
        out[c] = merged.astype(np.float32)
    out["opacity"] = logit(np.clip(a_s + w, 0.0, 0.999)).astype(np.float32)
    for c in SCALE_COLS:
        s = np.exp(out[c].astype(np.float64))
        smax = np.zeros(n_s, dtype=np.float64)
        smax[uniq] = np.maximum.reduceat(np.exp(data[c][pruned][order].astype(np.float64)), starts)
        out[c] = np.log(np.maximum(s, smax)).astype(np.float32)
    return out, dict(merged=int(n_p), merged_into=int(len(uniq)))


# ----------------------------------------------------------------------------------
# camera trajectory (for --dist-mode traj)
# ----------------------------------------------------------------------------------
def load_camera_centers(path):
    """Camera centres from a COLMAP images.txt (world2cam lines).

    ``images.txt`` lines alternate: an image line with 9 header values (id qw qx qy qz
    tx ty tz cam_id name) followed by a POINTS2D line.  The centre is ``C = -R^T t``.
    """
    if not path:
        raise SystemExit("--dist-mode traj needs --cameras <work>/sparse/0/images.txt")
    centers = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            f = line.split()
            if len(f) != 10:
                continue
            try:
                qw, qx, qy, qz, tx, ty, tz = (float(x) for x in f[1:8])
            except ValueError:
                continue
            n = qw * qw + qx * qx + qy * qy + qz * qz
            if n < 1e-12:
                continue
            s = 2.0 / n
            R = np.array([
                [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw), s * (qx * qz + qy * qw)],
                [s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
                [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw), 1 - s * (qx * qx + qy * qy)],
            ], dtype=np.float64)
            centers.append(-R.T @ np.array([tx, ty, tz], dtype=np.float64))
    if not centers:
        raise SystemExit(f"no camera centres parsed from {path}")
    return np.stack(centers, axis=0)


# ----------------------------------------------------------------------------------
# voxel importance
# ----------------------------------------------------------------------------------
def voxel_importance(xyz, scale, center, radius, args, op, lo, hi, cam_centers=None,
                     hole_points=None, error_points=None):
    """Per-voxel weight, occupancy statistics and the membership index."""
    keys = voxel_index(xyz, lo, hi, args.voxels)
    uk, inv, n_vox = unique_voxels(keys, args.voxels ** 3)
    counts = np.bincount(inv, minlength=n_vox).astype(np.float64)
    tex = 1.0 / np.maximum(scale, 1e-9)                       # small primitive -> texture
    tex_mean = np.bincount(inv, weights=tex, minlength=n_vox) / np.maximum(counts, 1)
    rank = np.empty(n_vox, dtype=np.float64)                  # empirical percentile [0,1]
    rank[np.argsort(tex_mean)] = np.linspace(0.0, 1.0, n_vox) if n_vox > 1 else 0.5
    cxyz = np.stack([np.bincount(inv, weights=xyz[:, i], minlength=n_vox)
                     for i in range(3)], axis=1) / np.maximum(counts, 1)[:, None]
    d_center = np.linalg.norm(cxyz - center, axis=1)
    mode = getattr(args, "dist_mode", "hard")
    if mode == "holes":
        # hole-targeted growth: weight = number of 3D hole points (from train-view
        # alpha<0.5 pixels unprojected with EDT-filled depth) inside the voxel
        if hole_points is None or not len(hole_points):
            raise SystemExit("--dist-mode holes needs --hole-points <npy>")
        hk = voxel_index(np.asarray(hole_points, dtype=np.float64), lo, hi, args.voxels)
        huk = np.unique(hk)
        pos = np.searchsorted(uk, huk)
        pos = np.clip(pos, 0, n_vox - 1)
        hcount = np.zeros(n_vox, dtype=np.float64)
        np.add.at(hcount, pos[uk[pos] == huk], 1.0)
        dist = hcount                      # interpretability only: hole density
        dist_ref = "3D hole-point density (train-view alpha<0.5, EDT depth)"
        far_threshold = 0.0
        far = hcount > 0
        dist_prior = hcount
        dist_median = float(np.median(hcount[hcount > 0])) if (hcount > 0).any() else 0.0
    elif mode == "errdensity":
        # error-density-targeted growth: weight = summed rendered squared error of the
        # TRAIN-view pixels whose unprojected 3D position falls in the voxel
        if error_points is None or not len(error_points):
            raise SystemExit("--dist-mode errdensity needs --error-points <npy>")
        ep = np.asarray(error_points, dtype=np.float64)
        ek = voxel_index(ep[:, :3], lo, hi, args.voxels)
        euk, einv = np.unique(ek, return_inverse=True)
        ewt = np.bincount(einv, weights=ep[:, 3], minlength=len(euk))
        pos = np.clip(np.searchsorted(uk, euk), 0, n_vox - 1)
        keepmask = uk[pos] == euk
        errw = np.zeros(n_vox, dtype=np.float64)
        errw[pos[keepmask]] = ewt[keepmask]
        dist = errw
        dist_ref = "TRAIN-view rendered squared-error density (3D unprojected)"
        far_threshold = 0.0
        far = errw > 0
        dist_prior = errw
        dist_median = float(np.median(errw[errw > 0])) if (errw > 0).any() else 0.0
    elif mode == "traj":
        if cam_centers is None or not len(cam_centers):
            raise SystemExit("--dist-mode traj needs --cameras <work>/sparse/0/images.txt")
        # nearest camera centre per voxel (a few thousand voxels x ~200 cameras)
        dist = np.empty(n_vox, dtype=np.float64)
        for i0 in range(0, n_vox, 2048):
            blk = cxyz[i0:i0 + 2048]
            dist[i0:i0 + 2048] = np.sqrt(
                ((blk[:, None, :] - cam_centers[None, :, :]) ** 2).sum(-1)).min(axis=1)
        dist_ref = "nearest camera centre (trajectory)"
    else:
        dist = d_center
        dist_ref = "scene centre"
    if mode == "traj":
        # "median distance as the reference" (task-9 G6): far = beyond
        # (1 + far_frac) x the median distance from a Gaussian to the camera orbit
        d_pts = dist[inv]
        dist_median = float(np.median(d_pts))
        far_threshold = (1.0 + float(args.far_frac)) * dist_median
    else:
        dist_median = None
        far_threshold = float(args.far_frac) * float(radius)
    far = dist > far_threshold
    if mode in ("holes", "errdensity"):
        # the "distance" IS the per-voxel target weight (hole density / error density);
        # it must not be overwritten by the threshold prior below
        dist_prior = dist
    elif mode == "ramp2":
        # continuous (d/R)^2 ramp: 1x at the centre -> far_weight at d >= R
        dist_prior = 1.0 + (float(args.far_weight) - 1.0) * np.clip(
            dist / max(float(radius), 1e-9), 0.0, 1.0) ** 2
    elif mode == "faronly":
        # growth budget only inside the far region, uniform within it
        dist_prior = far.astype(np.float64)
    elif mode == "inneronly":
        # sign test suggested by the train-view diagnostic: the init error sits in the
        # INNER region, so grow only inside d <= far_frac*R (uniform within it)
        dist_prior = (~far).astype(np.float64)
    else:                                   # "hard" and "traj" use the hard threshold
        dist_prior = np.where(far, float(args.far_weight), 1.0)
    tex_term = args.minimum_weight + args.texture_weight * np.power(rank, args.texture_gamma)
    if op in ("grow_distonly", "prune"):
        weight = dist_prior
    elif op == "grow_texonly":
        weight = tex_term
    else:
        weight = tex_term * dist_prior
    return dict(keys=uk, inv=inv, n_vox=n_vox, counts=counts, tex_mean=tex_mean,
                rank=rank, dist=dist, d_center=d_center, far=far,
                weight=weight.astype(np.float64), far_threshold=far_threshold,
                dist_ref=dist_ref, dist_mode=mode, dist_median=dist_median)


# ----------------------------------------------------------------------------------
# allocation of the growth budget over the under-Gaussian voxels
# ----------------------------------------------------------------------------------
UNDER_MODES = ("mean", "le1", "p10", "p25", "none")


def under_gaussian_plan(counts, mode="mean"):
    """Under-Gaussian mask + deficit (how many primitives each voxel is short of).

    ``mean`` (task-7 default): count < mean occupancy of the occupied voxels -- at coarse
    voxels this selects 87-96% of the voxels, i.e. it is almost non-discriminative.
    ``le1``  : count <= 1 (the strictly under-covered voxels).  Its deficit is 1 per
               voxel, so the +grow budget auto-expands the per-voxel cap when needed.
    ``p10``/``p25`` : count <= the 10th/25th percentile of the occupied-voxel counts.
    ``none`` : every occupied voxel is eligible (the far-region-only uniform arm).
    """
    occupied = counts >= 1
    if not occupied.any():
        return occupied, np.zeros(len(counts), dtype=np.int64), dict(
            mean_occupancy=0.0, median_occupancy=0.0, occupied_voxels=0,
            under_voxels=0, under_mode=mode)
    vals = counts[occupied]
    mean_occ, median_occ = float(vals.mean()), float(np.median(vals))
    if mode == "mean":
        under = occupied & (counts < mean_occ)
        if not under.any():
            under = occupied.copy()
        deficit = np.maximum(np.ceil(mean_occ) - counts, 1.0)
    elif mode in ("p10", "p25"):
        thresh = float(np.quantile(vals, 0.10 if mode == "p10" else 0.25))
        under = occupied & (counts <= thresh)
        deficit = np.maximum(np.ceil(thresh) - counts, 1.0)
    elif mode == "le1":
        under = occupied & (counts <= 1)
        deficit = np.ones(len(counts), dtype=np.float64)
    elif mode == "none":
        under = occupied.copy()
        deficit = np.ones(len(counts), dtype=np.float64)
    else:
        raise SystemExit(f"unknown --under-mode {mode}")
    deficit[~under] = 0.0
    return under, deficit.astype(np.int64), dict(
        mean_occupancy=mean_occ, median_occupancy=median_occ,
        occupied_voxels=int(occupied.sum()), under_voxels=int(under.sum()),
        under_mode=mode)


def allocate(weights, capacity, total):
    """Largest-remainder integer allocation proportional to `weights`, capped by `capacity`."""
    n = len(weights)
    alloc = np.zeros(n, dtype=np.int64)
    if total <= 0 or n == 0 or not (capacity > 0).any():
        return alloc
    remaining = int(total)
    for _ in range(256):
        if remaining <= 0:
            break
        room = capacity - alloc
        live = room > 0
        if not live.any():
            break
        ww = np.where(live, weights, 0.0)
        s = ww.sum()
        if not np.isfinite(s) or s <= 0:
            ww = live.astype(np.float64)
            s = ww.sum()
        share = remaining * ww / s
        add = np.minimum(np.floor(share).astype(np.int64), room)
        before = int(alloc.sum())
        alloc += add
        remaining = int(total - alloc.sum())
        if remaining > 0:
            room = capacity - alloc
            # only top up voxels that actually carry weight: zero-weight voxels (e.g. the
            # near region under --dist-mode faronly) must never receive children
            frac = np.where((room > 0) & (ww > 0), share - np.floor(share), -2.0)
            order = np.argsort(-frac)
            take = order[: remaining]
            take = take[(room[take] > 0) & (ww[take] > 0)]
            alloc[take] += 1
            remaining = int(total - alloc.sum())
        if int(alloc.sum()) == before and remaining > 0:
            break                                     # capacity exhausted
    return alloc


# ----------------------------------------------------------------------------------
# main transform
# ----------------------------------------------------------------------------------
def grow(data, args):
    t0 = time.time()
    n_in = int(len(data["x"]))
    opacity = sigmoid(data["opacity"]).astype(np.float64)
    if getattr(args, "no_prune", False):
        keep = np.ones(n_in, dtype=bool)
        prune_mode = "none (grow-only, --no-prune)"
    elif args.net_flat:
        # prune exactly the count that +grow_frac growth gives back: total == n_in, i.e.
        # zero module-B TIME penalty.  The lowest-opacity primitives are the ones dropped.
        n_after = int(round(n_in / (1.0 + float(args.grow_frac))))
        n_after = min(max(n_after, 1), n_in)
        keep = np.zeros(n_in, dtype=bool)
        keep[np.argsort(opacity, kind="stable")[n_in - n_after:]] = True
        prune_mode = "net-flat (lowest-opacity count = N - round(N/(1+grow_frac)))"
    else:
        keep = opacity >= float(args.prune_min_opacity)
        prune_mode = "opacity threshold (--prune-min-opacity)"
    n_pruned = int((~keep).sum())
    info = dict(op=args.op, net_flat=bool(args.net_flat), prune_mode=prune_mode,
                n_in=n_in, prune_min_opacity=float(args.prune_min_opacity),
                pruned_opacity_ceiling=float(opacity[~keep].max()) if n_pruned else None,
                kept_opacity_floor=float(opacity[keep].min()) if keep.any() else None,
                n_pruned=n_pruned, pruned_frac=n_pruned / max(n_in, 1),
                n_after_prune=int(keep.sum()))
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    center, radius, lo, hi = scene_frame(xyz[keep] if keep.any() else xyz)
    info.update(scene_center=[float(v) for v in center], scene_radius=radius)

    if args.op == "grow_texdist_merge":
        merged, minfo = merge_pruned_into_survivors(data, keep)
        info.update(minfo)
        base = merged
    else:
        base = {k: np.asarray(v[keep]) for k, v in data.items()}
    if args.op == "prune":
        # diagnosis arm: the same prune as net-flat but WITHOUT any growth, so the
        # quality delta of prune-alone vs prune+grow can be separated.
        info.update(n_out=int(len(base["x"])),
                    net_growth=(int(len(base["x"])) - n_in) / max(n_in, 1),
                    seconds=round(time.time() - t0, 2))
        return base, info

    n_base = int(len(base["x"]))
    xyz_b = np.stack([base["x"], base["y"], base["z"]], axis=1)
    scale_b = scale_mean(base)
    error_points = None
    if getattr(args, "dist_mode", "hard") == "errdensity":
        ep_path = getattr(args, "error_points", None)
        if not ep_path:
            raise SystemExit("--dist-mode errdensity needs --error-points <npy>")
        error_points = np.load(ep_path)
        info["error_points"] = ep_path
        info["n_error_points"] = int(len(error_points))
    hole_points = None
    if getattr(args, "dist_mode", "hard") == "holes":
        hp = getattr(args, "hole_points", None)
        if not hp:
            raise SystemExit("--dist-mode holes needs --hole-points <npy>")
        hole_points = np.load(hp)
        info["hole_points"] = hp
        info["n_hole_points"] = int(len(hole_points))
    cam_centers = None
    if getattr(args, "dist_mode", "hard") == "traj":
        cam_centers = load_camera_centers(args.cameras)
        info["cameras"] = args.cameras
        info["n_cameras"] = int(len(cam_centers))
        info["camera_center_radius"] = [float(v) for v in cam_centers.mean(0)]
    plan = voxel_importance(xyz_b, scale_b, center, radius, args, args.op, lo, hi,
                            cam_centers=cam_centers, hole_points=hole_points,
                            error_points=error_points)
    under, deficit, oinfo = under_gaussian_plan(plan["counts"], getattr(args, "under_mode", "mean"))
    info.update(oinfo)
    info.update(voxels=int(args.voxels), far_frac=float(args.far_frac),
                far_weight=float(args.far_weight), far_threshold=plan["far_threshold"])
    # fraction of surviving *Gaussians* whose voxel is beyond the far threshold
    # (the importance acts per voxel, so every primitive of a far voxel inherits it)
    info["far_gaussian_frac"] = float(plan["far"][plan["inv"]].mean())
    info["far_voxel_frac"] = float(plan["far"].mean())

    budget = int(round(float(args.grow_frac) * n_base))
    cap_scale = max(1, int(getattr(args, "cap_scale", 1)))
    capacity = deficit.copy()
    eligible = under & (plan["weight"] > 0)        # zero-weight voxels cannot host growth
    mult = 1
    while int((deficit[eligible] * mult * cap_scale).sum()) < budget and mult < 4096:
        mult *= 2
    capacity = deficit * mult * cap_scale
    info["capacity_mult"] = int(mult)
    info["cap_scale"] = int(cap_scale)
    info["budget"] = budget
    alloc = allocate(plan["weight"], capacity, budget)
    info["n_children"] = int(alloc.sum())
    info["voxels_with_children"] = int((alloc > 0).sum())

    # ---- vectorised child construction: one random member of the voxel per child ----
    order = np.argsort(plan["inv"], kind="stable")
    starts = np.zeros(plan["n_vox"], dtype=np.int64)
    starts[1:] = np.cumsum(plan["counts"].astype(np.int64))[:-1]
    rng = np.random.default_rng(int(args.seed))
    vox_children = np.repeat(np.arange(plan["n_vox"]), alloc)
    n_children = len(vox_children)
    if n_children:
        members_count = plan["counts"].astype(np.int64)[vox_children]
        off = rng.integers(0, members_count)
        src = order[starts[vox_children] + off]
    else:
        src = np.empty(0, dtype=np.int64)
    info["n_children"] = int(n_children)
    if n_children:
        info["children_per_voxel_mean"] = float(alloc[alloc > 0].mean())
        info["children_per_voxel_max"] = int(alloc.max())
    else:
        info["children_per_voxel_mean"] = 0.0
        info["children_per_voxel_max"] = 0

    out = {}
    for k, v in base.items():
        v = np.asarray(v)
        if k in ("x", "y", "z"):
            child = v[src]
        elif k in SCALE_COLS:
            child = v[src] - np.log(float(args.shrink))
        else:
            child = v[src]
        out[k] = np.concatenate([v, child]) if n_children else v
    if n_children:
        sigma = scale_b[src]
        jit = rng.normal(0.0, float(args.jitter), (n_children, 3)) * sigma[:, None]
        for i, ax in enumerate(("x", "y", "z")):
            out[ax] = out[ax].astype(np.float64)
            out[ax][n_base:] += jit[:, i]
            out[ax] = out[ax].astype(np.float32)

    # ---- where did the new primitives land (SCENE-CENTRE distance histogram, task-9) ----
    # always measured against the scene-centre distance so every arm is comparable,
    # even when the growth weight itself used the camera-trajectory distance
    d_pts = plan["d_center"][plan["inv"]]
    m_pts = plan["dist"][plan["inv"]]
    d_ch = d_pts[src] if n_children else np.empty(0)
    m_ch = m_pts[src] if n_children else np.empty(0)
    landing = dict(dist_mode=plan["dist_mode"], dist_ref=plan["dist_ref"],
                   mean_dR_all=float(d_pts.mean() / max(radius, 1e-9)),
                   mean_dR_children=(float(d_ch.mean() / max(radius, 1e-9))
                                     if n_children else None),
                   share_all_metric_far=float(np.mean(m_pts > plan["far_threshold"])),
                   share_children_metric_far=(float(np.mean(m_ch > plan["far_threshold"]))
                                              if n_children else None))
    for t in (0.25, 0.5, 0.75):
        landing[f"share_all_gt{t}R"] = float(np.mean(d_pts > t * radius))
        landing[f"share_children_gt{t}R"] = (float(np.mean(d_ch > t * radius))
                                             if n_children else None)
    # error-density arms: where do the children sit relative to the error distribution?
    if plan["dist_mode"] == "errdensity":
        errw = plan["weight"]
        npos = int((errw > 0).sum())
        if npos:
            # RANK-based top decile (a value threshold degenerates when the error mass is
            # extremely skewed: quantile(0.9) collapses to ~min and "top decile" = all)
            k = max(1, int(round(0.10 * npos)))
            top = np.zeros(len(errw), dtype=bool)
            top[np.argsort(-errw)[:k]] = True
            child_errw = errw[plan["inv"][src]] if n_children else np.empty(0)
            all_errw = errw[plan["inv"]]
            landing["top_error_decile_voxels"] = int(k)
            landing["share_all_top_error_decile"] = float(top[plan["inv"]].mean())
            landing["share_children_top_error_decile"] = (
                float(top[plan["inv"]][src].mean()) if n_children else None)
            landing["mean_err_weight_all"] = float(all_errw.mean())
            landing["mean_err_weight_children"] = (
                float(child_errw.mean()) if n_children else None)
    info["landing"] = landing

    info.update(n_out=int(len(out["x"])), net_growth=(int(len(out["x"])) - n_in) / max(n_in, 1),
                seconds=round(time.time() - t0, 2))
    return out, info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inputs", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--op", required=True, choices=list(OPS))
    p.add_argument("--prune-min-opacity", type=float, default=0.01, dest="prune_min_opacity")
    p.add_argument("--no-prune", action="store_true", dest="no_prune",
                   help="grow-only: keep every Gaussian (no opacity prune)")
    p.add_argument("--under-mode", default="mean", dest="under_mode", choices=list(UNDER_MODES),
                   help="under-Gaussian voxel criterion (mean=task-7 default)")
    p.add_argument("--dist-mode", default="hard", dest="dist_mode",
                   choices=["hard", "ramp2", "traj", "faronly", "inneronly", "holes",
                            "errdensity"],
                   help="distance prior: hard threshold / (d/R)^2 ramp / camera-trajectory "
                        "distance / far-region-only uniform")
    p.add_argument("--error-points", default=None, dest="error_points",
                   help=".npy (M,4) TRAIN-view covered-pixel points (x,y,z,squared error) "
                        "from research.init_error_diag (--dist-mode errdensity)")
    p.add_argument("--hole-points", default=None, dest="hole_points",
                   help=".npy (M,3) 3D hole points from research.init_error_diag "
                        "(--dist-mode holes)")
    p.add_argument("--cameras", default=None,
                   help="COLMAP images.txt with the train camera poses (for --dist-mode traj)")
    p.add_argument("--net-flat", action="store_true", dest="net_flat",
                   help="prune N - round(N/(1+grow_frac)) lowest-opacity primitives so that "
                        "after the +grow_frac split the TOTAL count is unchanged (no TIME cost)")
    p.add_argument("--grow-frac", type=float, default=0.2, dest="grow_frac")
    p.add_argument("--cap-scale", type=int, default=1, dest="cap_scale",
                   help="multiply the per-voxel growth cap after the auto expansion; >1 lets "
                        "the importance weight (instead of the cap) decide the distribution")
    p.add_argument("--voxels", type=int, default=32)
    p.add_argument("--far-frac", type=float, default=0.5, dest="far_frac")
    p.add_argument("--far-weight", type=float, default=2.0, dest="far_weight")
    p.add_argument("--texture-weight", type=float, default=4.0, dest="texture_weight")
    p.add_argument("--texture-gamma", type=float, default=1.25, dest="texture_gamma")
    p.add_argument("--minimum-weight", type=float, default=0.05, dest="minimum_weight")
    p.add_argument("--shrink", type=float, default=1.6)
    p.add_argument("--jitter", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary-out", default=None, dest="summary_out")
    args = p.parse_args(argv)

    data, names = load(args.inputs[0])
    out, info = grow(data, args)
    save(args.out, out, names)
    info["input"] = args.inputs[0]
    info["output"] = args.out
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
