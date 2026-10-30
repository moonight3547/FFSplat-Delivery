"""TRAIN-view coverage / error diagnostic for a ReSplat init PLY (bicycle study).

Question this answers (task-9): is the init PLY's photometric error concentrated in
*uncovered (hole)* pixels or in the *periphery*?  Two outputs:

1. **Robust coverage statistics.** Each sampled TRAIN view is rendered twice with the
   pinned FastGS rasteriser, once with a white and once with a black background, so the
   accumulated alpha is exact: ``alpha = 1 - mean(img_white - img_black)``.  (An earlier
   version used "no Gaussian centre lands in this pixel" as a coverage proxy; that is
   wrong because bicycle Gaussians project to ~10-15 px, so between-centre pixels are not
   holes.  This version reports alpha-based numbers.)
   Reported: pixel share with alpha<0.5 / alpha<0.1 and their share of squared error; the
   error-vs-radius table (0.5-99.5% bbox centre / radius, same definition as
   ``research/ply_voxgrow.py``) computed on alpha>=0.5 pixels only; and
   ``error_share(>0.5R) / gauss_share(>0.5R)``.

2. **3D hole point cloud** for hole-targeted growth (``--out-holes``): pixels with
   alpha<0.5 whose nearest alpha>=0.5 pixel is within ``--max-hole-px`` are unprojected
   with that nearest-covered depth (per-pixel centre z-buffer + EDT fill).  Sky/outside
   pixels (no covered pixel nearby) are counted but not emitted.  ``research/ply_voxgrow
   --dist-mode holes --hole-points <npy>`` then weights voxels by their hole-point count.

TRAIN VIEWS ONLY.  Diagnostic / input for a module-B design experiment; never used to
select per-scene settings, tune the recipe, or score a submission.

  python -m research.init_error_diag --source outputs/.../work \
      --ply .../resplat_world.ply --views 6 --out diag.json --out-holes holes.npy
"""
import argparse
import json
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "competition/upstream/baseline/fastgs"))

BINS = (0.0, 0.25, 0.5, 0.75, 1.0, np.inf)


def scene_frame(xyz, clip=0.005):
    lo = np.quantile(xyz, clip, axis=0)
    hi = np.quantile(xyz, 1.0 - clip, axis=0)
    return (lo + hi) / 2.0, float(np.linalg.norm(hi - lo) / 2.0)


def nearest_covered(mask):
    """For every pixel, the nearest True pixel in `mask` (scipy EDT)."""
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(~mask, return_distances=True, return_indices=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="prepared scene dir (module-A output)")
    p.add_argument("--ply", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--out-holes", default=None, help="save hole point cloud (.npy)")
    p.add_argument("--out-error-points", default=None, dest="out_error_points",
                   help="save covered-pixel points (M,4: x,y,z,squared_error) for "
                        "error-density-targeted growth (--dist-mode errdensity)")
    p.add_argument("--views", type=int, default=6)
    p.add_argument("--mult", type=float, default=1.0)
    p.add_argument("--alpha-covered", type=float, default=0.5, dest="alpha_covered")
    p.add_argument("--alpha-hole", type=float, default=0.5, dest="alpha_hole")
    p.add_argument("--max-hole-px", type=float, default=48.0, dest="max_hole_px")
    args = p.parse_args(argv)

    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import render_fastgs
    from scene import GaussianModel, Scene
    from utils.general_utils import safe_state

    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    known, _ = parser.parse_known_args(["-s", args.source, "--eval", "-r", "1", "--quiet"])
    lp_args = lp.extract(known)
    pipe = pp.extract(known)
    lp_args.model_path = tempfile.mkdtemp(prefix="init-diag-")
    safe_state(True)
    gm = GaussianModel(3, "default")
    scene = Scene(lp_args, gm)
    cams = scene.getTrainCameras()          # train split ONLY -- there is no code path
    idx = np.unique(np.linspace(0, len(cams) - 1, args.views).astype(int))
    # hard guard: the sampled cameras must be train-split cameras, never test cameras
    stem = lambda n: Path(n).stem
    split_path = Path(args.source) / "split.json"
    split = json.loads(split_path.read_text()) if split_path.is_file() else None
    cam_names = [cams[int(i)].image_name for i in idx]
    if split:
        train_stems = {stem(n) for n in split.get("train", [])}
        test_stems = {stem(n) for n in split.get("test", [])}
        bad = [n for n in cam_names if n in test_stems]
        if bad:
            raise SystemExit(f"ABORT: diagnostic sampled TEST cameras: {bad}")
        if train_stems:
            missing = [n for n in cam_names if n not in train_stems]
            if missing:
                raise SystemExit(f"ABORT: sampled cameras not in split.train: {missing}")
    model = GaussianModel(3, "default")
    model.load_ply(args.ply)
    bg_white = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    bg_black = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    xyz = model.get_xyz.detach()
    xyz_np = xyz.cpu().numpy()
    center, radius = scene_frame(xyz_np)
    r_gauss = np.linalg.norm(xyz_np - center, axis=1) / radius
    gauss_hist = np.histogram(r_gauss, bins=BINS)[0].astype(np.float64)
    gauss_hist /= gauss_hist.sum()

    err_sum = np.zeros(len(BINS) - 1)
    pix_count = np.zeros(len(BINS) - 1)
    total_err, total_pix = 0.0, 0.0
    hole_err, hole_pix, sky_err, sky_pix = 0.0, 0.0, 0.0, 0.0
    a05_err, a05_pix, a01_err, a01_pix = 0.0, 0.0, 0.0, 0.0
    hole_pts = []
    error_pts = []
    per_view = []
    for i in idx:
        cam = cams[int(i)]
        w, h = int(cam.image_width), int(cam.image_height)
        with torch.no_grad():
            out_w = render_fastgs(cam, model, pipe, bg_white, args.mult, 1.0)
            out_b = render_fastgs(cam, model, pipe, bg_black, args.mult, 1.0)
            img_w = out_w["render"].clamp(0, 1)
            img_b = out_b["render"].clamp(0, 1)
            alpha = (1.0 - (img_w - img_b).mean(0)).clamp(0, 1).cpu().numpy()
            rgb = img_w
            gt = cam.original_image.clamp(0, 1)
            err = ((rgb - gt) ** 2).mean(0).cpu().numpy()
            # camera-space projection of every Gaussian centre -> nearest-centre z-buffer
            R = cam.world_view_transform[:3, :3].T
            t = cam.world_view_transform[3, :3]
            cam_xyz = xyz @ R.T + t
            z = cam_xyz[:, 2]
            fx = 0.5 * w / np.tan(cam.FoVx * 0.5)
            fy = 0.5 * h / np.tan(cam.FoVy * 0.5)
            u = (cam_xyz[:, 0] / z.clamp_min(1e-6) * fx + w / 2).round().long()
            v = (cam_xyz[:, 1] / z.clamp_min(1e-6) * fy + h / 2).round().long()
            ok = (z > 1e-4) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            zbuf = torch.full((w * h,), float("inf"), device="cuda")
            zbuf.scatter_reduce_(0, v[ok] * w + u[ok], z[ok], reduce="amin",
                                 include_self=True)
            zbuf = zbuf.reshape(h, w).cpu().numpy()
            Rnp = R.T.detach().cpu().numpy()
            tnp = t.detach().cpu().numpy()
        covered = np.isfinite(zbuf)
        dist_cov, nn = nearest_covered(covered)
        depth_fill = zbuf[nn[0], nn[1]]
        depth_fill = np.where(np.isfinite(depth_fill), depth_fill, 0.0)

        total_err += float(err.sum())
        total_pix += float(err.size)
        is_hole = alpha < args.alpha_hole
        near_hole = is_hole & (dist_cov <= args.max_hole_px)
        sky = is_hole & (dist_cov > args.max_hole_px)
        a05 = alpha < args.alpha_covered
        a01 = alpha < 0.1
        a05_pix += float(a05.sum())
        a05_err += float(err[a05].sum())
        a01_pix += float(a01.sum())
        a01_err += float(err[a01].sum())
        hole_err += float(err[near_hole].sum())
        hole_pix += float(near_hole.sum())
        sky_err += float(err[sky].sum())
        sky_pix += float(sky.sum())

        # error-vs-radius on alpha-covered pixels only (EDT-filled depth)
        cov = alpha >= args.alpha_covered
        ys, xs = np.nonzero(cov)
        d = depth_fill[ys, xs]
        keep = d > 1e-6
        ys, xs, d = ys[keep], xs[keep], d[keep]
        p_cam = np.stack([(xs - w / 2) / fx * d, (ys - h / 2) / fy * d, d], axis=1)
        p_world = (p_cam - tnp[None, :]) @ Rnp
        r = np.linalg.norm(p_world - center, axis=1) / radius
        err_sum += np.histogram(r, bins=BINS, weights=err[ys, xs])[0]
        pix_count += np.histogram(r, bins=BINS)[0]
        if args.out_error_points:
            error_pts.append(np.concatenate(
                [p_world.astype(np.float32), err[ys, xs].astype(np.float32)[:, None]],
                axis=1))

        # 3D hole points (near-covered holes only)
        hy, hx = np.nonzero(near_hole)
        hd = depth_fill[hy, hx]
        okp = hd > 1e-6
        hy, hx, hd = hy[okp], hx[okp], hd[okp]
        if len(hd):
            p_cam = np.stack([(hx - w / 2) / fx * hd, (hy - h / 2) / fy * hd, hd], axis=1)
            hole_pts.append(((p_cam - tnp[None, :]) @ Rnp).astype(np.float32))
        per_view.append(dict(view=int(i), alpha_mean=float(alpha.mean()),
                             hole_px_frac=float(near_hole.mean()),
                             sky_px_frac=float(sky.mean()),
                             hole_err_frac=float(err[near_hole].sum() / max(err.sum(), 1e-12))))

    denom = max(err_sum.sum(), 1e-12)
    covered = max(pix_count.sum(), 1)
    res = dict(
        ply=str(args.ply), views=[int(i) for i in idx], n_train=len(cams),
        renderer="fastgs", coverage="alpha from white/black dual render",
        depth="per-pixel nearest-centre z-buffer + EDT fill",
        scene_center=[float(v) for v in center], scene_radius=radius,
        bins=[float(b) for b in BINS],
        gauss_share=[float(v) for v in gauss_hist],
        error_share_covered=[float(v / denom) for v in err_sum],
        pixel_share_covered=[float(v / covered) for v in pix_count],
        mean_sq_error_per_pixel=[float(e / max(c, 1)) for e, c in zip(err_sum, pix_count)],
        gauss_share_gt_half=float(gauss_hist[2:].sum()),
        error_share_gt_half=float((err_sum[2:] / denom).sum()),
        pixel_share_gt_half=float((pix_count[2:] / covered).sum()),
        alpha_mean=float(np.mean([v["alpha_mean"] for v in per_view])),
        hole_pixel_frac=float(hole_pix / max(total_pix, 1)),
        hole_error_share=float(hole_err / max(total_err, 1e-12)),
        sky_pixel_frac=float(sky_pix / max(total_pix, 1)),
        sky_error_share=float(sky_err / max(total_err, 1e-12)),
        alpha_lt_0_5_pixel_frac=float(a05_pix / max(total_pix, 1)),
        alpha_lt_0_5_error_share=float(a05_err / max(total_err, 1e-12)),
        alpha_lt_0_1_pixel_frac=float(a01_pix / max(total_pix, 1)),
        alpha_lt_0_1_error_share=float(a01_err / max(total_err, 1e-12)),
        train_views_only=True,
        split_source=str(split_path) if split else None,
        camera_names=cam_names,
        hole_meta=dict(alpha_threshold=float(args.alpha_hole),
                       max_hole_px=float(args.max_hole_px),
                       views=cam_names, train_views_only=True,
                       sky_exclusion="pixels whose nearest alpha>=threshold pixel is "
                                     "farther than max_hole_px are counted as sky and "
                                     "are NOT emitted into the hole cloud"),
        psnr=float(-10 * np.log10(max(total_err / max(total_pix, 1), 1e-12))),
        per_view=per_view,
    )
    res["error_to_gauss_ratio_gt_half"] = (
        res["error_share_gt_half"] / max(res["gauss_share_gt_half"], 1e-12))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    if args.out_error_points and error_pts:
        ep = np.concatenate(error_pts, 0)
        Path(args.out_error_points).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out_error_points, ep)
        res["n_error_points"] = int(len(ep))
        res["error_points"] = str(args.out_error_points)
        Path(args.out).write_text(json.dumps(res, indent=1))
    if args.out_holes and hole_pts:
        pts = np.concatenate(hole_pts, 0)
        Path(args.out_holes).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out_holes, pts)
        res["n_hole_points"] = int(len(pts))
        res["hole_points"] = str(args.out_holes)
        Path(args.out).write_text(json.dumps(res, indent=1))
    elif args.out_holes:
        res["n_hole_points"] = 0
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in
                      ("psnr", "alpha_mean", "alpha_lt_0_5_pixel_frac",
                       "alpha_lt_0_5_error_share", "alpha_lt_0_1_pixel_frac",
                       "alpha_lt_0_1_error_share", "hole_pixel_frac", "hole_error_share",
                       "sky_pixel_frac", "sky_error_share", "error_share_gt_half",
                       "gauss_share_gt_half", "error_to_gauss_ratio_gt_half")}, indent=1))


if __name__ == "__main__":
    main()
