"""Measure gsplat 1.5.3 RGB vs RGB+ED (depth) cost from a delivery FastGS PLY.

PGSR's depth half needs a dense depth buffer.  The vendored FastGS rasterizer
does not return depth, so the candidate implementation is a *second* render via
gsplat's rasterizer on the same Gaussian tensors.  This probe measures what that
second call would cost per camera (forward and forward+backward), so the
affordability of a low-frequency depth-consistency variant can be decided from a
number instead of an estimate.

  python -m research.depth_cost_probe --work outputs/prep/mipnerf360/bicycle/work \
      --ply <delivery point_cloud.ply> --views 3 --reps 20
"""
import argparse
import json
import time

import numpy as np
import torch
from gsplat.rendering import rasterization

from competition.bridge import parameter_arrays, read_training_views
from competition.train_utils import make_parameters
from research.mcmc_budget import read_world


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--views", type=int, default=3,
                        help="number of training views averaged per measurement")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--record", default=None)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    split, views = read_training_views(args.work)
    world = read_world(args.ply)
    poses = np.stack([v["c2w"] for v in views]).astype(np.float32)
    center = poses[:, :3, 3].mean(0)
    extent = float(np.max(np.abs(poses[:, :3, 3] - center)))
    scene_scale = 1.0 / extent
    params = make_parameters(parameter_arrays(world, center, scene_scale), device)
    poses[:, :3, 3] = (poses[:, :3, 3] - center) * scene_scale
    viewmats = torch.linalg.inv(torch.from_numpy(poses).to(device))
    height, width = split["resolution"]
    base_k = np.asarray(views[0]["K"], dtype=np.float32)
    k = base_k.copy()
    k[0, 2], k[1, 2] = width / 2, height / 2   # match FastGS's centred FoV projection
    ks = torch.from_numpy(np.ascontiguousarray(k)).to(device)
    near, far = 0.01 * scene_scale, 200.0 * scene_scale

    def call(view_idx, mode):
        rendered, alpha, info = rasterization(
            means=params["means"], quats=params["quats"],
            scales=params["scales"].exp(), opacities=params["opacities"].sigmoid(),
            colors=torch.cat((params["dc"][:, None], params["rest"]), dim=1),
            viewmats=viewmats[view_idx:view_idx + 1], Ks=ks[None],
            width=width, height=height, sh_degree=3,
            near_plane=near, far_plane=far, eps2d=0.3,
            rasterize_mode="classic", packed=False, render_mode=mode)
        return rendered, alpha, info

    picks = list(range(min(args.views, len(views))))
    results = {}
    n = int(params["means"].shape[0])
    for mode in ("RGB", "RGB+ED"):
        # warm-up
        for vi in picks:
            call(vi, mode)
        torch.cuda.synchronize()
        fwd_times, bwd_times = [], []
        for _ in range(args.reps):
            for vi in picks:
                for p in params.values():
                    p.grad = None
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                rendered, alpha, info = call(vi, mode)
                torch.cuda.synchronize()
                fwd = (time.perf_counter() - t0) * 1000.0
                loss = rendered[..., :3].mean()
                if rendered.shape[-1] > 3:
                    loss = loss + rendered[..., 3].mean()
                t0 = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize()
                bwd = (time.perf_counter() - t0) * 1000.0
                fwd_times.append(fwd)
                bwd_times.append(bwd)
        results[mode] = dict(
            forward_ms_mean=float(np.mean(fwd_times)),
            forward_ms_median=float(np.median(fwd_times)),
            forward_ms_p90=float(np.percentile(fwd_times, 90)),
            backward_ms_mean=float(np.mean(bwd_times)),
            forward_backward_ms_mean=float(np.mean(fwd_times) + np.mean(bwd_times)),
            channels=int(rendered.shape[-1]), num_gaussians=n,
            resolution=[height, width], views=len(picks), reps_per_view=args.reps)
        print(json.dumps({mode: results[mode]}, indent=2), flush=True)

    extra_depth_fwd = results["RGB+ED"]["forward_ms_mean"] - results["RGB"]["forward_ms_mean"]
    extra_depth_fb = (results["RGB+ED"]["forward_backward_ms_mean"]
                      - results["RGB"]["forward_backward_ms_mean"])
    fastgs_step_fb_ms = 4.84   # measured 2026-10-31, 1-cam FastGS fwd+bwd, bicycle
    summary = dict(
        num_gaussians=n, resolution=[height, width], views=picks,
        rgb_forward_ms=results["RGB"]["forward_ms_mean"],
        rgb_ed_forward_ms=results["RGB+ED"]["forward_ms_mean"],
        rgb_forward_backward_ms=results["RGB"]["forward_backward_ms_mean"],
        rgb_ed_forward_backward_ms=results["RGB+ED"]["forward_backward_ms_mean"],
        extra_depth_forward_ms=extra_depth_fwd,
        extra_depth_forward_backward_ms=extra_depth_fb,
        fastgs_1cam_step_fwd_bwd_ms_reference=fastgs_step_fb_ms,
        extra_depth_as_fraction_of_fastgs_step=extra_depth_fb / fastgs_step_fb_ms,
        full_rate_depth_b_pct=100.0 * extra_depth_fb / fastgs_step_fb_ms,
        every10steps_depth_b_pct=100.0 * (extra_depth_fb / fastgs_step_fb_ms) / 10.0,
    )
    # TIME cost if a depth pass is added every 10th step on both halves:
    # dTIME = -20 * (B * frac) / (0.9*budget); frac = extra_fb/(fastgs_step*10)
    frac10 = (extra_depth_fb / fastgs_step_fb_ms) / 10.0
    summary["every10steps_time_points_bicycle"] = -20.0 * frac10 / (0.9 * 95.0)
    print("[depth_cost_probe] " + json.dumps(summary), flush=True)
    if args.record:
        with open(args.record, "w") as fh:
            json.dump(dict(per_mode=results, summary=summary), fh, indent=2)
    return summary


if __name__ == "__main__":
    main()
