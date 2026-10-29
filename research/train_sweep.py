"""Research-only FastGS continuation trainer with snapshots and schedule knobs.

This is our own code. It imports the pinned organizer FastGS package read-only
(competition/upstream/baseline/fastgs) and never edits upstream files. The
training loop mirrors ``baseline/module_b/train_from_ply_fastgs.py`` but exposes
the knobs a short continuation actually needs (LR schedule length, densification
window, opacity reset, SH ramp) and can dump intermediate PLYs so a single run
yields a quality-vs-iteration curve.

Run in the fastgs conda env, from the repository root:

  python -m research.train_sweep \
      --source work/<case> --model-path /tmp/sweep/model \
      --init-ply .../resplat_world.ply --iterations 12000 \
      --snapshots 1000,2000,4000,8000,12000 --snap-dir /tmp/sweep/snaps
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "competition/upstream/baseline"
sys.path.insert(0, str(BASELINE / "fastgs"))

from argparse import ArgumentParser, Namespace  # noqa: E402

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402

try:
    from fused_ssim import fused_ssim as fast_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:  # pragma: no cover - environment check handles this
    fast_ssim = None
    FUSED_SSIM_AVAILABLE = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def quick_test_psnr(scene, gaussians, pipe, background):
    psnrs = []
    with torch.no_grad():
        for cam in scene.getTestCameras():
            out = render_fastgs(cam, gaussians, pipe, background, 1.0, 1.0)
            img = torch.clamp(out["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
            psnrs.append(psnr(img, gt).mean().item())
    return float(np.mean(psnrs)) if psnrs else None


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--init_ply", type=str, required=True)
    parser.add_argument("--snapshots", type=str, default="",
                        help="comma separated iterations to dump a PLY for")
    parser.add_argument("--snap_dir", type=str, default=None)
    parser.add_argument("--record", type=str, default=None, help="timing JSON path")
    parser.add_argument("--seed", type=int, default=20261015)
    # every other knob (--iterations, --position_lr_*, --densify_*, --mult,
    # --lambda_dssim, ...) is already declared by FastGS's OptimizationParams.
    parser.add_argument("--sh_up_every", type=int, default=1000)
    parser.add_argument("--no_densify", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    opt = op.extract(args)
    iterations = opt.iterations
    sh_up_every = sh_up_every
    if opt.opacity_reset_interval <= 0:
        # 0 means "never reset": push it past the run length.
        opt.opacity_reset_interval = iterations + 1

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    seed_all(args.seed)
    safe_state(args.quiet)
    torch.cuda.empty_cache()

    t_total0 = time.perf_counter()
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "cfg_args"), "w") as stream:
        stream.write(str(Namespace(**vars(dataset))))

    t0 = time.perf_counter()
    scene = Scene(dataset, gaussians)
    scene_load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    gaussians.load_ply(args.init_ply)
    n_init = int(gaussians.get_xyz.shape[0])
    gaussians.spatial_lr_scale = scene.cameras_extent
    gaussians.training_setup(op)
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    init_s = time.perf_counter() - t0

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    snapshots = sorted({int(x) for x in args.snapshots.split(",") if x.strip()})
    if args.snap_dir:
        os.makedirs(args.snap_dir, exist_ok=True)
    snaps = []
    trigger = {s: ("t_snap", None) for s in snapshots}

    train_cams = scene.getTrainCameras()
    viewpoint_stack = train_cams.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss = 0.0

    t_train0 = time.perf_counter()
    progress = tqdm(range(1, iterations + 1), desc="FastGS", disable=args.quiet,
                    file=sys.stdout, mininterval=1.0)
    for iteration in range(1, iterations + 1):
        gaussians.update_learning_rate(iteration)
        if iteration % sh_up_every == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = train_cams.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = random.randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, background, opt.mult, 1.0)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = 1.0
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if not args.quiet and iteration % 100 == 0:
                progress.set_postfix({"loss": f"{ema_loss:.5f}",
                                      "N": gaussians.get_xyz.shape[0]})
                progress.update(100)

            if not args.no_densify and iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if (iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    camlist = sampling_cameras(train_cams.copy())
                    importance_score, pruning_score = compute_gaussian_score_fastgs(
                        camlist, gaussians, pipe, background, op, DENSIFY=True)
                    gaussians.densify_and_prune_fastgs(
                        max_screen_size=size_threshold, min_opacity=0.005,
                        extent=scene.cameras_extent, radii=radii, args=op,
                        importance_score=importance_score, pruning_score=pruning_score)

            if not args.no_densify and (
                    iteration % opt.opacity_reset_interval == 0
                    or (dataset.white_background and iteration == opt.densify_from_iter)):
                gaussians.reset_opacity()

            gaussians.optimizer_step(iteration)

        if iteration in trigger:
            elapsed = time.perf_counter() - t_train0
            n_now = int(gaussians.get_xyz.shape[0])
            entry = dict(iteration=iteration, train_seconds=elapsed, num_gaussians=n_now)
            if args.snap_dir:
                path = os.path.join(args.snap_dir, f"iter{iteration:05d}.ply")
                gaussians.save_ply(path)
                entry["ply"] = path
            snaps.append(entry)

    train_s = time.perf_counter() - t_train0
    final_psnr = None
    try:
        final_psnr = quick_test_psnr(scene, gaussians, pipe, background)
    except Exception as error:  # noqa: BLE001
        print(f"[sweep] quick psnr failed: {error}")
    total_s = time.perf_counter() - t_total0

    record = dict(
        init_ply=str(args.init_ply), init_mode=f"ply:{os.path.basename(args.init_ply)}",
        num_init_gaussians=n_init, num_final_gaussians=int(gaussians.get_xyz.shape[0]),
        iterations=iterations, snapshots=snaps,
        scene_load_seconds=round(scene_load_s, 3), init_seconds=round(init_s, 3),
        train_seconds=round(train_s, 3), total_seconds=round(total_s, 3),
        iterations_per_second=round(iterations / train_s, 2) if train_s else None,
        quick_test_psnr=round(final_psnr, 4) if final_psnr is not None else None,
        num_train_views=len(train_cams), num_test_views=len(scene.getTestCameras()),
        settings=dict(position_lr_max_steps=opt.position_lr_max_steps,
                      position_lr_init=opt.position_lr_init,
                      position_lr_final=opt.position_lr_final,
                      densify_from_iter=opt.densify_from_iter,
                      densify_until_iter=opt.densify_until_iter,
                      densification_interval=opt.densification_interval,
                      opacity_reset_interval=opt.opacity_reset_interval,
                      mult=opt.mult, lambda_dssim=opt.lambda_dssim,
                      scaling_lr=opt.scaling_lr, opacity_lr=opt.opacity_lr,
                      position_lr_delay_mult=opt.position_lr_delay_mult,
                      no_densify=args.no_densify, seed=args.seed),
    )
    if args.record:
        os.makedirs(os.path.dirname(args.record), exist_ok=True)
        with open(args.record, "w") as stream:
            json.dump(record, stream, indent=2)
    print("[sweep] " + json.dumps({k: v for k, v in record.items() if k != "snapshots"}))
    return record


if __name__ == "__main__":
    main()
