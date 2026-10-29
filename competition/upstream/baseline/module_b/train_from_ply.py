"""Module B: classic 3DGS continuation training from ReSplat's Gaussians.

Pipeline position: AFTER ReSplat. Initializes the GaussianModel from the
world-space PLY exported by run_resplat.py (instead of a COLMAP point cloud),
then runs the classic 3DGS optimization (Inria graphdeco) for a fixed number
of iterations on ALL train views (llffhold=8 split), then saves the model.

Also supports:
  --init_random      initialize from random frustum points (for the classic
                     3DGS-from-scratch reference timing)

Usage (fastgs conda env, from baseline/):
  python module_b/train_from_ply.py \
      -s work/<group>/<scene> \
      -m results/<group>/<scene>/module_b \
      --init_ply results/<group>/<scene>/resplat/resplat_world.ply \
      --iterations 10000
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path
from random import randint

import numpy as np
import torch
from tqdm import tqdm

BASELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASELINE_ROOT / "3dgs"))

from argparse import ArgumentParser, Namespace  # noqa: E402

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.loss_utils import l1_loss, ssim as py_ssim  # noqa: E402

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False


def random_frustum_init(scene, num_points, sh_degree):
    """Uniformly sample points inside train-camera frustums."""
    cams = scene.getTrainCameras()
    r = max(scene.cameras_extent, 1e-6)
    per_cam = max(num_points // len(cams), 1)
    points, colors = [], []
    for cam in cams:
        c2w = cam.world_view_transform.inverse().cpu().numpy()  # world2cam inv
        R = c2w[:3, :3]
        C = c2w[:3, 3]
        H, W = cam.image_height, cam.image_width
        fovx, fovy = cam.FoVx, cam.FoVy
        us = np.random.rand(per_cam) * 2 - 1
        vs = np.random.rand(per_cam) * 2 - 1
        depths = np.exp(np.random.uniform(np.log(0.05 * r), np.log(1.5 * r), per_cam))
        tanx, tany = np.tan(fovx / 2), np.tan(fovy / 2)
        dirs_cam = np.stack([us * tanx * depths, vs * tany * depths, depths], axis=1)
        pts = (R @ dirs_cam.T).T + C
        points.append(pts)
        colors.append(np.random.rand(per_cam, 3))
    from scene.dataset_readers import BasicPointCloud
    xyz = np.concatenate(points, 0).astype(np.float32)
    rgb = np.concatenate(colors, 0).astype(np.float32)
    return BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros_like(xyz))


def training(dataset, opt, pipe, args):
    t_total0 = time.time()

    tb_writer = None
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)

    # prepare output dir + config dump (mirrors classic prepare_output_and_logger)
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(dataset))))

    t0 = time.time()
    scene = Scene(dataset, gaussians)
    scene_load_s = time.time() - t0

    t0 = time.time()
    if args.init_ply:
        gaussians.load_ply(args.init_ply)
        n_init = gaussians.get_xyz.shape[0]
        init_mode = f"ply:{os.path.basename(args.init_ply)}"
    elif args.init_random:
        pcd = random_frustum_init(scene, args.init_random_num, dataset.sh_degree)
        gaussians.create_from_pcd(pcd, scene.getTrainCameras(), scene.cameras_extent)
        n_init = gaussians.get_xyz.shape[0]
        init_mode = f"random:{n_init}"
    else:
        raise SystemExit("either --init_ply or --init_random is required")
    gaussians.spatial_lr_scale = scene.cameras_extent
    # exposure parameters (normally created in create_from_pcd; inert when
    # use_trained_exp is False, but training_setup expects them to exist)
    train_cams = scene.getTrainCameras()
    gaussians.exposure_mapping = {cam.image_name: idx for idx, cam in enumerate(train_cams)}
    gaussians.pretrained_exposures = None
    gaussians._exposure = torch.nn.Parameter(
        torch.eye(3, 4, device="cuda")[None].repeat(len(train_cams), 1, 1).requires_grad_(True))
    gaussians.training_setup(opt)
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    init_s = time.time() - t0

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0

    t_train0 = time.time()
    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"],
            render_pkg["visibility_filter"], render_pkg["radii"])

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = py_ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "N": gaussians.get_xyz.shape[0]})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor,
                                                  visibility_filter)
                if iteration > opt.densify_from_iter and \
                        iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                        size_threshold, radii)
                if iteration % opt.opacity_reset_interval == 0 or \
                        (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            gaussians.exposure_optimizer.step()
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

    train_s = time.time() - t_train0

    # save final model
    scene.save(opt.iterations)
    total_s = time.time() - t_total0

    # quick test PSNR at the end (diagnostic)
    test_psnr = None
    try:
        from utils.image_utils import psnr as img_psnr
        psnrs = []
        for cam in scene.getTestCameras():
            out = render(cam, gaussians, pipe, background)
            img = torch.clamp(out["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
            psnrs.append(img_psnr(img, gt).mean().item())
        if psnrs:
            test_psnr = float(np.mean(psnrs))
    except Exception as e:  # noqa: BLE001
        print(f"[B] test psnr failed: {e}")

    timing = {
        "init_mode": init_mode,
        "num_init_gaussians": int(n_init),
        "num_final_gaussians": int(gaussians.get_xyz.shape[0]),
        "iterations": int(opt.iterations),
        "num_train_views": len(scene.getTrainCameras()),
        "num_test_views": len(scene.getTestCameras()),
        "scene_load_seconds": round(scene_load_s, 2),
        "init_seconds": round(init_s, 2),
        "train_seconds": round(train_s, 2),
        "total_seconds": round(total_s, 2),
        "test_psnr_quick": round(test_psnr, 3) if test_psnr is not None else None,
    }
    with open(os.path.join(scene.model_path, "module_b.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print(f"[B] done: {timing}")


if __name__ == "__main__":
    parser = ArgumentParser(description="3DGS continuation from ReSplat PLY")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--init_ply", type=str, default=None)
    parser.add_argument("--init_random", action="store_true")
    parser.add_argument("--init_random_num", type=int, default=100_000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.save_iterations = [args.iterations]
    args.test_iterations = []

    safe_state(args.quiet)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)
    print("\nTraining complete.")
