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


def _make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


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
    parser.add_argument("--init_ply", type=str, default=None)
    parser.add_argument("--init_random", action="store_true",
                        help="module-B ablation: no ReSplat init, random frustum points")
    parser.add_argument("--init_random_num", type=int, default=100_000)
    parser.add_argument("--snapshots", type=str, default="",
                        help="comma separated iterations to dump a PLY for")
    parser.add_argument("--snap_dir", type=str, default=None)
    parser.add_argument("--record", type=str, default=None, help="timing JSON path")
    parser.add_argument("--seed", type=int, default=20261015)
    # every other knob (--iterations, --position_lr_*, --densify_*, --mult,
    # --lambda_dssim, ...) is already declared by FastGS's OptimizationParams.
    parser.add_argument("--sh_up_every", type=int, default=1000)
    parser.add_argument("--densify_count_thresh", type=int, default=None,
                        help="sampled-view count needed to densify (upstream hardcodes 5)")
    parser.add_argument("--max_gaussians", type=int, default=None,
                        help="hard cap on Gaussian count, weakest pruned first")
    parser.add_argument("--inject_holes", type=int, default=0,
                        help="add N low-opacity Gaussians in empty voxels at --inject_at")
    parser.add_argument("--inject_at", type=int, default=800,
                        help="iteration at which the hole injection happens")
    parser.add_argument("--inject_voxels", type=int, default=64,
                        help="voxel grid resolution used to define empty space")
    parser.add_argument("--snap_seconds", type=str, default="",
                        help="comma list of train-seconds at which to dump a PLY (curve)")
    parser.add_argument("--opt_step_schedule", type=str, default="",
                        help="e.g. '2500:2,3750:4' = dense Adam every N steps after that "
                             "iteration; the LR is scaled by N so the step size is kept")
    parser.add_argument("--time_breakdown", action="store_true",
                        help="accumulate per-section wall clock (render/backward/optimizer)")
    parser.add_argument("--realloc_frac", type=float, default=0.0,
                        help="budget-preserving reallocation: retire this fraction of the "
                             "lowest-contribution Gaussians and split the same number of "
                             "high-residual ones at --realloc_at")
    parser.add_argument("--realloc_at", type=str, default="1250",
                        help="one or more iterations, comma separated")
    parser.add_argument("--freeze_geometry", action="store_true",
                        help="optimise only the SH/colour parameters (position, scale, "
                             "rotation and opacity frozen); tests how much of the 5k gain "
                             "is appearance rather than geometry")
    parser.add_argument("--residual_weight", type=float, default=0.0,
                        help="soft attention on hard pixels (iSplat-style region-aware "
                             "enhancement): weight the term by an EMA of the per-pixel "
                             "residual so ambiguous regions get more gradient")
    parser.add_argument("--realloc_select", choices=("argsort", "topk"), default="argsort",
                        help="argsort is the version that produced the good multi-shot numbers; "
                             "topk was a speed micro-optimisation and must be validated")
    parser.add_argument("--realloc_mode", choices=("inplace", "concat"), default="concat",
                        help="inplace reuses the retired slots (O(n) writes); concat grows "
                             "the tensors through densification_postfix (3 full copies)")
    parser.add_argument("--max_sh_degree", type=int, default=3,
                        help="cap the ACTIVE SH degree (model keeps SH3 tensors, the "
                             "rasteriser evaluates fewer channels -> cheaper backward)")
    parser.add_argument("--fused_adam", action="store_true",
                        help="rebuild the per-Gaussian optimizers with fused=True")
    parser.add_argument("--legacy_sync", action="store_true",
                        help="restore the per-step loss.item() sync (for the P0-2 A/B)")
    parser.add_argument("--no_quick_test", action="store_true",
                        help="skip the diagnostic PSNR pass (the official eval follows)")
    parser.add_argument("--image_cache", type=str, default=None,
                        help="pre-decoded uint8 image cache dir (research.image_cache)")
    parser.add_argument("--limit_seconds", type=float, default=None,
                        help="stop before this module-B wall clock (train phase only)")
    parser.add_argument("--limit_reserve", type=float, default=1.5,
                        help="seconds kept back from --limit_seconds for export")
    parser.add_argument("--lpips_weight", type=float, default=0.0,
                        help="weight of VGG-LPIPS (the scored metric) in the training loss")
    parser.add_argument("--lpips_every", type=int, default=1,
                        help="apply the perceptual term every N iterations")
    parser.add_argument("--lpips_scale", type=float, default=0.5,
                        help="downscale factor for the perceptual term (1.0 = full res)")
    parser.add_argument("--res_scales", type=str, default="",
                        help="coarse-to-fine training resolutions, e.g. 0.5,0.75,1.0")
    parser.add_argument("--res_until", type=str, default="",
                        help="fraction of the run spent at each lower scale, e.g. 0.5,0.8")
    parser.add_argument("--no_densify", action="store_true")
    parser.add_argument("--no_save_final", action="store_true",
                        help="skip writing point_cloud/iteration_<N> for render_test.py")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    opt = op.extract(args)
    iterations = opt.iterations
    sh_up_every = args.sh_up_every
    if opt.opacity_reset_interval <= 0:
        # 0 means "never reset": push it past the run length.
        opt.opacity_reset_interval = iterations + 1

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    seed_all(args.seed)
    safe_state(args.quiet)
    torch.cuda.empty_cache()

    t_total0 = time.perf_counter()
    if args.densify_count_thresh is not None or args.max_gaussians is not None:
        from .dense_model import AggressiveGaussianModel as model_class
        if args.densify_count_thresh is not None:
            model_class.densify_count_thresh = args.densify_count_thresh
        if args.max_gaussians is not None:
            model_class.max_gaussians = args.max_gaussians
            # The aggressive model defaults its densify gate to 2 sampled views while the
            # standard model uses 5. Leaving it at 2 would change the densification policy
            # as well as the cap, so every "capped vs uncapped" comparison would be
            # confounded. Match the control unless the caller explicitly overrides it.
            if args.densify_count_thresh is None:
                model_class.densify_count_thresh = 5
        print(f"[sweep] densify gate > {model_class.densify_count_thresh} sampled views, "
              f"cap {model_class.max_gaussians}")
    else:
        model_class = GaussianModel
    gaussians = model_class(dataset.sh_degree, opt.optimizer_type)
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "cfg_args"), "w") as stream:
        stream.write(str(Namespace(**vars(dataset))))

    if args.image_cache:
        # Module B's wall clock includes decoding the prepared PNGs. The cache is a
        # module-A artifact holding the identical pixels, so this only removes
        # re-decoding; it is not a timing-boundary trick.
        import scene.dataset_readers as _readers
        from .image_cache import CachedImages
        from PIL import Image as _PILImage
        _cached = CachedImages(args.image_cache)
        _real_open = _readers.Image.open

        def _cached_open(path, _real=_real_open, _c=_cached):
            name = os.path.basename(path)
            idx = _c.index.get(name)
            if idx is None:
                return _real(path)
            return _PILImage.fromarray(np.asarray(_c.data[idx]))

        _readers.Image = type("CachedPIL", (), {"open": staticmethod(_cached_open)})
        print(f"[sweep] image cache: {len(_cached.index)} views")

    scales = [float(x) for x in args.res_scales.split(",") if x.strip()] or [1.0]
    cuts = [float(x) for x in args.res_until.split(",") if x.strip()]
    if len(cuts) != len(scales) - 1:
        raise SystemExit("--res_until needs one cut per resolution transition")
    t0 = time.perf_counter()
    scene = Scene(dataset, gaussians, resolution_scales=scales)
    scene_load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    if args.init_ply:
        gaussians.load_ply(args.init_ply)
        init_mode = f"ply:{os.path.basename(args.init_ply)}"
    elif args.init_random:
        # same frustum sampler the organizer's reference trainer uses
        cams = scene.getTrainCameras()
        r = max(scene.cameras_extent, 1e-6)
        per = max(args.init_random_num // len(cams), 1)
        points, colors = [], []
        for cam in cams:
            c2w = cam.world_view_transform.inverse().cpu().numpy()
            R, C = c2w[:3, :3], c2w[:3, 3]
            us = np.random.rand(per) * 2 - 1
            vs = np.random.rand(per) * 2 - 1
            depths = np.exp(np.random.uniform(np.log(0.05 * r), np.log(1.5 * r), per))
            tanx, tany = np.tan(cam.FoVx / 2), np.tan(cam.FoVy / 2)
            dirs = np.stack([us * tanx * depths, vs * tany * depths, depths], axis=1)
            points.append((R @ dirs.T).T + C)
            colors.append(np.random.rand(per, 3))
        from scene.dataset_readers import BasicPointCloud
        xyz = np.concatenate(points, 0).astype(np.float32)
        rgb = np.concatenate(colors, 0).astype(np.float32)
        gaussians.create_from_pcd(BasicPointCloud(xyz, rgb, np.zeros_like(xyz)),
                                  scene.cameras_extent)
        init_mode = f"random:{gaussians.get_xyz.shape[0]}"
    else:
        raise SystemExit("either --init_ply or --init_random is required")
    n_init = int(gaussians.get_xyz.shape[0])
    if args.freeze_geometry:
        frozen = []
        for name in ("_xyz", "_scaling", "_rotation", "_opacity"):
            tensor = getattr(gaussians, name, None)
            if tensor is not None:
                tensor.requires_grad_(False)
                frozen.append(name)
        print(f"[sweep] geometry frozen: {', '.join(frozen)}")
    gaussians.spatial_lr_scale = scene.cameras_extent
    gaussians.training_setup(opt)
    if args.max_sh_degree < 3:
        gaussians.active_sh_degree = min(gaussians.active_sh_degree, args.max_sh_degree)
        print(f"[sh] active SH degree capped at {gaussians.active_sh_degree}")
    if args.fused_adam:
        # Must run after training_setup: FastGS writes opt.percent_dense and creates
        # the Adam instances there. Only the kernel changes; param groups, LRs and the
        # update schedule are byte-identical to the control arm.
        rebuilt = []
        for name in ("optimizer", "shoptimizer"):
            opt_obj = getattr(gaussians, name, None)
            if opt_obj is None:
                continue
            # A passed param_group keeps its OWN fused/foreach value, overriding the
            # constructor default, so the flags must be overridden per group; otherwise
            # the rebuild silently stays on the non-fused path.
            groups = []
            for group in opt_obj.param_groups:
                clean = {k: v for k, v in group.items()
                         if k not in ("fused", "foreach", "capturable", "differentiable",
                                      "maximize")}
                clean["fused"] = True
                clean["foreach"] = False
                groups.append(clean)
            # the fused kernel needs one dtype/device/layout across params and state
            # Cast IN PLACE (.data) so the optimizer keeps the very same Parameter
            # objects the model uses; replacing them would silently detach the model.
            dtypes = {p.dtype for g in groups for p in g["params"]}
            grads = {p.grad.dtype for g in groups for p in g["params"] if p.grad is not None}
            if dtypes != {torch.float32} or grads - {torch.float32}:
                print(f"[fused] {name}: param dtypes {dtypes}, grad dtypes {grads}; casting")
                for g in groups:
                    for prm in g["params"]:
                        if prm.dtype != torch.float32:
                            prm.data = prm.data.float()
                        if prm.grad is not None and prm.grad.dtype != torch.float32:
                            prm.grad.data = prm.grad.data.float()
            try:
                new_opt = torch.optim.Adam(groups, lr=0.0, eps=1e-15, fused=True)
            except (TypeError, RuntimeError) as error:
                print(f"[fused] {name} not fused: {error}")
                continue
            setattr(gaussians, name, new_opt)
            rebuilt.append(name)
            print(f"[fused] {name} groups fused flags: "
                  f"{[g.get('fused') for g in new_opt.param_groups]}")
        print(f"[fused] rebuilt {', '.join(rebuilt) or 'nothing'}")
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    init_s = time.perf_counter() - t0

    perceptual = None
    if args.lpips_weight > 0:
        import lpips as lpips_pkg
        perceptual = lpips_pkg.LPIPS(net="vgg", verbose=False).cuda().eval()
        for parameter in perceptual.parameters():
            parameter.requires_grad_(False)
        print(f"[sweep] LPIPS loss weight={args.lpips_weight} every={args.lpips_every} "
              f"scale={args.lpips_scale}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    snapshots = sorted({int(x) for x in args.snapshots.split(",") if x.strip()})
    if args.snap_dir:
        os.makedirs(args.snap_dir, exist_ok=True)
    snaps = []
    trigger = {s: ("t_snap", None) for s in snapshots}

    cams_by_scale = {s: scene.getTrainCameras(s) for s in scales}

    def scale_for(iteration):
        if len(scales) == 1:
            return scales[0]
        fraction = iteration / iterations
        for cut, scale in zip(cuts, scales[:-1]):
            if fraction < cut:
                return scale
        return scales[-1]

    state = dict(scale=None, stack=[], indices=[])
    ema_loss = 0.0
    residual_map = None

    t_train0 = time.perf_counter()
    stop_reason = "iterations"
    progress = tqdm(range(1, iterations + 1), desc="FastGS", disable=args.quiet,
                    file=sys.stdout, mininterval=1.0)
    _snap_targets = sorted(float(x) for x in args.snap_seconds.split(",") if x.strip())
    timing = {"render": 0.0, "backward": 0.0, "optimizer": 0.0, "grad_to_step": 0.0}
    _sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda *a, **k: None)
    for iteration in range(1, iterations + 1):
        if args.limit_seconds is not None:
            spent = time.perf_counter() - t_train0
            if spent >= args.limit_seconds - args.limit_reserve:
                stop_reason = "wall_clock"
                break
        gaussians.update_learning_rate(iteration)
        if iteration % sh_up_every == 0 and gaussians.active_sh_degree < args.max_sh_degree:
            gaussians.oneupSHdegree()

        scale = scale_for(iteration)
        if scale != state["scale"]:
            state["scale"] = scale
            state["stack"] = cams_by_scale[scale].copy()
            state["indices"] = list(range(len(state["stack"])))
        if not state["stack"]:
            state["stack"] = cams_by_scale[scale].copy()
            state["indices"] = list(range(len(state["stack"])))
        rand_idx = random.randint(0, len(state["indices"]) - 1)
        viewpoint_cam = state["stack"].pop(rand_idx)
        state["indices"].pop(rand_idx)

        if args.time_breakdown:
            _sync()
            _t = time.perf_counter()
        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, background, opt.mult, 1.0)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if args.time_breakdown:
            _sync()
            timing["render"] += time.perf_counter() - _t
            _t = time.perf_counter()
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = 1.0
        if args.residual_weight > 0.0:
            # region-aware enhancement: keep an EMA of the per-pixel error and use it to
            # up-weight the pixels the model is still getting wrong. Zero extra renders.
            with torch.no_grad():
                cur = (image.detach() - gt_image).abs().mean(0, keepdim=True)
                if residual_map is None or tuple(residual_map.shape) != tuple(cur.shape):
                    residual_map = cur.clone()
                else:
                    residual_map.mul_(0.9).add_(0.1 * cur)
                w = 1.0 + args.residual_weight * (residual_map / (residual_map.mean() + 1e-6)).clamp(0, 3)
            Ll1 = ((image - gt_image).abs() * w).mean() / w.mean()
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        if perceptual is not None and iteration % args.lpips_every == 0:
            render_p, gt_p = image.unsqueeze(0), gt_image.unsqueeze(0)
            if args.lpips_scale != 1.0:
                size = (max(32, int(round(render_p.shape[2] * args.lpips_scale))),
                        max(32, int(round(render_p.shape[3] * args.lpips_scale))))
                render_p = torch.nn.functional.interpolate(render_p, size=size, mode="area")
                gt_p = torch.nn.functional.interpolate(gt_p, size=size, mode="area")
            perceptual_loss = perceptual(render_p * 2.0 - 1.0, gt_p * 2.0 - 1.0).mean()
            loss = loss + args.lpips_weight * perceptual_loss
        if args.time_breakdown:
            _sync()
            timing["grad_to_step"] += time.perf_counter() - _t
            _t = time.perf_counter()
        loss.backward()
        if args.time_breakdown:
            _sync()
            timing["backward"] += time.perf_counter() - _t
            _t = time.perf_counter()

        with torch.no_grad():
            # loss.item() forces a device sync; only pay it when something prints
            if args.legacy_sync or (not args.quiet and iteration % 100 == 0) or args.time_breakdown:
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
                    camlist = sampling_cameras(cams_by_scale[scale].copy())
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

            _every = 1
            if args.opt_step_schedule:
                for part in args.opt_step_schedule.split(","):
                    at, mult = part.split(":")
                    if iteration >= int(at):
                        _every = int(mult)
            if _every == 1 or iteration % _every == 0:
                if _every > 1:
                    # keep the per-step displacement comparable: undo the LR divisor
                    for group in gaussians.optimizer.param_groups:
                        group["lr"] *= _every
                gaussians.optimizer_step(iteration)
                if _every > 1:
                    for group in gaussians.optimizer.param_groups:
                        group["lr"] /= _every

        if args.time_breakdown:
            _sync()
            timing["optimizer"] += time.perf_counter() - _t

        _realloc_ats = [int(x) for x in str(args.realloc_at).split(",") if str(x).strip()]
        if args.realloc_frac > 0 and iteration in _realloc_ats:
            # One-shot, budget-preserving reallocation (Taming-3DGS style primitive
            # budget, RLSplat style allocation): retire low-contribution Gaussians and
            # spend the freed slots on high-residual regions instead of adding points.
            import torch as _t
            from utils.general_utils import inverse_sigmoid as _inv
            with _t.no_grad():
                contrib = (gaussians.get_opacity.view(-1)
                           * gaussians.get_scaling.mean(dim=1))
                n_rel = int(args.realloc_frac * contrib.numel())
                if n_rel >= 4:
                    # topk is O(N) for small k; a full argsort on 0.5-1M elements was the
                    # dominant per-shot cost (each shot otherwise costs 4-5 s).
                    if args.realloc_select == "topk":
                        low = _t.topk(contrib, n_rel, largest=False).indices
                    else:
                        low = _t.argsort(contrib)[:n_rel]
                    gaussians._opacity[low] = _inv(_t.full((n_rel, 1), 0.001,
                                                           device=contrib.device))
                    n_all = gaussians.get_xyz.shape[0]
                    if getattr(gaussians, "xyz_gradient_accum_abs", None) is None:
                        gaussians.xyz_gradient_accum_abs = _t.zeros((n_all, 1), device=contrib.device)
                    if getattr(gaussians, "denom", None) is None:
                        gaussians.denom = _t.zeros((n_all, 1), device=contrib.device)
                    grad = (gaussians.xyz_gradient_accum_abs / (gaussians.denom + 1e-8)).view(-1)
                    n_split = n_rel // 2
                    if args.realloc_select == "topk":
                        top = _t.topk(grad, n_split, largest=True).indices
                    else:
                        top = _t.argsort(grad, descending=True)[:n_split]
                    sel = gaussians.get_xyz[top].repeat(2, 1)
                    jitter = _t.randn_like(sel) * gaussians.get_scaling[top].repeat(2, 1) * 0.3
                    new_xyz = sel + jitter
                    new_scaling = _t.log(gaussians.get_scaling[top].repeat(2, 1) / 1.6)
                    new_rot = gaussians.get_rotation[top].repeat(2, 1)
                    new_op = gaussians._opacity[top].repeat(2, 1)
                    new_dc = gaussians._features_dc[top].repeat(2, 1, 1)
                    new_rest = gaussians._features_rest[top].repeat(2, 1, 1)
                    if args.realloc_mode == "inplace":
                        # Reuse the retired slots instead of growing the tensors: the
                        # budget-preserving reallocation keeps the count constant anyway,
                        # and cat_tensors_to_optimizer would copy the whole model (param
                        # + both Adam moments ~ 700 MB) on every shot.
                        n_pair = min(n_rel // 2, top.numel())
                        dst_a = low[0:2 * n_pair:2]
                        dst_b = low[1:2 * n_pair:2]
                        src = top[:n_pair]
                        for dst in (dst_a, dst_b):
                            gaussians._xyz.data[dst] = gaussians._xyz.data[src] + (
                                _t.randn_like(gaussians._xyz.data[src])
                                * gaussians.get_scaling.data[src] * 0.3)
                            gaussians._scaling.data[dst] = gaussians._scaling.data[src] - 0.4700036
                            gaussians._rotation.data[dst] = gaussians._rotation.data[src]
                            gaussians._opacity.data[dst] = gaussians._opacity.data[src]
                            gaussians._features_dc.data[dst] = gaussians._features_dc.data[src]
                            gaussians._features_rest.data[dst] = gaussians._features_rest.data[src]
                        touched = _t.cat((dst_a, dst_b))
                        for _opt in (getattr(gaussians, "optimizer", None),
                                     getattr(gaussians, "shoptimizer", None)):
                            if _opt is None:
                                continue
                            for _g in _opt.param_groups:
                                _st = _opt.state.get(_g["params"][0])
                                if _st and _st.get("exp_avg") is not None:
                                    _st["exp_avg"][touched] = 0.0
                                    _st["exp_avg_sq"][touched] = 0.0
                        print(f"[realloc] inplace retire {n_rel}, refill {2 * n_pair} "
                              f"at step {iteration} -> N={gaussians.get_xyz.shape[0]}")
                    else:
                        if getattr(gaussians, "tmp_radii", None) is None:
                            gaussians.tmp_radii = _t.zeros(gaussians.get_xyz.shape[0],
                                                           device=new_xyz.device)
                        gaussians.densification_postfix(new_xyz, new_dc, new_rest, new_op,
                                                        new_scaling, new_rot,
                                                        _t.zeros(new_xyz.shape[0], device=new_xyz.device))
                    if args.realloc_mode == "concat":
                        print(f"[realloc] retired {n_rel}, split {n_split*2} at step {iteration} "
                              f"-> N={gaussians.get_xyz.shape[0]}")
                    if args.max_gaussians and gaussians.get_xyz.shape[0] > args.max_gaussians:
                        excess = gaussians.get_xyz.shape[0] - args.max_gaussians
                        with _t.no_grad():
                            score = (gaussians.get_opacity.view(-1)
                                     * gaussians.get_scaling.mean(dim=1))
                            drop = _t.topk(score, excess, largest=False).indices
                            from utils.general_utils import inverse_sigmoid as _inv2
                            gaussians._opacity[drop] = _inv2(_t.full((excess, 1), 0.001,
                                                                     device=score.device))
                        print(f"[realloc] cap enforced: suppressed {excess} lowest-contribution "
                              f"Gaussians -> N={gaussians.get_xyz.shape[0]} (hard={args.max_gaussians})")
        if args.inject_holes and iteration == args.inject_at:
            # Mature hole-filling idea (Gaussian Frosting / PUP-3DGS): place fresh
            # low-opacity Gaussians where the current model has no primitives, then let
            # the normal opacity-based pruning delete the ones that earn no gradient.
            import torch as _t
            from utils.general_utils import inverse_sigmoid as _inv
            with _t.no_grad():
                xyz = gaussians.get_xyz.detach()
                lo = _t.quantile(xyz, 0.01, dim=0)
                hi = _t.quantile(xyz, 0.99, dim=0)
                n_vox = int(args.inject_voxels)
                cell = (hi - lo).clamp_min(1e-6) / n_vox
                idx = ((xyz - lo) / cell).floor().clamp_(0, n_vox - 1).long()
                flat = (idx[:, 0] * n_vox + idx[:, 1]) * n_vox + idx[:, 2]
                occupied = _t.zeros(n_vox ** 3, dtype=_t.bool, device=xyz.device)
                occupied[flat] = True
                keep, tries = [], 0
                while sum(k.numel() for k in keep) < args.inject_holes and tries < 20:
                    cand = lo + _t.rand(args.inject_holes * 4, 3, device=xyz.device) * (hi - lo)
                    cidx = ((cand - lo) / cell).floor().clamp_(0, n_vox - 1).long()
                    cflat = (cidx[:, 0] * n_vox + cidx[:, 1]) * n_vox + cidx[:, 2]
                    free = cand[~occupied[cflat]]
                    if free.numel():
                        keep.append(free)
                    tries += 1
                if keep:
                    new_xyz = _t.cat(keep, 0)[: args.inject_holes]
                    n_new = new_xyz.shape[0]
                    scale_med = gaussians.get_scaling.detach().median(dim=0).values
                    new_scaling = _t.log(scale_med * 0.5).unsqueeze(0).repeat(n_new, 1)
                    quat = _t.zeros(n_new, 4, device=xyz.device)
                    quat[:, 0] = 1.0
                    new_op = _inv(_t.full((n_new, 1), 0.01, device=xyz.device))
                    new_dc = _t.zeros(n_new, 1, 3, device=xyz.device)
                    new_rest = _t.zeros(n_new, gaussians._features_rest.shape[1],
                                        gaussians._features_rest.shape[2], device=xyz.device)
                    if getattr(gaussians, "tmp_radii", None) is None:
                        gaussians.tmp_radii = _t.zeros(gaussians.get_xyz.shape[0],
                                                       device=gaussians.get_xyz.device)
                    gaussians.densification_postfix(new_xyz, new_dc, new_rest, new_op,
                                                    new_scaling, quat,
                                                    _t.zeros(n_new, device=xyz.device))
                    print(f"[inject] added {n_new} hole Gaussians at step {iteration} "
                          f"(total {gaussians.get_xyz.shape[0]})")
        if _snap_targets and time.perf_counter() - t_train0 >= _snap_targets[0]:
            _sec = time.perf_counter() - t_train0
            _p = os.path.join(args.snap_dir or dataset.model_path, f"snap_{int(_snap_targets[0]):04d}s.ply")
            os.makedirs(os.path.dirname(_p), exist_ok=True)
            gaussians.save_ply(_p)
            snaps.append(dict(seconds=round(_sec, 3), iteration=iteration, ply=_p,
                              num_gaussians=int(gaussians.get_xyz.shape[0])))
            print(f"[curve] snapshot at {_sec:.2f}s -> {_p}")
            _snap_targets.pop(0)
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
    if args.time_breakdown:
        timing = {k: round(v, 3) for k, v in timing.items()}
        timing["train_seconds"] = round(train_s, 3)
        timing["iterations"] = iterations
    # standard 3DGS layout so the organizer's render_test.py can load this model
    final_ply = None
    if not args.no_save_final:
        final_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}")
        os.makedirs(final_dir, exist_ok=True)
        final_ply = os.path.join(final_dir, "point_cloud.ply")
        gaussians.save_ply(final_ply)
    total_s = time.perf_counter() - t_total0
    final_psnr = None
    if not args.no_quick_test:
        # Diagnostic only: the pipeline evaluates with the official renderer right
        # after this, so the pass is optional and its cost is outside module-B time.
        try:
            final_psnr = quick_test_psnr(scene, gaussians, pipe, background)
        except Exception as error:  # noqa: BLE001
            print(f"[sweep] quick psnr failed: {error}")

    record = dict(
        init_ply=str(args.init_ply), init_mode=init_mode,
        num_init_gaussians=n_init, num_final_gaussians=int(gaussians.get_xyz.shape[0]),
        iterations=iterations, completed_iterations=iteration, stop_reason=stop_reason,
        final_ply=final_ply,
        limit_seconds=args.limit_seconds, snapshots=snaps,
        scene_load_seconds=round(scene_load_s, 3), init_seconds=round(init_s, 3),
        train_seconds=round(train_s, 3), total_seconds=round(total_s, 3),
        timing_breakdown=(timing if args.time_breakdown else None),
        iterations_per_second=round(iteration / train_s, 2) if train_s else None,
        quick_test_psnr=round(final_psnr, 4) if final_psnr is not None else None,
        num_train_views=len(cams_by_scale[scales[-1]]),
        num_test_views=len(scene.getTestCameras()), res_scales=scales, res_until=cuts,
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
                      no_densify=args.no_densify, seed=args.seed,
                      res_scales=scales, res_until=cuts,
                      lpips_weight=args.lpips_weight, lpips_every=args.lpips_every,
                      lpips_scale=args.lpips_scale,
                      densify_count_thresh=args.densify_count_thresh,
                      max_gaussians=args.max_gaussians),
    )
    if args.record:
        os.makedirs(os.path.dirname(args.record), exist_ok=True)
        with open(args.record, "w") as stream:
            json.dump(record, stream, indent=2)
    print("[sweep] " + json.dumps({k: v for k, v in record.items() if k != "snapshots"}))
    return record


if __name__ == "__main__":
    main()
