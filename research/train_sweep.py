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


def _maybe_patch_official_adam():
    """Task-22 compatibility pilot: official Faster-GS CUDA Adam (FasterGSCudaBackend).

    Opt-in via FFSPLAT_OFFICIAL_ADAM=1 only; the delivery path never touches this.
    The official FusedAdam replaces torch.optim.Adam process-wide before
    GaussianModel.training_setup builds the two 3DGS optimizers.  Each 3DGS param
    group holds exactly one tensor, which is the official class's contract.
    """
    if os.environ.get("FFSPLAT_OFFICIAL_ADAM") != "1":
        return
    from FasterGSCudaBackend.torch_bindings.adam import FusedAdam
    import inspect
    torch.optim.Adam = FusedAdam
    print("[sweep] official Faster-GS FusedAdam patched: %s (%s)"
          % (FusedAdam, inspect.getfile(FusedAdam)))


def main(argv=None):
    _maybe_patch_official_adam()
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
    # ---- module-B engineering audit prototypes (default off; phase-1 math identical).
    parser.add_argument("--scene_cache", type=str, default=None,
                        help="path to an A-side scene tensor cache (research.scene_cache): "
                             "replaces the PNG/PIL scene load with one uint8 H->D upload + one "
                             "GPU convert; camera tensors are copied from the cache. Default "
                             "off; user-approved data residency (see B-ENGINEERING-20261030.md)")
    parser.add_argument("--train_cameras_only", action="store_true",
                        help="do not load/convert TEST cameras during module B: they are "
                             "only used by the post-training diagnostic PSNR pass, never by "
                             "optimization, but Scene converts their images to GPU inside B. "
                             "Implies quick_test_psnr=None.")
    parser.add_argument("--lean_step", action="store_true",
                        help="pure per-step eliminations: drop the no-op .cuda() on the "
                             "GPU-resident GT, hoist the realloc_at parse, index-based camera "
                             "sampling (same RNG stream => bit-identical training)")
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
    parser.add_argument("--lpips_start", type=int, default=1,
                        help="first iteration applying --lpips_weight (Codex VGG100: 4901)")
    parser.add_argument("--lpips_end", type=int, default=10**9,
                        help="last iteration applying --lpips_weight (inclusive)")
    # ---- optional staged continuation after the phase-1 delivery run (~S=5000).
    # Defaults keep phase 1 byte-identical to the delivery recipe; the stages only
    # run when the caller opts in.  See records/STAGED-LOSS-20261030.md.
    parser.add_argument("--stage2_iters", type=int, default=0,
                        help="post-S continuation steps (phase 2); 0 = disabled")
    parser.add_argument("--stage2_cams", type=int, default=1,
                        help="training cameras per phase-2/3 step (1 = standard 3DGS)")
    parser.add_argument("--stage2_trainer", choices=("standard", "absgrad", "prune"),
                        default="standard",
                        help="phase-2 optimiser: standard = 1-cam L1+DSSIM with no "
                             "topology ops; absgrad = same loss but periodic net-zero "
                             "relocation driven by the 2D-projected gradient norm "
                             "(AbsGrad/improved-GS densification criterion, single view)")
    parser.add_argument("--stage2_lr_final", type=float, default=None,
                        help="xyz LR reached at the END of phase 2+3 (default: 0.1x the "
                             "phase-1 LR at S); phase 1's own schedule is untouched")
    parser.add_argument("--stage2_lr_from_ratio", type=float, default=None,
                        help="override the continuation start LR as a ratio of "
                             "position_lr_init (used when phase 2/3 warm-starts a fresh "
                             "process from a PLY; e.g. 0.01 = position_lr_final/init)")
    parser.add_argument("--stage2_absgrad_every", type=int, default=200,
                        help="absgrad trainer: relocation window in steps")
    parser.add_argument("--stage2_split_frac", type=float, default=0.01,
                        help="absgrad trainer: fraction of Gaussians relocated per window "
                             "(net-zero: highest 2D-grad norm are split into the slots of "
                             "the lowest, so the primitive budget is unchanged)")
    parser.add_argument("--stage3_iters", type=int, default=0,
                        help="phase-3 steps after phase 2 (100 = the user's proposal); "
                             "adds the VGG-LPIPS term to the same loss")
    parser.add_argument("--stage3_lpips_weight", type=float, default=0.0,
                        help="VGG-LPIPS weight during phase 3 (reuses --lpips_every and "
                             "--lpips_scale, which default to every step / half res)")
    # ---- Mini-Splatting-style density control (pruning half).  Defaults off ->
    # phase 1 stays byte-identical.  Importance = sum over train views of
    # ||d loss / d mu_2d|| (research/ply_importance.py), i.e. rendered evidence.
    parser.add_argument("--prune_keep", type=float, default=1.0,
                        help="trainer=prune: fraction of Gaussians kept after phase 1")
    parser.add_argument("--prune_views", type=int, default=24,
                        help="train views used to accumulate the importance score")
    parser.add_argument("--mid_prune_at", type=int, default=0,
                        help="Mini-Splatting M1: prune at this phase-1 iteration "
                             "(0 = off); the run then continues at the reduced count")
    parser.add_argument("--mid_prune_keep", type=float, default=1.0,
                        help="M1: kept fraction at --mid_prune_at")
    parser.add_argument("--mid_prune_views", type=int, default=24,
                        help="M1: train views used for the importance score")
    # ---- PGSR hybrid: planar / normal priors (per-Gaussian, no second render pass).
    # Defaults 0 -> the delivery recipe stays bit-identical.
    parser.add_argument("--planar_weight", type=float, default=0.0,
                        help="PGSR planar regularizer: weight of mean((s_min/s_max)^2) "
                             "over all Gaussians (0 = off)")
    parser.add_argument("--normal_weight", type=float, default=0.0,
                        help="PGSR normal alignment: weight of mean(1-|cos|^2) between "
                             "the smallest-scale axis and the ReSplat PLY normal")
    parser.add_argument("--normal_from", type=int, default=1250,
                        help="rebuild the normal table by nearest-init lookup at this "
                             "iteration (after the densification window) when --normal_weight>0")
    parser.add_argument("--shape_stats_every", type=int, default=0,
                        help="log Gaussian shape statistics every N iterations "
                             "(0 = off; forced on when a PGSR weight is set)")
    # ---- PGSR depth-geometry regularizer (task-15): late-window, low frequency.
    # One extra gsplat RGB+ED pass per regularized step (~7.7 ms); defaults off.
    parser.add_argument("--depth_reg_weight", type=float, default=0.0,
                        help="weight of the single-view normal-depth consistency loss")
    parser.add_argument("--depth_reg_every", type=int, default=10,
                        help="apply the depth regularizer every N iterations")
    parser.add_argument("--depth_reg_from", type=int, default=4000,
                        help="first iteration of the depth-regularized window")
    parser.add_argument("--depth_reg_until", type=int, default=5000,
                        help="last iteration of the depth-regularized window")
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
    scene_cache_fallback = None
    if args.scene_cache:
        from .scene_cache import load as _load_scene_cache
        t0 = time.perf_counter()
        try:
            scene = _load_scene_cache(dataset.source_path, args.scene_cache,
                                      want_test=not args.no_quick_test)
        except (OSError, ValueError, KeyError) as error:
            # Corrupt/stale optional residency data must not change the recipe.
            # Failed load cost stays inside B, and native Scene preserves its RNG.
            args.scene_cache = None
            scene_cache_fallback = str(error)
            print(f'[sweep] scene cache fallback: {type(error).__name__}: {error}')
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        scene_load_s = time.perf_counter() - t0
        if args.scene_cache:
            print(f"[sweep] scene_cache load {scene_load_s:.3f}s "
                  f"timings={getattr(scene, 'load_timings', None)}")
    elif args.train_cameras_only:
        # Test cameras/images are loaded and converted to GPU by Scene but are only
        # used by the diagnostic quick_test after training; strip them before Scene
        # so their conversion cost leaves module B. Pure elimination (class A):
        # optimization never touches test cameras.
        import scene.dataset_readers as _readers

        def _make_no_test(_orig):
            def _no_test(*_a, **_k):
                _info = _orig(*_a, **_k)
                try:
                    return _info._replace(test_cameras=[])
                except Exception:  # pragma: no cover
                    return _info
            return _no_test

        for _key in ("Colmap", "Blender"):
            _orig = _readers.sceneLoadTypeCallbacks.get(_key)
            if _orig is None:
                continue
            _readers.sceneLoadTypeCallbacks[_key] = _make_no_test(_orig)
        print("[sweep] train_cameras_only: test cameras will not be loaded")
    if not args.scene_cache:
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
    # ---- PGSR hybrid side tables: ReSplat per-Gaussian normals from the init PLY.
    # The FastGS model does not carry normals and densification changes N, so the
    # table is valid before the first densification event and is rebuilt once after
    # the densification window (--normal_from) by nearest-init lookup.
    _init_means = _init_normals = _gauss_normals = None
    pgsr = dict(planar_weight=args.planar_weight, normal_weight=args.normal_weight,
                normal_from=args.normal_from, normal_rebuild=None)
    if args.planar_weight > 0.0 or args.normal_weight > 0.0:
        pgsr["enabled"] = True
    if args.normal_weight > 0.0 and args.init_ply:
        try:
            from plyfile import PlyData
            _verts = PlyData.read(args.init_ply)["vertex"].data
            if all(k in _verts.dtype.names for k in ("nx", "ny", "nz")):
                _n = torch.tensor(np.stack([np.asarray(_verts[k]) for k in
                                            ("nx", "ny", "nz")], axis=1),
                                  dtype=torch.float32)
                _n = _n / _n.norm(dim=1, keepdim=True).clamp_min(1e-8)
                _init_normals = _n.cuda()
                _init_means = torch.tensor(np.stack([np.asarray(_verts[k]) for k in
                                                     ("x", "y", "z")], axis=1),
                                           dtype=torch.float32).cuda()
                if _init_normals.shape[0] == n_init:
                    _gauss_normals = _init_normals.clone()
                    print(f"[pgsr] loaded {_init_normals.shape[0]} init normals")
            else:
                print("[pgsr] init PLY has no nx/ny/nz; normal_align disabled")
        except Exception as error:  # noqa: BLE001
            print(f"[pgsr] normal load failed: {error}; normal_align disabled")
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
    _need_perceptual = args.lpips_weight > 0 or (args.stage3_iters > 0
                                                 and args.stage3_lpips_weight > 0)
    if _need_perceptual:
        import lpips as lpips_pkg
        perceptual = lpips_pkg.LPIPS(net="vgg", verbose=False).cuda().eval()
        for parameter in perceptual.parameters():
            parameter.requires_grad_(False)
        print(f"[sweep] LPIPS loss weight={args.lpips_weight} every={args.lpips_every} "
              f"scale={args.lpips_scale} (phase3 weight={args.stage3_lpips_weight})")

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

    def _importance_prune(views, keep, n_views, label):
        """Mini-Splatting pruning: score train-view evidence, drop the lowest."""
        from .ply_importance import score_importance, prune_gaussians
        t0p = time.perf_counter()
        score, vis = score_importance(gaussians, views, pipe, background, opt,
                                      n_views)
        stats = prune_gaussians(gaussians, keep, score)
        stats["seconds"] = round(time.perf_counter() - t0p, 3)
        stats["views"] = min(int(n_views), len(views))
        stats["label"] = label
        print(f"[prune:{label}] N {stats['n_before']} -> {stats['n_after']} "
              f"(keep {keep}) over {stats['views']} train views in "
              f"{stats['seconds']:.2f}s")
        return stats

    def _smallest_axis():
        """World-space direction of each Gaussian's smallest scale axis."""
        rot = gaussians.get_rotation
        w, x, y, z = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
        R = torch.zeros((rot.shape[0], 3, 3), device=rot.device, dtype=rot.dtype)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        idx = gaussians.get_scaling.argmin(dim=1)
        return R[torch.arange(R.shape[0], device=R.device), :, idx]

    def _pgsr_terms(loss):
        """Add the free PGSR priors (planar + normal alignment) to the loss."""
        if args.planar_weight <= 0.0 and args.normal_weight <= 0.0:
            return loss
        scales = gaussians.get_scaling
        if args.planar_weight > 0.0:
            s_min = scales.min(dim=1).values
            s_max = scales.max(dim=1).values
            loss = loss + args.planar_weight * (
                s_min / s_max.detach().clamp_min(1e-12)).pow(2).mean()
        if (args.normal_weight > 0.0 and _gauss_normals is not None
                and _gauss_normals.shape[0] == scales.shape[0]):
            axis = _smallest_axis()
            cos_abs = (axis * _gauss_normals).sum(dim=1).abs()
            loss = loss + args.normal_weight * (1.0 - cos_abs.pow(2)).mean()
        return loss

    def _shape_stats(it):
        with torch.no_grad():
            scales = gaussians.get_scaling
            s_min = scales.min(dim=1).values
            s_max = scales.max(dim=1).values
            stats = dict(iteration=int(it), n=int(scales.shape[0]),
                         s_min_mean=float(s_min.mean()),
                         s_max_mean=float(s_max.mean()),
                         anisotropy_mean=float((s_min / s_max.clamp_min(1e-12)).mean()),
                         s_min_median=float(s_min.median()),
                         s_max_median=float(s_max.median()))
            if _gauss_normals is not None and _gauss_normals.shape[0] == scales.shape[0]:
                axis = _smallest_axis()
                stats["normal_cos_mean"] = float(
                    (axis * _gauss_normals).sum(dim=1).abs().mean())
            return stats

    def _rebuild_normals(it):
        """Nearest-init normal lookup once N is frozen (approximation, documented)."""
        nonlocal _gauss_normals
        if _init_means is None or _init_normals is None:
            return
        t0n = time.perf_counter()
        from scipy.spatial import cKDTree
        tree = cKDTree(_init_means.detach().cpu().numpy())
        _, idx = tree.query(gaussians.get_xyz.detach().cpu().numpy(), k=1,
                            workers=-1)
        _gauss_normals = _init_normals[torch.from_numpy(idx).to("cuda")]
        pgsr["normal_rebuild"] = dict(
            iteration=int(it), seconds=round(time.perf_counter() - t0n, 3),
            n=int(_gauss_normals.shape[0]))
        print(f"[pgsr] rebuilt {_gauss_normals.shape[0]} normals by nearest-init "
              f"lookup at {it} in {pgsr['normal_rebuild']['seconds']:.2f}s")

    shape_stats = []
    if (args.shape_stats_every > 0 or args.planar_weight > 0.0
            or args.normal_weight > 0.0):
        shape_stats.append(_shape_stats(0))
    depth_reg_events = []
    # pure elimination: this string parse used to run on EVERY iteration
    _realloc_ats = [int(x) for x in str(args.realloc_at).split(",") if str(x).strip()]

    state = dict(scale=None, stack=[], indices=[])
    ema_loss = 0.0
    residual_map = None
    prune_events = []
    densification_events = []

    t_train0 = time.perf_counter()
    stop_reason = "iterations"
    # stays 0 when --iterations 0 (a warm-start stage-only invocation, e.g. the
    # phase-3 VGG run of the chained MCMC experiment)
    iteration = 0
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
            state["stack"] = (cams_by_scale[scale] if args.lean_step
                              else cams_by_scale[scale].copy())
            state["indices"] = list(range(len(state["stack"])))
        if not state["indices"]:
            state["stack"] = (cams_by_scale[scale] if args.lean_step
                              else cams_by_scale[scale].copy())
            state["indices"] = list(range(len(state["stack"])))
        if args.lean_step:
            # same RNG stream and same sampled-camera sequence as the pop() version,
            # but without copying the camera list every step
            _pos = random.randint(0, len(state["indices"]) - 1)
            viewpoint_cam = state["stack"][state["indices"][_pos]]
            state["indices"].pop(_pos)
        else:
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
        gt_image = (viewpoint_cam.original_image if args.lean_step
                    else viewpoint_cam.original_image.cuda())
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
        if (perceptual is not None and args.lpips_start <= iteration <= args.lpips_end
                and iteration % args.lpips_every == 0):
            render_p, gt_p = image.unsqueeze(0), gt_image.unsqueeze(0)
            if args.lpips_scale != 1.0:
                size = (max(32, int(round(render_p.shape[2] * args.lpips_scale))),
                        max(32, int(round(render_p.shape[3] * args.lpips_scale))))
                render_p = torch.nn.functional.interpolate(render_p, size=size, mode="area")
                gt_p = torch.nn.functional.interpolate(gt_p, size=size, mode="area")
            perceptual_loss = perceptual(render_p * 2.0 - 1.0, gt_p * 2.0 - 1.0).mean()
            loss = loss + args.lpips_weight * perceptual_loss
        if (args.depth_reg_weight > 0.0
                and args.depth_reg_from <= iteration <= args.depth_reg_until
                and iteration % args.depth_reg_every == 0):
            from .gsplat_depth_loss import depth_normal_consistency
            _t_dr = time.perf_counter()
            _dloss, _dst = depth_normal_consistency(
                gaussians, viewpoint_cam, scene.cameras_extent)
            if _dloss is not None:
                loss = loss + args.depth_reg_weight * _dloss
                depth_reg_events.append(dict(
                    iteration=iteration,
                    seconds=round(time.perf_counter() - _t_dr, 4),
                    **{k: v for k, v in _dst.items() if k != "ok"}))
        loss = _pgsr_terms(loss)
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

            if (args.shape_stats_every > 0
                    and iteration % args.shape_stats_every == 0):
                shape_stats.append(_shape_stats(iteration))
            if not args.no_densify and iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if (iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    camlist = sampling_cameras(cams_by_scale[scale].copy())
                    # Pass the parsed OptimizationParams, not the raw argparse
                    # container.  Otherwise native densification silently uses the
                    # declared default grad_abs_thresh and BigD2500's 0.0008 never
                    # reaches the actual split rule (Codex audit finding).
                    importance_score, pruning_score = compute_gaussian_score_fastgs(
                        camlist, gaussians, pipe, background, opt, DENSIFY=True)
                    _n_before = int(gaussians.get_xyz.shape[0])
                    gaussians.densify_and_prune_fastgs(
                        max_screen_size=size_threshold, min_opacity=0.005,
                        extent=scene.cameras_extent, radii=radii, args=opt,
                        importance_score=importance_score, pruning_score=pruning_score)
                    densification_events.append(dict(
                        iteration=iteration, before=_n_before,
                        after=int(gaussians.get_xyz.shape[0]),
                        actual_native_parameters={k: getattr(opt, k) for k in
                            ("mult", "loss_thresh", "grad_abs_thresh",
                             "grad_thresh", "dense")}))

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
        if (args.normal_weight > 0.0 and iteration == args.normal_from
                and _init_means is not None):
            _rebuild_normals(iteration)
        if (args.mid_prune_at and iteration == args.mid_prune_at
                and args.mid_prune_keep < 1.0):
            prune_events.append(_importance_prune(
                cams_by_scale[scales[-1]], args.mid_prune_keep,
                args.mid_prune_views, "mid%d" % iteration))
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

    # ------------------------------------------------------------------
    # Optional staged continuation (phases 2/3).  With the defaults
    # (--stage2_iters 0 / --stage3_iters 0) nothing below executes and the
    # phase-1 delivery run stays byte-identical.
    #
    # phase 2 (--stage2_trainer):
    #   standard : 1-cam (--stage2_cams) L1 + lambda_dssim, no densify/prune/realloc
    #   absgrad  : same loss, plus a net-zero relocation every --stage2_absgrad_every
    #              steps: clone the Gaussians with the largest accumulated
    #              2D-projected gradient norm (|d loss / d mu_2d|, the AbsGrad /
    #              improved-GS densification criterion) into the slots of the
    #              lowest-scoring ones, so the primitive budget is unchanged.
    # phase 3: same loop with the VGG-LPIPS term added (reuses --lpips_every /
    #          --lpips_scale and the phase-3 weight).
    # The xyz LR continues decaying from the phase-1 schedule value at S down to
    # --stage2_lr_final over the whole extension; phase 1's own schedule is
    # untouched.  Every stage end dumps a snapshot PLY at the cumulative step
    # count so research.run_sweep evaluates it with a cumulative B time.
    if (args.shape_stats_every > 0 or args.planar_weight > 0.0
            or args.normal_weight > 0.0):
        shape_stats.append(_shape_stats(iteration))
    stage1_s = time.perf_counter() - t_train0
    stage2_s = 0.0
    stage3_s = 0.0
    stage2_done = 0
    stage3_done = 0
    lr_from = None
    lr_to = None
    _abs_iter = iteration
    _n_stage2 = max(0, int(args.stage2_iters))
    _n_stage3 = max(0, int(args.stage3_iters))
    if (_n_stage2 or _n_stage3) and stop_reason == "iterations":
        _extension = _n_stage2 + _n_stage3
        if args.stage2_lr_from_ratio is not None:
            lr_from = (float(args.stage2_lr_from_ratio) * opt.position_lr_init
                       * gaussians.spatial_lr_scale)
        else:
            lr_from = float(gaussians.xyz_scheduler_args(iterations))
        lr_to = (args.stage2_lr_final if args.stage2_lr_final is not None
                 else 0.1 * lr_from)
        _stage_pool = list(cams_by_scale[scales[-1]])
        # Cumulative train_seconds for the stage rows must use the SAME export
        # convention as research.run_sweep's phase-1 snapshot: the snapshot PLY
        # write is not part of B (the +0.5 in run_sweep is the flat export
        # allowance), so the phase-1 snapshot's own elapsed time -- recorded
        # before its PLY write -- is the base, and only the stage LOOP seconds
        # are added.  Otherwise every stage row would silently carry the ~6 s
        # phase-1 PLY write and its B delta would be inflated.
        _stage_train_base = float(snaps[-1]["train_seconds"]) if snaps else 0.0
        _stage_train_acc = 0.0

        def _set_xyz_lr(ext_step):
            if lr_from <= 0.0:
                return 0.0
            frac = min(1.0, ext_step / float(_extension))
            if lr_to > 0.0:
                lr = lr_from * (lr_to / lr_from) ** frac
            else:
                lr = lr_from * (1.0 - frac)
            for group in gaussians.optimizer.param_groups:
                if group["name"] == "xyz":
                    group["lr"] = lr
            return lr

        def _absgrad_relocate():
            """Net-zero clone relocation driven by the 2D gradient norm."""
            import torch as _t
            with _t.no_grad():
                n = gaussians.get_xyz.shape[0]
                n_ops = min(int(args.stage2_split_frac * n), n // 4)
                if n_ops < 2:
                    return 0
                score = (gaussians.xyz_gradient_accum.view(-1)
                         / (gaussians.denom.view(-1) + 1e-8))
                score[score.isnan()] = 0.0
                retired = _t.topk(score, n_ops, largest=False).indices
                parents = _t.topk(score, n_ops, largest=True).indices
                gaussians._xyz.data[retired] = (
                    gaussians._xyz.data[parents]
                    + _t.randn_like(gaussians._xyz.data[parents])
                    * gaussians.get_scaling.data[parents] * 0.3)
                gaussians._scaling.data[retired] = gaussians._scaling.data[parents] - 0.4700036
                gaussians._rotation.data[retired] = gaussians._rotation.data[parents]
                gaussians._opacity.data[retired] = gaussians._opacity.data[parents]
                gaussians._features_dc.data[retired] = gaussians._features_dc.data[parents]
                gaussians._features_rest.data[retired] = gaussians._features_rest.data[parents]
                for _opt in (getattr(gaussians, "optimizer", None),
                             getattr(gaussians, "shoptimizer", None)):
                    if _opt is None:
                        continue
                    for _g in _opt.param_groups:
                        _st = _opt.state.get(_g["params"][0])
                        if _st and _st.get("exp_avg") is not None:
                            _st["exp_avg"][retired] = 0.0
                            _st["exp_avg_sq"][retired] = 0.0
                gaussians.xyz_gradient_accum.zero_()
                gaussians.denom.zero_()
            return n_ops

        def _run_stage(name, n_steps, lpips_weight):
            nonlocal _abs_iter, _stage_train_acc
            if n_steps <= 0:
                return 0.0, 0
            if name == "stage2" and args.stage2_trainer == "absgrad":
                # start the window clean: the phase-1 accumulators are stale
                gaussians.xyz_gradient_accum.zero_()
                gaussians.denom.zero_()
            t_stage = time.perf_counter()
            prune_stats = None
            if name == "stage2" and args.stage2_trainer == "prune":
                prune_stats = _importance_prune(
                    _stage_pool, args.prune_keep, args.prune_views, "stage2")
            bar = tqdm(range(1, n_steps + 1), desc=name, disable=args.quiet,
                       file=sys.stdout, mininterval=1.0)
            for _k in range(1, n_steps + 1):
                _abs_iter += 1
                _lr = _set_xyz_lr(_abs_iter - iterations)
                pool = _stage_pool.copy()
                cams = []
                for _ in range(min(max(1, int(args.stage2_cams)), len(pool))):
                    cams.append(pool.pop(random.randint(0, len(pool) - 1)))
                loss_view = None
                viewspace = []
                for cam in cams:
                    render_pkg = render_fastgs(cam, gaussians, pipe, background,
                                               opt.mult, 1.0)
                    image = render_pkg["render"]
                    gt_image = cam.original_image.cuda()
                    Ll1 = l1_loss(image, gt_image)
                    if FUSED_SSIM_AVAILABLE:
                        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                    else:
                        ssim_value = 1.0
                    view_loss = ((1.0 - opt.lambda_dssim) * Ll1
                                 + opt.lambda_dssim * (1.0 - ssim_value))
                    if (lpips_weight > 0.0 and perceptual is not None
                            and _abs_iter % args.lpips_every == 0):
                        render_p, gt_p = image.unsqueeze(0), gt_image.unsqueeze(0)
                        if args.lpips_scale != 1.0:
                            size = (max(32, int(round(render_p.shape[2] * args.lpips_scale))),
                                    max(32, int(round(render_p.shape[3] * args.lpips_scale))))
                            render_p = torch.nn.functional.interpolate(
                                render_p, size=size, mode="area")
                            gt_p = torch.nn.functional.interpolate(
                                gt_p, size=size, mode="area")
                        perceptual_loss = perceptual(render_p * 2.0 - 1.0,
                                                     gt_p * 2.0 - 1.0).mean()
                        view_loss = view_loss + lpips_weight * perceptual_loss
                    loss_view = view_loss if loss_view is None else loss_view + view_loss
                    viewspace.append(render_pkg["viewspace_points"])
                _pgsr_terms(loss_view / len(cams)).backward()
                with torch.no_grad():
                    if (name == "stage2" and args.stage2_trainer == "absgrad"
                            and viewspace):
                        for _vp in viewspace:
                            if _vp.grad is None:
                                continue
                            gaussians.xyz_gradient_accum += _vp.grad[:, :2].norm(
                                dim=-1, keepdim=True)
                            gaussians.denom += 1.0
                        if _abs_iter % args.stage2_absgrad_every == 0:
                            _moved = _absgrad_relocate()
                            if not args.quiet and _moved:
                                print(f"[absgrad] step {_abs_iter}: relocated {_moved} "
                                      f"slots (net-zero)")
                    if not args.quiet and _k % 100 == 0:
                        bar.set_postfix({"it": _abs_iter, "lr": f"{_lr:.2e}"})
                        bar.update(100)
                    gaussians.optimizer_step(_abs_iter)
            bar.close()
            # record BEFORE the PLY save, exactly like the phase-1 snapshot
            # trigger: research.run_sweep's B convention is
            # scene_load + init + snapshot train_seconds + 0.5 (the flat 0.5 is
            # the export allowance), so the export write is not charged twice.
            elapsed = time.perf_counter() - t_stage
            _stage_train_acc += elapsed
            entry = dict(iteration=_abs_iter,
                         train_seconds=_stage_train_base + _stage_train_acc,
                         num_gaussians=int(gaussians.get_xyz.shape[0]),
                         stage=name, stage_seconds=round(elapsed, 3))
            if prune_stats is not None:
                entry["prune"] = prune_stats
            ply_dir = args.snap_dir or dataset.model_path
            os.makedirs(ply_dir, exist_ok=True)
            ply = os.path.join(ply_dir, f"iter{_abs_iter:05d}.ply")
            gaussians.save_ply(ply)
            entry["ply"] = ply
            snaps.append(entry)
            print(f"[stage] {name}: {n_steps} steps in {elapsed:.2f}s -> {ply}")
            return elapsed, n_steps

        print(f"[stage] S={iterations} -> +{_n_stage2} steps ({args.stage2_trainer}, "
              f"{args.stage2_cams} cam/step) then +{_n_stage3} VGG-LPIPS steps "
              f"(w={args.stage3_lpips_weight}); xyz LR {lr_from:.3e} -> {lr_to:.3e} "
              f"over {_extension} steps")
        stage2_s, stage2_done = _run_stage("stage2", _n_stage2, 0.0)
        stage3_s, stage3_done = _run_stage("stage3", _n_stage3, args.stage3_lpips_weight)
    elif _n_stage2 or _n_stage3:
        print(f"[stage] skipped: phase 1 stopped early ({stop_reason})")
    train_s = time.perf_counter() - t_train0
    if args.time_breakdown:
        timing = {k: round(v, 3) for k, v in timing.items()}
        timing["train_seconds"] = round(train_s, 3)
        timing["iterations"] = _abs_iter
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
        iterations=iterations, completed_iterations=_abs_iter, stop_reason=stop_reason,
        final_ply=final_ply,
        limit_seconds=args.limit_seconds, snapshots=snaps,
        scene_load_seconds=round(scene_load_s, 3), init_seconds=round(init_s, 3),
        train_seconds=round(train_s, 3), total_seconds=round(total_s, 3),
        # staged continuation bookkeeping (all zero with the default flags)
        stage1_seconds=round(stage1_s, 3), stage2_seconds=round(stage2_s, 3),
        stage3_seconds=round(stage3_s, 3), stage1_iterations=iterations,
        stage2_iterations=stage2_done, stage3_iterations=stage3_done,
        total_iterations=_abs_iter, stage2_trainer=args.stage2_trainer,
        stage2_cams=args.stage2_cams, stage2_lr_from=lr_from,
        stage2_lr_final=lr_to, stage3_lpips_weight=args.stage3_lpips_weight,
        prune_keep=args.prune_keep, prune_views=args.prune_views,
        eng=dict(train_cameras_only=bool(args.train_cameras_only),
                 lean_step=bool(args.lean_step),
                 scene_cache_fallback=scene_cache_fallback,
                 scene_cache=args.scene_cache),
        pgsr=dict(pgsr, shape_stats=shape_stats,
                  shape_stats_every=args.shape_stats_every),
        depth_reg=dict(weight=args.depth_reg_weight, every=args.depth_reg_every,
                       from_iter=args.depth_reg_from, until_iter=args.depth_reg_until,
                       events=depth_reg_events),
        mid_prune_at=args.mid_prune_at, mid_prune_keep=args.mid_prune_keep,
        prune_events=prune_events, densification_events=densification_events,
        timing_breakdown=(timing if args.time_breakdown else None),
        iterations_per_second=round(_abs_iter / train_s, 2) if train_s else None,
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
                      lpips_scale=args.lpips_scale, lpips_start=args.lpips_start,
                      lpips_end=args.lpips_end, grad_abs_thresh=opt.grad_abs_thresh,
                      stage2_iters=args.stage2_iters, stage2_cams=args.stage2_cams,
                      stage2_trainer=args.stage2_trainer,
                      stage2_absgrad_every=args.stage2_absgrad_every,
                      stage2_split_frac=args.stage2_split_frac,
                      stage2_lr_final=args.stage2_lr_final,
                      stage3_iters=args.stage3_iters,
                      stage3_lpips_weight=args.stage3_lpips_weight,
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
