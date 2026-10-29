"""Depth-regularized FastGS continuation for module B (research variant).

Same pipeline and every schedule knob as ``research/train_sweep.py`` -- 16
posed images -> frozen ReSplat -> world PLY -> FastGS continuation -- plus two
literature-grounded, opt-in regularizers:

1. **Edge-aware predicted-depth supervision** (FSGS / SparseGS / DepthSplat
   line).  A per-view depth sidecar produced from ReSplat's internal depth
   prediction (``research/export_resplat_depth.py``) supervises the
   alpha-weighted depth of the FastGS render.  ReSplat predicts depth at 1/4
   resolution; the sidecar is stored at the module-B training resolution and
   resized (bilinear) to whatever the renderer uses.
2. **Translation-only camera refinement** (``--pose-lr > 0``).  For the last M
   steps, one world-space translation per training view is optimized jointly
   with the Gaussians, with the same photometric + depth loss, frozen rotation,
   and a per-camera shift clamp.  Off by default; the implementation lives in
   ``research/pose_refine.py``.  The pinned rasterizer cannot backpropagate
   through ``viewmatrix``/``campos`` (they live inside the settings tuple), so
   the translation is applied as the mathematically exact world shift
   ``xyz - delta`` for the sampled view: the rendered image, alpha and depth are
   identical to moving the camera, and the rasterizer does differentiate
   ``means3D``.

Everything else (LR schedule, densification, opacity reset, SH ramp, coarse-to-
fine schedules, LPIPS term, snapshots, wall-clock stop, seeds) is copied
verbatim from ``train_sweep.py`` so a depth/pose run is comparable to a control
run with the same CLI.

GPU usage (from the repository root, fastgs conda env)::

    # 1. on the GPU host, dump ReSplat predicted depth once (see the exporter
    #    for the exact heredoc; it does NOT modify upstream run_resplat.py)
    python -m research.export_resplat_depth --print-gpu-command \
        --work work/<case> --resplat-out results/<case>/resplat

    # 2. convert the dump on any machine (numpy only)
    python -m research.export_resplat_depth \
        --raw results/<case>/resplat/depth_raw.npz \
        --split work/<case>/split.json \
        --out results/<case>/resplat/resplat_depth.npz

    # 3. depth-regularized continuation (~100 s budget)
    python -m research.depth_reg \
        --source work/<case> --model_path /tmp/depthreg/model \
        --eval -r 1 --sh_degree 3 --quiet \
        --init_ply results/<case>/resplat/resplat_world.ply \
        --iterations 12000 --limit_seconds 100 \
        --depth-npz results/<case>/resplat/resplat_depth.npz \
        --depth-weight 0.05 --depth-every 4 \
        --depth-start-frac 0.05 --depth-end-frac 0.8 \
        --snapshots 1000,2000,4000,8000,12000 --snap_dir /tmp/depthreg/snaps \
        --record /tmp/depthreg/train_record.json

    # 4. optional camera refinement (wrapper enables --pose-lr; keep it explicit)
    #    the same command works without --depth-npz (photometric-only refinement)
    python -m research.pose_refine \
        --source work/<case> --model_path /tmp/pose/model \
        --eval -r 1 --sh_degree 3 --quiet \
        --init_ply results/<case>/resplat/resplat_world.ply \
        --iterations 12000 --limit_seconds 100 \
        --depth-npz results/<case>/resplat/resplat_depth.npz \
        --depth-weight 0.05 --pose-lr 1e-4 \
        --pose-start-frac 0.75 \
        --record /tmp/pose/train_record.json

Depth renderer assumption: the pinned FastGS wrapper
(``gaussian_renderer.render_fastgs``) returns no depth.  We therefore accumulate
alpha-weighted camera-Z with one extra rasterization whose SH DC encodes
``(z / encode_max, 1, 0)`` in RGB through a read-only Gaussian proxy.  This
assumes the rasterizer applies the standard ``SH_C0 * dc + 0.5`` colour
affine without clamping and composites in linear RGB.  See
``research/DEPTH.md`` for the evidence, failure modes and diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "competition/upstream/baseline"
FASTGS = BASELINE / "fastgs"
SH_C0 = 0.28209479177387814

from .pose_refine import PoseRefiner, camera_forward  # noqa: E402

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - Mac import path
    torch = None
    F = None


# ---------------------------------------------------------------------------
# arguments / validation
# ---------------------------------------------------------------------------

def _fastgs_arguments():
    fastgs = str(FASTGS)
    if fastgs not in sys.path:
        sys.path.insert(0, fastgs)
    from arguments import ModelParams, OptimizationParams, PipelineParams
    return ModelParams, OptimizationParams, PipelineParams


def build_parser(argv=None):
    ModelParams, OptimizationParams, PipelineParams = _fastgs_arguments()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

    # --- depth regularization ------------------------------------------------
    parser.add_argument("--depth-npz", dest="depth_npz", default="",
                        help="depth sidecar .npz, or 'auto' for "
                             "<init_ply dir>/resplat_depth.npz; empty = control run")
    parser.add_argument("--depth-weight", dest="depth_weight", type=float, default=0.05,
                        help="weight of the edge-aware depth term (0 disables it)")
    parser.add_argument("--depth-every", dest="depth_every", type=int, default=4,
                        help="apply the extra depth render every N iterations")
    parser.add_argument("--depth-start-frac", dest="depth_start_frac", type=float,
                        default=0.05, help="first iteration (fraction) with depth loss")
    parser.add_argument("--depth-end-frac", dest="depth_end_frac", type=float,
                        default=0.8, help="last iteration (fraction) with depth loss")
    parser.add_argument("--depth-alpha-eps", dest="depth_alpha_eps", type=float,
                        default=0.01, help="minimum rendered alpha for a valid depth pixel")
    parser.add_argument("--depth-edge-beta", dest="depth_edge_beta", type=float,
                        default=4.0, help="edge sensitivity of the gradient term")
    parser.add_argument("--depth-grad-weight", dest="depth_grad_weight", type=float,
                        default=0.5, help="weight of the edge-aware gradient term")
    parser.add_argument("--depth-charbonnier-eps", dest="depth_charbonnier_eps",
                        type=float, default=1e-4, help="robust Charbonnier epsilon")
    parser.add_argument("--depth-min-valid", dest="depth_min_valid", type=float,
                        default=0.02, help="skip a step if fewer valid pixels than this")
    parser.add_argument("--depth-encode-max", dest="depth_encode_max", type=float,
                        default=0.0,
                        help="camera-Z range packed into RGB (0 = auto from scene extent)")

    # --- pose refinement -----------------------------------------------------
    parser.add_argument("--pose-lr", dest="pose_lr", type=float, default=0.0,
                        help="per-view camera translation LR; 0 keeps the feature off")
    parser.add_argument("--pose-start-frac", dest="pose_start_frac", type=float,
                        default=0.7,
                        help="start refinement after this fraction of --iterations")
    parser.add_argument("--pose-start-step", dest="pose_start_step", type=int, default=-1,
                        help="absolute start iteration (overrides --pose-start-frac)")
    parser.add_argument("--pose-steps", dest="pose_steps", type=int, default=-1,
                        help="M refinement steps (-1 = until the end)")
    parser.add_argument("--pose-max-shift", dest="pose_max_shift", type=float, default=0.0,
                        help="per-camera translation clamp (0 = 2%% of scene extent)")
    parser.add_argument("--pose-weight-decay", dest="pose_weight_decay", type=float,
                        default=0.0)

    parser.add_argument("--self-check", action="store_true",
                        help="CPU-only argument/loss checks (no CUDA, no FastGS)")
    return parser


def parse_schedule(scales_text, cuts_text):
    scales = [float(x) for x in str(scales_text).split(",") if x.strip()] or [1.0]
    cuts = [float(x) for x in str(cuts_text).split(",") if x.strip()]
    if len(cuts) != len(scales) - 1:
        raise ValueError("--res_until needs one cut per resolution transition")
    return scales, cuts


def resolve_depth_npz(args):
    """Return the sidecar path, or None when depth supervision is unavailable."""
    raw = str(getattr(args, "depth_npz", "") or "").strip()
    if not raw:
        return None
    if raw == "auto":
        if not args.init_ply:
            return None
        candidate = Path(args.init_ply).resolve().parent / "resplat_depth.npz"
        return candidate if candidate.exists() else None
    path = Path(raw)
    if not path.exists():
        raise ValueError(
            f"--depth-npz not found: {path}\n"
            "run `python -m research.export_resplat_depth --print-gpu-command "
            "--work <work> --resplat-out <resplat out>` on the GPU host, then "
            "convert the dump with --raw/--split/--out")
    return path


def compute_pose_window(args):
    """Return (start_iteration, end_iteration) for the refinement window."""
    iterations = int(args.iterations)
    if args.pose_start_step >= 0:
        start = int(args.pose_start_step)
    else:
        start = int(round(float(args.pose_start_frac) * iterations))
    start = max(1, min(iterations, start))
    if args.pose_steps >= 0:
        end = start + int(args.pose_steps) - 1
    else:
        end = iterations
    end = max(1, min(iterations, end))
    if start >= end:
        raise ValueError(
            f"empty pose window: start={start}, end={end} "
            f"(iterations={iterations})")
    return start, end


def validate_args(args):
    if int(args.iterations) < 1:
        raise ValueError("--iterations must be positive")
    if int(args.sh_up_every) < 1:
        raise ValueError("--sh_up_every must be positive")
    if args.limit_seconds is not None:
        if args.limit_seconds <= 0:
            raise ValueError("--limit_seconds must be positive")
        if args.limit_reserve < 0 or args.limit_reserve >= args.limit_seconds:
            raise ValueError("--limit_reserve must be in [0, --limit_seconds)")

    args.res_scales_list, args.res_until_list = parse_schedule(
        args.res_scales, args.res_until)
    if any(scale <= 0 or scale > 1.0 for scale in args.res_scales_list):
        raise ValueError("--res_scales values must be in (0, 1]")
    if args.res_scales_list[-1] != 1.0:
        raise ValueError("--res_scales must end at 1.0")

    if args.lpips_weight < 0:
        raise ValueError("--lpips_weight must be non-negative")
    if args.lpips_every < 1:
        raise ValueError("--lpips_every must be positive")
    if not 0 < args.lpips_scale <= 1.0:
        raise ValueError("--lpips_scale must be in (0, 1]")

    if args.depth_weight < 0:
        raise ValueError("--depth-weight must be non-negative")
    if args.depth_every < 1:
        raise ValueError("--depth-every must be positive")
    if not 0.0 <= args.depth_start_frac <= 1.0:
        raise ValueError("--depth-start-frac must be in [0, 1]")
    if not 0.0 <= args.depth_end_frac <= 1.0:
        raise ValueError("--depth-end-frac must be in [0, 1]")
    if args.depth_start_frac > args.depth_end_frac:
        raise ValueError("--depth-start-frac must not exceed --depth-end-frac")
    if args.depth_alpha_eps <= 0:
        raise ValueError("--depth-alpha-eps must be positive")
    if args.depth_edge_beta < 0:
        raise ValueError("--depth-edge-beta must be non-negative")
    if args.depth_grad_weight < 0:
        raise ValueError("--depth-grad-weight must be non-negative")
    if args.depth_charbonnier_eps <= 0:
        raise ValueError("--depth-charbonnier-eps must be positive")
    if not 0.0 <= args.depth_min_valid < 1.0:
        raise ValueError("--depth-min-valid must be in [0, 1)")
    if args.depth_encode_max < 0:
        raise ValueError("--depth-encode-max must be non-negative or 0 for auto")

    args.depth_start_iter = max(1, int(round(args.depth_start_frac * args.iterations)))
    args.depth_end_iter = max(args.depth_start_iter,
                              int(round(args.depth_end_frac * args.iterations)))

    args.depth_npz_path = resolve_depth_npz(args)

    if args.pose_lr < 0:
        raise ValueError("--pose-lr must be non-negative")
    if not 0.0 <= args.pose_start_frac <= 1.0:
        raise ValueError("--pose-start-frac must be in [0, 1]")
    if args.pose_start_step < -1:
        raise ValueError("--pose-start-step must be -1 or a non-negative iteration")
    if args.pose_steps < -1 or args.pose_steps == 0:
        raise ValueError("--pose-steps must be -1 (until end) or a positive count")
    if args.pose_max_shift < 0:
        raise ValueError("--pose-max-shift must be non-negative")
    if args.pose_weight_decay < 0:
        raise ValueError("--pose-weight-decay must be non-negative")
    args.pose_start_iter = args.pose_end_iter = None
    if args.pose_lr > 0:
        args.pose_start_iter, args.pose_end_iter = compute_pose_window(args)
    return args


# ---------------------------------------------------------------------------
# depth sidecar
# ---------------------------------------------------------------------------

class DepthSidecar:
    """Per-view predicted depth keyed by FastGS ``camera.image_name``."""

    def __init__(self, path):
        self.path = Path(path)
        arrays = {}
        with np.load(self.path, allow_pickle=False) as data:
            for key in data.files:
                value = np.asarray(data[key])
                if value.ndim != 2:
                    raise ValueError(
                        f"{self.path}: sidecar key {key!r} has shape "
                        f"{value.shape}; expected an HxW depth map")
                if not np.isfinite(value).all():
                    raise ValueError(f"{self.path}: non-finite depth for {key!r}")
                arrays[str(key)] = value.astype(np.float32, copy=False)
        if not arrays:
            raise ValueError(f"{self.path}: empty depth sidecar")
        self.arrays = arrays
        self.alias = {}
        for key in arrays:
            for alias in self._aliases(key):
                previous = self.alias.get(alias)
                if previous is not None and previous != key:
                    raise ValueError(
                        f"{self.path}: ambiguous alias {alias!r} for {previous!r} "
                        f"and {key!r}; re-export with --key-style name")
                self.alias[alias] = key
        self._cache = {}

    @staticmethod
    def _aliases(key):
        name = Path(key).name
        return {key, name, Path(key).stem}

    def match(self, image_name):
        for candidate in (str(image_name), Path(str(image_name)).name,
                          Path(str(image_name)).stem):
            if candidate in self.alias:
                return self.alias[candidate]
        return None

    def coverage(self, cameras):
        names = [str(cam.image_name) for cam in cameras]
        matched = [name for name in names if self.match(name) is not None]
        missing = [name for name in names if self.match(name) is None]
        return dict(total=len(names), matched=len(matched),
                    ratio=(len(matched) / len(names)) if names else 0.0,
                    missing=sorted(missing))

    def tensor_for(self, image_name, size, device):
        if torch is None:
            raise RuntimeError("depth tensors need torch")
        key = self.match(image_name)
        if key is None:
            return None
        height, width = int(size[0]), int(size[1])
        cache_key = (key, height, width, str(device))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        array = torch.from_numpy(np.ascontiguousarray(self.arrays[key]))
        array = array.to(device=device, dtype=torch.float32)
        if tuple(array.shape) != (height, width):
            array = F.interpolate(array[None, None], size=(height, width),
                                  mode="bilinear", align_corners=False)[0, 0]
        self._cache[cache_key] = array
        return array


# ---------------------------------------------------------------------------
# rendered depth accumulation (no depth output in the FastGS wrapper)
# ---------------------------------------------------------------------------

def encode_depth_dc(z, encode_max):
    """Pack camera-Z into SH DC so ``SH_C0*dc + 0.5`` recovers ``z/encode_max``.

    R: z/encode_max, G: 1 (alpha accumulation), B: 0 (background 0).
    """
    if encode_max <= 0:
        raise ValueError("encode_max must be positive")
    dc = z.new_empty((z.shape[0], 1, 3))
    dc[:, 0, 0] = (z / encode_max - 0.5) / SH_C0
    dc[:, 0, 1] = 0.5 / SH_C0
    dc[:, 0, 2] = -0.5 / SH_C0
    return dc


def decode_depth_rgb(rgb, encode_max, alpha_eps=1e-6):
    """Inverse of :func:`encode_depth_dc` for a rasterized RGB image [3,H,W].

    The denominator is detached: the depth *value* is still alpha-normalized,
    but gradients do not push opacity up merely to shrink the normalization
    (opacity gradients through the composited numerator remain).
    """
    numerator = rgb[0]
    alpha = rgb[1]
    depth = (numerator / alpha.clamp_min(alpha_eps).detach()) * encode_max
    return depth, alpha


class _DepthColorProxy:
    """Read-only Gaussian proxy that swaps only the SH DC features."""

    def __init__(self, gaussians, dc):
        self._gaussians = gaussians
        self._dc = dc
        self.active_sh_degree = 0

    @property
    def get_xyz(self):
        return self._gaussians.get_xyz

    @property
    def get_opacity(self):
        return self._gaussians.get_opacity

    @property
    def get_scaling(self):
        return self._gaussians.get_scaling

    @property
    def get_rotation(self):
        return self._gaussians.get_rotation

    @property
    def get_features_dc(self):
        return self._dc

    @property
    def get_features_rest(self):
        return self._gaussians.get_features_rest

    @property
    def get_features(self):
        return self._gaussians.get_features

    def get_covariance(self, scaling_modifier=1.0):
        return self._gaussians.get_covariance(scaling_modifier)

    def __getattr__(self, name):
        return getattr(self._gaussians, name)


def render_encoded_depth(cam, gaussians, render_fn, pipe, mult, encode_max,
                         alpha_eps=1e-6):
    """Return (depth [1,1,H,W], alpha [1,1,H,W], render_pkg).

    ``render_fn`` is ``gaussian_renderer.render_fastgs`` on the GPU; passing it
    in keeps this module importable on a CUDA-less machine.
    """
    if getattr(pipe, "convert_SHs_python", False):
        raise RuntimeError(
            "depth accumulation needs the rasterizer SH path; "
            "--convert_SHs_python must be False")
    forward = camera_forward(cam)
    z = (gaussians.get_xyz - cam.camera_center) @ forward
    proxy = _DepthColorProxy(gaussians, encode_depth_dc(z, encode_max))
    background = torch.zeros(3, dtype=z.dtype, device=z.device)
    pkg = render_fn(cam, proxy, pipe, background, mult, 1.0)
    depth, alpha = decode_depth_rgb(pkg["render"], encode_max, alpha_eps)
    return depth[None, None], alpha[None, None], pkg


# ---------------------------------------------------------------------------
# edge-aware depth loss (pure tensor math, CPU-testable)
# ---------------------------------------------------------------------------

def _as_bchw(value, name):
    if value.dim() == 2:
        return value[None, None]
    if value.dim() == 3:
        return value[None]
    if value.dim() == 4:
        return value
    raise ValueError(f"{name} must be [H,W], [1,H,W] or [1,1,H,W], "
                     f"got {tuple(value.shape)}")


def sobel_gradients(image):
    """Sobel dx/dy of a [B,1,H,W] image with replicate padding."""
    if image.dim() != 4 or image.shape[1] != 1:
        raise ValueError("sobel_gradients expects [B,1,H,W]")
    kernel = image.new_tensor([[-1.0, 0.0, 1.0],
                               [-2.0, 0.0, 2.0],
                               [-1.0, 0.0, 1.0]]) / 8.0
    pad_mode = "replicate" if min(image.shape[-2:]) >= 3 else "constant"
    padded = F.pad(image, (1, 1, 1, 1), mode=pad_mode)
    gx = F.conv2d(padded, kernel.view(1, 1, 3, 3))
    gy = F.conv2d(padded, kernel.t().view(1, 1, 3, 3))
    return gx, gy


def edge_aware_depth_loss(rendered_depth, predicted_depth, alpha=None, *,
                          edge_beta=4.0, grad_weight=0.5, alpha_eps=0.01,
                          charbonnier_eps=1e-4, min_valid=0.02):
    """Robust relative depth term + edge-aware gradient-matching term.

    ``rendered_depth`` is FastGS alpha-weighted camera-Z, ``predicted_depth`` is
    the ReSplat prediction in the same units.  Both are only compared where the
    predicted depth is positive, the rendered depth is positive and the render
    alpha exceeds ``alpha_eps`` (unobserved pixels are ignored).  The data term
    is Charbonnier on relative depth; the gradient term uses the predicted depth
    gradient as the edge reference and downweights depth discontinuities
    (``exp(-beta * |grad d_pred| / median(d_pred))``), i.e. it does not force the
    rasterizer's discretized boundaries to sit exactly on the prediction's.

    Returns ``(loss, stats)``.  ``loss`` is differentiable w.r.t.
    ``rendered_depth`` (and therefore Gaussians/cameras); ``stats`` contains
    detached floats for logging.
    """
    rendered = _as_bchw(rendered_depth, "rendered_depth").float()
    predicted = _as_bchw(predicted_depth, "predicted_depth").float()
    if rendered.shape[-2:] != predicted.shape[-2:]:
        raise ValueError(
            f"depth shapes differ: rendered {tuple(rendered.shape[-2:])}, "
            f"predicted {tuple(predicted.shape[-2:])}")
    alpha_t = None
    if alpha is not None:
        alpha_t = _as_bchw(alpha, "alpha").float()
        if alpha_t.shape[-2:] != rendered.shape[-2:]:
            raise ValueError("alpha and depth shapes differ")

    valid = (torch.isfinite(rendered) & torch.isfinite(predicted)
             & (predicted > 0) & (rendered > 0))
    if alpha_t is not None:
        valid = valid & (alpha_t > alpha_eps)
        confidence = alpha_t.clamp(0.0, 1.0).detach()
    else:
        confidence = torch.ones_like(rendered)
    valid_float = valid.to(rendered.dtype)
    fraction = valid_float.mean()
    stats = dict(valid_fraction=float(fraction.detach()),
                 data=0.0, grad=0.0, edge_weight=0.0, skipped=False)
    if not bool(valid.any()) or float(fraction.detach()) < float(min_valid):
        stats["skipped"] = True
        return rendered.sum() * 0.0, stats

    predicted_safe = torch.where(valid, predicted, torch.ones_like(predicted))
    scale = predicted_safe.detach()[valid].median().clamp_min(1e-6)
    relative = (rendered - predicted_safe) / predicted_safe.clamp_min(1e-6)
    charbonnier = torch.sqrt(relative * relative + charbonnier_eps ** 2) - charbonnier_eps
    weight = valid_float * confidence
    data = (charbonnier * weight).sum() / weight.sum().clamp_min(1e-6)

    gx_r, gy_r = sobel_gradients(torch.where(valid, rendered, predicted_safe))
    gx_p, gy_p = sobel_gradients(predicted_safe)
    gradient_magnitude = torch.sqrt(
        (gx_p * gx_p + gy_p * gy_p).clamp_min(0.0)) / scale
    edge_weight = torch.exp(
        -float(edge_beta) * gradient_magnitude.clamp(max=20.0)).detach()
    gradient_error = torch.sqrt((gx_r - gx_p) ** 2 + (gy_r - gy_p) ** 2 + 1e-12) / scale
    grad_weights = edge_weight * weight
    grad = (gradient_error * grad_weights).sum() / grad_weights.sum().clamp_min(1e-6)

    loss = data + float(grad_weight) * grad
    stats.update(data=float(data.detach()), grad=float(grad.detach()),
                 edge_weight=float((edge_weight * valid_float).sum().detach()
                                   / valid_float.sum().clamp_min(1.0)))
    return loss, stats


def _depth_diagnostics(rendered_depth, predicted_depth, alpha, alpha_eps):
    with torch.no_grad():
        alpha = _as_bchw(alpha, "alpha")
        rendered = _as_bchw(rendered_depth, "rendered_depth")
        predicted = _as_bchw(predicted_depth, "predicted_depth")
        mask = (alpha > alpha_eps) & (predicted > 0) & torch.isfinite(rendered)
        if mask.any():
            relative = ((rendered - predicted).abs() / predicted.clamp_min(1e-6))[mask]
            relative = float(relative.median())
        else:
            relative = None
        depth_values = rendered[torch.isfinite(rendered)]
        return dict(alpha_mean=float(alpha.mean()),
                    alpha_covered=float((alpha > alpha_eps).float().mean()),
                    depth_min=float(depth_values.min()) if depth_values.numel() else None,
                    depth_max=float(depth_values.max()) if depth_values.numel() else None,
                    depth_mean=float(depth_values.mean()) if depth_values.numel() else None,
                    median_relative_error=relative)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def quick_test_psnr(scene, gaussians, pipe, background, render_fn, psnr_fn):
    psnrs = []
    with torch.no_grad():
        for cam in scene.getTestCameras():
            out = render_fn(cam, gaussians, pipe, background, 1.0, 1.0)
            img = torch.clamp(out["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
            psnrs.append(psnr_fn(img, gt).mean().item())
    return float(np.mean(psnrs)) if psnrs else None


def _load_trainer():
    from gaussian_renderer import render_fastgs
    from scene import Scene, GaussianModel
    from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
    from utils.general_utils import safe_state
    from utils.image_utils import psnr
    from utils.loss_utils import l1_loss
    try:
        from fused_ssim import fused_ssim as fast_ssim
    except ImportError:  # pragma: no cover
        fast_ssim = None
    return dict(render_fastgs=render_fastgs, Scene=Scene,
                GaussianModel=GaussianModel,
                compute_gaussian_score_fastgs=compute_gaussian_score_fastgs,
                sampling_cameras=sampling_cameras, safe_state=safe_state,
                psnr=psnr, l1_loss=l1_loss, fast_ssim=fast_ssim)


def _mean(values):
    return float(np.mean(values)) if values else None


def run(args):
    if torch is None:
        raise SystemExit("research.depth_reg training needs torch; "
                         "use --self-check on a CUDA-less machine")
    trainer = _load_trainer()
    render_fastgs = trainer["render_fastgs"]
    Scene = trainer["Scene"]
    GaussianModel = trainer["GaussianModel"]
    compute_gaussian_score_fastgs = trainer["compute_gaussian_score_fastgs"]
    sampling_cameras = trainer["sampling_cameras"]
    safe_state = trainer["safe_state"]
    psnr = trainer["psnr"]
    l1_loss = trainer["l1_loss"]
    fast_ssim = trainer["fast_ssim"]

    # Argument classes live in the FastGS package; rebuild them to extract the
    # same Namespace objects train_sweep.py uses.
    ModelParams, OptimizationParams, PipelineParams = _fastgs_arguments()
    parser = argparse.ArgumentParser(add_help=False)
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    opt = op.extract(args)
    iterations = opt.iterations
    sh_up_every = args.sh_up_every
    if opt.opacity_reset_interval <= 0:
        opt.opacity_reset_interval = iterations + 1

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    seed_all(args.seed)
    safe_state(args.quiet)
    torch.cuda.empty_cache()

    scales = args.res_scales_list
    cuts = args.res_until_list

    t_total0 = time.perf_counter()

    # ---- depth sidecar (loaded before the scene; part of module-B setup) ---
    depth_path = args.depth_npz_path
    sidecar = None
    depth_load_s = 0.0
    if depth_path is not None:
        t0 = time.perf_counter()
        sidecar = DepthSidecar(depth_path)
        depth_load_s = time.perf_counter() - t0
        print(f"[depth] sidecar {depth_path} ({len(sidecar.arrays)} views, "
              f"{depth_load_s:.2f}s)")

    if args.densify_count_thresh is not None or args.max_gaussians is not None:
        from .dense_model import AggressiveGaussianModel as model_class
        if args.densify_count_thresh is not None:
            model_class.densify_count_thresh = args.densify_count_thresh
        if args.max_gaussians is not None:
            model_class.max_gaussians = args.max_gaussians
        print(f"[sweep] densify gate > {model_class.densify_count_thresh} sampled views, "
              f"cap {model_class.max_gaussians}")
    else:
        model_class = GaussianModel
    gaussians = model_class(dataset.sh_degree, opt.optimizer_type)
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "cfg_args"), "w") as stream:
        stream.write(str(argparse.Namespace(**vars(dataset))))

    t0 = time.perf_counter()
    scene = Scene(dataset, gaussians, resolution_scales=scales)
    scene_load_s = time.perf_counter() - t0

    cams_by_scale = {s: scene.getTrainCameras(s) for s in scales}

    # ---- init (identical to train_sweep.py) --------------------------------
    t0 = time.perf_counter()
    if args.init_ply:
        gaussians.load_ply(args.init_ply)
        init_mode = f"ply:{os.path.basename(args.init_ply)}"
    elif args.init_random:
        cams = scene.getTrainCameras()
        r = max(scene.cameras_extent, 1e-6)
        per = max(args.init_random_num // len(cams), 1)
        points, colors = [], []
        for cam in cams:
            c2w = cam.world_view_transform.inverse().cpu().numpy()
            rotation, center = c2w[:3, :3], c2w[:3, 3]
            us = np.random.rand(per) * 2 - 1
            vs = np.random.rand(per) * 2 - 1
            depths = np.exp(np.random.uniform(np.log(0.05 * r), np.log(1.5 * r), per))
            tanx, tany = np.tan(cam.FoVx / 2), np.tan(cam.FoVy / 2)
            dirs = np.stack([us * tanx * depths, vs * tany * depths, depths], axis=1)
            points.append((rotation @ dirs.T).T + center)
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
    gaussians.spatial_lr_scale = scene.cameras_extent
    gaussians.training_setup(opt)
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    init_s = time.perf_counter() - t0 + depth_load_s

    # ---- depth coverage and leakage guard ----------------------------------
    coverage = None
    depth_enabled = sidecar is not None and args.depth_weight > 0
    if sidecar is not None:
        test_names = {str(cam.image_name) for cam in scene.getTestCameras()}
        leaked = sorted(name for name in test_names if sidecar.match(name) is not None)
        if leaked:
            raise SystemExit(
                f"[depth] sidecar covers held-out views {leaked[:5]}; refusing to train")
        coverage = sidecar.coverage(cams_by_scale[scales[-1]])
        print(f"[depth] coverage {coverage['matched']}/{coverage['total']} train views "
              f"({coverage['ratio'] * 100:.1f}%)")
        if coverage["matched"] == 0:
            print("[depth] WARNING: no train view matches the sidecar; depth loss disabled")
            depth_enabled = False
    if args.depth_weight > 0 and not depth_enabled:
        print("[depth] depth loss disabled (no usable sidecar)")
    if not args.quiet and depth_enabled:
        print(f"[depth] weight={args.depth_weight} every={args.depth_every} "
              f"iters={args.depth_start_iter}..{args.depth_end_iter} "
              f"beta={args.depth_edge_beta} grad_weight={args.depth_grad_weight}")

    encode_max = args.depth_encode_max
    if encode_max <= 0:
        encode_max = max(1000.0, 20.0 * float(scene.cameras_extent))
    if depth_enabled:
        print(f"[depth] encoded camera-Z range 0..{encode_max:.1f}")

    # ---- pose refinement window --------------------------------------------
    refiner = None
    pose_start = args.pose_start_iter
    pose_end = args.pose_end_iter
    if args.pose_lr > 0:
        all_cams = [cam for scale in scales for cam in cams_by_scale[scale]]
        max_shift = args.pose_max_shift
        if max_shift <= 0:
            max_shift = 0.02 * float(scene.cameras_extent)
        refiner = PoseRefiner(all_cams, lr=args.pose_lr, max_shift=max_shift,
                             weight_decay=args.pose_weight_decay)
        print(f"[pose] {refiner.num_views} views, lr={args.pose_lr}, "
              f"iters={pose_start}..{pose_end}, max_shift={max_shift:.4f}")

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

    depth_applied = 0
    depth_skipped = 0
    depth_missing = 0
    depth_loss_sum = 0.0
    depth_frac_sum = 0.0
    depth_diagnostics_printed = False

    t_train0 = time.perf_counter()
    stop_reason = "iterations"
    progress = None
    from tqdm import tqdm
    progress = tqdm(range(1, iterations + 1), desc="FastGS", disable=args.quiet,
                    file=sys.stdout, mininterval=1.0)
    iteration = 0
    for iteration in range(1, iterations + 1):
        if args.limit_seconds is not None:
            spent = time.perf_counter() - t_train0
            if spent >= args.limit_seconds - args.limit_reserve:
                stop_reason = "wall_clock"
                break
        gaussians.update_learning_rate(iteration)
        if iteration % sh_up_every == 0:
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

        pose_active = (refiner is not None and pose_start is not None
                       and pose_start <= iteration <= pose_end)
        # Translation refinement is applied in Gaussian space (see
        # pose_refine.py): rendering the world shifted by -delta with the base
        # camera is exactly the same image/depth as moving the camera by +delta,
        # and the rasterizer backpropagates through means3D (not the view matrix).
        render_gaussians = (refiner.wrap(viewpoint_cam, gaussians)
                            if pose_active else gaussians)

        render_pkg = render_fastgs(viewpoint_cam, render_gaussians, pipe, background,
                                   opt.mult, 1.0)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if fast_ssim is not None:
            ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = 1.0
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

        depth_step = (depth_enabled
                      and args.depth_start_iter <= iteration <= args.depth_end_iter
                      and iteration % args.depth_every == 0)
        if depth_step:
            depth_target = sidecar.tensor_for(viewpoint_cam.image_name,
                                              image.shape[-2:], image.device)
            if depth_target is None:
                depth_missing += 1
            else:
                rendered_depth, rendered_alpha, _ = render_encoded_depth(
                    viewpoint_cam, render_gaussians, render_fastgs, pipe, opt.mult,
                    encode_max, alpha_eps=1e-6)
                if not depth_diagnostics_printed:
                    diagnostics = _depth_diagnostics(
                        rendered_depth, depth_target[None, None], rendered_alpha,
                        args.depth_alpha_eps)
                    print(f"[depth] render sanity (first supervised view "
                          f"{viewpoint_cam.image_name}): {diagnostics}")
                    depth_diagnostics_printed = True
                depth_loss, depth_stats = edge_aware_depth_loss(
                    rendered_depth, depth_target[None, None], rendered_alpha,
                    edge_beta=args.depth_edge_beta,
                    grad_weight=args.depth_grad_weight,
                    alpha_eps=args.depth_alpha_eps,
                    charbonnier_eps=args.depth_charbonnier_eps,
                    min_valid=args.depth_min_valid)
                if depth_stats["skipped"]:
                    depth_skipped += 1
                else:
                    loss = loss + args.depth_weight * depth_loss
                    depth_applied += 1
                    depth_loss_sum += float(depth_loss.detach())
                    depth_frac_sum += float(depth_stats["valid_fraction"])

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

            gaussians.optimizer_step(iteration)

        if pose_active:
            refiner.step()

        if iteration in trigger:
            elapsed = time.perf_counter() - t_train0
            entry = dict(iteration=iteration, train_seconds=elapsed,
                         num_gaussians=int(gaussians.get_xyz.shape[0]))
            if args.snap_dir:
                path = os.path.join(args.snap_dir, f"iter{iteration:05d}.ply")
                gaussians.save_ply(path)
                entry["ply"] = path
            snaps.append(entry)

    if progress is not None:
        progress.close()
    train_s = time.perf_counter() - t_train0
    if refiner is not None:
        if stop_reason == "wall_clock" and pose_start is not None and iteration < pose_start:
            print(f"[pose] WARNING: stopped at iteration {iteration} before the pose "
                  f"window starts at {pose_start}; lower --pose-start-frac or "
                  "use --pose-start-step")

    final_ply = None
    if not args.no_save_final:
        final_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}")
        os.makedirs(final_dir, exist_ok=True)
        final_ply = os.path.join(final_dir, "point_cloud.ply")
        gaussians.save_ply(final_ply)
    final_psnr = None
    try:
        final_psnr = quick_test_psnr(scene, gaussians, pipe, background,
                                     render_fastgs, psnr)
    except Exception as error:  # noqa: BLE001
        print(f"[sweep] quick psnr failed: {error}")
    total_s = time.perf_counter() - t_total0

    depth_record = dict(
        npz=str(depth_path) if depth_path is not None else None,
        enabled=bool(depth_enabled), coverage=coverage,
        weight=args.depth_weight, every=args.depth_every,
        start_iteration=args.depth_start_iter, end_iteration=args.depth_end_iter,
        alpha_eps=args.depth_alpha_eps, edge_beta=args.depth_edge_beta,
        grad_weight=args.depth_grad_weight,
        charbonnier_eps=args.depth_charbonnier_eps,
        min_valid=args.depth_min_valid, encode_max=encode_max,
        applied_steps=depth_applied, skipped_steps=depth_skipped,
        missing_view_steps=depth_missing,
        mean_loss=(depth_loss_sum / depth_applied) if depth_applied else None,
        mean_valid_fraction=(depth_frac_sum / depth_applied) if depth_applied else None)
    pose_record = dict(enabled=bool(refiner is not None),
                       start_iteration=pose_start, end_iteration=pose_end,
                       lr=args.pose_lr, steps=args.pose_steps,
                       start_frac=args.pose_start_frac,
                       max_shift=(refiner.max_shift if refiner is not None else None))
    if refiner is not None:
        pose_record.update(refiner.summary())

    record = dict(
        trainer="depth_reg",
        init_ply=str(args.init_ply), init_mode=init_mode,
        num_init_gaussians=n_init, num_final_gaussians=int(gaussians.get_xyz.shape[0]),
        iterations=iterations, completed_iterations=iteration, stop_reason=stop_reason,
        final_ply=final_ply,
        limit_seconds=args.limit_seconds, snapshots=snaps,
        scene_load_seconds=round(scene_load_s, 3),
        depth_load_seconds=round(depth_load_s, 3),
        init_seconds=round(init_s, 3), train_seconds=round(train_s, 3),
        total_seconds=round(total_s, 3),
        iterations_per_second=round(iteration / train_s, 2) if train_s else None,
        quick_test_psnr=round(final_psnr, 4) if final_psnr is not None else None,
        num_train_views=len(cams_by_scale[scales[-1]]),
        num_test_views=len(scene.getTestCameras()), res_scales=scales, res_until=cuts,
        depth=depth_record, pose=pose_record,
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
                      max_gaussians=args.max_gaussians,
                      depth_weight=args.depth_weight,
                      depth_every=args.depth_every,
                      depth_start_frac=args.depth_start_frac,
                      depth_end_frac=args.depth_end_frac,
                      pose_lr=args.pose_lr,
                      pose_start_frac=args.pose_start_frac,
                      pose_steps=args.pose_steps,
                      pose_max_shift=args.pose_max_shift))
    if args.record:
        os.makedirs(os.path.dirname(args.record), exist_ok=True)
        with open(args.record, "w") as stream:
            json.dump(record, stream, indent=2)
    print("[depth_reg] " + json.dumps({k: v for k, v in record.items()
                                       if k != "snapshots"}))
    return record


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def _expect_value_error(call, label, checks):
    try:
        call()
    except ValueError:
        return checks + 1
    raise AssertionError(f"self-check: expected ValueError for {label}")


def self_check():
    checks = 0

    def check(condition, message):
        nonlocal checks
        if not condition:
            raise AssertionError(f"self-check: {message}")
        checks += 1

    import tempfile

    parser = build_parser()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sidecar_path = tmp / "sidecar_fixture.npz"
        rng = np.random.default_rng(5)
        depth_arrays = {f"{i:04d}": np.exp(rng.normal(0, .2, (12, 16))).astype(np.float32)
                        for i in range(8)}
        np.savez_compressed(sidecar_path, **depth_arrays)
        base = ["--source", str(tmp / "work"), "--model_path", str(tmp / "model"),
                "--init_ply", str(tmp / "init.ply"), "--iterations", "1000",
                "--depth-npz", str(sidecar_path)]
        args = validate_args(parser.parse_args(base))
        check(args.iterations == 1000 and args.depth_weight == 0.05,
              "depth defaults parse")
        check(args.depth_start_iter == 50 and args.depth_end_iter == 800,
              "depth window rounds from fractions")
        check(args.depth_npz_path == sidecar_path, "explicit sidecar resolved")
        check(args.pose_lr == 0.0 and args.pose_start_iter is None,
              "pose off by default")
        check(args.res_scales_list == [1.0] and args.res_until_list == [],
              "single-resolution default")

        args = validate_args(parser.parse_args(base + [
            "--pose-lr", "1e-4", "--pose-start-frac", "0.5",
            "--pose-steps", "100"]))
        check(args.pose_start_iter == 500 and args.pose_end_iter == 599,
              "pose window from fraction + M steps")
        args = validate_args(parser.parse_args(base + [
            "--pose-lr", "1e-4", "--pose-start-step", "700"]))
        check(args.pose_start_iter == 700 and args.pose_end_iter == 1000,
              "pose window from absolute start to the end")

        checks = _expect_value_error(
            lambda: validate_args(parser.parse_args(base + ["--depth-weight", "-1"])),
            "negative depth weight", checks)
        checks = _expect_value_error(
            lambda: validate_args(parser.parse_args(base + [
                "--depth-start-frac", "0.9", "--depth-end-frac", "0.2"])),
            "inverted depth window", checks)
        checks = _expect_value_error(
            lambda: validate_args(parser.parse_args([
                "--source", str(tmp / "w"), "--model_path", str(tmp / "m"),
                "--init_ply", str(tmp / "i.ply"), "--iterations", "100",
                "--depth-npz", str(tmp / "missing.npz")])),
            "missing sidecar", checks)
        checks = _expect_value_error(
            lambda: validate_args(parser.parse_args(base + [
                "--pose-lr", "1e-4", "--pose-start-frac", "1.0"])),
            "pose window starts at/after the end", checks)
        checks = _expect_value_error(
            lambda: validate_args(parser.parse_args(base + ["--res_scales", "0.5"])),
            "schedule must end at 1.0", checks)

        auto = validate_args(parser.parse_args(base + ["--depth-npz", "auto"]))
        check(auto.depth_npz_path is None, "auto sidecar absent -> control run")
        (tmp / "init.ply").write_bytes(b"")
        np.savez_compressed(tmp / "resplat_depth.npz",
                            **{"0000": np.ones((4, 4), dtype=np.float32)})
        auto = validate_args(parser.parse_args(base + ["--depth-npz", "auto"]))
        check(auto.depth_npz_path == (tmp / "resplat_depth.npz").resolve(),
              "auto sidecar found next to init PLY")

        # ---- sidecar alias/resolution math ---------------------------------
        sidecar = DepthSidecar(sidecar_path)
        check(sidecar.match("0001") == "0001", "stem lookup")
        check(sidecar.match("0001.png") == "0001", "filename lookup")
        outside = type("Cam", (), {"image_name": "9999"})()
        inside = type("Cam", (), {"image_name": "0003"})()
        coverage = sidecar.coverage([inside, outside])
        check(coverage["matched"] == 1 and coverage["ratio"] == 0.5,
              "coverage accounting")
        ambiguous = tmp / "ambiguous.npz"
        np.savez_compressed(ambiguous,
                            **{"a/0001.png": np.ones((4, 4), dtype=np.float32),
                               "b/0001.jpg": np.ones((4, 4), dtype=np.float32)})
        checks = _expect_value_error(lambda: DepthSidecar(ambiguous),
                                     "ambiguous sidecar aliases", checks)
        if torch is not None:
            tensor = sidecar.tensor_for("0003.png", (6, 8), "cpu")
            check(tuple(tensor.shape) == (6, 8) and tensor.dtype == torch.float32,
                  "sidecar resized to the render resolution")
            check(sidecar.tensor_for("9999", (6, 8), "cpu") is None,
                  "unknown view has no depth")

    if torch is None:
        print(f"self-check OK: {checks} checks (arguments + sidecar; "
              "torch unavailable, loss math skipped)")
        return checks

    # ---- loss math ---------------------------------------------------------
    def close(a, b, tol=1e-6):
        return abs(float(a) - float(b)) <= tol

    predicted = torch.full((1, 1, 16, 16), 2.0)
    identical = predicted.clone()
    alpha = torch.ones_like(predicted)
    loss_same, stats = edge_aware_depth_loss(identical, predicted, alpha,
                                             edge_beta=4.0, grad_weight=0.5)
    check(float(loss_same) < 1e-3, "identical depth has ~zero loss")
    check(stats["valid_fraction"] == 1.0 and not stats["skipped"],
          "loss stats report full validity")

    shifted = predicted + 0.2
    loss_shift, _ = edge_aware_depth_loss(shifted, predicted, alpha)
    check(float(loss_shift) > float(loss_same), "depth offset increases the loss")

    masked_alpha = torch.ones_like(predicted)
    masked_alpha[:, :, :, 8:] = 0.0
    wild = predicted.clone()
    wild[:, :, :, 8:] = 1e6
    loss_masked, masked_stats = edge_aware_depth_loss(
        wild, predicted, masked_alpha, min_valid=0.0)
    check(close(masked_stats["valid_fraction"], 0.5),
          "alpha mask removes unobserved pixels")
    check(torch.isfinite(loss_masked).all().item(), "masked loss stays finite")
    loss_masked_ref, _ = edge_aware_depth_loss(
        predicted, predicted, masked_alpha, min_valid=0.0)
    check(float(loss_masked) < 1e-2,
          "masked corruption does not leak through the mask")

    skipped_rendered = predicted.clone().requires_grad_(True)
    skipped_loss, skipped_stats = edge_aware_depth_loss(
        skipped_rendered, predicted, torch.zeros_like(alpha), min_valid=0.0)
    check(skipped_stats["skipped"] and float(skipped_loss) == 0.0,
          "zero alpha -> skipped zero loss")
    skipped_loss.backward()
    check(skipped_rendered.grad is not None, "skipped loss keeps a gradient path")

    # edge-aware gradient term: shifting a pixel inside a flat region should be
    # punished harder than shifting the predicted discontinuity itself.
    step_pred = torch.ones((1, 1, 16, 16))
    step_pred[:, :, :, 8:] = 2.0
    flat_perturb = step_pred.clone()
    flat_perturb[:, :, :, 3] += 0.2
    edge_perturb = step_pred.clone()
    edge_perturb[:, :, :, 8] += 0.2
    _, stats_flat = edge_aware_depth_loss(
        flat_perturb, step_pred, alpha, edge_beta=8.0, grad_weight=1.0)
    _, stats_edge = edge_aware_depth_loss(
        edge_perturb, step_pred, alpha, edge_beta=8.0, grad_weight=1.0)
    check(stats_edge["grad"] < stats_flat["grad"],
          "edge-aware term downweights perturbations at predicted edges")
    _, stats_flat_pred = edge_aware_depth_loss(
        predicted, predicted, alpha, edge_beta=8.0, grad_weight=1.0)
    check(close(stats_flat_pred["edge_weight"], 1.0),
          "flat predicted depth has unit edge weight")

    # gradient flows into the rendered depth (and therefore Gaussians/cameras)
    rendered = (predicted + 0.1).clone().requires_grad_(True)
    camera_delta = torch.zeros(3, requires_grad=True)
    center = torch.zeros(3)
    forward = torch.tensor([0.0, 0.0, 1.0])
    z = forward @ (torch.tensor([0.0, 0.0, 2.0]) - (center + camera_delta))
    loss, _ = edge_aware_depth_loss(rendered, predicted, alpha)
    (loss + z).backward()
    check(rendered.grad is not None and torch.isfinite(rendered.grad).all(),
          "rendered-depth gradient is finite")
    check(camera_delta.grad is not None and torch.isfinite(camera_delta.grad).all(),
          "camera-translation gradient is finite")

    # encode/decode round-trip through the standard SH DC affine
    z_values = torch.tensor([0.5, 2.0, 37.5])
    encode_max = 100.0
    dc = encode_depth_dc(z_values, encode_max)
    rgb_red = SH_C0 * dc[:, 0, 0] + 0.5
    rgb_green = SH_C0 * dc[:, 0, 1] + 0.5
    rgb_blue = SH_C0 * dc[:, 0, 2] + 0.5
    check(bool(torch.allclose(rgb_green, torch.ones_like(rgb_green), atol=1e-6)),
          "green channel encodes alpha = 1")
    check(bool(torch.allclose(rgb_blue, torch.zeros_like(rgb_blue), atol=1e-6)),
          "blue channel is background")
    depth_image = torch.stack([rgb_red, rgb_green, rgb_blue])[:, :, None]
    decoded, decoded_alpha = decode_depth_rgb(depth_image, encode_max)
    check(bool(torch.allclose(decoded[:, 0], z_values, rtol=1e-5)),
          "encoded depth round-trips")
    check(bool(torch.allclose(decoded_alpha, torch.ones_like(decoded_alpha))),
          "decoded alpha is 1")

    # the proxy only swaps SH DC; geometry/opacity are the real tensors
    xyz = torch.nn.Parameter(torch.zeros(4, 3))
    proxy = _DepthColorProxy(type("G", (), {
        "get_xyz": property(lambda self: xyz),
        "get_opacity": property(lambda self: torch.ones(4, 1)),
        "get_scaling": property(lambda self: torch.ones(4, 3)),
        "get_rotation": property(lambda self: torch.ones(4, 4)),
        "get_features_rest": property(lambda self: torch.zeros(4, 15, 3)),
        "get_features": property(lambda self: torch.zeros(4, 16, 3)),
    })(), encode_depth_dc(torch.ones(4), 10.0))
    check(proxy.active_sh_degree == 0 and proxy.get_xyz is xyz,
          "depth proxy swaps color only")

    # end-to-end encoded depth render with a fake rasterizer (no CUDA): the
    # fake applies the standard SH DC affine and returns one composited pixel.
    point = torch.tensor([[0.1, 0.2, 3.0]])
    gaussians = type("FakeGaussians", (), {
        "get_xyz": property(lambda self: point),
    })()

    def fake_render(cam, pc, pipe, bg, mult, scaling_modifier=1.0):
        colors = SH_C0 * pc.get_features_dc[:, 0, :] + 0.5
        return {"render": colors.mean(dim=0).view(3, 1, 1)}

    class FakePipe:
        convert_SHs_python = False

    from research.pose_refine import _make_camera

    cam = _make_camera([1.0, -2.0, 0.5], [0.2, 0.9, 0.3], "fake")
    forward = torch.inverse(cam.world_view_transform)[2, :3]
    expected_depth = forward @ (point[0] - cam.camera_center)
    rendered_depth, rendered_alpha, pkg = render_encoded_depth(
        cam, gaussians, fake_render, FakePipe(), 0.5, 100.0)
    check(rendered_depth.shape == (1, 1, 1, 1) and
          bool(torch.allclose(rendered_depth.flatten(), expected_depth[None],
                              rtol=1e-5)),
          "encoded depth render recovers camera Z on a rotated camera")
    check(bool(torch.allclose(rendered_alpha.flatten(), torch.tensor([1.0]))),
          "encoded depth render recovers alpha")

    print(f"self-check OK: {checks} checks "
          "(arguments, sidecar, edge-aware depth loss, depth encoding, proxy, "
          "fake encoded-depth render)")
    return checks


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    try:
        args = validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
