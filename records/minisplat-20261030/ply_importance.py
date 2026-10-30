"""Mini-Splatting-style importance scoring and importance-based pruning.

Mini-Splatting (Fang & Wang, ECCV 2024) is a *density-control* method: keep a
constrained total number of Gaussians and prune the primitives that carry the
least rendered evidence, instead of following FastGS's multi-view heuristic.
This module is the pruning half, kept separate from `research/ply_ops.py`:

  * ``score_importance`` accumulates, over a fixed set of TRAIN views only, the
    per-Gaussian ``|d loss / d mu_2d|`` norm (the 2D-projected gradient of the
    photometric loss with respect to the screen-space mean).  This is the
    "importance / visibility" evidence the paper prunes by, and it needs no
    test images.
  * ``prune_gaussians`` removes the least important rows in place, through the
    pinned FastGS ``GaussianModel.prune_points`` (so the optimizer state,
    densification statistics and ``max_radii2D`` stay consistent).
  * ``main`` is a standalone CLI: load a delivery PLY, score it, write a pruned
    PLY + a JSON record (used for the M2 chain and for diagnostics).

Typical use (from the repository root, fastgs conda env):

  python -m research.ply_importance --work <prepared scene> \
      --ply <delivery point_cloud.ply> --out-ply /tmp/pruned.ply \
      --keep 0.7 --views 24 --record /tmp/prune.json
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

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "competition/upstream/baseline"
sys.path.insert(0, str(BASELINE / "fastgs"))

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402

try:
    from fused_ssim import fused_ssim as fast_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:  # pragma: no cover
    fast_ssim = None
    FUSED_SSIM_AVAILABLE = False


def _zero_param_grads(gaussians):
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity",
                 "_scaling", "_rotation"):
        tensor = getattr(gaussians, name, None)
        if tensor is not None and getattr(tensor, "grad", None) is not None:
            tensor.grad = None


def score_importance(gaussians, cams, pipe, background, opt, n_views=24,
                     rng=None):
    """Per-Gaussian importance = sum over train views of ||d loss / d mu_2d||.

    No optimizer step and no parameter-gradient carry-over: every view is a pure
    forward+backward on the current model.  Returns ``(score float32[n], vis
    float32[n])`` on CPU; ``vis`` counts the views in which the Gaussian had a
    nonzero projected radius.
    """
    rng = rng or random
    n = int(gaussians.get_xyz.shape[0])
    accum = torch.zeros((n, 1), device="cuda")
    vis = torch.zeros((n, 1), device="cuda")
    pool = list(cams)
    picks = []
    for _ in range(min(int(n_views), len(pool))):
        picks.append(pool.pop(rng.randint(0, len(pool) - 1)))
    for cam in picks:
        _zero_param_grads(gaussians)
        pkg = render_fastgs(cam, gaussians, pipe, background, opt.mult, 1.0)
        image = pkg["render"]
        gt = cam.original_image.cuda()
        loss = (1.0 - opt.lambda_dssim) * l1_loss(image, gt)
        if FUSED_SSIM_AVAILABLE:
            loss = loss + opt.lambda_dssim * (1.0 - fast_ssim(image.unsqueeze(0),
                                                             gt.unsqueeze(0)))
        loss.backward()
        viewspace = pkg["viewspace_points"]
        if viewspace.grad is not None:
            accum += viewspace.grad[:, :2].norm(dim=-1, keepdim=True)
        vis += (pkg["radii"] > 0).float().unsqueeze(1)
    _zero_param_grads(gaussians)
    return (accum.view(-1).float().cpu().numpy(),
            vis.view(-1).float().cpu().numpy())


def prune_gaussians(gaussians, keep_fraction, score):
    """Physically remove the (1-keep_fraction) least important Gaussians."""
    n_before = int(gaussians.get_xyz.shape[0])
    keep = int(round(float(keep_fraction) * n_before))
    keep = max(1, min(keep, n_before))
    s = torch.from_numpy(np.asarray(score, dtype=np.float32)).to("cuda")
    if s.numel() != n_before:
        raise ValueError("score length %d != gaussian count %d"
                         % (s.numel(), n_before))
    low = torch.argsort(s)[: n_before - keep]
    mask = torch.zeros(n_before, dtype=torch.bool, device="cuda")
    mask[low] = True
    gaussians.prune_points(mask)
    n_after = int(gaussians.get_xyz.shape[0])
    return dict(n_before=n_before, n_after=n_after,
                keep_requested=float(keep_fraction),
                keep_actual=(n_after / n_before if n_before else 1.0))


def load_for_scoring(work, ply, sh_degree=3, white_background=False):
    """Build a minimal FastGS scene + Gaussians from a delivery PLY (CLI path)."""
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    argv = ["--source_path", str(work), "--model_path", "/tmp/ply_importance_model",
            "--sh_degree", str(sh_degree), "--eval", "-r", "1"]
    if white_background:
        argv.append("--white_background")
    args = parser.parse_args(argv)
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    os.makedirs(dataset.model_path, exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, resolution_scales=[1.0])
    gaussians.load_ply(str(ply))
    return scene, gaussians, pipe, opt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, help="prepared scene dir")
    parser.add_argument("--ply", required=True, help="delivery point_cloud.ply")
    parser.add_argument("--out-ply", required=True, dest="out_ply")
    parser.add_argument("--keep", type=float, default=0.7,
                        help="fraction of Gaussians to keep (0, 1]")
    parser.add_argument("--views", type=int, default=24,
                        help="number of TRAIN views used to accumulate importance")
    parser.add_argument("--sh-degree", type=int, default=3, dest="sh_degree")
    parser.add_argument("--white-background", action="store_true",
                        dest="white_background")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--record", default=None, help="JSON record path")
    parsed = parser.parse_args(argv)
    if not 0.0 < parsed.keep <= 1.0:
        parser.error("--keep must be in (0, 1]")
    random.seed(parsed.seed)
    torch.manual_seed(parsed.seed)
    t0 = time.perf_counter()
    scene, gaussians, pipe, opt = load_for_scoring(parsed.work, parsed.ply,
                                                   parsed.sh_degree,
                                                   parsed.white_background)
    load_s = time.perf_counter() - t0
    background = torch.tensor([1, 1, 1] if parsed.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    t0 = time.perf_counter()
    score, vis = score_importance(gaussians, scene.getTrainCameras(), pipe,
                                  background, opt, parsed.views)
    score_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    stats = prune_gaussians(gaussians, parsed.keep, score)
    prune_s = time.perf_counter() - t0
    os.makedirs(os.path.dirname(os.path.abspath(parsed.out_ply)), exist_ok=True)
    t0 = time.perf_counter()
    gaussians.save_ply(parsed.out_ply)
    save_s = time.perf_counter() - t0
    record = dict(module="research.ply_importance", work=str(parsed.work),
                  ply=str(parsed.ply), out_ply=str(parsed.out_ply),
                  keep=parsed.keep, views=parsed.views,
                  load_seconds=round(load_s, 3),
                  score_seconds=round(score_s, 3),
                  prune_seconds=round(prune_s, 3),
                  save_seconds=round(save_s, 3),
                  score_mean=float(np.mean(score)), score_max=float(np.max(score)),
                  visible_views_mean=float(np.mean(vis)),
                  **stats)
    if parsed.record:
        os.makedirs(os.path.dirname(os.path.abspath(parsed.record)), exist_ok=True)
        with open(parsed.record, "w") as fh:
            json.dump(record, fh, indent=2)
    print("[ply_importance] " + json.dumps(record))
    return record


if __name__ == "__main__":
    main()
