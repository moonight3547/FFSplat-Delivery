"""Pre-module-B train-view LPIPS gate for the optional M1 mid-training prune.

Why this gate exists: Mini-Splatting-style pruning at iteration 1250 (M1) cuts
module-B time without a scoring loss ONLY on scenes whose delivered LPIPS is
already past the proxy's 0.25 cliff (the LPIPS term is then 0 and cannot get
worse).  This gate estimates that condition BEFORE module B starts, from the
ReSplat init PLY and TRAIN views only, so it costs nothing in the scored
module-B budget and never reads held-out RGB.

  python -m research.prune_gate --work <prepared scene> \
      --ply <resplat_world.ply> --tau 0.25 --keep 0.6 --record gate.json

Outputs {"train_lpips": ..., "keep": 0.6 or 1.0, "n_views": 8,
         "train_views_only": true, "tau": ...}.
"""
import argparse
import json
import sys
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

try:
    from fused_ssim import fused_ssim as fast_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:  # pragma: no cover
    fast_ssim = None
    FUSED_SSIM_AVAILABLE = False


def load_scene(work, ply, sh_degree, white_background):
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    argv = ["--source_path", str(work), "--model_path", "/tmp/prune_gate_model",
            "--sh_degree", str(sh_degree), "--eval", "-r", "1"]
    if white_background:
        argv.append("--white_background")
    args = parser.parse_args(argv)
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    Path(dataset.model_path).mkdir(parents=True, exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    # Scene's training default randomly shuffles cameras.  The gate must not use
    # that default: otherwise the "fixed" eight indices select different views
    # on every invocation and a scene can randomly cross the prune threshold.
    scene = Scene(dataset, gaussians, shuffle=False, resolution_scales=[1.0])
    gaussians.load_ply(str(ply))
    return scene, gaussians, pipe, dataset


def fixed_view_indices(n, count):
    """Deterministic, evenly spaced TRAIN-view indices (no RNG, no test views)."""
    count = min(int(count), n)
    return [int(round(i * (n - 1) / max(1, count - 1))) for i in range(count)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--tau", type=float, default=0.25,
                        help="train-LPIPS threshold above which M1 pruning is enabled")
    parser.add_argument("--keep", type=float, default=0.6,
                        help="keep fraction passed to --mid_prune_keep when the gate fires")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--sh-degree", type=int, default=3, dest="sh_degree")
    parser.add_argument("--white-background", action="store_true", dest="white_background")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--record", default=None)
    args = parser.parse_args(argv)
    torch.cuda.set_device(torch.device(args.device))
    white_bg = bool(args.white_background)
    scene, gaussians, pipe, dataset = load_scene(args.work, args.ply,
                                                 args.sh_degree, white_bg)
    train_cams = scene.getTrainCameras()
    picks = fixed_view_indices(len(train_cams), args.views)
    import lpips as lpips_pkg
    perceptual = lpips_pkg.LPIPS(net="vgg", verbose=False).cuda().eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)
    background = torch.tensor([1, 1, 1] if white_bg else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    lpips_vals, psnr_vals, ssim_vals = [], [], []
    with torch.no_grad():
        for idx in picks:
            cam = train_cams[idx]
            image = torch.clamp(render_fastgs(cam, gaussians, pipe, background,
                                              0.5, 1.0)["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
            lp = perceptual(image.unsqueeze(0) * 2.0 - 1.0,
                            gt.unsqueeze(0) * 2.0 - 1.0).mean().item()
            lpips_vals.append(lp)
            psnr_vals.append(psnr(image, gt).mean().item())
            if FUSED_SSIM_AVAILABLE:
                ssim_vals.append(fast_ssim(image.unsqueeze(0),
                                           gt.unsqueeze(0)).item())
    train_lpips = float(np.mean(lpips_vals))
    keep = float(args.keep) if train_lpips > float(args.tau) else 1.0
    payload = dict(train_lpips=round(train_lpips, 4),
                   train_psnr=round(float(np.mean(psnr_vals)), 4),
                   train_ssim=round(float(np.mean(ssim_vals)), 4) if ssim_vals else None,
                   keep=keep, tau=float(args.tau), n_views=int(len(picks)),
                   train_views_only=True, init_ply=str(args.ply),
                   work=str(args.work),
                   view_indices=[int(i) for i in picks],
                   view_names=[train_cams[i].image_name for i in picks],
                   camera_order="unshuffled")
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        Path(args.record).write_text(json.dumps(payload, indent=2))
    print("[prune_gate] " + json.dumps(payload))
    return payload


if __name__ == "__main__":
    main()
