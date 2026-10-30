"""Score a snapshot curve in one process: render test views from each PLY and
compute PSNR / SSIM / VGG-LPIPS inline, so many time points cost one model load.

  python -m research.curve_metrics --source work/<case> --out curve.json \
      --ply t05.ply --name 5 --ply t10.ply --name 10 ...
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "competition/upstream/baseline/fastgs"))

from argparse import ArgumentParser  # noqa: E402

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--ply", action="append", default=[])
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if len(args.ply) != len(args.name):
        raise SystemExit("--ply and --name must be given in pairs")

    safe_state(args.quiet)
    dataset = lp.extract(args)
    dataset.model_path = dataset.model_path or tempfile.mkdtemp(prefix="curve-")
    pipe = pp.extract(args)

    import lpips
    loss_fn = lpips.LPIPS(net="vgg").cuda().eval()
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

    gaussians = GaussianModel(3, "default")
    scene = Scene(dataset, gaussians)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    test_cams = scene.getTestCameras()

    rows = []
    for ply, name in zip(args.ply, args.name):
        if not Path(ply).is_file():
            continue
        model = GaussianModel(3, "default")
        model.load_ply(ply)
        ps, ss, lps = [], [], []
        for cam in test_cams:
            with torch.no_grad():
                out = render_fastgs(cam, model, pipe, bg, 1.0, 1.0)["render"].clamp(0, 1)
                gt = cam.original_image.clamp(0, 1)
                mse = ((out - gt) ** 2).mean()
                ps.append(float(-10.0 * torch.log10(mse)))
                ss.append(float(ssim_fn(out[None], gt[None], data_range=1.0)))
                lps.append(float(loss_fn(out[None] * 2 - 1, gt[None] * 2 - 1).item()))
        q = ((min(max(np.mean(ps) - 15, 0), 25) / 25 * 20)
             + (min(max(np.mean(ss) - 0.70, 0), 0.25) / 0.25 * 20)
             + (min(max(0.25 - np.mean(lps), 0), 0.20) / 0.20 * 40))
        rows.append(dict(name=name, psnr=float(np.mean(ps)), ssim=float(np.mean(ss)),
                         lpips=float(np.mean(lps)), quality=float(q),
                         gaussians=int(model.get_xyz.shape[0])))
        print(json.dumps(rows[-1]))
        del model
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
