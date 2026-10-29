"""Render test views from a 3DGS PLY with the pinned FastGS renderer.

Used by the research sweep so one process can score a whole snapshot curve
without reloading cameras per checkpoint. Writes the exact layout the official
``baseline/evaluate.py`` expects (``test_renders/`` + ``test_gt/``).

Run in the fastgs conda env, from the repository root:

  python -m research.render_eval --source work/<case> --out /tmp/eval \
      --ply a.ply --name iter00100 --ply b.ply --name iter01000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "competition/upstream/baseline"
sys.path.insert(0, str(BASELINE / "fastgs"))

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
    parser.add_argument("--mult", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if len(args.ply) != len(args.name):
        raise SystemExit("--ply and --name must be given in pairs")

    safe_state(args.quiet)
    dataset = lp.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(3, "default")
    scene = Scene(dataset, gaussians)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    test_cams = scene.getTestCameras()

    summary = []
    for ply, name in zip(args.ply, args.name):
        model = GaussianModel(3, "default")
        model.load_ply(ply)
        out_r = Path(args.out) / name / "test_renders"
        out_g = Path(args.out) / name / "test_gt"
        out_r.mkdir(parents=True, exist_ok=True)
        out_g.mkdir(parents=True, exist_ok=True)
        names = []
        for cam in test_cams:
            with torch.no_grad():
                out = render_fastgs(cam, model, pipe, background, args.mult, 1.0)
            img = torch.clamp(out["render"], 0.0, 1.0).detach()
            gt = torch.clamp(cam.original_image, 0.0, 1.0)
            fname = cam.image_name + ".png"
            Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save(
                out_r / fname)
            Image.fromarray((gt.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save(
                out_g / fname)
            names.append(fname)
        (Path(args.out) / name / "rendered_views.json").write_text(
            json.dumps(dict(ply=str(ply), test_views=names, mult=args.mult), indent=2))
        summary.append(dict(name=name, views=len(names), num_gaussians=int(model.get_xyz.shape[0])))
        del model
        torch.cuda.empty_cache()
    print(json.dumps(dict(source=str(dataset.source_path), rendered=summary), indent=2))


if __name__ == "__main__":
    main()
