"""Render test views with a trained Module-B model (fastgs env).

Backend:
  --backend fastgs      FastGS native renderer at mult=1.0 (full tile extent;
                        matches benchmarks/gs_compare_20261020 native parity)
  --backend classic3dgs vendored classic 3DGS renderer (legacy)

Usage (fastgs conda env, from baseline/):
  python render_test.py -s work/<group>/<scene> -m results/<group>/<scene>/module_b \
      --backend fastgs [--iteration 10000]
"""

import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

BASELINE_ROOT = Path(__file__).resolve().parent

# resolve backend code path BEFORE importing backend modules
_BACKEND = "fastgs"
if "--backend" in sys.argv:
    _BACKEND = sys.argv[sys.argv.index("--backend") + 1]
if _BACKEND == "fastgs":
    sys.path.insert(0, str(BASELINE_ROOT / "fastgs"))
else:
    sys.path.insert(0, str(BASELINE_ROOT / "3dgs"))

from argparse import ArgumentParser  # noqa: E402

from arguments import ModelParams, PipelineParams  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1,
                        help="model iteration to load (-1 = latest)")
    parser.add_argument("--backend", type=str, default="fastgs",
                        choices=["fastgs", "classic3dgs"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    safe_state(args.quiet)

    gaussians = GaussianModel(3, "default")
    scene = Scene(lp.extract(args), gaussians, load_iteration=args.iteration)
    bg = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    if args.backend == "fastgs":
        from gaussian_renderer import render_fastgs

        def render_fn(cam):
            return render_fastgs(cam, gaussians, pp.extract(args), bg, 1.0, 1.0)
    else:
        from gaussian_renderer import render

        def render_fn(cam):
            return render(cam, gaussians, pp.extract(args), bg)

    out_r = os.path.join(args.model_path, "test_renders")
    out_g = os.path.join(args.model_path, "test_gt")
    os.makedirs(out_r, exist_ok=True)
    os.makedirs(out_g, exist_ok=True)

    names = []
    for cam in scene.getTestCameras():
        with torch.no_grad():
            out = render_fn(cam)
        img = torch.clamp(out["render"], 0.0, 1.0).detach()
        gt = torch.clamp(cam.original_image, 0.0, 1.0)
        name = cam.image_name + ".png"
        Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save(
            os.path.join(out_r, name))
        Image.fromarray((gt.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save(
            os.path.join(out_g, name))
        names.append(name)

    with open(os.path.join(args.model_path, "rendered_views.json"), "w") as f:
        json.dump({"iteration": scene.loaded_iter, "test_views": names,
                   "backend": args.backend}, f, indent=2)
    print(f"[render:{args.backend}] {len(names)} test views -> {out_r}")


if __name__ == "__main__":
    main()
