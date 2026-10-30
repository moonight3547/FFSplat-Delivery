#!/usr/bin/env python3
"""Render a 3DGS PLY at an explicit list of prepared-scene cameras (held-out TRAIN views).

Used for the held-out triage metric of the DL3DV view-selection study: render the ReSplat init at
a fixed set of 8 train views that are NOT in any candidate context set, then score PSNR / SSIM /
VGG-LPIPS with the official competition evaluate.py (fastgs renderer + resplat metrics).

Run in the organizer-fastgs (fastgs) conda env, from the repository root:
  python research/holdout_render.py --work <prepared work> --ply <init.ply> \
      --views frame_00003.png,frame_00019.png,... --out <eval dir> --sh_degree 3

Writes <eval dir>/test_renders/*.png + <eval dir>/test_gt/*.png (layout expected by evaluate.py).
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

from arguments import PipelineParams  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from scene import GaussianModel  # noqa: E402
from scene.cameras import Camera  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def q2R(q):
    w, x, y, z = q
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def parse_cameras(path):
    cams = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 8 or f[1] != "PINHOLE":
            continue
        cams[int(f[0])] = dict(w=int(f[2]), h=int(f[3]), fx=float(f[4]), fy=float(f[5]))
    return cams


def parse_poses(path):
    out = {}
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.startswith("#"):
            i += 1
            continue
        f = line.split()
        if len(f) >= 10:
            try:
                qw, qx, qy, qz = map(float, f[1:5])
                tx, ty, tz = map(float, f[5:8])
                out[f[9]] = (q2R((qw, qx, qy, qz)), np.array([tx, ty, tz]))
            except ValueError:
                pass
        i += 2
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    pp = PipelineParams(parser)
    parser.add_argument("--work", required=True, help="prepared scene work dir (images + sparse/0)")
    parser.add_argument("--ply", required=True)
    parser.add_argument("--views", required=True, help="comma-separated image file names")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mult", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    safe_state(args.quiet)
    pipe = pp.extract(args)

    work = Path(args.work)
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    cams = parse_cameras(work / "sparse/0/cameras.txt")
    poses = parse_poses(work / "sparse/0/images.txt")
    if len(cams) != 1:
        raise SystemExit("expected a single PINHOLE camera")
    cam = cams[sorted(cams)[0]]
    fovx = 2.0 * np.arctan(cam["w"] / (2.0 * cam["fx"]))
    fovy = 2.0 * np.arctan(cam["h"] / (2.0 * cam["fy"]))

    model = GaussianModel(3, "default")
    model.load_ply(args.ply)
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    out = Path(args.out)
    (out / "test_renders").mkdir(parents=True, exist_ok=True)
    (out / "test_gt").mkdir(parents=True, exist_ok=True)
    for uid, name in enumerate(views):
        if name not in poses:
            raise SystemExit("no pose for view %s" % name)
        R_w2c, t_w2c = poses[name]
        R_c2w = R_w2c.T          # Camera() expects the camera-to-world ROTATION
        T = t_w2c                # ... but the world-to-camera TRANSLATION (3DGS convention)
        img = Image.open(work / "images" / name).convert("RGB")
        w, h = img.size
        if (w, h) != (cam["w"], cam["h"]):
            raise SystemExit("image %s size %s != camera %s" % (name, (w, h), (cam["w"], cam["h"])))
        img_t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        camera = Camera(colmap_id=uid, R=R_c2w, T=T, FoVx=fovx, FoVy=fovy,
                        image=img_t, gt_alpha_mask=None, image_name=name, uid=uid)
        with torch.no_grad():
            rendered = render_fastgs(camera, model, pipe, bg, args.mult, 1.0)["render"]
        r = torch.clamp(rendered, 0.0, 1.0).detach().permute(1, 2, 0).cpu().numpy()
        g = torch.clamp(camera.original_image, 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        Image.fromarray((r * 255).astype("uint8")).save(out / "test_renders" / name)
        Image.fromarray((g * 255).astype("uint8")).save(out / "test_gt" / name)
    (out / "rendered_views.json").write_text(json.dumps(
        dict(ply=str(args.ply), work=str(args.work), views=views, mult=args.mult,
             num_gaussians=int(model.get_xyz.shape[0])), indent=2))
    print(json.dumps(dict(ply=str(args.ply), views=len(views), gaussians=int(model.get_xyz.shape[0])), indent=1))


if __name__ == "__main__":
    main()
