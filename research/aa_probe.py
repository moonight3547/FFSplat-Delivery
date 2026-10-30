"""Find the camera convention that makes gsplat agree with the FastGS renderer.

Renders the same PLY with a small set of candidate view-matrix / intrinsic
conventions and reports PSNR against the held-out images. The variant with the
highest PSNR is the convention gsplat expects for this scene, which then makes the
antialiasing comparison meaningful.
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


def load_params(path, device):
    from plyfile import PlyData
    ply = PlyData.read(str(path))["vertex"]
    g = lambda n: torch.from_numpy(np.asarray(ply[n], dtype=np.float32)).to(device)
    xyz = torch.stack([g("x"), g("y"), g("z")], 1)
    scale = torch.stack([g(f"scale_{i}") for i in range(3)], 1)
    rot = torch.stack([g(f"rot_{i}") for i in range(4)], 1)
    dc = torch.stack([g(f"f_dc_{i}") for i in range(3)], 1)[:, None, :]
    rest = torch.stack([g(f"f_rest_{i}") for i in range(45)], 1).reshape(-1, 15, 3)
    return dict(means=xyz, scales=scale, quats=rot, opacities=g("opacity"),
                sh0=dc, shN=rest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--ply", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-views", type=int, default=8)
    args = p.parse_args()

    from argparse import ArgumentParser
    from arguments import ModelParams
    from scene import Scene, GaussianModel
    from gsplat.rendering import rasterization

    ap = ArgumentParser()
    lp = ModelParams(ap)
    lp_args = lp.extract(ap.parse_args(["-s", args.source, "--eval", "-r", "1"]))
    lp_args.model_path = tempfile.mkdtemp(prefix="aaprobe-")
    scene = Scene(lp_args, GaussianModel(3, "default"))
    cams = scene.getTestCameras()[: args.max_views]
    dev = torch.device("cuda")
    prm = load_params(args.ply, dev)
    extent = float(scene.cameras_extent)

    variants = {}
    for vname in ("w2c", "w2c_transposed", "w2c_yflip", "c2w"):
        rows = []
        for cam in cams:
            w, h = cam.image_width, cam.image_height
            fx = 0.5 * w / np.tan(cam.FoVx * 0.5)
            fy = 0.5 * h / np.tan(cam.FoVy * 0.5)
            K = torch.tensor([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]],
                             device=dev, dtype=torch.float32)
            base = cam.world_view_transform.clone().float()   # w2c^T
            if vname == "w2c":
                vm = base.transpose(0, 1)
            elif vname == "w2c_transposed":
                vm = base
            elif vname == "w2c_yflip":
                vm = base.transpose(0, 1).clone(); vm[1, :] *= -1
            else:
                vm = torch.linalg.inv(base.transpose(0, 1))
            with torch.no_grad():
                img, _, _ = rasterization(
                    means=prm["means"], quats=prm["quats"] / prm["quats"].norm(dim=-1, keepdim=True).clamp_min(1e-8),
                    scales=prm["scales"].exp(), opacities=prm["opacities"].sigmoid().squeeze(-1),
                    colors=torch.cat((prm["sh0"], prm["shN"]), dim=1),
                    viewmats=vm[None], Ks=K[None], width=w, height=h, sh_degree=3,
                    near_plane=0.01 * extent, far_plane=100.0 * extent,
                    eps2d=0.3, rasterize_mode="antialiased", packed=False)
                out = img[0].permute(2, 0, 1).clamp(0, 1)
                gt = cam.original_image.clamp(0, 1)
                mse = ((out - gt) ** 2).mean()
                rows.append(float(-10 * torch.log10(mse)))
        variants[vname] = float(np.mean(rows)) if rows else -99.0
    best = max(variants, key=variants.get)
    res = dict(psnr_by_variant=variants, best=best)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
