"""Render a PLY with gsplat's antialiased rasteriser instead of FastGS's.

Motivation: ReSplat's decoder runs gsplat in antialiased mode (eps2d=0.1) while the
pinned FastGS rasteriser hardcodes prefiltered=False and has no 3D filter. On
high-frequency scenes (foliage, grass) that mismatch shows up as aliasing, which the
VGG-LPIPS term punishes heavily. This script renders the same Gaussian set with
gsplat (eps2d configurable) so the two renderers can be compared on identical PLYs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "competition/upstream/baseline/fastgs"))


def load_ply_params(path, device):
    from plyfile import PlyData
    ply = PlyData.read(str(path))["vertex"]
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float32)
    opacity = np.asarray(ply["opacity"], dtype=np.float32)[:, None]
    scale = np.stack([ply[f"scale_{i}"] for i in range(3)], 1).astype(np.float32)
    rot = np.stack([ply[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    dc = np.stack([ply[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)
    rest = np.stack([np.asarray(ply[f"f_rest_{i}"], dtype=np.float32)
                     for i in range(45)], 1).reshape(-1, 15, 3)
    to = lambda a: torch.from_numpy(a).to(device)
    return dict(means=to(xyz), scales=to(scale), quats=to(rot), opacities=to(opacity),
                sh0=to(dc)[:, None, :], shN=to(rest))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="prepared scene dir")
    p.add_argument("--ply", required=True)
    p.add_argument("--out", required=True, help="output metrics json")
    p.add_argument("--eps2d", type=float, default=0.3)
    p.add_argument("--eval", action="store_true", default=True)
    args = p.parse_args(argv)

    from gsplat.rendering import rasterization
    from utils.general_utils import safe_state  # noqa: F401
    from scene import Scene, GaussianModel
    from argparse import ArgumentParser
    from arguments import ModelParams
    import lpips
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

    parser = ArgumentParser()
    lp = ModelParams(parser)
    lp_args = lp.extract(parser.parse_args(["-s", args.source, "--eval", "-r", "1"]))
    import tempfile
    lp_args.model_path = tempfile.mkdtemp(prefix="gsplat-aa-")
    gm = GaussianModel(3, "default")
    scene = Scene(lp_args, gm)
    cams = scene.getTestCameras()
    device = torch.device("cuda")
    params = load_ply_params(args.ply, device)
    loss_fn = lpips.LPIPS(net="vgg").to(device).eval()

    ps, ss, lps = [], [], []
    for cam in cams:
        w, h = cam.image_width, cam.image_height
        viewmat = cam.world_view_transform.transpose(0, 1)
        K = torch.tensor([[cam.FoVx, 0, w / 2], [0, cam.FoVy, h / 2], [0, 0, 1]],
                         device=device, dtype=torch.float32)
        # build a proper pinhole K from the camera FOV
        fx = 0.5 * w / np.tan(cam.FoVx * 0.5)
        fy = 0.5 * h / np.tan(cam.FoVy * 0.5)
        K = torch.tensor([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]],
                         device=device, dtype=torch.float32)
        with torch.no_grad():
            img, _, _ = rasterization(
                means=params["means"], quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True).clamp_min(1e-8), scales=params["scales"].exp(),
                opacities=params["opacities"].sigmoid().squeeze(-1),
                colors=torch.cat((params["sh0"], params["shN"]), dim=1),
                viewmats=viewmat[None], Ks=K[None], width=w, height=h, sh_degree=3,
                near_plane=0.01, far_plane=200.0, eps2d=args.eps2d,
                rasterize_mode="antialiased", packed=False)
            out = img[0].permute(2, 0, 1).clamp(0, 1)
            gt = cam.original_image.clamp(0, 1)
            mse = ((out - gt) ** 2).mean()
            ps.append(float(-10 * torch.log10(mse)))
            ss.append(float(ssim_fn(out[None], gt[None], data_range=1.0)))
            lps.append(float(loss_fn(out[None] * 2 - 1, gt[None] * 2 - 1).item()))
    res = dict(renderer="gsplat-antialiased", eps2d=args.eps2d, n=len(ps),
               psnr=float(np.mean(ps)), ssim=float(np.mean(ss)), lpips=float(np.mean(lps)))
    res["quality"] = (min(max(res["psnr"] - 15, 0), 25) / 25 * 20
                      + min(max(res["ssim"] - 0.70, 0), 0.25) / 0.25 * 20
                      + min(max(0.25 - res["lpips"], 0), 0.20) / 0.20 * 40)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
