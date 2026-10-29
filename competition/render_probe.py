"""Render reconstruction cameras for backend acceptance, outside timed policy runs.

Run each backend in its Conda runtime, then compare the two .npz files on CPU.
No RGB file is read. A numerical report is evidence, not automatic acceptance.
"""
import argparse
import math
from pathlib import Path
import sys
import numpy as np
from .contracts import sha256, write_json


def compare(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        for key in ("names", "camera_identity"):
            if not np.array_equal(a[key], b[key]): raise ValueError("Probe camera identities differ")
        x, y = a["renders"], b["renders"]
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Invalid probe renders")
        rows = []
        for name, first, second in zip(a["names"], x, y):
            error = first.astype(np.float64)-second
            mse = float(np.mean(error**2))
            rows.append(dict(name=str(name), mse=mse, psnr=None if mse == 0 else -10*math.log10(mse),
                             identical=mse == 0, max_abs=float(np.abs(error).max())))
    return dict(left_sha256=sha256(left), right_sha256=sha256(right), rows=rows,
                status="Diagnostic only; inspect before promoting MCMC transfer. Not benchmark quality metrics.")


def render(args):
    import torch
    from .bridge import read_training_views, load_world, parameter_arrays
    from .pairing import verify_inputs
    frozen = verify_inputs(args.work)
    split, views = read_training_views(args.work)
    chosen = np.linspace(0, len(views)-1, min(args.views, len(views)), dtype=int)
    h, w = split["resolution"]
    results = []
    if args.backend == "gsplat":
        from gsplat.rendering import rasterization
        from .train_utils import make_parameters
        poses = np.stack([v["c2w"] for v in views])
        center = poses[:, :3, 3].mean(0)
        extent = float(np.max(np.abs(poses[:, :3, 3]-center)))
        if not math.isfinite(extent) or extent <= 0: raise ValueError("Degenerate cameras")
        scale = 1/extent
        params = make_parameters(parameter_arrays(load_world(args.initial), center, scale), "cuda")
        poses[:, :3, 3] = (poses[:, :3, 3]-center)*scale
        matrices = torch.linalg.inv(torch.from_numpy(poses).cuda())
        K = views[0]["K"].copy()
        K[0, 2], K[1, 2] = w/2, h/2  # Official FastGS camera uses FoV, centered principal point.
        K = torch.from_numpy(K).cuda()
        for i in chosen:
            with torch.no_grad():
                output, _, _ = rasterization(means=params["means"], quats=params["quats"],
                    scales=params["scales"].exp(), opacities=params["opacities"].sigmoid(),
                    colors=torch.cat((params["dc"][:, None], params["rest"]), dim=1),
                    viewmats=matrices[i:i+1], Ks=K[None], width=w, height=h, sh_degree=3,
                    near_plane=.01*scale, far_plane=200.*scale, eps2d=.3, rasterize_mode="classic", packed=False)
            results.append(output[0].clamp(0, 1).cpu().numpy())
    else:
        from types import SimpleNamespace
        sys.path.insert(0, str(args.upstream/"baseline/fastgs"))
        from scene import GaussianModel
        from scene.cameras import Camera
        from gaussian_renderer import render_fastgs
        gaussians = GaussianModel(3, "default")
        gaussians.load_ply(str(args.initial))
        pipe = SimpleNamespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
        background = torch.zeros(3, device="cuda")
        for i in chosen:
            view = views[i]; w2c = np.linalg.inv(view["c2w"])
            camera = Camera(colmap_id=int(i), uid=int(i), R=w2c[:3, :3].T, T=w2c[:3, 3],
                FoVx=2*math.atan(w/(2*view["K"][0, 0])), FoVy=2*math.atan(h/(2*view["K"][1, 1])),
                image=torch.zeros(3, h, w), gt_alpha_mask=None, image_name=view["name"], data_device="cuda")
            with torch.no_grad():
                output = render_fastgs(camera, gaussians, pipe, background, 1.0, 1.0)
            results.append(output["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # File object avoids numpy silently appending another suffix.
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, renders=np.stack(results), names=np.array([views[i]["name"] for i in chosen]),
            camera_identity=np.array(frozen["geometry_sha256"]), backend=np.array(args.backend),
            initial_sha256=np.array(sha256(args.initial)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["gsplat", "fastgs"])
    p.add_argument("--work", type=Path)
    p.add_argument("--initial", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--upstream", type=Path, default=Path(__file__).parent/"upstream")
    p.add_argument("--views", type=int, default=3)
    p.add_argument("--compare", nargs=2, type=Path, metavar=("GSPLAT_NPZ", "FASTGS_NPZ"))
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError("Preserve previous probe evidence")
    if args.compare:
        write_json(args.output, compare(*args.compare))
    else:
        if not args.backend or not args.work or not args.initial or args.views < 1:
            p.error("Rendering requires --backend, --work, --initial and positive --views")
        render(args)


if __name__ == "__main__":
    main()
