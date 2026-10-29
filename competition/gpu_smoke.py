"""Small synthetic CUDA forward/backward acceptance; no benchmark RGB is read."""
import argparse
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from .contracts import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['gsplat', 'fastgs', 'classic', 'metrics'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    import torch
    torch.manual_seed(20261029)
    root = Path(__file__).parent/'upstream'
    result = dict(backend=a.backend, torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), scope='synthetic kernel acceptance, not benchmark quality')
    if a.backend == 'metrics':
        sys.path.insert(0, str(root/'resplat'))
        from src.evaluation.metrics import compute_psnr, compute_ssim, compute_lpips
        x = torch.rand(1, 3, 64, 64, device='cuda')
        y = (x*.9+.03).clamp(0, 1)
        result['metrics'] = {k: float(fn(x, y).mean()) for k, fn in
            [('psnr', compute_psnr), ('ssim', compute_ssim), ('vgg_lpips', compute_lpips)]}
        if not all(math.isfinite(v) for v in result['metrics'].values()):
            raise ValueError('Nonfinite synthetic metric')
    else:
        n = 128
        means = torch.rand(n, 3, device='cuda')-.5
        means[:, 2] += 3
        params = {'means': means, 'scales': torch.full((n, 3), -.3, device='cuda'),
                  'quats': torch.tensor([1., 0, 0, 0], device='cuda').repeat(n, 1),
                  'opacity': torch.zeros(n, device='cuda'), 'sh': torch.rand(n, 16, 3, device='cuda')*.05}
        params = {k: torch.nn.Parameter(v) for k, v in params.items()}
        optimizer = torch.optim.Adam(params.values(), lr=.001)
        if a.backend == 'gsplat':
            import pointops
            idx, dist = pointops.knn_query(4, means.contiguous(), torch.tensor([n], device='cuda', dtype=torch.int32))
            assert idx.shape == (n, 4) and torch.isfinite(dist).all()
            from gsplat.rendering import rasterization
            K = torch.tensor([[50., 0, 32.], [0, 50., 32.], [0, 0, 1.]], device='cuda')
            def render():
                rgb, _, _ = rasterization(means=params['means'], quats=params['quats'],
                    scales=params['scales'].exp(), opacities=params['opacity'].sigmoid(),
                    colors=params['sh'], viewmats=torch.eye(4, device='cuda')[None], Ks=K[None],
                    width=64, height=64, sh_degree=3, packed=False, rasterize_mode='classic', eps2d=.3)
                return rgb.permute(0, 3, 1, 2)
            from .train_utils import ssim_loss
            ssim = lambda x, y: 1-ssim_loss(x[0].permute(1, 2, 0), y[0].permute(1, 2, 0))
        else:
            sys.path.insert(0, str(root/'baseline'/('fastgs' if a.backend == 'fastgs' else '3dgs')))
            from simple_knn._C import distCUDA2
            assert torch.isfinite(distCUDA2(means)).all()
            from utils.graphics_utils import getProjectionMatrix
            module = __import__('diff_gaussian_rasterization_fastgs' if a.backend == 'fastgs' else 'diff_gaussian_rasterization')
            fields = dict(image_height=64, image_width=64, tanfovx=.64, tanfovy=.64,
                bg=torch.zeros(3, device='cuda'), scale_modifier=1., viewmatrix=torch.eye(4, device='cuda'),
                projmatrix=getProjectionMatrix(.01, 100., 2*math.atan(.64), 2*math.atan(.64)).T.contiguous().cuda(),
                sh_degree=3, campos=torch.zeros(3, device='cuda'), prefiltered=False, debug=False)
            if a.backend == 'fastgs':
                fields.update(mult=.5, get_flag=None, metric_map=torch.zeros(4096, dtype=torch.int32, device='cuda'))
            else:
                fields['antialiasing'] = False
            raster = module.GaussianRasterizer(module.GaussianRasterizationSettings(**fields))
            def render():
                colors = {'dc': params['sh'][:, :1].contiguous(), 'shs': params['sh'][:, 1:].contiguous()} if a.backend == 'fastgs' else {'shs': params['sh']}
                return raster(means3D=params['means'], means2D=torch.zeros(n, 4 if a.backend == 'fastgs' else 3,
                    device='cuda', requires_grad=True), opacities=params['opacity'].sigmoid()[:, None],
                    scales=params['scales'].exp(), rotations=params['quats'], colors_precomp=None,
                    cov3D_precomp=None, **colors)[0][None]
            from fused_ssim import fused_ssim as ssim
        target = torch.rand(1, 3, 64, 64, device='cuda')*.3
        result['steps'] = []
        for _ in range(2):
            optimizer.zero_grad()
            rgb = render()
            loss = .8*(rgb-target).abs().mean()+.2*(1-ssim(rgb, target))
            loss.backward()
            grads = {k: float(v.grad.abs().max()) for k, v in params.items() if v.grad is not None}
            if len(grads) != len(params) or not all(math.isfinite(v) for v in grads.values()) or not any(v > 0 for v in grads.values()):
                raise ValueError('Missing/nonfinite/empty backward gradients')
            optimizer.step()
            torch.cuda.synchronize()
            result['steps'].append(dict(loss=float(loss.detach()), gradient_max=grads))
    result['status'] = 'passed'
    write_json(a.output, result)
    print(result)


if __name__ == '__main__':
    main()
