"""PGSR-style depth-geometry regularizer for the FastGS trainer (late-window test).

The vendored FastGS rasterizer returns no depth/alpha, so the depth regularizer is
computed with a *second* gsplat 1.5.3 render on the same Gaussian tensors.  Only
ONE extra pass is needed: the pass is rendered with ``colors = smallest-scale
axis`` (the PGSR planar normal) in RGB+ED mode, so the RGB channels carry the
alpha-composited normal map and the ED channel carries expected depth.  The loss
is single-view normal-depth consistency:

    n_depth = normalize(cross(dD/dx, dD/dy))       (Sobel on the rendered depth)
    n_render = normalize(2 * rendered_rgb - 1)      (de-normalised axis map)
    L = mean_{valid pixels} (1 - |cos(n_depth, n_render)|)

with valid = alpha > 0.5, depth > 0, and finite depth gradients.

Cost (measured on N1, bicycle delivery PLY, 576x960, ~1M Gaussians): one gsplat
RGB+ED pass = 7.65 ms forward+backward, i.e. +7.65 ms on the depth steps.  At
every 10th step over iterations 4000-5000 (100 applications) that is +0.77 s of
module-B time (Delta-TIME -0.18 on a 95 s budget), matching the task-15 brief.

``depth_normal_consistency`` is called from ``research.train_sweep``; it returns
``(loss, stats)`` and does NOT touch the FastGS render graph.
"""
import math

import torch

_EDGES = {}


def _sobel_kernels(device, dtype):
    key = (str(device), str(dtype))
    if key not in _EDGES:
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                          device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
        ky = kx.transpose(2, 3).contiguous()
        _EDGES[key] = (kx, ky)
    return _EDGES[key]


def smallest_axis(gaussians):
    """World-space smallest-scale axis of every Gaussian (the planar normal)."""
    rot = gaussians.get_rotation
    w, x, y, z = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    R = torch.empty((rot.shape[0], 3, 3), device=rot.device, dtype=rot.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    idx = gaussians.get_scaling.argmin(dim=1)
    axis = R[torch.arange(R.shape[0], device=R.device), :, idx]
    return axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-8)


def _intrinsics_from_fov(cam):
    width, height = int(cam.image_width), int(cam.image_height)
    fx = width / (2.0 * math.tan(0.5 * float(cam.FoVx)))
    fy = height / (2.0 * math.tan(0.5 * float(cam.FoVy)))
    k = torch.tensor([[fx, 0.0, width / 2.0],
                      [0.0, fy, height / 2.0],
                      [0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    return k, width, height


def _depth_normals(depth, alpha):
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        depth = depth.unsqueeze(0)
    kx, ky = _sobel_kernels(depth.device, depth.dtype)
    dx = torch.nn.functional.conv2d(depth, kx, padding=1)
    dy = torch.nn.functional.conv2d(depth, ky, padding=1)
    n = torch.cat([-dx, -dy, torch.ones_like(dx)], dim=1)
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-8)
    valid = ((alpha > 0.5) & (depth > 0.0)
             & torch.isfinite(dx) & torch.isfinite(dy)
             & (dx.abs() < 1e3) & (dy.abs() < 1e3))
    if valid.dim() == 2:
        valid = valid.unsqueeze(1)
    return n, valid


def depth_normal_consistency(gaussians, cam, cameras_extent, near_far=None,
                             device="cuda"):
    """One gsplat RGB+ED pass -> single-view normal-depth consistency loss.

    Returns ``(loss, stats)``; returns ``(None, {"ok": False, ...})`` when the
    pass cannot be taken (non-CUDA, gsplat unavailable) so callers can skip it.
    """
    try:
        from gsplat.rendering import rasterization
    except Exception as error:  # pragma: no cover
        return None, dict(ok=False, reason="gsplat unavailable: %r" % (error,))
    axis = smallest_axis(gaussians)
    colors = axis * 0.5 + 0.5            # de-normalise back after compositing
    viewmats = cam.world_view_transform.transpose(0, 1).unsqueeze(0)
    ks, width, height = _intrinsics_from_fov(cam)
    near, far = (0.01 * cameras_extent, 200.0 * cameras_extent) if near_far is None else near_far
    rendered, alpha, _info = rasterization(
        means=gaussians.get_xyz, quats=gaussians.get_rotation,
        scales=gaussians.get_scaling, opacities=gaussians.get_opacity,
        colors=colors, viewmats=viewmats, Ks=ks.unsqueeze(0),
        width=width, height=height, sh_degree=None,
        near_plane=near, far_plane=far, eps2d=0.3,
        rasterize_mode="classic", packed=False, render_mode="RGB+ED")
    # rendered: (1, H, W, 4): RGB = composited normal colours, ED = expected depth
    normal_rgb = rendered[..., :3]
    depth = rendered[..., 3]
    n_render = normal_rgb * 2.0 - 1.0
    n_render = n_render / n_render.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    n_render = n_render.permute(0, 3, 1, 2)
    n_depth, valid = _depth_normals(depth, alpha)
    cos = (n_render * n_depth).sum(dim=1, keepdim=True).abs()
    valid_f = valid.float()
    loss = ((1.0 - cos) * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    with torch.no_grad():
        stats = dict(ok=True, valid_fraction=float(valid_f.mean()),
                     depth_mean=float(depth[alpha.squeeze(-1) > 0.5].mean())
                     if bool((alpha > 0.5).any()) else 0.0,
                     normal_cos_mean=float(cos[valid].mean()) if bool(valid.any()) else 0.0)
    return loss, stats
