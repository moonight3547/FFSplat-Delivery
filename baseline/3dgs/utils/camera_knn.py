import torch
import torch.nn.functional as F


def build_camera_knn(cameras, xyz, k=3):
    """3DGS2 Sec. 4.1: bounding sphere of GS kernels, project cameras onto it, spherical KNN."""
    mapping = {id(c): [] for c in cameras}
    if len(cameras) == 0 or xyz.numel() == 0:
        return mapping
    center = xyz.detach().mean(dim=0)
    radius = (xyz.detach() - center).norm(dim=-1).max().clamp(min=1e-6)
    cam_centers = torch.stack([c.camera_center.reshape(-1)[:3].to(xyz.device) for c in cameras], dim=0)
    dirs = F.normalize(cam_centers - center.unsqueeze(0), dim=-1, eps=1e-8)
    sim = dirs @ dirs.t()
    sim.fill_diagonal_(-2.0)
    k = min(int(k), max(len(cameras) - 1, 0))
    if k == 0:
        return mapping
    _, idx = torch.topk(sim, k=k, dim=1)
    for i, cam in enumerate(cameras):
        mapping[id(cam)] = [cameras[int(j)] for j in idx[i].tolist()]
    return mapping


class DownsampledView:
    """Secondary-target view with downsampled image (3DGS2 Sec. 4.3)."""

    def __init__(self, cam, factor=4):
        factor = max(int(factor), 1)
        self.FoVx = cam.FoVx
        self.FoVy = cam.FoVy
        self.world_view_transform = cam.world_view_transform
        self.full_proj_transform = cam.full_proj_transform
        self.camera_center = cam.camera_center
        self.image_name = getattr(cam, "image_name", "")
        self.alpha_mask = None
        img = cam.original_image
        if factor == 1:
            self.original_image = img
        else:
            self.original_image = F.interpolate(
                img.unsqueeze(0),
                scale_factor=1.0 / factor,
                mode="area",
                recompute_scale_factor=True,
            ).squeeze(0)
        self.image_height = int(self.original_image.shape[1])
        self.image_width = int(self.original_image.shape[2])
