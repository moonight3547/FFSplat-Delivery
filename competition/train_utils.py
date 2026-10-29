"""Small migrated Gaussian-only helpers, independent of legacy experiment runners."""
from functools import lru_cache
import torch
import torch.nn.functional as F


def make_parameters(arrays, device):
    keys = ("means", "scales", "quats", "dc", "rest", "opacities")
    if len(arrays) != len(keys): raise ValueError("Expected six Gaussian parameter arrays")
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.as_tensor(a, device=device).contiguous().clone())
                                     for k, a in zip(keys, arrays)})
    params["opacities"] = torch.nn.Parameter(params["opacities"].detach().reshape(-1))
    return params


def make_optimizers(params, cfg):
    lrs = dict(cfg["adam_lrs"], means=cfg["position_lr_initial"])
    return {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15, betas=(.9, .999)) for k in params}


@lru_cache(maxsize=8)
def ssim_kernel(device, dtype):
    axis = torch.arange(11, device=device, dtype=dtype)-5
    weights = torch.exp(-axis.square()/(2*1.5**2)); weights /= weights.sum()
    return (weights[:, None]*weights[None, :]).expand(3, 1, 11, 11).contiguous()


def ssim_loss(a, b):
    a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    inputs = torch.cat((a, b, a*a, b*b, a*b), dim=0)
    ma, mb, aa, bb, ab = F.conv2d(inputs, ssim_kernel(a.device, a.dtype), padding=5, groups=3).chunk(5, dim=0)
    return 1-(((2*ma*mb+.01**2)*(2*(ab-ma*mb)+.03**2))/
             ((ma.square()+mb.square()+.01**2)*(aa-ma.square()+bb-mb.square()+.03**2))).mean()


def regularization_loss(params, weights):
    return (weights["opacity"]*params["opacities"].sigmoid().mean() +
            weights["scale"]*params["scales"].exp().mean())
