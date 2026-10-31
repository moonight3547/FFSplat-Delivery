"""Frozen train-only LPIPS with explicit backbone and reproducible identity."""
import hashlib
from importlib.metadata import version
from pathlib import Path
import random
import time

import numpy as np
import torch


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_perceptual(net, constructor=None):
    import lpips
    from torchvision.models import AlexNet_Weights, VGG16_Weights
    if net not in ('alex', 'vgg'):
        raise ValueError(net)
    started = time.perf_counter()
    py_rng, np_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            model = (constructor or lpips.LPIPS)(net=net, version='0.1', verbose=False).cuda().eval()
    finally:
        random.setstate(py_rng)
        np.random.set_state(np_rng)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    weights = AlexNet_Weights.IMAGENET1K_V1 if net == 'alex' else VGG16_Weights.IMAGENET1K_V1
    backbone = Path(torch.hub.get_dir()) / 'checkpoints' / weights.url.rsplit('/', 1)[1]
    linear = Path(lpips.__file__).parent / 'weights' / 'v0.1' / (net + '.pth')
    identity = dict(net=net, lpips_package_version=version('lpips'), lpips_version='0.1',
                    backbone_weights=str(backbone), backbone_sha256=file_sha(backbone),
                    linear_weights=str(linear), linear_sha256=file_sha(linear),
                    eval_mode=not model.training,
                    parameters_frozen=all(not p.requires_grad for p in model.parameters()),
                    normalization='RGB [0,1] mapped to [-1,1]; LPIPS normalize=False',
                    rng_restored=True)
    torch.cuda.synchronize()
    identity['load_and_identity_seconds'] = time.perf_counter() - started
    return model, identity


def optimizer_clocks(model):
    result = {}
    for label in ('optimizer', 'shoptimizer'):
        optimizer = getattr(model, label)
        for group in optimizer.param_groups:
            parameter, = group['params']
            clock = optimizer.state[parameter].get('step')
            result[label + '/' + group['name']] = None if clock is None else float(clock)
    return result
