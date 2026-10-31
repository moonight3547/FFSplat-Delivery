import ast
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from research import cache_rng
from research.cache_integrity import scene_identity, sha256
from research.image_cache import build, valid, CachedImages

ROOT = Path(__file__).resolve().parents[1]


def prepared(root):
    (root / 'images').mkdir(parents=True)
    (root / 'sparse/0').mkdir(parents=True)
    for i in range(2):
        data = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, :, None], (4, 256, 3)).copy()
        Image.fromarray(data).save(root / 'images' / f'{i}.png')
    (root / 'split.json').write_text(json.dumps(dict(train=['0.png', '1.png'], test=[])))
    (root / 'sparse/0/cameras.txt').write_text('1 PINHOLE 256 4 100 100 128 2\n')
    (root / 'sparse/0/images.txt').write_text('fixed poses\n')
    build(root, root / 'image_cache')
    return root


def test_image_cache_reuses_exact_content_and_rejects_corruption(tmp_path):
    work = prepared(tmp_path)
    cache = work / 'image_cache'
    payload = cache / 'images_u8.npy'
    before = payload.stat().st_mtime_ns
    build(work, cache)
    assert valid(work, cache) and payload.stat().st_mtime_ns == before
    assert json.loads((cache / '_status.json').read_text())['status'] == 'reused'
    data = np.load(payload, mmap_mode='r+')
    data[0, 0, 0, 0] ^= 1
    data.flush()
    assert not valid(work, cache)
    with pytest.raises(ValueError):
        CachedImages(cache)


def test_scene_identity_binds_seed_pose_and_source_pixels(tmp_path):
    work = prepared(tmp_path)
    before = scene_identity(work, 2, 'v3')
    assert before != scene_identity(work, 3, 'v3')
    (work / 'sparse/0/cameras.txt').write_text('changed intrinsics\n')
    changed = scene_identity(work, 2, 'v3')
    assert before != changed
    Image.new('RGB', (256, 4), (1, 2, 3)).save(work / 'images/0.png')
    assert changed != scene_identity(work, 2, 'v3')


def test_rng_json_roundtrip_preserves_python_numpy_and_torch():
    state = cache_rng.capture(cuda=False)
    expected = (random.random(), np.random.random(), torch.rand(3))
    cache_rng.restore(json.loads(json.dumps(state)))
    assert random.random() == expected[0] and np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


def actual_loader():
    """Exercise real cache loading without importing the CUDA rasterizer."""
    namespace = dict(np=np, torch=torch, json=json, Path=Path, time=time, sha256=sha256,
        CACHE_VERSION='scene-cache-v3-exact',
        cache_rng=SimpleNamespace(capture=lambda: cache_rng.capture(cuda=False),
                                  identity=cache_rng.identity, restore=cache_rng.restore),
        version_of=lambda work, seed: scene_identity(work, seed, 'scene-cache-v3-exact'))
    tree = ast.parse((ROOT / 'research/scene_cache.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in ('CachedCamera', 'CachedScene', 'load')]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual_scene_cache', 'exec'), namespace)
    return namespace['load']


def make_scene_cache(work):
    out = work / 'scene_cache'
    out.mkdir()
    images = np.asarray(np.load(work / 'image_cache/images_u8.npy')).transpose(0, 3, 1, 2)
    np.save(out / 'images_u8.npy', np.ascontiguousarray(images))
    version = scene_identity(work, 2, 'scene-cache-v3-exact')
    np.savez(out / 'cams.npz', wvt=np.repeat(np.eye(4, dtype='f4')[None], 2, axis=0),
             fpt=np.repeat(np.eye(4, dtype='f4')[None], 2, axis=0), centers=np.zeros((2, 3), 'f4'),
             names=np.array(['0', '1']), fovx=np.array([1.123456789123]*2),
             fovy=np.array([.987654321987]*2), width=256, height=4,
             cameras_extent=np.float64(1.234567891234), version=version)
    before = cache_rng.capture(cuda=False)
    random.random(); np.random.random(); torch.rand(4)
    after = cache_rng.capture(cuda=False)
    cache_rng.restore(before)
    meta = dict(cache_version='scene-cache-v3-exact', seed=2, torch_version=str(torch.__version__),
                images_sha256=sha256(out / 'images_u8.npy'), cameras_sha256=sha256(out / 'cams.npz'),
                rng_before=before, rng_after=after, native_test_count=0)
    (out / 'meta.json').write_text(json.dumps(meta))
    return out, before, after


def test_real_loader_preserves_all_256_pixels_double_scalars_and_rng(tmp_path):
    work = prepared(tmp_path)
    out, before, after = make_scene_cache(work)
    loaded = actual_loader()(work, out, device='cpu')
    cam = loaded.getTrainCameras()[0]
    expected = torch.from_numpy(np.array(Image.open(work / 'images/0.png'))) / 255.0
    assert torch.equal(cam.original_image, expected.permute(2, 0, 1))
    assert cam.original_image.stride() == expected.permute(2, 0, 1).stride()
    assert cam.FoVx == 1.123456789123 and cam.FoVy == .987654321987
    assert loaded.cameras_extent == 1.234567891234
    assert cache_rng.identity(cache_rng.capture(cuda=False)) == cache_rng.identity(after)
    cache_rng.restore(before)
    random.random()
    with pytest.raises(ValueError, match='RNG'):
        actual_loader()(work, out, device='cpu')


def test_real_loader_rejects_changed_pose_or_payload(tmp_path):
    work = prepared(tmp_path)
    out, _, _ = make_scene_cache(work)
    (work / 'sparse/0/images.txt').write_text('different poses\n')
    with pytest.raises(ValueError, match='stale'):
        actual_loader()(work, out, device='cpu')
