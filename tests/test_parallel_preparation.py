"""Compare the actual upstream writer with the parallel writer on real pixels."""
import ast
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pytest

from research import stage1_prepare_parallel as parallel

ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)


@pytest.mark.parametrize('workers', [4, 8])
@pytest.mark.parametrize('width,height', [(160, 96), (96, 160)])
def test_actual_parallel_rgb_metadata_split_equal_native(tmp_path, monkeypatch, workers, width, height):
    namespace = dict(np=np, Image=Image, os=os, time=time, json=json)
    load_functions(ROOT / 'competition/upstream/resplat/scripts/infer_colmap.py',
                   {'compute_target_shape'}, namespace)
    load_functions(ROOT / 'competition/upstream/baseline/module_a/preprocess.py',
                   {'write_work_dir', 'rotmat2qvec'}, namespace)
    entries, poses = [], {}
    rng = np.random.default_rng(42)
    for index in range(24):
        name = f'frame_{index:04}.jpg'
        path = tmp_path / name
        Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(path)
        entries.append((name, str(path)))
        poses[name] = (np.eye(3), np.array([index / 24, 0, 0]))
    namespace['natural_image_list'] = lambda *a: entries
    for name in ('natural_image_list', 'compute_target_shape', 'rotmat2qvec'):
        monkeypatch.setattr(parallel, name, namespace[name], raising=False)
    monkeypatch.setattr(parallel, 'WORKERS', workers)
    intrinsics = dict(W=width, H=height, fx=100, fy=101, cx=width/2, cy=height/2, poses=poses)
    native, candidate = tmp_path / 'native', tmp_path / 'parallel'
    namespace['write_work_dir'](str(native), str(tmp_path), intrinsics, 16, 8, 960)
    parallel.write_work_dir(str(candidate), str(tmp_path), intrinsics, 16, 8, 960)
    for relative in ('split.json', 'sparse/0/cameras.txt', 'sparse/0/images.txt'):
        assert (native / relative).read_bytes() == (candidate / relative).read_bytes()
    for path in (native / 'images').iterdir():
        with Image.open(path) as expected, Image.open(candidate / 'images' / path.name) as actual:
            assert expected.mode == actual.mode == 'RGB'
            assert expected.tobytes() == actual.tobytes()
