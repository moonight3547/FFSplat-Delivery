"""TRAIN-only real-GPU acceptance for exact optional scene residency."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--init-ply', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('A fresh gate output is required')
    args.out.mkdir(parents=True)
    work = args.out / 'prepared'
    work.mkdir()
    (work / 'images').symlink_to((args.work / 'images').resolve(), target_is_directory=True)
    shutil.copytree(args.work / 'sparse', work / 'sparse')
    shutil.copy2(args.work / 'split.json', work / 'split.json')
    import torch
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from research import image_cache, scene_cache
    from research.cache_integrity import sha256
    image_cache.build(work, work / 'image_cache')
    scene_cache.build(work, work / 'scene_tensor_cache', seed=args.seed,
                      model_path=str(args.out / 'cache-builder'))
    exact = scene_cache._verify(work, work / 'scene_tensor_cache')
    if not exact['identical']:
        raise RuntimeError(f'Cache differs from native: {exact}')
    (args.out / 'exact-input-proof.json').write_text(json.dumps(exact, indent=2))
    timings = scene_cache._bench(work, work / 'scene_tensor_cache', reps=3)
    torch.cuda.empty_cache()
    records, walls = {}, {}
    for label in ('native_scene', 'cached_scene'):
        model = args.out / label
        command = [sys.executable, '-m', 'research.train_confirmed', '--source', str(work),
            '--model_path', str(model), '--init_ply', str(args.init_ply), '--record', str(model / 'train_record.json'),
            '--eval', '-r', '1', '--sh_degree', '3', '--quiet', '--seed', str(args.seed),
            '--iterations', '600', '--position_lr_max_steps', '600', '--opacity_reset_interval', '0',
            '--densify_from_iter', '500', '--densify_until_iter', '3500', '--densification_interval', '100',
            '--grad_abs_thresh', '.0008', '--lambda_dssim', '.6', '--scaling_lr', '.01', '--realloc_frac', '0',
            '--lpips_weight', '.1', '--lpips_start', '401', '--lpips_end', '600', '--lpips_scale', '1']
        if label == 'native_scene':
            command += ['--image_cache', str(work / 'image_cache')]
        else:
            command += ['--scene_cache', str(work / 'scene_tensor_cache')]
        (args.out / (label + '-command.json')).write_text(json.dumps(command, indent=2))
        began = time.perf_counter()
        with (args.out / (label + '.log')).open('w') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        walls[label] = time.perf_counter() - began
        records[label] = json.loads((model / 'train_record.json').read_text())
        audit = records[label]['runtime_audit']
        if (records[label]['completed_iterations'] != 600 or audit['optimizer_calls'] != 600
                or audit['vgg_steps'] != list(range(401, 601))
                or records[label]['eng']['scene_cache_fallback'] is not None):
            raise RuntimeError('600-step cache gate did not complete its exact recipe')
    first, second = [records[label]['runtime_audit'] for label in records]
    if first['sampling_audit'] != second['sampling_audit']:
        raise RuntimeError('Main camera or density RNG sequence changed with caching')
    if first['optimizer_clocks'] != second['optimizer_clocks']:
        raise RuntimeError('Optimizer clocks changed with caching')
    from plyfile import PlyData
    import numpy as np
    for record in records.values():
        ply = PlyData.read(record['final_ply'])['vertex'].data
        if not all(np.isfinite(ply[name]).all() for name in ply.dtype.names):
            raise RuntimeError('Nonfinite cache gate PLY')
    proof = dict(passed=True, exact_input=exact, isolated_load_benchmark=timings,
                 process_walls=walls, main_and_density_sampling_exact=True, clocks_exact=True,
                 source={name: sha256(Path(__file__).parent / name) for name in
                         ('train_confirmed.py', 'train_sweep.py', 'scene_cache.py', 'cache_rng.py', 'cache_integrity.py')},
                 scope='600 TRAIN steps; shortened LR and VGG401..600; no TEST selection or quality score')
    (args.out / 'acceptance.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
