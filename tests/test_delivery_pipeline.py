"""Submission command, accounting, completion and PLY compatibility contracts."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

import run_pipeline as pipeline
from research.vectorized_ply import save_ply_vectorized
from research.delivery_entry import module_b_args
from research.reference_native import main as reference_native


def plan(tmp_path, *args):
    return pipeline.main(['--scene', str(tmp_path / 'absent'), '--output', str(tmp_path / 'out'),
                          '--describe', *args])


def test_confirmed_default_and_explicit_d4000_r15(tmp_path):
    default = plan(tmp_path)
    argv = default['module_b_argv']
    assert default['b_recipe'] == 'bigd3500' and default['vgg_tail'] == 200
    assert default['realloc_frac'] == 0 and 'research.train_confirmed' in argv
    assert default['a_workers'] == 4 and 'research.stage1_prepare_parallel' in default['module_a_argv']
    for flag, value in [('--densify_until_iter', '3500'), ('--grad_abs_thresh', '0.0008'),
                        ('--lpips_start', '4801'), ('--lpips_end', '5000'),
                        ('--seed', '20261015')]:
        assert argv[argv.index(flag) + 1] == value
    assert '--limit_seconds' not in argv and '--no_quick_test' in argv
    override = plan(tmp_path, '--b-recipe', 'bigd4000', '--realloc-frac', '0.15')
    assert override['b_recipe'] == 'bigd4000' and override['realloc_frac'] == .15
    assert not (tmp_path / 'out').exists()


def test_legacy_snapshot_stays_explicit(tmp_path):
    legacy = plan(tmp_path, '--preset', 'legacy')
    argv = legacy['module_b_argv']
    assert legacy['b_recipe'] == 'delivery' and legacy['vgg_tail'] == 0
    assert legacy['realloc_frac'] == .15 and legacy['timing_scope'] == 'internal'
    assert 'research.train_sweep' in argv and '--seed' not in argv
    assert argv[argv.index('--densify_until_iter') + 1] == '1250'
    assert '--limit_seconds' in argv
    assert legacy['a_workers'] == 1 and 'research.stage1_prepare_parallel' not in legacy['module_a_argv']


def test_official_shaped_direct_b_defaults_match_pipeline(tmp_path):
    direct = module_b_args(['-s', 'prepared', '-m', 'model', '--init_ply', 'init.ply'])
    pipeline_args = plan(tmp_path)['module_b_argv']
    for flag in ('--iterations', '--position_lr_max_steps', '--densify_from_iter',
                 '--densify_until_iter', '--grad_abs_thresh', '--lambda_dssim',
                 '--scaling_lr', '--realloc_frac', '--lpips_start', '--lpips_end', '--seed'):
        assert direct[direct.index(flag) + 1] == pipeline_args[pipeline_args.index(flag) + 1]
    assert direct[direct.index('--record') + 1] == 'model/train_record.json'


def test_top_level_vendored_libraries_remain_byte_identical():
    for name in ('baseline/fastgs', 'baseline/3dgs', 'resplat'):
        source, mirror = pipeline.REPO / 'competition/upstream' / name, pipeline.REPO / name
        for path in source.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                assert (mirror / path.relative_to(source)).read_bytes() == path.read_bytes()


def test_official_shaped_group_entry_can_preview_default_three_cases():
    result = subprocess.run([sys.executable, str(pipeline.REPO / 'baseline/run_suite.py'), '--describe'],
                            cwd=pipeline.REPO, capture_output=True, text=True, check=True)
    assert result.stdout.count('"b_recipe": "bigd3500"') == 3


@pytest.mark.parametrize('args', [('--b-iterations', '100'), ('--budget-seconds', 'nan'),
                                 ('--budget-seconds', '0'), ('--realloc-frac', 'nan'),
                                 ('--b-fraction', '-1'), ('--num-refine', '3')])
def test_invalid_confirmed_settings_fail_before_running(tmp_path, args):
    with pytest.raises(SystemExit):
        plan(tmp_path, *args)


def test_full_process_b_and_reference_accounting(monkeypatch, tmp_path):
    scene, out = tmp_path / 'mipnerf360', tmp_path / 'out'
    scene.mkdir()
    class Runner:
        def __init__(self, profile):
            self.stage_seconds = {}
            self.optional_status = {}
        def conda(self, runtime, *args):
            return [str(a) for a in args]
        def optional(self, name, argv, logs):
            self.run(name, argv, logs)
            self.optional_status[name] = 'ok'
            return True
        def run(self, name, argv, logs):
            self.stage_seconds[name] = {'reference': 120, 'module_b': 40, 'prune_gate': 5}.get(name, 1)
            if name == 'reference':
                assert 'research.reference_native' in argv
                pipeline.write(out / 'results/reference_fastgs/module_b.json', {'total_seconds': 90})
            if name == 'prune_gate':
                pipeline.write(out / 'results/gate.json', {'keep': .6, 'train_lpips': .5, 'tau': .48})
            if name == 'module_b':
                assert 'research.train_confirmed' in argv
                pipeline.write(out / 'results/module_b/train_record.json', dict(
                    completed_iterations=5000, stop_reason='iterations', total_seconds=20,
                    num_final_gaussians=10, runtime_audit=dict(optimizer_calls=5000,
                    vgg_steps=list(range(4801, 5001)))))
            if name == 'metrics':
                pipeline.write(out / 'results/module_b/metrics.json', dict(mean=dict(psnr=27, ssim=.85, lpips=.2)))
    monkeypatch.setattr(pipeline, 'Runner', Runner)
    report = pipeline.main(['--scene', str(scene), '--output', str(out), '--mid-prune', 'auto'])
    assert report['budget_seconds'] == 120 and report['reference_internal_seconds'] == 90
    assert report['b_seconds'] == 45 and report['internal_b_seconds'] == 20
    assert report['proxy_score']['b_seconds'] == 45
    assert report['proxy_score']['valid']


def test_partial_optimizer_or_vgg_run_cannot_be_reported_valid():
    args = SimpleNamespace(preset='confirmed', b_iterations=5000, vgg_tail=200)
    record = dict(stop_reason='iterations', completed_iterations=5000,
                  runtime_audit=dict(optimizer_calls=5000, vgg_steps=list(range(4801, 5001))))
    pipeline.validate_completion(args, record)
    for change in [dict(optimizer_calls=4999), dict(vgg_steps=list(range(4802, 5001)))]:
        with pytest.raises(RuntimeError):
            pipeline.validate_completion(args, {**record, 'runtime_audit': {**record['runtime_audit'], **change}})
    with pytest.raises(RuntimeError):
        pipeline.validate_completion(args, {**record, 'stop_reason': 'wall_clock'})


def test_reference_budget_rejects_resplat_continuation_before_cuda_imports():
    with pytest.raises(ValueError):
        reference_native(['-m', 'out', '--init_random', '--init_ply', 'resplat.ply'])
    with pytest.raises(ValueError):
        reference_native(['-m', 'out'])


def test_export_without_git_remains_identified(tmp_path):
    (tmp_path / 'run_pipeline.py').write_text('print(1)\n')
    identity = pipeline.code_identity(tmp_path)
    assert identity['git_commit'] is None
    assert len(identity['source_sha256']['run_pipeline.py']) == 64


@pytest.mark.parametrize('count', [0, 7])
def test_vectorized_ply_matches_actual_upstream_writer_bytes(tmp_path, count):
    source = pipeline.REPO / 'competition/upstream/baseline/fastgs/scene/gaussian_model.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GaussianModel')
    names = ('save_ply', 'construct_list_of_attributes')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(np=np, PlyData=PlyData, PlyElement=PlyElement,
                     os=__import__('os'), mkdir_p=lambda p: Path(p).mkdir(parents=True, exist_ok=True))
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)
    model = SimpleNamespace(_xyz=torch.randn(count, 3), _features_dc=torch.randn(count, 1, 3),
                            _features_rest=torch.randn(count, 15, 3), _opacity=torch.randn(count, 1),
                            _scaling=torch.randn(count, 3), _rotation=torch.randn(count, 4))
    model.construct_list_of_attributes = lambda: namespace['construct_list_of_attributes'](model)
    native, vector = tmp_path / 'native.ply', tmp_path / 'vector.ply'
    namespace['save_ply'](model, str(native))
    save_ply_vectorized(model, vector)
    assert native.read_bytes() == vector.read_bytes()


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_scores_rejected(bad):
    with pytest.raises(ValueError):
        pipeline.proxy(27, .8, bad, 20, 100)
