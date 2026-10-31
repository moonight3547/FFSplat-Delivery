"""Confirmed FastGS runtime, leaving the pinned upstream implementation untouched."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import random


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--record', required=True)
    parser.add_argument('--adam-backend', choices=('foreach', 'official'), default='foreach')
    parser.add_argument('--adam-manifest', type=Path)
    known, remaining = parser.parse_known_args(argv)
    argv = [*remaining, '--record', known.record]
    if os.environ.get('FFSPLAT_OFFICIAL_ADAM') == '1' or '--fused_adam' in argv:
        raise ValueError('Use explicit Adam backend selection; legacy environment/fused switching is disabled')
    from research import train_sweep as trainer
    from research.perceptual_loss import load_perceptual, optimizer_clocks
    from research.vectorized_ply import save_ply_vectorized
    from research.adam_backend import AdamBackend
    import lpips
    import torch
    backend = AdamBackend(known.adam_backend, known.adam_manifest)

    if not trainer.FUSED_SSIM_AVAILABLE:
        raise RuntimeError('fused_ssim is required; refusing the silent SSIM=1 fallback')
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    state = dict(seed=0, iteration=0, optimizer_calls=0, resets=[],
                 vgg_steps=[], vgg_values=[], exports=[], models=[])
    camera_hash = hashlib.sha256()
    sampling = dict(render_calls=0, density=[])
    originals = []

    def patch(obj, name, replacement):
        originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    seed_all, safe_state = trainer.seed_all, trainer.safe_state
    def seed(seed):
        state['seed'] = seed
        seed_all(seed)
    def safe(quiet):
        safe_state(quiet)
        seed_all(state['seed'])
    patch(trainer, 'seed_all', seed)
    patch(trainer, 'safe_state', safe)

    cls = trainer.GaussianModel
    setup, update, step, reset = (cls.training_setup, cls.update_learning_rate,
                                 cls.optimizer_step, cls.reset_opacity)
    def training_setup(model, options):
        backend.training_setup(model, options, setup)
        state['models'].append(model)
    def update_lr(model, iteration):
        state['iteration'] = iteration
        return update(model, iteration)
    def optimizer_step(model, iteration):
        state['optimizer_calls'] += 1
        return step(model, iteration)
    def reset_opacity(model):
        state['resets'].append(state['iteration'])
        return reset(model)
    def save_ply(model, path):
        began = time.perf_counter()
        save_ply_vectorized(model, path)
        state['exports'].append(dict(path=str(path), bytes=Path(path).stat().st_size,
                                     seconds=time.perf_counter() - began))
    patch(cls, 'training_setup', training_setup)
    patch(cls, 'update_learning_rate', update_lr)
    patch(cls, 'optimizer_step', optimizer_step)
    patch(cls, 'reset_opacity', reset_opacity)
    patch(cls, 'save_ply', save_ply)

    original_render, original_score = trainer.render_fastgs, trainer.compute_gaussian_score_fastgs
    def render(camera, *args, **kwargs):
        camera_hash.update((str(camera.image_name) + '\n').encode())
        sampling['render_calls'] += 1
        return original_render(camera, *args, **kwargs)
    def score(cameras, *args, **kwargs):
        event = dict(iteration=state['iteration'], cameras=[c.image_name for c in cameras],
                     rng_before=hashlib.sha256(repr(random.getstate()).encode()).hexdigest())
        result = original_score(cameras, *args, **kwargs)
        event['rng_after'] = hashlib.sha256(repr(random.getstate()).encode()).hexdigest()
        sampling['density'].append(event)
        return result
    patch(trainer, 'render_fastgs', render)
    patch(trainer, 'compute_gaussian_score_fastgs', score)

    constructor = lpips.LPIPS
    def perceptual(*args, **kwargs):
        if args or kwargs.get('net') != 'vgg':
            raise ValueError('Confirmed perceptual backbone must be VGG')
        model, identity = load_perceptual('vgg', constructor=constructor)
        state['perceptual'] = identity
        def observe(module, inputs, output):
            state['vgg_steps'].append(state['iteration'])
            state['vgg_values'].append(output.detach().mean())
        model.register_forward_hook(observe)
        return model
    patch(lpips, 'LPIPS', perceptual)
    try:
        backend.__enter__()
        if '--no_quick_test' not in argv:
            argv.append('--no_quick_test')
        record = trainer.main(argv)
        torch.cuda.synchronize()
        model, = state['models']
        optimizer_flags = [dict(name=g['name'], foreach=g.get('foreach'), fused=g.get('fused'))
                           for o in (model.optimizer, model.shoptimizer) for g in o.param_groups]
        backend_proof = backend.audit(model)
        values = (torch.stack(state['vgg_values']).cpu().tolist()
                  if state['vgg_values'] else [])
        record['runtime_audit'] = dict(
            seed=state['seed'], seed_restored_after_safe_state=True,
            precision='fp32', matmul_tf32=False, cudnn_tf32=False,
            optimizer=('torch_adam_foreach' if known.adam_backend == 'foreach' else 'official_faster_gs_adam'),
            backend=backend_proof, optimizer_groups=optimizer_flags,
            optimizer_calls=state['optimizer_calls'], optimizer_clocks=optimizer_clocks(model),
            opacity_reset_steps=state['resets'], vgg_steps=state['vgg_steps'],
            vgg_values=values, perceptual=state.get('perceptual'),
            ply_backend='vectorized_lossless', exports=state['exports'],
            sampling_audit=dict(**sampling, view_sequence_sha256=camera_hash.hexdigest()),
            quick_test_disabled=True, timing_scope='internal scene load through final export')
        Path(known.record).write_text(json.dumps(record, indent=2) + '\n')
        return record
    finally:
        backend.__exit__(None, None, None)
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)


if __name__ == '__main__':
    main()
