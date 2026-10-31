"""Run the unchanged native reference, excluding only its post-training TEST probe.

The parent measures this complete process, including imports, image loading and
native PLY export. Random initialization, optimizer, resets and update clocks
remain native. The official TEST evaluation runs separately after candidate B.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-m', '--model_path', required=True)
    args, _ = parser.parse_known_args(argv)
    baseline = Path(__file__).resolve().parents[1] / 'competition/upstream/baseline'
    sys.path.insert(0, str(baseline / 'fastgs'))
    from scene import Scene, GaussianModel
    from fused_ssim import fused_ssim  # fail rather than silently disable SSIM
    import torch
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    state = dict(iteration=0, optimizer_calls=0, opacity_reset_steps=[], scenes=[])
    original_init, original_test = Scene.__init__, Scene.getTestCameras
    original_lr = GaussianModel.update_learning_rate
    original_step, original_reset = GaussianModel.optimizer_step, GaussianModel.reset_opacity
    def scene_init(scene, *a, **k):
        original_init(scene, *a, **k)
        state['scenes'].append(scene)
    def update(model, iteration):
        state['iteration'] = iteration
        return original_lr(model, iteration)
    def step(model, iteration):
        state['optimizer_calls'] += 1
        return original_step(model, iteration)
    def reset(model):
        state['opacity_reset_steps'].append(state['iteration'])
        return original_reset(model)
    Scene.__init__ = scene_init
    Scene.getTestCameras = lambda scene, *a, **k: []
    GaussianModel.update_learning_rate = update
    GaussianModel.optimizer_step = step
    GaussianModel.reset_opacity = reset
    original_argv = sys.argv
    source = baseline / 'module_b/train_from_ply_fastgs.py'
    try:
        sys.argv = [str(source), *argv]
        runpy.run_path(str(source), run_name='__main__')
        torch.cuda.synchronize()
        scene, = state['scenes']
        record_path = Path(args.model_path) / 'module_b.json'
        record = json.loads(record_path.read_text())
        record['num_test_views'] = len(original_test(scene))
        if state['optimizer_calls'] != record['iterations'] - 1:
            raise RuntimeError('Native reference optimizer call count changed')
        if record['iterations'] == 30000 and state['opacity_reset_steps'] != [3000, 6000, 9000, 12000]:
            raise RuntimeError('Native 30K reference opacity reset window changed')
        record['runtime_audit'] = dict(
            optimizer_calls=state['optimizer_calls'],
            opacity_reset_steps=state['opacity_reset_steps'], quick_test_disabled=True,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            optimizer='native_unmodified', ply_backend='native_unmodified',
            precision='fp32', matmul_tf32=False, cudnn_tf32=False)
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        return record
    finally:
        sys.argv = original_argv
        Scene.__init__, Scene.getTestCameras = original_init, original_test
        GaussianModel.update_learning_rate = original_lr
        GaussianModel.optimizer_step, GaussianModel.reset_opacity = original_step, original_reset


if __name__ == '__main__':
    main()
