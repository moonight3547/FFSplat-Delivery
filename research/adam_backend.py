"""Explicit optimizer component selection; caller retains all schedules and losses.

Use ``with AdamBackend(requested, manifest) as backend`` around training, call
``backend.training_setup(model, options, native_setup)`` in the existing setup
hook, and store ``backend.audit(model)`` after the final update. Missing official
dependencies raise; switching back requires an explicit ``foreach`` selection.
"""
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

from competition.contracts import sha256 as file_sha

OFFICIAL_COMMIT = 'd23bb784d8f19b2432d847b80418d56f0494f599'
OFFICIAL_CLASS_SHA = '9ea0b47233f08dbf318bb60e693e3f5417d45f9c6107a7e4cdf5e3db60601ea6'
OFFICIAL_CUDA_SHA = 'f7fed10a3d9f97b7392797272bff83d069e7e6a1d78c5e1725ec5726201ab237'


class AdamBackend:
    def __init__(self, requested, manifest=None):
        if requested not in ('foreach', 'official'):
            raise ValueError('Choose foreach or official explicitly')
        if requested == 'official' and manifest is None:
            raise ValueError('Requested official Adam requires its installation manifest; no automatic fallback')
        if os.environ.get('FFSPLAT_OFFICIAL_ADAM') == '1':
            raise ValueError('Legacy environment switching is disabled; use explicit backend selection')
        self.requested, self.manifest = requested, manifest
        self.restores, self.model = [], None
        self.counts = dict(official_kernel=0, _multi_tensor_adam=0, _single_tensor_adam=0, _fused_adam=0)

    def _observe(self, module, name, counter):
        original = getattr(module, name)
        def observed(*args, **kwargs):
            self.counts[counter] += 1
            return original(*args, **kwargs)
        self.restores.append((module, name, original))
        setattr(module, name, observed)

    def __enter__(self):
        import torch
        self.native_class = torch.optim.Adam
        if self.requested == 'foreach':
            self.constructor = self.native_class
            path = Path(inspect.getfile(self.native_class)).resolve()
            self.identity = dict(requested='foreach', actual='foreach', actual_class=self.native_class.__qualname__,
                                 actual_module=self.native_class.__module__, class_file=str(path), class_SHA=file_sha(path),
                                 extension_file=None, extension_SHA=None, official_source_commit=None)
        else:
            manifest = json.loads(Path(self.manifest).read_text())
            source, site = Path(manifest['official_source']).resolve(), Path(manifest['isolated_site']).resolve()
            if manifest['official_commit'] != OFFICIAL_COMMIT:
                raise RuntimeError('Official Adam source pin mismatch')
            if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != OFFICIAL_COMMIT:
                raise RuntimeError('Installed official source HEAD changed')
            source_class = source/'FasterGSCudaBackend/FasterGSCudaBackend/torch_bindings/adam.py'
            source_cuda = source/'FasterGSCudaBackend/FasterGSCudaBackend/adam/src/adam.cu'
            if file_sha(source_class) != OFFICIAL_CLASS_SHA or file_sha(source_cuda) != OFFICIAL_CUDA_SHA:
                raise RuntimeError('Official class or CUDA source changed')
            if (torch.__version__, torch.version.cuda, bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                    f'{sys.version_info.major}.{sys.version_info.minor}') != (manifest['Torch'], manifest['CUDA'], manifest['ABI'], manifest['Python']):
                raise RuntimeError('Official Adam Torch/CUDA/Python/ABI mismatch; install in an isolated compatible target')
            sys.path.insert(0, str(site))
            try:
                from FasterGSCudaBackend import _C
                from FasterGSCudaBackend.torch_bindings.adam import FusedAdam
            except Exception as error:
                raise RuntimeError('Requested official Adam is unavailable; explicitly select foreach to use the fallback') from error
            class_path, extension = Path(inspect.getfile(FusedAdam)).resolve(), Path(_C.__file__).resolve()
            if (class_path != Path(manifest['class_file']).resolve() or not class_path.is_relative_to(site)
                    or extension != Path(manifest['extension_file']).resolve() or not extension.is_relative_to(site)
                    or file_sha(class_path) != OFFICIAL_CLASS_SHA or file_sha(extension) != manifest['extension_SHA']):
                raise RuntimeError('Actual imported official Adam class/extension differs from its manifest')
            self.constructor = FusedAdam
            self.identity = dict(requested='official', actual='official', actual_class=FusedAdam.__qualname__,
                                 actual_module=FusedAdam.__module__, class_file=str(class_path), class_SHA=file_sha(class_path),
                                 extension_file=str(extension), extension_SHA=file_sha(extension),
                                 official_source_commit=OFFICIAL_COMMIT, official_source=str(source),
                                 CUDA_source_SHA=file_sha(source_cuda), install_manifest_SHA=file_sha(self.manifest),
                                 build_flags=manifest['build_flags'], numerical_bitwise_equivalence_claim=False)
            self._observe(_C, 'adam_step', 'official_kernel')
        adam = importlib.import_module('torch.optim.adam')
        for name in ('_multi_tensor_adam', '_single_tensor_adam', '_fused_adam'):
            self._observe(adam, name, name)
        self.identity.update(Torch=torch.__version__, CUDA=torch.version.cuda, ABI=bool(torch._C._GLIBCXX_USE_CXX11_ABI))
        return self

    def training_setup(self, model, options, native_setup):
        import torch
        if self.model is not None:
            raise RuntimeError('Each backend session handles one fresh model')
        parameters = [id(getattr(model, name)) for name in ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')]
        old_factory = torch.optim.Adam
        try:
            torch.optim.Adam = self.constructor
            result = native_setup(model, options)
        finally:
            torch.optim.Adam = old_factory
        assert parameters == [id(getattr(model, name)) for name in ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')]
        for optimizer in (model.optimizer, model.shoptimizer):
            assert type(optimizer) is self.constructor and not optimizer.state
            if self.requested == 'foreach':
                optimizer.defaults.update(foreach=True, fused=False)
                for group in optimizer.param_groups:
                    group.update(foreach=True, fused=False)
        self.model = model
        return result

    def audit(self, model):
        import torch
        if model is not self.model:
            raise RuntimeError('Backend audit must inspect its actual model')
        groups = {}
        for label in ('optimizer', 'shoptimizer'):
            optimizer = getattr(model, label)
            assert type(optimizer) is self.constructor
            for group in optimizer.param_groups:
                parameter, = group['params']
                assert group['eps'] == 1e-15 and list(group['betas']) == [.9, .999]
                clock = optimizer.state[parameter].get('step')
                groups[label+'/'+group['name']] = dict(lr=group['lr'], eps=group['eps'], betas=list(group['betas']),
                    foreach=group.get('foreach'), fused=group.get('fused'), state_step=None if clock is None else float(clock),
                    parameter_dtype=str(parameter.dtype), parameter_device=str(parameter.device))
                if self.requested == 'official':
                    assert parameter.dtype == torch.float32 and parameter.is_cuda and parameter.is_contiguous()
                else:
                    assert group['foreach'] and not group['fused']
        assert len(groups) == 6
        if self.requested == 'official':
            assert self.counts['official_kernel'] > 0 and not any(v for k, v in self.counts.items() if k != 'official_kernel')
        else:
            assert self.counts['_multi_tensor_adam'] > 0 and self.counts['_single_tensor_adam'] == self.counts['_fused_adam'] == self.counts['official_kernel'] == 0
        return dict(self.identity, calls=dict(self.counts), groups=groups,
                    optimizer_rebuilt=False, model_parameters_replaced_by_backend=False,
                    schedule_or_gradient_accumulation_modified=False, explicit_foreach_fallback_available=True)

    def __exit__(self, *error):
        for module, name, original in reversed(self.restores):
            setattr(module, name, original)
        self.restores.clear()
