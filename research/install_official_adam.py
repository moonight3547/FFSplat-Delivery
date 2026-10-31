"""Pinned official Git source -> isolated package target; never mutates an environment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from urllib.parse import urlsplit

from competition.contracts import sha256 as sha, write_json as write

PIN = 'd23bb784d8f19b2432d847b80418d56f0494f599'
CLASS_SHA = '9ea0b47233f08dbf318bb60e693e3f5417d45f9c6107a7e4cdf5e3db60601ea6'
CUDA_SHA = 'f7fed10a3d9f97b7392797272bff83d069e7e6a1d78c5e1725ec5726201ab237'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--git-source', default='https://github.com/nerficg-project/faster-gaussian-splatting.git')
    p.add_argument('--jobs', type=int, choices=range(1, 5), default=4)
    p.add_argument('--arch', default=None)
    p.add_argument('--reuse-binary', type=Path)
    p.add_argument('--binary-proof', type=Path)
    args = p.parse_args()
    parsed = urlsplit(args.git_source)
    if parsed.username or parsed.password or parsed.scheme not in ('https', 'git'):
        raise ValueError('Use a credential-free official Git URL or an existing local Git mirror')
    root = args.root.resolve()
    if root.exists():
        raise FileExistsError('Choose a fresh installation root; preserve existing artifacts')
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < 2*2**30+300000000:
        raise RuntimeError('Installation needs a 300MB allowance plus a 2GiB filesystem buffer')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Install and validate on the target CUDA node')
    root.mkdir()
    source, site = root/'official', root/'site'
    with (root/'install.log').open('w') as log:
        def run(argv, cwd=None, environment=None):
            log.write(json.dumps(argv)+'\n'); log.flush()
            subprocess.run(argv, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        source.mkdir()
        run(['git', 'init'], source)
        run(['git', 'remote', 'add', 'origin', args.git_source], source)
        run(['git', 'fetch', '--depth', '1', 'origin', PIN], source)
        run(['git', 'sparse-checkout', 'set', 'FasterGSCudaBackend'], source)
        run(['git', 'checkout', '--detach', PIN], source)
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() == PIN
        backend = source/'FasterGSCudaBackend'
        class_source = backend/'FasterGSCudaBackend/torch_bindings/adam.py'
        cuda_source = backend/'FasterGSCudaBackend/adam/src/adam.cu'
        assert sha(class_source) == CLASS_SHA and sha(cuda_source) == CUDA_SHA
        runtime = dict(Torch=torch.__version__, CUDA=torch.version.cuda, ABI=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                       Python=f'{sys.version_info.major}.{sys.version_info.minor}')
        arch = args.arch or '.'.join(map(str, torch.cuda.get_device_capability()))
        if args.reuse_binary:
            if args.binary_proof is None:
                raise ValueError('Binary artifact reuse requires its SHA/build identity proof')
            proof = json.loads(args.binary_proof.read_text())
            assert all(proof[k] == value for k, value in runtime.items())
            assert proof['official_commit'] == PIN and proof['class_SHA'] == CLASS_SHA and proof['CUDA_source_SHA'] == CUDA_SHA
            assert sha(args.reuse_binary) == proof['extension_SHA']
            (site/'FasterGSCudaBackend').mkdir(parents=True)
            shutil.copytree(backend/'FasterGSCudaBackend/torch_bindings', site/'FasterGSCudaBackend/torch_bindings')
            shutil.copyfile(args.reuse_binary, site/'FasterGSCudaBackend'/('_C'+sysconfig.get_config_var('EXT_SUFFIX')))
            flags = proof.get('build_flags', proof.get('original_build_flags'))
            assert flags
        else:
            environment = dict(os.environ, MAX_JOBS=str(args.jobs), TORCH_CUDA_ARCH_LIST=arch)
            run([sys.executable, 'setup.py', 'build_py', '--build-lib', str(site), 'build_ext',
                 '--build-temp', str(root/'build-temp'), '--build-lib', str(site)], backend, environment)
            flags = ['-O3', '-use_fast_math', '-std=c++17', 'TORCH_CUDA_ARCH_LIST='+arch]
        extension, = (site/'FasterGSCudaBackend').glob('_C*.so')
        manifest = dict(runtime, official_commit=PIN, official_source=str(source), isolated_site=str(site),
                        class_file=str(site/'FasterGSCudaBackend/torch_bindings/adam.py'), class_SHA=CLASS_SHA,
                        CUDA_source_SHA=CUDA_SHA, extension_file=str(extension), extension_SHA=sha(extension),
                        build_flags=flags, source_transport='Git', isolated_target=True, shared_environment_modified=False,
                        binary_reused=bool(args.reuse_binary), compiler_jobs=args.jobs,
                        runtime_validation_required='CUDA numeric/state + TRAIN600 before production acceptance')
        write(root/'manifest.json', manifest)
        environment = dict(os.environ, PYTHONPATH=str(site))
        run([sys.executable, '-c', 'import torch; from FasterGSCudaBackend import _C; print(torch.__version__, _C.__file__)'], environment=environment)
    print(str(root/'manifest.json'), flush=True)


if __name__ == '__main__':
    main()
