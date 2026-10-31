"""Export committed runnable source in the official baseline/resplat layout."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent


def include(name):
    parts = Path(name).parts
    if parts[0] == 'research' and len(parts) > 1:
        return parts[1] not in ('advisory', 'experiments', 'evidence', '__pycache__')
    if parts[0] == 'records':
        return len(parts) > 1 and parts[1] in ('h3-param-confirm-20261031', 'delivery-runtime-gate-20261031', 'official-adam-prior')
    return parts[0] in ('baseline', 'resplat', 'competition', 'configs', 'docs') or name in (
        'run_pipeline.py', 'README.md', 'AGENTS.md', 'ProblemDesc.md', 'DELIVERY.md')


def build(output, revision='HEAD'):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    commit = subprocess.check_output(['git', 'rev-parse', revision], cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(['git', 'archive', '--format=tar', commit], cwd=ROOT)
    files = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source, zipfile.ZipFile(
            output, 'x', compression=zipfile.ZIP_DEFLATED) as target:
        for member in source:
            if not member.isfile() or not include(member.name):
                continue
            data = source.extractfile(member).read()
            target.writestr(member.name, data)
            files[member.name] = hashlib.sha256(data).hexdigest()
        required = ('baseline/run_train.sh', 'baseline/run_eval.sh', 'baseline/run_resplat.py',
                    'baseline/module_a/preprocess.py', 'baseline/module_b/train_from_ply_fastgs.py',
                    'resplat/README.md', 'run_pipeline.py', 'configs/confirmed_recipe.json')
        if any(name not in files for name in required):
            raise ValueError('Committed revision is missing submission entrypoints')
        manifest = dict(commit=commit, file_count=len(files), sha256=files,
                        weights_included=False, datasets_included=False,
                        excluded='Git history, research reports/queues/raw evidence, private credentials')
        target.writestr('SUBMISSION.json', json.dumps(manifest, indent=2) + '\n')
    return dict(output=str(output.resolve()), commit=commit, files=len(files), bytes=output.stat().st_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', default='HEAD', help='committed revision, default HEAD')
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.revision), indent=2))
