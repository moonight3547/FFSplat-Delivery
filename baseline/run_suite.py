"""Official-shaped group driver, also supporting a dry run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_pipeline import CHECKPOINT, DEFAULT_PROFILE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups', nargs='*', metavar='GROUP')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets/examples')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'baseline/results')
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    scenes = json.loads((ROOT / 'baseline/scenes.json').read_text())
    if any(group not in scenes for group in args.groups):
        parser.error('groups must be dl3dv, hw or mipnerf360')
    for group in args.groups or scenes:
        spec = scenes[group]
        family = Path(spec['source']).name
        command = [sys.executable, str(ROOT / 'run_pipeline.py'), '--group', group,
                   '--skip-eval',
                   '--scene', str(args.data_root / family / spec['scene']),
                   '--output', str(args.output_root / group / spec['scene']),
                   '--checkpoint', args.checkpoint, '--profile', str(args.profile)]
        if args.describe:
            command.append('--describe')
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
