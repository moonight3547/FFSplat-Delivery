"""Re-evaluate all TEST views of a completed single-scene delivery output."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_pipeline import DEFAULT_PROFILE, Runner, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    out = args.output.resolve()
    report = read(out / 'report.json')
    model = out / 'results/module_b'
    runner = Runner(args.profile)
    runner.run('render_repeat', runner.conda('fastgs', ROOT / 'baseline/render_test.py',
        '-s', out / 'work', '-m', model, '--backend', 'fastgs',
        '--iteration', report['b_iterations'], '--eval', '-r', '1', '--sh_degree', '3', '--quiet'), out / 'logs')
    runner.run('metrics_repeat', runner.conda('resplat', ROOT / 'baseline/evaluate.py',
        '--model_path', model, '--device', args.device), out / 'logs')
    print(json.dumps(read(model / 'metrics.json')['mean'], indent=2))


if __name__ == '__main__':
    main()
