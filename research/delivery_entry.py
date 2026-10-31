"""Defaults for direct invocation of the official-shaped module-B entry."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from run_pipeline import recipe_flags


def module_b_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('--iterations', type=int, default=5000)
    args, _ = parser.parse_known_args(argv)
    result = list(argv)
    def missing(flag):
        return not any(a == flag or a.startswith(flag + '=') for a in result)
    defaults = ['--iterations', str(args.iterations),
                '--position_lr_max_steps', str(args.iterations),
                '--opacity_reset_interval', '0',
                '--densify_from_iter', str(max(20, args.iterations // 10)),
                '--lambda_dssim', '0.6', '--scaling_lr', '0.01', '--realloc_frac', '0.0',
                '--sh_degree', '3',
                '--seed', '20261015', '--record', str(Path(args.model_path) / 'train_record.json')]
    defaults += recipe_flags(SimpleNamespace(b_recipe='bigd3500', b_iterations=args.iterations,
                                            vgg_tail=min(200, args.iterations)))
    for flag, value in zip(defaults[::2], defaults[1::2]):
        if missing(flag):
            result += [flag, value]
    if '--no_quick_test' not in result:
        result.append('--no_quick_test')
    if '--eval' not in result:
        result.append('--eval')
    if missing('-r') and missing('--resolution'):
        result += ['-r', '1']
    return result
