"""User-confirmed native random FastGS30K budget; not classic 3DGS."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.reference_native import main

if __name__ == '__main__':
    argv = list(sys.argv[1:])
    for flag, value in (('--iterations', '30000'), ('--position_lr_max_steps', '30000'),
                        ('--densify_until_iter', '15000')):
        if not any(a == flag or a.startswith(flag + '=') for a in argv):
            argv += [flag, value]
    if '--init_random' not in argv:
        argv.append('--init_random')
    if '--eval' not in argv:
        argv.append('--eval')
    if not any(a in ('-r', '--resolution') or a.startswith('--resolution=') for a in argv):
        argv += ['-r', '1']
    main(argv)
