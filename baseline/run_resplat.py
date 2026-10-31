"""Delivery ReSplat entry: original network/weights, refine=2, corrected SH export."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.resplat_variant import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--num_refine', type=int, default=2)
    args, _ = parser.parse_known_args()
    if args.num_refine != 2:
        parser.error('delivery fixes num_refine=2')
    if not any(a == '--num_refine' or a.startswith('--num_refine=') for a in sys.argv):
        sys.argv += ['--num_refine', '2']
    if '--checkpoint' not in sys.argv:
        sys.argv += ['--checkpoint', str(Path(__file__).resolve().parents[1] / 'resplat/pretrained/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth')]
    if '--fix-sh-basis' not in sys.argv:
        sys.argv.append('--fix-sh-basis')
    main()
