"""Delivery ReSplat entry: original network/weights, refine=2, corrected SH export."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.resplat_variant import main

if __name__ == '__main__':
    if '--num_refine' not in sys.argv:
        sys.argv += ['--num_refine', '2']
    if '--checkpoint' not in sys.argv:
        sys.argv += ['--checkpoint', str(Path(__file__).resolve().parents[1] / 'resplat/pretrained/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth')]
    if '--fix-sh-basis' not in sys.argv:
        sys.argv.append('--fix-sh-basis')
    main()
