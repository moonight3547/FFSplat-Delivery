"""Run the pinned ReSplat entrypoint with an explicit inference knob set.

Identical to ``competition.resplat`` (same seeding) but forwards every CLI flag
through, so research can vary e.g. ``--num_refine`` without editing the wrapper.

Run in the resplat conda env, from the repository root:

  python -m research.resplat_variant --work work/<case> --out <fresh dir> \
      --checkpoint .../resplat-base-dl3dv-540x960-view16-a72dc6d0.pth \
      --num_refine 4 --device cuda:0
"""
import sys
from pathlib import Path

from competition.resplat import load_runner

REPO = Path(__file__).resolve().parents[1]


def main():
    upstream = load_runner(REPO / "competition/upstream")
    sys.argv = ["organizer_run_resplat", *sys.argv[1:]]
    upstream.main()


if __name__ == "__main__":
    main()
