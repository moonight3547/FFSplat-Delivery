"""Run the pinned ReSplat entrypoint with an explicit inference knob set.

Identical to ``competition.resplat`` (same seeding) but forwards every CLI flag
through, so research can vary e.g. ``--num_refine`` without editing the wrapper.

Optional SH export correction (``--fix-sh-basis``): the pinned export fits a
rotation between two SH bases where the first-order z term is negated relative to
the official 3DGS/FastGS convention (audit: research/advisory/SH_EXPORT_AUDIT_20261030.md).
Negating that basis column before the fit yields S·D_bad·S, i.e. the correct
rotation in the official convention, without touching the protected upstream.
The upstream file is never modified; the patch lives in this process only.

Run in the resplat conda env, from the repository root:

  python -m research.resplat_variant --work work/<case> --out <fresh dir> \
      --checkpoint .../resplat-base-dl3dv-540x960-view16-a72dc6d0.pth \
      --num_refine 4 --device cuda:0 [--fix-sh-basis]
"""
import sys
from pathlib import Path

from competition.resplat import load_runner

REPO = Path(__file__).resolve().parents[1]


def _patch_sh_basis(upstream):
    """Flip the first-order z column so the exported rotation matches FastGS."""
    original = upstream.eval_sh_basis

    def eval_sh_basis_fixed(dirs, sh_degree):
        Y = original(dirs, sh_degree)
        Y = Y.copy()
        Y[:, 2] *= -1.0
        return Y

    upstream.eval_sh_basis = eval_sh_basis_fixed
    return upstream


def main():
    fix = "--fix-sh-basis" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--fix-sh-basis"]
    upstream = load_runner(REPO / "competition/upstream")
    if fix:
        _patch_sh_basis(upstream)
        print("[resplat] SH export basis patched (first-order z sign)", file=sys.stderr)
    sys.argv = ["organizer_run_resplat", *sys.argv[1:]]
    upstream.main()


if __name__ == "__main__":
    main()
