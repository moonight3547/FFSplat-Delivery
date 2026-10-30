"""F0 acceptance check for the SH export fix (CPU only).

A rotation of the SH coefficients must leave the evaluated colour unchanged.
With the pinned export basis the first-order z column has the wrong sign, so a
rotation that mixes z with another axis changes the colour. This script fits the
rotation with the original and with the corrected basis and reports the colour
error of each, in the official FastGS convention.
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

# The pinned entrypoint imports torch at module scope; the SH helpers only need
# numpy, so stub the heavy imports to keep this a CPU-only check.
for _name in ("torch", "torch.nn", "torch.nn.functional", "torchvision", "torchvision.transforms"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
sys.modules["torch"].nn = sys.modules["torch.nn"]
sys.modules["torch"].nn.functional = sys.modules["torch.nn.functional"]

REPO = Path(__file__).resolve().parents[1]
UP = REPO / "competition/upstream/baseline/run_resplat.py"


def load():
    """Exec only the two SH helpers out of the pinned file (no heavy imports)."""
    import ast
    src = UP.read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("eval_sh_basis", "sh_rotation_matrix")]
    ns = {"np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(UP), "exec"), ns)
    return ns


def official_basis(dirs, sh_degree):
    """Official 3DGS/FastGS basis (first-order z term is +C1*z)."""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    C0, C1 = 0.28209479177387814, -0.4886025119029199
    cols = [np.full_like(x, C0)]
    cols += [C1 * y, -C1 * z, C1 * x]
    return np.stack(cols, axis=1)


def colour(sh, Y):
    return (sh @ Y.T) + 0.5


def main():
    mod = load()
    rng = np.random.default_rng(20261030)
    dirs = rng.normal(size=(1024, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    sh = np.zeros(4)
    sh[2] = 0.1  # first-order z coefficient
    R = np.array([[np.cos(np.pi / 2), 0, np.sin(np.pi / 2)],
                  [0, 1, 0],
                  [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2)]])
    rows = []
    for label, patched in (("original basis", False), ("fixed basis", True)):
        basis = official_basis
        _saved = mod['eval_sh_basis']
        if patched:
            orig = mod['eval_sh_basis']

            def fixed(d, deg, _orig=orig):
                Y = _orig(d, deg).copy()
                Y[:, 2] *= -1.0
                return Y
            mod['eval_sh_basis'] = fixed
        D = mod["sh_rotation_matrix"](R, sh_degree=1, n_samples=4000)

        c_before = colour(sh, basis(dirs, 1))
        c_after = colour(sh @ D, basis(dirs @ R.T, 1))
        rows.append((label, float(np.abs(c_before - c_after).max())))
        if patched:  # restore
            mod['eval_sh_basis'] = _saved
    for label, err in rows:
        print(f"{label:16s} max colour error under 90deg y-rotation: {err:.3e}")


if __name__ == "__main__":
    sys.exit(main())
