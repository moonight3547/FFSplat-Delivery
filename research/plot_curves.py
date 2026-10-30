"""Plot LPIPS-vs-module-B-time curves for the low-scoring scenes.

Reads the curve.json files written by research.curve_metrics (name = snapshot second)
and produces one PNG per scene plus a combined figure. Snapshots are scored after
training, so the x axis is pure module-B time.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("fastgs", "FastGS delivery", "#1f77b4"),
        ("mcmc", "3DGS-MCMC", "#d62728"),
        ("inject", "hole injection", "#2ca02c")]


def load(curve_dir):
    p = Path(curve_dir)
    if not p.is_file():
        return []
    rows = json.load(open(p))
    return sorted((float(r["name"]), r) for r in rows if r.get("name"))


def main(root, out):
    root = Path(root)
    scenes = sorted([d for d in root.iterdir() if d.is_dir()])
    fig, axes = plt.subplots(1, max(1, len(scenes)), figsize=(5 * max(1, len(scenes)), 4),
                             squeeze=False)
    for ax, scene in zip(axes[0], scenes):
        for arm, label, colour in ARMS:
            pts = load(scene / arm / "curve.json")
            if not pts:
                continue
            xs = [x for x, _ in pts]
            lpips = [r["lpips"] for _, r in pts]
            q = [r["quality"] for _, r in pts]
            ax.plot(xs, lpips, "o-", color=colour, label=f"{label} (q {q[-1]:.1f})")
        ax.set_title(scene.name[:28], fontsize=8)
        ax.set_xlabel("module-B seconds")
        ax.set_ylabel("VGG-LPIPS (lower is better)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/curve",
         sys.argv[2] if len(sys.argv) > 2 else "curves_lowcase.png")
