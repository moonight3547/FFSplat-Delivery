"""Inspect and rewrite ReSplat's world-space Gaussian PLY before module B.

These are module-B *approaches* (not hyper-parameters): what we hand to the
optimizer changes the solution, not just its speed.

  inspect   print count / opacity / scale statistics
  prune     drop floaters (low opacity, huge anisotropic scale) before training
  sharpen   re-scale opacities toward 0/1 so FastGS pruning is not dominated by mush
  densify   split every Gaussian into N children (scale down, jittered mean) so the
            optimizer starts with enough capacity to fit detail

Run in the fastgs env:
  python -m research.ply_ops --in resplat_world.ply --out new.ply --op densify --factor 2
"""
import argparse
import json

import numpy as np
from plyfile import PlyData, PlyElement


def load(path):
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    names = [p.name for p in vertex.properties]
    data = {n: np.asarray(vertex[n]) for n in names}
    return data, names


def save(path, data, names):
    dtype = [(n, "f4") for n in names]
    array = np.empty(len(data["x"]), dtype=dtype)
    for n in names:
        array[n] = data[n].astype("f4")
    PlyData([PlyElement.describe(array, "vertex")]).write(path)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def scale_stats(data):
    cols = [n for n in data if n.startswith("scale_")]
    s = np.stack([np.exp(data[c]) for c in cols], axis=1)
    return s


def stats(data):
    op = sigmoid(data["opacity"])
    s = scale_stats(data)
    return dict(count=int(len(op)), opacity=dict(mean=float(op.mean()),
                  p05=float(np.quantile(op, 0.05)), p95=float(np.quantile(op, 0.95)),
                  below_001=int((op < 0.01).sum())),
                scale=dict(mean=float(s.mean()), max=float(s.max()),
                           p99=float(np.quantile(s, 0.99))))


def op_prune(data, min_opacity=0.01, max_scale=0.5):
    op = sigmoid(data["opacity"])
    s = scale_stats(data).max(axis=1)
    keep = (op >= min_opacity) & (s <= max_scale)
    return {k: v[keep] for k, v in data.items()}, dict(kept=int(keep.sum()), dropped=int((~keep).sum()))


def op_sharpen(data, gain=2.0):
    out = dict(data)
    out["opacity"] = logit(np.clip(sigmoid(data["opacity"]), 1e-4, 1 - 1e-4)) * gain
    return out, dict(gain=gain)


def op_densify(data, factor=2, shrink=0.7, jitter=0.15):
    rng = np.random.default_rng(20261030)
    index = []
    for child in range(factor):
        index.append(np.arange(len(data["x"])))
    index = np.concatenate(index)
    out = {k: v[index] for k, v in data.items()}
    total = len(index)
    child_id = np.tile(np.arange(factor), len(data["x"]))
    # child 0 keeps the original; every other child is a shrunk, jittered copy
    is_copy = child_id > 0
    cols = [n for n in data if n.startswith("scale_")]
    for c in cols:
        out[c] = out[c].copy()
        out[c][is_copy] = np.log(np.exp(out[c][is_copy]) * shrink)
    # jitter the mean inside one original sigma
    sigma = np.exp(np.mean(np.stack([data[c] for c in cols], axis=1), axis=1))
    offset = np.zeros((total, 3), dtype=np.float32)
    offset[is_copy] = rng.normal(0.0, 1.0, (int(is_copy.sum()), 3)) * (jitter * sigma[is_copy])[:, None]
    for i, axis in enumerate(("x", "y", "z")):
        out[axis] = out[axis] + offset[:, i]
    return out, dict(factor=factor, shrink=shrink, jitter=jitter, out_count=int(total))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--op", required=True, choices=["inspect", "prune", "sharpen", "densify"])
    p.add_argument("--factor", type=int, default=2)
    p.add_argument("--shrink", type=float, default=0.7)
    p.add_argument("--jitter", type=float, default=0.15)
    p.add_argument("--gain", type=float, default=2.0)
    p.add_argument("--min-opacity", type=float, default=0.01, dest="min_opacity")
    p.add_argument("--max-scale", type=float, default=0.5, dest="max_scale")
    args = p.parse_args(argv)

    data, names = load(args.input)
    before = stats(data)
    if args.op == "inspect":
        print(json.dumps(dict(path=args.input, **before), indent=2))
        return
    if args.op == "prune":
        out, info = op_prune(data, args.min_opacity, args.max_scale)
    elif args.op == "sharpen":
        out, info = op_sharpen(data, args.gain)
    else:
        out, info = op_densify(data, args.factor, args.shrink, args.jitter)
    save(args.out, out, names)
    print(json.dumps(dict(op=args.op, input=args.input, output=args.out, info=info,
                          before=before, after=stats(out)), indent=2))


if __name__ == "__main__":
    main()
