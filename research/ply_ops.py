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



def voxel_merge(data, voxels=128, max_scale_quantile=0.6, keep_small=True):
    """Merge Gaussians that share a voxel, but only the large/low-frequency ones.

    Rationale: a feed-forward init spends primitives roughly uniformly in space, so
    smooth regions are over-sampled and textured regions under-sampled. Merging only
    the primitives whose scale is above `max_scale_quantile` concentrates the budget
    on fine detail while leaving small (high-frequency) primitives untouched.
    """
    import numpy as np
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    lo = np.quantile(xyz, 0.005, axis=0)
    hi = np.quantile(xyz, 0.995, axis=0)
    cell = np.maximum(hi - lo, 1e-6) / voxels
    idx = np.clip(((xyz - lo) / cell).astype(np.int64), 0, voxels - 1)
    keys = (idx[:, 0] * voxels + idx[:, 1]) * voxels + idx[:, 2]
    opacity = sigmoid(data["opacity"].reshape(-1))
    scale = np.stack([np.exp(data[c]) for c in data if c.startswith("scale_")], axis=1).mean(axis=1)
    thresh = np.quantile(scale, max_scale_quantile)
    mergeable = scale >= thresh if keep_small else np.ones_like(scale, dtype=bool)
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    merged = data
    keep_mask = np.ones(len(xyz), dtype=bool)
    # for every voxel that contains at least two mergeable gaussians, keep the most
    # opaque one and drop the rest (a cheap, deterministic merge)
    from collections import defaultdict
    groups = defaultdict(list)
    for pos in order:
        if mergeable[pos]:
            groups[keys[pos]].append(pos)
    drop = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda i: -opacity[i])
        # keep two strongest, drop the rest
        drop.extend(members[2:])
    if drop:
        keep_mask[np.asarray(drop, dtype=np.int64)] = False
    out = {}
    for k, v in data.items():
        try:
            out[k] = v[keep_mask]
        except Exception:
            out[k] = v
    return out, dict(count=int(keep_mask.sum()), merged_away=int((~keep_mask).sum()))


def voxel_fuse(data, voxels=128, max_scale_quantile=1.0, target_frac=1.0):
    """Proper voxel FUSION (InfiniSplat/AnySplat style) instead of dropping primitives.

    Gaussians sharing a voxel are merged into one primitive whose alpha is the
    composited alpha of the group (1 - prod(1 - a_i)), with opacity-weighted centroid,
    scale and SH. This keeps the coverage that the plain "keep the strongest" merge
    throws away.
    """
    import numpy as np
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    lo = np.quantile(xyz, 0.005, axis=0)
    hi = np.quantile(xyz, 0.995, axis=0)
    cell = np.maximum(hi - lo, 1e-6) / voxels
    idx = np.clip(((xyz - lo) / cell).astype(np.int64), 0, voxels - 1)
    keys = (idx[:, 0] * voxels + idx[:, 1]) * voxels + idx[:, 2]
    a = sigmoid(data["opacity"].reshape(-1))
    scale_cols = [c for c in data if c.startswith("scale_")]
    scl = np.stack([np.exp(data[c]) for c in scale_cols], axis=1)
    big = scl.mean(axis=1) >= np.quantile(scl.mean(axis=1), max_scale_quantile)
    order = np.argsort(keys, kind="stable")
    out_rows = []
    names = list(data.keys())
    i = 0
    keys_sorted = keys[order]
    # process the most crowded voxels first so a target_FRAC can stop early
    counts = {}
    for k in keys:
        counts[int(k)] = counts.get(int(k), 0) + 1
    dense = {k for k, c in counts.items() if c >= 2}
    remaining = len(xyz) - int(round(target_frac * len(xyz)))
    fused_so_far = 0
    while i < len(order):
        cur_key = int(keys_sorted[i])
        j = i
        while j + 1 < len(order) and keys_sorted[j + 1] == keys_sorted[i]:
            j += 1
        members = order[i:j + 1]
        i = j + 1
        if len(members) == 1 or cur_key not in dense or fused_so_far >= remaining:
            out_rows.extend(members); continue
        fused_so_far += len(members) - 1
        w = a[members]
        wsum = w.sum() + 1e-8
        row = {n: data[n][members[0]] for n in names}
        centroid = (xyz[members] * w[:, None]).sum(0) / wsum
        row["x"], row["y"], row["z"] = centroid
        combined = 1.0 - np.prod(1.0 - np.clip(w, 0, 0.999))
        row["opacity"] = logit(np.clip(combined, 1e-6, 1 - 1e-6))
        for c_i, c in enumerate([n for n in names if n.startswith("scale_")]):
            row[c] = np.log(max((scl[members][:, c_i] * w).sum() / wsum, 1e-8))
        for c in [n for n in names if n.startswith("rot_")]:
            vals = data[c][members]
            row[c] = vals[int(np.argmax(w))]
        for c in [n for n in names if n.startswith("f_dc_") or n.startswith("f_rest_")]:
            row[c] = (data[c][members] * w).sum() / wsum
        out_rows.append(row)
    out = {n: np.asarray([r[n] if isinstance(r, dict) else data[n][r] for r in out_rows])
           for n in names}
    return out, dict(count=len(out_rows), merged_away=int(len(xyz) - len(out_rows)))


def op_scale_mult(data, mult=1.1):
    """Bake a mild low-pass filter into the PLY by dilating every Gaussian.

    The pinned FastGS rasteriser hardcodes prefiltered=False (no 2D filter), while the
    ReSplat decoder ran gsplat in antialiased mode. On high-frequency scenes that
    mismatch shows up as aliasing, which VGG-LPIPS punishes. Since the deliverable is
    the PLY (the official renderer is fixed), the filter has to live in the parameters:
    a small uniform scale-up softens the reconstruction the same way a prefilter would.
    """
    import numpy as np
    out = dict(data)
    cols = [c for c in data if c.startswith("scale_")]
    for c in cols:
        out[c] = data[c] + float(np.log(mult))
    return out, dict(count=len(data["x"]), mult=float(mult))


def op_texresample(data, voxels=128, seed=0):
    """Texture-adaptive resampling at CONSTANT primitive count.

    Split the scene into voxels, score each one by a texture proxy (the mean inverse
    Gaussian scale: small primitives == high frequency detail), then redistribute a
    fixed budget proportionally to that score. Dense-texture voxels receive extra
    primitives by randomly splitting their Gaussians; smooth voxels lose them by random
    pruning. Unlike voxelmerge/voxelfuse the total never changes, so this isolates the
    effect of the density distribution from the effect of the budget.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    lo, hi = np.quantile(xyz, 0.005, axis=0), np.quantile(xyz, 0.995, axis=0)
    cell = np.maximum(hi - lo, 1e-6) / voxels
    idx = np.clip(((xyz - lo) / cell).astype(np.int64), 0, voxels - 1)
    keys = (idx[:, 0] * voxels + idx[:, 1]) * voxels + idx[:, 2]
    scale_cols = [c for c in data if c.startswith("scale_")]
    scl = np.stack([np.exp(data[c]) for c in scale_cols], axis=1).mean(axis=1)
    tex = 1.0 / np.maximum(scl, 1e-6)          # small primitive -> high texture
    uniq, inv = np.unique(keys, return_inverse=True)
    n_vox = len(uniq)
    counts = np.bincount(inv, minlength=n_vox).astype(np.float64)
    tex_mean = np.bincount(inv, weights=tex, minlength=n_vox) / np.maximum(counts, 1)
    target = len(xyz) * (tex_mean * counts) / np.maximum((tex_mean * counts).sum(), 1e-9)
    target = np.maximum(target, 1.0)
    # integer allocation that keeps the total exactly
    alloc = np.floor(target).astype(np.int64)
    short = len(xyz) - int(alloc.sum())
    if short > 0:
        order = np.argsort(-(target - alloc))
        alloc[order[:short]] += 1
    elif short < 0:
        order = np.argsort(target - alloc)
        for k in order[:(-short)]:
            if alloc[k] > 1:
                alloc[k] -= 1
    keep = np.ones(len(xyz), dtype=bool)
    dup_src = []
    for vi in range(n_vox):
        members = np.nonzero(inv == vi)[0]
        want = int(alloc[vi])
        have = len(members)
        if want < have:
            drop = rng.choice(members, size=have - want, replace=False)
            keep[drop] = False
        elif want > have:
            extra = want - have
            dup_src.extend(rng.choice(members, size=extra, replace=True).tolist())
    out = {}
    for k, v in data.items():
        try:
            arr = v[keep]
            if dup_src and k in ("x", "y", "z"):
                arr = np.concatenate([arr, v[np.asarray(dup_src)]])
            elif dup_src and k.startswith("scale_"):
                arr = np.concatenate([arr, (v[np.asarray(dup_src)] - np.log(1.6))])
            else:
                arr = np.concatenate([arr, v[np.asarray(dup_src)]])
            out[k] = arr
        except Exception:
            out[k] = v
    # jitter the duplicates' positions so they are not exact copies
    n_dup = len(dup_src)
    if n_dup:
        base = len(xyz) - int((~keep).sum())
        jit = rng.normal(scale=0.3, size=(n_dup, 3)) * np.stack(
            [np.exp(out[c][base:]) for c in scale_cols], axis=1)
        out["x"][base:] += jit[:, 0]
        out["y"][base:] += jit[:, 1]
        out["z"][base:] += jit[:, 2]
    # enforce an exact constant count regardless of rounding in the allocation
    cur, tgt = len(out["x"]), len(xyz)
    if cur > tgt:
        drop = rng.choice(cur, size=cur - tgt, replace=False)
        out = {k: np.delete(v, drop, axis=0) for k, v in out.items()}
    elif cur < tgt:
        extra = rng.choice(cur, size=tgt - cur, replace=True)
        out = {k: np.concatenate([v, v[extra]], axis=0) for k, v in out.items()}
    return out, dict(count=len(out["x"]), target_total=len(xyz), n_dup=n_dup)

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
    p.add_argument("--in", dest="inputs", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--op", required=True,
                   choices=["inspect", "prune", "sharpen", "densify", "merge", "voxelmerge",
                            "voxelfuse", "scalemult", "texresample"])
    p.add_argument("--voxels", type=int, default=128)
    p.add_argument("--target-frac", type=float, default=1.0, dest="target_frac")
    p.add_argument("--scale-mult", type=float, default=1.1, dest="scale_mult")
    p.add_argument("--factor", type=int, default=2)
    p.add_argument("--shrink", type=float, default=0.7)
    p.add_argument("--jitter", type=float, default=0.15)
    p.add_argument("--gain", type=float, default=2.0)
    p.add_argument("--min-opacity", type=float, default=0.01, dest="min_opacity")
    p.add_argument("--max-scale", type=float, default=0.5, dest="max_scale")
    args = p.parse_args(argv)

    if args.op == "merge":
        # concatenate several Gaussian sets (e.g. two ReSplat runs on disjoint
        # 16-view subsets) into one init for module B
        parts = []
        for path in args.inputs:
            d, names = load(path)
            parts.append(d)
        merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
        save(args.out, merged, names)
        print(json.dumps(dict(op="merge", inputs=args.inputs, output=args.out,
                              counts=[len(p["x"]) for p in parts],
                              out_count=int(len(merged["x"])), after=stats(merged)), indent=2))
        return

    data, names = load(args.inputs[0])
    before = stats(data)
    if args.op == "voxelmerge":
        import numpy as _np
        before = len(data["x"])
        data, vinfo = voxel_merge(data, voxels=args.voxels)
        print(f"[voxelmerge] {before} -> {vinfo['count']} gaussians "
              f"({vinfo['merged_away']} merged away)")
    if args.op == "inspect":
        print(json.dumps(dict(path=args.inputs[0], **before), indent=2))
        return
    if args.op == "prune":
        out, info = op_prune(data, args.min_opacity, args.max_scale)
    elif args.op == "sharpen":
        out, info = op_sharpen(data, args.gain)
    elif args.op == "voxelmerge":
        out, info = data, vinfo  # already filtered above
    elif args.op == "texresample":
        out, info = op_texresample(data, args.voxels)
    elif args.op == "scalemult":
        out, info = op_scale_mult(data, args.scale_mult)
    elif args.op == "voxelfuse":
        out, info = voxel_fuse(data, voxels=args.voxels, target_frac=args.target_frac)
    else:
        out, info = op_densify(data, args.factor, args.shrink, args.jitter)
    save(args.out, out, names)
    print(json.dumps(dict(op=args.op, input=args.inputs[0], output=args.out, info=info,
                          before=before, after=stats(out)), indent=2))


if __name__ == "__main__":
    main()
