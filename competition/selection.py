"""Bounded16-view proposals and ranking using reconstruction-only probes."""
import math
import numpy as np


def proposals(views, context, probe_count=8, max_changes=2):
    names = [v["name"] for v in views]
    if len(set(names)) != len(names) or not set(context) <= set(names):
        raise ValueError("Context must be unique training views")
    remaining = [n for n in names if n not in context]
    if len(remaining) < 2:
        return [], [list(context)]
    probes = [remaining[i] for i in np.linspace(0, len(remaining)-1, min(probe_count, len(remaining)), dtype=int)]
    allowed = [i for i, n in enumerate(names) if n not in probes]
    poses = np.stack([v["c2w"] for v in views]).astype(np.float64)
    positions, directions = poses[:, :3, 3], poses[:, :3, 2]
    distance = np.linalg.norm(positions[:, None]-positions[None, :], axis=-1)
    positive = distance[distance > 1e-12]
    scale = np.median(positive) if len(positive) else 1.
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    cost = (distance/scale)**2 + (1-np.clip(directions@directions.T, -1, 1))
    chosen = [names.index(n) for n in context]
    candidates = [list(context)]
    objective = np.min(cost[:, chosen], axis=1).mean()
    for _ in range(max_changes):
        best = None
        for slot in range(len(chosen)):
            others = chosen[:slot]+chosen[slot+1:]
            nearest = np.min(cost[:, others], axis=1) if others else np.full(len(names), np.inf)
            for candidate in allowed:
                if candidate in chosen: continue
                value = np.minimum(nearest, cost[:, candidate]).mean()
                if value < objective-1e-12 and (best is None or value < best[0]-1e-12):
                    best = (value, slot, candidate)
        if best is None: break
        objective, slot, candidate = best
        chosen[slot] = candidate
        proposal = [names[i] for i in sorted(chosen)]
        if proposal not in candidates: candidates.append(proposal)
    return probes, candidates


def rank_candidates(candidates, evaluate, relative_gain=.02):
    """Evaluate every bounded proposal; keep uniform unless declared gain is met."""
    if not 0 <= relative_gain < 1: raise ValueError("Invalid gain threshold")
    rows = []
    for context in candidates:
        row = dict(evaluate(context), context=list(context))
        if not math.isfinite(row["loss"]) or row["loss"] < 0:
            raise ValueError("Nonfinite/negative reconstruction probe loss")
        rows.append(row)
    if not rows: raise ValueError("No proposals")
    best = min(range(len(rows)), key=lambda i: rows[i]["loss"])
    selected = best if rows[best]["loss"] < rows[0]["loss"]*(1-relative_gain) else 0
    return selected, rows


def evaluate_context(upstream, model, views, context, probes, shape, device):
    """Refine on context only, then render probes without another model update."""
    import torch
    from skimage.metrics import structural_similarity
    encoder, decoder, shim = model
    by_name = {v["name"]: v for v in views}
    if not set(context).isdisjoint(probes) or not set(context+probes) <= set(by_name):
        raise ValueError("Probes must be disjoint reconstruction-only views")
    h, w = shape
    def inputs(names):
        chosen = [by_name[n] for n in names]
        rgb = upstream.load_and_preprocess_images([v["path"] for v in chosen], h, w)
        poses = torch.tensor(np.stack([v["c2w"] for v in chosen]), dtype=torch.float32)
        ks = np.stack([v["K"] for v in chosen]).copy()
        ks[:, 0] /= w; ks[:, 1] /= h
        return rgb, poses, torch.tensor(ks, dtype=torch.float32)
    images, poses, ks = inputs(context)
    probe_images, probe_poses, probe_ks = inputs(probes)
    # Never send probe RGB to forward_update: its target is exactly context.
    batch = shim(upstream.build_batch(images, images, poses, poses, ks, ks, .01, 200., "scene", device))
    probe_batch = shim(upstream.build_batch(images, probe_images, poses, probe_poses, ks, probe_ks,
                                           .01, 200., "scene", device))
    with torch.no_grad():
        gaussians, _, _, _ = upstream.run_inference(encoder, decoder, batch, 2, render_chunk_size=8, save_depth=False)
        target = probe_batch["target"]
        predicted = []
        for i in range(0, len(probes), 4):
            rendered = decoder.forward(gaussians, target["extrinsics"][:, i:i+4], target["intrinsics"][:, i:i+4],
                target["near"][:, i:i+4], target["far"][:, i:i+4], (h, w), depth_mode=None)
            predicted.append(rendered.color[0].clamp(0, 1).cpu())
        prediction = torch.cat(predicted).numpy()
        reference = target["image"][0].clamp(0, 1).cpu().numpy()
    l1 = float(np.abs(prediction-reference).mean())
    ssim = float(np.mean([structural_similarity(gt, pred, win_size=11, gaussian_weights=True,
                        channel_axis=0, data_range=1.) for gt, pred in zip(reference, prediction)]))
    return dict(loss=.8*l1+.2*(1-ssim), l1=l1, ssim=ssim)
