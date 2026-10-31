"""Compute PSNR / SSIM / LPIPS for rendered test views (resplat env).

Uses ReSplat's official metric implementations (identical to the competition
leaderboard scoring: PSNR/SSIM/LPIPS, VGG-LPIPS, official chunked code).

Inputs:  <model_path>/test_renders/*.png + <model_path>/test_gt/*.png
Output:  <model_path>/metrics.json  (+ per-view metrics)

Usage (resplat conda env, from baseline/):
  python evaluate.py --model_path results/<group>/<scene>/module_b
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as tf
from PIL import Image

REPLAT_ROOT = Path(__file__).resolve().parent.parent / "resplat"
sys.path.insert(0, str(REPLAT_ROOT))

from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim  # noqa: E402


def load_batch(directory, names):
    to_tensor = tf.ToTensor()
    imgs = []
    for n in names:
        with Image.open(os.path.join(directory, n)) as im:
            imgs.append(to_tensor(im.convert("RGB")))
    return torch.stack(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    renders_dir = os.path.join(args.model_path, "test_renders")
    gt_dir = os.path.join(args.model_path, "test_gt")
    names = sorted(f for f in os.listdir(renders_dir) if f.endswith(".png"))
    if not names:
        sys.exit(f"no renders found in {renders_dir}")

    device = torch.device(args.device)
    rendered = load_batch(renders_dir, names).to(device)
    gt = load_batch(gt_dir, names).to(device)
    assert rendered.shape == gt.shape, "render/GT shape mismatch"

    psnr = compute_psnr(gt, rendered)
    ssim = compute_ssim(gt, rendered)
    lpips = compute_lpips(gt, rendered, chunk_size=4)

    per_view = []
    for i, n in enumerate(names):
        per_view.append({
            "name": n,
            "psnr": round(psnr[i].item(), 4),
            "ssim": round(ssim[i].item(), 4),
            "lpips": round(lpips[i].item(), 4),
        })
    results = {
        "num_views": len(names),
        "mean": {
            "psnr": round(psnr.mean().item(), 3),
            "ssim": round(ssim.mean().item(), 4),
            "lpips": round(lpips.mean().item(), 4),
        },
        "per_view": per_view,
    }
    out = os.path.join(args.model_path, "metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] {len(names)} views | PSNR {results['mean']['psnr']} | "
          f"SSIM {results['mean']['ssim']} | LPIPS {results['mean']['lpips']} -> {out}")


if __name__ == "__main__":
    main()
