"""Single-pass parallel I/O; metadata/split body copied from pinned official A.

Only the image loop is parallelized and PNG compression is disabled. The
protected upstream file is unchanged; pixel and metadata equivalence is audited.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

WORKERS = 4

def write_work_dir(out_dir, scene_dir, pose_dict, num_context, llffhold, max_resolution):
    os.makedirs(os.path.join(out_dir, "sparse/0"), exist_ok=True)
    img_out = os.path.join(out_dir, "images")
    os.makedirs(img_out, exist_ok=True)

    entries = natural_image_list(scene_dir, pose_dict)
    if not entries:
        raise RuntimeError("no images resolved")
    names = [n for n, _ in entries]

    with Image.open(entries[0][1]) as im:
        ow, oh = im.size

    # Resolution policy (mirrors benchmarks/gs_compare_20261020): landscape
    # long side <= max_resolution (ReSplat preset, 960); portrait long side
    # capped at 768 (images stay upright via the portrait-permitting patch in
    # resplat/src/model/encoder/unimatch/mv_unimatch.py; a 960-long-side
    # portrait blows the cross-view cost volume past 24GB). H/W are rounded
    # to multiples of 64. Images are NEVER rotated.
    max_long = 768 if oh > ow else max_resolution
    target_h, target_w = compute_target_shape(oh, ow, max_long, None)
    sx, sy = target_w / float(ow), target_h / float(oh)

    # Resize + save images as PNG
    t0 = time.time()
    def convert(entry):
        name, path = entry
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
        stem = os.path.splitext(name)[0]
        resized.save(os.path.join(img_out, stem + ".png"), compress_level=0, optimize=False)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(convert, entries))
    resize_time = time.time() - t0

    # COLMAP txt (intrinsics rescaled from the pose source's resolution)
    intr = pose_dict
    sx, sy = target_w / float(intr["W"]), target_h / float(intr["H"])
    cameras_txt = os.path.join(out_dir, "sparse/0/cameras.txt")
    with open(cameras_txt, "w") as f:
        f.write(f"1 PINHOLE {target_w} {target_h} "
                f"{intr['fx'] * sx:.10g} {intr['fy'] * sy:.10g} "
                f"{intr['cx'] * sx:.10g} {intr['cy'] * sy:.10g}\n")
    images_txt = os.path.join(out_dir, "sparse/0/images.txt")
    kept = []
    with open(images_txt, "w") as f:
        for idx, name in enumerate(names):
            if name not in intr["poses"]:
                print(f"  [warn] no pose for {name}, dropping")
                continue
            R, t = intr["poses"][name]
            q = rotmat2qvec(R)
            stem = os.path.splitext(name)[0]
            f.write(f"{idx + 1} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g} "
                    f"{t[0]:.12g} {t[1]:.12g} {t[2]:.12g} 1 {stem}.png\n\n")
            kept.append(stem + ".png")

    # Split + context selection (mirrors vanilla 3DGS llffhold split)
    order = sorted(kept)
    test_idx = [i for i in range(len(order)) if i % llffhold == 0]
    train_idx = [i for i in range(len(order)) if i % llffhold != 0]
    if num_context >= len(train_idx):
        ctx = list(range(len(train_idx)))
    else:
        sel = np.linspace(0, len(train_idx) - 1, num_context, dtype=int)
        ctx = sorted(set(int(i) for i in sel))
    split = {
        "images": order,
        "train": [order[i] for i in train_idx],
        "test": [order[i] for i in test_idx],
        "context": [order[train_idx[i]] for i in ctx],
        "llffhold": llffhold,
        "resolution": [target_h, target_w],
    }
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=2)
    return split, resize_time

def main():
    global WORKERS, natural_image_list, compute_target_shape, rotmat2qvec
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workers", type=int, default=4, choices=(4, 8))
    args, remaining = parser.parse_known_args()
    WORKERS = args.workers
    script = Path(__file__).resolve().parents[1] / "competition/upstream/baseline/module_a/preprocess.py"
    spec = importlib.util.spec_from_file_location("official_a_parallel_io", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    natural_image_list = module.natural_image_list
    compute_target_shape = module.compute_target_shape
    rotmat2qvec = module.rotmat2qvec
    module.write_work_dir = write_work_dir
    sys.argv = [str(script), *remaining]
    module.main()


if __name__ == "__main__":
    main()
