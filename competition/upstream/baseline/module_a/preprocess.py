"""Module A: pose acquisition + uniform view selection (competition baseline).

Pipeline position: BEFORE ReSplat. Produces a canonical COLMAP-format work dir
that both ReSplat (run_resplat.py) and Module B (3DGS continuation) consume.

Pose sources (first available wins, "skip if present"):
  1. sparse/0/{cameras,images}.{bin,txt}   -> use directly (e.g. mipnerf360)
  2. transforms.json (Blender/DL3DV style) -> convert to COLMAP (no estimation)
  3. frames_info.txt (B011/hw AR dump)     -> AR positions are geometrically
     invalid (validated against an independent COLMAP reconstruction: rotations
     match to 0.5 deg, positions fail every rigid-alignment test), so we RUN
     COLMAP from scratch on the images instead.
  4. nothing                               -> run COLMAP from scratch.

Outputs (<out_dir>):
  images/<name>.png            images resized (LANCZOS) to the evaluation
                               resolution (ReSplat rule: max side <= 960,
                               multiples of 64)
  sparse/0/cameras.txt         PINHOLE intrinsics scaled to the eval resolution
  sparse/0/images.txt          COLMAP w2c poses
  split.json                   train/test (llffhold) + 16 context indices
  module_a.json                provenance + wall-clock timing

Usage (resplat conda env):
  python module_a/preprocess.py \
      --scene_dir datasets/examples/DL3DV/<hash> \
      --group dl3dv \
      --out work/dl3dv/<hash>
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Reuse ReSplat's COLMAP readers / resolution rule (torch is imported at module
# load in infer_colmap, so this script must run in the resplat environment).
REPLAT_ROOT = Path(__file__).resolve().parent.parent.parent / "resplat"
sys.path.insert(0, str(REPLAT_ROOT / "scripts"))
from infer_colmap import (  # noqa: E402
    compute_target_shape,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def rotmat2qvec(R):
    """Rotation matrix -> COLMAP wxyz quaternion (round-trip verified)."""
    import scipy.spatial.transform as sst
    q_xyzw = sst.Rotation.from_matrix(R).as_quat()
    if q_xyzw[3] < 0:
        q_xyzw = -q_xyzw
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


# ----------------------------------------------------------------------------
# Pose source 1: existing COLMAP
# ----------------------------------------------------------------------------

def load_colmap_poses(scene_dir):
    """Return {name: (R_w2c 3x3, t_w2c 3,)} + one camera dict or None."""
    sp = os.path.join(scene_dir, "sparse/0")
    for ext in ("bin", "txt"):
        cpath, ipath = f"{sp}/cameras.{ext}", f"{sp}/images.{ext}"
        if not (os.path.exists(cpath) and os.path.exists(ipath)):
            continue
        if ext == "bin":
            cams = read_intrinsics_binary(cpath)
            imgs = read_extrinsics_binary(ipath)
        else:
            cams = read_intrinsics_text(cpath)
            imgs = read_extrinsics_text(ipath)
        cam = sorted(cams.values())[0]
        model = cam.model
        W, H = cam.width, cam.height
        params = np.asarray(cam.params, dtype=np.float64)
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1:3]
        elif model in ("OPENCV", "SIMPLE_RADIAL", "RADIAL"):
            # distortion ignored (video data, small distortion) - PINHOLE approx
            fx = fy = params[0]
            if model == "SIMPLE_RADIAL":
                fx = fy = params[0]
                cx, cy = W / 2.0, H / 2.0
            else:
                fx, fy = params[:2]
                cx, cy = params[2:4]
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {model}")
        poses = {}
        for img in imgs.values():
            poses[img.name] = (qvec2rotmat(img.qvec), np.asarray(img.tvec, dtype=np.float64))
        return {"model": "PINHOLE", "W": W, "H": H, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "poses": poses, "source": f"colmap.{ext}"}
    return None


# ----------------------------------------------------------------------------
# Pose source 2: transforms.json (DL3DV / Blender style)
# ----------------------------------------------------------------------------

def load_transforms_json(scene_dir):
    """DL3DV/Blender transforms.json -> COLMAP-equivalent dict.

    Stored transform_matrix is c2w in OpenGL convention (Y up, Z back).
    COLMAP uses OpenCV cameras (Y down, Z forward): flip columns 1 and 2.
    Intrinsics are given at the transforms.json's own w/h; they are rescaled
    later when we resize images.
    """
    path = os.path.join(scene_dir, "transforms.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    W, H = int(d["w"]), int(d["h"])
    fx, fy = float(d["fl_x"]), float(d["fl_y"])
    cx, cy = float(d["cx"]), float(d["cy"])

    def find_image(frame):
        for key in ("file_path",):
            rel = frame.get(key)
            if not rel:
                continue
            for cand in (rel, os.path.join("images_4", os.path.basename(rel)),
                         os.path.join("images_2", os.path.basename(rel)),
                         os.path.join("images", os.path.basename(rel))):
                p = os.path.join(scene_dir, cand)
                if os.path.exists(p):
                    return p
        return None

    poses = {}
    for frame in d["frames"]:
        img_path = find_image(frame)
        if img_path is None:
            continue
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV camera axes
        w2c = np.linalg.inv(c2w)
        poses[os.path.basename(img_path)] = (w2c[:3, :3], w2c[:3, 3])
    return {"model": "PINHOLE", "W": W, "H": H, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "poses": poses, "source": "transforms.json",
            "image_resolver": "find_image"}


# ----------------------------------------------------------------------------
# Pose source 3/4: run COLMAP
# ----------------------------------------------------------------------------

def run_colmap(image_paths, work_tmp):
    """Run COLMAP (feature extraction + sequential matching + mapper).

    Returns the COLMAP-equivalent dict (poses in COLMAP world) or raises.
    """
    os.makedirs(work_tmp, exist_ok=True)
    img_dir = os.path.join(work_tmp, "input_images")
    os.makedirs(img_dir, exist_ok=True)
    for p in image_paths:
        dst = os.path.join(img_dir, os.path.basename(p))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(p), dst)
    db = os.path.join(work_tmp, "database.db")
    sparse_out = os.path.join(work_tmp, "sparse")
    os.makedirs(sparse_out, exist_ok=True)

    def colmap(*args):
        cmd = ["colmap", *args]
        print("  $ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    use_gpu = 1
    try:
        colmap("feature_extractor",
               "--database_path", db, "--image_path", img_dir,
               "--ImageReader.single_camera", "1",
               "--ImageReader.camera_model", "OPENCV",
               "--SiftExtraction.use_gpu", str(use_gpu))
    except subprocess.CalledProcessError:
        if use_gpu:
            print("  [warn] GPU SIFT failed, retrying CPU")
            if os.path.exists(db):
                os.remove(db)
            use_gpu = 0
            colmap("feature_extractor",
                   "--database_path", db, "--image_path", img_dir,
                   "--ImageReader.single_camera", "1",
                   "--ImageReader.camera_model", "OPENCV",
                   "--SiftExtraction.use_gpu", "0")
        else:
            raise
    n_imgs = len(image_paths)
    if n_imgs <= 120:
        # exhaustive matching: globally consistent poses (mirrors the
        # benchmarks/gs_compare_20261020 calibration: their full-BA COLMAP
        # poses let ReSplat reach ~18 PSNR on b011 scenes, while quick
        # sequential matching is noticeably drifter)
        colmap("exhaustive_matcher",
               "--database_path", db,
               "--SiftMatching.use_gpu", "0")
    else:
        colmap("sequential_matcher",
               "--database_path", db,
               "--SequentialMatching.overlap", "10",
               "--SequentialMatching.quadratic_overlap", "1",
               "--SiftMatching.use_gpu", "0")
    colmap("mapper",
           "--database_path", db, "--image_path", img_dir,
           "--output_path", sparse_out)

    # pick the sub-model with most images
    models = sorted(os.listdir(sparse_out))
    best = None
    for m in models:
        d = os.path.join(sparse_out, m)
        if os.path.exists(os.path.join(d, "images.bin")):
            n = len(read_extrinsics_binary(os.path.join(d, "images.bin")))
            if best is None or n > best[1]:
                best = (d, n)
    if best is None:
        raise RuntimeError("COLMAP mapper produced no model")
    model_dir, n = best
    print(f"  COLMAP registered {n} images")

    # Undistort (classic 3DGS convert.py path): the AR-captured jpgs have real
    # lens distortion; forcing PINHOLE during mapping bakes the distortion into
    # drifty poses (validated: 5.5 dB ReSplat quality loss vs the benchmark's
    # undistorted-image calibration). image_undistorter warps the images to
    # PINHOLE and adjusts K; poses stay globally consistent.
    undist_dir = os.path.join(work_tmp, "undist")
    colmap("image_undistorter",
           "--image_path", img_dir,
           "--input_path", model_dir,
           "--output_path", undist_dir)
    model_dir = os.path.join(undist_dir, "sparse")
    img_dir = os.path.join(undist_dir, "images")
    cams = read_intrinsics_binary(os.path.join(model_dir, "cameras.bin"))
    imgs = read_extrinsics_binary(os.path.join(model_dir, "images.bin"))
    cam = sorted(cams.values())[0]
    params = np.asarray(cam.params, dtype=np.float64)
    if cam.model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif cam.model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        fx = fy = params[0]
        cx, cy = cam.width / 2.0, cam.height / 2.0
    poses = {}
    for img in imgs.values():
        poses[img.name] = (qvec2rotmat(img.qvec), np.asarray(img.tvec, dtype=np.float64))
    return {"model": "PINHOLE", "W": cam.width, "H": cam.height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "poses": poses, "source": "colmap_run",
            "images_dir": img_dir}


# ----------------------------------------------------------------------------
# Work-dir writing
# ----------------------------------------------------------------------------

def natural_image_list(scene_dir, pose_dict):
    """Images to use, in canonical (sorted) order.

    For pose-carrying sources the pose dict defines the name set. When the
    same filename exists in several folders (e.g. images/ and images_4/), the
    copy matching the pose intrinsics' resolution is preferred.
    """
    if pose_dict is not None and pose_dict["poses"]:
        names = sorted(pose_dict["poses"].keys())
        want = (pose_dict.get("W"), pose_dict.get("H"))
        candidates = {}
        roots = []
        if pose_dict.get("images_dir"):
            # the pose source's own images (e.g. COLMAP undistorted output)
            roots.append(pose_dict["images_dir"])
        for root, _, _ in os.walk(scene_dir):
            roots.append(root)
        for root in roots:
            for f in sorted(os.listdir(root)):
                p = os.path.join(root, f)
                if os.path.isfile(p) and f in names:
                    candidates.setdefault(f, []).append(p)
        out = []
        for n in names:
            paths = candidates.get(n)
            if not paths:
                continue
            if len(paths) > 1 and want[0]:
                for p in paths:
                    try:
                        with Image.open(p) as im:
                            if im.size == want:
                                paths = [p]
                                break
                    except Exception:
                        pass
            if len(paths) > 1:
                paths = [max(paths, key=lambda p: os.path.getsize(p))]
            out.append((n, paths[0]))
        return out
    # fallback: all images in <scene>/images*
    for cand in ("images", "images_4", "images_2", "images_8"):
        d = os.path.join(scene_dir, cand)
        if os.path.isdir(d):
            names = sorted(f for f in os.listdir(d)
                           if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
            return [(n, os.path.join(d, n)) for n in names]
    raise FileNotFoundError(f"No images found under {scene_dir}")




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
    for name, path in entries:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
        stem = os.path.splitext(name)[0]
        im.save(os.path.join(img_out, stem + ".png"))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_context", type=int, default=16)
    ap.add_argument("--llffhold", type=int, default=8)
    ap.add_argument("--max_resolution", type=int, default=960)
    ap.add_argument("--tmp_root", default=None,
                    help="scratch dir for COLMAP runs (default: <out>_colmap_tmp)")
    args = ap.parse_args()

    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)

    # ---- pose acquisition -------------------------------------------------
    pose_dict = load_colmap_poses(args.scene_dir)
    if pose_dict is not None:
        print(f"[A] using existing COLMAP poses ({pose_dict['source']})")
    else:
        tj = load_transforms_json(args.scene_dir)
        if tj is not None:
            print("[A] using transforms.json poses (converted, no estimation)")
            pose_dict = tj
        else:
            fi = os.path.join(args.scene_dir, "frames_info.txt")
            reason = ("frames_info.txt present but AR positions are geometrically "
                      "invalid (see README); running COLMAP" if os.path.exists(fi)
                      else "no poses found; running COLMAP")
            print(f"[A] {reason}")
            t0 = time.time()
            if os.path.exists(fi):
                # images referenced by frames_info only (excludes poster PNG etc.)
                keep = set()
                with open(fi) as f:
                    for line in f:
                        if line.startswith("#") or not line.strip():
                            continue
                        p = line.split()
                        if len(p) >= 15:
                            keep.add(p[1] + ".jpg")
                all_imgs = []
                d = os.path.join(args.scene_dir, "images")
                for n in sorted(os.listdir(d)):
                    if os.path.splitext(n)[1].lower() in IMAGE_EXTS and n in keep:
                        all_imgs.append(os.path.join(d, n))
            else:
                all_imgs = [p for _, p in natural_image_list(args.scene_dir, None)]
            tmp = args.tmp_root or (args.out.rstrip("/") + "_colmap_tmp")
            pose_dict = run_colmap(all_imgs, tmp)
            pose_dict["colmap_seconds"] = time.time() - t0
            print(f"[A] COLMAP done in {pose_dict['colmap_seconds']:.1f}s")

    # ---- work dir ----------------------------------------------------------
    split, resize_time = write_work_dir(
        args.out, args.scene_dir, pose_dict,
        args.num_context, args.llffhold, args.max_resolution)

    elapsed = time.time() - t_start
    info = {
        "group": args.group,
        "scene_dir": os.path.abspath(args.scene_dir),
        "pose_source": pose_dict["source"],
        "pose_estimation_seconds": pose_dict.get("colmap_seconds", 0.0),
        "resize_seconds": resize_time,
        "total_seconds": elapsed,
        "num_images": len(split["images"]),
        "num_train": len(split["train"]),
        "num_test": len(split["test"]),
        "num_context": len(split["context"]),
        "resolution": split["resolution"],
    }
    with open(os.path.join(args.out, "module_a.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[A] done: {len(split['images'])} imgs "
          f"({len(split['train'])} train / {len(split['test'])} test), "
          f"{len(split['context'])} context, res={split['resolution']}, "
          f"{elapsed:.1f}s")


if __name__ == "__main__":
    main()
