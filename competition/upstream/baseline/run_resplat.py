"""ReSplat stage: 16-context-view feed-forward reconstruction (baseline).

Consumes Module A's work dir (COLMAP txt + split.json), runs the pretrained
ReSplat model (dl3dv_16v_540x960 preset) on the 16 uniformly-selected views
(refinement targets = the same 16 views; no extra views enter ReSplat), then
exports the Gaussians as a world-space PLY in the classic 3DGS layout for
Module B's continuation training.

Coordinate note: infer_colmap.build_batch normalizes all poses to the middle
context view (pivot). The predicted Gaussians live in that normalized space;
we transform them back to the COLMAP world frame before export:
    X_world = pivot_c2w @ X_norm        (means, rotations)

Usage (resplat conda env):
  python run_resplat.py --work work/<group>/<scene> \
      --out results/<group>/<scene>/resplat \
      --checkpoint resplat/pretrained/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image

BASELINE_ROOT = Path(__file__).resolve().parent
REPLAT_ROOT = BASELINE_ROOT.parent / "resplat"
sys.path.insert(0, str(REPLAT_ROOT))
sys.path.insert(0, str(REPLAT_ROOT / "scripts"))

from infer_colmap import (  # noqa: E402
    build_model,
    build_batch,
    load_colmap_scene,
    load_and_preprocess_images,
    run_inference,
)

PRESET_OVERRIDES = [
    "model.encoder.gaussian_adapter.gaussian_scale_max=3.",
    "model.encoder.depth_pred_half_res=true",
    "+model.encoder.render_chunk_size=8",
]


def eval_sh_basis(dirs, sh_degree):
    """3DGS/gsplat real SH basis (C++ eval_sh convention) for directions [K,3]."""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    C0 = 0.28209479177387814
    C1 = -0.4886025119029199
    C2 = np.array([1.0925484305920792, -1.0925484305920792,
                   0.31539156525252005, -1.0925484305920792,
                   0.5462742152960396])
    C3 = np.array([-0.5900435899266435, 2.890611442640554,
                   -0.4570457994644658, 0.3731763325901154,
                   -0.4570457994644658, 1.445305721320277,
                   -0.5900435899266435])
    cols = [np.full_like(x, C0)]
    cols += [C1 * y, C1 * z, C1 * x]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    cols += [C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy),
             C2[3] * xz, C2[4] * (xx - yy)]
    cols += [C3[0] * y * (3 * xx - yy), C3[1] * xy * z,
             C3[2] * y * (4 * zz - xx - yy),
             C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
             C3[4] * x * (4 * zz - xx - yy), C3[5] * z * (xx - yy),
             C3[6] * x * (xx - 3 * yy)]
    return np.stack(cols, axis=1)


def sh_rotation_matrix(R, sh_degree=3, n_samples=4000):
    """Real Wigner-D rotation for SH coefficients (block-diagonal per degree).

    The rasterizers evaluate SH with world-frame view directions, so a rigid
    world rotation R requires coefficient rotation c' = D(R) c to preserve
    colors. Built numerically: Y(R d) = D Y(d) on sample directions.
    """
    # quasi-uniform directions (golden spiral)
    i = np.arange(n_samples) + 0.5
    phi = np.arccos(1 - 2 * i / n_samples)
    theta = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)
    Y = eval_sh_basis(dirs, sh_degree)
    Yr = eval_sh_basis(dirs @ R.T, sh_degree)
    blocks = []
    start = 0
    for _l in range(sh_degree + 1):
        n = 2 * _l + 1
        A, B = Y[:, start:start + n], Yr[:, start:start + n]
        Dt, _, _, _ = np.linalg.lstsq(A, B, rcond=None)  # B = A @ D^T
        blocks.append(Dt.T)
        start += n
    D = np.zeros((start, start))
    start = 0
    for b in blocks:
        n = b.shape[0]
        D[start:start + n, start:start + n] = b
        start += n
    return D


def quat_xyzw_to_rotmat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotmat_to_quat_xyzw(R):
    """Shepperd's method; returns unit quaternion (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def export_world_ply(path, means, scales, quats_xyzw, harmonics, opacities):
    """Write Gaussians in the classic 3DGS PLY layout (world space).

    means [N,3], scales [N,3] (positive), quats_xyzw [N,4] (unit),
    harmonics [N,3,d_sh] (d_sh = 16), opacities [N] in (0,1).
    Flat f_rest layout matches classic save_ply: index = channel * (d_sh-1) + sh.
    """
    from plyfile import PlyData, PlyElement

    N = means.shape[0]
    d_sh = harmonics.shape[-1]
    num_rest = 3 * (d_sh - 1)

    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    dtype_full = [(a, "f4") for a in attributes]

    f_dc = harmonics[:, :, 0]
    f_rest = harmonics[:, :, 1:].reshape(N, num_rest)
    op = np.clip(opacities, 1e-6, 1 - 1e-6)
    logit_op = np.log(op / (1 - op))
    log_scales = np.log(np.clip(scales, 1e-9, None))
    rots_wxyz = np.stack([quats_xyzw[:, 3], quats_xyzw[:, 0],
                          quats_xyzw[:, 1], quats_xyzw[:, 2]], axis=1)
    zeros = np.zeros_like(means)

    elements = np.empty(N, dtype=dtype_full)
    arrs = (means, zeros, f_dc, f_rest, logit_op, log_scales, rots_wxyz)
    flat = np.concatenate([a.reshape(N, -1) if a.ndim > 1 else a[:, None] for a in arrs], axis=1)
    elements[:] = list(map(tuple, flat))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="Module A output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--num_refine", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--near", type=float, default=0.01)
    ap.add_argument("--far", type=float, default=200.0)
    args = ap.parse_args()

    if args.checkpoint is None:
        args.checkpoint = str(REPLAT_ROOT / "pretrained" /
                               "resplat-base-dl3dv-540x960-view16-a72dc6d0.pth")
    if not os.path.exists(args.checkpoint):
        sys.exit(f"checkpoint not found: {args.checkpoint}\n"
                 "download it per resplat/MODEL_ZOO.md into resplat/pretrained/")

    t_total0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    split = json.load(open(os.path.join(args.work, "split.json")))
    target_h, target_w = split["resolution"]

    # ---- load scene --------------------------------------------------------
    scene_data = load_colmap_scene(args.work, "sparse/0", "images")
    names = scene_data["image_names"]
    assert names == split["images"], "work dir image order mismatch"
    ctx_idx = [names.index(n) for n in split["context"]]
    Vc = len(ctx_idx)
    print(f"[resplat] {len(names)} views, {Vc} context views, "
          f"res {target_h}x{target_w}")

    context_images = load_and_preprocess_images(
        [scene_data["image_paths"][i] for i in ctx_idx], target_h, target_w)
    context_c2w = torch.tensor(scene_data["c2w"][ctx_idx], dtype=torch.float32)
    context_K = torch.tensor(scene_data["intrinsics"][ctx_idx], dtype=torch.float32)

    # ---- model -------------------------------------------------------------
    t0 = time.time()
    encoder, decoder, data_shim = build_model(
        experiment="dl3dv",
        checkpoint=args.checkpoint,
        num_refine=args.num_refine,
        image_shape=(target_h, target_w),
        overrides=list(PRESET_OVERRIDES),
        device=args.device,
    )
    model_load_s = time.time() - t0
    print(f"[resplat] model loaded in {model_load_s:.1f}s")

    # ---- inference ---------------------------------------------------------
    # targets = the 16 context views themselves (self-supervised refinement;
    # no additional views are fed to ReSplat in this baseline)
    batch = build_batch(
        context_images, context_images,
        context_c2w, context_c2w,
        context_K, context_K,
        args.near, args.far,
        "scene", args.device,
    )
    batch = data_shim(batch)

    t0 = time.time()
    gaussians, rendered, _, _ = run_inference(
        encoder, decoder, batch, args.num_refine, render_chunk_size=8,
        save_depth=False,
    )
    infer_s = time.time() - t0
    print(f"[resplat] inference done in {infer_s:.1f}s "
          f"({gaussians.means.shape[1]} gaussians)")

    # ---- sanity render of the context views --------------------------------
    try:
        os.makedirs(os.path.join(args.out, "context_render"), exist_ok=True)
        for i, n in enumerate(split["context"]):
            img = rendered[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            Image.fromarray((img * 255).astype(np.uint8)).save(
                os.path.join(args.out, "context_render", n))
        gt = context_images.to(rendered.device)
        psnr = -10 * torch.log10(((rendered.clamp(0, 1) - gt) ** 2).mean()).item()
        print(f"[resplat] context self-render PSNR {psnr:.2f} dB")
    except Exception as e:  # noqa: BLE001
        print(f"[resplat] sanity render skipped: {e}")

    # ---- un-normalize to COLMAP world --------------------------------------
    # ReSplat renders with precomputed covariances: for a gaussian generated by
    # context view v, the world-frame rotation is
    #   R_world = R_c2w[v] @ R_local        (encoder_resplat.py builds cov this way)
    # while gaussians.rotations only stores the LOCAL rotation. The means are
    # already world (batch-frame) coordinates, so they only need the pivot
    # transform: X_world = pivot_c2w @ X_norm.
    t0 = time.time()
    mid = Vc // 2
    pivot = context_c2w[mid].numpy().astype(np.float64)  # world c2w of pivot view
    R_p, t_p = pivot[:3, :3], pivot[:3, 3]

    means_n = gaussians.means[0].detach().cpu().numpy().astype(np.float64)
    scales = gaussians.scales[0].detach().cpu().numpy().astype(np.float64)
    quats_n = gaussians.rotations[0].detach().cpu().numpy().astype(np.float64)
    harmonics = gaussians.harmonics[0].detach().cpu().numpy().astype(np.float64)
    opacities = gaussians.opacities[0].detach().cpu().numpy().astype(np.float64)

    N = means_n.shape[0]
    n_per_view, rem = divmod(N, Vc)
    assert rem == 0, f"gaussian count {N} not divisible by {Vc} context views"
    view_of_gaussian = np.repeat(np.arange(Vc), n_per_view)

    means_w = means_n @ R_p.T + t_p
    R_ctx_w = context_c2w.numpy().astype(np.float64)[:, :3, :3]  # [Vc,3,3] world c2w rotations
    R_local = np.stack([quat_xyzw_to_rotmat(q) for q in quats_n])       # [N,3,3]
    R_w = np.einsum("nij,njk->nik", R_ctx_w[view_of_gaussian], R_local)  # [N,3,3]
    quats_w = np.stack([rotmat_to_quat_xyzw(R) for R in R_w])

    # Cull gaussians beyond the far plane for every context view: ReSplat's own
    # rendering (gsplat, far_plane=args.far) never shows them, but the 3DGS
    # rasterizer has no far culling, so keeping them would corrupt Module B's
    # initialization. (They are depth-outlier splats.)
    ctx_centers = context_c2w.numpy().astype(np.float64)[:, :3, 3]       # [Vc,3]
    min_dist = np.linalg.norm(
        means_w[:, None, :] - ctx_centers[None, :, :], axis=2).min(axis=1)
    keep = (min_dist <= args.far) & (opacities >= 0.005)
    n_culled = int((~keep).sum())
    if n_culled:
        print(f"[resplat] culling {n_culled} gaussians (beyond far={args.far} "
              f"or opacity < 0.005; {100 * n_culled / N:.1f}%)")
        means_w, scales, quats_w = means_w[keep], scales[keep], quats_w[keep]
        harmonics, opacities = harmonics[keep], opacities[keep]

    # SH coefficients live in the (normalized) world frame the rasterizer
    # evaluates view directions in; rotating to the COLMAP world frame
    # requires the matching Wigner-D coefficient rotation.
    sh_degree = int(np.sqrt(harmonics.shape[-1])) - 1
    D_sh = sh_rotation_matrix(R_p, sh_degree=sh_degree)
    harmonics = np.einsum("ab,ncb->nca", D_sh, harmonics)
    export_s = time.time() - t0

    ply_path = os.path.join(args.out, "resplat_world.ply")
    export_world_ply(ply_path, means_w, scales, quats_w, harmonics, opacities)
    print(f"[resplat] world PLY -> {ply_path} "
          f"({os.path.getsize(ply_path) / 1e6:.1f} MB)")

    # store raw tensors as well (for debugging / player convenience)
    np.savez_compressed(
        os.path.join(args.out, "resplat_world.npz"),
        means=means_w, scales=scales, rotations_xyzw=quats_w,
        harmonics=harmonics, opacities=opacities)

    total_s = time.time() - t_total0
    info = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "num_context": Vc,
        "num_refine": args.num_refine,
        "resolution": [target_h, target_w],
        "num_gaussians": int(means_w.shape[0]),
        "model_load_seconds": round(model_load_s, 2),
        "inference_seconds": round(infer_s, 2),
        "export_seconds": round(export_s, 2),
        "total_seconds": round(total_s, 2),
    }
    with open(os.path.join(args.out, "resplat.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[resplat] stage total {total_s:.1f}s "
          f"(model {model_load_s:.1f}s + infer {infer_s:.1f}s + export {export_s:.1f}s)")


if __name__ == "__main__":
    main()
