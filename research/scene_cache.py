"""A-side tensor cache for module B's scene load (user-approved data residency).

Rationale (2026-10-31 user decision): module A already loads the same images/poses;
preparing decoded pixels + camera tensors in A and keeping them resident for B is
latency hiding, not an illegitimate timing shift. Only moving actual optimization
work into A would be out of scope.

The bundle is a versioned pair:
  images_u8.npy   (N, 3, H, W) uint8 NCHW, train order (same pixels as image_cache)
  cams.npz        world_view_transform / full_proj_transform / camera_center /
                  FoVx / FoVy / image names / order / cameras_extent, all f32 CPU
  meta.json       version hash + build seconds + bytes

B-side load: one H->D upload (uint8) + one on-GPU uint8->float/255 convert, then
Camera-compatible objects are built from the cached transforms - no PNG/PIL decode,
no per-camera matrix inversions, no per-camera clamp/ones. Pixel/RNG/order semantics
are identical (verified bitwise against a normally-constructed Scene).

  # A side (build once per prepared work dir; ~2-3 s, mostly the normal Scene load)
  python -m research.scene_cache build --work <work> --out <cache dir>
  # verification + measurement
  python -m research.scene_cache bench --work <work> --cache <cache dir> --reps 3
"""
import argparse
import hashlib
import random
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "competition/upstream/baseline"
sys.path.insert(0, str(BASELINE / "fastgs"))

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from research.image_cache import CachedImages  # noqa: E402
from research.cache_integrity import sha256, scene_identity
from research import cache_rng

CACHE_VERSION = "scene-cache-v3-exact"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _args_for(work, model_path):
    p = argparse.ArgumentParser()
    lp = ModelParams(p)
    op = OptimizationParams(p)
    pp = PipelineParams(p)
    a = p.parse_args(["--source_path", str(work), "--model_path", str(model_path),
                      "--sh_degree", "3", "--eval", "-r", "1"])
    return lp.extract(a), op.extract(a), pp.extract(a)


def version_of(work, seed=20261015):
    return scene_identity(work, seed, CACHE_VERSION)


def install_image_cache(work):
    import scene.dataset_readers as readers
    from PIL import Image as PILImage
    cache = CachedImages(str(Path(work) / "image_cache"))
    orig_open = readers.Image.open

    def cached_open(path, *a, **k):
        name = os.path.basename(str(path))
        if name in cache.index:
            return PILImage.fromarray(cache.get(name))
        return orig_open(path, *a, **k)

    readers.Image = type("CachedPIL", (), {"open": staticmethod(cached_open)})
    return cache


def _seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build(work, out_dir, model_path="/tmp/scene_cache_model", seed=20261015):
    work = Path(work)
    _seed_all(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    current_version = version_of(work, seed)
    try:
        old = json.loads((out / "meta.json").read_text())
        if (old.get("cache_version") == CACHE_VERSION and old.get("seed") == seed
                and old.get("version") == current_version
                and old.get("torch_version") == str(torch.__version__)
                and old.get("images_sha256") == sha256(out / "images_u8.npy")
                and old.get("cameras_sha256") == sha256(out / "cams.npz")):
            (out / "_status.json").write_text(json.dumps(dict(status="reused")))
            return old
    except (OSError, ValueError, KeyError):
        pass
    t0 = time.perf_counter()
    cache = install_image_cache(work)
    ds, opt, pipe = _args_for(work, model_path)
    os.makedirs(model_path, exist_ok=True)
    g = GaussianModel(3, "default")
    rng_before = cache_rng.capture()
    scene = Scene(ds, g, resolution_scales=[1.0])
    rng_after = cache_rng.capture()
    stable_version = version_of(work, seed)
    if stable_version != current_version:
        # Native loaders can create a fallback points3D.ply on the first read.
        # B will see that existing file, so capture its steady input/RNG path.
        del scene, g
        _seed_all(seed)
        g = GaussianModel(3, "default")
        rng_before = cache_rng.capture()
        scene = Scene(ds, g, resolution_scales=[1.0])
        rng_after = cache_rng.capture()
        if version_of(work, seed) != stable_version:
            raise ValueError("Prepared inputs changed while building the scene cache")
    torch.cuda.synchronize()
    build_scene_s = time.perf_counter() - t0
    cams = scene.getTrainCameras()
    n = len(cams)
    H, W = cams[0].image_height, cams[0].image_width
    # pixels: uint8 NCHW from the same cache (same pixels as Camera.original_image)
    cache_names = list(cache.index.keys())
    stem_to_idx = {os.path.splitext(n)[0]: i for i, n in enumerate(cache_names)}
    images = np.lib.format.open_memmap(out / "images_u8.npy", mode="w+",
                                       dtype=np.uint8, shape=(n, 3, H, W))
    t0 = time.perf_counter()
    for i, cam in enumerate(cams):
        stem = os.path.splitext(os.path.basename(cam.image_name))[0]
        if stem not in stem_to_idx:
            raise SystemExit("cache image %r not found in image_cache (%d names)"
                             % (stem, len(cache_names)))
        name = cache_names[stem_to_idx[stem]]
        arr = np.asarray(cache.get(name))  # HWC uint8
        if arr.shape != (H, W, 3):
            raise SystemExit("cache image shape %r != %r" % (arr.shape, (H, W, 3)))
        images[i] = np.transpose(arr, (2, 0, 1))
    images.flush()
    save_images_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    wvt = torch.stack([c.world_view_transform.cpu() for c in cams]).numpy().astype(np.float32)
    fpt = torch.stack([c.full_proj_transform.cpu() for c in cams]).numpy().astype(np.float32)
    centers = torch.stack([c.camera_center.cpu() for c in cams]).numpy().astype(np.float32)
    fovx = np.array([float(c.FoVx) for c in cams], dtype=np.float64)
    fovy = np.array([float(c.FoVy) for c in cams], dtype=np.float64)
    names = np.array([os.path.basename(c.image_name) for c in cams])
    np.savez(out / "cams.npz", wvt=wvt, fpt=fpt, centers=centers, fovx=fovx,
             fovy=fovy, names=names, width=np.int32(W), height=np.int32(H),
             cameras_extent=np.float64(scene.cameras_extent),
             version=np.array(stable_version))
    save_cams_s = time.perf_counter() - t0
    total = time.perf_counter() - t0
    bytes_total = int((out / "images_u8.npy").stat().st_size + (out / "cams.npz").stat().st_size)
    meta = dict(cache_version=CACHE_VERSION, seed=int(seed),
                work=str(work.resolve()), n_train=n,
                resolution=[H, W], cameras_extent=float(scene.cameras_extent),
                build_scene_seconds=round(build_scene_s, 3),
                save_images_seconds=round(save_images_s, 3),
                save_cams_seconds=round(save_cams_s, 3), bytes=bytes_total,
                images_bytes=int((out / "images_u8.npy").stat().st_size),
                gpu_images_f32_bytes=int(n * 3 * H * W * 4),
                version=stable_version,
                images_sha256=sha256(out / "images_u8.npy"),
                cameras_sha256=sha256(out / "cams.npz"),
                rng_before=rng_before, rng_after=rng_after,
                torch_version=str(torch.__version__),
                native_test_count=len(scene.getTestCameras()))
    meta["build_total_seconds"] = time.perf_counter() - began
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "_status.json").write_text(json.dumps(dict(status="built")))
    del scene, g
    torch.cuda.empty_cache()
    return meta


class CachedCamera:
    """Duck-typed Camera with the attributes FastGS training actually uses."""
    __slots__ = ("original_image", "world_view_transform", "full_proj_transform",
                 "camera_center", "FoVx", "FoVy", "image_width", "image_height",
                 "image_name", "uid", "data_device", "znear", "zfar", "R", "T")

    def __init__(self, image, wvt, fpt, center, fovx, fovy, width, height, name, uid):
        self.original_image = image
        self.world_view_transform = wvt
        self.full_proj_transform = fpt
        self.camera_center = center
        self.FoVx = float(fovx)
        self.FoVy = float(fovy)
        self.image_width = int(width)
        self.image_height = int(height)
        self.image_name = name
        self.uid = int(uid)
        self.data_device = image.device
        self.znear, self.zfar = 0.01, 100.0
        self.R, self.T = None, None


class CachedScene:
    def __init__(self, cams, test_cams, cameras_extent):
        self._cams = cams
        self._test = test_cams
        self.cameras_extent = cameras_extent

    def getTrainCameras(self, scale=1.0):
        return self._cams

    def getTestCameras(self):
        return self._test


def load(work, cache_dir, want_test=False, device="cuda"):
    """B-side: rebuild the train cameras from the cache in ~one upload + one convert."""
    work = Path(work)
    out = Path(cache_dir)
    timings = {}
    t0 = time.perf_counter()
    cams_npz = np.load(out / "cams.npz", allow_pickle=False)
    meta = json.loads((out / "meta.json").read_text())
    timings["read_meta_seconds"] = time.perf_counter() - t0
    if (meta.get("cache_version") != CACHE_VERSION
            or str(cams_npz["version"]) != version_of(work, meta["seed"])
            or meta.get("torch_version") != str(torch.__version__)):
        raise ValueError("stale scene cache: rebuild it (%s)" % out)
    if (sha256(out / "images_u8.npy") != meta["images_sha256"]
            or sha256(out / "cams.npz") != meta["cameras_sha256"]):
        raise ValueError("corrupt scene cache: rebuild it (%s)" % out)
    if cache_rng.identity(cache_rng.capture()) != cache_rng.identity(meta["rng_before"]):
        raise ValueError("Scene-cache seed/RNG does not match the native loader")
    images = np.load(out / "images_u8.npy", mmap_mode="r")
    timings["open_npy_seconds"] = time.perf_counter() - t0 - timings["read_meta_seconds"]
    n, C, H, W = images.shape
    t0 = time.perf_counter()
    u8_gpu = torch.from_numpy(np.ascontiguousarray(images)).to(device, non_blocking=False)
    timings["h2d_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    # CPU division exactly matches the pinned PILtoTorch operation for all 256
    # possible channel values; lookup avoids CPU/CUDA reciprocal-rounding drift.
    lut = torch.arange(256, dtype=torch.float32).div_(255.0).to(device)
    f32 = lut[u8_gpu.to(torch.int32)]
    # Native PILtoTorch creates HWC storage viewed as CHW. Retain those strides
    # so VGG/SSIM do not select a different layout-dependent kernel.
    f32 = f32.contiguous(memory_format=torch.channels_last)
    del u8_gpu
    timings["gpu_convert_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    wvt = torch.from_numpy(cams_npz["wvt"]).to(device)
    fpt = torch.from_numpy(cams_npz["fpt"]).to(device)
    centers = torch.from_numpy(cams_npz["centers"]).to(device)
    names = [str(x) for x in cams_npz["names"]]
    fovx, fovy = cams_npz["fovx"], cams_npz["fovy"]
    width, height = int(cams_npz["width"]), int(cams_npz["height"])
    cams = [CachedCamera(f32[i], wvt[i], fpt[i], centers[i], fovx[i], fovy[i],
                         width, height, names[i], i) for i in range(n)]
    timings["build_cams_seconds"] = time.perf_counter() - t0
    timings["total_seconds"] = sum(timings.values())
    test_cams = []
    if want_test:
        import scene.dataset_readers as readers
        from utils.camera_utils import cameraList_from_camInfos
        info = readers.sceneLoadTypeCallbacks["Colmap"](str(work), "images", True)
        ds, opt, pipe = _args_for(work, "/tmp/scene_cache_model")
        test_cams = cameraList_from_camInfos(info.test_cameras, 1.0, ds)
    scene = CachedScene(cams, test_cams, float(cams_npz["cameras_extent"]))
    scene.load_timings = timings
    scene.native_test_count = meta["native_test_count"]
    cache_rng.restore(meta["rng_after"])
    return scene


# ---------------------------------------------------------------- CLI
def _verify(work, cache_dir):
    """Bitwise identity: cached cams/images vs a normally constructed Scene."""
    meta = json.loads((Path(cache_dir) / "meta.json").read_text())
    _seed_all(meta.get("seed", 20261015))
    install_image_cache(work)
    ds, opt, pipe = _args_for(work, "/tmp/scene_cache_model")
    os.makedirs("/tmp/scene_cache_model", exist_ok=True)
    g = GaussianModel(3, "default")
    ref = Scene(ds, g, resolution_scales=[1.0])
    expected_rng = cache_rng.capture()
    _seed_all(meta.get("seed", 20261015))
    cached = load(work, cache_dir, want_test=False)
    out_rng_equal = cache_rng.identity(cache_rng.capture()) == cache_rng.identity(expected_rng)
    ref_cams = ref.getTrainCameras()
    out = dict(n_train=len(ref_cams), images_maxdiff=None,
               wvt_maxdiff=None, fpt_maxdiff=None, center_maxdiff=None,
               fov_maxdiff=None, rng_exact=out_rng_equal,
               names_exact=([c.image_name for c in ref.getTrainCameras()] ==
                            [c.image_name for c in cached.getTrainCameras()]),
               extent_exact=bool(ref.cameras_extent == cached.cameras_extent))
    out['image_strides_exact'] = all(a.original_image.stride() == b.original_image.stride()
        for a, b in zip(ref_cams, cached.getTrainCameras()))
    md = 0.0
    for a, b in zip(ref_cams, cached.getTrainCameras()):
        md = max(md, float((a.original_image - b.original_image).abs().max()))
        md_w = float((a.world_view_transform - b.world_view_transform).abs().max())
        md_f = float((a.full_proj_transform - b.full_proj_transform).abs().max())
        md_c = float((a.camera_center - b.camera_center).abs().max())
        md_fov = max(abs(a.FoVx - b.FoVx), abs(a.FoVy - b.FoVy))
        out["images_maxdiff"] = max(out["images_maxdiff"] or 0.0, md)
        out["wvt_maxdiff"] = max(out["wvt_maxdiff"] or 0.0, md_w)
        out["fpt_maxdiff"] = max(out["fpt_maxdiff"] or 0.0, md_f)
        out["center_maxdiff"] = max(out["center_maxdiff"] or 0.0, md_c)
        out["fov_maxdiff"] = max(out["fov_maxdiff"] or 0.0, md_fov)
    # The v3 LUT, float64 scalar metadata and saved native RNG progression must
    # all agree exactly. Approximate pixels are diagnostic only, never a pass.
    transform_keys = ("wvt_maxdiff", "fpt_maxdiff", "center_maxdiff")
    out["transforms_bitwise"] = all((out[k] or 0.0) == 0.0 for k in transform_keys)
    out["pixels_bitwise"] = (out.get("images_maxdiff") or 0.0) == 0.0
    out["pixels_1ulp_or_better"] = (out.get("images_maxdiff") or 0.0) <= 1e-6
    out["identical"] = bool(out["transforms_bitwise"] and out["pixels_bitwise"]
                            and (out["fov_maxdiff"] or 0.0) == 0.0
                            and out["rng_exact"] and out["names_exact"] and out["extent_exact"]
                            and out['image_strides_exact'])
    out["seed"] = meta.get("seed", 20261015)
    out["cached_load_timings"] = cached.load_timings
    return out


def _bench(work, cache_dir, reps=3):
    meta = json.loads((Path(cache_dir) / "meta.json").read_text())
    _seed_all(meta.get("seed", 20261015))
    install_image_cache(work)
    ds, opt, pipe = _args_for(work, "/tmp/scene_cache_model")
    os.makedirs("/tmp/scene_cache_model", exist_ok=True)
    full_times, cache_times = [], []
    for i in range(reps):
        order = ["full", "cache"] if i % 2 == 0 else ["cache", "full"]
        for kind in order:
            if kind == "full":
                _seed_all(meta.get("seed", 20261015))
                random_state = None
                g = GaussianModel(3, "default")
                t0 = time.perf_counter()
                scene = Scene(ds, g, resolution_scales=[1.0])
                torch.cuda.synchronize()
                full_times.append(time.perf_counter() - t0)
                del scene, g
            else:
                t0 = time.perf_counter()
                _seed_all(meta.get("seed", 20261015))
                scene = load(work, cache_dir, want_test=False)
                torch.cuda.synchronize()
                cache_times.append(time.perf_counter() - t0)
                del scene
            torch.cuda.empty_cache()
    return dict(reps=reps,
                full_seconds=[round(x, 4) for x in full_times],
                cache_seconds=[round(x, 4) for x in cache_times],
                mean_full=round(float(np.mean(full_times)), 4),
                mean_cache=round(float(np.mean(cache_times)), 4),
                mean_saving=round(float(np.mean(full_times) - np.mean(cache_times)), 4))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--work", required=True); b.add_argument("--out", required=True); b.add_argument("--seed", type=int, default=20261015)
    v = sub.add_parser("verify"); v.add_argument("--work", required=True); v.add_argument("--cache", required=True)
    m = sub.add_parser("bench"); m.add_argument("--work", required=True); m.add_argument("--cache", required=True); m.add_argument("--reps", type=int, default=3)
    args = p.parse_args(argv)
    if args.cmd == "build":
        print(json.dumps(build(args.work, args.out, seed=args.seed), indent=2))
    elif args.cmd == "verify":
        print(json.dumps(_verify(args.work, args.cache), indent=2))
    else:
        print(json.dumps(_bench(args.work, args.cache, args.reps), indent=2))
    return 0


if __name__ == "__main__":
    main()
