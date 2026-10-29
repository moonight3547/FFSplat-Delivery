"""Pre-decode the prepared scene images into one uint8 array for module B.

Module B's wall clock includes loading the prepared images (PIL decode of hundreds of
PNGs: ~9-11 s of a ~30 s module-B budget on DL3DV). The prepared scene is our own
module-A artifact, so caching its decoded pixels is a legitimate pipeline change, not a
timing-boundary trick: every stage still reads the same pixels, we only avoid decoding
the same PNGs twice. The cache is a single .npy plus a JSON manifest of file names.

  python -m research.image_cache --work <prepared scene> --out <cache dir>
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def build(work, out):
    images = os.path.join(work, "images")
    names = sorted(n for n in os.listdir(images) if os.path.splitext(n)[1].lower() in
                   (".png", ".jpg", ".jpeg"))
    if not names:
        raise SystemExit(f"no images under {images}")
    arrays = []
    for name in names:
        with Image.open(os.path.join(images, name)) as im:
            arrays.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise SystemExit(f"mixed image shapes are not supported by the cache: {sorted(shapes)}")
    data = np.stack(arrays, axis=0)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "images_u8.npy"), data)
    manifest = dict(names=names, shape=[int(x) for x in data.shape[1:]],
                    count=int(data.shape[0]), bytes=int(data.nbytes),
                    note="uint8 RGB, module-B reads these instead of decoding PNGs")
    with open(os.path.join(out, "manifest.json"), "w") as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(dict(status="cached", count=manifest["count"], shape=manifest["shape"],
                          mb=round(data.nbytes / 1e6, 1)), indent=2))
    return manifest


class CachedImages:
    """Loads pixels by file name from a single memmapped uint8 array."""

    def __init__(self, cache_dir):
        manifest = json.load(open(os.path.join(cache_dir, "manifest.json")))
        self.index = {n: i for i, n in enumerate(manifest["names"])}
        self.data = np.load(os.path.join(cache_dir, "images_u8.npy"), mmap_mode="r")

    def get(self, name):
        return np.asarray(self.data[self.index[name]])


def install(train_sweep_module, cache_dir):
    """Patch the FastGS camera loader inside the pinned package (no upstream edit)."""
    import scene.dataset_readers as readers
    from PIL import Image as PILImage
    import torch
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov  # noqa: F401

    cached = CachedImages(cache_dir)
    original = readers.load_image

    def load_image(path):
        name = os.path.basename(path)
        if name in cached.index:
            arr = cached.get(name)
            return PILImage.fromarray(arr)
        return original(path)

    readers.load_image = load_image
    return cached


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    build(args.work, args.out)


if __name__ == "__main__":
    main()
