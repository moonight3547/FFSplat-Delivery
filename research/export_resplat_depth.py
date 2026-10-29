"""Export ReSplat's per-view predicted depth into the module-B depth sidecar.

This helper never runs ReSplat itself.  It only *converts an existing depth
dump* into the sidecar consumed by ``research.depth_reg``:

    npz: <image stem> -> float32 [H, W] depth at the module-B training
         resolution (the same units as the ReSplat world PLY)

The raw depth stack is produced once on the GPU host by ``run_resplat.py`` with
``save_depth=True``.  The pinned ``run_resplat.py`` is not edited; the exact
command below monkey-patches ``run_inference`` at runtime, dumps
``depth_fullres`` for the 16 context views next to the normal ReSplat outputs
and then executes the stock entry point unchanged:

    cd <repo root>
    conda run --no-capture-output -n resplat python - <<'PY'
    import json, sys
    import numpy as np
    sys.path.insert(0, "competition/upstream/baseline")
    import run_resplat

    work = "outputs/<run>/work/<case>"
    out = "outputs/<run>/results/<case>/resplat"
    context = json.load(open(work + "/split.json", encoding="utf-8"))["context"]
    original = run_resplat.run_inference

    def capture(*args, **kwargs):
        kwargs["save_depth"] = True
        gaussians, rendered, rendered_depth, dump = original(*args, **kwargs)
        depth = dump["depth_fullres"][0].detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(out + "/depth_raw.npz", depth=depth,
                            names=np.asarray(context))
        print("[depth] dumped", depth.shape, "->", out + "/depth_raw.npz")
        return gaussians, rendered, rendered_depth, dump

    run_resplat.run_inference = capture
    sys.argv = ["run_resplat.py", "--work", work, "--out", out]
    run_resplat.main()
    PY

Use ``--print-gpu-command --work <work> --resplat-out <out>`` to print the same
command with the paths filled in (it is printed, never executed here).  Add
``--checkpoint <pth>`` to the printed script if the checkpoint is not the
default one.

Then convert the raw dump to the sidecar (CPU only; numpy is enough, so this can
run on the Mac after copying the npz):

    python -m research.export_resplat_depth \
        --raw outputs/<run>/results/<case>/resplat/depth_raw.npz \
        --split outputs/<run>/work/<case>/split.json \
        --out outputs/<run>/results/<case>/resplat/resplat_depth.npz

The sidecar keys are image stems by default because FastGS stores
``camera.image_name = Path(name).stem``; ``--key-style name`` keeps full
filenames instead.  ``research.depth_reg`` resolves both.  Held-out/test views
are rejected, never exported.

Accepted raw formats (all through ``np.load(..., allow_pickle=False)``):
  * ``npz`` with ``depth``/``depths``/``depth_preds``/``depth_fullres`` +
    optional ``names`` (unicode array);
  * ``npz`` whose keys already are image names and whose values are 2-D maps;
  * ``npy`` stack plus ``--names-from``.

``--self-check`` exercises the whole converter on synthetic arrays, including
key style, resizing, leakage rejection and the printed GPU command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STACK_KEYS = ("depth", "depths", "depth_preds", "depth_fullres", "rendered_depth")
META_SUFFIX = ".meta.json"

# The runtime monkey-patch above.  Kept as a template so the printed command and
# the docstring cannot drift apart.
GPU_TEMPLATE = """import json, sys
import numpy as np
sys.path.insert(0, "competition/upstream/baseline")
import run_resplat

work = {work!r}
out = {out!r}
context = json.load(open(work + "/split.json", encoding="utf-8"))["context"]
original = run_resplat.run_inference

def capture(*args, **kwargs):
    kwargs["save_depth"] = True
    gaussians, rendered, rendered_depth, dump = original(*args, **kwargs)
    depth = dump["depth_fullres"][0].detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(out + "/depth_raw.npz", depth=depth, names=np.asarray(context))
    print("[depth] dumped", depth.shape, "->", out + "/depth_raw.npz")
    return gaussians, rendered, rendered_depth, dump

run_resplat.run_inference = capture
sys.argv = ["run_resplat.py", "--work", work, "--out", out{checkpoint}]
run_resplat.main()
"""


def gpu_depth_command(work, out, checkpoint=None, conda_env="resplat"):
    """Return the exact GPU-host command that dumps ReSplat predicted depth."""
    checkpoint_arg = ""
    if checkpoint:
        checkpoint_arg = f", \"--checkpoint\", {str(checkpoint)!r}"
    body = GPU_TEMPLATE.format(work=str(work), out=str(out),
                               checkpoint=checkpoint_arg)
    return (f"conda run --no-capture-output -n {conda_env} python - <<'PY'\n"
            + body + "PY")


def _bilinear_resize(array, out_h, out_w):
    """Bilinear resize of a 2-D float map (align_corners=False semantics)."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D depth map, got shape {array.shape}")
    h, w = array.shape
    if (h, w) == (out_h, out_w):
        return np.ascontiguousarray(array, dtype=np.float32)
    if h == 0 or w == 0 or out_h <= 0 or out_w <= 0:
        raise ValueError("empty depth map or invalid target size")
    source = array.astype(np.float64)
    ys = (np.arange(out_h, dtype=np.float64) + 0.5) * (h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float64) + 0.5) * (w / out_w) - 0.5
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    wy = np.clip(ys - np.floor(ys), 0.0, 1.0)[:, None]
    wx = np.clip(xs - np.floor(xs), 0.0, 1.0)[None, :]
    top = source[y0][:, x0] * (1.0 - wx) + source[y0][:, x1] * wx
    bottom = source[y1][:, x0] * (1.0 - wx) + source[y1][:, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def _squeeze_stack(array):
    """Normalise a depth stack to [V, H, W] without reordering views."""
    array = np.asarray(array, dtype=np.float32)
    while array.ndim > 3:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[1] == 1:
            array = array[:, 0]
        elif array.shape[-1] == 1:
            array = array[..., 0]
        else:
            raise ValueError(f"cannot interpret depth stack shape {array.shape}")
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"expected a [V, H, W] depth stack, got {array.shape}")
    return array


def _read_names_file(path, key):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [str(x) for x in value]
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON list or object")
    if key not in value:
        raise ValueError(f"{path}: no {key!r} key (available: {sorted(value)})")
    return [str(x) for x in value[key]]


def _names_from_npz(value):
    names = np.asarray(value)
    if names.ndim != 1:
        names = names.reshape(-1)
    if names.dtype.kind == "S":
        return [x.decode("utf-8") for x in names]
    return [str(x) for x in names]


def _name_matches(name, wanted):
    """True when a raw name matches a split entry by basename or stem."""
    name = Path(str(name))
    if name.name in wanted:
        return True
    return name.stem in {Path(w).stem for w in wanted}


def load_raw_depths(raw, names_from=None, names_key="context"):
    """Load a raw ReSplat depth dump -> (names, [V, H, W] float32 stack)."""
    raw = Path(raw)
    if not raw.exists():
        raise FileNotFoundError(
            f"raw depth dump not found: {raw}\n"
            "Run the GPU command printed by "
            "`python -m research.export_resplat_depth --print-gpu-command "
            "--work <work> --resplat-out <resplat out>` first.")
    embedded_names = None
    if raw.suffix == ".npy":
        stack = _squeeze_stack(np.load(raw, allow_pickle=False))
    else:
        with np.load(raw, allow_pickle=False) as data:
            keys = list(data.files)
            stack_key = next((k for k in STACK_KEYS if k in keys), None)
            if stack_key is not None:
                stack = _squeeze_stack(np.asarray(data[stack_key]))
                if "names" in keys:
                    embedded_names = _names_from_npz(np.asarray(data["names"]))
            else:
                mapping = {}
                for key in keys:
                    value = np.asarray(data[key])
                    if value.ndim != 2:
                        raise ValueError(
                            f"{raw}: key {key!r} has shape {value.shape}, expected "
                            "a 2-D depth map or a 'depth' stack")
                    mapping[key] = value.astype(np.float32, copy=False)
                if not mapping:
                    raise ValueError(f"{raw}: no depth arrays found")
                stack = np.stack([mapping[k] for k in mapping], axis=0)
                embedded_names = list(mapping)

    if not np.isfinite(stack).all():
        bad = int((~np.isfinite(stack)).sum())
        raise ValueError(f"{raw}: {bad} non-finite depth samples; refuse to export")

    names = None
    if names_from is not None:
        names = _read_names_file(names_from, names_key)
        if len(names) != stack.shape[0]:
            raise ValueError(
                f"{names_from}: {len(names)} names for {stack.shape[0]} depth maps")
    elif embedded_names is not None:
        names = embedded_names
    else:
        raise ValueError(
            f"{raw}: no names embedded in the archive; pass --names-from "
            "(a split.json or a JSON list)")

    if len(names) != stack.shape[0]:
        raise ValueError(f"{len(names)} names for {stack.shape[0]} depth maps")
    if any(not str(n) for n in names):
        raise ValueError("empty image name in depth dump")
    return [str(n) for n in names], stack


def _dedupe_names(names):
    seen = set()
    output = []
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate image name in depth dump: {name!r}")
        seen.add(name)
        output.append(name)
    return output


def convert(raw, out, split=None, names_from=None, names_key="context",
            key_style="stem", resolution=None, keep="all"):
    """Convert a raw depth dump into the depth sidecar.  Returns metadata."""
    raw_names, stack = load_raw_depths(raw, names_from=names_from, names_key=names_key)
    raw_names = _dedupe_names(raw_names)
    invalid = int((stack <= 0).sum())
    if invalid:
        positive = float((stack > 0).mean())
        if positive < 0.5:
            raise ValueError(
                f"{raw}: only {positive * 100:.1f}% of depth samples are positive; "
                "refusing to export (wrong dtype/layout or a corrupt dump)")
        print(f"[export] warning: {invalid} non-positive depth samples set to 0 "
              "(masked by the trainer)")
        stack = np.where(stack > 0, stack, 0.0)

    split_data = None
    if split is not None:
        split_data = json.loads(Path(split).read_text(encoding="utf-8"))
        test = {str(n) for n in split_data.get("test", [])}
        test_stems = {Path(n).stem for n in test}
        leaked = sorted({Path(n).name for n in raw_names
                         if Path(n).name in test or Path(n).stem in test_stems})
        if leaked:
            raise ValueError(
                "depth dump contains held-out/test views: "
                f"{leaked[:5]}; refusing to export")
        if keep == "context":
            wanted = {str(n) for n in split_data.get("context", [])}
            index = [i for i, n in enumerate(raw_names) if _name_matches(n, wanted)]
        elif keep == "train":
            wanted = {str(n) for n in split_data.get("train", [])}
            index = [i for i, n in enumerate(raw_names) if _name_matches(n, wanted)]
        elif keep == "all":
            index = list(range(len(raw_names)))
        else:
            raise ValueError(f"unknown --keep {keep!r}")
        if not index:
            raise ValueError(
                f"none of the {len(raw_names)} raw depth views are in the "
                f"{keep!r} split")
        raw_names = [raw_names[i] for i in index]
        stack = stack[index]

    target_hw = None
    if resolution is not None:
        target_hw = (int(resolution[0]), int(resolution[1]))
    elif split_data is not None and split_data.get("resolution"):
        target_hw = (int(split_data["resolution"][0]),
                     int(split_data["resolution"][1]))

    keys = []
    for name in raw_names:
        key = Path(name).stem if key_style == "stem" else Path(name).name
        keys.append(key)
    if len(set(keys)) != len(keys):
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(
            f"key style {key_style!r} is ambiguous for {duplicates[:5]}; "
            "use --key-style name")

    arrays = {}
    heights = set()
    for key, name, depth in zip(keys, raw_names, stack):
        if target_hw is not None:
            depth = _bilinear_resize(depth, target_hw[0], target_hw[1])
        else:
            depth = np.ascontiguousarray(depth, dtype=np.float32)
        arrays[key] = depth
        heights.add(depth.shape)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)

    meta = dict(raw=str(raw), split=str(split) if split else None,
                out=str(out), count=len(arrays), key_style=key_style,
                invalid_samples=invalid,
                keep=keep, keys=sorted(arrays),
                source_names=raw_names,
                resolution=[int(x) for x in next(iter(heights))] if heights else None,
                resized=target_hw is not None)
    meta_path = out.parent / (out.stem + META_SUFFIX)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def _expect_error(call, label, checks):
    try:
        call()
    except (ValueError, FileNotFoundError):
        return checks + 1
    raise AssertionError(f"self-check: expected an error for {label}")


def self_check():
    import tempfile

    checks = 0

    def check(condition, message):
        nonlocal checks
        if not condition:
            raise AssertionError(f"self-check: {message}")
        checks += 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        names = [f"{i:04d}.png" for i in range(24)]
        train = [n for i, n in enumerate(names) if i % 8]
        split = dict(images=names, train=train, test=names[::8],
                     context=train[:16], llffhold=8, resolution=[32, 48])
        (tmp / "split.json").write_text(json.dumps(split), encoding="utf-8")

        rng = np.random.default_rng(11)
        depth = np.exp(rng.normal(0.0, 0.3, size=(16, 16, 24))).astype(np.float32)

        # stack + unicode names, 1/4-resolution style [V, H, W, 1, 1]
        raw = tmp / "depth_raw.npz"
        np.savez_compressed(raw, depth=depth[None, ..., None, None],
                            names=np.asarray(split["context"]))
        loaded_names, loaded = load_raw_depths(raw)
        check(loaded.shape == (16, 16, 24), "squeeze depth_fullres stack")
        check(loaded_names == split["context"], "embedded names")
        np.testing.assert_allclose(loaded, depth)
        checks += 1

        sidecar = tmp / "resplat_depth.npz"
        meta = convert(raw, sidecar, split=tmp / "split.json")
        check(meta["count"] == 16 and meta["resolution"] == [32, 48],
              "sidecar resized to split resolution")
        check(meta["key_style"] == "stem", "default key style is stem")
        with np.load(sidecar, allow_pickle=False) as data:
            keys = set(data.files)
            check(keys == {Path(n).stem for n in split["context"]}, "stem keys")
            sample = data[Path(split["context"][0]).stem]
            check(sample.dtype == np.float32 and sample.shape == (32, 48),
                  "sidecar dtype/shape")
            check(np.isfinite(sample).all() and (sample > 0).all(),
                  "sidecar finite positive")
        checks += 1

        # flat name -> 2-D mapping
        flat = tmp / "flat.npz"
        np.savez_compressed(flat, **{n: depth[i] for i, n in enumerate(split["context"])})
        names2, stack2 = load_raw_depths(flat)
        check(names2 == split["context"] and stack2.shape == depth.shape,
              "flat mapping round-trip")
        sidecar2 = tmp / "flat_sidecar.npz"
        meta2 = convert(flat, sidecar2, split=tmp / "split.json", key_style="name")
        check(meta2["key_style"] == "name" and len(meta2["keys"]) == 16,
              "full-name key style")

        # resizing a constant map preserves the value
        resized = _bilinear_resize(np.full((5, 7), 2.5, dtype=np.float32), 11, 13)
        check(resized.shape == (11, 13) and np.allclose(resized, 2.5),
              "bilinear resize constant")

        # explicit resolution override and no split
        sidecar3 = tmp / "no_split.npz"
        meta3 = convert(flat, sidecar3, resolution=(8, 10))
        check(meta3["resolution"] == [8, 10], "--resolution override")

        # keep=context filters a dump that also contains other train views
        allraw = tmp / "all.npz"
        np.savez_compressed(allraw,
                            depth=np.repeat(depth[:1], len(train), axis=0),
                            names=np.asarray(train))
        sidecar_ctx = tmp / "context_only.npz"
        meta_ctx = convert(allraw, sidecar_ctx, split=tmp / "split.json",
                           keep="context")
        check(meta_ctx["count"] == 16, "--keep context selects the 16 context views")
        with np.load(sidecar_ctx, allow_pickle=False) as data:
            check(set(data.files) == {Path(n).stem for n in split["context"]},
                  "context-only sidecar keys")

        # names-from takes precedence
        names_json = tmp / "names.json"
        names_json.write_text(json.dumps({"context": split["context"]}),
                              encoding="utf-8")
        names3, _ = load_raw_depths(tmp / "depth_raw.npz", names_from=names_json)
        check(names3 == split["context"], "--names-from resolution")

        # test-view leakage is rejected
        leaked = tmp / "leaked.npz"
        np.savez_compressed(leaked, depth=depth[:1], names=np.asarray([split["test"][0]]))
        checks = _expect_error(
            lambda: convert(leaked, tmp / "x.npz", split=tmp / "split.json"),
            "test-view leakage", checks)

        # duplicate stems are rejected
        dup = tmp / "dup.npz"
        np.savez_compressed(dup, depth=np.stack([depth[0], depth[1]]),
                            names=np.asarray(["a/x.png", "b/x.png"]))
        checks = _expect_error(
            lambda: convert(dup, tmp / "dup_sidecar.npz"), "duplicate stems", checks)

        # missing raw dump
        checks = _expect_error(
            lambda: load_raw_depths(tmp / "does_not_exist.npz"), "missing raw", checks)

        # missing names
        nonames = tmp / "nonames.npz"
        np.savez_compressed(nonames, depth=depth)
        checks = _expect_error(
            lambda: load_raw_depths(nonames), "missing names", checks)

        command = gpu_depth_command("outputs/run/work/case", "outputs/run/resplat")
        check("save_depth" in command and "depth_fullres" in command,
              "GPU command requests depth")
        check("outputs/run/work/case" in command and "run_resplat.main()" in command,
              "GPU command paths")

    print(f"self-check OK: {checks} checks "
          "(raw formats, squeeze, resize, keys, leakage, GPU command)")
    return checks


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=None,
                        help="existing raw ReSplat depth dump (.npz/.npy)")
    parser.add_argument("--out", type=Path, default=None, help="sidecar npz to write")
    parser.add_argument("--split", type=Path, default=None,
                        help="module-A split.json (leakage guard + resolution)")
    parser.add_argument("--names-from", type=Path, default=None,
                        help="JSON split/list giving the depth stack order")
    parser.add_argument("--names-key", default="context",
                        help="key to read from --names-from (default context)")
    parser.add_argument("--key-style", choices=["stem", "name"], default="stem",
                        help="sidecar keys: FastGS image_name stems (default) or filenames")
    parser.add_argument("--resolution", type=int, nargs=2, default=None,
                        metavar=("H", "W"), help="sidecar resolution (default split)")
    parser.add_argument("--keep", choices=["all", "context", "train"], default="all",
                        help="which non-test views to export")
    parser.add_argument("--print-gpu-command", action="store_true",
                        help="print (do not run) the GPU depth-dump command")
    parser.add_argument("--work", type=Path, default=None,
                        help="module-A work dir for --print-gpu-command")
    parser.add_argument("--resplat-out", type=Path, default=None,
                        help="ReSplat output dir for --print-gpu-command")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="optional ReSplat checkpoint for --print-gpu-command")
    parser.add_argument("--conda-env", default="resplat",
                        help="conda env used by --print-gpu-command (profile: ffsplat-enhance)")
    parser.add_argument("--self-check", action="store_true",
                        help="CPU-only converter unit checks")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    if args.print_gpu_command:
        if args.work is None or args.resplat_out is None:
            parser.error("--print-gpu-command needs --work and --resplat-out")
        print(gpu_depth_command(args.work, args.resplat_out, args.checkpoint,
                                args.conda_env))
        return 0
    if args.raw is None or args.out is None:
        parser.error("--raw and --out are required (or use --self-check / "
                     "--print-gpu-command)")
    try:
        meta = convert(args.raw, args.out, split=args.split,
                       names_from=args.names_from, names_key=args.names_key,
                       key_style=args.key_style, resolution=args.resolution,
                       keep=args.keep)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
