# Module A: 16-view context selection (round 4)

`research/select_views.py` replaces the organizer's filename-uniform context
with a geometry-driven choice while keeping the exact same ReSplat contract:
16 images, all from `split.json["train"]`, no held-out RGB, no network change.
It reads only `split.json` and `sparse/0/{cameras,images}.txt`, is pure
numpy/CPU, and never writes into the prepared scene dir.

**COLMAP convention.** `images.txt` rows are world-to-camera:
`X_cam = R(qvec) X_world + tvec`. The parser therefore uses
`camera_center = -R(qvec)^T tvec` and `R_c2w = R(qvec)^T`. `--self-test`
writes a synthetic cameras/images pair, round-trips those poses, and checks a
world point through both transforms before testing selection determinism.

## Policies

`--policy uniform` is the organizer baseline: 16 indices from
`np.linspace(0, len(train)-1, 16, dtype=int)` in sorted train order. It is kept
only for paired comparison.

`--policy fps` normalises camera centres by the scene scale
`s = median_{i<j} ||C_i - C_j||`, starts at the camera closest to the train
centroid, then repeatedly adds the candidate with the largest distance to its
nearest already-selected centre. Ties go to the lower train index. This is a
deterministic maximin-spacing baseline that ignores orientation.

`--policy coverage` greedily maximises weighted view coverage of a scene proxy.
The proxy is the camera-centre bounding box, inflated by `0.5 s` on every axis
and shifted by `0.5 s` along the mean of the unit optical axes (so
forward-facing rigs include the space in front; a 360 ring has near-zero
shift). The box is
voxelised with 32 cells on its longest side. A voxel is covered by a view when
its centre projects, with `z > 0`, into the **central 50% x 50% of that view's
image** (25% of the area, distortion ignored); the crop prevents one wide-FOV
view from claiming the whole proxy. Each voxel is weighted
`1 / sqrt(frequency)`, where `frequency` is the number of candidate views that
can see it, and each greedy step picks the view with the largest total weight
of newly covered voxels. Ties go to the lower train index. The central crop is
a proxy for the high-confidence part of a view; it is a fixed rule, not a
per-scene parameter.

The emitted `context` is always sorted in train order, exactly like the
baseline, so ReSplat's pivot (`context[8]`) keeps the same convention across
policies. The greedy order is recorded in `scores.selection_order`; per-view
marginal scores are in `scores.views`.

`<out>/context16.json` is exactly
`{"policy": str, "context": [16 names], "scores": {...}}`. `scores`
contains `scene_scale`, the policy `objective`, `min_separation`,
`mean_nearest_separation`, `selection_order`, per-view rows, and (for
coverage) the full grid/coverage record. All floats are rounded to 12
decimals and all ties break on the lower train index.

## Why this should help the feed-forward init

ReSplat sees the 16 context views both as inputs and as self-supervised
refinement targets, so near-duplicate views spend capacity on information the
network already has while under-covered parts of the scene produce weak or
missing Gaussians. FPS directly increases baseline/parallax and avoids the
filename-order clustering that `np.linspace` can inherit from capture order.
Coverage adds the actual frustum/visibility footprint and weights rare proxy
voxels more heavily, so the chosen set spans what the scene exposes instead of
re-covering the same central volume. Neither policy looks at RGB, test views,
or per-scene labels, and both use the frozen train split unchanged.

## CPU checks

From the repository root (no CUDA, no torch import):

```bash
conda run --no-capture-output -n ffsplat-enhance \
  python -m research.select_views --self-test

conda run --no-capture-output -n ffsplat-enhance \
  python -m research.select_views --work outputs/O0-20261030/work/hw/billboard \
  --out /tmp/context_probe --policy coverage --dry-run
```

Expected self-test tail: `[self-test] PASS all 7 checks`. `--dry-run` prints
the one-line summary plus the JSON record and writes nothing.

## GPU commands

Run from the repository root on the GPU host. The selector stage is CPU-only;
the ReSplat/scoring stage is gated by `RUN_GPU=1` and is **not** run by the
plain command.

```bash
cd /workspace/FFSplat-Hackathon   # use this checkout's actual path
export PYTHONPATH=$PWD

# optional: only if the prepared work dir does not exist yet
conda run --no-capture-output -n resplat \
  python competition/upstream/baseline/module_a/preprocess.py \
  --scene_dir /workspace/organizer-assets/datasets/hw/billboard --group hw \
  --out outputs/O0-20261030/work/hw/billboard \
  --num_context 16 --llffhold 8 --max_resolution 960

# selector only: outputs/experiments/round4/<case>/<policy>/context16.json
bash research/experiments/round4_a_selection.sh hw/billboard
bash research/experiments/round4_a_selection.sh mipnerf360/counter

# stage a copy, run ReSplat num_refine=2, score the init (B=0)
RUN_GPU=1 bash research/experiments/round4_a_selection.sh hw/billboard
RUN_GPU=1 bash research/experiments/round4_a_selection.sh mipnerf360/counter
```

The script defaults to `WORK=outputs/O0-20261030/work/$CASE` and
`OUT=outputs/experiments/round4/$CASE`; override with `WORK=... OUT=...`.
Each policy gets its own full work-dir copy under `$OUT/<policy>/work`, so
budget roughly three prepared-work-dir copies of disk.
`RUN_GPU=1` copies `$WORK` to `$OUT/<policy>/work` (never modifying `$WORK`),
rewrites only `split.json["context"]`, then runs:

```bash
conda run --no-capture-output -n ffsplat-enhance python -m research.resplat_variant \
  --work "$OUT/<policy>/work" --out "$OUT/<policy>/resplat" \
  --checkpoint /workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth \
  --num_refine 2 --device cuda:0

conda run --no-capture-output -n ffsplat-enhance python -m research.score_ply \
  --work "$OUT/<policy>/work" --out "$OUT/<policy>/score-init" \
  --ply "$OUT/<policy>/resplat/resplat_world.ply" --name "<policy>"
```

`score-init` is the official PSNR/SSIM/VGG-LPIPS on the held-out views with
B=0, i.e. the initialization quality that module B has to repair. If a B
continuation is wanted after comparing init quality, use `research.run_sweep`
with the same `--work` and `--init-ply` and the case budget from
`records/fastgs-random30k-calibration.json`; that is outside this round.

## Not verified without the GPU

Only the CPU parser, self-test, dry-run, and offline shell staging were run
here. Actual selection quality on hw/billboard and mipnerf360/counter, ReSplat
memory/time, and whether better spread improves the init are open until the GPU
commands above are run. Coverage uses geometry only (no image content) and
cannot repair wrong poses or a bad COLMAP reconstruction.
