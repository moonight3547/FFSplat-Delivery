# Depth regularization and translation-only pose refinement (module B)

Status: implemented, CPU-checked, **not yet run on a GPU**.  This note covers
the two remaining literature-grounded module-B ideas, their design, the exact
GPU commands, the expected effect under the ~100 s module-B budget and the
assumptions that still need GPU validation.  Nothing here changes
`research/train_sweep.py` or any upstream file.

| file | role |
|---|---|
| `research/depth_reg.py` | depth-regularized FastGS continuation; same knobs as `train_sweep.py`; `--pose-lr` enables translation refinement |
| `research/pose_refine.py` | translation table + exact world-shift wrapper; thin CLI that defaults `--pose-lr` on |
| `research/export_resplat_depth.py` | raw ReSplat depth dump -> per-view depth sidecar (CPU, numpy only) |

CPU checks (no CUDA, torch optional; the torch branch checks loss math):

```bash
python -m research.depth_reg --self-check
python -m research.pose_refine --self-check
python -m research.export_resplat_depth --self-check

# or in the Mac env that has CPU torch:
/opt/anaconda3/envs/ffsplat-enhance-macos/bin/python -m research.depth_reg --self-check
```

## 1. Why these two ideas

* **Predicted-depth supervision.** FSGS ("Fast and Sparse 3D Gaussian
  Splatting"), SparseGS and the DepthSplat line all regularize 3DGS geometry
  with depth instead of relying only on RGB.  ReSplat already predicts a depth
  map per context view (at 1/4 resolution internally; `run_resplat.py` sets
  `depth_pred_half_res=true`), so the depth is free at inference time.  The
  initialization dominates our score; a depth term is a direct way to stop
  densification from placing Gaussians along the wrong surface during the first
  few hundred steps, when the 16-view photometric signal is weakest.
* **Camera refinement.** COLMAP poses are frozen before ReSplat, but module B
  can still absorb a small pose bias.  Translation-only refinement (rotation
  and intrinsics frozen) has 3 parameters per view and cannot collapse the
  scene, so it is a low-risk knob when it is off by default and clamped.

## 2. Depth-regularized continuation

### 2.1 Sidecar and coverage

`research/export_resplat_depth.py` writes an `.npz` mapping an image key to a
`float32 [H, W]` depth at the module-B training resolution.  Keys are image
stems by default because FastGS stores `camera.image_name = Path(name).stem`;
full filenames are also resolved by the trainer.  Test/held-out views are
rejected by the exporter and by the trainer (leakage guard).

**Important coverage limit:** ReSplat only sees the 16 `split["context"]`
views, so the sidecar only covers 16 of the 56 training views.  Module B samples
all 56, so a depth-eligible step has the sidecar for roughly 29 % of draws.  The
trainer reports `depth.coverage` and `depth.missing_view_steps`; the other
draws stay photometric-only.  This is a property of the pipeline, not a bug.

### 2.2 Rendered depth (the renderer has no depth output)

The pinned `gaussian_renderer.render_fastgs` returns only
`{"render", "viewspace_points", "visibility_filter", "radii",
"accum_metric_counts"}`.  `research/depth_reg.py` first checks `render_pkg`
for a native `"depth"` (and optional `"alpha"`) and uses it if a future build
adds one -- assumed alpha-weighted camera-Z in sidecar units.  With the current
build it accumulates alpha-weighted camera-Z with **one extra rasterization**:

* a read-only Gaussian proxy swaps only `get_features_dc`; geometry, opacity,
  scale and rotation are the real tensors;
* SH DC encodes `(z / encode_max, 1, 0)` in RGB, so the standard
  `SH_C0 * dc + 0.5` affine produces `R = sum(w z)/encode_max`, `G = alpha`,
  `B = 0`; with a zero background the decoded depth is `R/G * encode_max`;
* `encode_max = max(1000, 20 * cameras_extent)` by default, so normal scene
  depths stay inside `[0, 1]` and are insensitive to a possible rasterizer
  clamp;
* the denominator alpha is detached (the value is still alpha-normalized) so
  the loss cannot cheaply reduce itself by only raising opacity.

This assumes the FastGS rasterizer uses the standard unclamped DC->RGB affine
and linear alpha compositing.  The first supervised step prints a
`[depth] render sanity ...` line (alpha coverage, depth min/max, median relative
error against the prediction).  If alpha is not ~`A` or the depth range is
degenerate, the encoding assumption is wrong on that build: stop and fall back
to `--depth-weight 0` until the rasterizer is inspected.  This is the largest
unverified assumption in this variant.

### 2.3 Loss

For valid pixels (prediction > 0, rendered depth > 0, alpha > `--depth-alpha-eps`):

```
data = mean( Charbonnier( (d_render - d_pred) / d_pred ) )
gx_p, gy_p = Sobel(d_pred)          # prediction is the edge reference
w_edge = exp(-beta * |grad d_pred| / median(d_pred))
grad = mean( w_edge * |grad d_render - grad d_pred| / median(d_pred) )
L_depth = data + grad_weight * grad
L = L_photometric + depth_weight * L_depth      (when the window is active)
```

* Relative error makes the term scale-free across scenes.
* The edge-aware gradient term does **not** force the rasterizer's discretized
  boundary to sit exactly on the (1/4-resolution, bilinearly upsampled)
  prediction; gradients are downweighted at predicted discontinuities.  This is
  the "edge-aware" part, following the DepthSplat-style gradient matching.
* Steps with fewer than `--depth-min-valid` valid pixels are skipped, not
  zeroed into the graph.
* Defaults: `--depth-weight 0.05`, `--depth-every 4`,
  `--depth-start-frac 0.05`, `--depth-end-frac 0.8`, `--depth-edge-beta 4.0`,
  `--depth-grad-weight 0.5`.

### 2.4 Producing the sidecar on the GPU host

ReSplat is **not** run by this repository's CPU checks.  The one-off command
below monkey-patches `run_inference` at runtime, asks it for
`visualization_dump["depth_fullres"]` for the 16 context views, writes
`depth_raw.npz` next to the normal ReSplat outputs and then executes the stock
`run_resplat.py` unchanged (no upstream edit).  Print the exact command with
the profile's resplat environment:

```bash
cd <repo root>            # e.g. /workspace/FFSplat-Hackathon
python -m research.export_resplat_depth --print-gpu-command \
  --work outputs/O0-20261030/work/hw/billboard \
  --resplat-out outputs/O0-20261030/results/hw/billboard/resplat \
  --conda-env ffsplat-enhance
```

which prints and can be pasted as:

```bash
conda run --no-capture-output -n ffsplat-enhance python - <<'PY'
import json, sys
import numpy as np
sys.path.insert(0, "competition/upstream/baseline")
import run_resplat

work = 'outputs/O0-20261030/work/hw/billboard'
out = 'outputs/O0-20261030/results/hw/billboard/resplat'
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
sys.argv = ["run_resplat.py", "--work", work, "--out", out]
run_resplat.main()
PY
```

This is the same ReSplat pass as the baseline (same checkpoint, `num_refine=2`,
same 16 views); it additionally writes `depth_raw.npz`.  If the resplat stage
was already run, this re-runs it once; alternatively supply any existing raw
depth dump to the converter.  The converter itself only needs numpy, so it can
run on the Mac after copying `depth_raw.npz`:

```bash
python -m research.export_resplat_depth \
  --raw outputs/O0-20261030/results/hw/billboard/resplat/depth_raw.npz \
  --split outputs/O0-20261030/work/hw/billboard/split.json \
  --out outputs/O0-20261030/results/hw/billboard/resplat/resplat_depth.npz
```

The sidecar is resized to `split["resolution"]` (the module-B training
resolution used by `-r 1`); `depth_reg` resizes again only if a coarse-to-fine
scale is active.  A `.meta.json` next to the sidecar records source, keys and
resolution.

### 2.5 Train with depth (fastgs env, ~100 s)

`organizer-fastgs` is the profile's fastgs environment; use the env name from
your own `competition/profiles/*.json` if it differs.

```bash
cd <repo root>
conda run --no-capture-output -n organizer-fastgs python -m research.depth_reg \
  --source outputs/O0-20261030/work/hw/billboard \
  --model_path outputs/experiments/depth/hw-billboard-d05/model \
  --eval -r 1 --sh_degree 3 --quiet \
  --init_ply outputs/O0-20261030/results/hw/billboard/resplat/resplat_world.ply \
  --iterations 12000 --limit_seconds 100 --limit_reserve 1.5 \
  --depth-npz outputs/O0-20261030/results/hw/billboard/resplat/resplat_depth.npz \
  --depth-weight 0.05 --depth-every 4 \
  --depth-start-frac 0.05 --depth-end-frac 0.8 \
  --record outputs/experiments/depth/hw-billboard-d05/train_record.json
```

A control run is the same command with `--depth-weight 0` (or without
`--depth-npz`); every other flag is identical to `train_sweep.py`, so the
comparison is seed- and schedule-matched.  `--depth-npz auto` looks for
`resplat_depth.npz` next to `--init_ply` and silently stays a control run when
the file does not exist.

Score it with the same official path used by `run_sweep.py`:

```bash
EXP=outputs/experiments/depth/hw-billboard-d05
ITER=$(python3 -c 'import json;print(json.load(open("'"$EXP"'/train_record.json"))["completed_iterations"])')
conda run --no-capture-output -n organizer-fastgs python -m research.render_eval \
  --source outputs/O0-20261030/work/hw/billboard --out "$EXP/eval" \
  --eval -r 1 --sh_degree 3 \
  --ply "$EXP/model/point_cloud/iteration_$ITER/point_cloud.ply" --name "iter$ITER"
conda run --no-capture-output -n ffsplat-enhance \
  python competition/upstream/baseline/evaluate.py \
  --model_path "$EXP/eval/iter$ITER" --device cuda:0
```

## 3. Translation-only camera refinement

### 3.1 Why the Gaussians move

The pinned rasterizer takes `viewmatrix`, `projmatrix` and `campos` inside
`GaussianRasterizationSettings`, which is not a tensor argument of the autograd
Function; there is no gradient to camera tensors.  A pure translation is exactly
equivalent to shifting the world:

```
camera at C + delta sees X   ==   camera at C sees X - delta
```

so `PoseRefiner.wrap(cam, gaussians)` returns a read-only view with
`get_xyz = xyz - delta[image_name]` and leaves the camera object untouched.
The rendered RGB, alpha and depth are identical to moving the camera, and
`means3D` **is** differentiated by the rasterizer, so both the photometric and
depth losses update `delta`.  This is why `depth_reg.py` can claim "same loss"
for the pose parameters.

Notes:

* `delta` is shared by all resolution variants of the same `image_name`; one
  3-vector per training view.
* Rotation and intrinsics are never optimized (3 DoF only).
* `--pose-max-shift` (default `0` = 2 % of `cameras_extent`) projects the
  translation back onto a ball after every Adam step, so a noisy photometric
  step cannot teleport a camera.
* FastGS's multi-view densification helper still renders the 10 sampled cameras
  with the unshifted model.  The clamp keeps this second-order inconsistency
  small; it is a known approximation.
* Test cameras are untouched; the PLY is the only artifact that carries the
  refinement, so held-out evaluation sees the original poses.
* When depth is enabled, the predicted depth map is kept in the original camera
  frame; the loss is pixelwise, so it is a **soft anchor** that resists pose
  drift (and is only approximately correct once the shift has parallax).  It is
  not a pose solver: the photometric loss drives the correction, the depth term
  keeps it from wandering.  With the ≤2 % extent clamp the parallax is small.

### 3.2 Window and defaults

`N` = `--pose-start-step` (or `round(--pose-start-frac * --iterations)`),
`M` = `--pose-steps` (`-1` = until the end).  Examples: `--pose-start-step 7000
--pose-steps 3000` refines iterations 7000..9999; the default
`--pose-start-frac 0.7 --pose-steps -1` refines the last 30 %.

With `--limit_seconds`, a wall-clock stop can happen before the window opens;
the trainer warns and records `completed_iterations`.  Lower `--pose-start-frac`
or use `--pose-start-step` after a first control run shows the real step count.

### 3.3 Commands (fastgs env)

Pure photometric pose refinement (no depth sidecar):

```bash
conda run --no-capture-output -n organizer-fastgs python -m research.depth_reg \
  --source outputs/O0-20261030/work/hw/billboard \
  --model_path outputs/experiments/pose/hw-billboard/model \
  --eval -r 1 --sh_degree 3 --quiet \
  --init_ply outputs/O0-20261030/results/hw/billboard/resplat/resplat_world.ply \
  --iterations 12000 --limit_seconds 100 \
  --pose-lr 1e-4 --pose-start-frac 0.75 --pose-max-shift 0.02 \
  --record outputs/experiments/pose/hw-billboard/train_record.json
```

Depth + pose (the wrapper `research.pose_refine` injects `--pose-lr 5e-5` if it
is omitted; keeping it explicit is clearer):

```bash
conda run --no-capture-output -n organizer-fastgs python -m research.pose_refine \
  --source outputs/O0-20261030/work/hw/billboard \
  --model_path outputs/experiments/pose-depth/hw-billboard/model \
  --eval -r 1 --sh_degree 3 --quiet \
  --init_ply outputs/O0-20261030/results/hw/billboard/resplat/resplat_world.ply \
  --iterations 12000 --limit_seconds 100 \
  --depth-npz outputs/O0-20261030/results/hw/billboard/resplat/resplat_depth.npz \
  --depth-weight 0.05 --pose-lr 1e-4 \
  --pose-start-frac 0.75 --pose-max-shift 0.02 \
  --record outputs/experiments/pose-depth/hw-billboard/train_record.json
```

Scoring is identical to §2.5 (`render_eval` + `evaluate.py`).

## 4. Expected effect under ~100 s

The budget is short, so the useful depth signal is the early/mid geometry, not
the final photometric polish:

* **Depth term cost.** One extra rasterization forward+backward every
  `--depth-every` iterations, inside `--depth-start-frac 0.05` ..
  `--depth-end-frac 0.8`, **and only when the sampled view is one of the 16
  context views**.  At 12 k steps, a 9 k-step window and `--depth-every 4`,
  that is ~2.25 k eligible draws × 16/56 ≈ 640 extra render/backward passes
  (~7 % of the window's step count; each such step costs roughly two
  rasterizations, so ~7 % extra rasterizer work overall).  With
  `--depth-every 1` every context-view draw is supervised (~29 % extra) —
  this is the knob to trade quality signal for wall clock.  FastGS step time is
  not purely rasterization, so the wall-clock overhead is smaller, but it is
  not free.
* **Expected gain.** The initial ReSplat Gaussians are unprojected *with* the
  same predicted depths, so they already lie close to the depth surfaces; the
  term mainly anchors newly densified Gaussians, penalizes floaters that
  photometric loss tolerates from the 16-view signal, and resists depth drift
  after the first few thousand steps.  The edge-aware gradient term avoids
  fighting true depth discontinuities, so thin structures should regress less
  than with a plain L1 depth loss.  Do not expect a large gain at iteration 0;
  the hypothesis is about the quality-vs-time curve under densification, and it
  must be measured against the `--depth-weight 0` control on the same 3 cases.
* **Pose cost.** Negligible: 3 parameters per view, no extra render.
* **Pose risk.** A translation-only fit can absorb a real COLMAP bias (good) or
  overfit the training views (bad for test PSNR).  The 2 % extent clamp and
  off-by-default guard are the mitigation; run it as a separate variant, not
  stacked, until both are measured.

Because only 16/56 training views have depth, the effective depth supervision is
≈29 % of draws × 75 % of the run.  Do not expect the full benefit reported by
papers that supervise every view.

## 5. What was not verified (no GPU here)

* The encoded-depth render on the actual FastGS rasterizer (DC->RGB affine,
  no clamp, alpha compositing, background handling).
* Whether the rasterizer's backward through `means3D` is stable for a shifted
  model (it should be identical to the unshifted render mathematically).
* Wall-clock overhead and step count under `--limit_seconds` on the target GPU.
* PSNR/SSIM/LPIPS effect of either variant; no score is claimed.
* The ReSplat depth dump command end to end (the resplat checkout is only
  present on the GPU host; the exporter is tested on synthetic raw dumps).
* Whether the 1/4-resolution predicted depth is accurate enough on reflective or
  thin geometry; the edge-aware term only softens the penalty, it does not fix
  a wrong prediction.
* Zero-valued "invalid" depth samples (if any) are exported as 0 and masked by
  the trainer, but bilinear resizing can bleed them into neighbours.  Export at
  the exact training resolution (the default when `--split` is given) to avoid
  resizing in the common `-r 1` case.

## 6. A/B protocol

1. Control: `train_sweep.py` (or `depth_reg.py` with `--depth-weight 0`), same
   seed `20261015`, same `--iterations`, same `--limit_seconds`.
2. Depth: `depth_reg.py` with `--depth-weight 0.05`, `--depth-every 4`.
3. Pose (separate): `depth_reg.py` with `--pose-lr 1e-4` and no depth; only
   stack with depth after both are individually measured.
4. Render every PLY with `research.render_eval.py` and score with the official
   `baseline/evaluate.py` (resplat env), exactly as `run_sweep.py` does.
5. Report B-only seconds from `train_record.json` (`scene_load_seconds +
   init_seconds + train_seconds`; `init_seconds` already includes
   `depth_load_seconds`) and the proxy /100 from `research/run_sweep.py`.
