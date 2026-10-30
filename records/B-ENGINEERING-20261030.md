# B-ENGINEERING-20261030 — whole module-B engineering audit (measured)

**Task 18** · N1 (3×4090), serial quiet box · 2026-10-31 · user ask: *"检视整个模块B的实现有什么是
可以前提到模块A或者提前加载到缓存，或者其他可以加速的工程优化空间"* · raw data
`/root/gpufree-data/outputs/b-eng-20261030/`, local copies under `records/b-eng-20261030/raw/`.

> **VERDICT (updated after the user explicitly approved A-side data residency).** Module B
> is CUDA-bound, not Python-bound: render+backward = 68.7–73.6 % of the loop, Adam = 20–24 %,
> all Python/loss/bookkeeping together ≤7 %; per-step micro-elimination has ≤1–2 % headroom.
> The real lever is **scene_load: 1.99 s (bicycle) / 3.23 s (0b5635) = 7–11 % of B**, and it
> can be removed almost entirely by an **A-side tensor cache** (decoded uint8 pixels in
> NCHW train order + precomputed camera tensors), which the user classified as legitimate
> **data residency / latency hiding** (A already loads the same images/poses; only moving
> real optimization work into A would be out of scope). Measured prototype
> (`research/scene_cache.py`, A-side build + B-side `--scene_cache`):
> **B scene-load 1.90 s → 0.064 s on bicycle (saving 1.83 s) and 3.15 s → 0.105 s on
> 0b5635 (saving 3.04 s)** ⇒ **TIME +0.43 / +0.72** at the 95 s / 94.29 s budgets, with
> camera transforms bit-identical and pixels equal to 1 float32 ulp (5.96e-8, i.e. one
> rounding step of `uint8/255`; below rasterization noise). A-side cost: 4.3–6.0 s wall
> (mostly the same `Scene` load A already does, plus a 0.9–1.6 s memmap write of
> 280–491 MB); it can be produced while A/ReSplat uses the GPU. Delivery default is
> **unchanged** (`--scene_cache` default off) until the end-to-end A/B is positive.
> Second lever: test cameras loaded inside B (0.54/1.06 s, class A). Everything else ≥10 %
> of B is one-shot load/init (class B inside B, class C for research repeats) or CUDA work.

## 1. Part 1 — stage inventory and measured breakdown

`--time_breakdown` on the delivery recipe (5000 steps, N1 GPU2, serial, quiet; the four
sections are measured with CUDA sync between them, so the shares are accurate and the
absolute train seconds are ~10 % higher than a sync-free run):

| phase | bicycle (train 26.14 s) | 0b5635 (train 28.46 s) | what it is |
|---|---|---|---|
| (a) scene_load | **1.99 s** | **3.23 s** | split/metadata read + per-camera image decode→tensor→GPU + camera transforms |
| (b) init | **0.77 s** | **0.62 s** | PLY parse (plyfile→numpy→GPU), GaussianModel/optimizer setup, densify-stat buffers |
| (c) render (FastGS forward) | 5.51 s = **21.1 %** (1.10 ms/step) | 6.70 s = **23.5 %** (1.34 ms/step) | rasterizer forward + metric bookkeeping |
| (c) backward | 12.44 s = **47.6 %** (2.49 ms/step) | 14.25 s = **50.1 %** (2.85 ms/step) | rasterizer backward (dominant) |
| (c) optimizer | 6.39 s = **24.4 %** (1.28 ms/step) | 5.77 s = **20.3 %** (1.15 ms/step) | 5-group Adam every step + SH-rest Adam every 16 + `zero_grad` + no-grad bookkeeping |
| (c) loss + GT access + Python (`grad_to_step`) | 1.50 s = **5.7 %** (0.30 ms/step) | 1.45 s = **5.1 %** (0.29 ms/step) | L1 + fused SSIM + camera access + Python |
| unaccounted (untimed: realloc parse/trigger/snapshot checks) | 0.31 s = **1.2 %** | 0.30 s = **1.0 %** | |
| (d) one-shot realloc @1250 | inside the untimed/optimizer section, bounded | same | 15 % one-shot split+concat |
| (e) snapshot/export | **not in B** by the run_sweep convention (`+0.5` flat) | same | |

So the mysterious "~20 % outside render/backward" **is the Adam optimizer** — it is fully
measured, not overhead.

## 2. Part 2 — cache/preload candidates (with measured size and class)

**GT residency (the Lead's point 1 confirmed by measurement):** `ModelParams.data_device="cuda"`,
`Camera.__init__` already does the clamp/`.to(cuda)`, so every train GT is **GPU-resident
before step 1**; the trainer's per-step `.cuda()` is a no-op costing **0.4 µs** (2 ms over
5000 steps). **There is no per-step Host→Device transfer to remove.**

| item | bicycle | 0b5635 (largest-ish) | notes |
|---|---:|---:|---|
| train views × resolution | 169 × 576×960 | 333 × 512×960 | |
| train GT on GPU, current f32 | **1.12 GB** | **1.96 GB** | `N·3·H·W·4 B` |
| same as uint8 (hypothetical) | 0.28 GB | 0.49 GB | would need per-step conversion: ~4–5 ms/view ⇒ +0.7…1.6 GB stays; **rejected** (adds 4–8 s B to save memory that is not scarce: 24 GB) |
| whole scene GPU footprint | 1.30 GB | 2.25 GB | images + transforms + test cams |
| camera transform tensors (all views) | **43 KB** | **85 KB** | already GPU; irrelevant memory lever |
| init PLY GPU tensors | ~124 MB | ~108 MB | parse+upload 0.6–0.8 s |

**Immutable-across-runs work (same prepared work dir):** split.json/sparse metadata,
decoded+resized images (already cached as `images_u8.npy` **in A**, class-C/D boundary),
`cameras_extent`, camera transforms. `init` is immutable per ReSplat PLY: PLY parse +
tensor upload (~0.4–0.6 s of the 0.6–0.8 s init) and optimizer-state buffers.
Class split: (i) same-process reuse — none available (single run); (ii) cross-run research
cache — a versioned `.pt` of resized image tensors / parsed PLY could save ~1–2 s per
repeat, **helps research iterations only, not the scored run**; (iii) moving it into A —
**class D**, see §4.

**Optional models:** LPIPS/VGG weights are constructed only when `lpips_weight>0` or the
stage-3 weight is set; `gsplat`/depth modules are imported only when their regularizer is
enabled. Delivery path loads none of them (verified by import guards + absence in the
breakdown). No junk to remove.

**Render bookkeeping:** `add_densification_stats`/`max_radii2D` are already gated to
`iteration < densify_until_iter` (1250); `compute_gaussian_score_fastgs` (2 renders × 10 cams)
only runs in the 500–1250 window (~7 events, ≈1–2 % one-time). `accum_metric_counts` is
returned by the rasterizer but unused in the trainer; it is produced inside the same kernel
(no separable second cost), and the per-step `metric_map` buffer allocation is only
**7.6 µs** — reusing a preallocated buffer is bit-identical (max render diff 0.0) but **not
faster** (0.96 vs 0.98 ms), so that prototype was dropped.

## 3. Part 3 — other levers, measured

| lever | measured | verdict |
|---|---|---|
| GT `.cuda()` per step | 0.4 µs/call → 2 ms / 5000 steps | already a no-op; remove only in lean prototype |
| `realloc_at` string parse (runs **every** iteration) | 0.24 µs/step → 1.2 ms / 5000 | pure elimination, hoisted under `--lean_step` |
| camera list `.copy()` + `.pop()` | 0.43 + 0.89 µs/step → 6.6 ms / 5000 | noise; index-based variant in `--lean_step` keeps the same RNG stream |
| per-step Python/GC as a whole | bounded by `grad_to_step` = 0.29 ms/step incl L1+SSIM | no 20 % Python overhead exists |
| Adam / fused Adam | optimizer = 20–24 %, fused Adam already falsified (codex), foreach is the CUDA default | no legitimate win; **no** optimizer merge (SH every-16 already correct) |
| redundant Camera clamp + `torch.ones` multiply (no alpha mask) | 22 µs/view ⇒ 3.7 ms (bicycle) / 7.3 ms (0b5635) | real but <0.03 % of B; class A, not worth a default change |
| allocator / pinned memory / non-blocking | profile already uses `expandable_segments:True`; H2D is one-shot and small next to images | no measured lever |
| **test cameras loaded inside B** | isolated alternating A/B: **0.54 s bicycle / 1.06 s 0b5635** | **the one real lever** (below) |
| CUDA render+backward | 68.7–73.6 % | algorithm/hardware domain, not engineering |

## 3b. A-side scene tensor cache (user-approved data residency) — measured

`research/scene_cache.py` (new, 307 lines; build = A side, load = B side; no upstream change):

* **Bundle** (versioned by sha256 of `split.json` + `image_cache/manifest.json`):
  `images_u8.npy` — `(N,3,H,W)` uint8 in **train order**, memmapped on disk;
  `cams.npz` — `world_view_transform`, `full_proj_transform`, `camera_center`, `FoVx/FoVy`,
  image names/order, `cameras_extent`, all f32; `meta.json` — version/seed/bytes/timings.
* **Build (A side)**: runs the normal `Scene` load once (so the train-camera order matches
  what the trainer sees for the same seed), extracts the exact camera tensors, writes the
  uint8 NCHW stack to a memmap. No B work moves into A: A already reads the same pixels.
* **Load (B side)**: read npz + meta (1–2 ms), **one H→D upload** of the uint8 stack
  (0.043 s / 0.18 s), **one on-GPU `uint8→float/255` convert** (4.6 ms), then build
  Camera-compatible objects from the stored transforms — **no PNG/PIL decode, no CPU
  clamp/ones, no per-camera matrix inversions**.

| scene | N×res | bundle bytes | A build (wall / scene / write) | B normal load | B cached load | **B saving** | camera tensors | pixels |
|---|---:|---:|---|---:|---:|---:|---|---|
| bicycle | 169×576×960 | 280.4 MB | 4.26 s / 2.41 s / 0.90 s | 1.898 s | **0.064 s** | **1.834 s** | **bitwise 0.0** | max|Δ| 5.96e-8 (1 ulp of f32) |
| 0b5635 | 333×512×960 | 491.1 MB | 5.93 s / 3.46 s / 1.56 s | 3.150 s | **0.105 s** | **3.045 s** | **bitwise 0.0** | max|Δ| 5.96e-8 (1 ulp of f32) |

Memory accounting: B keeps the same delivery VRAM (f32 images 1.12/1.96 GB) plus a
transient 0.28/0.49 GB uint8 upload buffer (freed after convert); CPU/RSS peak during load
≈ the bundle size (mmap + one contiguous copy); disk 0.28/0.49 GB per scene.
A-side wall can be hidden: only the `Scene` part touches the GPU (2.4–3.5 s) and it is the
same load A would do anyway; the memmap write (0.9–1.6 s) is CPU-only and can overlap A's
GPU work (user-approved data residency), in which case the E2E cost is ≈0.
TIME equivalent: ΔB −1.83 s ⇒ **+0.43 TIME** (bicycle, 95 s budget); −3.05 s ⇒ **+0.72**
(0b5635, 94.29 s). Per-13-case mean expectation ≈ **+0.4…+0.6 TIME** (DL3DV scenes have the
largest test-view counts and gain most; hw scenes have few views and gain least).

**Real-loop B measurement (`--scene_cache`, delivery recipe, no quick-test):**

| scene | ctrl scene_load (same session) | cache scene_load | **ΔB** | ΔTIME | train_seconds (ctrl → cache) | N |
|---|---:|---:|---:|---:|---|---|
| bicycle | 2.181 s | **0.212 s** | **−1.97 s** | **+0.46** | 23.574 → 22.955 | 1,009,597 → 974,664 |
| dl3dv/0b5635 | 3.644 s | **0.241 s** | **−3.40 s** | **+0.80** | 25.970 → 25.801 | 795,534 → 798,641 |

(`num_final_gaussians` still varies ±3 % run-to-run — rasterizer atomics — so N is not an
identity check; the identity check is the bitwise tensor comparison above.)

Residual 1-ulp pixel difference: cached path computes `u8.float()/255` on GPU, upstream
computes `PIL→np.array/255` then `clamp`+`×ones`; the two differ by one float32 rounding
step (5.96e-8) on some pixels. This is far below rasterization/render nondeterminism
(run-to-run `num_final_gaussians` varies ±3 % on the same recipe) and is reported rather
than hidden. Camera transforms are bitwise identical because they are copied verbatim from
a normally constructed Scene.

## 4. Part 4 — legal/timing classification (all proposed changes)

| change | class | measured/expected saving | memory | risk | TIME Δ (budget 95) |
|---|---|---:|---|---|---|
| **A1 `--train_cameras_only`** (do not load/convert TEST cams in B) | **A pure elimination** | **0.54 s bicycle / 1.06 s 0b5635** (isolated; 4 alternating reps; same-code drift control −0.15/+0.02 s) | −0.17 GB / −0.28 GB | disables the diagnostic quick-PSNR only; optimization & official eval untouched | **+0.13…+0.25** |
| A2 `--lean_step` (drop no-op `.cuda()`, hoist realloc parse, index camera sampling) | A | ~2–7 ms measured microbenches ⇒ <0.05 s | 0 | none (same RNG stream) | <+0.01 |
| A3 Camera-init clamp/ones removal | A | 3.7–7.3 ms | 0 | none | <+0.01 |
| A4 metric_map reuse | A | 0 (measured not faster) | 0 | none | 0 (dropped) |
| B1 anything to warm up inside B | B | none identified: GT/transforms/PLY are loaded once and used; no lazy model loads in the delivery path | — | — | 0 |
| C1 versioned tensor cache of resized images / parsed PLY | **C cross-run research cache** | ~1–2 s per *repeat* | 1–2 GB disk | stale-cache risk if A outputs change (needs versioning) | **0 for the scored run** (helps only research reruns) |
| **A5 A-side scene tensor cache** (`research/scene_cache.py`, `--scene_cache`): uint8 NCHW images + camera tensors prepared in A; B does one H→D + one GPU convert | **A (user-approved data residency / latency hiding)** — explicitly *not* class D: A already loads the same images/poses, no optimization work moves | **1.83 s bicycle / 3.05 s 0b5635** | disk 0.28/0.49 GB; VRAM +0.28/0.49 GB transient; A wall 4.3–6.0 s (hideable) | stale-cache risk → version hash; 1-ulp pixel diff | **+0.43 / +0.72** |
| D2 pre-parse PLY / pre-allocate optimizer state in A | **D time-shifting** | 0.4–0.8 s init | — | same | not counted |

Rule applied: A is reported as a real score gain; C is explicitly **not** score-relevant;
D is labelled **time-shifting** and is not presented as free TIME.

## 5. Prototype proofs (default off; delivery unchanged)

`research/train_sweep.py` now has two opt-in prototypes (`records/b-eng-20261030/patch.diff`):

* `--train_cameras_only`: monkeypatches the research-local scene loader to strip
  `test_cameras` before `Scene(...)`; `quick_test_psnr` becomes `None` (the official eval
  re-renders from the PLY with its own loader, so scored metrics are unaffected by
  construction). **Log/diagnostic only difference.**
* `--lean_step`: per-step no-ops removed; camera sampling keeps the identical RNG stream
  (index permutation of the same stack), realloc parse hoisted.

End-to-end A/B (single runs, delivery recipe, no quick-test) — **noise-limited**:
`bicycle`: ctrl scene_load 2.18 s → camonly 2.00 s (−0.18), both 2.10 (−0.09); `0b5635`:
ctrl 3.64 → camonly 2.72 (−0.93), both 3.65 (+0.00). The run-to-run `scene_load` σ is
0.2–0.5 s and run-to-run `num_final_gaussians` varies ±3 % (rasterizer atomics), so a
single-run A/B cannot resolve a 0.1–1 s effect. The defensible measurement is therefore
the **isolated alternating A/B of exactly the changed work** (§3/§4 A1): 4 reps, both
orders, same-process, same-code drift control:

| scene | test views | mean saving | per-rep savings | same-code drift control |
|---|---:|---:|---|---:|
| bicycle | 25 | **0.538 s** | 0.46 / 0.63 / 0.71 / 0.35 | −0.15 s |
| dl3dv/0b5635 | 48 | **1.059 s** | 1.08 / 1.08 / 0.85 / 1.22 | +0.02 s |

**Recommendation:** if a delivery change is allowed, adopt **A1** (`--train_cameras_only`
default-off flag; enable in run_pipeline only with approval) — it is a strict pure
elimination worth **+0.13…+0.25 TIME** on these scenes with zero optimization difference,
and it costs no memory. Do **not** pursue per-step micro-optimizations (≤0.01 TIME),
fused Adam (falsified), resolution/SH reductions (lossy), or A-side caching (class D) as
score gains.

## 6. Exact commands / artifacts

```bash
# breakdown (Part 1)
conda run -n fastgs python -m research.train_sweep ... --time_breakdown --no_quick_test
# isolated A/B of the changed work (Part 2/3): alternating Scene builds with/without test cams
python /tmp/stagedloss/beng_camonly2.py
# end-to-end prototype A/B (noise-limited)
python /tmp/stagedloss/beng_ab.py     # ctrl / camonly / lean / both, bicycle + 0b5635
# microbenches (no-op .cuda, list copy/pop, realloc parse, metric_map, render reuse identity)
python /tmp/stagedloss/beng_micro.py
```

Artifacts: `records/b-eng-20261030/{microbench.json, breakdown/, ab/ab.jsonl,
camonly_isolated{,_2}.json, scene_cache/results.json, cache_hits/cache_hits.json,
scene_cache.py, patch.diff (train_sweep/run_sweep prototypes: scene_cache,
train_cameras_only, lean_step), run.sh}`; raw on N1
`/root/gpufree-data/outputs/b-eng-20261030/`.

No git push; `competition/upstream/` untouched; all prototypes default-off; delivery
recipe bit-identical when the flags are absent.
