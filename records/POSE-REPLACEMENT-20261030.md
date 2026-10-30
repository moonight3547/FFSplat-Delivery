# Pose-pipeline replacement experiment — hw COLMAP (2026-10-30)

**Verdict: KEEP the current COLMAP module-A pipeline. Do not adopt a rotation-known
pose pipeline as the default.** The AR `frames_info.txt` rotations are a valuable
*validation / QA* signal (they independently confirm COLMAP's rotations to
**0.60° mean / 1.22° max**, matching the organizer README's 0.54°) but they are
**not** a better pose estimator for NVS: replacing COLMAP's joint-BA poses with
"AR rotations + epipolar-optimal translations" costs **30.5 / 36.6 quality
points** on hw/car / hw/trashbin, and **79% of the ReSplat context-PSNR gap is the
pose geometry itself** (not intrinsics, images or world scale).

Separately, **the premise that module A is nondeterministic is only half true**:
on both scenes the current pipeline is *structurally* deterministic (same image
set, byte-identical `split.json`, same 16 context views) and *numerically*
reproducible to **0.02–0.04° rotation / 0.03–0.08 % of scene scale translation**,
though not bit-identical. End-to-end repeats on identical inputs reproduce the
proxy score to **≤ 0.37 points** (car 32.72/32.39, trashbin 49.58/49.21/49.54).
So the historical hw/trashbin q=9.0 ↔ 49.1 swing cannot be explained by A or B
nondeterminism on a fixed input set — it must come from a *different input image
set* (H1 re-uploaded several raw scenes on 2026-10-29/30) or a different
recipe/environment.

All numbers below are measured on H1 (the only COLMAP+ReSplat node, 1×4090),
`ffsplat-enhance` / `organizer-fastgs` envs, colmap 3.7 CPU-SIFT, 20 cores.
Raw artifacts: `/workspace/organizer-projects/FFSplat-Hackathon/outputs/pose-exp-20261030/`.

---

## 1. What was run

| stage | what | where |
|---|---|---|
| P0 | current module A ×3 repeats × hw/{car,trashbin} into fresh work dirs, COLMAP tmp kept | `outputs/pose-exp-20261030/p0/` |
| P1 | deterministic-COLMAP decomposition (feature/matching vs mapper) | `.../p1/` — **parked by the clean-13 CPU freeze, see §9** |
| P2 | `research/pose_rotfixed.py` — `check`, `rotonly`, `transavg` candidates | `.../C/{car,trashbin}/` |
| P3 | end-to-end delivery recipe on 9 arms (current r1/r2/r3, rotonly, transavg) | `.../p3/` |
| diag | 2 controlled ReSplat-only ablations explaining the candidate gap | `.../diag/` |

Delivery recipe (node-facts recipe, identical for every arm):
`research.run_sweep --iterations 5000 --position-lr-max-steps 5000
--opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 --densify-from 500
--densify-until 1250 --realloc-frac 0.15 --realloc-at 1250 --image-cache <arm>/cache
--budget 95.0`, init = ReSplat (`research.resplat_variant --num_refine 2
--device cuda:0 --fix-sh-basis`, checkpoint
`resplat-base-dl3dv-540x960-view16-a72dc6d0.pth`). Proxy score as in
`ProblemDesc` §4 (quality 80 + TIME 20; official total unconfirmed).

---

## 2. `frames_info.txt` — verified facts (independently valuable)

Everything below was measured against COLMAP reconstructions of the same images
(`research.pose_rotfixed check`, results in `.../pose_report.json`).

1. **Declared image size is wrong.** `frames_info.txt` declares 1080×1440 but the
   actual jpgs are **720×960**. The intrinsics must therefore be scaled by
   `720/1080 = 2/3`: `fx = 1027.94·2/3 = 685.29`, `fy = 1029.8·2/3 = 686.53`.
   COLMAP's own OPENCV estimate on car is `fx 684.13, fy 680.43` → agreement
   **+0.17 % / +0.90 %** (trashbin: −1.16 % / +0.18 %). So the dump's focal
   lengths are real and only the declared resolution is stale.
2. **`principalX/Y = 0` means "offset from image centre"**, not absolute
   coordinates (an absolute (0,0) would be the top-left corner). `cx = W/2 + px`,
   `cy = H/2 + py` → `cx=360, cy=480` at 720×960, which is exactly what COLMAP
   uses (it holds the principal point at the centre with `ba_refine_principal_point=0`).
3. **Lens distortion is negligible.** COLMAP's car calibration:
   `k1=0.00991, k2=−0.01419, p1=0.00141, p2=0.00024` → radial displacement
   ≤ 0.5 px at the corner of the 720×960 frame. Undistortion is therefore *not*
   what protects ReSplat quality (it only changes pixels by <1 px), and the AR
   pinhole model is adequate for the raw jpgs.
4. **Rotation convention = `w2c_cv`**: the quaternion `(rotX, rotY, rotZ, rotW)`
   is a **world→camera** rotation in OpenCV camera axes. Gauge-fit residual vs an
   independent COLMAP model: **car 0.598° mean / 0.612° median / 1.223° max;
   trashbin 0.578° / 0.522° / 1.519°** — this reproduces the README's 0.54° claim.
   Falsified alternatives (mean residual on car / trashbin):
   | reading | car | trashbin |
   |---|---|---|
   | **`w2c_cv` (adopted)** | **0.60°** | **0.58°** |
   | `c2w_cv` (identity + c2w) | 9.15° | 17.9° |
   | `c2w_gl` (ARKit c2w, GL axes) | 80.8° | 86.3° |
   | `w2c_gl` | 80.8° | 83.7° |
5. **AR positions are unusable as translations**, as the README says: after the
   best rigid (Umeyama) alignment to COLMAP camera centres, residual is
   **13.5 % median / 36.2 % max** of scene scale on car and **11.2 % / 25.4 %** on
   trashbin, with a trajectory-length ratio of 1.27 — i.e. the positions are not
   related to COLMAP's by any similarity. Only the rotations are kept.

---

## 3. P0 — stability matrix of the current pipeline (3 repeats × 2 scenes)

Fresh output dirs, same input images, same command
(`module_a/preprocess.py --num_context 16 --llffhold 8 --max_resolution 960`).

| scene | repeat | registered | A pose s | A total s | res |
|---|---|---:|---:|---:|---|
| car | r1 | 70/70 | 131.7 | 143.2 | 576×768 |
| car | r2 | 70/70 | 132.4 | 144.0 | 576×768 |
| car | r3 | 70/70 | 138.1 | 151.2 | 576×768 |
| trashbin | r1 | 75/75 | 298.5 | 313.3 | 576×768 |
| trashbin | r2 | 75/75 | 298.6 | 313.1 | 576×768 |
| trashbin | r3 | 75/75 | 301.3 | 315.7 | 576×768 |

Pose disagreement between repeats (gauge-invariant metrics; the *primary* numbers,
because a centre-only Umeyama rotation is ill-conditioned for a near-circular
orbit and can absorb a constant world-origin shift as a spurious rotation —
it gave a misleading 17° before that was diagnosed):

| scene | pairs | rel. rotation median / max | rel. translation in camera frame median / max | rotation-gauge trans residual |
|---|---|---|---:|---:|
| car | r1–r2 | 0.012° / 0.039° | 0.009° / 0.70° | 0.03 % scale |
| car | r1–r3 | 0.023° / 0.073° | 0.017° / 0.84° | 0.03 % |
| car | r2–r3 | 0.023° / 0.058° | 0.019° / 1.26° | 0.02 % |
| trashbin | r1–r2 | 0.042° / 0.12° | 0.040° / 3.23° | 0.08 % |
| trashbin | r1–r3 | 0.031° / 0.14° | 0.031° / 1.46° | 0.04 % |
| trashbin | r2–r3 | 0.037° / 0.13° | 0.035° / 2.92° | 0.07 % |

(Max values are isolated very-short-baseline pairs where the direction is
ill-conditioned; the medians and p90s are the meaningful numbers. `rel. rotation` =
`R_a_i^T R_a_j` vs `R_b_i^T R_b_j`; `rel. translation` = `R_wc_i (C_j − C_i)`
direction — both invariant to the arbitrary world frame.)

**Byte-level input comparison (sha256, `.../input_hashes.json`):**

| item | car | trashbin |
|---|---|---|
| raw jpgs (all frames) | `b1b3c22a459da864…` | `73b3836061e56815…` |
| `frames_info.txt` | `bdb7d7e3ea1708c5…` | `439ca9de97d166af…` |
| `split.json` (all 3 repeats) | **identical** `f29a052c2714` | **identical** `4d618e368f24` |
| `sparse/0/images.txt` | differs (`4a8fabd0` / `de867d88` / …) | differs |
| `sparse/0/cameras.txt` | differs (`4ee9defe` / `86e638bd` / …) | differs |
| resized `images/*.png` | differs (`fc364244` / `0f8382ef` / …) | differs (`1a63b9f7` / `15e9ff94` / `4c6a44cc`) |

**Precise statement of what still varies:** module A is **structurally
deterministic** (identical image list, byte-identical `split.json` = identical
train/test split and 16 context views) and **numerically reproducible to
0.02–0.04° / 0.03–0.08 %**, but **not bit-identical**: `image_undistorter` warps
with each run's estimated OPENCV distortion and the mapper's BA converges to
slightly different focal lengths / poses. That is the only input difference
between repeats (see §4: it is worth ≤0.37 proxy points end-to-end).

---

## 4. End-to-end repeatability on identical inputs (the q=9↔49 question)

Current-A repeats through the full delivery recipe (ReSplat → 5 k module B →
official metrics), sequential on the 4090:

| scene | repeat | PSNR | SSIM | LPIPS | q | B s | total | ReSplat ctx PSNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| car | r1 | 23.03 | 0.8284 | 0.1699 | 32.72 | 25.7 | 48.92 | 21.08 |
| car | r2 | 22.94 | 0.8279 | 0.1710 | 32.39 | 36.0* | 46.18 | 21.09 |
| trashbin | r1 | 27.93 | 0.9005 | 0.1340 | 49.58 | 22.6 | 66.53 | 25.67 |
| trashbin | r2 | 27.79 | 0.9022 | 0.1360 | 49.21 | 27.0* | 65.11 | 25.66 |
| trashbin | r3 | 27.92 | 0.9026 | 0.1350 | 49.54 | 23.4 | 66.28 | 25.77 |

\* the r2 arms' `B` is **inflated 20–40 %** because another teammate's COLMAP A
stage was running on the same 20 CPU cores during those runs (car 36.0 s vs
25.7 s = +40 %; trashbin 27.0 s vs 22.6/23.4 s = +15–20 %). Quality is
unaffected (q spread 0.33 / 0.37), and the ReSplat context PSNR is identical
(21.08 vs 21.09; 25.67 vs 25.66), i.e. the effect is purely wall-clock.

> **Measurement caveat (applies to every timing number in this repo):** module-B
> wall clock is **CPU-contention sensitive as well as GPU-contention sensitive**.
> Any scored `B_seconds`/TIME must come from a run where the 20 CPU cores are
> quiet too (the clean-13 re-measurement is loadavg-gated for this reason). This
> was measured here: an unrelated CPU-only COLMAP job inflated car's B from
> 25.7 s to 36.0 s (+40 %) with identical quality.

**⇒ With identical inputs, both A and the downstream pipeline are reproducible
to ≤0.4 proxy points. The historical q=9.0 vs q=49.1 for hw/trashbin cannot be
reproduced from a fixed input set; it is consistent with a different raw image
set (H1 re-uploaded raw data on 2026-10-29/30) or a different recipe/environment,
not with intrinsic A/B nondeterminism.**

---

## 5. The replacement candidate — `research/pose_rotfixed.py`

New file, no modification of `competition/upstream/`; writes module-A-format work
dirs by calling the *unmodified* upstream `module_a/preprocess.py::write_work_dir`
(same images/split/cameras/images.txt layout), so the rest of the pipeline is
unchanged.

Modes: `check` (convention/intrinsics/position QA), `rotonly` (replace only
rotations in an existing work dir, keep centres), `transavg` (full rotation-known
replacement), `self-check` (synthetic CPU test — passes: translation averaging
recovers a synthetic camera network to 0.01 % of scale, triangulation 0.0008).

`transavg` pipeline:
1. parse `frames_info.txt` (actual 720×960 intrinsics, `w2c_cv` rotations);
2. COLMAP SIFT keypoints+matches from a `database.db` (can be produced once and
   reused; everything after it is deterministic);
3. per pair, known relative rotation `R_ij = R_wc_b R_wc_a^T` → relative
   translation direction by **2-point RANSAC** on `x_j^T [t]_x R_ij x_i = 0`
   (linear in t; cheirality resolves the sign);
4. **global translation averaging**: `min Σ ||(I − d d^T)(C_i − C_j)||²` (linear
   SVD, Huber IRLS, one gauge camera, global mirror resolved by the `s_ij > 0`
   vote);
5. union-find tracks + linear DLT triangulation with a minimum-ray-angle test;
6. **alternating known-rotation BA** (re-triangulate; Gauss–Newton on camera
   translations and points, rotations pinned, Huber);
7. write the work dir.

Determinism: two runs on the same database are **bit-identical**
(rel. rotation 0.00000°, rel. translation 0.00000°).

Internal diagnostics / cost:

| scene | frames | usable pairs | pair epipolar median | TA residual median / p90 | points | reproj median / p90 | pose-est s |
|---|---:|---:|---:|---|---:|---|---:|
| car | 70 | 1084/2129 (inl. ratio 0.84) | 0.39 px | 0.226° / 3.5° | 23 834 | **0.58 px** / 1.86 px | 48.7–67.0 |
| trashbin | 75 | 1337/2775 (inl. ratio 0.55) | 0.49 px | 0.980° / 6.4° | 27 404 | **0.77 px** / — | 51.3 |

Accuracy *relative to COLMAP* (correct gauge from the AR rotations; COLMAP is not
ground truth, but it is the current delivered reference):

| scene | rotation median / max | translation median | translation max | pairwise rel. rotation | pairwise rel. translation |
|---|---|---:|---:|---|---|
| car | 0.612° / 1.223° | 5.78 % scale | 23.2 % | 0.71° median | 0.67° median, p90 3.9° |
| trashbin | 0.522° / 1.519° | 1.15 % scale | — | 0.72° median | 0.52° median, p90 1.7° |

The rotation difference is expected (it *is* the AR-vs-COLMAP error, 0.52–0.61°);
the translation difference is the real candidate error (5.8 % on car, 1.15 % on
trashbin).

Efficiency (separate column — module-B TIME is what is scored, so this is not
"free points", it is iteration/E2E speed):

| scene | current A pose s | candidate pose s | speedup | full A (incl. resize) |
|---|---:|---:|---:|---|
| car | 131.7 / 132.4 / 138.1 | 48.7–67.0 | **2.0–2.8×** | ~81 s vs 143–151 s |
| trashbin | 298.5 / 298.6 / 301.3 | 51.3 | **5.8×** | ~65 s vs 313–316 s |

---

## 6. P3 — end-to-end quality, candidates vs current (delivery recipe)

| scene | arm | PSNR | SSIM | LPIPS | q | B s | total | ReSplat ctx PSNR |
|---|---|---|---|---:|---:|---:|---:|---:|
| car | **current A r1** | **23.03** | **0.8284** | **0.1699** | **32.72** | 25.7 | **48.92** | **21.08** |
| car | rotonly | 21.39 | 0.7069 | 0.2333 | 9.00 | 27.4 | 24.81 | 20.41 |
| car | transavg | 17.83 | 0.5197 | 0.3411 | 2.27 | 29.4 | 17.61 | 17.65 |
| trashbin | **current A r1** | **27.93** | **0.9005** | **0.1340** | **49.58** | 22.6 | **66.53** | **25.67** |
| trashbin | rotonly | 22.70 | 0.7590 | 0.2554 | 10.88 | 30.8 | 25.90 | 22.30 |
| trashbin | transavg | 22.91 | 0.7611 | 0.2400 | 13.22 | 23.4 | 29.97 | 22.31 |

(trashbin current repeats for reference: r2 q 49.21, r3 q 49.54, ctx 25.66/25.77.)

Both candidates lose on **both** scenes; nothing wins anywhere. **Verdict:
keep COLMAP.**

`rotonly` is **confounded** and is *supporting* evidence only: it replaces
rotations while keeping COLMAP translations, so the resulting pose set is not
geometrically self-consistent by construction. It still shows how sensitive NVS
is: a 0.52–0.61° rotation edit alone costs 23.7 (car) / 38.7 (trashbin) quality
points and 0.7 / 3.4 dB of context PSNR.

The falsification of the *rotation-known* hypothesis rests primarily on
`transavg`, which is internally self-consistent (reprojection 0.58/0.77 px — as
good as a joint BA) yet 30.5 (car) / 36.6 (trashbin) quality points worse.

---

## 7. Why the rotation-known candidate loses — controlled ablations

ReSplat context self-render PSNR on hw/car (higher = the 16 context views are
better explained; the best no-GT pose-quality proxy):

| input | ctx PSNR | Δ vs candidate |
|---|---:|---:|
| current A poses + COLMAP-undistorted images + COLMAP K | **21.08** | +3.43 |
| candidate poses + COLMAP-undistorted images + COLMAP K | 18.38 | +0.73 |
| candidate poses, world rescaled ×6.76 to COLMAP scale | 17.95 | +0.30 |
| candidate poses + candidate images/K (baseline) | 17.65 | — |

So world scale (+0.30 dB) and raw-vs-undistorted images/intrinsics (+0.73 dB) are
each minor; **≈79 % of the 3.43 dB gap stays with the pose geometry itself.**

Mechanistic explanation (corollary): the AR rotations carry a **0.5–0.6° error
relative to the image-consistent optimum**. In a known-R estimator that error
cannot be corrected by rotations; it is **absorbed into the estimated
translations**, which biases exactly the epipolar relationships NVS depends on.
COLMAP's joint BA has no such constraint — rotations and translations co-adapt to
minimise reprojection error, and the 3DGS renderer rewards that self-consistency,
not absolute pose "truth". Hence "AR rotations + epipolar-optimal translations"
is a *worse* estimator for NVS than joint BA.

**Implication for the follow-up question "can AR rotations give a faster
high-precision COLMAP result": not directly.** AR rotations must not replace the
rotation estimation; the only sound use is as an **initializer / regularizer for a
later rotations-free BA** (i.e. seed, then jointly optimise), which is what
task-6 S3 tests.

---

## 8. Recommended default (exact commands)

Keep the current module A unchanged — the winning command is exactly the current
one:

```bash
CONDA=/usr/local/miniconda3/bin/conda
$CONDA run --no-capture-output -n ffsplat-enhance python \
  competition/upstream/baseline/module_a/preprocess.py \
  --scene_dir /workspace/organizer-assets/datasets/hw/<scene> --group hw \
  --out outputs/<exp>/work/hw/<scene> \
  --num_context 16 --llffhold 8 --max_resolution 960
```

Use the AR dump only as a **QA / validation gate** (cheap, no estimation):

```bash
$CONDA run --no-capture-output -n ffsplat-enhance python -m research.pose_rotfixed check \
  --scene_dir /workspace/organizer-assets/datasets/hw/<scene> \
  --model outputs/<exp>/work/hw/<scene>_colmap_tmp/sparse/0
# expected on a healthy hw scene: best convention = w2c_cv,
# rotation residual <= ~1.2 deg max, AR positions 10-40 % of scale (they are invalid)
```

If the QA residual jumps (e.g. >5° mean, or a non-`w2c_cv` convention wins), the
AR dump and the reconstruction disagree — that is the signal a raw input set
changed or the reconstruction is degenerate, which is exactly the failure mode
that could explain a junk delivery run.

For iteration speed only (not for delivery): `research.pose_rotfixed transavg`
gives a 2–5.8× faster A stage and bit-identical reruns, at a measured cost of
30–37 quality points — do not use it for submission.

---

## 9. Deterministic COLMAP attempt (P1) — status and what is already known

A mapper-determinism decomposition is prepared and queued in
`outputs/pose-exp-20261030/p1run.sh` (db-level keypoint/match hashes for the three
P0 repeats; `feature_extractor` twice with/without `--SiftExtraction.num_threads 1`;
`mapper` ×3 with default flags and ×3 with
`--random_seed 0 --Mapper.num_threads 1
--Mapper.ba_min_num_residuals_for_multi_threading 100000000
--Mapper.ba_global_max_refinements 3` on a fixed database). It was **parked on
purpose** by the clean-13 CPU freeze (a mapper run would pollute another
teammate's module-B wall clock) and will be completed and appended here after the
freeze.

What is already established — and it locates the variation precisely (light
sqlite-only pass over the three P0 databases, `tools/db_hash.py`):

| item | car | trashbin |
|---|---|---|
| SIFT keypoint count per repeat | **353 795 / 353 795 / 353 795** (identical) | **343 014 ×3** (identical) |
| keypoint blob md5 | `3f59be0e` / `8ff8ad77` / `96263d7d` → **differs** | differs |
| raw match count | 389 444 / 389 576 / 389 278 (±0.04 %) | 548 805 / 548 957 / 548 729 (±0.02 %) |
| match blob md5 | differs | differs |
| initial camera (feature-extractor prior) | identical | identical |
| `split.json` | **byte-identical** | **byte-identical** |
| mapper `cameras.txt` (focal) | differs at ~1e-4 relative | differs |
| mapper poses | 0.02–0.04° / 0.03–0.08 % | same order |

So the chain is: **SIFT extraction already returns the same keypoints in the same
count but not bit-identically** (float/parallel reduction differences), which
changes the ratio-test match set by ±0.04 %, which changes the mapper's BA
trajectory by 0.02–0.04°, and the undistorter then renders slightly different
pixels. Nothing structural varies: same image list, same split, same 16 context
views, same registered count (70/70/70, 75/75/75), same initial camera.

**Can it be made bit-identical?** A CLI-only deterministic recipe would require
`--SiftExtraction.num_threads 1 --SiftMatching.num_threads 1 --random_seed 0
--Mapper.num_threads 1 --Mapper.ba_min_num_residuals_for_multi_threading
100000000`, which is prepared in `p1run.sh` (parked by the clean-13 CPU freeze);
the expectation is that single-threading removes the float-order variation at the
SIFT/matching level. **It is not needed for the delivery:** the residual
0.02–0.04° variation is worth **≤0.4 proxy points** end-to-end (§4) and the
pipeline registers every image on every run. Stated honestly: A is
*structurally deterministic* and *numerically reproducible*, not bit-identical —
which is exactly what the repeat data supports.

---

## 10. Falsified / corrected claims (so they are not re-litigated)

1. **"Module A nondeterminism explains the q=9↔49 swing"** — falsified for a fixed
   input set: A poses reproduce to 0.02–0.04°, and five end-to-end repeats
   (car r1/r2, trashbin r1/r2/r3) reproduce q to ≤0.37 on identical inputs.
2. **"COLMAP rotations are the main pose error source"** — falsified: replacing
   them with the AR rotations (which are 0.6° closer to "truth") *lowers* quality
   by 23.7/38.7 points (`rotonly`, confounded) and the self-consistent known-R
   solution (`transavg`) is 30.5/36.6 points worse. NVS needs image-consistent
   joint poses, not absolute pose truth.
3. **"frames_info declares 1080×1440"** — false for the shipped jpgs (720×960);
   using the declared size directly would mis-scale focals by 1.5×.
4. **"The AR quaternion is an ARKit c2w transform"** — false; it is `w2c_cv`
   (mean residual 0.60°/0.58° vs COLMAP; the ARKit-c2w reading is 80° off).
5. **"AR pinhole intrinsics are unusable because the jpgs are distorted"** —
   largely false: COLMAP's own distortion estimate implies <1 px warping.
6. **"Only GPU contention matters for module-B timing"** — false: sharing the 20
   CPU cores inflated B by 20–40 % with identical quality.

## 11. §speed — faster high-precision module A (task-6)

User question ①: "can AR rotations give a faster high-precision COLMAP result?"
Task-4 answered the accuracy half: **AR rotations cannot replace rotation
estimation** (§6–§7). This section is the speed half, measured on H1 with a quiet
machine (nothing else running, so both CPU and GPU wall clocks are valid).

**Cost structure (hw/car, current A):** feature extraction 7 s + exhaustive
matching **83 s** + mapper 43 s = 133 s ≈ the 131.7 s A-pose baseline. So **63 %
of A is exhaustive matching**, not the mapper. hw/trashbin is different: A's pose
stage is 298.5 s and its mapper dominates (fewer, harder images).

### 11.1 CPU candidates (hw/car, 70 images, same GPU node)

| # | config | feature | match | map | register | BA | pose-est total | speedup | registered | pose diff vs current-A |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| — | current A | 7 | 83 | 43 | — | — | 133 | 1.00× | 70/70 | — |
| S0 | lower BA caps (`ba_global_max_num_iterations 25`, `ba_local_max_num_iterations 15`, `ba_global_max_refinements 3`) | 7 | 83 | 41 | — | — | 131 | **1.02×** | 70/70 | n/a |
| S1 | `max_num_features 4096` + `max_image_size 1280` | 7 | 83 | 43 | — | — | 133 | **1.00× (no-op)** | 70/70 | none |
| S1b | matcher `max_num_trials 2000` | 7 | 81 | 43 | — | — | 131 | **1.02×** | 70/70 | n/a |
| S4 | AR-chosen init pair (`--Mapper.init_image_id1 66 --init_image_id2 67 --init_num_trials 1`; 1043 matches, 8.98° separation) | 7 | 83 | 43 | — | — | 133 | **1.00×** | 70/70 | n/a |
| **S2** | **subset-SfM stride 2** | 7 | 20+33 | 12 | 1 | 2 | **75** | **1.77×** | **70/70** | rot 0.048° med / 0.118° max; trans 0.057 % med / 0.171 % max |
| **S2** | **subset-SfM stride 3** | 7 | 10+33 | 5 | 1 | 1 | **57** | **2.33×** | **70/70** | rot 0.064° / 0.127°; trans 0.128 % / 0.242 % |

hw/trashbin (75 images, current A pose stage **298.5 s** — the scene S2 helps most):

| config | pose-est total | speedup | registered | pose diff vs current-A |
|---|---:|---:|---|---|
| S2 subset-SfM stride 2 | **124** | **2.41×** | **75/75** | rot 0.147° med / 0.492° max; trans 0.303 % / 0.878 % |
| S2 subset-SfM stride 3 | **87** | **3.43×** | **75/75** | rot 0.175° / 0.886°; trans 0.387 % / 1.738 % |

Pose diff = rotation-gauge alignment of the candidate COLMAP model against the
current-A model (rotation error per camera, translation residual relative to scene
scale). For comparison the *native* current-A run-to-run variance is
0.017–0.026° / 0.03–0.08 %, so S2 on car is ~2× the noise floor (very safe) and on
trashbin ~6× (a real, if small, change). Full A adds the undisort+resize constant
(~14 s car, ~15 s trashbin): car 143 s → S2 stride 2 ≈ **89 s (1.6×)** / stride 3
≈ **71 s (2.0×)**; trashbin 313 s → **139 s (2.25×)** / **102 s (3.1×)**.

**S1 is a proven no-op in this build, stated explicitly rather than claimed as a
win:**
* `--SiftExtraction.max_image_size 1280` is a **cap** and COLMAP never upscales;
  the hw jpgs are 720×960 < 1280 (and < the 3200 default) → no change.
* `--SiftExtraction.max_num_features` is **ignored by this CPU-SIFT build**: a
  controlled probe with `--max_num_features 1000` still produced **1260–1329
  keypoints/image** (full extraction is ~5054/img with the 8192 default; the
  "4096" database had the identical 353 795 keypoints). `estimate_affine_shape`
  and `domain_size_pooling` are already 0 (defaults) → also non-levers.
* `--SiftMatching.max_num_trials 2000` saves only 2 s of 83 s → not worth a
  quality risk.

**S4** shows COLMAP's default init-pair search is not a meaningful cost: forcing
the best AR-rotation-separated co-visible pair keeps mapper time at 43 s. AR
rotations do not help incremental SfM.

### 11.2 S3 — AR-seeded rotations-free BA (the only route that "uses AR")

`research.pose_rotfixed transavg --ba-rot-free-from 2`: AR rotations are the seed,
held fixed for 2 iterations, then rotations + translations + points are optimised
jointly (new implementation in `research/pose_rotfixed.py`; synthetic self-check
passes).

| scene | pose-est s | speedup vs A | rot deviation from AR seed med/max | reprojection med | rel. rotation vs current-A (pairwise) | e2e q |
|---|---:|---:|---|---:|---|---:|
| car | **32.4** | 4.1× | 0.074° / 0.294° | 0.541 px | 0.699° | **2.17** |
| trashbin | **39.9** | 7.5× | 0.117° / 0.357° | 0.607 px | 0.667° | (not scored; same regime) |

The decisive negative: **releasing the rotations does not recover COLMAP-level
accuracy.** BA moves each camera only 0.07–0.12° from its AR seed and keeps a
0.67–0.70° relative-rotation disagreement with current A — the known-R solution is
a local minimum, and the joint point adjustment absorbs the 0.6° AR rotation error
instead of correcting the rotations. Car e2e confirms it: **q 2.17, total 17.55**
(vs 32.31 / 48.54), i.e. the same failure regime as `transavg`. AR seeding buys
4–7× speed but not precision.

### 11.3 End-to-end delivery check (quiet H1, serial GPU)

Delivery recipe identical to §1; ReSplat init PLY regenerated **per candidate work
dir** (its world frame must match; reusing the current-A PLY would be a frame
mismatch). Controls are the quiet-gated clean-13 rows.

| scene | arm | PSNR | SSIM | LPIPS | ctx PSNR | q | B s | total | Δq vs control | Δtotal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| car | **control (clean-13)** | — | — | — | 21.08 | **32.31** | 25.63 | **48.54** | — | — |
| car | S2 stride 2 | 22.96 | 0.8253 | 0.1715 | 21.04 | **32.09** | 26.09 | **48.21** | **−0.22** | **−0.33** |
| car | S2 stride 3 | 22.91 | 0.8195 | 0.1731 | 21.03 | 31.27 | 25.97 | 47.41 | −1.04 | −1.13 |
| car | S3 AR-seeded free BA | 17.72 | 0.5144 | 0.3459 | 17.69 | 2.17 | 29.28 | 17.55 | −30.1 | −31.0 |
| trashbin | **control (clean-13)** | — | — | — | 25.67 | **48.07** | 22.74 | **64.98** | — | — |
| trashbin | S2 stride 2 | 26.85 | 0.8868 | 0.1492 | 25.17 | 44.58 | 22.72 | 61.49 | **−3.49** | **−3.49** |
| trashbin | S2 stride 3 | 26.15 | 0.8751 | 0.1541 | 24.97 | 42.11 | 22.95 | 58.96 | −5.96 | −6.02 |

### 11.4 S2 + AR-intrinsics prior (targeted follow-up: **negative**)

The trashbin loss in §11.3 correlates with a focal-length drift: plain S2's final
BA lands on `fy 670.9` vs current-A's `685.3` (−2.1 %). To test whether the AR
intrinsics can fix that, S2 stride 2 was rerun with the camera frozen to the
AR-derived PINHOLE `fx=685.293, fy=686.533, cx=360, cy=480` (720×960) and a
focal-frozen BA (`--BundleAdjustment.refine_focal_length 0` etc.):

| arm | ctx PSNR | q | B s | total | Δq vs control |
|---|---:|---:|---:|---:|---:|
| trashbin S2 stride 2 (plain) | 25.17 | 44.58 | 22.72 | 61.49 | −3.49 |
| trashbin S2 stride 2 + AR-K frozen | 24.83 | **41.98** | 23.17 | 58.78 | −6.09 |
| car S2 stride 2 (plain) | 21.04 | **32.09** | 26.09 | 48.21 | −0.22 |
| car S2 stride 2 + AR-K frozen | 20.03 | **24.05** | 26.58 | 40.05 | −8.26 |

**Rejected — the AR focal prior makes it worse on both scenes.** Two lessons:
1. For NVS the *effective* focal is what matters, and AR's focal (685.3/686.5,
   which matched the *A* reconstruction to 0.2–0.9 %) is a worse operating point
   than what each scene's own BA estimates from the delivers — a 0.5–2 % focal
   change costs 8–10 q. So AR intrinsics are useful as a *validation* signal
   (§2), not as a hard prior.
2. The trashbin S2 loss is therefore **not** fixed by pinning the AR focal; it is
   a genuine subset-SfM accuracy cost on a mapper-hard scene.

### 11.5 Verdict

**Adopt criterion (Δq ≤ 0.5 AND Δtotal ≤ 0.5 while A is meaningfully faster):**

* **S2 stride 2 on hw/car: PASS** (Δq −0.22, Δtotal −0.33, A pose stage 1.77×,
  full A 1.6×).
* **S2 stride 2/3 on hw/trashbin: FAIL** (Δq −3.5 / −6.0). Root cause measured:
  mapping only every 2nd/3rd frame leaves the focal length under-constrained —
  the final BA drifts to `fy 670.9` vs current-A's `685.3` (**−2.1 %**) and the
  pose set is 0.15° / 0.30 % from current A (vs 0.05° / 0.06 % on car), which the
  renderer sees as blurred/wrong geometry.
* **S3 AR-seeded free BA: FAIL by a wide margin** (q 2.2 vs 32.3).
* S0/S1/S4: ≤2 % → **no measurable win, not adopted.**
* **Uniform-hyperparameter rule (one stride for all scenes): NEITHER stride
  qualifies.** stride 3 fails both scenes (car Δq −1.04, trashbin Δq −5.96);
  stride 2 passes car (Δq −0.22, Δtotal −0.33) but fails trashbin (Δq −3.49).
  Since both scenes are in the 13-case set, the uniform rule resolves to **keep
  current A**. S2 stride 2 may be used **per-scene** only where its ctx PSNR is
  within ~0.1 dB of current A (measured safe: hw/car; measured unsafe:
  hw/trashbin) — i.e. as an opt-in throughput switch, never as the default.

**Recommendation: KEEP the current module A as the default.** A is not part of
the scored TIME (only module B is), so the S2 win is pure iteration/E2E
throughput; and it is not uniform: safe on car, −3.5 q on trashbin. If a faster
A is wanted for iteration, use **S2 stride 2** and accept the risk on
mapper-hard scenes (command below). The next thing worth trying (not done here)
is S2 + an AR-intrinsics prior: freeze the focal length at the AR-derived
`fx=685.3, fy=686.5` after registration (`--BundleAdjustment.refine_focal_length 0`),
which directly targets the measured −2.1 % focal drift on trashbin.

The exact S2 stride-2 command (candidate for an optional fast mode; not the
delivery default):

```bash
REPO=/workspace/organizer-projects/FFSplat-Hackathon
DB=$REPO/outputs/<exp>/speed/<scene>/db.db ; SUB=.../subset.txt
# 1. features for ALL images (unchanged)
colmap feature_extractor --database_path $DB --image_path $IMG \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 0
# 2. exhaustive matching INSIDE every-2nd-frame subset only (db copy, ~4x fewer pairs),
#    merge those matches back into $DB, then sequential links for the registrator
colmap exhaustive_matcher --database_path $DB_SUB --SiftMatching.use_gpu 0
colmap sequential_matcher --database_path $DB --SiftMatching.use_gpu 0 \
  --SequentialMatching.overlap 5 --SequentialMatching.quadratic_overlap 0
# 3. map the subset, register the rest, refine everything
mkdir -p $SPARSE $REG $REG_BA
colmap mapper --database_path $DB --image_path $IMG --output_path $SPARSE --image_list_path $SUB
colmap image_registrator --database_path $DB --input_path $SPARSE/0 --output_path $REG
colmap bundle_adjuster --input_path $REG --output_path $REG_BA
# 4. existing image_undistorter + module_a work-dir writer (unchanged)
```

**One-line recommendation to the collaborator:** keep the current COLMAP module A
for delivery (A is not in the scored TIME); use S2 stride 2 only as an opt-in
iteration-speed switch (full-A 1.6× on car, 2.25× on trashbin) after checking that
the scene's ReSplat context PSNR is within ~0.1 dB of its COLMAP-A value — it is
quality-neutral on hw/car but −3.5 q on hw/trashbin; do not use stride 3, S3
(AR-seeded free BA) or the AR-focal prior.

Full scripts: `outputs/pose-exp-20261030/s2b.sh` (car strides 2/3),
`s2c.sh` (trashbin), `speed_prep.sh` (S1/S0/S4), `s3_arseed.sh` (S3),
`finalize.sh` (work dirs + ctx triage + e2e). Raw rows:
`outputs/pose-exp-20261030/speed/results.jsonl`, e2e rows in
`outputs/pose-exp-20261030/p3/results.jsonl` (`"kind": "speed"`).

## 12. Raw artifact index (H1)

```
outputs/pose-exp-20261030/
  p0/A/{car,trashbin}/r{1,2,3}/          current-A work dirs + module_a.json
  p0/tmp/{car,trashbin}/r{1,2,3}/        COLMAP db + sparse + undist (kept)
  p0/stability.json                      pairwise P0 stability matrix
  p1run.sh                               queued mapper-determinism decomposition
  C/{car,trashbin}/{transavg,transavg2,rotonly}/
  p3/results.jsonl  p3/run.log           all 9 end-to-end arms
  p3/<scene>_<arm>/{resplat,sweep,cache,run.log}
  diag/{car_transavg_scaled,car_transavg_colmap_imgs}/   §7 ablations
  speed/                                 §11 task-6 speed benchmark
    results.jsonl                        S1/S0/S4/S2/S3 CPU stage timings
    s2b_stride{2,3}/ s2c_trash_stride{2,3}/   subset-SfM models + registered_ba
    work_{car,trashbin}_s2_stride{2,3}/  module-A work dirs for the e2e runs
    ctx_*/                               ReSplat context self-render PSNR
    S3/{car,trashbin}/                   AR-seeded rotations-free BA work dirs
    work_*_ark/                          AR-focal-prior negative result (§11.4)
  s2b.sh s2c.sh speed_prep.sh s1b.sh s4b.sh s3_arseed.sh s2fix.sh finalize.sh
  input_hashes.json                      dataset + per-arm input sha256
  pose_report.json                       full aggregate (stability, candidates, check)
  tools/                                 analysis helpers (pose_report.py, hash_inputs.py, diag_gpu.py)
```
Reproduction: `research/pose_eval.py` (metrics), `research/pose_rotfixed.py`
(candidate pipeline + `check` QA).
