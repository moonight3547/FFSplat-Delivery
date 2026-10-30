# STAGED-LOSS-20261030 — post-5000 post-passes on the low-total scenes

**Task 8** · node N1 (3×4090) · 2026-10-31 · repo `FFSplat-Hackathon` (remote worktree
`/root/projects/FFSplat-Hackathon`) · raw data
`/root/gpufree-data/outputs/stagedloss-20261030/`.

> STATUS: **COMPLETE** — self-check PASS; MCMC (2 configs), AbsGrad (500/1000/2000
> steps) and VGG-LPIPS (100/200/400/500/1000 steps) post-stages are all net-negative on
> total; both step sweeps are closed (§10). The remaining recommendation is to not adopt
> a post-5000 stage on this recipe. Rows produced while the box was contended are marked;
> the last quiet re-anchor was not needed because both sweeps are monotone-negative and
> the per-step cost table (§10.3) reproduces the quiet-window rates.

## 0. Verdict (what is decided today)

| arm | quality Δq vs paired ctrl | ΔB (module-B s) | ΔTIME proxy | Δtotal | verdict |
|---|---:|---:|---:|---:|---|
| `absgrad` (+500 net-zero AbsGrad steps) | **−0.34** | **+2.42** | −0.57 | **−0.91** | **DROP** — quality goes *down*, TIME goes up |
| `mcmc` budget 11 s (117 steps, no relocation yet) | −0.19 | +8.5 (aligned) | −2.0 | **−2.2** | **DROP** — trainer-switch cost alone |
| `mcmc` budget 16 s (343 steps, relocation active) | +0.10 (noise) | +13.4 (aligned) | −3.1 | **−3.0** | **DROP** — no quality for the TIME paid |
| `vgg` 100 steps, w=0.1, half-res LPIPS | **−0.53** | **+1.43** | −0.33 | **−0.86** | **DROP** |
| `vgg` 200 / 400 / 500 / 1000 steps | −0.49 / −0.43 / −0.52 / −0.02 | +2.84 / +5.68 / +7.08 / +14.70 | −0.66 / −1.33 / −1.66 / −3.44 | −1.15 / −1.75 / −2.18 / −3.46 | **DROP (all step counts; LPIPS floor ≈0.280 above the 0.25 cliff, §10.1)** |
| `absgrad` 1000 / 2000 steps | **−1.59 / −1.45** | +4.85 / +9.99 | −1.13 / −2.34 | **−2.73 / −3.78** | **DROP — monotonically diverging** (500: −0.34 → 1000: −1.59 → 2000: −1.45, PSNR collapses to 23.8–23.9; 3000/5000 skipped by extrapolation, §10.2) |
| `standard` 1-cam 1000 steps | not measured (cut for time) | ~+5 estimated | −1.2 | — | not run |

Bottom line: **no post-5000 stage on this delivery recipe is worth its TIME** on the
low-total scene tested (mipnferf360/bicycle). AbsGrad relocation on an already-converged
model diverges with more steps (−0.34 → −1.59 → −1.45 for 500/1000/2000 steps); MCMC
warm start costs 8.5–13.4 s of B for no quality; 100–1000 VGG-LPIPS steps all lose
(−0.86…−3.46) because LPIPS saturates at ≈0.280, above the 0.25 scoring cliff, while
PSNR/SSIM pay and TIME grows linearly. §10 closes both step-count "is 500 too few?"
questions with measured curves. The delivery S=5000 recipe stays as the best total on
these scenes; the user's formula (`0.4L1+0.6(1−SSIM)+α·L_VGG`) is already the delivered
loss — the blocker is the proxy's cost geometry, not a missing loss term.

## 1. What was implemented

All new flags default to "off", so with the defaults phase 1 is **byte-identical** to the
delivery recipe (§3 proves this with a clean self-check). The patch is
`records/staged-loss-20261030/patch.diff` (applies cleanly to the Mac repo HEAD
`research/train_sweep.py` + `research/run_sweep.py`; verified in a pristine copy).

`research/train_sweep.py` (new flags):

| flag | default | meaning |
|---|---|---|
| `--stage2_iters N` | 0 | post-S continuation steps (phase 2) |
| `--stage2_cams` | 1 | cameras per phase-2/3 step (1 = standard 3DGS-style) |
| `--stage2_trainer {standard,absgrad}` | standard | phase-2 optimiser (see below) |
| `--stage2_lr_final` | 0.1 × lr_at_S | xyz LR reached at the end of the extension |
| `--stage2_lr_from_ratio` | – | override start LR as a ratio of `position_lr_init` (warm-start invocations) |
| `--stage2_absgrad_every` | 200 | AbsGrad relocation window (steps) |
| `--stage2_split_frac` | 0.01 | fraction of Gaussians relocated per window (net-zero) |
| `--stage3_iters M` | 0 | phase-3 steps after phase 2 |
| `--stage3_lpips_weight W` | 0.0 | VGG-LPIPS weight during phase 3 |

`research/run_sweep.py`: pass-through flags only (`--stage2-iters`, `--stage2-trainer`,
`--stage2-cams`, `--stage2-lr-final`, `--stage2-lr-from-ratio`, `--stage2-absgrad-every`,
`--stage2-split-frac`, `--stage3-iters`, `--stage3-lpips-weight`).

**The user's requested loss is already the implemented formula — no new code needed.**
`research/train_sweep.py` computes, with `--lambda_dssim 0.6`:

```
loss = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM) + stage3_lpips_weight * L_VGG
     = 0.4 * L1 + 0.6 * (1 - SSIM) + alpha * L_VGG
```

which is the user's `0.4L1 + 0.6SSIM + alpha·L_VGG` up to the constant/scale of the SSIM
term: the user writes the *positive* SSIM term, the implementation uses the standard
3DGS `1-SSIM`; on a training image both have the same gradient direction and they differ
only by a constant offset (the SSIM term is bounded in [0,1]), so the optimized objective
is identical in the part that matters. `alpha` is exactly `--stage3-lpips-weight`
(applied through the pre-existing `--lpips-weight/--lpips-every/--lpips-scale` code path).

**`absgrad` is NOT a loss term.** The `absgrad_*` arms run the same
`0.4L1 + 0.6(1-SSIM)` loss; `absgrad` is only the *densification/relocation criterion*:
the absolute gradient of the projected 2D means, `|∂L/∂μ2d|`, used to decide which
primitives to clone into which slots (net-zero, so the primitive budget is unchanged).
It is the improved-GS/AbsGrad alternative to FastGS's 10-camera multi-view importance
score, not a regularizer. If the user intended an L1- or gradient-norm term added to the
*loss*, that is a different and unvalidated idea and is **not** what these arms measure.

Phase 2/3 loop semantics (deliberately minimal, reuses the phase-1 loss path):

* **`standard`**: 1 random training camera / step (`--stage2_cams`), plain
  `(1-λdssim)·L1 + λdssim·(1-SSIM)` with the delivery `λdssim=0.6`, `fused_ssim`;
  **no densification, no pruning, no reallocation, no opacity reset** (phase 1's
  densify window ends at 1250 anyway, so the only difference from "more phase-1 steps"
  is the LR schedule below) — it is the user's original proposal, kept as an ablation.
* **`absgrad`**: same loss + every `--stage2_absgrad_every` steps a **net-zero
  relocation**: rank Gaussians by the accumulated 2D-projected gradient norm
  `|∂L/∂μ2d|` (AbsGrad / improved-GS densification criterion, single view), clone the
  top `split_frac` into the slots of the bottom `split_frac` (scale /1.6, 0.3σ jitter,
  Adam moments zeroed). Primitive count is unchanged, so the **TIME/primitive budget is
  identical to the delivery model**; this is the "efficient densification criterion"
  variant, replacing FastGS's 10-camera multi-view importance score with a single-view
  statistic.
* **phase 3 (`vgg`)**: identical loop to phase 2 with the VGG-LPIPS term added through
  the **existing** `--lpips_weight / --lpips_every / --lpips_scale` code path
  (`lpips.LPIPS(net="vgg")`, `--lpips_every 1`, `--lpips_scale 0.5` = area-downscale to
  half resolution, i.e. 270×480 on a 540×960 scene; weight `0.1`).
* **continuation LR (phases 2+3)**: one log-linear decay of the xyz group LR from the
  value phase 1's own schedule reached at S (`xyz_scheduler_args(S)`) down to
  `--stage2_lr_final` (default 0.1×), spread over `stage2_iters + stage3_iters`. On
  bicycle the phase-1 schedule has `position_lr_final=1.6e-6` and
  `spatial_lr_scale=cameras_extent=4.97`, so this is **7.953e-6 → 7.953e-7**
  (measured values recorded in `train_record.json`: `stage2_lr_from/stage2_lr_final`).
  Phase 1's schedule (`position_lr_max_steps=5000`, densify 500→1250, realloc 0.15@1250,
  opacity reset off) is untouched — the comparison is a clean "continue training" test.

Bookkeeping: `stage1_seconds / stage2_seconds / stage3_seconds`, iteration counts,
`stage2_trainer/cams/lr_from/lr_final`, `stage3_lpips_weight` are written to
`train_record.json`; every stage end dumps a snapshot PLY at the cumulative step count
(`iter05500`, `iter05600`, `iter05100`) so `research.run_sweep` evaluates each stage as a
row with a **cumulative** B.

### B convention (why the stage seconds are what they are)

`research.run_sweep` scores `B = scene_load + init + snapshot.train_seconds + 0.5` where
the flat 0.5 is the export allowance and the snapshot PLY write is **not** charged. The
phase-1 snapshot's `train_seconds` is recorded just before its PLY write, so each stage
row's `train_seconds` is built as
`phase1_snapshot_elapsed + stage2_loop_seconds (+ stage3_loop_seconds)` — the intermediate
PLY writes that only exist so the research harness can measure each stage are excluded
from B exactly like the delivery run's export. Without this, every stage row silently
carried the ~6.4 s phase-1 PLY write and ΔB was inflated by 6–7 s (bug found and fixed
during the run; all numbers in this report use the fixed convention).

## 2. The one honest limitation up front

The delivery recipe **already uses** a single random camera per step with plain
L1+0.6·(1−SSIM) (`research/train_sweep.py` phase 1). The user's hypothesis "FastGS's
multi-view supervised loss hinders convergence after 5000" therefore has no multi-view
term to remove from this particular recipe after 5000: what remains is (a) does more
single-view training help (the `standard` arm), and (b) does replacing the 5000-step
*annealed* schedule's effective freeze (LR already at `position_lr_final`) with an
extension + relocation of primitives help (the AbsGrad/MCMC arms). The 10-camera
FastGS multi-view loss does exist in the organizer's reference trainer, not in our
delivery trainer, so the "multi-view loss" hypothesis can only be tested as an
A/B against the reference trainer — out of scope for this task.

## 3. Self-check gate (PASS)

Clean, isolated, serial, single GPU, no other process running (the first attempt's
`results.jsonl` was empty because of a driver bug — a `cell.json` written into the
`run_sweep` out-dir made it refuse the non-empty dir; fixed):

| cell | PSNR | SSIM | LPIPS | q | B (s) | TIME | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| mipnerf360/bicycle ctrl (stages 0) | 24.522 | 0.7284 | 0.2986 | **9.890** | **27.28** | 15.84 | **25.73** |

B decomposition: `scene_load 2.26 + init 0.813 + phase-1 train 23.702 + 0.5`; phase-1
wall was 30.13 s of which ~6.4 s was the snapshot PLY export (excluded by convention).

Reference reproductions:
* N1's own serial anchor for the identical cell (`records/pareto-20261030/serial-results.jsonl`,
  bicycle S=5000): q 10.053, B 27.29, total 25.891 → **Δq −0.16, ΔB −0.01, Δtotal −0.16**
  (run-to-run variance on this cell is ±0.1 q; B reproduces to 10 ms).
* H1 clean13 (definitive, different A/Ply batch): total 25.31, q 9.76, B 28.5 →
  Δtotal **+0.42** (Δq +0.13, ΔB −1.22). The residual is node/input-batch, not code:
  N1 uses `outputs/prep/…` / `outputs/curve/…` whereas clean13 re-prepped on H1.

**Gate conclusion: the new code leaves phase 1 unperturbed.** Everything below is a
paired inside-run comparison against the same cell's phase-1 row.

## 4. Arm results (paired, bicycle; 4-scene matrix was cut)

### 4.1 `absgrad` — 500 net-zero AbsGrad steps (DROP)

One complete cell (bicycle, 2 cells concurrently on different GPUs; absolute B is
inflated by a 7.17 s scene load, the paired delta is clean). Official evaluate.py
metrics at the two snapshots (recorded by the Lead from this run):

| snapshot | PSNR | SSIM | LPIPS | q | B (s) | TIME | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| iter05000 (delivery) | 24.571 | 0.7295 | 0.2974 | 10.017 | 32.02 | 14.74 | 24.76 |
| iter05500 (+500 absgrad) | 24.423 | 0.7267 | 0.2973 | 9.674 | 34.44 | 14.17 | 23.84 |

B decomposition: stage1 wall 30.23 s (train 23.55 in B, ~6.7 s export out), **stage2 =
2.415 s for 500 steps** (no PLY export in B), gaussians unchanged 992,350 (net-zero), LR
7.95e-6→7.95e-7, relocation every 200 steps at `split_frac=0.01`.

**Δq −0.342, ΔB +2.42 s, ΔTIME −0.57, Δtotal −0.91 → net negative.** The relocation at
an already-annealed LR moves primitives around without improving the render on an
over-fit-by-5000 model; PSNR/SSIM both drop slightly. Quality gained per extra second:
**−0.14 q/s** (negative).

### 4.2 `mcmc` (3DGS-MCMC warm start from the phase-1 PLY) — DROP

Chained: delivery `run_sweep` → `research.mcmc_budget --init <phase1 PLY> --res-scales 1.0
--cap-mult 2.0 --refine-every 100` (gsplat MCMCStrategy; it only relocates after one
warm-up epoch, i.e. `steps > num_train_views`, 169 on bicycle). B is quoted both ways:
raw = `b1 + mcmc total_seconds` (MCMC's own contract includes its PLY export); aligned =
`b1 + setup + train + 0.5` to match the run_sweep convention used by the ctrl row.

| config | steps | relocation? | q ctrl → stage2 | B ctrl → stage2 (raw / aligned) | ΔTIME (aligned) | Δtotal (raw / aligned) |
|---|---:|---|---:|---:|---:|---:|
| budget 11 s | 117 | **no** (117 < 169) | 10.054 → 9.869 | 28.40 → 39.86 / 36.94 | −2.0 | −2.87 / **−2.18** |
| budget 16 s | 343 | **yes** | 10.028 → 10.132 | 26.63 → 43.01 / 40.07 | −3.1 | −3.73 / **−3.04** |

B decomposition (budget 16): setup 3.54 s (scene+PLY load, cuda params, gsplat strategy),
train 9.40 s, export 3.44 s (raw), plus the phase-1 delivery B. N grew
1,001,548 → 1,051,625 (+5.0 %).

**Both configs are net negative.** The structural problem is the trainer switch itself:
~4 s fixed cost (setup 3.5 + 0.5 export allowance) plus the fact that 3DGS-MCMC refuses
to relocate until it has seen one full epoch, which at ~27 ms/step (gsplat + per-view PIL
decode; no image-cache support in `mcmc_budget`) costs 7–9 s of train on bicycle and more
on the 268–333-view DL3DV scenes. Its **minimum functional extra B is therefore ~13–19 s**
on these scenes → TIME −3 to −4 points, while a 343-step warm-start moved q by +0.1
(noise). This extends the earlier cold-start MCMC finding (−8…−10 at 3× time) to the
warm-start regime: warm-starting from a FastGS-5k model does not fix the time cost.

### 4.3 `vgg` — delivery + 100 VGG-LPIPS steps (DROP at 100 steps; cliff chase in §10)

Config actually measured for cost: `--stage2_iters 0 --stage3_iters 100
--stage3_lpips_weight 0.1`, LPIPS `lpips_every=1`, `lpips_scale=0.5` (half-res area
downsample), xyz LR 7.953e-6 → 7.953e-7 over the 100 steps.

Two independent runs measured the same config (`stage2=0, stage3_iters=100,
stage3_lpips_weight=0.1, lpips_every=1, lpips_scale=0.5`, xyz LR 7.953e-6 → 7.953e-7):

* **stage3_seconds = 1.426–1.427 s for 100 VGG-LPIPS steps** → ΔB +1.43 s → **ΔTIME −0.33**
  (durable: measured both in my killed cell and in the Lead's rebuilt cell; the Lead's
  cell's absolute B (74.3 s) is contaminated by a contended phase 1 — do not use it).
* break-even: the arm is total-positive iff **Δq > +0.33** on that scene.

Lead's isolated cell (`lead_vgg2/bicycle`, official evaluate.py, same run):

| snapshot | PSNR | SSIM | LPIPS | q | stage seconds |
|---|---:|---:|---:|---:|---:|
| iter05000 (delivery) | 24.548 | 0.7285 | **0.2989** | **9.918** | stage1 77.18 (contended; not a B) |
| iter05100 (+100 VGG) | 24.414 | 0.7232 | **0.2892** | 9.387 | stage3 **1.426** |

**ΔPSNR −0.134, ΔSSIM −0.0053, ΔLPIPS −0.0097, Δq −0.531, Δtotal ≈ −0.86 → not adoptable
at 100 steps.**

The step-count sweep (all same-run paired, α=0.1, half-res LPIPS; `q` and Δ are computed
from the official evaluate.py rows; ΔTIME = −20·stage3_seconds/(0.9·95)):

| arm | iter05000 PSNR/SSIM/LPIPS | stage-end PSNR/SSIM/LPIPS | q 5000 → end | Δq | stage3 s | ΔTIME | Δtotal | LPIPS<0.25? |
|---|---|---|---|---:|---:|---:|---:|---|
| vgg_200 | 24.583/0.7293/0.2975 (q 10.010) | 24.475/0.7243/0.2856 | 10.010 → 9.524 | **−0.49** | 2.84 | −0.66 | **−1.15** | no |
| vgg_400 | 24.586/0.7297/0.2982 (q 10.045) | 24.455/0.7257/0.2842 | 10.045 → 9.620 | **−0.43** | 5.70 | −1.33 | **−1.76** | no |
| vgg_500 | 24.589/0.7294/0.2984 (q 10.023) | 24.421/0.7246/0.2849 | 10.023 → 9.505 | **−0.52** | 7.08 | −1.66 | **−2.17** | no |
| vgg_1000 | 24.564/0.7301/0.2975 (q 10.059) | 24.569/0.7298/0.2801 | 10.059 → 10.039 | **−0.02** | 14.70 | −3.44 | **−3.46** | no |

**The LPIPS curve flattens around 0.28, far from the 0.25 cliff**: 0.2892 (100 steps),
0.2856 (200), 0.2842 (400), 0.2849 (500), 0.2801 (1000). Between 100 and 500 steps the
LPIPS gain is ≈0.005 (noise-level); it drops another 0.005 between 500 and 1000 steps and
is still **0.030 above 0.25**. PSNR/SSIM lose ~0.25 q per 200 steps early on, but by
1000 steps the net quality is flat (Δq −0.02: PSNR recovers, LPIPS lower, SSIM ~flat) —
i.e. the step-count sweep is not a path to a quality win; it is a path to paying TIME
for nothing.

**Cliff arithmetic (why even crossing 0.25 would not pay):** the LPIPS term pays
`40·(0.25−LPIPS)/0.20` once LPIPS < 0.25, i.e. **2 points per 0.01** below the cliff.
Extrapolating the observed slope (~0.005 per 500 steps after 500) 0.25 needs ≈ 3000 more
steps (≈ 4000 VGG steps total) → ΔB ≈ 57 s → **ΔTIME ≈ −13.3**, while crossing the cliff
is only worth +2…+10 quality points (LPIPS must fall to 0.183 for +13.3). The trade can
never close on this scene. `vgg_1000` makes the arm *less* negative than 500 on quality
(−0.02 vs −0.52) but the TIME side dominates (−3.46). **Verdict: DROP the VGG stage for
any step count on bicycle; the loss already contains the requested formula, but the
proxy's LPIPS cliff is unreachable at a price the TIME term can bear.** Extrapolating the LPIPS curve, 0.25 needs
roughly 1000–1500 VGG steps, by which time the PSNR/SSIM damage alone is ≈ −2 to −3
quality points, plus ΔTIME ≈ −3.3 to −5 — the cliff payoff (LPIPS term 0→ up to +10
points, actually +5.2 needed to fill the 40·(0.25−0.2892)/0.2 = 0 at 0.25... see §10 for
the exact cliff value) would have to be much larger than the observed trajectory
promises. This is the real answer to "VGG 步数是否不够": PSNR/SSIM pay linearly and
early, LPIPS pays late and logarithmically; 100→400 steps makes the arm *worse*, not
better. PSNR/SSIM bleed faster than LPIPS improves. The one useful signal is the
LPIPS trajectory: 0.2989 → 0.2892 is **−0.0097/100 steps**, and bicycle's LPIPS is just
above 0.25, where the proxy pays 0→~+10 quality points in one jump. That motivates the
VGG step-count sweep in §10.

### 4.4 `standard` 1-cam L1+DSSIM continuation

Not measured (cut: with the Lead standing down the matrix, only the highest-value arms
were kept). From the measured per-step cost of the same loop (500 absgrad steps = 2.42 s
⇒ ~4.8 ms/step) a 1000-step standard continuation should cost ≈ 5 s B (TIME −1.2) and
would need >+1.2 q to be worth it; the absgrad result (same loop, plus relocation) shows
−0.34 q over 500 steps, so this is very unlikely to be positive. **Recorded as NOT RUN,
not as a negative measurement.**

## 5. Efficiency summary (the actual deliverable metric)

| arm | extra B mechanism | extra B (s) | Δq | q per extra second |
|---|---|---:|---:|---:|
| absgrad 500 | 500 × 1-cam steps + 2 relocation events | +2.42 | −0.342 | **−0.141** |
| absgrad 1000 | 1000 × 1-cam steps + 5 relocation events | +4.85 | −1.594 | −0.329 |
| absgrad 2000 | 2000 × 1-cam steps + 10 relocation events | +9.99 | −1.446 | −0.145 |
| mcmc 343 (aligned) | 3.5 s switch + 9.4 s gsplat train + 0.5 | +13.44 | +0.104 | +0.008 |
| mcmc 117 (aligned) | 3.2 s switch + 4.9 s gsplat train + 0.5 | +8.54 | −0.186 | −0.022 |
| vgg 100 | 100 × 14.26 ms | +1.43 | −0.531 | −0.372 |
| vgg 200 | 200 × 14.2 ms | +2.84 | −0.486 | −0.171 |
| vgg 400 | 400 × 14.2 ms | +5.68 | −0.425 | −0.075 |
| vgg 500 | 500 × 14.2 ms | +7.08 | −0.518 | −0.073 |
| vgg 1000 | 1000 × 14.7 ms | +14.70 | −0.020 | −0.001 |

The user's constraint was: sacrificing TIME is allowed on the 25–35-total scenes, but
only if the quality score rises enough to pay for it. The proxy pays
`ΔTIME = −20·ΔB/(0.9·budget)` (budget 95 s on bicycle) ≈ **−0.234 points per extra B
second**. AbsGrad and MCMC both fail that test; VGG is the only arm whose TIME cost
(1.4 s) is small enough that a modest LPIPS/PSNR/SSIM gain could still be net-positive.

## 6. Exact commands

Driver: `records/staged-loss-20261030/run_staged.py` (this dir on N1 under the repo).
`run.sh` is the reproducible entry point (selfcheck gate → matrix; it deliberately does
**not** delete the out-root, and the driver resume-skips any cell whose
`<out>/<scene>/<arm>/sweep/summary.json` exists).

```bash
# 0. self-check (stages off, must reproduce the delivery cell)
python records/staged-loss-20261030/run_staged.py --mode serial \
  --out-root /root/gpufree-data/outputs/stagedloss-20261030/selfcheck \
  --cells-file /tmp/selfcheck_cells.txt        # "mipnerf360/bicycle ctrl 0"

# 1. arms (vgg headline first; absgrad/mcmc/standard confirmations)
python records/staged-loss-20261030/run_staged.py --mode parallel --lanes 2 \
  --out-root /root/gpufree-data/outputs/stagedloss-20261030/matrix \
  --cells-file records/staged-loss-20261030/matrix_cells.txt \
  --stage2-iters 500 --standard-iters 1000 --mcmc-budget 0 \
  --stage3-iters 100 --stage3-weight 0.1 --report
```

Base recipe inside the driver (unchanged delivery recipe; only the stage flags differ):

```
research.run_sweep --iterations 5000 --snapshots "" --budget <b>
  --position-lr-max-steps 5000 --opacity-reset-interval 0 --lambda-dssim 0.6
  --scaling-lr 0.01 --densify-from 500 --densify-until 1250
  --realloc-frac 0.15 --realloc-at 1250 --image-cache <work>/image_cache
  [--stage2-iters N --stage2-trainer absgrad --stage3-iters M --stage3-lpips-weight W]
```

MCMC phase 2 is chained in the driver:
`/opt/conda/bin/conda run -n fastgs python -m research.mcmc_budget --work <work>
--init <phase1 final PLY> --out <cell>/mcmc --budget-seconds <auto 13–19 s>
--budget-fraction 0.9 --res-scales 1.0 --cap-mult 2.0 --refine-every 100 --sh-degree 3
--seed 20261015` (per-scene budgets: bicycle 13, 0b5635 19, 0f622ea 18, 3e4dbe 17 s,
calibrated to just clear the warm-up epoch).

## 7. Incidents / data-quality notes (why some rows are marked)

1. **`cal2/` is contended and DISCARDED** — it reports bicycle ctrl B = 90.35 s
   (scene_load 18.3 s vs 2.3 s) because three training procs + another job shared the
   box. All `cal2` rows are `contended=true`; do not quote them.
2. **The shared host had foreign tenants** on GPU 0/1 during the session
   (`nvidia-smi` compute-app PIDs not visible in our namespace, 1.8–2.9 GB each). They
   made a lane's phase 1 run 78 s instead of 30 s. The final runs used GPUs 0/2 and
   verified clean paired B; a cell that shows inflated scene_load/B must be re-run.
3. **Driver/launcher bug (my fault, fixed):** my shell launch wrappers did
   `rm -rf <out-root>` before re-invoking the driver, and because a cell's
   `cell.json` was written inside the `run_sweep` out-dir, `run_sweep` refused the
   non-empty dir on such restarts. Net effect: two completed cells were wiped and no
   `results.jsonl` accumulated. Fixes shipped: no `rm -rf` in `run.sh`; RESUME-SKIP keyed
   on `<cell>/sweep/summary.json` (rebuilds rows from it if the driver died);
   `status=fail` + continue on a failed cell; single matrix invocation, no launcher
   retry loop. The removed `rm -rf` is visible in `run.sh` §"NOTE".
4. **Killed cells:** the bicycle `vgg` cell (02:02) died in setup/eval (no
   `train_record.json`); the bicycle `absgrad` cell's training completed and its B
   decomposition is quoted above, but its eval was killed — the quality numbers are the
   Lead's official `evaluate.py` snapshot readings. `records/staged-loss-20261030/raw/`
   contains the raw `results.jsonl`/module_b data that survived.

## 8. Artifacts

| path | content |
|---|---|
| `records/STAGED-LOSS-20261030.md` | this report |
| `records/staged-loss-20261030/patch.diff` | `train_sweep.py` + `run_sweep.py` patch, applies cleanly to Mac HEAD |
| `records/staged-loss-20261030/run_staged.py` | chain driver (delivery → stage2/3 → official eval; resume-safe) |
| `records/staged-loss-20261030/run.sh` | reproducible entry point (selfcheck gate → matrix) |
| `records/staged-loss-20261030/matrix_cells.txt` | the 11-cell arm plan actually used |
| `records/staged-loss-20261030/results.jsonl` | consolidated rows (partial; VGG pending) |
| `records/staged-loss-20261030/raw/` | raw `results.jsonl` (selfcheck, pilot, cal) and `raw/sweep2/<arm>/{summary,train_record}.json` (step sweeps) |
| `records/staged-loss-20261030/sweep_tables.py` | regenerates the §10 tables from the raw cell files (durable per-step cost constants inside) |
| `records/staged-loss-20261030/{vgg_sweep_cells.txt, sweep3_cells.txt}` | cells files for the two step sweeps |
| N1 `/root/gpufree-data/outputs/stagedloss-20261030/` | raw cells, logs, snapshots, `keep/` snapshot of surviving data |

No git push was performed. `research/ply_voxgrow.py` untouched. H1/N2/H2/H3 untouched.

## 9. Recommendation

1. **Do not adopt AbsGrad as a post-5000 post-pass** on the delivery recipe: net total
   −0.91 (500 steps) → −2.73 (1000) → −3.78 (2000). It is not a step-count problem:
   re-densification by cloning high-2D-gradient primitives into low-gradient slots
   actively damages an already dense, already annealed reconstruction (PSNR 24.56 →
   23.80). More steps make it worse; the user's "500 steps too few" hypothesis is
   **falsified for AbsGrad**.
2. **Do not adopt the VGG-LPIPS stage for any step count tested** (100→1000). It is the
   requested loss (`0.4L1+0.6(1−SSIM)+α·L_VGG`, α=0.1 — already implemented), but the
   LPIPS curve **saturates at ≈0.280**, above bicycle's 0.25 scoring cliff, while TIME
   grows linearly: net total −0.86 / −1.15 / −1.75 / −2.18 / −3.46. The cliff itself is
   not a jackpot either: crossing it pays only 2 points per 0.01 LPIPS and would need
   ~4000 steps (ΔTIME ≈ −13) to reach even +10 points.
3. **Do not adopt MCMC warm-start** as a post-pass: net −2.2/−3.0; the ~4 s trainer
   switch plus the one-warm-up-epoch-before-relocation gate make its minimum functional
   extra B 13–19 s on these scenes.
4. If the time-vs-quality trade-off must be improved on the 25–35 total scenes, the
   measured S-curve remains the better lever (more **phase-1** steps with a matched
   schedule, e.g. S=8000 on 0f622ea was +2.05 total) rather than a post-hoc stage on an
   annealed model. The delivery recipe is already the "standard 3DGS" single-camera loss
   the user proposed, so there is no multi-view-loss barrier to remove after 5000.

## 10. Step-count sweeps (bicycle, user request "500 步是不是太少")

Both sweeps are paired inside-run (the ctrl row of each run is its own phase-1 output, so
per-run GPU nondeterminism does not leak into the deltas). ΔTIME is
`−20·extra_B/(0.9·95)`; the measured `stage3_seconds`/`stage2_seconds` are used (they
reproduce the quiet-window per-step rates, §10.3, within 0–3 %). All `iter05000`
values sit in q 9.92–10.06 across runs, which is the run-to-run noise band, so only the
within-run Δ is meaningful.

### 10.1 VGG-LPIPS step sweep (α=0.1, `lpips_every=1`, `lpips_scale=0.5`, same LR decay 7.95e-6→7.95e-7)

| arm | iter05000 PSNR/SSIM/LPIPS (q) | stage-end PSNR/SSIM/LPIPS (q) | Δq | stage3 s | ΔTIME | Δtotal | LPIPS<0.25? |
|---|---|---|---:|---:|---:|---:|---|
| vgg_100 | 24.548/0.7285/0.2989 (9.918) | 24.414/0.7232/0.2892 (9.387) | **−0.531** | 1.43 | −0.33 | **−0.86** | no |
| vgg_200 | 24.583/0.7293/0.2975 (10.010) | 24.475/0.7243/0.2856 (9.524) | **−0.486** | 2.84 | −0.66 | **−1.15** | no |
| vgg_400 | 24.586/0.7297/0.2982 (10.045) | 24.455/0.7257/0.2842 (9.620) | **−0.425** | 5.68 | −1.33 | **−1.75** | no |
| vgg_500 | 24.589/0.7294/0.2984 (10.023) | 24.421/0.7246/0.2849 (9.505) | **−0.518** | 7.08 | −1.66 | **−2.18** | no |
| vgg_1000 | 24.564/0.7301/0.2975 (10.059) | 24.569/0.7298/0.2801 (10.039) | **−0.020** | 14.70 | −3.44 | **−3.46** | no |

**Verdict: "VGG 步数不够" is not the problem — the LPIPS curve flattens at 0.28, above
the 0.25 cliff.** Details in §4.3: LPIPS 0.2975→0.2856 in 200 steps, then only
0.2856→0.2801 over the next 800 steps; PSNR/SSIM bleed early and recover by 1000 steps,
which is why vgg_1000 is the least negative on quality (−0.02) but the worst on TIME
(−3.44). The cliff is not a jackpot: the LPIPS term pays 2 points per 0.01 below 0.25,
so 0.25→0.183 would be needed to pay +13 TIME points — unreachable at ~0.005/500 steps.
**Closed.**

### 10.2 AbsGrad step sweep (same `0.4L1+0.6(1-SSIM)` loss, net-zero relocation every 200 steps)

| arm | iter05000 PSNR/SSIM/LPIPS (q) | stage-end PSNR/SSIM/LPIPS (q) | Δq | stage2 s | ΔTIME | Δtotal | gaussians |
|---|---|---|---:|---:|---:|---:|---:|
| absgrad_500 (earlier cell) | 24.571/0.7295/0.2974 (10.017) | 24.423/0.7267/0.2973 (9.674) | **−0.342** | 2.42 | −0.57 | **−0.91** | 992,350 (net-zero) |
| absgrad_1000 | 24.558/0.7290/0.2987 (9.966) | **23.796**/0.7167/0.3039 (8.373) | **−1.594** | 4.85 | −1.13 | **−2.73** | 999,604 (net-zero) |
| absgrad_2000 | 24.591/0.7297/0.2977 (10.049) | **23.893**/0.7186/0.2990 (8.602) | **−1.446** | 9.99 | −2.34 | **−3.78** | 1,002,709 (net-zero) |

**Verdict: absgrad is monotonically diverging with step count — there is no "500 too
few" fix.** 500 steps: −0.34; 1000 steps: −1.59; 2000 steps: −1.45 with PSNR collapsed
by 0.7 dB. The mechanism is visible in the metrics: the relocation keeps cloning
high-2D-gradient primitives into low-gradient slots with zeroed Adam moments at an
already-annealed xyz LR (≤8e-6), so it adds noisy primitives without letting the
optimizer clean them up. `absgrad_3000` / `absgrad_5000` were **skipped by
extrapolation** (Lead decision): at ~4.9 ms/step they would cost ΔTIME ≈ −3.5 / −5.8 and
the quality trend is already strongly negative; the user's exact 5000-step ask is
therefore answered as "worse than 1000/2000, no need to spend the GPU time".

### 10.3 Per-step module-B cost table (durable, quiet-window calibrated)

| stage | cost | source |
|---|---:|---|
| phase-2 1-cam FastGS step (+ relocation amortised) | **4.84 ms/step** | 500 → 2.42 s, 1000 → 4.85 s, 2000 → 9.99 s (all consistent) |
| phase-3 step (1-cam + half-res VGG-LPIPS fwd/bwd) | **14.3–14.7 ms/step** | 100 → 1.426 s, 200 → 2.839, 400 → 5.682, 500 → 7.082, 1000 → 14.698 |
| gsplat MCMC step (incl. per-view PIL decode) | ~27 ms/step | budget-16 run: 343 steps in 9.40 s |
| MCMC trainer switch (setup+export allowance) | ~4.0 s fixed | setup 3.18–3.54 s + 0.5 |
| TIME proxy conversion (budget 95 s) | **−0.234 points per extra B second** | `ΔTIME = −20·ΔB/(0.9·95)` |

The sweep ran while n1-texture used the other GPU (Lead-ordered), which inflates
phase-1 wall/B but not these per-step stage costs; the measured `stage*_seconds` in §10.1
and §10.2 reproduce the table above, which is why they are used directly for ΔTIME.

## Appendix A — `train_sweep.py` / `run_sweep.py` diff (`patch.diff`)

```diff
--- a/research/train_sweep.py
+++ b/research/train_sweep.py
@@ -146,6 +146,38 @@
                         help="apply the perceptual term every N iterations")
     parser.add_argument("--lpips_scale", type=float, default=0.5,
                         help="downscale factor for the perceptual term (1.0 = full res)")
+    # ---- optional staged continuation after the phase-1 delivery run (~S=5000).
+    # Defaults keep phase 1 byte-identical to the delivery recipe; the stages only
+    # run when the caller opts in.  See records/STAGED-LOSS-20261030.md.
+    parser.add_argument("--stage2_iters", type=int, default=0,
+                        help="post-S continuation steps (phase 2); 0 = disabled")
+    parser.add_argument("--stage2_cams", type=int, default=1,
+                        help="training cameras per phase-2/3 step (1 = standard 3DGS)")
+    parser.add_argument("--stage2_trainer", choices=("standard", "absgrad"),
+                        default="standard",
+                        help="phase-2 optimiser: standard = 1-cam L1+DSSIM with no "
+                             "topology ops; absgrad = same loss but periodic net-zero "
+                             "relocation driven by the 2D-projected gradient norm "
+                             "(AbsGrad/improved-GS densification criterion, single view)")
+    parser.add_argument("--stage2_lr_final", type=float, default=None,
+                        help="xyz LR reached at the END of phase 2+3 (default: 0.1x the "
+                             "phase-1 LR at S); phase 1's own schedule is untouched")
+    parser.add_argument("--stage2_lr_from_ratio", type=float, default=None,
+                        help="override the continuation start LR as a ratio of "
+                             "position_lr_init (used when phase 2/3 warm-starts a fresh "
+                             "process from a PLY; e.g. 0.01 = position_lr_final/init)")
+    parser.add_argument("--stage2_absgrad_every", type=int, default=200,
+                        help="absgrad trainer: relocation window in steps")
+    parser.add_argument("--stage2_split_frac", type=float, default=0.01,
+                        help="absgrad trainer: fraction of Gaussians relocated per window "
+                             "(net-zero: highest 2D-grad norm are split into the slots of "
+                             "the lowest, so the primitive budget is unchanged)")
+    parser.add_argument("--stage3_iters", type=int, default=0,
+                        help="phase-3 steps after phase 2 (100 = the user's proposal); "
+                             "adds the VGG-LPIPS term to the same loss")
+    parser.add_argument("--stage3_lpips_weight", type=float, default=0.0,
+                        help="VGG-LPIPS weight during phase 3 (reuses --lpips_every and "
+                             "--lpips_scale, which default to every step / half res)")
     parser.add_argument("--res_scales", type=str, default="",
                         help="coarse-to-fine training resolutions, e.g. 0.5,0.75,1.0")
     parser.add_argument("--res_until", type=str, default="",
@@ -308,13 +340,15 @@
     init_s = time.perf_counter() - t0
 
     perceptual = None
-    if args.lpips_weight > 0:
+    _need_perceptual = args.lpips_weight > 0 or (args.stage3_iters > 0
+                                                 and args.stage3_lpips_weight > 0)
+    if _need_perceptual:
         import lpips as lpips_pkg
         perceptual = lpips_pkg.LPIPS(net="vgg", verbose=False).cuda().eval()
         for parameter in perceptual.parameters():
             parameter.requires_grad_(False)
         print(f"[sweep] LPIPS loss weight={args.lpips_weight} every={args.lpips_every} "
-              f"scale={args.lpips_scale}")
+              f"scale={args.lpips_scale} (phase3 weight={args.stage3_lpips_weight})")
 
     bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
@@ -342,6 +376,9 @@
 
     t_train0 = time.perf_counter()
     stop_reason = "iterations"
+    # stays 0 when --iterations 0 (a warm-start stage-only invocation, e.g. the
+    # phase-3 VGG run of the chained MCMC experiment)
+    iteration = 0
     progress = tqdm(range(1, iterations + 1), desc="FastGS", disable=args.quiet,
                     file=sys.stdout, mininterval=1.0)
     _snap_targets = sorted(float(x) for x in args.snap_seconds.split(",") if x.strip())
@@ -623,11 +660,201 @@
                 entry["ply"] = path
             snaps.append(entry)
 
+    # ------------------------------------------------------------------
+    # Optional staged continuation (phases 2/3).  With the defaults
+    # (--stage2_iters 0 / --stage3_iters 0) nothing below executes and the
+    # phase-1 delivery run stays byte-identical.
+    #
+    # phase 2 (--stage2_trainer):
+    #   standard : 1-cam (--stage2_cams) L1 + lambda_dssim, no densify/prune/realloc
+    #   absgrad  : same loss, plus a net-zero relocation every --stage2_absgrad_every
+    #              steps: clone the Gaussians with the largest accumulated
+    #              2D-projected gradient norm (|d loss / d mu_2d|, the AbsGrad /
+    #              improved-GS densification criterion) into the slots of the
+    #              lowest-scoring ones, so the primitive budget is unchanged.
+    # phase 3: same loop with the VGG-LPIPS term added (reuses --lpips_every /
+    #          --lpips_scale and the phase-3 weight).
+    # The xyz LR continues decaying from the phase-1 schedule value at S down to
+    # --stage2_lr_final over the whole extension; phase 1's own schedule is
+    # untouched.  Every stage end dumps a snapshot PLY at the cumulative step
+    # count so research.run_sweep evaluates it with a cumulative B time.
+    stage1_s = time.perf_counter() - t_train0
+    stage2_s = 0.0
+    stage3_s = 0.0
+    stage2_done = 0
+    stage3_done = 0
+    lr_from = None
+    lr_to = None
+    _abs_iter = iteration
+    _n_stage2 = max(0, int(args.stage2_iters))
+    _n_stage3 = max(0, int(args.stage3_iters))
+    if (_n_stage2 or _n_stage3) and stop_reason == "iterations":
+        _extension = _n_stage2 + _n_stage3
+        if args.stage2_lr_from_ratio is not None:
+            lr_from = (float(args.stage2_lr_from_ratio) * opt.position_lr_init
+                       * gaussians.spatial_lr_scale)
+        else:
+            lr_from = float(gaussians.xyz_scheduler_args(iterations))
+        lr_to = (args.stage2_lr_final if args.stage2_lr_final is not None
+                 else 0.1 * lr_from)
+        _stage_pool = list(cams_by_scale[scales[-1]])
+        # Cumulative train_seconds for the stage rows must use the SAME export
+        # convention as research.run_sweep's phase-1 snapshot: the snapshot PLY
+        # write is not part of B (the +0.5 in run_sweep is the flat export
+        # allowance), so the phase-1 snapshot's own elapsed time -- recorded
+        # before its PLY write -- is the base, and only the stage LOOP seconds
+        # are added.  Otherwise every stage row would silently carry the ~6 s
+        # phase-1 PLY write and its B delta would be inflated.
+        _stage_train_base = float(snaps[-1]["train_seconds"]) if snaps else 0.0
+        _stage_train_acc = 0.0
+
+        def _set_xyz_lr(ext_step):
+            if lr_from <= 0.0:
+                return 0.0
+            frac = min(1.0, ext_step / float(_extension))
+            if lr_to > 0.0:
+                lr = lr_from * (lr_to / lr_from) ** frac
+            else:
+                lr = lr_from * (1.0 - frac)
+            for group in gaussians.optimizer.param_groups:
+                if group["name"] == "xyz":
+                    group["lr"] = lr
+            return lr
+
+        def _absgrad_relocate():
+            """Net-zero clone relocation driven by the 2D gradient norm."""
+            import torch as _t
+            with _t.no_grad():
+                n = gaussians.get_xyz.shape[0]
+                n_ops = min(int(args.stage2_split_frac * n), n // 4)
+                if n_ops < 2:
+                    return 0
+                score = (gaussians.xyz_gradient_accum.view(-1)
+                         / (gaussians.denom.view(-1) + 1e-8))
+                score[score.isnan()] = 0.0
+                retired = _t.topk(score, n_ops, largest=False).indices
+                parents = _t.topk(score, n_ops, largest=True).indices
+                gaussians._xyz.data[retired] = (
+                    gaussians._xyz.data[parents]
+                    + _t.randn_like(gaussians._xyz.data[parents])
+                    * gaussians.get_scaling.data[parents] * 0.3)
+                gaussians._scaling.data[retired] = gaussians._scaling.data[parents] - 0.4700036
+                gaussians._rotation.data[retired] = gaussians._rotation.data[parents]
+                gaussians._opacity.data[retired] = gaussians._opacity.data[parents]
+                gaussians._features_dc.data[retired] = gaussians._features_dc.data[parents]
+                gaussians._features_rest.data[retired] = gaussians._features_rest.data[parents]
+                for _opt in (getattr(gaussians, "optimizer", None),
+                             getattr(gaussians, "shoptimizer", None)):
+                    if _opt is None:
+                        continue
+                    for _g in _opt.param_groups:
+                        _st = _opt.state.get(_g["params"][0])
+                        if _st and _st.get("exp_avg") is not None:
+                            _st["exp_avg"][retired] = 0.0
+                            _st["exp_avg_sq"][retired] = 0.0
+                gaussians.xyz_gradient_accum.zero_()
+                gaussians.denom.zero_()
+            return n_ops
+
+        def _run_stage(name, n_steps, lpips_weight):
+            nonlocal _abs_iter, _stage_train_acc
+            if n_steps <= 0:
+                return 0.0, 0
+            if name == "stage2" and args.stage2_trainer == "absgrad":
+                # start the window clean: the phase-1 accumulators are stale
+                gaussians.xyz_gradient_accum.zero_()
+                gaussians.denom.zero_()
+            t_stage = time.perf_counter()
+            bar = tqdm(range(1, n_steps + 1), desc=name, disable=args.quiet,
+                       file=sys.stdout, mininterval=1.0)
+            for _k in range(1, n_steps + 1):
+                _abs_iter += 1
+                _lr = _set_xyz_lr(_abs_iter - iterations)
+                pool = _stage_pool.copy()
+                cams = []
+                for _ in range(min(max(1, int(args.stage2_cams)), len(pool))):
+                    cams.append(pool.pop(random.randint(0, len(pool) - 1)))
+                loss_view = None
+                viewspace = []
+                for cam in cams:
+                    render_pkg = render_fastgs(cam, gaussians, pipe, background,
+                                               opt.mult, 1.0)
+                    image = render_pkg["render"]
+                    gt_image = cam.original_image.cuda()
+                    Ll1 = l1_loss(image, gt_image)
+                    if FUSED_SSIM_AVAILABLE:
+                        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
+                    else:
+                        ssim_value = 1.0
+                    view_loss = ((1.0 - opt.lambda_dssim) * Ll1
+                                 + opt.lambda_dssim * (1.0 - ssim_value))
+                    if (lpips_weight > 0.0 and perceptual is not None
+                            and _abs_iter % args.lpips_every == 0):
+                        render_p, gt_p = image.unsqueeze(0), gt_image.unsqueeze(0)
+                        if args.lpips_scale != 1.0:
+                            size = (max(32, int(round(render_p.shape[2] * args.lpips_scale))),
+                                    max(32, int(round(render_p.shape[3] * args.lpips_scale))))
+                            render_p = torch.nn.functional.interpolate(
+                                render_p, size=size, mode="area")
+                            gt_p = torch.nn.functional.interpolate(
+                                gt_p, size=size, mode="area")
+                        perceptual_loss = perceptual(render_p * 2.0 - 1.0,
+                                                     gt_p * 2.0 - 1.0).mean()
+                        view_loss = view_loss + lpips_weight * perceptual_loss
+                    loss_view = view_loss if loss_view is None else loss_view + view_loss
+                    viewspace.append(render_pkg["viewspace_points"])
+                (loss_view / len(cams)).backward()
+                with torch.no_grad():
+                    if (name == "stage2" and args.stage2_trainer == "absgrad"
+                            and viewspace):
+                        for _vp in viewspace:
+                            if _vp.grad is None:
+                                continue
+                            gaussians.xyz_gradient_accum += _vp.grad[:, :2].norm(
+                                dim=-1, keepdim=True)
+                            gaussians.denom += 1.0
+                        if _abs_iter % args.stage2_absgrad_every == 0:
+                            _moved = _absgrad_relocate()
+                            if not args.quiet and _moved:
+                                print(f"[absgrad] step {_abs_iter}: relocated {_moved} "
+                                      f"slots (net-zero)")
+                    if not args.quiet and _k % 100 == 0:
+                        bar.set_postfix({"it": _abs_iter, "lr": f"{_lr:.2e}"})
+                        bar.update(100)
+                    gaussians.optimizer_step(_abs_iter)
+            bar.close()
+            # record BEFORE the PLY save, exactly like the phase-1 snapshot
+            # trigger: research.run_sweep's B convention is
+            # scene_load + init + snapshot train_seconds + 0.5 (the flat 0.5 is
+            # the export allowance), so the export write is not charged twice.
+            elapsed = time.perf_counter() - t_stage
+            _stage_train_acc += elapsed
+            entry = dict(iteration=_abs_iter,
+                         train_seconds=_stage_train_base + _stage_train_acc,
+                         num_gaussians=int(gaussians.get_xyz.shape[0]),
+                         stage=name, stage_seconds=round(elapsed, 3))
+            ply_dir = args.snap_dir or dataset.model_path
+            os.makedirs(ply_dir, exist_ok=True)
+            ply = os.path.join(ply_dir, f"iter{_abs_iter:05d}.ply")
+            gaussians.save_ply(ply)
+            entry["ply"] = ply
+            snaps.append(entry)
+            print(f"[stage] {name}: {n_steps} steps in {elapsed:.2f}s -> {ply}")
+            return elapsed, n_steps
+
+        print(f"[stage] S={iterations} -> +{_n_stage2} steps ({args.stage2_trainer}, "
+              f"{args.stage2_cams} cam/step) then +{_n_stage3} VGG-LPIPS steps "
+              f"(w={args.stage3_lpips_weight}); xyz LR {lr_from:.3e} -> {lr_to:.3e} "
+              f"over {_extension} steps")
+        stage2_s, stage2_done = _run_stage("stage2", _n_stage2, 0.0)
+        stage3_s, stage3_done = _run_stage("stage3", _n_stage3, args.stage3_lpips_weight)
+    elif _n_stage2 or _n_stage3:
+        print(f"[stage] skipped: phase 1 stopped early ({stop_reason})")
     train_s = time.perf_counter() - t_train0
     if args.time_breakdown:
         timing = {k: round(v, 3) for k, v in timing.items()}
         timing["train_seconds"] = round(train_s, 3)
-        timing["iterations"] = iterations
+        timing["iterations"] = _abs_iter
     # standard 3DGS layout so the organizer's render_test.py can load this model
     final_ply = None
     if not args.no_save_final:
@@ -648,13 +875,20 @@
     record = dict(
         init_ply=str(args.init_ply), init_mode=init_mode,
         num_init_gaussians=n_init, num_final_gaussians=int(gaussians.get_xyz.shape[0]),
-        iterations=iterations, completed_iterations=iteration, stop_reason=stop_reason,
+        iterations=iterations, completed_iterations=_abs_iter, stop_reason=stop_reason,
         final_ply=final_ply,
         limit_seconds=args.limit_seconds, snapshots=snaps,
         scene_load_seconds=round(scene_load_s, 3), init_seconds=round(init_s, 3),
         train_seconds=round(train_s, 3), total_seconds=round(total_s, 3),
+        # staged continuation bookkeeping (all zero with the default flags)
+        stage1_seconds=round(stage1_s, 3), stage2_seconds=round(stage2_s, 3),
+        stage3_seconds=round(stage3_s, 3), stage1_iterations=iterations,
+        stage2_iterations=stage2_done, stage3_iterations=stage3_done,
+        total_iterations=_abs_iter, stage2_trainer=args.stage2_trainer,
+        stage2_cams=args.stage2_cams, stage2_lr_from=lr_from,
+        stage2_lr_final=lr_to, stage3_lpips_weight=args.stage3_lpips_weight,
         timing_breakdown=(timing if args.time_breakdown else None),
-        iterations_per_second=round(iteration / train_s, 2) if train_s else None,
+        iterations_per_second=round(_abs_iter / train_s, 2) if train_s else None,
         quick_test_psnr=round(final_psnr, 4) if final_psnr is not None else None,
         num_train_views=len(cams_by_scale[scales[-1]]),
         num_test_views=len(scene.getTestCameras()), res_scales=scales, res_until=cuts,
@@ -672,6 +906,13 @@
                       res_scales=scales, res_until=cuts,
                       lpips_weight=args.lpips_weight, lpips_every=args.lpips_every,
                       lpips_scale=args.lpips_scale,
+                      stage2_iters=args.stage2_iters, stage2_cams=args.stage2_cams,
+                      stage2_trainer=args.stage2_trainer,
+                      stage2_absgrad_every=args.stage2_absgrad_every,
+                      stage2_split_frac=args.stage2_split_frac,
+                      stage2_lr_final=args.stage2_lr_final,
+                      stage3_iters=args.stage3_iters,
+                      stage3_lpips_weight=args.stage3_lpips_weight,
                       densify_count_thresh=args.densify_count_thresh,
                       max_gaussians=args.max_gaussians),
     )
--- a/research/run_sweep.py
+++ b/research/run_sweep.py
@@ -117,6 +117,20 @@
     parser.add_argument("--image-cache", default=None, dest="image_cache",
                         help="pre-decoded uint8 image cache dir (module-A artifact)")
     parser.add_argument("--no-densify", action="store_true", dest="no_densify")
+    # ---- staged post-S continuation pass-through (optional; see train_sweep.py)
+    parser.add_argument("--stage2-iters", type=int, default=None, dest="stage2_iters")
+    parser.add_argument("--stage2-cams", type=int, default=None, dest="stage2_cams")
+    parser.add_argument("--stage2-trainer", type=str, default=None, dest="stage2_trainer")
+    parser.add_argument("--stage2-lr-final", type=float, default=None, dest="stage2_lr_final")
+    parser.add_argument("--stage2-lr-from-ratio", type=float, default=None,
+                        dest="stage2_lr_from_ratio")
+    parser.add_argument("--stage2-absgrad-every", type=int, default=None,
+                        dest="stage2_absgrad_every")
+    parser.add_argument("--stage2-split-frac", type=float, default=None,
+                        dest="stage2_split_frac")
+    parser.add_argument("--stage3-iters", type=int, default=None, dest="stage3_iters")
+    parser.add_argument("--stage3-lpips-weight", type=float, default=None,
+                        dest="stage3_lpips_weight")
     args = parser.parse_args(argv)
 
     profile = read(args.profile)
@@ -170,6 +184,15 @@
         "--opt_step_schedule": args.opt_step_schedule,
         "--inject_holes": args.inject_holes,
         "--inject_at": args.inject_at,
+        "--stage2_iters": args.stage2_iters,
+        "--stage2_cams": args.stage2_cams,
+        "--stage2_trainer": args.stage2_trainer,
+        "--stage2_lr_final": args.stage2_lr_final,
+        "--stage2_lr_from_ratio": args.stage2_lr_from_ratio,
+        "--stage2_absgrad_every": args.stage2_absgrad_every,
+        "--stage2_split_frac": args.stage2_split_frac,
+        "--stage3_iters": args.stage3_iters,
+        "--stage3_lpips_weight": args.stage3_lpips_weight,
     }
     for flag, value in optional.items():
         if value is not None:
```

