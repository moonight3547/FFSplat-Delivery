# LIT.md — how to spend ~100 s of module B on a frozen feed-forward init

Written 2026-10-30. Scope: module-B literature + a falsifiable experiment plan.
No GPU was touched and no training was run for this note.

**Provenance of numbers.** Everything marked *(measured)* is quoted from
`docs/RESEARCH-20261030.md` / `records/o0-baseline-20261030.json` on the same
4090 (FastGS 30 K random-init budget 92.6–99.5 s for the three default cases;
an earlier same-host calibration in `records/fastgs-random30k-calibration.json`
gave 93.7–102.7 s — both are our data, ~7 % apart, which is the timing noise
floor for every comparison in this file). Everything marked *(reported)* is from
the cited abstract and was **not** reproduced here. No number in this file is
invented.

**Fixed by the competition.** ReSplat [arXiv:2510.08575](https://arxiv.org/abs/2510.08575)
frozen, 16 posed views, no extra training data, no network edits, module B only.

## 0. The arithmetic that decides every design choice

Proxy score = quality 80 (PSNR 20 / SSIM 20 / LPIPS 40) + time 20, where time is
full marks at 10 % of budget and 0 at 100 %. Our own hw/billboard trace *(measured)*:

| B steps | Gaussians | B s | PSNR | SSIM | LPIPS | quality | time | total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 408 K | 8.1 | 22.289 | 0.7650 | 0.2055 | 19.93 | 20.00 | 39.93 |
| 2 000 | 413 K | 15.5 | 22.448 | 0.7703 | 0.1990 | 21.78 | 18.76 | 40.54 |
| 4 000 | 368 K | 29.7 | 22.595 | 0.7843 | 0.1841 | 26.00 | 15.58 | 41.58 |
| 8 000 | 330 K | 51.6 | 22.621 | 0.7963 | 0.1646 | 30.88 | 10.69 | 41.57 |
| 10 000 | 323 K | 49.3 | 22.648 | 0.7987 | 0.1598 | 32.05 | 11.21 | 43.26 |
| 20 000 | 307 K | 99.0 | 22.680 | 0.7957 | 0.1682 | 30.16 | 0.12 | 30.28 |

Two slopes, same units (points per second):

* time term costs `20 / (0.9 × 99.49) = 0.223` pts/s;
* quality buys `0.25` (1→2 K), `0.30` (2→4 K), `0.22` (4→8 K), `≤0` (10→20 K, quality actually falls) pts/s.

They cross at **~8 K steps ≈ 52 s**: beyond that, extra steps cost more time than
they earn. counter shows the same knee (4→8 K = 0.48 q/s, 10→20 K = 0.03 q/s).
The observed argmax is 10 K (43.26), 1.7 pts above 8 K — but the 10 K row was
also *faster* than the 8 K row, so that ordering is timing noise, not a gain.
Note also counter's 8 K row (64.4 s) being slower than its 10 K row (55.5 s):
repeat any timing before believing a <10 % difference.

**Consequences that constrain every experiment below**

1. The objective is **q(t), not q(100 s)**. At equal quality each second saved is
   worth 0.223 pts, so reproducing the 8 K-step result at 4 K steps (22 s saved)
   would be worth up to ~5 pts, and 20 % faster steps at the knee are worth ~2.
   Extending 8 K→20 K costs 47 s *and* loses quality. Score everything at
   2/4/6/8 K snapshots and take the argmax of `quality + time`.
2. Any new loss/regularizer costing fraction `c` of step time must repay
   `0.223 × c × T` quality points (`T ≈ 50 s` → `11 × c`; a 15 % overhead must
   buy 1.7 pts).
3. The total is flat between 4 K and 10 K (41.6 / 41.6 / 43.3) and collapses at
   20 K (30.28), while the observed spread between reasonable module-B designs is
   only ~2 pts *(measured: 59 / 60 / 58 — dropping the init entirely gives 31)*.
   **Adopt nothing below +2 pts on a 3-scene mean**, and never from a single run.

## 1. Literature (all identifiers verified 2026-10-30)

Coverage map for this brief: FastGS, ReSplat, ForeSplat, iSplat, Learn2Splat,
ProSplat, VoxelTTO, InstantSplat, MVSGaussian, Splatt3R, SplatFormer, MCMC,
Taming-3DGS, FSGS, SparseGS, DepthSplat, 3DGS-LM, plus GIFSplat (CVPR 2026).

### Table A — usable with a frozen FF model and no new data

| work | what it changes | reported cost | verdict at ~100 s |
|---|---|---|---|
| **FastGS** [2511.04283](https://arxiv.org/abs/2511.04283) | MV-consistency densify+prune, no budget mechanism; 3.32× vs DashGaussian, 15.45× vs 3DGS *(reported)* | ~100 s for 30 K steps *(our own 92.6–99.5 s matches the headline)* | Its densify/prune is tuned for SfM-init from-scratch runs. Under 100 s the *prune* is the useful part (buys steps), the densify is not |
| **ReSplat** [2510.08575](https://arxiv.org/abs/2510.08575) | recurrent refine on render error, init at 16× subsampled space, 16× fewer Gaussians *(reported)* | 2 refine passes | The init is deliberately coarse ⇒ systematic under-density. num_refine 4/6 saturated *(measured)* |
| **InstantSplat** [2403.20309](https://arxiv.org/abs/2403.20309) | dense prior points → **co-visibility geometry init** → photometric fit **without adaptive density control**; >30× faster, SSIM 0.3755→0.7624 *(reported)* | "in seconds" (title) | The closest published precedent for E1+E4: dedupe/prune the prior's points, then attribute-only |
| **MCMC** [2404.09591](https://arxiv.org/abs/2404.09591) | Gaussians as MCMC samples: SGLD noise, relocation instead of clone/split, opacity+scale regularizer, "robustness to initialization", easy count control | minutes | We measured 58 vs 60 *(measured)*; correct thesis (init-robust), wrong budget |
| **Taming-3DGS** [2406.15643](https://arxiv.org/abs/2406.15643) | MCMC + regularizers for "limited resources" training | targets consumer-class GPUs (title claim) | Source of our `mcmc_budget.py` recipe; wall clock 3–4× over budget *(measured)* |
| **FSGS** [2312.00451](https://arxiv.org/abs/2312.00451) | monocular-depth-guided optimization + Gaussian Unpooling to infill vacant regions | few-shot, minutes | Depth *loss* is portable (5–10 % step cost); Unpooling needs a depth net and adds primitives we cannot afford to re-fit |
| **SparseGS** [2312.00206](https://arxiv.org/abs/2312.00206) | depth priors, novel-depth rendering, floater pruning, unseen-viewpoint regularization | minutes | Floater pruning is the portable half; the unseen-viewpoint term targets extrapolation we are not scored on |
| **3DGS-LM** [2409.12892](https://arxiv.org/abs/2409.12892) | LM instead of Adam, cached gradients, custom CUDA JVPs; 20 % faster at equal quality *(reported)* | ICCV'25 | Out: needs CUDA kernels, and 20 % < the ~2 pts available from cheaper knobs |

### Table B — needs a trained network ⇒ out of scope (frozen ReSplat, no data)

| work | why it is out | what we still steal from it |
|---|---|---|
| **ForeSplat** [2605.22020](https://arxiv.org/abs/2605.22020) | MetaGrad unrolls the 3DGS optimizer and back-props into the prediction head: fine-tuning the FF network. We cannot touch ReSplat | The framing: predict-then-refine with **fixed primitive count, attribute-only** updates, few steps. It is the published justification for our 58-pt arm |
| **iSplat** [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_iSplat_Iterative_Learning_for_Fine-Grained_Gaussian_Splatting_CVPR_2026_paper.html) | 3 learned passes with a GRU optimizer, 42.6 M params; 27.67 vs DepthSplat 27.47 dB, +2.88 dB cross-dataset *(reported)* | Iterative refinement only pays when the update rule is *learned*; more passes of a frozen recurrent net do not (matches our num_refine saturation) |
| **Learn2Splat** [2605.15760](https://arxiv.org/abs/2605.15760) | meta-learned optimizer, horizon-stable, zero-shot across datasets; code/models "will be released" *(reported)* | Its insight is implementable by hand: encode gradient *scale* per parameter and keep updates stable over a short horizon → E2/E6 |
| **ProSplat** [2506.07670](https://arxiv.org/abs/2506.07670) | one-step diffusion improvement model + MORI/DWEA; +1 dB *(reported)* | Nothing usable: it adds a per-render network and needs training |
| **GIFSplat** [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_GIFSplat_Generative_Prior-Guided_Iterative_Feed-Forward_3D_Gaussian_Splatting_from_Sparse_CVPR_2026_paper.pdf) | generative-prior-guided iterative FF | Same family; confirms 2026 trend of paying at train time |
| **VoxelTTO** [2609.21498](https://arxiv.org/abs/2609.21498) | voxel-aligned FF + LoRA test-time optimization with pose supervision; **80 GPU-hours** training *(reported)* | TTO adapts *poses*, not appearance — wrong axis for us |
| **SplatFormer** [2411.06390](https://arxiv.org/abs/2411.06390) | point transformer that refines a whole 3DGS set in one pass, trained multi-scene; targets **OOD** test views | The only true "post-optimization of a Gaussian set" method. Watch item only: our test views are held-out *interpolation* (llffhold 8 inside the capture), not OOD |
| **DepthSplat** [2410.13862](https://arxiv.org/abs/2410.13862) | multi-view depth model on pretrained monocular features; 12 views at 512×960 in 0.6 s *(reported)*; depth ↔ 3DGS mutual benefit | Any gain requires retraining the depth/GS pair. Its *portable* idea is cross-view depth consistency as the regularizer, which is what E8 approximates |
| **MVSGaussian** [2405.12218](https://arxiv.org/abs/2405.12218) | MVS cost volume → geometry-aware Gaussians; per-view point clouds aggregated as init for short fine-tuning | The init idea (multi-view-consistent aggregation) maps to E1's dedupe/merge, which we can do without the net |
| **Splatt3R** [2408.13912](https://arxiv.org/abs/2408.13912) | zero-shot Gaussians from uncalibrated pairs, no per-scene optimization | Establishes the ceiling of pure FF output; not a refinement method |

### The four findings that actually change our plan

1. **Every large published gain on FF refinement is paid at train time** (ForeSplat,
   iSplat, Learn2Splat, ProSplat, GIFSplat, VoxelTTO). Under a frozen network +
   no data, the literature leaves only optimizer/init/loss changes, and all of
   those have measured out at ±2 pts for us. This is the single most important
   scoping result of the review.
2. **ReSplat's 16×-subsampled primitive budget is a systematic prior for
   under-density and over-smooth scales.** That is the one defect a *model-free
   init transform* can attack, and it is cheap enough to test with B=0 renders.
3. **InstantSplat is the precedent for our best remaining idea**: clean the
   prior's points (co-visibility/dedupe/prune) and then fit **without adaptive
   density control**. It reports >30× speedup and a large SSIM jump vs SfM+3DGS
   *(reported)* — the mechanism is init conditioning, not a better optimizer.
4. **No 2025–2026 paper studies budget allocation for post-optimizing a frozen FF
   init.** Targeted arXiv searches for "predict-then-refine", "post-optimization"
   and "time budget" × Gaussian splatting return nothing on this question. Our
   q(t)/score(t) curve (§0) is the missing piece, and it says: shorten and
   steepen, do not extend.

## 2. What this implies for module B (design rules)

* Keep the primitive count fixed and update attributes (ForeSplat, InstantSplat);
  treat densify as a *pruning* device that buys step time, not as a quality device.
* Judge every change by quality at ~40–50 s, because that is where the two slopes cross.
* Spend on init conditioning first; it is the only lever with >2 pts of measured headroom (28 pts).
* Position LR must be re-tuned for a *localized* init. FastGS's schedule is
  `1.6e-4 → 1.6e-6` geometric over `position_lr_max_steps = 30 000`
  *(verified in `baseline/fastgs/arguments/__init__.py`)*, so at 8 K steps the
  LR is still `4.7e-5` (29 % of max) and the settle phase is never reached.
* Any loss term must be time-accounted (§0 rule 2).

## 3. Ranked experiments (all runnable with `research/run_sweep.py`, `research/ply_ops.py`)

Common harness: 3 cases, snapshots `2000,4000,6000,8000`, `--opacity-reset-interval 0`,
report quality+time at every snapshot, and the argmax total. "Falsify" is the
single-GPU wall clock for the whole decision.

| # | experiment | predicted Δ (3-case mean) | impl | falsify | kill signal |
|---|---|---|---|---|---|
| 1 | E1 init conditioning (B=0) | +0.5…+2 | 30 min | 8 min | <0.5 pts on ≥2 cases |
| 2 | E2 LR shaping at 40–50 s | +1…+3 | 1 h | 12 min | <+1 on hw at 6 K, or any case negative |
| 3 | E3 step-cost reduction | +1…+2 | 2 h | 4 min | no ≥15 % step gain |
| 4 | E4 structural schedule | ±1 (confirmation) | 0 | 8 min | any densify arm must win by >1.5 |
| 5 | E5 late LPIPS | +0…+2 | 1 h | 5 min | net total ≤ control |
| 6 | E6 trust region to init | +0.5…+1 | 1 h | 5 min | PSNR < control |
| 7 | E7 MCMC-lite warm start | 0 | 2 h | 20 min | <+2 → abandon family |
| 8 | E8 depth anchor (ReSplat depth) | ±0.5 | 3 h | 10 min | <+1.7 (its own overhead) |

### E1 — Init conditioning screen, no training (rank 1)
* **init:** six transforms of the same `resplat_world.ply`: V0 as-is; V1 `ply_ops sharpen --gain 1.5`; V2 `--gain 2.0`; V3 `prune --min-opacity 0.05`; V4 `prune --min-opacity 0.02 --max-scale 0.25` then `merge` (dedupe overdraw); V5 = V4 with all scales ×0.7.
* **optimizer / loss / structure:** none — this is a B=0 measurement (`research/score_ply.py`).
* **predicted:** hw init scores 3.81/80 with LPIPS 0.2916 *(measured)*, the worst number in the report; a calibrated init should move it ≥1 quality point, other cases 0–1.
* **falsify (<8 min):** 6 variants × 3 cases, render 8 held-out views + VGG-LPIPS, ≈15 s/case (estimate).
* **then:** take the winner into E2/E4 — a B=0 win can be erased by 8 K Adam steps, so it only counts if it survives at the operating point.

### E2 — Per-parameter LR shaped for a localized init (rank 2)
* **init:** E1 winner.
* **optimizer:** Adam per group; position LR `3e-5 → 3e-6` with `--position-lr-max-steps` = real step count; scaling LR ×3, opacity LR ×2, SH LR ×2 (`--scaling-lr`, `--opacity-lr`).
* **loss / structure:** `0.8·L1 + 0.2·DSSIM` (existing), `--no-densify`.
* **predicted:** +1…+3 on hw at 6 K. The stock schedule decays over 30 K steps, so at 6 K the position LR is still `6.4e-5` and at 8 K it is `4.7e-5` — we never reach the settle phase, and that LR level is tuned for SfM points that are wrong by centimeters, not for means already within a Gaussian radius.
* **falsify (<12 min):** control + 4 LR sets on hw, 6 K steps, snapshots 2/4/6 K, compare totals at equal seconds.

### E3 — Cut the cost per step to buy steps (rank 3)
* **init:** E1 winner, one-shot prune to ~60 % of count (FastGS's own gate takes 455 K→~210 K *(measured)*; do it up front instead of at step 500).
* **optimizer:** E2 winner (same `0.8·L1 + 0.2·DSSIM` loss); `--sh-up-every` delayed so SH1 is used until 60 % of the budget, then SH3.
* **structure:** none.
* **predicted:** 15–30 % more steps in fixed seconds; at the knee that is worth up to `0.2·T·0.223 ≈ 2` pts, minus an LPIPS penalty from SH truncation (LPIPS is 40/80 of quality — verify it separately).
* **falsify (<4 min):** one case, two variants, log steps completed at 40 s and quality at equal seconds.

### E4 — Structural-op schedule at fixed seconds (rank 4)
* **arms:** (a) none, (b) densify first 20 % of steps then off, (c) FastGS default with opacity reset disabled, (d) prune at 30 % then attribute-only.
* **optimizer / loss:** held constant across arms (`0.8·L1 + 0.2·DSSIM`, default Adam) so the only variable is the structural schedule.
* **predicted:** ±1 — we already measured 58/59/60 for the attribute-only / densify+prune / sharpened-init trio *(measured)*. This is a confirmation run that stops the team re-litigating it.
* **falsify (<8 min):** 4 arms × 1 case at 8 K steps.

### E5 — LPIPS only in the last quarter (rank 5)
* **loss:** `+0.05…0.1 · LPIPS` every 4th step over the last 25 % of the time budget, on top of L1+DSSIM.
* **rationale:** the falsified variant was LPIPS **at every step** (cost + gradient-scale instability). Late-only is a different experiment against the 40-point metric.
* **predicted:** a 0.01 LPIPS gain = 2 pts; cost `c ≈ 0.1` needs >1.1 pts. Marginal, hence rank 5.
* **falsify (<5 min):** two weights vs E2 winner on one case.

### E6 — Trust region toward the init (rank 6)
* **loss/opt:** decaying L2 pull on means (`1e-2 → 0` over the run) plus a scale clamp to `[0.5, 2] ×` init for the first 50 %; E2's optimizer and `0.8·L1 + 0.2·DSSIM` unchanged.
* **rationale:** the cheap, hand-built version of Learn2Splat's horizon stability and 3DGS-LM's trust region; protects the good init from floater drift, which is exactly hw's failure.
* **predicted:** +0.5…+1, low variance.
* **falsify (<5 min):** two variants vs control on hw.

### E7 — MCMC-lite warm start (rank 7)
* **init/optimizer/loss:** 2–3 K steps of `MCMCStrategy` (SGLD noise + relocation, no cap growth) on the E1 init with the same `0.8·L1 + 0.2·DSSIM`, then E2's Adam attribute-only phase.
* **predicted:** 0. MCMC's own thesis is init-robustness; our init is already good and we measured 58 vs 60 *(measured)*.
* **falsify (<20 min):** one arm vs E2 winner on two cases; abandon the family if <+2.

### E8 — Depth anchor from ReSplat's own rendered depth (rank 8)
* **init:** re-run inference with the upstream `--save-depth` path (it already returns `rendered_depth`), keep per-view depth.
* **loss:** truncated L1 between rendered depth and the init depth, weight 0.01–0.02, first 50 % of steps.
* **predicted:** ±0.5 and partly circular (the depth comes from the same primitives); it costs 5–10 % step time, so it must buy >0.6–1.1 pts to break even.
* **falsify (<10 min):** one arm vs control.

## 4. What NOT to try

1. **More ReSplat passes** — num_refine 2→4→6 saturated *(measured)*; iSplat shows more passes pay only with a *learned* update rule.
2. **Coarse-to-fine resolution** — already falsified *(measured)*; it also re-caches uint8 views per stage.
3. **LPIPS in the loss at every step** — already falsified *(measured)*; per-step cost violates §0 rule 2.
4. **Lowering the densify gate / raising `--max-gaussians`** — already falsified *(measured)*; FastGS prunes 455 K→~210 K for a reason: fewer, better-placed primitives render faster.
5. **FPS / coverage view selection** — already falsified *(measured)*; ranked16 did not change the selected contexts on any of the three cases. (Also an A-side lever, not module B.)
6. **Anything that retrains a network**: ForeSplat MetaGrad, iSplat's GRU optimizer, Learn2Splat, ProSplat/GIFSplat diffusion, VoxelTTO (80 GPU-h *(reported)*). Rules freeze ReSplat; we also have no training data.
7. **SplatFormer as a drop-in module B** — needs multi-scene training on Gaussian sets and targets extreme OOD views; our held-out views are interpolated inside the same capture.
8. **3DGS-LM** — custom CUDA Jacobian-vector kernels, days of work, 20 % *(reported)* < the ~2 pts E2/E3 can win.
9. **Full-length MCMC** — 276–449 s per case historically *(measured)*, i.e. 3–4× over budget.
10. **Depth-net-guided densification (FSGS unpooling, SparseGS)** — extra network, extra per-step cost, and the regions are under-observed for the FF model too, so the prior adds little that the 16 views do not already say.

## 5. Executive summary

* **The 100 s budget is a tent, not a wall.** At ~8 K steps the marginal quality
  (0.22 pts/s) equals the marginal time penalty (0.22 pts/s) *(measured)*, and the
  observed total is flat from 4 K to 10 K steps (41.6–43.3) before collapsing at
  20 K (30.28). There is no score to be won by spending the full 99 s; the wins
  are in reaching the plateau sooner.
* **Module B is not where the points are.** Init is worth ~28 points (59 vs 31);
  module-B designs span 58–60. The only real headroom is a **steeper q(t)** —
  reach 8 K-step quality in 4 K steps — which is an init-conditioning and
  LR-shaping problem, not an optimizer-choice problem.
* **The literature's big wins are all out of scope.** ForeSplat, iSplat,
  Learn2Splat, ProSplat, GIFSplat and VoxelTTO all pay at train time; with ReSplat
  frozen and no extra data, the transferable recipes (fixed count, attribute-only,
  prior-point cleanup) are worth ±2 points by our own measurements.
* **Do E1 then E2 this week, nothing else first.** E1 is a training-free B=0 screen
  of six PLY transforms (minutes, no GPU training); E2 re-tunes per-parameter LR
  for a localized init at the 40–50 s operating point. E3–E8 are confirmations or
  low-EV; run them only if E1/E2 leave budget.
* **Measurement discipline.** Primary metric becomes quality at ~40 s plus the
  argmax total over snapshots; require +2 pts on a 3-case mean; repeat any timing
  whose difference is <10 % (counter's 8 K vs 10 K rows already contradict each
  other). Keep the E2E-vs-B-only numerator ambiguity flagged in every report.

## 6. Verification status

* Verified identifiers/abstracts (2026-10-30): 2511.04283, 2510.08575, 2605.22020,
  2605.15760, 2609.21498, 2506.07670, 2403.20309, 2405.12218, 2408.13912,
  2411.06390, 2404.09591, 2406.15643, 2312.00451, 2312.00206, 2410.13862,
  2409.12892; iSplat and GIFSplat via CVF open access (no arXiv id found for iSplat).
* *(reported)* and not reproduced: every paper timing, all paper PSNR/SSIM/LPIPS,
  the 80 GPU-hour figure, InstantSplat's "seconds"-scale and ">30×" claims,
  3DGS-LM's 20 %.
* *(measured)* and reproducible from this repo: all §0 tables, the 58/59/60/31
  module-B comparison, num_refine saturation, the 10 K prunes.
* Open risk: TIME's numerator (B-only vs E2E) is still unconfirmed by the
  organizers. If it is E2E, A+ReSplat setup alone may consume the budget and every
  module-B experiment here becomes moot — settle that before E2.

## 7. Implementation note — E8 depth anchor and a new E9 pose arm (2026-10-30)

Added after the literature review above (which is unchanged).  Detailed design,
commands and failure modes: `research/DEPTH.md`.  CPU checks only; no GPU run.

**Files.** `research/depth_reg.py` (depth-regularized continuation, same knobs as
`train_sweep.py`), `research/pose_refine.py` (translation-only refinement, off by
default), `research/export_resplat_depth.py` (raw depth dump -> sidecar).

**E8 as implemented — predicted depth, not rendered depth.**  The upstream
`run_resplat.py` now returns `rendered_depth`, but ReSplat also exposes its
internal depth prediction (`visualization_dump["depth_fullres"]`, predicted at
1/4 resolution).  We use the *prediction*, which removes E8's circularity (the
depth is not rendered from the same primitives being trained).  A one-off
`save_depth=True` monkey-patch dumps it without editing upstream; the sidecar
stores image-stem -> `float32 [H, W]` at the module-B training resolution.  Only
the 16 `split["context"]` views have a prediction, so the term fires on ~29 % of
training draws and the other 40 train views stay photometric (reported in
`train_record.json`).  Loss: alpha-masked Charbonnier on relative depth plus an
edge-aware Sobel gradient-matching term (predicted-discontinuity weight
`exp(-beta |grad d_pred| / median d_pred)`), `weight 0.05`, every 4th iteration,
iterations 5–80 % of the run.  A rendered-depth pass costs one extra
rasterization and only runs when the sampled view has a sidecar (~7 % extra
rasterizer work at the defaults), which sits inside E8's 5–10 % estimate and
under §0 rule 2 needs ~0.8 pts at T≈50 s.  Keep E8's kill signal: **< +1.7 pts on
the 3-case mean at equal seconds -> abandon**.

**E9 — translation-only pose refinement.**  3 DoF per view, no rotation or
intrinsics; last M steps; shift clamp 2 % of scene extent; `--pose-lr 0` by
default.  The pinned rasterizer cannot backprop through `viewmatrix`, so the
translation is applied as the exact world shift `xyz - delta` for the sampled
view (same RGB/alpha/depth as moving the camera; `means3D` is differentiable).
Pure-photometric and depth-anchored variants; run separately until both are
measured.  Predicted effect is small (±0.5–1) and the main risk is test-PSNR
regression from overfitting train poses, hence the default-off guard.

**Status.** Argument parsing, sidecar I/O, loss math, depth encoding and the
pose transform are covered by `--self-check` on CPU (`depth_reg` 41 checks,
`pose_refine` 26, `export_resplat_depth` 22 with CPU torch).  Not verified: the
encoded-depth rasterization, real step timing and any metric change.
