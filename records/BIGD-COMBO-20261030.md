# BIGD-COMBO-20261030 — BigD2500 × M1 × A-side scene cache (isolated factors)

**Task 19** · N1 (3×4090), serial quiet box on GPU2 · 2026-10-31 · base commit `c607ca6`
(BigD2500 merge: `op→opt` fix + actual-native-parameter densification event log) · raw
`/root/gpufree-data/outputs/bigd-combo-20261030/`, local copies under
`records/bigd-combo-20261030/raw/`. Official metrics (resplat PSNR/SSIM/VGG-LPIPS);
no VGG tail, no post-5000 stages; seed 20261015.

> **VERDICT.** BigD2500 is a **real quality win** (+2.60 q bicycle, +5.31 q 0b5635) that
> costs +8.40 / +7.27 s of B; **M1 (importance prune keep 0.6 @1250) and the A-side scene
> tensor cache recover most of that cost while keeping BigD's quality**:
> bicycle **bigd_m1_cache = 28.13 total, +1.59 vs bigd and +2.22 vs delivery**, with quality
> identical to bigd within noise (Δq **−0.006**); 0b5635 **bigd_cache = 36.61 total,
> +0.98 vs bigd and +4.57 vs delivery** (Δq +0.04). Answering the task's questions:
> **Q1 yes** (M1 keeps quality within 0.5 q and raises total), **Q2 yes** (m1+cache
> dominates plain bigd), **Q3 yes** (cache leaves quality unchanged within noise and
> reproduces the 2.1–3.5 s scene_load saving under the bigger model).

## 1. Recipe verification (commit `c607ca6`)

| check | bicycle | 0b5635 |
|---|---|---|
| BigD densification events | **19** | **19** |
| event iterations | 600, 700, …, 2400 | 600, 700, …, 2400 |
| actual native `grad_abs_thresh` at every event | **0.0008** | **0.0008** |
| `mult` (FastGS box multiplier) | 0.5 | 0.5 |
| delivery control events / threshold | 7 / 0.0012 (native default) | 7 / 0.0012 |
| final N (bigd vs delivery) | 1,528,177 vs 998,924 | 1,100,210 vs 793,594 |

So the `op→opt` fix works: the parsed `OptimizationParams` reaches
`densify_and_prune_fastgs`, and the trainer records the actual parameters per event.

## 2. Results — bicycle (budget 95.0 s)

B = scene_load + init + **snapshot** `train_seconds` + 0.5 (PLY export and eval are outside
B; the record's `train_seconds` includes the final PLY export — both are listed).

| arm | N final | dev | PSNR | SSIM | LPIPS | q | load | init | snap_train (rec train) | **B** | TIME | **total** | Δq vs bigd | ΔB vs bigd | Δtotal vs bigd | Δtotal vs delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| delivery (ctrl) | 998,924 | 7 | 24.540 | 0.7295 | 0.2980 | 9.992 | 2.157 | 0.909 | 23.39 (31.86) | 26.95 | 15.917 | 25.909 | −2.596 | −8.40 | −0.631 | — |
| **bigd** | 1,528,177 | 19 | 24.925 | 0.7581 | 0.2586 | **12.588** | 2.375 | 0.893 | 31.59 (42.50) | 35.35 | 13.952 | 26.540 | — | — | — | +0.631 |
| bigd_cache | 1,562,185 | 19 | 24.897 | 0.7561 | 0.2608 | 12.406 | **0.224** | 0.865 | 31.37 (42.55) | 32.96 | 14.512 | 26.917 | −0.182 | −2.39 | +0.377 | +1.008 |
| bigd_m1 | 1,272,524 | 19 | 24.868 | 0.7571 | 0.2636 | 12.462 | 2.231 | 0.771 | 27.72 (36.45) | 31.22 | 14.919 | 27.382 | **−0.126** | **−4.13** | **+0.841** | +1.472 |
| **bigd_m1_cache** | 1,204,926 | 19 | 24.917 | 0.7581 | 0.2623 | **12.582** | **0.219** | 0.811 | 27.02 (35.20) | **28.54** | 15.545 | **28.127** | **−0.006** | **−6.81** | **+1.587** | **+2.218** |

M1 prune event: `bigd_m1` 1,140,113 → 684,068 (keep 0.6, 0.098 s); `bigd_m1_cache`
1,078,021 → 646,813 (0.090 s). After pruning the run re-densifies (window still open to
2500) to 1.27M / 1.20M — which is why quality is preserved: M1 removes the lowest-importance
Gaussians early and BigD rebuilds useful ones with the remaining budget.

## 3. Results — dl3dv/0b5635 (budget 94.29 s)

| arm | N final | dev | PSNR | SSIM | LPIPS | q | load | init | snap_train (rec train) | **B** | TIME | **total** | Δq vs bigd | ΔB vs bigd | Δtotal vs bigd | Δtotal vs delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| delivery (ctrl) | 793,594 | 7 | 23.844 | 0.8158 | 0.2462 | 17.099 | 3.685 | 0.657 | 26.04 (31.35) | 30.89 | 14.943 | 32.042 | −5.308 | −7.27 | −3.594 | — |
| **bigd** | 1,100,210 | 19 | 24.119 | 0.8309 | 0.2268 | **22.407** | 3.765 | 0.646 | 33.25 (53.59) | 38.16 | 13.230 | 35.637 | — | — | — | +3.594 |
| **bigd_cache** | 1,088,029 | 19 | 24.154 | 0.8319 | 0.2260 | **22.450** | **0.280** | 0.646 | 32.75 (52.9) | **34.19** | 14.165 | **36.615** | **+0.042** | **−3.97** | **+0.978** | **+4.573** |

0b5635 was the prior "M1-negative, gate-must-not-fire" case, so no M1 arm was run there
(per task) — BigD+cache already dominates delivery by +4.57 total.

## 4. Answers to the task's questions

1. **Does `bigd_m1` preserve BigD2500 quality within Δq ≤ 0.5 while reducing B enough to
   raise total?** **YES.** Δq −0.126 (well inside 0.5), ΔB −4.13 s, ΔTIME +0.97,
   **Δtotal +0.841** vs bigd (and +1.472 vs delivery). The quality cost of the prune is
   essentially zero because the BigD window continues to 2500 and refills.
2. **Does `bigd_m1_cache` dominate plain BigD2500 on total?** **YES.** **+1.587 total**
   (+6.81 s B: −4.13 M1, −2.39 cache, ±noise; Δq −0.006). It also dominates the delivery
   control by **+2.218** while its B is only **+1.59 s** above delivery's B.
3. **Does the scene cache leave quality unchanged within noise and reproduce its saving
   under the larger BigD model?** **YES.** Δq −0.182 (bicycle) / +0.042 (0b5635), i.e.
   inside single-run noise (run-to-run `num_final_gaussians` varies ±3 % on this recipe);
   scene_load **2.375 → 0.224 s** (−2.15 s, bicycle) and **3.765 → 0.280 s** (−3.49 s,
   0b5635); B deltas −2.39 / −3.97 s. Camera transforms bitwise identical; cached pixels
   1 f32 ulp (5.96e-8) as in the audit.

**Recommendation:** the best measured bicycle configuration is **BigD2500 + M1 keep 0.6 @1250
+ A-side scene cache = 28.13 total vs 25.91 delivery (+2.22)**; for 0b5635 use
**BigD2500 + cache = 36.62 vs 32.04 delivery (+4.57)**. All three factors are additive
(quality from BigD, B recovery from M1, load recovery from the cache). Delivery default
stays unchanged until the full 13-case validation (and note the bicycle LPIPS 0.2586 is
still above the 0.25 scoring cliff, so BigD's bicycle quality gain comes from PSNR+SSIM;
a 13-case run must confirm the cliff stays crossed).

## 5. Cache reuse / A-side accounting

The A-side bundles from task-18 were reused: their version hashes matched the current
`split.json` + `image_cache/manifest.json` (`be2e4a48…` bicycle, `4438bcdb…` 0b5635).
The task-18 strict verify check treats the known **1 f32 ulp** pixel difference as
"not identical", so the matrix script rebuilt both bundles anyway — **A-side only**:
bicycle **5.32 s**, 0b5635 **7.32 s** wall (≈2.5–3.8 s of that is the normal `Scene` load
A performs; 0.9/1.6 s is the memmap write). None of it is counted in B; camera tensors
were bitwise identical to a normal Scene (`maxdiff 0.0`).

## 6. Caveats

* One run per arm (the task's matrix); quality/N noise is ±3 % N and ≈±0.3 q, so Δq of
  ±0.2 between arms is not resolvable. The Δtotal conclusions (+0.84 … +1.59) are larger
  than that noise but a 2–3-repeat confirmation is the obvious next step.
* Absolute B in this session is slightly higher than the clean-13 delivery table
  (bicycle delivery B 26.95 vs clean-13 27.3–29.4 depending on session); all arms were run
  back-to-back on the same quiet GPU, so the deltas are within-session comparisons.
* The optional **init-PLY tensor cache** prototype (skip PLY parse during B init) was not
  attempted: the main matrix finished first and the task said not to delay it. Measured
  init is 0.65–0.91 s; a PLY tensor cache could remove ~0.4–0.6 s of it (class A,
  A-side build), worth ~+0.1 TIME if validated.

## 7. Exact commands / artifacts

```bash
# 8-cell serial matrix (resume-safe), GPU2 quiet box
python /tmp/stagedloss/bigd_combo.py       # -> /root/gpufree-data/outputs/bigd-combo-20261030/matrix
python /tmp/stagedloss/bigd_analyze.py     # -> .../results.json
# per-arm recipe flags:
#  delivery   : --densify-until 1250
#  bigd       : --densify-until 2500 --grad-abs-thresh 0.0008
#  bigd_m1    : bigd + --mid-prune-at 1250 --mid-prune-keep 0.6
#  bigd_cache : bigd + --scene-cache <bundle>
#  common     : --iterations 5000 --snapshots "" --position-lr-max-steps 5000
#               --densify-from 500 --densification-interval 100 --opacity-reset-interval 0
#               --lambda-dssim 0.6 --scaling-lr 0.01 --realloc-frac 0.15 --realloc-at 1250
#               --image-cache <work>/image_cache --no-quick-test --seed 20261015
#               --profile competition/profiles/gpufree-2.json
```

Artifacts: `records/bigd-combo-20261030/{results.jsonl (8 rows with Δ vs bigd and Δ vs
delivery), run.sh, bigd_combo.py, bigd_analyze.py, raw/<scene>/<arm>/{summary.json,
train_record.json}}`; raw on N1 `/root/gpufree-data/outputs/bigd-combo-20261030/` (matrix,
matrix.out, results.json, cache-info from the matrix log). No git push; defaults unchanged.
