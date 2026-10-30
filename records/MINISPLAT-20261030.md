# MINISPLAT-20261030 — Mini-Splatting-style density control on the delivery recipe

**Task 10** · node N1 (3×4090) · 2026-10-31 · scenes `mipnerf360/bicycle` (low total) and
`mipnerf360/counter` (mid total) · raw data on N1
`/root/gpufree-data/outputs/minisplat-20261030/`, local copies under
`records/minisplat-20261030/raw/`.

> STATUS: complete. **M2 (post-5000 importance prune + refine) is net-negative at every
> fraction** — the prune itself is almost free (0.09 s) and 70–90 % keep barely moves
> quality, but the post-pass only *adds* B time, so it can never win. **M1 (prune in the
> middle of phase 1 and let the remaining steps train the smaller model) is the real
> B-time lever and produced the first positive-Δtotal arm of the session on bicycle
> (+0.76), but it is scene-dependent and negative on counter (−1.55)** because counter's
> LPIPS is below the 0.25 scoring cliff and pays for the quality loss. Details, the
> measured B-vs-count law and the adoption caveat are below.

## 0. Verdict

| arm | N after | ΔN | ΔB (s) | ΔTIME | Δq | Δtotal | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| M2 prune→0.90 + 400 refine (bicycle) | 895,927 | −10.0 % | **+1.89** | −0.44 | −0.05 | **−0.50** | DROP (post-pass adds B) |
| M2 prune→0.70 + 400 refine (bicycle) | 698,426 | −30.0 % | **+1.60** | −0.37 | −0.11 | **−0.48** | DROP |
| M2 prune→0.50 + 400 refine (bicycle) | 500,757 | −50.0 % | **+1.33** | −0.31 | −0.89 | **−1.20** | DROP |
| M2 prune→0.70 + 400 refine (counter) | 433,559 | −30.0 % | **+1.65** | −0.40 | −2.12 | **−2.52** | DROP |
| **M1 prune@1250→0.80 (bicycle)** | 795,215 | −20.6 % | **−2.06** | **+0.48** | −0.08 | **+0.41** | positive |
| **M1 prune@1250→0.70 (bicycle)** | 692,840 | **−30.8 %** | **−3.56** | **+0.83** | −0.08 | **+0.76** | positive |
| **M1 prune@1250→0.60 (bicycle)** | 597,851 | −40.3 % | **−4.81** | **+1.13** | −0.29 | **+0.84** | positive |
| **M1 prune@1250→0.50 (bicycle)** | 495,678 | −50.5 % | **−6.06** | **+1.42** | −0.57 | **+0.85** | **best total found** |
| M1 prune@1250→0.40 (bicycle) | 397,994 | −60.2 % | −7.04 | +1.65 | −1.16 | +0.49 | past the optimum |
| M1 prune@1250→0.70 (counter) | 433,425 | −30.0 % | −2.20 | +0.53 | −2.08 | **−1.55** | negative (LPIPS pays) |
| M1 prune@1250→0.70 (dl3dv/0b5635) | 559,037 | −29.8 % | −1.86 | +0.44 | −0.86 | **−0.42** | slightly negative (LPIPS pays a little) |

**The honest one-liner:** Mini-Splatting's pruning *half* is real — on bicycle 20–50 % of
the Gaussians can be removed at iteration 1250 with essentially no quality cost, and the
whole-run module-B drops 2.1–6.1 s. The fraction curve peaks at **keep 0.5–0.6:
Δtotal +0.84…+0.85** (bicycle was the lowest-total delivery scene). But the same recipe
is **negative on counter (−1.55)** and **slightly negative on the low-total DL3DV scene
0b5635 (−0.42)**, because those scenes' delivered LPIPS sits *below* the 0.25 scoring
cliff, where every +0.01 LPIPS costs 2 quality points. Mean over the three tested scenes
at keep 0.70 is **−0.40**, and the competition rules forbid per-scene test-time tuning
(`AGENTS.md`: no per-scene hyperparameter selection) — so **not adoptable as a uniform
recipe**, but it is the only direction in this project where B time was actually moved,
and a *legitimate train-time gate* (the importance pass already yields train-view
PSNR/SSIM/LPIPS for 24 views at +0.09 s) is the concrete follow-up.

## 1. What was implemented

New module `research/ply_importance.py` (does not touch `ply_ops.py` / `ply_voxgrow.py`):

* `score_importance(gaussians, cams, pipe, bg, opt, n_views)` — for a fixed random set of
  **TRAIN views only** (default 24), forward+backward the standard
  `0.4·L1 + 0.6·(1−SSIM)` loss and accumulate per-Gaussian
  `importance_g = Σ_views || d loss / d μ_2d ||` (2D-projected gradient norm — the
  "rendered evidence / visibility" importance of Mini-Splatting, no test images).
  It also accumulates visibility counts for diagnostics. It leaves the parameter grads
  cleared so the trainer's state is untouched.
* `prune_gaussians(gaussians, keep_fraction, score)` — physically removes the
  least-important rows through the pinned FastGS `GaussianModel.prune_points`, so
  optimiser state, densification stats and `max_radii2D` all stay consistent.
* `main()` — standalone CLI (`--work --ply --out-ply --keep --views`) that scores a
  delivery PLY and writes the pruned PLY + a JSON record; this is the exact code the
  trainer calls.

`research/train_sweep.py` (new flags, **defaults off → phase 1 byte-identical**; same
snapshot/B convention as the delivered task-8 patch):

* `--stage2_trainer prune` + `--prune_keep F` + `--prune_views K` — **M2**: after the
  S=5000 delivery run, score → prune to `F`, then run `--stage2_iters` standard 1-cam
  refinement steps (no densification) with the continuation LR (scheduler(S) → 0.1×).
* `--mid_prune_at T` + `--mid_prune_keep F` + `--mid_prune_views K` — **M1**: at phase-1
  iteration T (placed after the T=1250 reallocation) score + prune in place, then let
  the *same* run continue to 5000. The optimiser state is preserved, so the schedule is
  exactly the delivery recipe — the only difference is that steps T+1..5000 train a
  smaller model. This is the part that can actually reduce B.
* Records `prune_keep/prune_views/mid_prune_at/mid_prune_keep/prune_events`
  (`n_before`, `n_after`, seconds, views) and the existing per-stage seconds.

`research/run_sweep.py`: pass-through for the five new flags.
Driver: the task-8 `records/staged-loss-20261030/run_staged.py` gained arms
`m2_50/m2_70/m2_90` and `m1_1250_60/70/80/90`, `m1_2000_70`; cells file
`records/minisplat-20261030/cells.txt`; entry point `run.sh`.

**B accounting** (identical convention to the delivery run):
`B = scene_load + init + snapshot.train_seconds + 0.5`, where the snapshot `train_seconds`
for M2 includes the score+prune+refine stage and for M1 is the phase-1 snapshot elapsed
time (the mid-run prune is inside the loop). The flat 0.5 is the export allowance; the
research snapshot PLY writes are excluded exactly like before.

## 2. Controls (measured in the same session, serial, one GPU)

| scene | N | PSNR | SSIM | LPIPS | q | B (s) | TIME | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mipnerf360/bicycle ctrl | 1,001,119 | 24.592 | 0.7294 | 0.2975 | 10.026 | 26.93 | 15.92 | 25.95 |
| mipnerf360/counter ctrl | 619,595 | 27.833 | 0.9011 | 0.1808 | 40.194 | 27.15 | 15.86 | 56.05 |

Both reproduce the known N1 series (bicycle selfcheck/anchor q 9.89–10.05, B 27.3;
counter pareto serial q 40.14, B 27.65) within run-to-run noise (bicycle B varies
26.9–27.3 s between sessions; Δ comparisons below are within-run for M2 and
same-session ctrl for M1).

## 3. M2 — importance prune after S=5000, then 400 refine steps (post-pass)

Paired inside each run (iter05000 → stage end). `prune_s` = score+prune, `refine_s` =
400 standard steps at the pruned count.

| scene | keep | N before → after | prune_s | refine_s | PSNR 5000 → end | SSIM | LPIPS | q 5000 → end | Δq | ΔB | Δtotal |
|---|---:|---|---:|---:|---|---|---|---:|---:|---:|---:|
| bicycle | 0.90 | 995,474 → 895,927 | 0.086 | 1.892 | 24.579 → 24.551 | 0.7289 → 0.7285 | 0.2988 → 0.2979 | 9.975 → 9.921 | −0.05 | +1.89 | **−0.50** |
| bicycle | 0.70 | 997,751 → 698,426 | 0.086 | 1.599 | 24.548 → 24.496 | 0.7287 → 0.7279 | 0.2984 → 0.2992 | 9.934 → 9.829 | −0.11 | +1.60 | **−0.48** |
| bicycle | 0.50 | 1,001,514 → 500,757 | 0.084 | 1.330 | 24.541 → 24.282 | 0.7298 → 0.7213 | 0.2980 → 0.3146 | 10.017 → 9.130 | −0.89 | +1.33 | **−1.20** |
| counter | 0.70 | 619,370 → 433,559 | 0.091 | 1.650 | 27.793 → 27.383 | 0.9006 → 0.8967 | 0.1813 → 0.1887 | 40.022 → 37.902 | −2.12 | +1.65 | **−2.52** |

**M2 verdict: DROP.** The importance criterion itself is cheap (0.09 s) and 10–30 %
removal is nearly quality-free on bicycle (−0.05/−0.11), which validates the paper's
pruning half; but a post-pass can only *add* B, so Δtotal is negative at every fraction.
It also shows the quality cliff: at 50 % keep the model has lost the fine detail LPIPS
sees (0.2980 → 0.3146, −0.89 q), and on counter even 30 % costs −2.1 q because counter's
LPIPS is in the paying range.

## 4. M1 — prune in the middle of phase 1 (the B-time lever)

Prune at iteration 1250 (after densification + one-shot reallocation), then continue the
same run to 5000 (LR schedule untouched). Measured against the same-session control.

| scene | keep | N at prune → final | ΔN | B ctrl → arm | ΔB | ΔTIME | q ctrl → arm | Δq | Δtotal |
|---|---:|---|---:|---|---:|---:|---|---:|---:|
| bicycle | 0.80 | 994,019 → 795,215 | −20.6 % | 26.93 → 24.87 | −2.06 | +0.48 | 10.026 → 9.950 | −0.075 | **+0.41** |
| bicycle | 0.70 | 989,772 → 692,840 | −30.8 % | 26.93 → 23.37 | **−3.56** | **+0.83** | 10.026 → 9.950 | −0.075 | **+0.76** |
| bicycle | 0.60 | 996,419 → 597,851 | −40.3 % | 26.93 → 22.12 | −4.81 | +1.13 | 10.026 → 9.740 | −0.286 | **+0.84** |
| bicycle | 0.50 | 991,355 → 495,678 | −50.5 % | 26.93 → 20.87 | −6.06 | +1.42 | 10.026 → 9.458 | −0.567 | **+0.85** |
| bicycle | 0.40 | 994,984 → 397,994 | −60.2 % | 26.93 → 19.89 | −7.04 | +1.65 | 10.026 → 8.864 | −1.162 | +0.49 |
| counter | 0.70 | 619,179 → 433,425 | −30.0 % | 27.15 → 24.95 | −2.20 | +0.53 | 40.194 → 38.117 | −2.078 | **−1.55** |
| dl3dv/0b5635 | 0.70 | 798,624 → 559,037 | −29.8 % | 29.31 → 27.45 | −1.86 | +0.44 | 17.187 → 16.331 | −0.856 | **−0.42** |

Δ decomposition for the winner (bicycle keep 0.50): PSNR 24.592→24.493 (−0.10 dB),
SSIM 0.7294→0.7233 (−0.006), LPIPS 0.2975→0.3171 (+0.020, **free** because the LPIPS term
is already 0 above 0.25), q −0.57, B −6.06 s (22.5 % of B), ΔTIME +1.42 → **+0.85**.

The trade-off curve is monotone up to keep≈0.5 (B saving grows faster than the PSNR/SSIM
loss), then rolls over at 0.4 when the visible damage starts to show. The uniform
(3-scene, keep 0.70) mean is **−0.40 total** because LPIPS-below-cliff scenes lose more
quality than the B saving buys.

The prune itself costs 0.09 s (24 scoring views), i.e. 0.4 % of the B saving. Bicycle is
the first positive-total arm found in the whole session: quality is flat (PSNR −0.03,
SSIM −0.0006, and **LPIPS 0.2975 → 0.3015 costs nothing because bicycle's LPIPS is
already above the 0.25 cliff**) while module-B drops 13 %. Counter loses because its
LPIPS (0.1808) is *below* the cliff: +0.008 LPIPS = −1.6 quality points, on top of
−0.27 dB PSNR and −0.003 SSIM.

A `m1_1250_90` fraction was registered but not run (the curve's flat region is already
covered by 0.80 -> 0.70); `m1_2000_70` (later prune = smaller B saving) was skipped once
the 1250 point proved dominant.

## 5. The measured "B ∝ Gaussian count" law (the premise check)

The task premise was "a 20 % count cut ≈ +3.6 TIME". Measured per-step module-B cost from
the 400-step refine runs at known counts (same scene/step settings) and the delivery
control:

| scene | N | per-step cost | fit |
|---|---:|---:|---|
| bicycle refine @ 896 k | 895,927 | 4.73 ms | cost(N) ≈ **1.82 ms fixed + 3.02 ms per 1 M Gaussians** |
| bicycle refine @ 698 k | 698,426 | 4.00 ms | (60 % of a 1 M-Gaussian step is count-dependent) |
| bicycle refine @ 501 k | 500,757 | 3.33 ms | |
| bicycle delivery | 1,001,119 | ~4.84 ms (task-8 measurement) | |
| counter refine @ 434 k | 433,559 | 4.13 ms | cost(N) ≈ **2.66 ms fixed + 3.32 ms per 1 M Gaussians** |
| counter delivery | 619,595 | ~4.72 ms | (56 % of a 620 k-Gaussian step is fixed) |
| dl3dv/0b5635 whole run | 796 k → 559 k | ΔB −1.86 s for −30 % count | whole-run elasticity 0.21 (DL3DV's larger scene load / view count keeps the fixed part high) |

Consequences, all confirmed by the M1 measurement: a 30 % count cut reduces the
*post-prune* per-step cost by only ~20 % on bicycle (~11 % on counter), and since the
cut can only be applied after some training has already happened (1250 of 5000 steps),
the **whole-run B cut is 8–13 %, not 30 %**. The correct conversion on these budgets is
**≈ +0.028 TIME points per 1 % whole-run B cut** (bicycle: 13.2 % → +0.83; counter:
8.1 % → +0.53), i.e. a 20 % count cut purchased mid-run is worth ≈ +0.5 TIME points,
not +3.6. The remaining path to larger B cuts is to prune *earlier* (e.g. at iteration
500–800, before the bulk of densification) or to constrain the final count during
densification itself (Mini-Splatting's densification half); both are untested and would
trade more quality.

## 6. Why M1 is positive only on bicycle (scoring geometry)

The proxy's LPIPS term is `40·clip((0.25−LPIPS)/0.20)`, so:
* bicycle LPIPS 0.30 → the LPIPS term is **already 0**; any further LPIPS damage is free,
  and only PSNR/SSIM matter (bicycle is low-quality: q ≈ 10, PSNR 24.6, SSIM 0.73).
* counter LPIPS 0.18 → each +0.01 LPIPS costs 2 quality points; a 30 % prune costs
  −1.6 points there.
* the same holds for the DL3DV low-total scenes (LPIPS 0.23–0.25, i.e. just below or at
  the cliff) — M1 at 70 % is expected to be near break-even or slightly negative there,
  which the §4.1 `0b5635` cell tests.

Adoption rule implication: M1 cannot be selected per scene from test metrics (forbidden),
but a **train-time gate** is legitimate and plausible: the importance pass already gives
train-view PSNR/SSIM/LPIPS for 24 views at +0.09 s, so "prune if train-view LPIPS > 0.25"
(or a learned threshold) can be computed from train data only. That gate is the concrete
follow-up if the direction is pursued.

## 7. Incidents / data quality

* Serial, one GPU (lane 2 = GPU2) throughout; GPU0 was held by another teammate's run +
  a foreign tenant for the first minutes, so the driver ran with `--allow-busy` on the
  free GPU. Controls were re-measured in the same session; `ctrl` B 26.93/27.15 sits on
  the known N1 values (27.3/27.65) → the B comparison is sound (no contention observed
  on GPU2: phase-1 wall 78 s/83 s as usual).
* No `rm -rf` of out-roots; the driver RESUME-SKIPs completed cells via
  `<out>/<scene>/<arm>/sweep/summary.json`; one 0-byte cells-file incident at launch was
  fixed by writing the cells file in a separate ssh call.
* `research/ply_ops.py`, `research/ply_voxgrow.py`, `competition/upstream/` untouched.
  No git push.

## 8. Deliverables

| path | content |
|---|---|
| `records/MINISPLAT-20261030.md` | this report |
| `records/minisplat-20261030/ply_importance.py` | importance scoring + pruner + standalone CLI |
| `records/minisplat-20261030/run.sh` | serial entry point (M2 + M1 + controls, resume-safe) |
| `records/minisplat-20261030/cells.txt` | the cell matrix actually run |
| `records/minisplat-20261030/results.jsonl` | consolidated rows |
| `records/minisplat-20261030/summarize.py` | regenerates the tables above from `raw/` |
| `records/minisplat-20261030/raw/` | raw `summary.json` + `train_record.json` per cell |
| N1 `/root/gpufree-data/outputs/minisplat-20261030/` | raw cells, logs, snapshots |

## 9. Recommendation

1. **Adopt M1 (`--mid_prune_at 1250 --mid_prune_keep 0.5–0.6`) only behind a train-time
   gate.** On bicycle it is the best total of the whole project: **+0.85** (keep 0.50) to
   +0.84 (0.60), with B down 6.1/4.8 s. On counter −1.55 and on 0b5635 −0.42. The gate
   can be computed from train data only: the importance pass already gives train-view
   PSNR/SSIM/LPIPS for 24 views at +0.09 s, so "prune when train-view LPIPS is
   saturated" is legal (no test-RGB selection) and cheap. Without a gate, do not adopt.
2. **Do not adopt M2** (post-pass pruning): quality-neutral-ish at 70–90 % keep but it
   strictly adds B time (Δtotal −0.48…−2.52).
3. The B-time lever is real but sublinear: measured `cost(N) ≈ fixed + c·N` with
   40–60 % of a step independent of the Gaussian count, and B measured over the whole run
   (not just the pruned part). A 30 % count cut at iteration 1250 is worth **+0.4…+0.8
   TIME**, not the hoped +3.6; a 20 % cut ≈ +0.5.
4. The compression/density-control family is **closed for the post-pass framing**; the
   open variant is Mini-Splatting's densification half with an early constrained budget
   (prune at iteration ≤800, before the 500–1250 densification window, or cap the count
   inside that window). By the measured law that could be worth ~+2 TIME points but at a
   larger quality risk — worth one test only if the density direction is still a
   priority for the user.
