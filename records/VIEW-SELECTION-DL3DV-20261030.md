# VIEW-SELECTION-DL3DV-20261030 — view selection on the DL3DV bad cases: held-out triage, arc-uniform e2e, and next-gen variants

Owner: `h1-prep` (task-14 + task-16). Date: 2026-10-31 (Asia/Shanghai). Node: H1 (stage-A triage) + N1 (decisive e2e).

## TL;DR

1. **The held-out triage works and found a candidate.** Ranking on 8 fixed held-out *train* views
   (not context self-render), `arc-uniform` — 16 context frames uniform in **cumulative camera-centre
   arc length** instead of frame index — beats the index-uniform control on **4/4 scenes**:
   0b5635 +0.61 dB / LPIPS −0.025; 0f622ea +0.35 / −0.019; 3e4dbe +0.63 / −0.022;
   001dccbc (good case) +0.07 / −0.003. `covis-*` variants are consistently worse.
2. **But it does not survive end-to-end. arc-uniform is DROPPED.** Same-node paired delivery runs on
   N1 give **Δq = −0.089 (0b5635), −0.521 (0f622ea), −0.393 (3e4dbe)** — all ≤ 0, one beyond the
   −0.5 regression bound → adoption bar (Δq ≥ +0.5 on ≥2/3) fails. **Projected 13-case gain = −0.08
   points.** Mechanism: the init gains are real but the B-side optimisation absorbs them — e2e PSNR
   moves only +0.03/−0.09/+0.05 dB and e2e LPIPS gets *marginally worse* (+0.0009/+0.0017/+0.0019).
3. **A-side view selection is not a lever for the final score** (third independent negative:
   task-11 co-visibility, task-14 DL3DV held-out, now e2e). Stop spending GPU on it; the ceiling is
   ReSplat quality itself, and B + official metrics converge to the same neighbourhood for any
   reasonable 16-view subset.
4. Part B (next-gen cadence-aware variants) is **designed and CPU-selected but NOT triaged** (H1 GPU
   reserved). The cadence-aligned variants are flagged **protocol-aware / grey** and are not
   recommended for adoption.
5. `coverage-max` and `extreme-baseline` produced **identical selections** in task-14 (v1 rule's
   overlap constraint was non-binding → fallback collision). Fixed v2 rule (τ=60th pct of pairwise J)
   gives distinct selections but is untriaged. Documented in §5.

## 0. Background: why arc-length (the split structure)

The organizer split is exactly `llffhold=8`: **test = image index ≡ 0 (mod 8)** of the sequence in
sorted order; train indices are everything else (gaps 1–2). Module A's `context` = 16 **train** frames
chosen **uniformly by frame index** (`np.linspace`). For a video sequence whose camera speed is not
constant (DL3DV and hw captures), uniform-in-index samples the camera path unevenly — some context
frames are spatially near-duplicates, some parts of the path are under-sampled. Uniform in
**arc length** (cumulative camera-centre path length) fixes that with the same 16-frame budget and one
ReSplat forward pass. This is the hypothesis that stage A tested with a held-out metric.

## 1. Held-out triage metric (task-14 stage A; the fix to the task-11 lesson)

Context self-render PSNR is misleading (task-11: +1.6 dB while e2e −3.8 q). This task triages with a
**held-out train-view render**:

* For each scene pick **8 TRAIN views that are not in any candidate context set** — chosen once per
  scene, identical across candidates (evenly spaced over `train \ uniform_context`; candidate pools
  exclude them by construction).
* For each candidate: rewrite only `split["context"]` in a copied work dir, run ReSplat
  (`research.resplat_variant --num_refine 2 --fix-sh-basis`, one forward pass), then **render the
  init PLY at the 8 held-out views** with the pinned FastGS renderer
  (`research/holdout_render.py`, adapted from `render_eval.py`) and score with the **official**
  `competition/upstream/baseline/evaluate.py` (PSNR/SSIM/VGG-LPIPS vs the train GT images).
* Ranking uses **held-out LPIPS (asc) / PSNR (desc)**. Context self-render PSNR is recorded for
  continuity only. No test RGB is used anywhere in selection or triage.

Source work dirs: H1 `outputs/multi/dl3dv-{0b5635d5,0f622eaa,3e4dbe41,001dccbc}/work`
(512×960, budget 94.29). 21 candidates: 6 selections × 3 bad cases + 3 selections × 001dccbc.

Held-out triage table (task-14 stage A, H1; held-out = 8 fixed train views, never in any context set)

| scene | selection | hl PSNR | hl SSIM | hl LPIPS | ctx self-render PSNR | Δhl-PSNR | Δhl-LPIPS | gaussians | beats uniform (both §)? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| dl3dv-0b5635d5 | uniform | 19.313 | 0.5466 | 0.4152 | 22.689 | — | — | 450206 | control |
| dl3dv-0b5635d5 | covis-fps | 19.204 | 0.5456 | 0.4230 | 23.336 | -0.109 | +0.0078 | 459641 | no |
| dl3dv-0b5635d5 | covis-maxmin | 18.002 | 0.4857 | 0.4645 | 22.056 | -1.311 | +0.0493 | 435469 | no |
| dl3dv-0b5635d5 | coverage-max | 19.036 | 0.5347 | 0.4224 | 22.420 | -0.277 | +0.0072 | 443334 | no |
| dl3dv-0b5635d5 | extreme-baseline | 19.036 | 0.5347 | 0.4224 | 22.420 | -0.277 | +0.0072 | 443334 | no |
| dl3dv-0b5635d5 | arc-uniform | 19.921 | 0.5873 | 0.3902 | 22.337 | +0.608 | -0.0250 | 448276 | **YES** |
| dl3dv-0f622eaa | uniform | 18.306 | 0.4656 | 0.4016 | 20.854 | — | — | 476635 | control |
| dl3dv-0f622eaa | covis-fps | 17.586 | 0.4342 | 0.4217 | 20.879 | -0.720 | +0.0201 | 476160 | no |
| dl3dv-0f622eaa | covis-maxmin | 16.985 | 0.3906 | 0.4572 | 21.212 | -1.321 | +0.0556 | 480283 | no |
| dl3dv-0f622eaa | coverage-max | 18.460 | 0.4716 | 0.4049 | 20.799 | +0.154 | +0.0033 | 475110 | no |
| dl3dv-0f622eaa | extreme-baseline | 18.460 | 0.4716 | 0.4049 | 20.799 | +0.154 | +0.0033 | 475110 | no |
| dl3dv-0f622eaa | arc-uniform | 18.651 | 0.4952 | 0.3828 | 20.745 | +0.345 | -0.0188 | 476735 | **YES** |
| dl3dv-3e4dbe41 | uniform | 19.649 | 0.5292 | 0.4052 | 22.769 | — | — | 466887 | control |
| dl3dv-3e4dbe41 | covis-fps | 19.526 | 0.5200 | 0.4074 | 23.030 | -0.123 | +0.0022 | 469610 | no |
| dl3dv-3e4dbe41 | covis-maxmin | 19.673 | 0.5204 | 0.4059 | 22.914 | +0.024 | +0.0007 | 463641 | no |
| dl3dv-3e4dbe41 | coverage-max | 19.892 | 0.5333 | 0.4020 | 22.922 | +0.243 | -0.0032 | 465065 | **YES** |
| dl3dv-3e4dbe41 | extreme-baseline | 19.892 | 0.5333 | 0.4020 | 22.922 | +0.243 | -0.0032 | 465065 | **YES** |
| dl3dv-3e4dbe41 | arc-uniform | 20.283 | 0.5601 | 0.3831 | 22.764 | +0.634 | -0.0221 | 465278 | **YES** |
| dl3dv-001dccbc | uniform | 27.415 | 0.8717 | 0.2139 | 30.948 | — | — | 455234 | control |
| dl3dv-001dccbc | covis-fps | 24.905 | 0.8466 | 0.2502 | 31.668 | -2.510 | +0.0363 | 462439 | no |
| dl3dv-001dccbc | arc-uniform | 27.487 | 0.8736 | 0.2112 | 30.923 | +0.072 | -0.0027 | 455248 | **YES** |

**Ranking:** `arc-uniform` is the best-by-held-out selection on **all 4 scenes** and beats uniform on
both held-out metrics (PSNR and LPIPS) on all 4. `covis-fps`/`covis-maxmin` — which score highest on
context self-render — are the worst on held-out. `coverage-max` is 2nd on 3e4dbe (and identical to
`extreme-baseline`, see §5).

ctx self-render vs held-out ranking (bad cases)

| scene | rank by ctx self-render | rank by held-out LPIPS | the two agree? |
|---|---|---|---|
| dl3dv-0b5635d5 | covis-fps > coverage-max > extreme-baseline > arc-uniform > covis-maxmin | arc-uniform > coverage-max > extreme-baseline > covis-fps > covis-maxmin | no |
| dl3dv-0f622eaa | covis-maxmin > covis-fps > coverage-max > extreme-baseline > arc-uniform | arc-uniform > coverage-max > extreme-baseline > covis-fps > covis-maxmin | no |
| dl3dv-3e4dbe41 | covis-fps > coverage-max > extreme-baseline > covis-maxmin > arc-uniform | arc-uniform > coverage-max > extreme-baseline > covis-maxmin > covis-fps | no |

The inversion is unambiguous: context self-render ranks `arc-uniform` 4th/5th of 5 on all three bad
cases, held-out ranks it **1st**. This is the second independent confirmation of the task-11 lesson.

## 2. Arc-uniform e2e on N1 (the decisive test, task-16 part A): DROPPED

**Setup.** The 3 `arc-uniform` init PLYs (+ the 3 matching clean-13 `uniform` init PLYs) were copied
H1→N1 to `/root/gpufree-data/data/{arc-uniform,uniform-clean13}/<scene>/resplat_world.ply`
(byte sizes verified = H1 source). N1 work dirs were verified **identical** to H1's (split.json and
`sparse/0/images.txt` sha256 match, same image counts, image_cache present), so the comparison is
paired: same work dir, same recipe, same node, same session, **init is the only variable**.

Delivery recipe unchanged (module B untouched): `research.run_sweep --iterations 5000
--position-lr-max-steps 5000 --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01
--densify-from 500 --densify-until 1250 --realloc-frac 0.15 --realloc-at 1250 --seed 20261015
--budget 94.29 --image-cache <work>/image_cache --profile competition/profiles/gpufree-0.json`.
Serial on one GPU, each cell quiet-gated (GPU < 50 MiB, no foreign research process, loadavg < 8).

Same-node paired e2e (N1, serial quiet-gated, delivery recipe S=5000)

| scene | arm | PSNR | SSIM | LPIPS | q | B(s) | TIME | total | Δq | Δtotal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dl3dv-0b5635d5 | uniform ctrl | 23.911 | 0.8160 | 0.2465 | 17.109 | 30.88 | 14.945 | 32.053 | — | — |
| dl3dv-0b5635d5 | **arc-uniform** | 23.945 | 0.8168 | 0.2474 | 17.020 | 30.75 | 14.975 | 31.995 | **-0.089** | -0.059 |
| dl3dv-0f622eaa | uniform ctrl | 22.558 | 0.7955 | 0.2297 | 17.746 | 32.63 | 14.532 | 32.279 | — | — |
| dl3dv-0f622eaa | **arc-uniform** | 22.472 | 0.7941 | 0.2314 | 17.226 | 33.02 | 14.439 | 31.665 | **-0.521** | -0.614 |
| dl3dv-3e4dbe41 | uniform ctrl | 24.304 | 0.8185 | 0.2290 | 21.123 | 31.16 | 14.879 | 36.003 | — | — |
| dl3dv-3e4dbe41 | **arc-uniform** | 24.358 | 0.8178 | 0.2309 | 20.730 | 30.17 | 15.112 | 35.842 | **-0.393** | -0.160 |

**Verdict: NOT ADOPTED.** The adopt bar was Δq ≥ +0.5 on ≥2 of 3 bad cases, no case regressing >0.5.
Observed Δq: **−0.089 / −0.521 / −0.393** (0/3 pass; 0f622ea breaches the −0.5 bound).
**Projected 13-case gain = mean Δq over the 3 bad cases / 13 = −1.00/13 ≈ −0.08 points.**
Same-node controls reproduce clean-13 *quality* almost exactly (q 17.109/17.746/21.123 vs clean-13
17.16/17.93/21.09; totals differ by ~+0.7–1.0 via N1's lower B/TIME), confirming the paired design.

**Why the init gain vanishes.** The +0.35–0.63 dB init gain on 8 held-out views is smaller than what
the 5k-step B-side optimisation converges past for any init: e2e PSNR ends +0.03/−0.09/+0.05 dB
relative to the uniform control, and e2e **LPIPS** (40 of the 80 quality points) is *slightly worse*
(+0.0009/+0.0017/+0.0019) — enough to make q net negative. B seconds are unchanged (±1 s), so this is
not a timing artefact. The final metric is dominated by what the 5k-step optimisation + official
metrics can extract, not by which 16 views seeded the init. Consequence: **do not pursue A-side
context-view selection further** — it is the third independent negative for this lever.

## 3. Next-generation variants (task-16 part B; designed + CPU-selected, triage pending H1 GPU)

Motivated by the same split structure, four variants were built (`select_context3.py`, CPU only; same
fixed held-out 8-view set so numbers stay comparable when triage runs):

* `arc-uniform` — baseline (uniform cumulative arc length).
* `arc-posediv` — endpoints fixed, then greedy max-min in the combined space
  `d = sqrt(Δs_norm² + (0.25·Δθ/π)²)` (equal arc spacing **and** angular diversity).
* `arc-cadence-mid` — **GREY**: only train frames with index ≡ 4 (mod 8) (midway between test frames
  ≡0 mod 8), 16 uniform in arc.
* `arc-cadence-adj` — **GREY**: only train frames with index ≡ ±1 (mod 8) (adjacent to a test frame),
  16 uniform in arc.

Next-gen variants (task-16 part B, CPU selection only — triage pending H1 GPU)

| scene | variant | mean arc gap | rotation span | mean pair angle | grey? | status |
|---|---|---:|---:|---:|---|---|
| dl3dv-0b5635d5 | arc-uniform | 0.0667 | 150.8° | 85.8° | no | selection built; triage pending H1 |
| dl3dv-0b5635d5 | arc-posediv | 0.0667 | 150.9° | 85.9° | no | selection built; triage pending H1 |
| dl3dv-0b5635d5 | arc-cadence-mid | 0.0663 | 153.0° | 85.4° | yes | selection built; triage pending H1 |
| dl3dv-0b5635d5 | arc-cadence-adj | 0.0656 | 150.4° | 85.6° | yes | selection built; triage pending H1 |
| dl3dv-0f622eaa | arc-uniform | 0.0667 | 175.2° | 78.7° | no | selection built; triage pending H1 |
| dl3dv-0f622eaa | arc-posediv | 0.0667 | 170.6° | 77.1° | no | selection built; triage pending H1 |
| dl3dv-0f622eaa | arc-cadence-mid | 0.0656 | 175.8° | 79.4° | yes | selection built; triage pending H1 |
| dl3dv-0f622eaa | arc-cadence-adj | 0.0650 | 175.2° | 78.9° | yes | selection built; triage pending H1 |
| dl3dv-3e4dbe41 | arc-uniform | 0.0667 | 173.1° | 74.8° | no | selection built; triage pending H1 |
| dl3dv-3e4dbe41 | arc-posediv | 0.0667 | 171.2° | 76.9° | no | selection built; triage pending H1 |
| dl3dv-3e4dbe41 | arc-cadence-mid | 0.0655 | 170.9° | 75.0° | yes | selection built; triage pending H1 |
| dl3dv-3e4dbe41 | arc-cadence-adj | 0.0653 | 173.3° | 74.3° | yes | selection built; triage pending H1 |

**Status: selections built (`candidates2.json`, 12 candidates); held-out triage and e2e were NOT run**
because H1's GPU is reserved (Lead's exact-13 measurement + VGG200 arm) and the task requires Lead
coordination before any H1 ReSplat. Given §2 (arc-uniform's held-out win did not survive e2e), the
expected value of triaging these is low; `arc-posediv` is the only non-grey variant still worth a
cheap held-out check if H1 GPU is ever free.

**Grey-area statement (explicit).** `arc-cadence-mid` and `arc-cadence-adj` use only the *known split
rule* (test ≡ 0 mod 8) and **never any test RGB**, so they are not "uses test data" cheating. But they
are **protocol-aware**: they exploit knowledge of the organiser's evaluation cadence to place context
views where they most benefit the test frames — a grey exploit, not a scene-understanding improvement,
and it would not generalise to an unknown hold convention. Recommendation: **do not adopt
cadence-aware selection**; keep it documented as an analysis variant only. The optional B-side
"up-weight train cameras adjacent to held-out frames" variant is the same grey category and was
therefore not implemented.

## 4. Constraints / legal checks

* Selection and triage use **train views only**; test RGB was never read. The held-out 8 views are
  *train* views (never in any candidate context set), rendered from the init and compared to their GT.
* 16-frame context cap and one ReSplat forward pass respected (12-frame variant is **not runnable**
  without touching upstream: `competition/upstream/baseline/run_resplat.py` pins
  `len(context) == min(16, len(train))`).
* Module B and `competition/upstream/` untouched; delivery recipe unchanged; no per-scene
  hyperparameters; no git push.
* e2e cells are same-node paired, serial, quiet-gated; the uniform control was re-run in-session.
* N1 was otherwise idle; H1 GPU was not used for any GPU stage after the Lead's stop request
  (task-14 e2e was superseded by this N1 e2e).

## 5. Two implementation notes

* **`coverage-max` == `extreme-baseline` (fallback collision).** In task-14 stage A the two produced
  *identical* context sets on all 3 bad cases (same gaussians / ctx / held-out metrics): the v1
  `extreme-baseline` rule's min-overlap constraint (25th pct of pairwise J) was non-binding, so the
  rule degenerated to pure camera-centre maximin = `coverage-max`. Fixed version (`--tau-pct 60`,
  `select_context2.py` + `selection-v2/`) produces distinct sets (3–4 views overlap with v1) but is
  untriaged; this report treats the `extreme-baseline` rows as `coverage-max` rows.
* **12-frame variant**: blocked by the upstream wrapper's exactly-16 validation (would require
  modifying protected code) — documented as untested.

## 6. Files

Mac (versioned): `FFSplat-Hackathon/records/view-selection-dl3dv-20261030/`
* `e2e_results.jsonl` / `e2e_results.json` — the 6 N1 e2e cells + paired Δq/Δtotal/ΔB
* `results-stage1.json` — 21 held-out-triage rows (task-14 stage A)
* `candidates2.json` — 12 next-gen candidates (context names, grey flag, stats, planned pipeline)
* `select_context2.py` (task-14 selections + τ option), `select_context3.py` (next-gen variants),
  `prepare2.py`, `run2.py`, `run2_finish.py`, `build_v2.py`, `holdout_render.py`, `run.sh`
* `selection/{scene}.json`, `selection-v2/{scene}.json`, `selection3/{scene}.json`,
  `holdout/*/init/metrics.json`, `logs/`

H1 raw: `outputs/viewsel-dl3dv-20261030/` (21 candidate work dirs with rewritten `split.json`,
`resplat/<scene>/<method>/` init PLYs + `resplat.json`, `holdout/`, `logs/`).
N1 raw: `/root/gpufree-data/outputs/arcuniform-20261030/` (`runs/<scene>/{uniform-clean13,arc-uniform}/`,
`e2e_results.json`, `results.jsonl`, `heldout/`, `selection3/`, `logs/`).
Init PLYs on N1: `/root/gpufree-data/data/{arc-uniform,uniform-clean13}/<scene>/resplat_world.ply`.

## 7. Exact commands

```bash
# held-out metric (fastgs env): render init PLY at 8 held-out train views
python research/holdout_render.py --work <prepared work> --ply <init.ply> \
  --views <8 held-out train names> --out <eval dir> --quiet
# official metrics (resplat env)
python competition/upstream/baseline/evaluate.py --model_path <eval dir> --device cuda:0

# H1: task-14 selection variants (CPU) then ReSplat per candidate
python /root/select_context2.py --work <work> --ply <world.ply> --out selection/<scene>.json --scene <scene>
python -m research.resplat_variant --work <work> --out resplat/<scene>/<method> \
  --checkpoint /workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth \
  --num_refine 2 --device cuda:0 --fix-sh-basis

# N1 e2e (delivery recipe), init = arc-uniform / uniform-clean13 PLY
python -m research.run_sweep --case '<case>' --work <n1 work> --init-ply <init.ply> \
  --out runs/<scene>/<method> --iterations 5000 --snapshots '' --budget 94.29 \
  --position-lr-max-steps 5000 --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 \
  --densify-from 500 --densify-until 1250 --realloc-frac 0.15 --realloc-at 1250 \
  --image-cache <n1 work>/image_cache --profile competition/profiles/gpufree-0.json --seed 20261015
```
