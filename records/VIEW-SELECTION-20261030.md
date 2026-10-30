# VIEW-SELECTION-20261030 — co-visibility 16-view context selection for the ReSplat init

Owner: `h1-prep` (task-11). Date: 2026-10-31 (Asia/Shanghai). Node: H1 (1×4090 48G), serial.
Question: is *which* 16 frames we feed ReSplat a legal quality lever worth taking over the current
uniform `np.linspace` selection? **Answer: no — co-visibility / coverage-based selection does not
improve the e2e delivery score; the context self-render PSNR triage metric is actively misleading.
Keep the uniform selection. This direction is closed.**

## TL;DR

| candidate family | triage (Δ context self-render PSNR vs uniform) | e2e (Δ quality points vs uniform control) | adopted? |
|---|---|---|---|
| covis-cover / covis-maxmin / covis-fps | −1.11 … **+1.60 dB** | bicycle **+0.36**, counter **−1.62 / −3.77**, 001dccbc **−2.33** | **NO** |

Adoption rule from the task (Δq ≥ +0.5 on ≥2 of 3 scenes, no scene regressing > 0.5):
**FAILS** — the best e2e quality delta is +0.36 (below +0.5 and inside single-run noise), while two
scenes lose 1.6–3.8 quality points. The direction is not adopted; the uniform selection stays.

Key secondary finding (worth keeping): **context self-render PSNR is not a valid proxy for init
quality.** It can be raised by +0.75 … +1.60 dB while the e2e quality gets *worse* — the metric
rewards selections whose 16 views are mutually easy to reconstruct (redundant / high-overlap views)
rather than views that generalise to the held-out test cameras. The *largest* triage gain
(counter, covis-maxmin, +1.60 dB) produced the *worst* e2e regression (−3.77 q).
→ Any future A-side selection work must triage with a held-out (train-only) render metric, not the
context self-render.

## 1. What was varied (and what was not)

Only `split["context"]` (the 16 image names ReSplat consumes as context) was rewritten in a copied
work dir; `images`, `train`, `test`, `sparse/`, `module_a.json` and `image_cache` are symlinked to
the unchanged source work dirs. Module A, ReSplat weights/architecture (`num_refine=2`, one forward
pass, ≤16 frames), and module B (delivery recipe S=5000) are untouched. A/ReSplat are not in the
scored TIME, so this is a pure quality lever.

- Scenes (3): `mipnerf360/bicycle` (worst LPIPS ≈0.30), `mipnerf360/counter`, `dl3dv/001dccbc` (best-case sanity).
- Sources: `outputs/cand/mipnerf360-bicycle/work`, `outputs/cand/mipnerf360-counter/work`,
  `outputs/multi/dl3dv-001dccbc/work` (the clean13 work dirs; uniform control = their `split["context"]`).
- Selection candidates are always a subset of `split["train"]` (169 / 210 / 288 candidates) — **test
  images are never touched in selection**; they are only used by the official e2e metrics.

## 2. Co-visibility graph (CPU, `select_context.py`)

No COLMAP tracks were available for these prepared dirs (the exported `sparse/0` has only
`cameras.txt`/`images.txt`; the raw mip datasets' `sparse/0` also has no `points3D.*`; no
`database.db` for these scenes). So co-visibility is derived from the **ReSplat world model itself**:

1. sample `N=30000` 3-D points (Gaussian centres) from the scene's exported PLY
   (`outputs/clean13-20261030/resplat/<scene>/resplat_world.ply`);
2. project them into every **train** camera (PINHOLE intrinsics from `sparse/0`; visible = in front,
   inside the image); `V[v,p] = 1`;
3. pairwise co-visibility kernel `J_ij = |V_i ∩ V_j| / |V_i ∪ V_j|` (Jaccard, 0…1).

Selections (all exactly 16 train views):
- **uniform** — control: the existing `np.linspace` context of the prepared split.
- **covis-maxmin** — greedy max-min Jaccard distance (start from the highest-J pair, repeatedly add
  the view with the largest minimum `1−J` to the selected set; tie-break larger visible count).
- **covis-cover** — greedy marginal coverage of the visible-point union (add the view covering the
  most not-yet-covered points; tie-break larger min Jaccard distance).
- **covis-fps** — farthest-point sampling on camera centres restricted to the top-60 %
  co-visibility core (diversity inside the mutually-covisible component).

Kernel statistics of the selected sets (coverage = union of visible sampled points / all-train union):

| scene | selection | mean pair J | min pair J | coverage | mean visible pts |
|---|---|---:|---:|---:|---:|
| bicycle | uniform | 0.305 | 0.050 | 0.993 | 9,410 |
| bicycle | covis-cover | 0.219 | 0.007 | 0.999 | 7,726 |
| bicycle | covis-maxmin | 0.128 | 0.000 | 0.995 | 5,698 |
| bicycle | covis-fps | 0.491 | 0.222 | 0.921 | 10,279 |
| counter | uniform | 0.590 | 0.007 | 0.996 | 18,510 |
| counter | covis-cover | 0.545 | 0.034 | 1.000 | 18,087 |
| counter | covis-maxmin | 0.414 | 0.007 | 0.972 | 13,962 |
| counter | covis-fps | 0.692 | 0.511 | 0.956 | 19,388 |
| 001dccbc | uniform | 0.661 | 0.438 | 0.999 | 20,263 |
| 001dccbc | covis-cover | 0.646 | 0.437 | 1.000 | 20,444 |
| 001dccbc | covis-maxmin | 0.659 | 0.434 | 0.997 | 20,156 |
| 001dccbc | covis-fps | 0.724 | 0.517 | 0.937 | 19,837 |

Note: the uniform control is already near-optimal on this kernel (coverage 0.993–0.999 and high
mean overlap on 001dccbc) — consistent with earlier fps/coverage findings being neutral: for these
ordered captures `linspace` already spans the scene.

## 3. Stage 1 — ReSplat init per candidate (triage)

Each candidate: copy work dir with rewritten `context`, re-run
`research.resplat_variant --num_refine 2 --device cuda:0 --fix-sh-basis`, then measure the
**context self-render PSNR** (mean PSNR of ReSplat's 16 context renders vs the GT context images;
same script as clean13). All serial, quiet-gated. Control determinism check: re-running the
**uniform** selection reproduces the clean13 numbers to 10 decimals
(21.40678823605526 / 26.638701895514906 / 30.947608092563776) → ReSplat is deterministic and the
clean13 controls are valid baselines.

| scene | selection | gaussians | ctx self-render PSNR | Δ vs uniform | eligibility (≥+0.3 dB) |
|---|---|---:|---:|---:|---|
| bicycle | uniform | 520,980 | 21.4068 | — | control |
| bicycle | covis-cover | 518,044 | 20.2999 | −1.1069 | no |
| bicycle | covis-maxmin | 532,106 | 20.7934 | −0.6134 | no |
| bicycle | **covis-fps** | 506,702 | 22.1557 | **+0.7489** | **yes → e2e** |
| counter | uniform | 488,831 | 26.6387 | — | control |
| counter | covis-cover | 480,626 | 27.0953 | +0.4566 | yes → e2e (extra run) |
| counter | **covis-maxmin** | 508,846 | 28.2381 | **+1.5994** | **yes → e2e** |
| counter | covis-fps | 480,130 | 25.8586 | −0.7801 | no |
| 001dccbc | uniform | 455,234 | 30.9476 | — | control |
| 001dccbc | covis-cover | 459,140 | 30.7827 | −0.1649 | no |
| 001dccbc | covis-maxmin | 454,062 | 30.4695 | −0.4781 | no |
| 001dccbc | **covis-fps** | 462,439 | 31.6677 | **+0.7201** | **yes → e2e** |

## 4. Stage 2 — full delivery e2e on the triage-passing candidates

Delivery recipe unchanged (S=5000, `position_lr_max_steps 5000`, no opacity reset, λdssim 0.6,
scaling_lr 0.01, densify 500→1250, realloc 0.15@1250, same image_cache, same budgets); the new
`resplat_world.ply` is the init. Controls = clean13 delivery rows (same work dirs, same recipe,
quiet-gated) for these 3 scenes.

| scene | selection | Δ ctx PSNR | quality (ctrl → cand) | **Δq** | B(s) | total (ctrl → cand) | **Δtotal** |
|---|---|---:|---|---:|---:|---|---:|
| mipnerf360/bicycle | covis-fps | +0.749 | 9.76 → 10.12 | **+0.362** | 31.67 | 25.310 → 24.935 | −0.375 |
| mipnerf360/counter | covis-maxmin | +1.599 | 40.28 → 36.51 | **−3.774** | 30.28 | 55.172 → 51.459 | −3.713 |
| mipnerf360/counter | covis-cover | +0.457 | 40.28 → 38.66 | **−1.618** | 28.82 | 55.172 → 53.966 | −1.206 |
| dl3dv/001dccbc | covis-fps | +0.720 | 64.46 → 62.13 | **−2.332** | 31.54 | 78.942 → 76.916 | −2.026 |

Full metrics of the e2e candidates:

| scene | selection | PSNR | SSIM | LPIPS | q | B(s) | TIME | total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | covis-fps | 24.683 | 0.7297 | 0.2951 | 10.1224 | 31.67 | 14.813 | 24.935 |
| counter | covis-maxmin | 27.293 | 0.8944 | 0.1944 | 36.5064 | 30.28 | 14.953 | 51.459 |
| counter | covis-cover | 27.853 | 0.8980 | 0.1873 | 38.6624 | 28.82 | 15.304 | 53.966 |
| 001dccbc | covis-fps | 33.885 | 0.9639 | 0.1149 | 62.1280 | 31.54 | 14.788 | 76.916 |

(controls: bicycle 25.310 total / q 9.76 / B 28.52; counter 55.172 / 40.28 / B 30.55;
001dccbc 78.942 / 64.46 / B 32.83 — from `records/clean13-20261030/`.)

**Rule check (task):** Δq ≥ +0.5 on ≥2 of 3 scenes → only bicycle is positive (+0.36 < +0.5);
counter regresses −1.62/−3.77 and 001dccbc −2.33 (>0.5 regression). **Not adopted.**

## 5. Why the triage metric fails (the reusable finding)

The context self-render PSNR measures how well ReSplat reproduces *the very 16 frames it was given*.
A selection can maximise it by choosing mutually redundant views (high pairwise Jaccard): the
network sees the same content many times and renders those views very well, while providing less
diverse coverage for the actual held-out test cameras. This is exactly what happens here:

- counter `covis-maxmin` has the *lowest* mean pair overlap of its selections (0.414) yet the
  highest ctx PSNR (+1.60 dB) — because its 16 views are easy to fit individually? No: the key is
  the metric rewards self-consistency, and the e2e init quality moved the opposite way (−3.77 q).
- bicycle/001dccbc `covis-fps` selects the high-overlap core (mean J 0.49 / 0.72, coverage 0.92 /
  0.94) and gets higher ctx PSNR (+0.75/+0.72) but flat-to-negative e2e (bicycle +0.36 q,
  001dccbc −2.33 q).
- Across the 4 e2e points the rank correlation between Δctx and Δq is **negative**
  (Δctx +0.75/+1.60/+0.46/+0.72 → Δq +0.36/−3.77/−1.62/−2.33): using ctx self-render PSNR to pick
  among selections would pick the worst candidate.

**Recommendation:** do not use context self-render PSNR as a triage for view selection again; if a
future selection experiment is attempted, score candidates by a *train-only held-out render*
(e.g. render held-out train views that are not in the 16 context views), which is what the e2e
quality actually tracks.

## 6. Constraints / caveats

- 16-frame cap and exactly one ReSplat forward pass preserved; A/ReSplat are not scored, so the
  only adoption criterion used is quality (Δq) — as in the task rule.
- Module B, `competition/upstream/`, `run_pipeline.py`, `DELIVERY.md` untouched; no git push; N1 not used.
- Single e2e run per candidate (no repeats). Single-scene delivery rerun noise is ~0.2–0.7 points
  (task-2 σ≈0.22 on 13-case means), so bicycle's +0.36 q is inside noise; the counter (−1.6…−3.8)
  and 001dccbc (−2.3) regressions are far outside it → the negative conclusion is robust.
- e2e runs were quiet-gated (`gpu<50 MiB`, `loadavg1<4`, no foreign research/colmap process ×2
  samples); the 4 runs show B = 28.8–31.7 s. B is init-dependent and secondary here (TIME is 20
  points); the adoption decision uses Δq.
- The co-visibility graph uses the exported Gaussian centres as scene geometry (documented proxy);
  it is not COLMAP-track covisibility because these prepared scenes do not carry `points3D`/matches.
- The 4th e2e row (counter `covis-cover`, the only other triage-eligible candidate) was run as an
  extra check; its small logging wrapper hit `KeyError('total_points')` *after* `summary.json` was
  written, so its metrics were reconstructed from `runs/mipnerf360-counter/covis-cover/summary.json`
  (noted in `results-e2e-extra.json`). The 12 triage rows and the first 3 e2e rows come from the
  main runner unchanged.

## 7. Files

Mac (versioned): `FFSplat-Hackathon/records/view-selection-20261030/`
- `results.jsonl` — 12 triage rows + 4 e2e rows (16 lines)
- `results-stage1.json`, `results-e2e.json`, `results-e2e-all.json`, `results-e2e-extra.json`
- `select_context.py` (co-visibility + selections), `viewsel_prepare.py` (candidate work dirs),
  `viewsel_run.py` (serial quiet-gated runner), `run.sh` (exact commands)
- `selection/{scene}.json` — per-scene kernel stats + the 16 chosen image names for every method
- `candidates.json`, `resplat/*/*/resplat.json`, `logs/` (run/stage logs, ctx PSNR, e2e summaries)

H1 raw: `/workspace/organizer-projects/FFSplat-Hackathon/outputs/viewsel-20261030/`
(`selection/`, `work/` candidate dirs with rewritten `split.json` + symlinked inputs, `resplat/`,
`runs/`, `logs/`, `results*.json[l]`).
