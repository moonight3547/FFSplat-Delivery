# Module-B evidence summary (2026-10-30)

Consolidated from the archived experiment JSONs only: 11 `summary.json` files under `records/artifacts-20261030/**` (40 scored module-B rows), the official baseline `records/o0-baseline-20261030.json`, and the narrative in `docs/RESEARCH-20261030.md`. Rows are parsed with the existing `research/collect.py::rows_from` parser; no metric, score or timing is recomputed here beyond re-deriving the proxy from the stored PSNR/SSIM/LPIPS/B columns as a cross-check. Nothing was run on a GPU; no file outside `records/SUMMARY.md` and `records/best-recipe.json` was written.

## 1. Scoring proxy and budget

Total = quality (80) + TIME (20). Quality = PSNR 20 (linear 15–40 dB) + SSIM 20 (linear 0.70–0.95) + LPIPS 40 (linear 0.25→0.05, lower is better); each component is clamped to its range, so a metric outside its band contributes 0 or full. TIME = 20 · clamp((1 − r)/(1 − 0.10), 0, 1) with r = B_seconds / budget, i.e. linear in B/budget and full marks at ≤ 10% of the budget, zero at ≥ 100%.

Budget is the *same-GPU* FastGS reference with **random init, 30,000 steps** (not classic-3DGS 30k, not ReSplat-continued 30k): **94.29 s (dl3dv), 99.49 s (hw/billboard), 92.57 s (mipnerf360/counter)**. These are the `budget_seconds` values stored in `records/o0-baseline-20261030.json` and repeated in the header of every module-B `summary.json`; the older `records/fastgs-random30k-calibration.json` still holds the 10-29 calibration totals (102.68 / 100.13 / 93.72), which are *not* the budgets used here.

Cross-check: re-deriving quality and TIME from the stored PSNR/SSIM/LPIPS/B columns reproduces the stored `quality_points`, `time_points` and `total_points` for all 40 rows (max deviation 1.1e-3 points, entirely from the 3–4-decimal rounding of the stored metrics). `total_points` equals `quality_points + time_points` for all 40 rows. Every row with `b_seconds_estimate` > budget also has `score.valid = false` and `time_points = 0.0` — the flag and the arithmetic agree, 9 rows in total.

## 2. Module-B sweep rows (best row per experiment tag × case)

All 40 archived rows were collected; the table keeps the best-`total_points` row per (config, case) to stay readable — 11 rows. Rows sorted by case, then by total. **INVALID** = `b_seconds_estimate` exceeds that case's budget, so the run would not be scored under the rules (the stored `score.valid` is false and TIME is 0 in those rows).

| case | config (tag) | iters | B (s) | B/budget | PSNR | SSIM | LPIPS | Q | TIME | total | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dl3dv | noreset-30kLR (`dl3dv-noreset`) | 8,000 | 51.71 | 54.8% | 33.336 | 0.9511 | 0.1324 | 58.189 | 10.036 | **68.225** | ok |
| hw/billboard | noreset-30kLR (`hw-noreset`) | 4,000 | 22.64 | 22.8% | 22.684 | 0.7885 | 0.1761 | 28.007 | 17.166 | **45.173** | ok |
| hw/billboard | dense-900k-noreset (`hw-dense-noreset`) | 8,000 | 48.49 | 48.7% | 22.729 | 0.7975 | 0.1573 | 32.523 | 11.391 | **43.914** | ok |
| hw/billboard | default-30kLR (reset 3000) (`hw-default`) | 4,000 | 29.73 | 29.9% | 22.595 | 0.7843 | 0.1841 | 26.000 | 15.583 | **41.583** | ok |
| hw/billboard | res-ramp 0.5>1.0 (`hw-resramp`) | 8,000 | 57.28 | 57.6% | 22.545 | 0.7836 | 0.1775 | 27.224 | 9.428 | **36.652** | ok |
| hw/billboard | default-30kLR, early probe (`hw-probe`) | 400 | 8.63 | 8.7% | 22.171 | 0.7499 | 0.2177 | 16.189 | 20.000 | **36.189** | ok |
| hw/billboard | lpips-0.2 (reset 3000) (`hw-lpips`) | 4,000 | 54.91 | 55.2% | 22.313 | 0.7649 | 0.1938 | 22.282 | 9.957 | **32.239** | ok |
| mipnerf360/counter | noreset-30kLR (`counter-noreset`) | 8,000 | 53.95 | 58.3% | 27.552 | 0.8833 | 0.2079 | 33.126 | 9.270 | **42.396** | ok |
| mipnerf360/counter | default-30kLR (reset 3000) (`counter-default`) | 8,000 | 64.40 | 69.6% | 27.330 | 0.8808 | 0.2142 | 31.488 | 6.763 | **38.251** | ok |
| mipnerf360/counter | res-ramp 0.5>1.0 (`counter-resramp`) | 8,000 | 89.28 | 96.4% | 27.009 | 0.8712 | 0.2230 | 28.703 | 0.790 | **29.494** | ok |
| mipnerf360/counter | lpips-0.2 (reset 3000) (`counter-lpips`) | 8,000 | 128.87 | 139.2% | 26.716 | 0.8559 | 0.2199 | 27.865 | 0.000 | **27.865** | INVALID (over budget) |

Reading notes:

- `counter-lpips` 8k is over budget (128.87 s vs 92.57 s), so that tag's best *scoring* row is 4,000 iters / 20.885 total (B 69.89 s); even that is the weakest counter result in the archive.

- `hw-probe` is the same configuration as `hw-default`, just scored at 200/400 iterations; it is listed separately because it is a separate artifact.

- Config identity: *default-30kLR* = official FastGS schedule (opacity reset every 3,000, λ_dssim 0.2, scaling_lr 0.005, `position_lr_max_steps` 30000, densify 500–15000). *noreset-30kLR* = the same but with `opacity_reset_interval` (12001 for hw/dl3dv, 8001 for counter) beyond the run length. *lpips-0.2* adds a λ_lpips 0.2 term to B's loss at reset 3000. *res-ramp* adds `res_scales` [0.5, 0.75, 1.0] / `res_until` [0.6, 0.85]. *dense-900k-noreset* adds `densify_count_thresh` 2 and `max_gaussians` 900000 to noreset.

- Coverage gap: 8 of the 11 tags cover a single case; the remaining three are the per-case noreset artifacts. At config level the default schedule covers only hw + counter (`hw-default`, `counter-default`, plus `hw-probe`), and there is **no dl3dv run of the default schedule, of lpips, of res-ramp or of dense**. So only the noreset schedule and the O0 baseline are 3-case-complete and comparable as a uniform configuration. `summary.json` files under `sweeps/` predate the `lpips_weight`/`res_scales` keys (older schema, keys absent rather than zero); round2/round3 files carry them.

## 3. Best per case (best valid configuration found)

| case | config (tag) | iters | B (s) | PSNR | SSIM | LPIPS | Q | TIME | total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dl3dv | noreset-30kLR (`dl3dv-noreset`) | 8,000 | 51.71 | 33.336 | 0.9511 | 0.1324 | 58.189 | 10.036 | **68.225** |
| hw/billboard | noreset-30kLR (`hw-noreset`) | 4,000 | 22.64 | 22.684 | 0.7885 | 0.1761 | 28.007 | 17.166 | **45.173** |
| mipnerf360/counter | noreset-30kLR (`counter-noreset`) | 8,000 | 53.95 | 27.552 | 0.8833 | 0.2079 | 33.126 | 9.270 | **42.396** |
| **mean** | | | 42.77 | | | | | | **51.931** |

Best-per-case uses three different iteration counts of the *same* noreset-30kLR configuration: 8k for dl3dv, 4k for hw, 8k for counter. So "best per case" buys only **51.931** mean over the single best uniform configuration (51.735), i.e. +0.196.

## 4. Best uniform configuration (one setting for all 3 cases)

| rank | uniform configuration | dl3dv | hw/billboard | mipnerf360/counter | mean total | vs baseline |
|---:|---|---:|---:|---:|---:|---:|
| 1 | noreset-30kLR, 8,000 iters (one setting, three cases) | 68.225 | 44.584 | 42.396 | **51.735** | +2.106 |
| 2 | baseline: default-30kLR, 10,000 iters (official O0) | 62.933 | 43.263 | 42.692 | **49.629** | +0.000 |
| 3 | noreset-30kLR, 4,000 iters (one setting, three cases) | 66.350 | 45.173 | 36.384 | **49.302** | -0.327 |

Only three 3-case-complete, valid candidates exist in the archive: the O0 baseline (default schedule, 10k), noreset-30kLR at 4k, and noreset-30kLR at 8k. **The winner is noreset-30kLR at 8,000 iterations, mean 51.735 vs baseline 49.629 (+2.106).** Its per-case settings differ only in the literal `opacity_reset_interval` (12001 vs 8001), each chosen to exceed that case's run length, i.e. the same "reset disabled" setting in all three.

What does **not** reproduce the hint: the hinted winner (no opacity reset + λ_dssim 0.6 + scaling_lr 0.01 + matched position-LR schedule + 4k steps, mean ≈ 60) has **no JSON artifact anywhere in the repository**. It appears only in `docs/RESEARCH-20261030.md` §10 as 59.02 (FastGS densify) / 60.03 (opacity-sharpened init), and the supporting λ_dssim sweep (§8) and matched-LR runs (§9, mean 58.24) are documented text only; those runs happened on other machines and were never archived under `records/`. The archived data therefore caps the uniform winner at **51.735**, not ~60. See §5.2.

## 5. Falsified ideas

### 5.1 Things the JSON directly kills

| # | falsified idea | evidence | source |
|---:|---|---|---|
| 1 | More ReSplat refine rounds (num_refine 2 -> 4 -> 6) keep improving the initialization | hw quality 3.813 (r2) -> 3.874 (r4) -> 3.880 (r6): +0.067 for 4 extra rounds, and counter goes the other way 5.158 (r2) -> 4.985 (r4) -> 4.926 (r6). Recursion saturates after round 2; these are B=0 (initialization-only) rows with time_points=null. | `records/artifacts-20261030/probe/init-{dl3dv,hw,counter}/score_summary.json, records/artifacts-20261030/experiments/round1/*-score/score_summary.json` |
| 2 | Training the default schedule longer (more iterations) is better | counter 20,000 iters has the best quality of its tag (35.538) but B=114.38 s > 92.57 s budget, so the row is INVALID (time_points 0); hw 20,000 iters totals 30.278 vs 39.931 at 1,000 iters. | `records/artifacts-20261030/sweeps/{counter,hw}-default/summary.json` |
| 3 | Using the whole budget (late snapshots) is the goal | hw total falls 41.583 (4k, 29.7 s) -> 30.278 (20k, 99.0 s = 99.5% of budget, TIME 0.118); TIME is linear in B/budget, so filling the budget is the single most expensive thing you can do. | `records/artifacts-20261030/sweeps/hw-default/summary.json` |
| 4 | Adding VGG-LPIPS (lambda_lpips=0.2) to the B loss buys the 40-point LPIPS term | hw 4k: total 32.239 (Q 22.282) vs 41.583 (Q 26.000) for plain default and 45.173 for noreset on identical iterations; hw 8k at 108.14 s is already over budget. | `records/artifacts-20261030/experiments/round2/{hw,counter}-lpips/summary.json` |
| 5 | A coarse-to-fine resolution ramp (0.5 -> 0.75 -> 1.0) buys more optimization steps per second | hw 8k: 36.652 vs 41.574 (default) and 44.584 (noreset); counter 8k: 29.494 vs 42.396 noreset. It costs seconds without adding quality. | `records/artifacts-20261030/experiments/round2/{hw,counter}-resramp/summary.json` |
| 6 | Looser densification (densify_count_thresh=2) + a 900K Gaussian cap recovers detail the default prunes | hw 8k: +32,424 Gaussians (401,031 vs 368,607) but PSNR 22.729 vs 22.707, SSIM 0.7975 vs 0.7978, LPIPS 0.1573 vs 0.1574; total 43.914 vs 44.584. | `records/artifacts-20261030/experiments/round3/hw-dense-noreset/summary.json` |
| 7 | Snapshots taken at/near an opacity reset are usable quality points | reset points collapse the render: counter 3k PSNR 10.767/SSIM 0.3580 and hw 12k PSNR 5.116/SSIM 0.1090 both score Q=0.000; 6 of 40 archived rows are such degenerate snapshots. | `records/artifacts-20261030/sweeps/{counter,hw}-default/summary.json` |
| 8 | Turning off opacity reset is a uniform win across cases | counter 8k noreset 42.396 is 0.296 BELOW the 10k baseline (42.692); the reset change only pays on dl3dv (+5.292) and hw (+1.910). | `records/artifacts-20261030/experiments/round2/hw-noreset + round3/*-noreset vs records/o0-baseline-20261030.json` |

### 5.2 Wins claimed in the narrative that the archive cannot confirm

| claim | source | status |
|---|---|---|
| matched position-LR schedule (`position_lr_max_steps` = actual iterations) lifts hw 8k quality 32.51 → 35.39 | `docs/RESEARCH-20261030.md` §8 | no JSON: no run with a matched schedule exists under `records/` |
| no reset + λ_dssim 0.4 + matched LR at 4k → 76.31 / 50.05 / 48.37 (mean 58.24) | `docs/RESEARCH-20261030.md` §9 | no JSON: 4-GPU-machine runs unarchived |
| λ_dssim 0.6 + scaling_lr 0.01 + matched LR at 4k/5k → mean 59.02 / 60.03 | `docs/RESEARCH-20261030.md` §10 | no JSON: this is the hinted “mean ~60” recipe and it is not reproducible from the archived sweeps |

These are not counted as falsified — they are simply untested as far as this repository is concerned, and they are the highest-value experiments to re-run/archive because the archived knobs (reset, step count, densification, resolution ramp, LPIPS term) have all been exhausted.

## 6. What the numbers say about where the remaining gains are

- One second of B costs 20/(0.9·budget) ≈ 0.223–0.240 points of TIME, so extra optimization only pays above ≈ **0.24 quality points per second**.

- Measured marginal return 4k→8k with noreset: hw +4.502 Q for +22.8 s (0.198 Q/s → net -0.59, below the bar), counter +11.332 Q for +22.2 s (0.511 Q/s → net +6.01), dl3dv +6.547 Q for +19.8 s (0.330 Q/s → net +1.87). Step count is a per-case trade, not a global win.

- **hw/billboard is a quality wall, not a time wall**: its best valid quality is 34.271/80 (noreset 12k, total 41.375); even with TIME forced to 20 it would score ~54, while dl3dv already sits at 68.225 (quality 58.189) with TIME 10.036 — the 3-case mean is gated by hw and counter, not by dl3dv.

- counter converts seconds to quality worst: 0.614 Q per B-second at its best row, vs 1.237 for hw and 1.125 for dl3dv — any trainer speedup is worth more there than any knob combination in this archive.

- Per-case tuning of {reset, iterations} is worth +0.196 mean over the best uniform setting — there is no scene-specific hyperparameter gold mine left in these knobs, which matches the repo rule of not selecting hyperparameters per scene.

- Every archived run still uses the default 30,000-step position-LR schedule; the doc's matched-schedule lever is the one large effect missing from the archive (unverifiable here).

- Net: the remaining gains are (a) faster B per quality point (implementation/trainer efficiency, density control) and (b) hw geometry/pose quality on the A side — not further step-count or reset tuning.

## 7. Reproduce

Every row with its stored score (the raw evidence behind tables 2–4):

```bash
python3 -m research.collect records/artifacts-20261030 --kind B-sweep
```

Regenerate the grouped table, the per-case bests, the uniform means and the INVALID flags exactly as printed above:

```bash
python3 - <<'PY'
from pathlib import Path
from research.collect import rows_from
BUD={"dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f":94.29,
     "hw/billboard":99.49,"mipnerf360/counter":92.57}
rows=[]
for p in sorted(Path("records/artifacts-20261030").rglob("summary.json")):
    for r in rows_from(p): r["tag"]=p.parent.name; rows.append(r)
best={}
for r in rows:
    k=(r["tag"],r["case"])
    if k not in best or r["total"]>best[k]["total"]: best[k]=r
for k in sorted(best,key=lambda k:(k[1],-best[k]["total"])):
    r=best[k]; print(k[1][:8],k[0],r["iteration"],round(r["b_seconds"],2),round(r["total"],3),
                     "INVALID" if r["b_seconds"]>BUD[r["case"]] else "ok")
for c in BUD:
    v=[r for r in rows if r["case"]==c and r["b_seconds"]<=BUD[c]]
    r=max(v,key=lambda r:r["total"]); print("best",c[:8],r["tag"],r["iteration"],round(r["total"],3))
u=[r for r in rows if r["tag"] in ("hw-noreset","counter-noreset","dl3dv-noreset") and r["iteration"]==8000]
print("uniform 8k mean",round(sum(r["total"] for r in u)/3,3))
PY
```

## 8. Sources

| source | what it contributes |
|---|---|
| `records/artifacts-20261030/sweeps/{hw,counter}-default/summary.json` | 16 default-schedule rows (incl. the degenerate reset-point snapshots and the over-budget 20k) |
| `records/artifacts-20261030/sweeps/hw-probe/summary.json` | 2 early default-schedule rows (200/400 iters) |
| `records/artifacts-20261030/experiments/round2/{hw,counter}-{lpips,resramp,noreset}/summary.json` | lpips, resolution-ramp and hw/counter noreset rows |
| `records/artifacts-20261030/experiments/round3/{dl3dv,counter,hw-dense}-noreset/summary.json` | dl3dv noreset, counter noreset, dense-900k rows |
| `records/artifacts-20261030/experiments/round1/*-score/score_summary.json`, `probe/init-*/score_summary.json` | initialization-only (B=0) quality for num_refine 2/4/6; `time_points` is null, so no totals |
| `records/o0-baseline-20261030.json` | official baseline (3 cases, 10k iters) and the budgets used throughout |
| `docs/RESEARCH-20261030.md` | narrative; §8–§10 claims are quoted only where marked unverified |

`records/best-recipe.json` is the machine-readable version of §3–§5 (same numbers, plus the budget map, baseline block, per-row source paths and the unverified-claim list).
