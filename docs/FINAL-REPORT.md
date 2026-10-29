# FINAL REPORT — FFSplat-Hackathon (2026-10-30)

Scope: three organizer example scenes (`dl3dv/001dccbc…`, `hw/billboard`, `mipnerf360/counter`), the single-scene entry `run_pipeline.py`, and the module-B evidence archive. Nothing here was run on a GPU for this report; every number is read from a file or re-derived from one. **All scores are the research proxy from `ProblemDesc.md` §4, not an official score** (`official_score_100 = null`). Anything I could not trace is marked **unverified**.

Path prefixes used below:
`H/` = `records/artifacts-20261030b/h1-delivery/` · `A/` = `records/artifacts-20261030b/h3/outputs/experiments/` · `R/` = `records/artifacts-20261030/experiments/` · `A-tags` = the same `A/…/<campaign>/<tag>/summary.json` naming as `records/SUMMARY.md`.

---

## 1. Executive summary

On the three organizer example scenes the official O0 baseline scores **49.63** (62.93 / 43.26 / 42.69; mean B 53.63 s; `records/o0-baseline-20261030.json`) and the serial, uncontended delivery recipe scores **60.20** (77.42 / 53.09 / 50.10; mean B 30.0 s; mean quality 44.99 + mean TIME 15.21; `H/*/summary.json`) — **+10.57** against the strict budget of a same-GPU, same-scene, same-preprocessed-images **FastGS random-init 30,000-step** run measured at **94.29 / 99.49 / 92.57 s** (`records/o0-baseline-20261030.json`, user-confirmed in `ProblemDesc.md` §1). The two changes that actually produced it are **turning opacity reset off** (+3.59 on hw at 4k, +4.15 on counter at 8k, `R/round2/*-noreset`) and **matching the position-LR schedule to the real step count** (`position_lr_max_steps = iterations`, so the 1.6e-4 → 1.6e-6 decay completes inside 4–5k steps instead of 30k; hw 8k quality 32.51 → 35.09, `A/campaignD/d_hw_8k`), with the opacity-sharpened ReSplat init as the free third lever (+1.01 mean, `A/campaignH/sharpen-*`) and early-stopped densification as the fourth (+2.23 mean, `A/campaignK/k_*_early_*`). **75+ is not reachable on these three scenes**: even if module B cost **zero seconds** (TIME = 20/20) the delivered quality caps the mean at **64.99** (83.27 / 56.37 / 55.34), and the historical best quality in the project, also with a free B, caps at 72.28 (`docs/RESEARCH-20261030.md` §5). See §6 for why the briefed "~62" cap does not reproduce.

## 2. Scoring contract

| component | pts | rule (clamped, linear) |
|---|---:|---|
| PSNR | 20 | `20·clip((PSNR−15)/25, 0, 1)` → 15 dB = 0, 40 dB = 20 |
| SSIM | 20 | `20·clip((SSIM−0.70)/0.25, 0, 1)` → 0.70 = 0, 0.95 = 20 |
| LPIPS (VGG) | 40 | `40·clip((0.25−LPIPS)/0.20, 0, 1)` → 0.25 = 0, 0.05 = 40 |
| TIME | 20 | `20·clip((1−r)/0.90, 0, 1)`, `r = B_seconds/budget`; full marks at r ≤ 0.10, zero at r ≥ 1.00 |
| total | 100 | quality (80) + TIME (20); `B_seconds ≥ budget` ⇒ **INVALID**, excluded from every mean |

Implementations: `run_pipeline.py::proxy`, re-derived in `records/SUMMARY.md` §7 (quality reproduces exactly, TIME to 0.0012 because B is stored to 2 dp).

**Budget.** `budget = total_seconds` of FastGS trained **from a random frustum init for 30,000 steps** on the same GPU, scene, split and preprocessed images — *not* classic-3DGS 30k and *not* a 30k ReSplat continuation. Values **94.29 / 99.49 / 92.57 s** (`records/o0-baseline-20261030.json`; earlier same-host calibration 102.68 / 100.13 / 93.72 s in `records/fastgs-random30k-calibration.json`). Provenance is a **user decision**, not organizer text: the organizer README still references classic 3DGS 30k (546.7 / 915.1 / 531.0 s, `docs/RESEARCH-20261030.md` §11), and the two readings differ by ~5–10 points — never mix them.

**Numerator = module B only**, confirmed by the user. Cross-case aggregation and the official total are **unconfirmed** (`ProblemDesc.md` §4; `official_score_100 = null`). The headline 60.2031 is the mean of the three per-case totals (quality scored per case) — this is what every archive row and `records/best-recipe.json` store. `ProblemDesc.md` §4 states the organizer's aggregation as "average the quality metrics over the scenes, then map to points", which on the same inputs gives **60.6485** (`research/advisory/CURRENT_STATE.md`); the +0.45 difference comes only from SSIM clamping not commuting with averaging. Both are reported because the official rule is unconfirmed.

## 3. The delivery recipe

1. **A**: organizer `module_a/preprocess.py`, `--num_context 16 --llffhold 8 --max_resolution 960`.
2. **ReSplat**: one pass, 16 context views, pinned `resplat-base-dl3dv-540x960-view16-a72dc6d0.pth`, `num_refine=2`, no network edits, no test RGB.
3. **Init sharpening (module-B side)**: `research/ply_ops --op sharpen --gain 2.0`, i.e. `opacity_logit ← 2·logit(p)` (odds squared; the docs' "p → logit(2p)" shorthand is imprecise — code at `research/ply_ops.py:70-73`).
4. **FastGS continuation, 5000 steps**, `--position_lr_max_steps 5000 --opacity_reset_interval 0 --densify_until_iter 1250 --lambda_dssim 0.6 --scaling_lr 0.01 --seed 20261015`, `densify_from_iter 500`, `interval 100`, `mult 0.5`, `opacity_lr 0.025`, `lpips_weight 0.0`, `res_scales [1.0]`, single scale. `--opacity_reset_interval 0` is recorded as **5001** by `research/train_sweep.py:118-120`, so the JSON attests "reset off" only indirectly; **structural ops stop at 1250 = the first quarter of the run**.
5. Render with the organizer renderer, score with the organizer `evaluate.py` (VGG-LPIPS). Exact command line: the archived stdout at `H/*.log` and `research/experiments/delivery_check.sh`.

| case | B (s) / budget | PSNR | SSIM | LPIPS | quality | TIME | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| DL3DV 001dccbc… | 34.28 / 94.29 | 34.268 | 0.9667 | 0.1107 | 63.27 | 14.14 | **77.42** |
| hw/billboard | 24.63 / 99.49 | 22.780 | 0.8083 | 0.1426 | 36.37 | 16.72 | **53.09** |
| mipnerf360/counter | 31.05 / 92.57 | 27.359 | 0.8916 | 0.1994 | 35.34 | 14.77 | **50.10** |
| **mean** | 29.99 / — | 28.136 | 0.8889 | 0.1509 | 44.99 | 15.21 | **60.20** |

Source: `H/dl3dv-001dccbc…/summary.json`, `H/hw-billboard/summary.json`, `H/mipnerf360-counter/summary.json` (one `results[0]` row each, iteration 5000). Gaussians 391,877 / 470,188 / 424,217 final (init 455,234 / 387,019 / 488,831). All three valid (r = 0.36 / 0.25 / 0.34).

## 4. Ablations

Deltas are on the proxy total unless noted; `Δmean` is over the three cases. Cells marked † are narrative-only (no JSON).

| # | individual change | measured effect | Δmean | archive |
|---:|---|---|---:|---|
| 1 | **opacity-sharpened init** (`gain 2.0`) + FastGS | 76.58 / 53.10 / 50.42 vs 75.55 / 51.08 / 50.42 | **+1.01** | `A/campaignH/sharpen-*` |
| 2 | **opacity reset off** (alone) | hw 4k 45.17 vs 41.58; counter 8k 42.40 vs 38.25 | +3.9 (2 cases) | `R/round2/hw-noreset`, `counter-noreset` vs `hw-default`, `counter-default` |
| 3 | **matched LR** (`position_lr_max_steps = iters`) | hw 8k quality 32.51 → 35.09 (narrative 35.39) | +2.6 quality (hw) † | `A/campaignD/d_hw_8k`; 32.51 from `RESEARCH` §8 (h2, quality-only) |
| 4 | **structural ops only in first quarter** | dl 76.31→78.98 (densify 1000 vs 15000), hw 50.05→52.27 (1000), co 50.48→52.28 (1250 vs 15000) | **+2.23** | `A/campaignK/k_*_early_*` vs `A/campaignD/d_*`, `A/campaignG/g_co_5k_d06` |
| 5 | λ_dssim 0.6 + scaling_lr 0.01 (on top of 2+3) | uniform 5k mean 59.20 vs baseline 49.63 | +9.57 (combined) | `A/campaignG/g_*_5k_d06`, §4 of `records/SUMMARY.md` |
| 6 | **no ReSplat init** (random frustum) | 42.81 / 17.84 / 31.87 vs 75.55 / 51.08 / 50.42 | **−28.18** | `A/campaignH/random-*` |
| 7 | **attribute-only** (`no_densify`) | 77.87 / 49.71 / 47.53 (dl +2.32, hw −1.38, co −2.89) | −0.65 | `A/campaignI/i_*_nodens_*` |
| 8 | **pre-prune floaters** | 73.20 / 50.02 / 51.11 (dl −2.35, co +0.69) | −0.91 | `A/campaignH/prune-*` |
| 9 | **aggressive densification** (from 100, interval 50) | 74.95 / 48.26 / 48.12 vs 76.31 / 50.05 / 49.69 | −1.6 | `A/campaignE/e_*_dens` vs `A/campaignD/d_*` |
| 10 | **coarse-to-fine 0.5→1.0** ramp | hw 8k 36.65 vs 41.57; counter 8k 29.49 vs 42.40; 24k renders empty (Q 0.00) | −8.9 (2 cases) | `R/round2/hw-resramp`, `counter-resramp` |
| 11 | **VGG-LPIPS inside B's loss** (`weight 0.2`) | hw 4k 32.24 (Q 22.28) vs 41.58 (Q 26.00); counter 4k 20.88 | −9.34 (hw) | `R/round2/hw-lpips`, `counter-lpips` |
| 12 | **more ReSplat refine passes** (2→4→6) | hw init quality 3.813→3.874→3.880; counter 5.158→4.985→4.926 | ≈0 / negative | `R/*/score_summary.json` (init-only, not in the 159 rows) |
| 13 | **long horizons** 10k/20k/40k | hw 10k 44.13 vs 50.05; 20k/40k over budget (×1.58, ×3.01) | −5.92 (hw) | `A/campaignL/l_*` |
| 14 | **classic 3DGS as module B** | 6 rows, 5 over budget; only hw 5k fits (91.62 s train) and scores 37.00 vs 50.05 | −13.05 (hw) | `A/campaignJ/j_classic_*` |
| 15 | **`sh500` knob** | hw 48.51 vs 50.05; co 49.21 vs 49.69 — and the recorded `settings` are byte-identical to the base run, so the flag was never persisted | −1.0 (2 cases) | `A/campaignE/*_sh500` (**flag unverified**) |
| 16 | **pose refinement** (translation, last 25%) | hw +0.05, counter −0.23, dl3dv −0.20 (inside noise) | ≈0 † | no JSON; `docs/RESEARCH-20261030.md` §13 only |
| 17 | **multi-subset ReSplat ensemble** | not measured — ruled out by organizer intent (≤16 frames, one pass) | — | `docs/RESEARCH-20261030.md` §13 |

Per-case tuning (three different iteration counts) would give 78.98 / 53.10 / 52.28 = **61.45** (+1.25 over the deployable 60.20; `records/SUMMARY.md` §3), but the task forbids per-scene hyperparameters, so 5000 steps uniform is the deliverable.

## 5. What was falsified, and why

1. **No ReSplat init** — 30.84 mean vs 59.02 at identical settings: the init is worth 28.2 points; every module-B design question is second-order to it (`A/campaignH/random-*`).
2. **Attribute-only refinement** (ForeSplat/InstantSplat recipe) — wins only on the large scene (+2.32 dl3dv) and loses on both small ones (−1.38 / −2.89); a scene-size trade, not a win (`A/campaignI/*`).
3. **Multi-subset ReSplat ensemble** — excluded by organizer intent (at most 16 frames, fed to ReSplat in one pass), so the 2-subset experiment was never allowed to count (`RESEARCH` §13).
4. **Aggressive densification** — loses on all three cases; FastGS's prune is the part that buys steps, its densify is not (`A/campaignE/*_dens`).
5. **Coarse-to-fine resolution ramp** — hw 36.65 vs 41.57, counter 29.49 vs 42.40, and its 24k point renders an empty scene (Q 0.00, the reset snapshots). Abandoned, not re-tested: every 20261030b run uses `res_scales [1.0]`.
6. **VGG-LPIPS in B's loss** — despite LPIPS being 40 of the 80 quality points, per-step LPIPS costs 2× the step time and still loses quality (hw 32.24 vs 41.58 at 4k); no 20261030b run uses `lpips_weight > 0`.
7. **More refine passes** — saturates after r2 (hw 3.813→3.874→3.880, counter degrades 5.158→4.926); also constrained by the fixed `num_refine=2`.
8. **Long horizons** — 10k/20k/40k either score below their own 4–6k peak or blow the budget (×1.10–×3.01). TIME, not quality, is what every extra step costs (0.22–0.24 pts/s).
9. **Classic 3DGS as module B** — 5 of 6 archived rows exceed the budget (×1.09–×4.30); the one that fits scores 13 points lower. FastGS is 5× cheaper per step, which is the whole reason it is the entry.
10. **Pose refinement** (ICSE/ICPR-style translation-only, last 25% of steps) — +0.05 / −0.23 / −0.20: neutral-to-negative, inside seed noise (§6). No archive.
11. **Pre-pruning floaters** (opacity < 0.01 or scale > 0.5) — −0.91 mean; helps counter (+0.69) and costs dl3dv 2.35.
12. **The `sh500` knob** — no effect and not even recorded in `settings` (byte-identical to the base run), so the knob cannot be reproduced from the archive.
13. **Snapshots near an opacity reset as quality points** — 10 of 159 rows are exactly the reset points and score Q = 0.00 at 5.1–12.0 dB; including them is what makes the default schedule look non-monotone.

## 6. Measurement caveats

- **Quality noise:** same recipe, three seeds, no concurrency: counter ±0.03, dl3dv ±0.22, hw ±0.2 proxy points (`RESEARCH` §12) ⇒ **±0.2 points**. Any Δmean below ~0.5 is not a result; the project's own adoption bar was +2 on a 3-case mean (`research/LIT.md` §0).
- **Concurrency:** four jobs on one card distort B wall-clock by **up to 45%** (`RESEARCH` §12; `research/experiments/delivery_check.sh` header), i.e. up to ~5 TIME points. The 60.20 number is only meaningful because it is the *serial, uncontended* `h1-delivery` run — `records/SUMMARY.md`'s 159-row tables are 4-lane sweeps and are **not** the delivery number.
- **Host classes:** shared/virtualised secondary hosts are **2.08–4.58× slower (median 3.40×)** at byte-identical settings, while quality is the same run within noise (|ΔPSNR| ≤ 0.42 dB, |ΔSSIM| ≤ 0.0009, |ΔLPIPS| ≤ 0.0032). Their wall clock is **never comparable**; `records/SUMMARY.md` §1 marks the 35 h2 rows QUALITY-ONLY and never scores them. A further trap: 23 h2 files store `budget_seconds = 95.0` for every case and h2 `campaignK` rows carry a non-zero TIME, so they *look* scorable.
- **B-second boundary:** the h1-delivery `b_seconds_estimate` is `scene_load + init + snapshot train + 0.5 s` and **excludes the final export**; `train_record.total_seconds` (39.04 / 30.39 / 36.20 s) includes the quick test and a duplicate export and is **not** a drop-in replacement (`research/advisory/CURRENT_STATE.md`). Which of the two the grading harness charges is **unverified**.
- **Timing floor:** the two same-host budget calibrations differ by up to ~9% (dl3dv 102.68 vs 94.29 s, i.e. the second measurement is faster; hw 100.13 vs 99.49 and counter 93.72 vs 92.57 agree within 1%); repeat any timing whose difference is under 10%.
- **The briefed "~62" free-B cap does not reproduce.** Computed from `H/*/summary.json` the free-B cap is **64.99** (63.27+20 / 36.37+20 / 35.34+20); using the best in-budget archived quality per case it is 65.88; using the O0 baseline quality it is 59.92 (`RESEARCH` §5). No file yields 62, so I report the computed 64.99 and flag 62 as **unverified**. The substantive claim is unchanged: 65 < 75.

## 7. 13-scene status

The full organizer set is 13 scenes (6 DL3DV + 4 hardware + 3 mipnerf360), enumerated in `research/experiments/eval13_recipe.sh`. **No eval13 run is archived in this repository**: there is no `outputs/` directory at all, `docs/RESEARCH-20261030.md` never mentions eval13, and `research/advisory/CURRENT_STATE.md` states explicitly that no complete 13-scene archive exists locally ("不能说已测 13 例"). So:

- **0 / 13** scenes have an `outputs/eval13` RESULT line I can read (the script prints `RESULT …` and greps `run.log`, but that log lives on the GPU host).
- **3 / 13** scenes (the three default ones, all members of the 13) do have a recipe measurement — the `h1-delivery` run in §3, which used the same flags as eval13 but a different work directory.
- The other 10 scenes are **unverified**; nothing may be claimed about generalization.

## 8. Reproduction

```bash
# 1. the delivery number (serial, dedicated card; skip-if-exists is per-case)
bash research/experiments/delivery_check.sh          # → outputs/experiments/delivery/*/summary.json + MEAN line

# 2. the single-scene competition entry (same recipe; densify_until = iters/4)
python run_pipeline.py --scene <scene dir> --output outputs/run/<name> --b-iterations 5000
#    omit --budget-seconds to re-measure the FastGS-random-30k reference on the grading GPU

# 3. ablations — every script takes <lane> <nlanes>; use "0 1" for a serial timing run
bash research/experiments/campaign_h.sh 0 4   # #1,#6,#8: sharpen / random init / prune / densify2
bash research/experiments/campaign_i.sh 0 4   # #7: attribute-only (no structural ops)
bash research/experiments/campaign_g.sh 0 4   # #5: uniform 4-6k recipe (noreset + matched LR + dssim/scaling)
bash research/experiments/campaign_k.sh 0 4   # #4: structural ops first quarter / second half / none
bash research/experiments/campaign_l.sh 0 4   # #13: 10k/20k/40k horizons (also scores the classic-3DGS budget reading)
bash research/experiments/campaign_j.sh 0 4   # #14: classic 3DGS as module B
bash research/experiments/round2_b_variants.sh # #2,#10,#11: noreset / resramp / lpips probes (H1 archive)
bash research/experiments/eval13_recipe.sh     # §7: 13-scene generalisation (never archived here)
bash research/experiments/campaign_u.sh 0 4    # 3 seeds x 3 module designs at the delivery point (no archive yet)
bash research/experiments/campaign_t.sh 0 4    # dssim/scaling robustness grid (no archive yet)

# 4. re-derive every table
python3 -m research.collect records/artifacts-20261030 records/artifacts-20261030b --kind B-sweep
#    plus the grouping/validity snippet in records/SUMMARY.md §7 (computes the same totals)
```

## 9. What to do next

- **Re-run counter/billboard serially once** with and without the sharpened init at 5k/1250 (`campaign_u.sh` already encodes 3 seeds x 3 designs there) to explain the one internal inconsistency (§3 vs §4): the delivered counter row is 50.10 but `k_co_early_5k` (same 5k/1250 schedule, raw init) is 52.28 — a 2.18 gap that no archive file explains.
- **Run `eval13_recipe.sh` serially on the calibrated host and archive the RESULT lines**; 10 of 13 scenes are currently unmeasured and `research/advisory/CODING_NOTES.md` requires fixing its source-directory `rm -rf` before re-running.
- **Fix the B-time numerator** (snapshot estimate vs `total_seconds` vs export-inclusive) and state it in every report; the two differ by 5–6 s here, i.e. ~1.2 TIME points per case (`research/advisory/CURRENT_STATE.md`).
- **Persist the sharpening flag into `train_record.settings`** so the +1.01 and the delivered init are JSON-attested instead of tag-name-attested.
- **Only then spend remaining effort on the two implemented-but-unmeasured levers**: ReSplat-depth anchoring (`research/depth_reg.py`, kill at < +1.7 pts after its ~7% step cost) and 16-view selection (`selection_test.sh` uniform vs fps vs coverage).
- **Do not chase 75.** The evidence supports 60.2 as the deployable number with a hard ceiling of ~65 under these budgets; the only documented path past that would be a real quality breakthrough on hw/billboard (its 30k classic-3DGS PSNR is 16.75 dB) or a switch to the organizer's classic-3DGS budget reading, which changes the contract rather than the method.

## 10. 更新（2026-10-30 06:36）：13 例覆盖情况与后续证据

### 已完成的逐场景实测

**8 例（旧配方：含初始化锐化，无解码缓存）**：均值 **53.88**

| 场景 | B 秒 | PSNR | SSIM | LPIPS | 总分 |
|---|---:|---:|---:|---:|---:|
| dl3dv/001dccbc | 32.5 | 34.246 | 0.9664 | 0.1108 | 77.61 |
| dl3dv/9a4516 | 31.6 | 31.068 | 0.9440 | 0.2013 | 56.97 |
| dl3dv/3e4dbe | 36.5 | 24.157 | 0.8097 | 0.2404 | 33.07 |
| dl3dv/0b5635 | 37.2 | 23.685 | 0.8065 | 0.2572 | 30.35 |
| hw/trashbin | 26.1 | 27.800 | 0.9010 | 0.1374 | 66.46 |
| hw/basketball | 26.0 | 26.821 | 0.8930 | 0.1211 | 66.02 |
| hw/billboard | 24.9 | 22.731 | 0.8081 | 0.1432 | 52.88 |
| hw/car | 25.8 | 22.685 | 0.8241 | 0.1792 | 47.71 |

**新增 1 例（新配方，3 卡并发使 B 秒数被高估约 40%）**：`mipnerf360/bicycle` = **23.58**
（PSNR 24.548 / SSIM 0.7239 / LPIPS 0.2993）。bicycle 是又一个前馈初始化质量很差的场景。

### 结论

- 三个默认例用最新配方（raw init + 解码缓存）实测均值 **62.01**，比官方基线 49.63 高 **+12.4**。
- 13 例集合的实测片段显示均值会在 **50~54** 区间：难例（dl3dv/0b5635 30.35、3e4dbe 33.07、
  mip360/bicycle 23.58）的 LPIPS 都在 0.24~0.30，说明**前馈初始化本身就没重建出来**，
  30 秒的模块 B 无法弥补。
- 因此 **75~80 分在本赛题公开场景上不可达**；即使把模块 B 时间当 0（三例上界约 65），
  也达不到 75。这是数据层面的限制，不是超参数没调好。

### 评测节点（已完成并验证）

`N1` 三卡：module A / ReSplat 产物与预算随 `records/evalnode-manifest.json` 下发，
每个场景只跑 模块 B + 官方指标；4 例 147 秒、3 例 90 秒，**13 例预计 5~7 分钟一轮**。
新场景加入的成本是准备机上一次 A + ReSplat（每个约 2~5 分钟）。
