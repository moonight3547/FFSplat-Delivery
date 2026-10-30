# AUDIT — `docs/STATE-INHERIT-20261030.md` vs on-disk evidence

Auditor: `h1-prep` (independent verifier; did not write the audited doc).
Date: 2026-10-30 (Asia/Shanghai). Mode: **read-only**; only this file under `records/` was written.
Audited doc sha256: `6bc3b5870f7ad0406ff6174b799b31e8e793703bd823bf8409eadb722e0e0cd0`
(135 lines, 10,296 B, mtime 22:55). H1 GPU was left untouched (shell/python only).

## VERDICT: **PASS-with-corrections**

No numeric error was found in the two headline tables. Every §3.1 cell and the
mean `41.371 → 49.599 (+8.228)` recompute exactly; all 7 present §3.3 cells recompute
exactly and the claimed missing cells are genuinely missing. The corrections are
**1 citation range, 1 provenance attribution, 1 command-completeness, 1 now-false code
statement, 2 stale-count/snapshot items** — none changes a conclusion or a number that
was used for a decision. The manifest sha256 matches.

Evidence hashes used (files as audited):

```
6bc3b587…  docs/STATE-INHERIT-20261030.md        (audited)
8f0343b0…  docs/RESEARCH-20261030.md              (55 sections, mtime 22:16)
b796524c…  docs/FINAL-REPORT.md
e4060a8b…  research/experiments/scene_eval_r15.sh (identical sha on Mac and H1)
c2aac742…  records/evalnode-manifest-13.json      (matches the requested sha)
113f1890…  research/advisory/HANDOFF_MULTISHOT_REALLOC_20261030.md
```

---

## 1) §3.1 — 13-case per-scene table + means: **PASS, 0 cells differ**

Source: H1 `/workspace/organizer-projects/FFSplat-Hackathon/outputs/scene13r15/results.jsonl`
— exactly 13 JSON lines, arms `baseline` / `delivery`, all recomputed from raw `total_points`:

| case | base (calc) | doc | del (calc) | doc | Δ (calc) | doc | q base→del | B base→del |
|---|---:|---:|---:|---:|---:|---:|---|---|
| dl3dv/001dccbc | 66.640 | 66.64 | 78.990 | 78.99 | +12.350 | +12.35 | 56.44→64.59 | 51.01→33.20 |
| dl3dv/0b5635 | 25.610 | 25.61 | 31.130 | 31.13 | +5.530 | +5.53 | 15.08→17.20 | 49.63→35.20 |
| dl3dv/0f622ea | 22.150 | 22.15 | 31.860 | 31.86 | +9.710 | +9.71 | 12.83→18.24 | 54.75→36.50 |
| dl3dv/2dad2d | 36.490 | 36.49 | 49.600 | 49.60 | +13.110 | +13.11 | 25.59→35.33 | 48.04→33.75 |
| dl3dv/3e4dbe | 26.030 | 26.03 | 35.290 | 35.29 | +9.260 | +9.26 | 15.71→21.13 | 50.48→34.24 |
| dl3dv/9a4516 | 52.130 | 52.13 | 59.560 | 59.56 | +7.430 | +7.43 | 40.59→44.38 | 45.35→29.89 |
| hw/billboard | 45.560 | 45.56 | 53.380 | 53.38 | +7.820 | +7.82 | 32.47→36.50 | 40.87→23.90 |
| hw/basketball | 58.960 | 58.96 | 67.550 | 67.55 | +8.590 | +8.59 | 45.46→50.48 | 37.29→22.05 |
| hw/car | 58.240 | 58.24 | 67.520 | 67.52 | +9.280 | +9.28 | 44.92→50.48 | 38.03→22.11 |
| hw/trashbin | 21.910 | 21.91 | 25.410 | 25.41 | +3.500 | +3.50 | 8.11→9.01 | 36.00→24.89 |
| mipnerf360/bicycle | 20.820 | 20.82 | 25.320 | 25.32 | +4.500 | +4.50 | 10.79→9.92 | 52.10→29.18 |
| mipnerf360/bonsai | 56.530 | 56.53 | 63.930 | 63.93 | +7.400 | +7.40 | 44.60→48.31 | 44.00→28.19 |
| mipnerf360/counter | 46.740 | 46.74 | 55.230 | 55.23 | +8.490 | +8.49 | 35.43→40.28 | 45.46→30.31 |

**Means (raw, unrounded):** base `41.370893` → 41.371 ✓; delivery `49.598723` → 49.599 ✓;
delta `8.227830` → +8.228 ✓. The q/B columns also match after rounding (B convention note
in §Notes-1: two cells rely on round-half-up: 45.35→45.4 and 22.05→22.1).
Claim "mip/bicycle is the only delivery quality regression" ✓ (10.79→9.92, all others up).

## 2) §3.3 — ocurve hard-scene table: **PASS; missing cells confirmed**

Source: H1 `outputs/ocurve/hard-*/s*/summary.json` (7 files). Recomputed
(`b_seconds_estimate` / `quality_points` / `total_points`):

| case | S=500 | S=2000 | S=5000 |
|---|---|---|---|
| 0b5635 | q 5.19 / B 12.48 / 24.47 ✓doc | q 10.67 / B 18.49 / 28.54 ✓doc | q 17.06 / B 36.05 / 30.79 ✓doc |
| 0f622ea | q 3.78 / B 10.33 / 23.57 ✓doc | q 6.02 / B 17.58 / 24.10 ✓doc | q 17.82 / B 36.57 / 31.42 ✓doc |
| 3e4dbe | q 5.41 / B 9.85 / 25.31 ✓doc | **absent** ✓doc says 缺 | **absent** ✓doc says 缺 |

Doc's "难例上步数越多总分越高" is consistent with both complete rows (24.47→28.54→30.79
and 23.57→24.10→31.42). No correction.

## 3) §3.2 — delivery command vs real executed commands: **PASS-with-corrections (C3)**

`outputs/scene13r15/run.log` **does** contain the executed shell lines (`+ … conda run …
research.train_sweep …`): 13× base-arm and 13× delivery-arm, and all 13 delivery-arm lines
are byte-identical in their flag set (`sort -u` → 1 unique line). Verbatim delivery arm
(run.log line 57, paths elided):

```
research.train_sweep --source <work> --model_path <out>/model --eval -r 1 --sh_degree 3 --quiet
  --iterations 5000 --snapshots 5000 --seed 20261015 --position_lr_max_steps 5000
  --densify_until_iter 1250 --opacity_reset_interval 0 --lambda_dssim 0.6 --scaling_lr 0.01
  --image_cache <cache> --realloc_frac 0.15 --realloc_at 1250
  --opt_step_schedule "" --inject_holes 0 --inject_at 800
```

Base arm (run.log line 29): `--iterations 10000 --snapshots 10000 --seed 20261015
--position_lr_max_steps 10000 --densify_until_iter 15000 --opacity_reset_interval 3000
--image_cache … --realloc_frac 0.0 --realloc_at 1250 …` → doc's baseline O0 line ✓ exact.

Every flag listed in doc §3.2 is real and matches; **nothing is wrong**, but the doc's
"实测命令" is an excerpt and cannot be run as-is (see C3). Corroboration of the
non-echoed stages: `[A] done: … 16 context, res=[512, 960]` (max_resolution 960),
`Running refinement (2 iterations)` (num_refine 2), `[resplat] SH export basis patched
(first-order z sign)` (fix-sh-basis), `"status": "cached"` (image cache in use).
Also: recorded `time_points` reproduce from the §5 budgets with the TIME formula
(e.g. 001dccbc B=33.2, budget 94.29 → t=14.398 = JSONL 14.3977; counter 30.31/92.57 →
14.946 ≈ 14.9456; billboard 23.9/99.49 → 16.884 ≈ 16.8833), so the runs did use those budgets ✓.

## 4) §1/§2 node facts: **snapshot was true; several values now drifted (C5, C6)**

Verified against H1/N1 at audit time:

- H1: 1×4090 48 GB ✓ (`memory.total` 49140 MiB); root overlay 200 GB ✓; SSH `ffh1`
  → `cpod-1wqzm4msan2q-s3.podtcp.compshare.cn:22304` ✓ (pod short hostname
  `cpod-1wqzm4msan2q`); intermittently `Connection closed` ✓ (observed during audit).
  HEAD `cd355cb` ✓; conda envs exactly the 4 listed ✓; 13 raw datasets ✓ (DL3DV 6 / hw 4 /
  mipnerf360 3); checkpoint present ✓.
- N1: 3×4090 24 GB ✓ (`memory.total` 24564 MiB each); root 30 GB / 99% ✓ (now 309 MiB free,
  doc said 312 MiB); `/root/gpufree-data` 246 GB ✓; HEAD `c3c1ee5` ✓; conda envs only `fastgs` ✓;
  no raw datasets ✓ (only prepared work + PLY, plus today's `inputs13/` + `backup/`).
- Mac repo: HEAD `ad90ed4` == `origin/main` ✓ (pushed).

Nothing in §1/§2 was false at the stated 22:47–23:00 snapshot; the drifts are in C5/C6.

## 5) §3.4 — "falsified / confirmed wins" spot-checks: **PASS (5/5 verified, 1 citation to add)**

Spot-checked against `docs/RESEARCH-20261030.md` (line anchors given):

| §3.4 claim | Research section | Match |
|---|---|---|
| 去每步 `loss.item()` 同步（B 28.22→24.55 s） | §34 (lines 592–605): legacy 28.22 s / 177.2 it/s → new 24.55 s / 203.7 it/s, −13% | ✓ exact |
| uint8 解码缓存（+0.9 TIME，质量不变） | §16 (266–283): 61.13 → 62.01 mean, quality unchanged, gain all TIME | ✓ exact |
| MCMC/Taming-3DGS 难例同预算 −8~10 分 | §27 (466–482): −8.66 / −8.05 / −10.00 on 0b5635/0f622ea/3e4dbe | ✓ exact |
| 纯外观/冻结几何 −13~−30 | §33 (572–592): −12.82 / −14.84 / −30.14 | ✓ (−13 after rounding) |
| 渲染端抗锯齿（膨胀高斯单调变差） | §52 (908–928): m=1.00→1.20 LPIPS 0.2817→0.3070, "单调变差" | ✓ exact |
| 孔洞注入（质量 +0.3 但 B 翻倍） | §29 (501–518): quality +0.0~0.5, B doubled, total −6 | ✓ ("+0.3" in range) |
| 多 shot = 高方差 6.55 vs 10.86 | **not a RESEARCH §** — verbatim in `research/advisory/HANDOFF_MULTISHOT_REALLOC_20261030.md` lines 23–36: same config, same GPU, sequential, same init, 4.3 q / 15.7 s apart | ✓ but see C7 |
| 10K / 4.5K 步 | §22 (lines 375/377): 10K 四例总分全降; 4.5K 51.00 vs 5K 51.12 | ✓ |
| 粗到细分辨率 / LPIPS 入损 / num_refine>2 / 激进致密化 / 预剪枝 | §6 table "运行中" (line 111) is not a conclusion; the closed-out list lives in `docs/FINAL-REPORT.md` §12 (lines 188–194, list at 192–194); 预剪 numbers 58.11 vs 59.02 are RESEARCH §10 (line 182); RESEARCH §4 (lines 62–71) supports num_refine>2 | ✓ (basis for some items is FINAL-REPORT, not RESEARCH) |

## 6) Manifest sha256: **PASS**

`sha256sum records/evalnode-manifest-13.json` (Mac) =
`c2aac742d859f1e7c82b592c04b2788a760eeefdc37a132ad9ebf5435d9c2667` — exactly the requested
value. N1 copy has the same sha (verified in task-1). 13 entries, budgets match doc §5.

---

## Numbered corrections (exact)

**C1 — §6 evidence index, line 131.**
Says: `| 旧会话研究叙事（§1–§37） | docs/RESEARCH-20261030.md |`
Should say: `§1–§55`. `grep -c "^## " docs/RESEARCH-20261030.md` = **55** sections
(§1…§55); the file is older (mtime 22:16) than the doc (22:55), so at writing time the
narrative already ran to §55.

**C2 — §3.1 note, lines 72–73.**
Says: `旧会话压缩摘要里出现过 42.38 → 51.35，与这份落盘 JSONL 不一致；…以本表（落盘证据）为准，51.35 不再引用。`
Should say: the 42.38/51.35 pair is **on disk in `docs/RESEARCH-20261030.md` §24** (table at
line 422: `13 例均值 42.38 / 51.35 / +8.97`), not merely a compressed summary. And the two
tables are not like-for-like: §24's O0 baseline used `position_lr_max_steps 30000` (30K
schedule, §24 lines 405–407), whereas `scene13r15/results.jsonl` used the matched
`--position-lr-max-steps 10000`; the resulting 1.01 baseline gap is already documented at
RESEARCH line 728 (`42.38 vs 41.37`). Keep "以本表为准", but cite §24 + the schedule
difference instead of "压缩摘要".

**C3 — §3.2, lines 77–78 ("实测命令").**
Says: module-B del line = `--iterations 5000 --position-lr-max-steps 5000
--opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 --densify-until 1250
--realloc-frac 0.15 --realloc-at 1250 --image-cache <cache>`, baseline line = 4 flags,
"A 用 --max_resolution 960", "ReSplat 用 --num_refine 2 --fix-sh-basis".
Should say: this is a flag excerpt, not the executed command; for reproduction add
`--budget <budget>` (run_sweep-level; default `None`, so without it the TIME term of
§3.1 cannot be reproduced), `--seed 20261015` (run_sweep default, but state it),
`--snapshots ""` (run_sweep injects `iterations` itself), and note `--densify-from`
was not passed (FastGS default **500**, giving the measured 500→1250 window);
A also passes `--num_context 16 --llffhold 8`; ReSplat also passes
`--checkpoint <ckpt> --device cuda:0`. Recommend pasting the verbatim del command from
run.log line 57 (Appendix A) and saying the A/ReSplat command lines are not echoed in
run.log but are exactly those in `scene_eval_r15.sh` (sha `e4060a8b…`, identical on H1
and Mac). Baseline line should also show `--realloc_frac 0.0 --realloc_at 1250`
(passed explicitly in the real base arm).

**C4 — §3.4, line 94 (⚠️ sentence).**
Says: `本地 run_pipeline.py 目前默认是 8 次 shot，与实测配方/DELIVERY.md 不一致，已列为本轮修复项。`
Now false: `run_pipeline.py` defaults to **one shot** — `--realloc-frac 0.15` and
`"--realloc_at", str(max(1, args.b_iterations // 4))`, with the comment
"ONE shot at the end of the structural window … Multi-shot (8 shots at iterations*k/20)
was retracted"; `DELIVERY.md` line 29 lists the same single-shot default, and
`git diff run_pipeline.py` shows the old 8-shot generator removed. Should say:
`已修复（本轮）：run_pipeline.py / DELIVERY.md 已改为单 shot（realloc_frac 0.15,
realloc_at = iterations//4），与实测配方一致。`

**C5 — §2 table, line 39 (uncommitted-file counts).**
Says: H1 `cd355cb（+ 3 个未提交文件，+264 行 train_sweep，多 shot realloc 实现）`;
N1 `c3c1ee5（+ 8 个未提交文件）`.
Partially stale/ambiguous: H1 now has **3 tracked modifications** (`research/resplat_variant.py`,
`research/run_sweep.py`, `research/train_sweep.py`) **+ 13 untracked files** = 16 `git status`
entries (`+264` train_sweep is right: 255 insertions + 9 deletions); N1 now has **8 tracked
changes (6 M + 2 D) + 11 untracked = 19** entries (2 of the untracked are task-1 artifacts:
`records/STATE-SHIP-20261030.md`, `records/evalnode-manifest-13.json`). Should say e.g.
"3 tracked + 13 untracked (H1) / 8 tracked + 11 untracked (N1), as of …, task-1 artifacts
added later".

**C6 — §1 table lines 25–26 + §3 line 49 (snapshot values now drifted; mark as snapshot).**
- H1 root free: doc `剩 132G` → now **124G** (env backup 7.25G + source 235M + repo growth).
- N1 `/root/gpufree-data`: doc `剩 142G` → now **118G** (inputs13 3.1G, image_cache 4.3G,
  backup 8.4G, Pareto sweep outputs).
- N1 root free: doc `312M` → now **309M**.
- §3 line 49 `两台全部空闲，无作业在跑` → now false by design: H1 GPU is running the Lead's
  LPIPS screening; N1's 3 GPUs are running the E2 Pareto sweep (~45% / 1% / 0% util at audit).
These were true at 22:47–23:00; recommend adding "(快照：22:50)" and updating the live
numbers in the next revision. Same for §4 statuses (line 103–105): E1 now **done** (13/13),
E2 **running**, E3 **pending** (blocked by E2), not "进行中 / 待放行 / 分析器先行".

**C7 — §3.4, line 94 (citation for the 6.55/10.86 pair).**
Says: `同配置两次 6.55 vs 10.86（质量分），已交 codex 调查`.
Supported, but the evidence is not a RESEARCH §: it is
`research/advisory/HANDOFF_MULTISHOT_REALLOC_20261030.md` lines 23–36 ("完全相同的配置,
同 GPU、同顺序执行、同 init。质量差 4.3 分、B 差 15.7 秒"). Suggest citing that file.

## Notes (non-blocking, no edit required)

1. **Rounding convention in §3.1.** B(s) cells `45.35→45.4` (dl3dv/9a4516 base) and
   `22.05→22.1` (hw/basketball delivery) are round-half-up; banker's rounding would print
   45.3 / 22.0. All other cells are unambiguous. Means are computed from unrounded totals
   (41.370893 / 49.598723 / 8.227830); averaging the *displayed* 2-dp cells gives
   41.370 / 49.598 (last-digit difference) — a "均值由未舍入值计算" footnote avoids nitpicks.
2. **§3.3 "3 个难例 × 500/2000/5000"** is a slight over-generalisation (3e4dbe has only S=500);
   the doc itself says 缺, so no change needed.
3. **§3.4 items closed elsewhere.** "粗到细分辨率 / LPIPS 入损 / num_refine>2 / 激进致密化 /
   预剪枝" are all in `docs/FINAL-REPORT.md` §12 (lines 188–194; 预剪 numbers at RESEARCH §10
   line 182: 58.11 vs 59.02); RESEARCH §6 still lists several as "运行中" (line 111), so the
   correct pointer for those is FINAL-REPORT, not RESEARCH.
4. **Budget usage is operationally confirmed** by the TIME formula on the recorded `b` values
   (§3.2/§5), so the §5 budget table is internally consistent with §3.1.

## Appendix A — verbatim executed delivery-arm command (run.log line 57, first case)

```
+ /usr/local/miniconda3/bin/conda run --no-capture-output -n organizer-fastgs python -m research.train_sweep \
  --source=outputs/scene13r15/work --model_path=outputs/scene13r15/del/model --eval -r 1 --sh_degree 3 --quiet \
  --iterations 5000 --snapshots 5000 --seed 20261015 --position_lr_max_steps 5000 --densify_until_iter 1250 \
  --opacity_reset_interval 0 --lambda_dssim 0.6 --scaling_lr 0.01 --image_cache outputs/scene13r15/cache \
  --realloc_frac 0.15 --realloc_at 1250 --opt_step_schedule  --inject_holes 0 --inject_at 800
```

Shell-level invocation in `research/experiments/scene_eval_r15.sh` (sha `e4060a8b…`) is:
`python -m research.run_sweep --case <case> --work <work> --init-ply <ply> --out <out>
--iterations 5000 --snapshots "" --budget <budget> --opacity-reset-interval 0 --lambda-dssim 0.6
--scaling-lr 0.01 --position-lr-max-steps 5000 --densify-until 1250 --image-cache <cache>
--realloc-frac 0.15 --realloc-at 1250` (run_sweep adds `--seed 20261015`,
`--snapshots 5000`, and maps to the `train_sweep` flags above; `--densify-from` stays default 500).
