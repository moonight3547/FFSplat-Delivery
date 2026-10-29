# Module-B evidence summary — 159 archived rows (2026-10-30)

Built from **130 `summary.json` files / 159 scored module-B rows**: 11 under `records/artifacts-20261030/**` (40 rows) and 119 under `records/artifacts-20261030b/{h2,h3}/outputs/experiments/<campaign>/<tag>/summary.json`, plus the official baseline `records/o0-baseline-20261030.json`. Rows come from the existing parser `research/collect.py::rows_from` (`case, name, iteration, gaussians, b_seconds, psnr, ssim, lpips, quality, time, total`). The proxy is re-derived from the stored PSNR/SSIM/LPIPS/B columns as a cross-check: `quality_points` reproduces exactly (max |Δ| = 0.0000) and `time_points` to 0.0012 (B is stored rounded to 2 dp). Nothing ran on a GPU; only `records/SUMMARY.md` and `records/best-recipe.json` were written. Anything not traceable to a file here is marked **unarchived/unverified**.

## 1. Scoring contract, budgets, host classes

| component | pts | rule |
|---|---:|---|
| PSNR / SSIM / LPIPS (VGG) | 20 / 20 / 40 | linear and clamped: PSNR 15 → 40 dB, SSIM 0.70 → 0.95, LPIPS 0.25 → 0.05 (lower is better) |
| TIME | 20 | `20·clamp((1−r)/0.90, 0, 1)`, `r = B_seconds/budget`: full marks at r ≤ 0.10, zero at r ≥ 1.00 |
| total | 100 | quality (80) + TIME (20) |

**Budgets** (FastGS *random-init, 30,000 steps*, same GPU — not classic-3DGS 30k, not ReSplat-continued 30k): dl3dv **94.29 s**, hw/billboard **99.49 s**, mipnerf360/counter **92.57 s**. A row with `b_seconds ≥ budget` is **INVALID** — excluded from every mean, listed in §3.2. The organiser's *official* total is **unconfirmed**; everything here is the proxy above.

**H1 vs h3.** The old archive is host H1 (the 4090 that produced the baseline and the budgets); h3 is a second full-speed host. Both are TIMING-VALID by the brief, but no identical config+iteration pair exists across H1 and h3 (so cross-host timing equality is *assumed*, not measured) — every winner below is an h3 row, and the baseline is H1, which is the same class but not the same board.

**Host classes.** `records/artifacts-20261030/**` (40 rows) and `records/artifacts-20261030b/h3/**` (84 rows) are full-speed → **TIMING-VALID**: B, TIME and validity are meaningful. `records/artifacts-20261030b/h2/**` (35 rows) is the shared/throttled host → **QUALITY-ONLY**: PSNR/SSIM/LPIPS usable, B/TIME/validity meaningless and never scored here.

**Host-class check** — 12 `campaignK` tags were run on *both* hosts at identical settings and iteration (all 12 settings dicts and iteration counts verified byte-identical): h2/h3 B ratio **2.08–4.58×, median 3.40×** (`k_hw_early_4k` @4k 97.18 s vs 21.24 s = 4.58×; `k_dl_nodens_4k_sharp` @4k 95.89 s vs 46.04 s = 2.08×) while quality is the same run within noise (|ΔPSNR| ≤ 0.42 dB, |ΔSSIM| ≤ 0.0009, |ΔLPIPS| ≤ 0.0032). The measured 2–4.6× is narrower than the 4–6× in the brief (§9.4). Second trap: the 23 h2 files of campaignB/campaignC store `budget_seconds = 95.0` for every case (wrong for dl3dv and counter) and h2 `campaignK` rows carry a non-zero TIME — so h2 rows *look* scorable. They are never scored.

## 2. Best row per (campaign tag, case) — 95 TIMING-VALID groups (of 118)

One row per tag (its best *scoring* configuration), sorted by case then total ↓; `TV` = TIMING-VALID, `INV` = every archived point of that tag is over budget (×ratio). All 14 INVALID rows are itemised in §3.2. `campaignK` runs on both hosts and the h3 copy is shown. `- <hash>` is elided from `dl3dv` tag names.

| case | tag | it | B (s) | PSNR | SSIM | LPIPS | Q | TIME | total | class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dl3dv | `k_dl_early_4k` | 4000 | 29.0 | 34.371 | 0.9662 | 0.1095 | 63.60 | 15.38 | **78.98** | TV |
| dl3dv | `i_dl_nodens_4k` | 4000 | 31.6 | 34.062 | 0.9645 | 0.1108 | 63.09 | 14.78 | **77.87** | TV |
| dl3dv | `f_dl_3k_s2x` | 3000 | 25.2 | 33.941 | 0.9622 | 0.1197 | 61.21 | 16.29 | **77.50** | TV |
| dl3dv | `f_dl_4k_s2x_d06` | 4000 | 30.6 | 33.824 | 0.9647 | 0.1149 | 62.08 | 15.01 | **77.09** | TV |
| dl3dv | `g_dl_5k_d06` | 5000 | 35.4 | 34.199 | 0.9664 | 0.1117 | 63.02 | 13.88 | **76.90** | TV |
| dl3dv | `sharpen-dl3dv` | 4000 | 27.7 | 33.397 | 0.9639 | 0.1191 | 60.90 | 15.68 | **76.58** | TV |
| dl3dv | `d_dl_4k` | 4000 | 30.6 | 33.832 | 0.9627 | 0.1188 | 61.31 | 15.00 | **76.31** | TV |
| dl3dv | `e_dl_3k` | 3000 | 28.4 | 33.850 | 0.9615 | 0.1215 | 60.78 | 15.53 | **76.31** | TV |
| dl3dv | `k_dl_half_4k` | 4000 | 31.4 | 33.373 | 0.9613 | 0.1189 | 60.92 | 14.83 | **75.75** | TV |
| dl3dv | `f_dl_4k_s2x` | 4000 | 35.4 | 33.896 | 0.9635 | 0.1170 | 61.72 | 13.88 | **75.59** | TV |
| dl3dv | `e_dl_5k` | 5000 | 36.4 | 34.076 | 0.9642 | 0.1166 | 61.94 | 13.64 | **75.58** | TV |
| dl3dv | `fastgs-dl3dv` | 4000 | 35.4 | 33.665 | 0.9644 | 0.1163 | 61.67 | 13.88 | **75.55** | TV |
| dl3dv | `d_dl_6k` | 6000 | 39.4 | 34.056 | 0.9649 | 0.1137 | 62.50 | 12.95 | **75.45** | TV |
| dl3dv | `e_dl_4k_sh500` | 4000 | 33.9 | 33.789 | 0.9628 | 0.1193 | 61.17 | 14.23 | **75.40** | TV |
| dl3dv | `g_dl_4k_d05` | 4000 | 33.0 | 33.174 | 0.9637 | 0.1182 | 60.90 | 14.43 | **75.33** | TV |
| dl3dv | `g_dl_6k_d06` | 6000 | 43.2 | 33.990 | 0.9669 | 0.1103 | 63.13 | 12.04 | **75.18** | TV |
| dl3dv | `e_dl_4k_dens` | 4000 | 34.0 | 33.858 | 0.9627 | 0.1217 | 60.75 | 14.20 | **74.95** | TV |
| dl3dv | `d_dl_8k` | 8000 | 49.1 | 34.373 | 0.9666 | 0.1115 | 63.20 | 10.65 | **73.85** | TV |
| dl3dv | `e_dl_4k_d06` | 4000 | 33.3 | 30.981 | 0.9608 | 0.1194 | 58.90 | 14.37 | **73.28** | TV |
| dl3dv | `prune-dl3dv` | 4000 | 28.2 | 33.121 | 0.9625 | 0.1343 | 57.64 | 15.56 | **73.20** | TV |
| dl3dv | `k_dl_nodens_4k_sharp` | 4000 | 46.0 | 33.648 | 0.9630 | 0.1171 | 61.50 | 11.37 | **72.87** | TV |
| dl3dv | `k_dl_half_4k_sharp` | 4000 | 43.4 | 33.022 | 0.9609 | 0.1216 | 60.10 | 11.99 | **72.09** | TV |
| dl3dv | `d_dl_10k` | 10000 | 69.0 | 34.684 | 0.9679 | 0.1095 | 63.85 | 5.97 | **69.82** | TV |
| dl3dv | `dl3dv-noreset` | 8000 | 51.7 | 33.336 | 0.9511 | 0.1324 | 58.19 | 10.04 | **68.22** | TV |
| dl3dv | `d_dl_12k` | 12000 | 77.3 | 34.514 | 0.9684 | 0.1093 | 63.75 | 3.99 | **67.74** | TV |
| dl3dv | `random-dl3dv` | 4000 | 24.5 | 26.162 | 0.8933 | 0.2401 | 26.37 | 16.44 | **42.81** | TV |
| dl3dv | `l_dl_20k` | 20000 | 103.9 | 35.664 | 0.9750 | 0.0932 | 67.89 | — | — | **INV** ×1.10 |
| hw/billboard | `sharpen-hw-billboard` | 4000 | 22.4 | 22.814 | 0.8075 | 0.1448 | 35.89 | 17.21 | **53.10** | TV |
| hw/billboard | `k_hw_early_4k` | 4000 | 21.2 | 22.625 | 0.8026 | 0.1476 | 34.79 | 17.48 | **52.27** | TV |
| hw/billboard | `k_hw_half_4k_sharp` | 4000 | 23.4 | 22.585 | 0.8016 | 0.1502 | 34.16 | 16.99 | **51.14** | TV |
| hw/billboard | `fastgs-hw-billboard` | 4000 | 28.3 | 22.629 | 0.8040 | 0.1462 | 35.18 | 15.90 | **51.08** | TV |
| hw/billboard | `g_hw_4k_d06_op2x` | 4000 | 22.5 | 22.530 | 0.8012 | 0.1512 | 33.88 | 17.20 | **51.08** | TV |
| hw/billboard | `f_hw_4k_s2x_d06` | 4000 | 27.7 | 22.627 | 0.8038 | 0.1471 | 34.99 | 16.03 | **51.02** | TV |
| hw/billboard | `g_hw_4k_d05` | 4000 | 27.1 | 22.658 | 0.8034 | 0.1482 | 34.76 | 16.16 | **50.92** | TV |
| hw/billboard | `f_hw_4k_s3x` | 4000 | 26.8 | 22.672 | 0.8025 | 0.1488 | 34.58 | 16.24 | **50.82** | TV |
| hw/billboard | `e_hw_4k_scale2x` | 4000 | 26.8 | 22.689 | 0.8027 | 0.1496 | 34.45 | 16.25 | **50.69** | TV |
| hw/billboard | `f_hw_3k_s2x` | 3000 | 22.1 | 22.573 | 0.7994 | 0.1534 | 33.33 | 17.30 | **50.63** | TV |
| hw/billboard | `f_hw_4k_s2x` | 4000 | 26.8 | 22.685 | 0.8024 | 0.1501 | 34.32 | 16.23 | **50.55** | TV |
| hw/billboard | `g_hw_5k_d06` | 5000 | 32.7 | 22.608 | 0.8029 | 0.1451 | 35.30 | 14.91 | **50.21** | TV |
| hw/billboard | `e_hw_5k` | 5000 | 28.0 | 22.645 | 0.8010 | 0.1503 | 34.14 | 15.97 | **50.10** | TV |
| hw/billboard | `d_hw_4k` | 4000 | 26.8 | 22.696 | 0.8018 | 0.1524 | 33.82 | 16.23 | **50.05** | TV |
| hw/billboard | `prune-hw-billboard` | 4000 | 22.9 | 22.117 | 0.7971 | 0.1527 | 32.92 | 17.10 | **50.02** | TV |
| hw/billboard | `e_hw_4k_mult10` | 4000 | 28.2 | 22.683 | 0.8024 | 0.1518 | 33.98 | 15.92 | **49.90** | TV |
| hw/billboard | `d_hw_3k` | 3000 | 22.3 | 22.583 | 0.7984 | 0.1566 | 32.62 | 17.24 | **49.86** | TV |
| hw/billboard | `i_hw_nodens_4k` | 4000 | 21.7 | 22.517 | 0.7960 | 0.1568 | 32.33 | 17.37 | **49.71** | TV |
| hw/billboard | `e_hw_4k_d06` | 4000 | 32.1 | 22.622 | 0.8031 | 0.1485 | 34.65 | 15.06 | **49.70** | TV |
| hw/billboard | `i_hw_sharp_nodens_4k` | 4000 | 19.6 | 22.530 | 0.7959 | 0.1605 | 31.60 | 17.85 | **49.45** | TV |
| hw/billboard | `d_hw_6k` | 6000 | 37.0 | 22.658 | 0.8033 | 0.1475 | 34.89 | 13.96 | **48.85** | TV |
| hw/billboard | `e_hw_4k_sh500` | 4000 | 34.3 | 22.698 | 0.8023 | 0.1520 | 33.94 | 14.56 | **48.51** | TV |
| hw/billboard | `e_hw_4k_d08` | 4000 | 38.1 | 22.529 | 0.8032 | 0.1485 | 34.58 | 13.71 | **48.29** | TV |
| hw/billboard | `e_hw_4k_dens` | 4000 | 33.3 | 22.558 | 0.8001 | 0.1529 | 33.47 | 14.78 | **48.26** | TV |
| hw/billboard | `k_hw_half_4k` | 4000 | 36.5 | 22.517 | 0.7996 | 0.1501 | 33.96 | 14.07 | **48.03** | TV |
| hw/billboard | `d_hw_8k` | 8000 | 47.2 | 22.601 | 0.8034 | 0.1463 | 35.09 | 11.69 | **46.78** | TV |
| hw/billboard | `k_hw_nodens_5k_sharp` | 5000 | 36.7 | 22.602 | 0.7986 | 0.1569 | 32.59 | 14.03 | **46.62** | TV |
| hw/billboard | `d_hw_10k` | 10000 | 51.7 | 22.667 | 0.8041 | 0.1442 | 35.62 | 10.68 | **46.30** | TV |
| hw/billboard | `hw-noreset` | 4000 | 22.6 | 22.684 | 0.7885 | 0.1761 | 28.01 | 17.17 | **45.17** | TV |
| hw/billboard | `l_hw_10k` | 10000 | 64.9 | 22.597 | 0.8041 | 0.1400 | 36.41 | 7.72 | **44.13** | TV |
| hw/billboard | `hw-dense-noreset` | 8000 | 48.5 | 22.729 | 0.7975 | 0.1573 | 32.52 | 11.39 | **43.91** | TV |
| hw/billboard | `hw-default` | 4000 | 29.7 | 22.595 | 0.7843 | 0.1841 | 26.00 | 15.58 | **41.58** | TV |
| hw/billboard | `hw-resramp` | 8000 | 57.3 | 22.545 | 0.7836 | 0.1775 | 27.22 | 9.43 | **36.65** | TV |
| hw/billboard | `hw-probe` | 400 | 8.6 | 22.171 | 0.7499 | 0.2177 | 16.19 | 20.00 | **36.19** | TV |
| hw/billboard | `hw-lpips` | 4000 | 54.9 | 22.313 | 0.7649 | 0.1938 | 22.28 | 9.96 | **32.24** | TV |
| hw/billboard | `random-hw-billboard` | 4000 | 24.1 | 16.252 | 0.5780 | 0.3778 | 1.00 | 16.84 | **17.84** | TV |
| hw/billboard | `l_hw_20k` | 20000 | 157.4 | 22.465 | 0.8033 | 0.1377 | 36.70 | — | — | **INV** ×1.58 |
| hw/billboard | `l_hw_40k` | 40000 | 299.6 | 22.523 | 0.8000 | 0.1375 | 36.52 | — | — | **INV** ×3.01 |
| mipnerf360/counter | `k_co_early_5k` | 5000 | 33.7 | 27.708 | 0.8961 | 0.1885 | 38.15 | 14.12 | **52.28** | TV |
| mipnerf360/counter | `prune-mipnerf360-counter` | 5000 | 35.3 | 27.429 | 0.8953 | 0.1910 | 37.37 | 13.74 | **51.11** | TV |
| mipnerf360/counter | `g_co_5k_d06` | 5000 | 39.4 | 27.466 | 0.8962 | 0.1898 | 37.71 | 12.78 | **50.48** | TV |
| mipnerf360/counter | `f_co_6k_s2x_d06` | 6000 | 45.2 | 27.637 | 0.8997 | 0.1851 | 39.07 | 11.37 | **50.43** | TV |
| mipnerf360/counter | `fastgs-mipnerf360-counter` | 5000 | 40.0 | 27.506 | 0.8966 | 0.1896 | 37.81 | 12.61 | **50.42** | TV |
| mipnerf360/counter | `sharpen-mipnerf360-counter` | 5000 | 32.7 | 27.366 | 0.8940 | 0.1968 | 36.05 | 14.37 | **50.42** | TV |
| mipnerf360/counter | `e_co_6k_d06` | 6000 | 44.0 | 27.518 | 0.8976 | 0.1886 | 38.10 | 11.67 | **49.77** | TV |
| mipnerf360/counter | `g_co_4k_d05` | 4000 | 30.4 | 27.189 | 0.8887 | 0.2001 | 34.83 | 14.93 | **49.76** | TV |
| mipnerf360/counter | `d_co_6k` | 6000 | 39.3 | 27.594 | 0.8950 | 0.1939 | 36.90 | 12.79 | **49.69** | TV |
| mipnerf360/counter | `e_co_5k` | 5000 | 34.0 | 27.424 | 0.8916 | 0.1991 | 35.45 | 14.06 | **49.50** | TV |
| mipnerf360/counter | `f_co_6k_s2x` | 6000 | 43.8 | 27.728 | 0.8964 | 0.1908 | 37.73 | 11.71 | **49.44** | TV |
| mipnerf360/counter | `g_co_4k_d06` | 4000 | 34.2 | 27.191 | 0.8897 | 0.1977 | 35.39 | 14.02 | **49.41** | TV |
| mipnerf360/counter | `e_co_6k_sh500` | 6000 | 40.5 | 27.538 | 0.8936 | 0.1941 | 36.70 | 12.51 | **49.21** | TV |
| mipnerf360/counter | `d_co_8k` | 8000 | 50.4 | 27.794 | 0.8989 | 0.1876 | 38.63 | 10.13 | **48.75** | TV |
| mipnerf360/counter | `k_co_half_5k` | 5000 | 37.9 | 27.160 | 0.8897 | 0.1974 | 35.42 | 13.12 | **48.55** | TV |
| mipnerf360/counter | `d_co_4k` | 4000 | 30.1 | 27.122 | 0.8858 | 0.2060 | 33.36 | 15.00 | **48.37** | TV |
| mipnerf360/counter | `e_co_6k_dens` | 6000 | 42.4 | 27.457 | 0.8933 | 0.1968 | 36.07 | 12.05 | **48.12** | TV |
| mipnerf360/counter | `f_co_4k_s2x` | 4000 | 33.8 | 27.166 | 0.8866 | 0.2036 | 33.94 | 14.12 | **48.06** | TV |
| mipnerf360/counter | `d_co_10k` | 10000 | 59.8 | 27.954 | 0.9022 | 0.1823 | 40.08 | 7.87 | **47.95** | TV |
| mipnerf360/counter | `i_co_nodens_5k` | 5000 | 36.1 | 27.058 | 0.8842 | 0.2021 | 33.96 | 13.56 | **47.53** | TV |
| mipnerf360/counter | `d_co_3k` | 3000 | 25.0 | 26.619 | 0.8766 | 0.2165 | 30.12 | 16.22 | **46.34** | TV |
| mipnerf360/counter | `k_co_half_5k_sharp` | 5000 | 44.7 | 27.017 | 0.8862 | 0.2062 | 33.27 | 11.50 | **44.77** | TV |
| mipnerf360/counter | `counter-noreset` | 8000 | 54.0 | 27.552 | 0.8833 | 0.2079 | 33.13 | 9.27 | **42.40** | TV |
| mipnerf360/counter | `k_co_nodens_6k_sharp` | 6000 | 44.2 | 26.720 | 0.8773 | 0.2164 | 30.28 | 11.61 | **41.89** | TV |
| mipnerf360/counter | `counter-default` | 8000 | 64.4 | 27.330 | 0.8808 | 0.2142 | 31.49 | 6.76 | **38.25** | TV |
| mipnerf360/counter | `random-mipnerf360-counter` | 5000 | 28.7 | 23.557 | 0.8211 | 0.2972 | 16.53 | 15.33 | **31.87** | TV |
| mipnerf360/counter | `counter-resramp` | 8000 | 89.3 | 27.009 | 0.8712 | 0.2230 | 28.70 | 0.79 | **29.49** | TV |
| mipnerf360/counter | `counter-lpips` | 4000 | 69.9 | 24.970 | 0.7933 | 0.2787 | 15.44 | 5.44 | **20.88** | TV |
| mipnerf360/counter | `l_co_10k` | 10000 | 112.5 | 28.022 | 0.9066 | 0.1738 | 42.19 | — | — | **INV** ×1.21 |
| mipnerf360/counter | `l_co_20k` | 20000 | 183.8 | 28.771 | 0.9185 | 0.1562 | 47.26 | — | — | **INV** ×1.99 |

**QUALITY-ONLY rows (h2) are excluded from the table** — 23 tag×case groups whose B/TIME/validity is meaningless; only their quality is evidence, listed here as `tag Q`:

- **dl3dv**: `c_dl_lrm_8k_d06` 64.01 · `c_dl_lrm_8k` 63.04 · `c_dl_lrm_4k` 61.26
- **hw/billboard**: `c_hw_lrm_8k_d06` 35.59 · `nb_hw_dssim04_12k` 35.49 · `nb_hw_dssim04_lr8k` 35.39 · `c_hw_lrm_8k_mult10` 35.35 · `c_hw_lrm_6k_d06` 35.33 · `c_hw_lrm_8k` 35.12 · `c_hw_lrm_6k` 34.89 · `nb_hw_dssim08` 34.14 · `nb_hw_dssim06` 34.05 · `c_hw_lrm_4k` 33.78 · `nb_hw_dssim04_mult10` 33.76 · `nb_hw_dssim04_dens` 33.47 · `c_hw_lrm_3k` 32.63 · `nb_hw_dssim04_4k` 29.50
- **mipnerf360/counter**: `c_co_lrm_8k_d06` 40.32 · `c_co_lrm_8k` 38.84 · `nb_co_dssim06` 36.66 · `nb_co_dssim04` 35.49 · `c_co_lrm_4k_d06` 34.59 · `c_co_lrm_4k` 33.10

## 3. Best per case (TIMING-VALID only)

| case | tag | iters | what differs from default | B (s)/budget | PSNR | SSIM | LPIPS | Q | TIME | total |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| dl3dv | `k_dl_early_4k` | 4000 | matched LR + noreset, λdssim 0.6, scaling_lr 0.01, densify stops at **1000** | 29.03/94.29 | 34.371 | 0.9662 | 0.1095 | 63.60 | 15.38 | **78.98** |
| hw/billboard | `sharpen-hw-billboard` | 4000 | opacity-sharpened ReSplat init + FastGS, matched LR + noreset, λdssim 0.6, scaling_lr 0.01 | 22.45/99.49 | 22.814 | 0.8075 | 0.1448 | 35.89 | 17.21 | **53.10** |
| mipnerf360/counter | `k_co_early_5k` | 5000 | matched LR + noreset, λdssim 0.6, scaling_lr 0.01, densify stops at **1250** | 33.74/92.57 | 27.708 | 0.8961 | 0.1885 | 38.15 | 14.12 | **52.28** |
| **mean** | | | | 28.41 | 28.298 | | | 45.88 | 15.57 | **61.45** |

Per-case tuning gains **+2.25** over the best uniform setting (61.45 vs 59.20) but needs three different iteration counts, so it is not one deployable recipe.

### 3.2 INVALID (B ≥ budget) — 14 of 159 rows, excluded from every mean

**dl3dv**: `l_dl_20k` 20k 103.9s ×1.10 (Q 67.89) · **hw/billboard**: `hw-lpips` 8k 108.1s ×1.09 (Q 28.18), `hw-lpips` 12k 161.6s ×1.62 (Q 0.00), `hw-resramp` 16k 103.3s ×1.04 (Q 10.40), `hw-resramp` 24k 133.7s ×1.34 (Q 0.00), `l_hw_20k` 20k 157.4s ×1.58 (Q 36.70), `l_hw_40k` 40k 299.6s ×3.01 (Q 36.52) · **mipnerf360/counter**: `counter-lpips` 8k 128.9s ×1.39 (Q 27.86), `counter-lpips` 12k 185.8s ×2.01 (Q 0.00), `counter-resramp` 16k 136.1s ×1.47 (Q 20.47), `counter-resramp` 24k 164.4s ×1.78 (Q 0.00), `counter-default` 20k 114.4s ×1.24 (Q 35.54), `l_co_10k` 10k 112.5s ×1.21 (Q 42.19), `l_co_20k` 20k 183.8s ×1.99 (Q 47.26)

## 4. Best uniform configuration (one setting for all 3 cases, TIMING-VALID only)

A candidate must be the same recipe **and the same iteration count** on all three cases with byte-identical `settings` (only `position_lr_max_steps`/`opacity_reset_interval` may scale with the iteration count); eight qualify.

| rank | uniform setting, named by its actual flags | it | dl3dv | hw | counter | mean | vs base |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `campaignG`: noreset (`opacity_reset_interval=5001`) + matched LR (`position_lr_max_steps=5000`) + λdssim 0.6 + scaling_lr 0.01, densify 500–15000/100 | 5000 | 76.90 | 50.21 | 50.48 | **59.20** | +9.57 |
| 2 | `campaignG`: noreset (`opacity_reset_interval=4001`) + matched LR (`position_lr_max_steps=4000`) + λdssim 0.5 + scaling_lr 0.01, densify 500–15000/100 | 4000 | 75.33 | 50.92 | 49.76 | **58.67** | +9.04 |
| 3 | `campaignE`: noreset (`opacity_reset_interval=5001`) + matched LR (`position_lr_max_steps=5000`) + λdssim 0.4 + scaling_lr 0.005, densify 500–15000/100 | 5000 | 75.58 | 50.10 | 49.50 | **58.40** | +8.77 |
| 4 | `campaignD`: noreset (`opacity_reset_interval=4001`) + matched LR (`position_lr_max_steps=4000`) + λdssim 0.4 + scaling_lr 0.005, densify 500–15000/100 | 4000 | 76.31 | 50.05 | 48.37 | **58.24** | +8.61 |
| 5 | `campaignF`: noreset (`opacity_reset_interval=4001`) + matched LR (`position_lr_max_steps=4000`) + λdssim 0.4 + scaling_lr 0.01, densify 500–15000/100 | 4000 | 75.59 | 50.55 | 48.06 | **58.07** | +8.44 |
| 6 | `campaignD`: noreset (`opacity_reset_interval=6001`) + matched LR (`position_lr_max_steps=6000`) + λdssim 0.4 + scaling_lr 0.005, densify 500–15000/100 | 6000 | 75.45 | 48.85 | 49.69 | **57.99** | +8.37 |
| 7 | `campaignD`: noreset (`opacity_reset_interval=8001`) + matched LR (`position_lr_max_steps=8000`) + λdssim 0.4 + scaling_lr 0.005, densify 500–15000/100 | 8000 | 73.85 | 46.78 | 48.75 | **56.46** | +6.83 |
| 8 | `campaignD`: noreset (`opacity_reset_interval=10001`) + matched LR (`position_lr_max_steps=10000`) + λdssim 0.4 + scaling_lr 0.005, densify 500–15000/100 | 10000 | 69.82 | 46.30 | 47.95 | **54.69** | +5.06 |
| — | baseline O0: FastGS default schedule (reset 3000, 30k LR, λdssim 0.2), densify 500–15000/100, 10,000 it | 10000 | 62.93 | 43.26 | 42.69 | **49.63** | 0.00 |

**New best uniform mean = 59.20** (`campaignG` at 5000 iterations) vs baseline 49.63 → **+9.57**. Flags, identical on all three cases: `iterations=5000`, `opacity_reset_interval=5001`, `position_lr_max_steps=5000`, `position_lr_init 1.6e-4`, `position_lr_final 1.6e-6`, `position_lr_delay_mult 0.01`, `lambda_dssim 0.6`, `scaling_lr 0.01`, `opacity_lr 0.025`, `mult 0.5`, `densify_from_iter 500`, `densify_until_iter 15000`, `densification_interval 100`, `lpips_weight 0.0`, `res_scales [1.0]`, `densify_count_thresh null`, `max_gaussians null`, `no_densify false`, `seed 20261015`.

Not uniform, so not ranked: `campaignH` `sharpen-*` has the best 3-case mean in the archive (**60.03** = 76.58/53.10/50.42) but counter ran 5,000 iterations while dl3dv/hw ran 4,000. Its mechanism (opacity sharpening of the init, `p → logit(2p)`) is **not in `train_record.settings`** — `sharpen-*` and `fastgs-*` have identical settings dicts — so only the tag name and `docs/RESEARCH-20261030.md` §10 attest it (**unverified from JSON**).

## 5. Falsified ideas (one line each, JSON-traceable)

| # | idea | evidence |
|---:|---|---|
| 1 | **No ReSplat init** (random frustum points) is competitive | `random-*` 42.81/17.84/31.87 (mean **30.84**) vs `fastgs-*` 75.55/51.08/50.42 (mean **59.02**) at identical settings → the init is worth 28.2 points. `campaignH/random-*/summary.json` |
| 2 | **Attribute-only** B (`no_densify=true`, all structural ops off) is a uniform win | mean **58.37** vs 59.02: dl3dv **+2.32** (77.87 vs 75.55) but hw **−1.37** (49.71 vs 51.08) and counter **−2.89** (47.53 vs 50.42) — a scene-size trade, not a win. `campaignI/i_*/summary.json` |
| 3 | **Pre-pruning floaters** before FastGS helps | `prune-*` 73.20/50.02/51.11 (mean **58.11**, −0.91); counter +0.69 but dl3dv −2.35. `campaignH/prune-*/summary.json` |
| 4 | **Aggressive densification** (`densify_from_iter 100`, `interval 50`) recovers detail | loses on all three: hw 48.26 vs 50.05, counter 48.12 vs 49.69, dl3dv 74.95 vs 76.31. `campaignE/*_dens` vs `campaignD/d_*` |
| 5 | **Coarse-to-fine resolution** ramp (0.5→0.75→1.0) buys steps per second | hw 8k 36.65 vs 41.57 (`hw-default`); counter 8k 29.49 vs 42.40 (`counter-noreset`); its 16k points are over budget and 24k renders an empty scene (Q 0.00). Every 20261030b run has `res_scales [1.0]` — abandoned, not re-tested. `round2/*-resramp/summary.json` |
| 6 | **VGG-LPIPS in B's loss** (`lpips_weight 0.2`) buys the 40-point LPIPS term | hw 4k **32.24** (Q 22.28) vs 41.58 (Q 26.00) plain default at the same iteration; counter 4k **20.88** (Q 15.44) is the weakest non-degenerate scorable row in the archive; hw 8k already costs 108.14 s > 99.49 s. No 20261030b run uses `lpips_weight > 0`. `round2/{hw,counter}-lpips/summary.json` |
| 7 | **More ReSplat refine passes** keep improving the init | hw init quality 3.813 (r2) → 3.874 (r4) → 3.880 (r6); counter 5.158 → 4.985 → 4.926 (worse). Saturated after r2. B=0 rows, so `time_points` is null. `records/artifacts-20261030/**/score_summary.json` (not one of the 159) |
| 8 | **Long horizons** (10k/20k/40k) pay | `l_hw_10k` 44.13 < 50.05 at 4k; `l_hw_20k` 157.38 s ×1.58 and `l_hw_40k` 299.64 s ×3.01 INVALID; `l_co_10k` 112.47 s, `l_co_20k` 183.81 s, `l_dl_20k` 103.94 s INVALID; the valid `d_*_10k` rows (69.82/46.30/47.95) are below their own 4k–6k peaks. `campaignL/*` + `campaignD/*_10k` |
| 9 | **Classic 3DGS as module B** | 6 archived rows (`campaignJ`): B = 91.6–398.0 s vs budgets 92.57–99.49 s → 5 of 6 over budget (×1.09–4.30); the only one that fits, hw 5k at 91.62 s (×0.92), totals 37.00 vs 50.05. Source `campaignJ/j_classic_*/{module_b,metrics}.json` — **no `summary.json`**, outside the 159 |
| 10 | the **`sh500`** knob changes anything | `e_hw_4k_sh500` 48.51 vs 50.05; `e_co_6k_sh500` 49.21 vs 49.69 — and its `settings` dict is byte-identical to the base run, so the flag is **not recorded** (unarchived/unverified). `campaignE/*_sh500` |
| 11 | Snapshots near an **opacity reset** are usable quality points | 10 of 159 rows have Q = 0.00 at PSNR 5.1–12.0 dB (`hw-default` 3k/6k/12k, `counter-default` 3k/6k/12k, `hw-resramp` 24k, `counter-resramp` 24k, `hw-lpips` 12k, `counter-lpips` 12k) — they are exactly the reset points. `sweeps/*` + `round2/*` |

## 6. Marginal return: points of total per second of B

TIME costs **0.223 (hw), 0.236 (dl3dv), 0.240 (counter) points per second of B**, so extra iterations pay only while the config still returns more than that per second. Each line is the chain of Δtotal/ΔB over adjacent iteration points of the *same* config (settings identical apart from `position_lr_max_steps=iters`, `opacity_reset_interval=iters+1`); `peak` is the highest-total valid point; family C skips its degenerate reset snapshots (3k/6k/12k), so its steps are non-monotone.

| family | case | Δtotal/ΔB per adjacent step (pts/s) | peak | first drop |
|---|---|---|---|---|
| A λdssim 0.4, scaling_lr 0.005, matched LR, noreset | dl3dv | 3k→4k +0.000 · 4k→5k -0.126 · 5k→6k -0.044 · 6k→8k -0.164 · 8k→10k -0.203 · 10k→12k -0.247 | 4k | 4k→5k |
| A λdssim 0.4, scaling_lr 0.005, matched LR, noreset | hw/billboard | 3k→4k +0.043 · 4k→5k +0.048 · 5k→6k -0.140 · 6k→8k -0.203 · 8k→10k -0.106 | 5k | 5k→6k |
| A λdssim 0.4, scaling_lr 0.005, matched LR, noreset | mipnerf360/counter | 3k→4k +0.399 · 4k→5k +0.288 · 5k→6k +0.035 · 6k→8k -0.084 · 8k→10k -0.086 | 6k | 6k→8k |
| B λdssim 0.6, scaling_lr 0.01, matched LR, noreset | dl3dv | 4k→5k -0.041 · 5k→6k -0.221 | 4k | 4k→5k |
| B λdssim 0.6, scaling_lr 0.01, matched LR, noreset | hw/billboard | 4k→5k -0.161 · 5k→10k -0.189 | 4k | 4k→5k |
| B λdssim 0.6, scaling_lr 0.01, matched LR, noreset | mipnerf360/counter | 4k→5k +0.209 · 5k→6k -0.009 | 5k | 5k→6k |
| C legacy default (reset 3000, 30k LR, λdssim 0.2) | hw/billboard | 200→400 +0.868 · 1k→2k +0.082 · 2k→4k +0.073 · 4k→8k -0.000 · 8k→20k -0.239 | 4k | 4k→8k |
| C legacy default (reset 3000, 30k LR, λdssim 0.2) | mipnerf360/counter | 1k→2k +0.016 · 2k→4k -0.070 · 4k→8k +0.240 | 8k | 2k→4k |

**Crossover.** The total turns down at the first step whose ΔQ/ΔB falls below the TIME penalty. Winning recipe (family B): **dl3dv 4k→5k** (0.195 < 0.236), **hw/billboard 4k→5k** (0.062 < 0.223), **mipnerf360/counter 5k→6k** (0.232 < 0.240); the older family A crosses at dl3dv 4k→5k (0.110), hw/billboard 5k→6k (0.084), mipnerf360/counter 6k→8k (0.156). So the score-optimal horizon is **3k–4k dl3dv, 4k–5k hw, 5k–6k counter** — roughly a quarter of the budget: TIME, not quality, is what every extra step costs.

## 7. Reproduce

```bash
# 1. all 159 rows with their stored scores
python3 -m research.collect records/artifacts-20261030 records/artifacts-20261030b --kind B-sweep

# 2. grouping / host class / re-derived totals (what §§2–6 are computed from)
python3 - <<'PY'
from pathlib import Path
from research.collect import rows_from
BUD={'dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f':94.29,'hw/billboard':99.49,'mipnerf360/counter':92.57}
q=lambda r:20*min(max((r['psnr']-15)/25,0),1)+20*min(max((r['ssim']-.7)/.25,0),1)+40*min(max((.25-r['lpips'])/.2,0),1)
R=[dict(r,tag=p.parent.name,h2='/h2/' in str(p)) for root in ('records/artifacts-20261030','records/artifacts-20261030b')
   for p in sorted(Path(root).rglob('summary.json')) for r in rows_from(p) if r['kind']=='B-sweep']
for r in R: b=BUD[r['case']]; r['ok']=not r['h2'] and r['b_seconds']<b; r['tot']=q(r)+(20*max((1-r['b_seconds']/b)/.9,0) if r['ok'] else 0)
key=lambda x:(not x['h2'],x['tot']); best={}
for r in R: best[(r['tag'],r['case'])]=max(best.get((r['tag'],r['case']),r),r,key=key)
for c in BUD: print(c[:8],max([r for r in R if r['case']==c and r['ok']],key=key)['tag'])
PY
```

## 8. Sources

`records/artifacts-20261030/**/summary.json` (11 files, 40 rows: `sweeps/*`, `round2/*`, `round3/*`), `records/artifacts-20261030b/h3/**/summary.json` (84 files, 84 rows: campaigns D–L), `records/artifacts-20261030b/h2/**/summary.json` (35 files, 35 quality-only rows: campaigns B, C, K), `records/o0-baseline-20261030.json` (baseline + budgets), `records/artifacts-20261030/**/score_summary.json` (init-only rows) and `records/artifacts-20261030b/h3/.../campaignJ/j_classic_*/{module_b,metrics}.json` (classic-3DGS timings, no `summary.json`).

## 9. Remaining mismatches with `docs/RESEARCH-20261030.md`

1. §9 (76.31/50.05/48.37, mean 58.24) and §10 (59.02 fastgs / 60.03 sharpen) **now reproduce exactly** from `campaignD` / `campaignH`; the previous version of this file called them unarchived. The §8 λ_dssim sweep is archived only as h2 quality-only rows (`campaignB/nb_*`), so its *scores* still cannot be reproduced; the 4-GPU curve is covered only up to what campaignD archives.
2. §11 quotes classic-3DGS-as-B 5k at 108–152 s; the archive has dl 102.5, hw 91.6, co 131.3 s → the range is roughly right but **hw 5k fits its budget** (91.6 < 99.49), so "5k classic is over budget" is not true for every case.
3. §8's matched-LR effect (hw 8k 32.51 → 35.39) is right in direction, wrong in magnitude: the archive has hw 8k matched-LR = **35.09** (`d_hw_8k`) and 35.30 at 5k (`g_hw_5k_d06`).
4. The brief says h2 is 4–6× slower; the measured equal-iteration ratio is **2.08–4.58× (median 3.40×)**.

Budget caveat that still bites: under the organiser's *original* wording (classic-3DGS 30k budget = 547/915/531 s) long-B runs would be valid and every TIME would shift by ~5–10 points. Every number here uses the stricter self-measured FastGS-random-30k budgets and must not be mixed with classic-3DGS-budget numbers.
