# N1 steps-Pareto sweep — 13 scenes x 6 step counts (raw evidence)

**Status: COMPLETE — 78/78 grid cells valid, 0 failures; 24 serial anchors.**
Grid: 13 scenes x S in {500, 1000, 2000, 3000, 5000, 8000}, delivery recipe with a
step-matched schedule (only the module-B step count varies). All cells used
h1-prep's `<work>/image_cache` and the fastgs conda env for both training and the
official PSNR/SSIM/VGG-LPIPS metrics (`competition/upstream/baseline/evaluate.py`,
profile `gpufree-0|1|2`).

## 0. Headline numbers (13-scene mean of the organiser proxy total, /100)

| policy | mean total | B source |
|---|---|---|
| (a) **fixed S=5000 = current delivery** | **51.908** | serial, single card (13/13 measured) |
| (b) best *uniform* S | **S=5000 -> 51.908** (measured, same cell as (a)); 51.845 if concurrent quality is mixed with corrected B | serial-corrected |
| (c) per-scene oracle (argmax S) | **52.567 (+0.660 vs (a))** | serial where anchored: all 13 argmax cells AND all 13 S=5000 cells were measured serially, so this oracle is almost fully measured, not modelled. Mixing concurrent quality with corrected B gives an equivalent estimate of 52.491 (+0.583) |

- Uniform-S means (serial-corrected B): S=500 **40.67**, S=1000 **45.23**, S=2000
  **49.02**, S=3000 **50.31**, S=5000 **51.91**, S=8000 **50.81**. 5000 is the best
  uniform S; 3000 is 1.60 behind, 8000 is 1.10 behind.
- Oracle gain over delivery is only **+0.660**. Only 5/13 scenes gain >0.5:
  `dl3dv/0f622ea` +2.05 (S=8000), `hw/trashbin` +1.62 (S=3000), `dl3dv/9a451604`
  +1.05 (S=3000), `dl3dv/2dad2d29` +1.03 (S=8000), `mipnerf360/bonsai` +1.03
  (S=8000); `dl3dv/001dccbc` +0.80 (S=3000), `mip/bicycle` +0.38 (S=2000), and the
  remaining 6 scenes gain <=0.25 (two gain exactly 0 at S=5000).
- **Conclusion: the steps axis is closed for the uniform delivery policy — fixed
  5000 stays optimal.** A legitimate adaptive stop rule would have at most ~+0.66
  headroom and would need to predict per-scene argmax from a pre-B/train-time
  signal; that is below the +1.5 threshold agreed with the Lead, so no adaptive
  rule / no train-view init probe is warranted. (The 4 S=8000 optima and 6 S=3000
  optima are not predictable from a single cheap scene statistic with 13 samples.)

Per-scene argmax distribution: S=8000 x4, S=3000 x6, S=2000 x1, S=5000 x2.

## 0b. Per-scene argmax, measured serially (the oracle evidence)

| scene | argmax S (all cells serial) | total@argmax (serial) | total@5000 (serial) | gain |
|---|---|---|---|---|
| dl3dv/0f622eaabc | 8000 | 34.328 | 32.273 | +2.054 |
| hw/trashbin | 3000 | 67.489 | 65.865 | +1.624 |
| dl3dv/9a451604b8 | 3000 | 61.462 | 60.413 | +1.049 |
| dl3dv/2dad2d2989 | 8000 | 51.544 | 50.511 | +1.034 |
| mipnerf360/bonsai | 8000 | 65.843 | 64.814 | +1.029 |
| dl3dv/001dccbc1f | 3000 | 81.026 | 80.231 | +0.795 |
| mipnerf360/bicycle | 2000 | 26.272 | 25.891 | +0.382 |
| hw/billboard | 3000 | 54.273 | 54.025 | +0.248 |
| mipnerf360/counter | 8000 | 55.870 | 55.726 | +0.143 |
| hw/basketball | 3000 | 67.976 | 67.867 | +0.109 |
| hw/car | 3000 | 49.031 | 48.925 | +0.106 |
| dl3dv/0b5635d53b | 5000 | 32.157 | 32.157 | +0.000 |
| dl3dv/3e4dbe4135 | 5000 | 36.106 | 36.106 | +0.000 |

Oracle mean = **52.567**, delivery mean = **51.908**, gain = **+0.660**.

## 1. Policy comparison (13-scene mean of total proxy)

| policy | mean total | B source | n |
|---|---|---|---|
| (a) fixed S=5000 = delivery | **51.409** | concurrent | 13 |
| (a) fixed S=5000 = delivery | **51.908** | serial (clean) | 13 |
| (b) best uniform S = 5000 | 51.409 | concurrent | 13 |
| (b') best uniform S = 5000 | 51.845 | serial-corrected (est) | 13 |
| (c) per-scene oracle (argmax) | 52.197 | concurrent | 13 |
| (c') per-scene oracle (argmax) | 52.491 | serial-corrected (est) | 13 |

## 2. Per-scene table (S ascending; q / B / t / total)

Concurrent rows are the raw 3-lane measurement; `B*`/`total*` use the serial-corrected B (measured anchor where it exists, else scaled by that scene's S=5000 anchor).

| scene | S | q | B conc | t | total | B* | total* |
|---|---|---|---|---|---|---|---|
| dl3dv/001dccbc1f | 500 | 53.54 | 10.95 | 19.64 | 73.182 | 10.67 | 73.248 |
| dl3dv/001dccbc1f | 1000 | 58.04 | 10.72 | 19.70 | 77.735 | 10.44 | 77.800 |
| dl3dv/001dccbc1f | 2000 | 61.09 | 16.74 | 18.28 | 79.370 | 16.31 | 79.471 |
| dl3dv/001dccbc1f **[serial]** | 3000 | 62.76 | 18.81 | 17.79 | 80.549 | 17.34 | 80.896 |
| dl3dv/001dccbc1f **[serial]** | 5000 | 64.54 | 27.96 | 15.63 | 80.171 | 27.24 | 80.341 |
| dl3dv/001dccbc1f | 8000 | 65.95 | 40.25 | 12.74 | 78.690 | 39.21 | 78.934 |
| dl3dv/0b5635d53b | 500 | 5.23 | 27.41 | 15.76 | 20.991 | 23.69 | 21.867 |
| dl3dv/0b5635d53b | 1000 | 8.59 | 12.12 | 19.37 | 27.956 | 10.48 | 28.343 |
| dl3dv/0b5635d53b | 2000 | 12.19 | 15.39 | 18.60 | 30.782 | 13.30 | 31.274 |
| dl3dv/0b5635d53b | 3000 | 14.33 | 33.79 | 14.26 | 28.587 | 29.21 | 29.666 |
| dl3dv/0b5635d53b **[serial]** | 5000 | 17.11 | 35.85 | 13.77 | 30.879 | 30.99 | 32.024 |
| dl3dv/0b5635d53b | 8000 | 22.51 | 71.06 | 5.47 | 27.984 | 61.43 | 30.255 |
| dl3dv/0f622eaabc | 500 | 3.83 | 7.66 | 20.00 | 23.833 | 7.49 | 23.833 |
| dl3dv/0f622eaabc | 1000 | 4.42 | 12.51 | 19.27 | 23.695 | 12.24 | 23.760 |
| dl3dv/0f622eaabc | 2000 | 7.95 | 15.94 | 18.47 | 26.414 | 15.59 | 26.497 |
| dl3dv/0f622eaabc | 3000 | 10.79 | 22.73 | 16.87 | 27.655 | 22.23 | 27.772 |
| dl3dv/0f622eaabc **[serial]** | 5000 | 17.90 | 33.69 | 14.28 | 32.187 | 32.95 | 32.361 |
| dl3dv/0f622eaabc **[serial]** | 8000 | 24.12 | 55.08 | 9.24 | 33.362 | 51.30 | 34.253 |
| dl3dv/2dad2d2989 | 500 | 9.67 | 6.96 | 20.00 | 29.669 | 6.92 | 29.669 |
| dl3dv/2dad2d2989 | 1000 | 12.90 | 9.19 | 20.00 | 32.895 | 9.13 | 32.895 |
| dl3dv/2dad2d2989 | 2000 | 21.34 | 13.79 | 18.97 | 40.311 | 13.71 | 40.331 |
| dl3dv/2dad2d2989 | 3000 | 28.79 | 19.15 | 17.71 | 46.497 | 19.03 | 46.524 |
| dl3dv/2dad2d2989 **[serial]** | 5000 | 35.38 | 29.64 | 15.24 | 50.620 | 29.46 | 50.662 |
| dl3dv/2dad2d2989 **[serial]** | 8000 | 40.14 | 46.61 | 11.24 | 51.376 | 45.83 | 51.560 |
| dl3dv/3e4dbe4135 | 500 | 5.48 | 26.64 | 15.94 | 21.425 | 25.37 | 21.725 |
| dl3dv/3e4dbe4135 | 1000 | 8.33 | 10.75 | 19.69 | 28.018 | 10.24 | 28.139 |
| dl3dv/3e4dbe4135 | 2000 | 12.04 | 16.93 | 18.23 | 30.267 | 16.12 | 30.458 |
| dl3dv/3e4dbe4135 | 3000 | 14.64 | 32.60 | 14.54 | 29.177 | 31.04 | 29.544 |
| dl3dv/3e4dbe4135 **[serial]** | 5000 | 21.23 | 33.12 | 14.42 | 35.651 | 31.54 | 36.023 |
| dl3dv/3e4dbe4135 | 8000 | 27.03 | 70.88 | 5.52 | 32.552 | 67.50 | 33.349 |
| dl3dv/9a451604b8 | 500 | 37.54 | 6.29 | 20.00 | 57.539 | 5.78 | 57.539 |
| dl3dv/9a451604b8 | 1000 | 40.40 | 9.95 | 19.88 | 60.274 | 9.14 | 60.397 |
| dl3dv/9a451604b8 | 2000 | 42.18 | 14.67 | 18.76 | 60.942 | 13.48 | 61.223 |
| dl3dv/9a451604b8 **[serial]** | 3000 | 43.56 | 19.45 | 17.64 | 61.201 | 18.19 | 61.498 |
| dl3dv/9a451604b8 **[serial]** | 5000 | 44.21 | 28.37 | 15.54 | 59.747 | 26.07 | 60.289 |
| dl3dv/9a451604b8 | 8000 | 44.89 | 38.68 | 13.11 | 57.993 | 35.54 | 58.732 |
| hw/basketball | 500 | 35.50 | 4.91 | 20.00 | 55.500 | 4.31 | 55.500 |
| hw/basketball | 1000 | 40.82 | 7.21 | 20.00 | 60.822 | 6.34 | 60.822 |
| hw/basketball | 2000 | 46.06 | 10.48 | 19.77 | 65.832 | 9.21 | 66.062 |
| hw/basketball **[serial]** | 3000 | 48.66 | 13.57 | 19.05 | 67.706 | 13.03 | 67.833 |
| hw/basketball **[serial]** | 5000 | 49.85 | 24.01 | 16.61 | 66.457 | 21.10 | 67.138 |
| hw/basketball | 8000 | 51.64 | 33.03 | 14.50 | 66.137 | 29.03 | 67.073 |
| hw/billboard | 500 | 23.68 | 9.02 | 20.00 | 43.676 | 8.69 | 43.676 |
| hw/billboard | 1000 | 28.27 | 6.07 | 20.00 | 48.274 | 5.85 | 48.274 |
| hw/billboard | 2000 | 32.96 | 11.17 | 19.73 | 52.691 | 10.76 | 52.783 |
| hw/billboard **[serial]** | 3000 | 35.38 | 14.73 | 18.93 | 54.317 | 14.60 | 54.346 |
| hw/billboard **[serial]** | 5000 | 36.85 | 23.84 | 16.90 | 53.748 | 22.96 | 53.945 |
| hw/billboard | 8000 | 37.52 | 37.90 | 13.76 | 51.276 | 36.50 | 51.589 |
| hw/car | 500 | 16.78 | 4.76 | 20.00 | 36.779 | 4.24 | 36.779 |
| hw/car | 1000 | 22.15 | 6.74 | 20.00 | 42.149 | 6.01 | 42.149 |
| hw/car | 2000 | 27.86 | 11.15 | 19.61 | 47.470 | 9.94 | 47.753 |
| hw/car **[serial]** | 3000 | 30.02 | 15.92 | 18.50 | 48.514 | 15.58 | 48.594 |
| hw/car **[serial]** | 5000 | 32.25 | 27.89 | 15.70 | 47.949 | 24.86 | 48.658 |
| hw/car | 8000 | 33.30 | 39.88 | 12.89 | 46.198 | 35.55 | 47.212 |
| hw/trashbin | 500 | 39.27 | 4.35 | 20.00 | 59.271 | 4.08 | 59.271 |
| hw/trashbin | 1000 | 43.47 | 6.77 | 20.00 | 63.474 | 6.35 | 63.474 |
| hw/trashbin | 2000 | 46.80 | 10.45 | 19.78 | 66.581 | 9.81 | 66.732 |
| hw/trashbin **[serial]** | 3000 | 48.23 | 14.34 | 18.87 | 67.097 | 14.21 | 67.128 |
| hw/trashbin **[serial]** | 5000 | 49.11 | 23.06 | 16.83 | 65.942 | 21.64 | 66.275 |
| hw/trashbin | 8000 | 49.85 | 33.27 | 14.44 | 64.289 | 31.22 | 64.768 |
| mipnerf360/bicycle | 500 | 5.70 | 25.90 | 16.16 | 21.866 | 22.24 | 22.722 |
| mipnerf360/bicycle | 1000 | 6.23 | 10.72 | 19.71 | 25.940 | 9.21 | 26.226 |
| mipnerf360/bicycle **[serial]** | 2000 | 6.89 | 12.12 | 19.39 | 26.274 | 11.98 | 26.307 |
| mipnerf360/bicycle | 3000 | 7.51 | 29.08 | 15.42 | 22.929 | 24.97 | 23.890 |
| mipnerf360/bicycle **[serial]** | 5000 | 10.02 | 31.78 | 14.79 | 24.805 | 27.29 | 25.855 |
| mipnerf360/bicycle | 8000 | 12.19 | 66.83 | 6.59 | 18.779 | 57.39 | 20.988 |
| mipnerf360/bonsai | 500 | 24.82 | 7.51 | 20.00 | 44.824 | 7.37 | 44.824 |
| mipnerf360/bonsai | 1000 | 32.64 | 9.01 | 20.00 | 52.636 | 8.84 | 52.636 |
| mipnerf360/bonsai | 2000 | 39.37 | 12.93 | 19.20 | 58.565 | 12.68 | 58.623 |
| mipnerf360/bonsai | 3000 | 44.03 | 17.53 | 18.12 | 62.154 | 17.19 | 62.233 |
| mipnerf360/bonsai **[serial]** | 5000 | 48.44 | 25.55 | 16.25 | 64.685 | 25.06 | 64.799 |
| mipnerf360/bonsai **[serial]** | 8000 | 52.39 | 37.08 | 13.55 | 65.935 | 36.81 | 65.998 |
| mipnerf360/counter | 500 | 18.67 | 12.04 | 19.33 | 37.999 | 11.79 | 38.060 |
| mipnerf360/counter | 1000 | 23.14 | 9.77 | 19.88 | 43.022 | 9.57 | 43.071 |
| mipnerf360/counter | 2000 | 30.92 | 14.06 | 18.85 | 49.769 | 13.77 | 49.840 |
| mipnerf360/counter | 3000 | 35.25 | 18.66 | 17.74 | 52.996 | 18.27 | 53.090 |
| mipnerf360/counter **[serial]** | 5000 | 40.03 | 28.24 | 15.44 | 55.470 | 27.65 | 55.612 |
| mipnerf360/counter **[serial]** | 8000 | 43.36 | 41.17 | 12.34 | 55.697 | 40.50 | 55.858 |

## 3. Per-scene argmax and gain over fixed S=5000

| scene | argmax (conc) | total@argmax | total@5000 | gain | argmax (serial-corrected) | total@argmax* | total@5000* | gain* |
|---|---|---|---|---|---|---|---|---|
| dl3dv/001dccbc1f | 3000 | 80.549 | 80.171 | +0.378 | 3000 | 80.896 | 80.341 | +0.555 |
| dl3dv/0b5635d53b | 5000 | 30.879 | 30.879 | +0.000 | 5000 | 32.024 | 32.024 | +0.000 |
| dl3dv/0f622eaabc | 8000 | 33.362 | 32.187 | +1.175 | 8000 | 34.253 | 32.361 | +1.891 |
| dl3dv/2dad2d2989 | 8000 | 51.376 | 50.620 | +0.757 | 8000 | 51.560 | 50.662 | +0.898 |
| dl3dv/3e4dbe4135 | 5000 | 35.651 | 35.651 | +0.000 | 5000 | 36.023 | 36.023 | +0.000 |
| dl3dv/9a451604b8 | 3000 | 61.201 | 59.747 | +1.454 | 3000 | 61.498 | 60.289 | +1.209 |
| hw/basketball | 3000 | 67.706 | 66.457 | +1.249 | 3000 | 67.833 | 67.138 | +0.695 |
| hw/billboard | 3000 | 54.317 | 53.748 | +0.568 | 3000 | 54.346 | 53.945 | +0.401 |
| hw/car | 3000 | 48.514 | 47.949 | +0.565 | 5000 | 48.658 | 48.658 | +0.000 |
| hw/trashbin | 3000 | 67.097 | 65.942 | +1.155 | 3000 | 67.128 | 66.275 | +0.853 |
| mipnerf360/bicycle | 2000 | 26.274 | 24.805 | +1.469 | 2000 | 26.307 | 25.855 | +0.452 |
| mipnerf360/bonsai | 8000 | 65.935 | 64.685 | +1.250 | 8000 | 65.998 | 64.799 | +1.199 |
| mipnerf360/counter | 8000 | 55.697 | 55.470 | +0.227 | 8000 | 55.858 | 55.612 | +0.246 |

## 4. Serial anchors (single card, other two GPUs idle)

| scene | S | B serial | B concurrent | inflation | q ser | total ser | psnr ser |
|---|---|---|---|---|---|---|---|
| dl3dv/001dccbc1f | 3000 | 17.34 | 18.81 | x1.085 | 62.89 | 81.027 | 34.164 |
| dl3dv/001dccbc1f | 5000 | 27.24 | 27.96 | x1.026 | 64.43 | 80.230 | 34.560 |
| dl3dv/0b5635d53b | 5000 | 30.99 | 35.85 | x1.157 | 17.24 | 32.157 | 23.903 |
| dl3dv/0f622eaabc | 5000 | 32.95 | 33.69 | x1.022 | 17.82 | 32.274 | 22.567 |
| dl3dv/0f622eaabc | 8000 | 51.30 | 55.08 | x1.074 | 24.20 | 34.327 | 23.089 |
| dl3dv/2dad2d2989 | 5000 | 29.46 | 29.64 | x1.006 | 35.23 | 50.510 | 25.349 |
| dl3dv/2dad2d2989 | 8000 | 45.83 | 46.61 | x1.017 | 40.12 | 51.543 | 25.853 |
| dl3dv/3e4dbe4135 | 5000 | 31.54 | 33.12 | x1.050 | 21.32 | 36.107 | 24.358 |
| dl3dv/9a451604b8 | 3000 | 18.19 | 19.45 | x1.069 | 43.53 | 61.462 | 31.273 |
| dl3dv/9a451604b8 | 5000 | 26.07 | 28.37 | x1.088 | 44.34 | 60.413 | 31.639 |
| hw/basketball | 3000 | 13.03 | 13.57 | x1.041 | 48.80 | 67.975 | 26.556 |
| hw/basketball | 5000 | 21.10 | 24.01 | x1.138 | 50.58 | 67.867 | 26.865 |
| hw/billboard | 3000 | 14.60 | 14.73 | x1.009 | 35.31 | 54.272 | 22.669 |
| hw/billboard | 5000 | 22.96 | 23.84 | x1.038 | 36.93 | 54.026 | 22.720 |
| hw/car | 3000 | 15.58 | 15.92 | x1.022 | 30.45 | 49.031 | 22.901 |
| hw/car | 5000 | 24.86 | 27.89 | x1.122 | 32.52 | 48.925 | 23.043 |
| hw/trashbin | 3000 | 14.21 | 14.34 | x1.009 | 48.59 | 67.489 | 27.743 |
| hw/trashbin | 5000 | 21.64 | 23.06 | x1.066 | 48.70 | 65.864 | 27.830 |
| mipnerf360/bicycle | 2000 | 11.98 | 12.12 | x1.012 | 6.85 | 26.272 | 23.565 |
| mipnerf360/bicycle | 5000 | 27.29 | 31.78 | x1.165 | 10.05 | 25.891 | 24.556 |
| mipnerf360/bonsai | 5000 | 25.06 | 25.55 | x1.020 | 48.45 | 64.813 | 29.826 |
| mipnerf360/bonsai | 8000 | 36.81 | 37.08 | x1.007 | 52.23 | 65.843 | 30.644 |
| mipnerf360/counter | 5000 | 27.65 | 28.24 | x1.021 | 40.14 | 55.727 | 27.848 |
| mipnerf360/counter | 8000 | 40.50 | 41.17 | x1.017 | 43.37 | 55.869 | 28.171 |

Run-to-run repeatability (serial - concurrent, same config): dPSNR -0.069..0.411, dSSIM -0.0048..0.0030, dLPIPS -0.0012..0.0007 (n=24).

## 5. B(S) decomposition and per-step rate

| scene | S | load | init | train(record) | snap_train | B | B* | iters/s | gaussians |
|---|---|---|---|---|---|---|---|---|---|
| dl3dv/001dccbc1f | 500 | 6.37 | 1.19 | 6.57 | 2.89 | 10.95 | 10.67 | 76.07 | 529374 |
| dl3dv/001dccbc1f | 1000 | 3.39 | 0.65 | 9.73 | 6.18 | 10.72 | 10.44 | 102.74 | 529252 |
| dl3dv/001dccbc1f | 2000 | 5.08 | 0.67 | 14.32 | 10.49 | 16.74 | 16.31 | 139.68 | 531675 |
| dl3dv/001dccbc1f | 3000 | 2.57 | 0.65 | 18.61 | 15.09 | 18.81 | 17.34 | 161.25 | 539077 |
| dl3dv/001dccbc1f | 5000 | 3.24 | 0.63 | 27.16 | 23.59 | 27.96 | 27.24 | 184.12 | 543288 |
| dl3dv/001dccbc1f | 8000 | 3.29 | 0.63 | 39.32 | 35.84 | 40.25 | 39.21 | 203.45 | 537589 |
| dl3dv/0b5635d53b | 500 | 23.60 | 0.67 | 6.61 | 2.64 | 27.41 | 23.69 | 75.70 | 574610 |
| dl3dv/0b5635d53b | 1000 | 6.07 | 0.67 | 8.77 | 4.88 | 12.12 | 10.48 | 114.08 | 575665 |
| dl3dv/0b5635d53b | 2000 | 4.62 | 0.60 | 13.89 | 9.67 | 15.39 | 13.30 | 143.97 | 626042 |
| dl3dv/0b5635d53b | 3000 | 17.65 | 0.64 | 19.65 | 15.00 | 33.79 | 29.21 | 152.70 | 706384 |
| dl3dv/0b5635d53b | 5000 | 8.38 | 0.64 | 31.48 | 26.32 | 35.85 | 30.99 | 158.84 | 795926 |
| dl3dv/0b5635d53b | 8000 | 26.56 | 0.68 | 48.96 | 43.32 | 71.06 | 61.43 | 163.40 | 865080 |
| dl3dv/0f622eaabc | 500 | 3.72 | 0.70 | 7.59 | 2.74 | 7.66 | 7.49 | 65.86 | 608021 |
| dl3dv/0f622eaabc | 1000 | 4.79 | 1.09 | 10.39 | 6.13 | 12.51 | 12.24 | 96.28 | 612600 |
| dl3dv/0f622eaabc | 2000 | 3.55 | 1.15 | 15.24 | 10.74 | 15.94 | 15.59 | 131.26 | 667381 |
| dl3dv/0f622eaabc | 3000 | 3.33 | 0.77 | 23.21 | 18.13 | 22.73 | 22.23 | 129.26 | 764993 |
| dl3dv/0f622eaabc | 5000 | 3.25 | 0.71 | 35.09 | 29.23 | 33.69 | 32.95 | 142.48 | 882976 |
| dl3dv/0f622eaabc | 8000 | 3.63 | 0.76 | 56.92 | 50.19 | 55.08 | 51.30 | 140.54 | 960580 |
| dl3dv/2dad2d2989 | 500 | 3.25 | 0.65 | 7.00 | 2.55 | 6.96 | 6.92 | 71.45 | 598426 |
| dl3dv/2dad2d2989 | 1000 | 3.35 | 0.67 | 8.78 | 4.67 | 9.19 | 9.13 | 113.90 | 589546 |
| dl3dv/2dad2d2989 | 2000 | 3.31 | 0.68 | 14.08 | 9.31 | 13.79 | 13.71 | 142.01 | 643929 |
| dl3dv/2dad2d2989 | 3000 | 3.31 | 0.64 | 20.34 | 14.70 | 19.15 | 19.03 | 147.46 | 733697 |
| dl3dv/2dad2d2989 | 5000 | 3.10 | 0.63 | 30.94 | 25.41 | 29.64 | 29.46 | 161.63 | 814154 |
| dl3dv/2dad2d2989 | 8000 | 3.84 | 0.63 | 47.59 | 41.64 | 46.61 | 45.83 | 168.10 | 875693 |
| dl3dv/3e4dbe4135 | 500 | 21.88 | 1.35 | 6.96 | 2.91 | 26.64 | 25.37 | 71.85 | 580908 |
| dl3dv/3e4dbe4135 | 1000 | 4.46 | 0.69 | 9.10 | 5.10 | 10.75 | 10.24 | 109.94 | 578702 |
| dl3dv/3e4dbe4135 | 2000 | 3.40 | 0.76 | 16.48 | 12.26 | 16.93 | 16.12 | 121.34 | 611565 |
| dl3dv/3e4dbe4135 | 3000 | 14.09 | 2.09 | 20.71 | 15.92 | 32.60 | 31.04 | 144.87 | 692430 |
| dl3dv/3e4dbe4135 | 5000 | 2.71 | 0.67 | 34.53 | 29.23 | 33.12 | 31.54 | 144.80 | 764858 |
| dl3dv/3e4dbe4135 | 8000 | 25.85 | 0.91 | 48.78 | 43.62 | 70.88 | 67.50 | 164.01 | 799443 |
| dl3dv/9a451604b8 | 500 | 2.34 | 0.66 | 6.78 | 2.79 | 6.29 | 5.78 | 73.79 | 515634 |
| dl3dv/9a451604b8 | 1000 | 3.77 | 0.66 | 8.47 | 5.02 | 9.95 | 9.14 | 118.11 | 515167 |
| dl3dv/9a451604b8 | 2000 | 3.69 | 0.65 | 13.96 | 9.82 | 14.67 | 13.48 | 143.22 | 516619 |
| dl3dv/9a451604b8 | 3000 | 3.56 | 0.61 | 18.13 | 14.79 | 19.45 | 18.19 | 165.46 | 510842 |
| dl3dv/9a451604b8 | 5000 | 5.77 | 0.60 | 24.80 | 21.50 | 28.37 | 26.07 | 201.61 | 487198 |
| dl3dv/9a451604b8 | 8000 | 3.45 | 0.61 | 37.03 | 34.12 | 38.68 | 35.54 | 216.02 | 449926 |
| hw/basketball | 500 | 1.65 | 0.57 | 5.34 | 2.19 | 4.91 | 4.31 | 93.68 | 465449 |
| hw/basketball | 1000 | 1.00 | 0.62 | 8.21 | 5.09 | 7.21 | 6.34 | 121.81 | 461368 |
| hw/basketball | 2000 | 0.94 | 0.51 | 11.73 | 8.53 | 10.48 | 9.21 | 170.42 | 480539 |
| hw/basketball | 3000 | 0.94 | 0.59 | 15.05 | 11.54 | 13.57 | 13.03 | 199.28 | 504024 |
| hw/basketball | 5000 | 2.23 | 1.10 | 23.75 | 20.19 | 24.01 | 21.10 | 210.50 | 528987 |
| hw/basketball | 8000 | 1.09 | 0.52 | 34.59 | 30.92 | 33.03 | 29.03 | 231.29 | 535093 |
| hw/billboard | 500 | 1.59 | 1.79 | 8.42 | 5.14 | 9.02 | 8.69 | 59.37 | 477360 |
| hw/billboard | 1000 | 0.74 | 0.51 | 7.51 | 4.32 | 6.07 | 5.85 | 133.11 | 477348 |
| hw/billboard | 2000 | 0.82 | 0.89 | 12.19 | 8.96 | 11.17 | 10.76 | 164.03 | 505896 |
| hw/billboard | 3000 | 0.77 | 0.52 | 16.56 | 12.94 | 14.73 | 14.60 | 181.16 | 541336 |
| hw/billboard | 5000 | 0.74 | 0.54 | 25.71 | 22.05 | 23.84 | 22.96 | 194.46 | 572979 |
| hw/billboard | 8000 | 0.80 | 0.54 | 40.12 | 36.06 | 37.90 | 36.50 | 199.40 | 600383 |
| hw/car | 500 | 1.00 | 0.60 | 6.57 | 2.66 | 4.76 | 4.24 | 76.14 | 521539 |
| hw/car | 1000 | 0.97 | 0.56 | 8.17 | 4.71 | 6.74 | 6.01 | 122.46 | 511693 |
| hw/car | 2000 | 0.93 | 0.60 | 12.63 | 9.12 | 11.15 | 9.94 | 158.34 | 539805 |
| hw/car | 3000 | 0.85 | 0.60 | 17.97 | 13.96 | 15.92 | 15.58 | 166.91 | 585464 |
| hw/car | 5000 | 2.48 | 1.18 | 28.07 | 23.73 | 27.89 | 24.86 | 178.11 | 642200 |
| hw/car | 8000 | 0.90 | 0.59 | 42.64 | 37.89 | 39.88 | 35.55 | 187.61 | 693335 |
| hw/trashbin | 500 | 0.92 | 0.55 | 5.66 | 2.37 | 4.35 | 4.08 | 88.39 | 490682 |
| hw/trashbin | 1000 | 1.29 | 0.58 | 7.69 | 4.40 | 6.77 | 6.35 | 130.00 | 488930 |
| hw/trashbin | 2000 | 0.88 | 0.62 | 11.73 | 8.45 | 10.45 | 9.81 | 170.44 | 497996 |
| hw/trashbin | 3000 | 0.89 | 0.59 | 15.80 | 12.36 | 14.34 | 14.21 | 189.83 | 510674 |
| hw/trashbin | 5000 | 1.03 | 1.03 | 23.92 | 20.50 | 23.06 | 21.64 | 208.99 | 516183 |
| hw/trashbin | 8000 | 1.00 | 0.61 | 34.53 | 31.16 | 33.27 | 31.22 | 231.68 | 508771 |
| mipnerf360/bicycle | 500 | 19.84 | 2.72 | 7.47 | 2.84 | 25.90 | 22.24 | 66.89 | 665030 |
| mipnerf360/bicycle | 1000 | 2.26 | 0.75 | 12.34 | 7.21 | 10.72 | 9.21 | 81.03 | 671159 |
| mipnerf360/bicycle | 2000 | 2.22 | 0.71 | 13.60 | 8.69 | 12.12 | 11.98 | 147.10 | 721764 |
| mipnerf360/bicycle | 3000 | 14.11 | 1.19 | 19.13 | 13.28 | 29.08 | 24.97 | 156.82 | 835357 |
| mipnerf360/bicycle | 5000 | 5.97 | 1.18 | 30.95 | 24.12 | 31.78 | 27.29 | 161.58 | 1004155 |
| mipnerf360/bicycle | 8000 | 20.64 | 2.08 | 51.93 | 43.62 | 66.83 | 57.39 | 154.06 | 1140667 |
| mipnerf360/bonsai | 500 | 3.67 | 0.74 | 6.77 | 2.59 | 7.51 | 7.37 | 73.81 | 611130 |
| mipnerf360/bonsai | 1000 | 3.17 | 0.77 | 8.67 | 4.57 | 9.01 | 8.84 | 115.29 | 608442 |
| mipnerf360/bonsai | 2000 | 3.10 | 0.78 | 12.78 | 8.55 | 12.93 | 12.68 | 156.45 | 621445 |
| mipnerf360/bonsai | 3000 | 3.01 | 0.78 | 22.23 | 13.24 | 17.53 | 17.19 | 134.97 | 639358 |
| mipnerf360/bonsai | 5000 | 3.65 | 0.71 | 25.16 | 20.69 | 25.55 | 25.06 | 198.77 | 650462 |
| mipnerf360/bonsai | 8000 | 3.45 | 0.75 | 36.76 | 32.38 | 37.08 | 36.81 | 217.61 | 649198 |
| mipnerf360/counter | 500 | 8.02 | 0.72 | 6.74 | 2.80 | 12.04 | 11.79 | 74.20 | 580879 |
| mipnerf360/counter | 1000 | 3.49 | 0.65 | 8.86 | 5.13 | 9.77 | 9.57 | 112.83 | 579847 |
| mipnerf360/counter | 2000 | 2.95 | 0.62 | 14.10 | 9.99 | 14.06 | 13.77 | 141.89 | 589574 |
| mipnerf360/counter | 3000 | 2.93 | 0.71 | 18.57 | 14.52 | 18.66 | 18.27 | 161.60 | 609768 |
| mipnerf360/counter | 5000 | 3.00 | 0.74 | 28.28 | 24.00 | 28.24 | 27.65 | 176.80 | 619753 |
| mipnerf360/counter | 8000 | 2.85 | 0.71 | 41.26 | 37.11 | 41.17 | 40.50 | 193.88 | 615149 |


## 6. Caveats / anomalies (read before quoting section 1)

1. **Concurrent B inflation.** For the 13 anchored S=5000 cells the 3-lane
   concurrent B was only **1.006–1.165x** the serial B (mean **1.071x**), i.e.
   ~7% on average — not the ~45% worst case. But individual cells were polluted
   much more whenever another driver overlapped on N1 (phase-2 was killed at
   23:30:28 and recovery drivers ran at 23:31/23:35/23:37): e.g.
   `dl3dv/0b5635` S=8000 scene_load=26.6 s -> B=71.1 s (serial 5000 anchor implies
   ~61 s corrected), `dl3dv/3e4dbe` S=8000 B=70.9 s, `mip/bicycle` S=8000 B=66.8 s,
   `0b5635` S=3000 B=33.8 s, plus `scene_load` 20–26 s on several cells. **All TIME
   conclusions use the serial anchors**; contaminated concurrent B values are
   still printed raw in section 2 and are marked indirectly via B*(scaled).
2. **S=500 warm-up bias.** All three S=500 cells of phase 1 were the first cell on
   their lane: cold page cache / CUDA warm-up inflated `scene_load`+`init`
   (hw/billboard 1.59+1.79 s vs 0.74+0.51 s at S=1000; mip/counter load 8.02 s;
   dl3dv/001 load 6.37 s). This is why `hw/billboard` S=500 B=9.02 s > S=1000
   B=6.07 s. **Do not read the S=500 column's B as a B(S) point.** (No serial
   S=500 anchors were taken: with clean B the S=500 TIME term is already 20.00 for
   every scene, so B there cannot change any score.)
3. **B definition.** `b_seconds_estimate` = `scene_load + init + snap_train + 0.5`
   (identical to the 13-case reference table), where `snap_train` is measured just
   before the snapshot PLY export. The record's `train_seconds` is 3.2–7.0 s
   higher and the gap scales with Gaussian count (477k -> ~3.2 s, 1.03M -> ~7.0 s)
   = PLY export cost. An organiser harness timing module B end-to-end would see
   roughly `load+init+record_train`, so **our TIME points are slightly optimistic
   vs a strict harness** (order +0.3–0.7 TIME points at S=5000, less at smaller
   S); we keep the reference definition for comparability and do not re-score.
4. **Run-to-run quality noise.** Serial vs concurrent re-runs of the same config
   (n=24): dPSNR mean +0.035, sd 0.094, range [−0.069, +0.411]; dSSIM
   [−0.0048, +0.0030]; dLPIPS [−0.0012, +0.0007]. So per-cell total differences
   below ~0.3–0.5 points are within noise (this covers most of the "gain" list
   except 0f622 / trashbin / 9a451604 / 2dad / bonsai / 001).
5. **`quick_test_psnr` is TEST-camera PSNR**, not a train-view signal
   (`research/train_sweep.py::quick_test_psnr` renders `scene.getTestCameras()`).
   It is recorded per row for comparison only and was **not** used for any
   selection. No train-view quality trace exists in the current `train_sweep.py`;
   the quality-vs-S trajectory of section 2 (independent runs, official metrics)
   is what is available for the marginal-value analysis.
6. **Serial anchors used `--no-quick-test`** (the quick test runs after module-B
   timing and cannot affect B; it only saves wall time). Official metrics are
   still computed for every serial cell, which is why the repeatability check in
   point 4 is possible.

## 7. Reproduction / provenance

- Driver (versioned locally and on N1):
  `/root/gpufree-data/outputs/pareto-20261030/run_sweep_pareto.py`
  (md5 compared with `FFSplat-Hackathon/records/pareto-20261030/run_sweep_pareto.py`).
- Per cell: `research.run_sweep --profile competition/profiles/gpufree-<lane>.json
  --case C --work W --init-ply P --out <root>/<tag>/s<S> --iterations S
  --snapshots "" --budget B --position-lr-max-steps S --opacity-reset-interval 0
  --lambda-dssim 0.6 --scaling-lr 0.01 --densify-from max(20,S//10)
  --densify-until max(40,S//4) --realloc-frac 0.15 --realloc-at max(20,S//4)
  --image-cache W/image_cache` (-b-schedule matched to S; at S=5000 it is exactly
  the measured delivery recipe).
- Resumable: a cell is skipped iff its `summary.json` parses **and** its recorded
  `work` resolves to the manifest's `work`; every cell dir keeps `summary.json`
  and `train_record.json` (train trace fields: `scene_load_seconds`,
  `init_seconds`, `train_seconds`, `total_seconds`, `iterations_per_second`,
  `num_init_gaussians`, `num_final_gaussians`, `quick_test_psnr`, `settings`).
- Provenance note: phase 2 (10 remaining scenes) was killed at 23:30:28 after 48
  cells; the missing 12 cells (all 10 S=8000 + `hw/trashbin`/`mip/bonsai` S=5000)
  were finished by recovery runs at 23:31–23:44. Those are the cells with
  pollution in point 1.

## 8. Artifacts

| path | content |
|---|---|
| N1 `/root/gpufree-data/outputs/pareto-20261030/results.jsonl` | 78 rows, one per (case,S): q, B, t, total + train breakdown + trace fields |
| N1 `.../pareto-20261030/queue.log`, `logs/*.log`, `cells/*.json` | run provenance, per-cell stdout, per-cell meta+cmd |
| N1 `.../pareto-20261030/<tag>/s<S>/{summary.json,train_record.json}` | raw per-cell outputs (kept, 78/78) |
| N1 `/root/gpufree-data/outputs/pareto-20261030-serial/results.jsonl` | 24 serial anchor rows |
| N1 `.../pareto-20261030-serial/serial-anchors.jsonl` (= Mac copy) | raw serial anchors in the Lead's requested schema |
| Mac `FFSplat-Hackathon/records/pareto-20261030/{results.jsonl, serial-results.jsonl, serial-anchors.jsonl, N1-SWEEP-SUMMARY.md}` | versioned copies |
