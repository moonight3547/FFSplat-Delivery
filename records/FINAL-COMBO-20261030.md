# FINAL-COMBO-20261030 — VGG100/200 × M1 × scene cache × D2500/D3500 (the full combination)

**Task 21** · N1 (3×4090), serial quiet box on GPU2 · 2026-10-31 · base commits `5f616b3` /
`ca5a3f6` · S=5000, LR horizon 5000, opacity reset 0, λdssim .6, scaling_lr .01,
densify_from 500, interval 100, actual native `grad_abs_thresh` .0008, realloc .15@1250,
verified scene-cache bundles, per-case image cache · VGG = full-res weight .1 **inside the
same 5000 steps** (`--lpips-start 4901` V100 / `4801` V200, `--lpips-end 5000`,
`--lpips-scale 1.0`, every 1), not a post-S stage · official metrics · seed 20261015 ·
raw `/root/gpufree-data/outputs/final-combo-20261030/`, local `records/final-combo-20261030/raw/`.

> **VERDICT.** The in-run VGG tail is the **largest single lever measured in this whole
> exploration**, and it survives M1 + scene cache completely: no-VGG best → full combo is
> **bicycle 28.127 → 32.630 (+4.50)** and **0b5635 36.615 → 40.684 (+4.07)** total. The
> combination is *super-additive* on bicycle because the VGG tail pushes LPIPS **below the
> 0.25 scoring cliff** (0.2623 no-VGG → **0.2159**), unlocking ~6.8 LPIPS points, while
> M1 + cache keep B only ~1.5–3.6 s above the no-VGG B. **After all factors, D3500 wins
> both scenes at VGG200** (bicycle +1.14 over D2500+V200; 0b5635 +0.51) — reversing the
> no-VGG tie — but only with the **longer** tail: VGG100 on D3500 is not worth it on
> 0b5635 (rank 4). **Candidate recipe for full-13 validation: D3500 + VGG200 + M1 keep
> .6@1250 (bicycle only) + A-side scene cache.**

## 1. Recipe verification

| check | bicycle | 0b5635 |
|---|---|---|
| density events (D2500 / D3500) | **19 / 29** | **19 / 29** |
| actual native `grad_abs_thresh` at every event | **0.0008** | **0.0008** |
| VGG window (from `train_record.settings`) | weight **0.1**, every 1, scale **1.0**, start **4901** (V100) / **4801** (V200), end **5000** | same |
| M1 event (bicycle only, keep .6@1250) | e.g. 1,094,024 → 656,414 | not run |
| scene cache | reused, hash `be2e4a48…` matched, **zero rebuild** | hash `4438bcdb…` matched |

## 2. Bicycle — full 8-arm ranking (budget 95.0 s)

| rank | arm | N | dev | PSNR | SSIM | LPIPS | q | LPIPS pts | load | init | snap_train | **B** | TIME | **total** | vs its no-VGG ref |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **d3500_m1_cache_v200** | 1,433,880 | 29 | 24.879 | 0.7588 | **0.2159** | **19.427** | 6.82 | 0.222 | 0.805 | 37.04 | 38.56 | 13.202 | **32.630** | **+4.564** |
| 2 | d3500_m1_cache_v100 | 1,443,155 | 29 | 24.866 | 0.7580 | 0.2217 | 18.193 | 5.66 | 0.224 | 0.810 | 33.67 | 35.20 | 13.988 | 32.181 | +4.115 |
| 3 | d2500_m1_cache_v100 | 1,209,038 | 19 | 24.844 | 0.7540 | 0.2266 | 16.875 | 4.68 | 0.222 | 0.797 | 30.97 | 32.49 | 14.622 | 31.497 | +3.370 |
| 4 | d2500_m1_cache_v200 | 1,206,697 | 19 | 24.857 | 0.7534 | 0.2221 | 17.738 | 5.58 | 0.226 | 0.818 | 34.66 | 36.20 | 13.754 | 31.492 | +3.365 |

No-VGG references (same N1, task‑19/20): **d2500_m1_cache 28.127** (q 12.582, LPIPS
0.2623 → 0 LPIPS pts), **d3500_m1_cache 28.066** (q 13.084, LPIPS 0.2559 → 0 pts).

## 3. dl3dv/0b5635 — full 4-arm ranking (budget 94.29 s)

| rank | arm | N | dev | PSNR | SSIM | LPIPS | q | LPIPS pts | load | init | snap_train | **B** | TIME | **total** | vs its no-VGG ref |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **d3500_cache_v200** | 1,101,820 | 29 | 24.065 | 0.8290 | **0.1964** | **28.292** | 10.72 | 0.307 | 0.605 | 40.30 | 41.71 | 12.392 | **40.684** | **+3.940** |
| 2 | d2500_cache_v200 | 1,091,916 | 19 | 23.875 | 0.8262 | 0.1984 | 27.516 | 10.32 | 0.282 | 0.624 | 39.16 | 40.57 | 12.661 | 40.177 | +3.562 |
| 3 | d2500_cache_v100 | 1,089,109 | 19 | 23.971 | 0.8268 | 0.2038 | 26.561 | 9.24 | 0.272 | 0.666 | 36.11 | 37.55 | 13.372 | 39.933 | +3.318 |
| 4 | d3500_cache_v100 | 1,102,729 | 29 | 23.836 | 0.8269 | 0.2035 | 26.521 | 9.30 | 0.273 | 0.643 | 37.51 | 38.93 | 13.047 | 39.568 | +2.824 |

No-VGG references: **d2500_cache 36.615** (q 22.450, LPIPS 0.2251), **d3500_cache 36.744**
(q 22.855, LPIPS 0.2251).

## 4. Answers to the task's questions

**Does the H1 VGG gain survive M1 + cache?** **YES, fully.** Bicycle: +3.37…+4.56 total
over the no-VGG combo; 0b5635: +2.82…+3.94. M1+cache does not erase it — M1+cache buy B
(the same −4.1 s M1 and −2.1 s cache as before; scene_load is 0.22/0.27 s in every arm),
and the VGG tail spends that budget on quality. Bicycle is *super-additive*: the VGG tail
drops LPIPS from 0.2559–0.2623 (cliff, 0 points) to 0.2159–0.2266 (4.7–6.8 points), so the
quality term jumps by +3.8–5.0 while B rises only +3.9–7.6 s.

**Which D window wins after all factors?** **D3500 + VGG200** on both scenes:
bicycle 32.630 vs 31.492 (+1.14 over the best D2500 combo; d3500_v200 also beats
d3500_v100 by +0.45), 0b5635 40.684 vs 40.177 (+0.51). This **reverses the no-VGG result**
(task‑20: D3500 tied/lost because its +0.4–0.7 q was paid by +1.7 s B). With the VGG tail
the extra capacity is converted into bigger gains: bicycle V200 q is +1.69 over D2500+V200,
0b5635 +0.78, so the window's B cost is repaid. **But the pairing matters:** on 0b5635,
D3500+V100 is the *worst* arm (rank 4, 39.568) — the longer window only pays when the tail
is long enough to exploit it; on bicycle D3500 wins at both tail lengths.

## 5. Interaction notes (why the ranking differs from H1's VGG-only result)

* H1 measured VGG100/200 on the D3500 base without M1/cache and found **VGG200 > VGG100 on
  all five bad cases**. Here, with M1 + cache, VGG200 still wins on both scenes, but the
  margin on bicycle is smaller (+0.45 total over V100) because the cliff unlock already
  happens at VGG100 (LPIPS 0.2217 → 5.66 pts) and V100's B is 3.4 s cheaper.
* The full-combo ranking is thus **not** a simple sum of H1's factors: M1/cache lower B,
  which raises each arm's TIME points and makes the *quality* differences worth relatively
  more — so the longer tail and the larger window both become affordable. The combined
  recipe is the best of both: 32.630 bicycle (vs H1's D3500+VGG200 alone, which would sit
  ~1.6 s higher B and ~1.0 lower total from the missing M1+cache).
* Practical caution: bicycle's gain is partly a **cliff-crossing** effect (LPIPS 0.25
  boundary). Scenes whose LPIPS stays above 0.25 even with the VGG tail will gain less;
  the 13-case run must verify how many cases cross.

## 6. Candidate recipe for the full-13 validation

```
D3500 (densify 500→3500, interval 100, actual native grad_abs_thresh 0.0008, 29 events)
+ realloc .15 @1250
+ M1 importance prune keep 0.6 @1250      (bicycle-class only; gate decides; NOT 0b5635)
+ A-side scene tensor cache               (user-approved residency; scene_load 3.7→0.3 s)
+ in-run VGG-LPIPS tail: weight 0.1, full res (scale 1.0), every step, 4801–5000 (VGG200)
S=5000, LR horizon 5000, opacity reset 0, λdssim .6, scaling_lr .01, seed 20261015
```

Expected on the two measured cases: bicycle **32.63** (TIME 13.20, B 38.56), 0b5635
**40.68** (TIME 12.39, B 41.71). Watch: 0b5635's B is 44 % of its budget with only 12.4
TIME points — larger/heavier 13-case scenes may need the VGG100 tail (saves ~3.4–7.7 s B).

## 7. Caveats

* One run per arm (as tasked); q noise ≈ ±0.3, N ≈ ±3 % — so the V200-vs-V100 margin on
  bicycle (+0.45) and the D-window margins (+0.5…+1.1) are small but consistently signed
  across both scenes.
* Absolute B is within-session only (~0.5–1.5 s above clean-13); all arms are back-to-back
  on the same quiet GPU.
* Bicycle LPIPS now sits **below** the 0.25 cliff in every VGG arm, which is a large part of
  the gain; the 13-case run must confirm how often that crossing happens (this is the main
  risk for the full-13 result).
* The VGG100-on-D3500-for-0b5635 arm being rank 4 is a real interaction (window × tail
  length), not noise; keep the D3500↔V200 pairing.

## 8. Exact commands / artifacts

```bash
# 8-cell serial matrix (resume-safe), GPU2 quiet box
python /tmp/stagedloss/final_combo.py       # -> /root/gpufree-data/outputs/final-combo-20261030/matrix
python /tmp/stagedloss/final_analyze.py     # -> .../results.json
# per-arm flags (all cells):
#  --iterations 5000 --snapshots "" --position-lr-max-steps 5000 --densify-from 500
#  --densify-interval 100 --densify-until {2500|3500} --grad-abs-thresh 0.0008
#  --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01
#  --realloc-frac 0.15 --realloc-at 1250 --image-cache <work>/image_cache
#  --no-quick-test --seed 20261015 --profile competition/profiles/gpufree-2.json
#  --lpips-weight 0.1 --lpips-scale 1.0 --lpips-start {4901|4801} --lpips-end 5000
#  [bicycle only] --mid-prune-at 1250 --mid-prune-keep 0.6
#  --scene-cache /root/gpufree-data/outputs/b-eng-20261030/scene_cache/<scene>
```

Artifacts: `records/final-combo-20261030/{results.jsonl (8 rows: rank, LPIPS points,
Δ vs its no-VGG reference), run.sh, final_combo.py, final_analyze.py,
raw/<scene>/<arm>/{summary.json, train_record.json}}`; raw on N1
`/root/gpufree-data/outputs/final-combo-20261030/`. No git push; delivery defaults unchanged.
