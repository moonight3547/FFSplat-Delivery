# FRAMEWORK-OPT-20261030 — framework optimization on N1 (GT-LPIPS feature cache, official CUDA Adam)

**Task 23** · N1 (3×4090), one quiet GPU, serial · 2026-10-31 · current best recipe:
**D3500 (500→3500, native .0008, 29 events) + full-res VGG-LPIPS tail w=.1 at 4801–5000 +
A-side scene tensor cache + deterministic M1 gate (keep .6 on bicycle-class)** · official
metrics · all changes **default off**, no run_pipeline integration · raw
`/root/gpufree-data/outputs/framework-opt-20261030/`, local `records/framework-opt-20261030/raw/`.

> **VERDICT.**
> 1. **GT VGG-LPIPS feature cache: exact, but REJECTED.** The cached path is **bit-identical**
>    to the live LPIPS loss (max |Δ| = 0.0 on real GT views) and removes **9.62 ms/step** of
>    GT-forward work (live 20.40 ms → cached 10.77 ms per LPIPS call, −47 %), but the
>    tail's sampler draws **without replacement per epoch**, so a 200-step tail touches
>    ~169 distinct GT views (bicycle) / 200 of 333 (0b5635): measured **hits 9/200** and
>    **0/200**. Exact features cost **257–270 MB per view** (VGG16's first layer is
>    full-resolution), so a 40-view LRU alone is **9.6–10.8 GB VRAM**; the missed builds and
>    eviction overhead *add* B: **+4.76 s / +4.41 s**, total **−1.17 / −0.86**. The structural
>    ceiling is 31 hits (15.5 %) on bicycle and 0 on 0b5635. **Do not adopt.**
> 2. **Official Faster-GS CUDA Adam: CONFIRMED under the D3500 recipe.** Built from official
>    source (`d23bb784`), numeric/state smoke pass, and two new paired cells today:
>    **bicycle B −2.57 s (−6.67 %) → Δtotal +0.695**, **0b5635 B −2.70 s (−6.43 %) → Δtotal
>    +0.675**, quality within noise. Combined with the 3 D4000 compatibility cells from
>    task-22 (B −5.2…−7.0 %), this is the strongest remaining framework win — keep it
>    **opt-in** and adopt pending paired repeats on more cases.
> 3. Priority 3 (init-PLY tensor cache) skipped as allowed: the GT-cache investigation used
>    the remaining window; init is only 0.6–0.9 s of B.

## 1. Priority 1 — GT VGG-LPIPS feature cache (measured)

Implementation: `research/gt_feature_cache.py` (`GTCachedLPIPS`, subclasses the pinned
`lpips.LPIPS`, caches the **already scaled/normalized GT feature maps** per camera, then
evaluates only the render branch with the package's exact math). Opt-in:
`--gt_feature_cache [--gt_feature_cache_views K]`; `--lpips_timing` records per-step LPIPS GPU
time via CUDA events. Weight/start/end/scale and the 200 updates are unchanged; **the render
forward and the whole LPIPS backward remain** — only the GT forward is removed on a hit.

**Identity proof (3 real bicycle GT views, miss and hit paths):**

| item | value |
|---|---|
| cached vs live LPIPS loss, max abs diff | **0.0** (all three views, miss and hit identical to live to the bit) |
| example losses (view 0/40/120) | live = miss = hit = 0.0599194616 / 0.1877640784 / 0.0975679681 |
| VGG16 weights sha256 | `a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868` |
| lpips package sha256 (`lpips.py`) | `780d09b907cb9b661e0ae28b2d163ddfba92f9e870d7feba34d4790cc6590658` |
| per-call time | live **20.398 ms** · cached hit **10.775 ms** · cached miss 10.796 ms |
| saving per hit | **9.62 ms** (47 % of the LPIPS call) |
| feature size | **269,844,480 B/view (257 MB)** bicycle 576×960; ~240 MB/view 0b5635 512×960 |
| build cost | ~10 ms/view measured in-tail (0.029 s for 3 views standalone) |

**A/B with the full recipe (D3500 + VGG200 + cache + M1; same-day same-node controls):**

| scene | arm | q | B (s) | TIME | total | lpips tail GPU (200 calls) | hits | cache VRAM | peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | control | 19.538 | 38.53 | 13.209 | **32.748** | 4.243 s | — | — | 5.06 GB |
| bicycle | GT cache (K=40) | 19.486 | **43.29** | 12.096 | **31.582 (−1.17)** | 8.893 s | **9/200** | 10.79 GB | 14.79 GB |
| 0b5635 | control | 28.054 | 41.99 | 12.322 | **40.380** | 3.445 s | — | — | 5.31 GB |
| 0b5635 | GT cache (K=40) | 28.236 | **46.40** | 11.285 | **39.522 (−0.86)** | 7.800 s | **0/200** | 9.59 GB | 13.97 GB |

**Why it cannot pay (the important part):** the tail camera sampler pops from a copy of the
train-camera list **without replacement**, so within one epoch each GT view is visited at most
once; the 200-step tail is ~1.2 epochs on bicycle (169 views) and *less than one epoch* on
0b5635 (333 views). Therefore the **maximum possible hit count is 200 − distinct views = 31
(15.5 %) on bicycle and 0 on 0b5635**, and even perfect caching would save only
31 × 9.62 ms ≈ **0.30 s** there. Holding a whole bicycle epoch to serve the repeats would need
169 × 257 MB = **43 GB VRAM** (impossible on 24 GB). Alternatives considered and rejected:
fp16 feature cache (breaks the bit-identity requirement), CPU-side cache (270 MB H2D per step
≈ 20 ms > 9.62 ms saved), A-side precompute (43 GB artifact), changing the sampler
(a recipe change, out of scope).

## 2. Priority 2 — official Faster-GS CUDA Adam (confirmed)

Deployment (from task-22, reproduced here): official backend source, official main
`d23bb784`; no prebuilt `_C`, so it was compiled on N1 against torch **2.4.0+cu124** /
nvcc 12.8 / sm_89 in ~2.5 min. Identity and smoke:

| item | value |
|---|---|
| extension | `FasterGSCudaBackend/_C.cpython-312-x86_64-linux-gnu.so`, **6,007,648 B**, sha256 `c7b176953fd4f8eec51dca21114635fb4a82f77dd3340006d7f04bef7586ebd6` |
| class | `FasterGSCudaBackend.torch_bindings.adam.FusedAdam` (subclass of `torch.optim.Adam`; `adam.py` sha256 `9ea0b472…`; calls `_C.adam_step`) |
| numeric/state smoke | 35 steps vs `torch.optim.Adam` incl. state expansion at step 12 and a zero-grad step: max |Δparam| **7.15e-7**, |Δm| 1.79e-7, |Δv| 5.91e-7 → **PASS** (Codex N2 evidence 1.19e-7/5.96e-8/7.45e-9) |
| opt-in | `FFSPLAT_OFFICIAL_ADAM=1` env hook in `research/train_sweep.py`; delivery path untouched |

**D3500 + VGG200 + scene cache + M1 paired cells (today, same node, same recipe):**

| case | backend | q | B (s) | TIME | total | Δq | ΔB | Δtotal |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mipnerf360/bicycle | control | 19.538 | 38.53 | 13.209 | 32.748 | — | — | — |
| mipnerf360/bicycle | **official FusedAdam** | 19.631 | **35.96** | 13.811 | **33.443** | +0.093 | **−2.57 (−6.67 %)** | **+0.695** |
| dl3dv/0b5635 | control | 28.054 | 41.99 | 12.322 | 40.380 | — | — | — |
| dl3dv/0b5635 | **official FusedAdam** | 28.093 | **39.29** | 12.962 | **41.055** | +0.039 | **−2.70 (−6.43 %)** | **+0.675** |

Together with the task-22 D4000 compatibility cells (bicycle −6.96 %/+0.735; 0b5635 −6.2 %/+0.625;
3e4dbe −5.2 %/+0.646), the B win is **consistent at −5.2…−7.0 %** with quality deltas inside
run noise. **Recommendation: keep as a separate opt-in flag and adopt after paired repeats**
(it is not part of the delivery default).

## 3. Priority 3 — init Gaussian tensor cache (skipped)

Not attempted (optional per task). Known size: init is 0.6–0.9 s of B (PLY parse + upload +
optimizer state); an A-side tensor cache could remove ~0.4–0.6 s of it, identity-verifiable
the same way as the scene cache. Leave for a future task.

## 4. Prioritized recommendation table

| # | item | class | measured saving | quality cost | memory cost | recommendation |
|---|---|---|---|---|---|---|
| 1 | **Official Faster-GS CUDA Adam** | A (kernel replacement) | **B −2.57/−2.70 s (−6.7/−6.4 %)** on D3500; −5.2…−7.0 % across 5 cells | ≤ +0.1 q (noise) | none | **adopt after paired repeats; keep opt-in now** |
| 2 | scene tensor cache | A residency (user-approved) | B −2.1…−3.4 s | identical | 0.28–0.49 GB bundle + transient | adopted (shipped) |
| 3 | **GT VGG-LPIPS feature cache** | A (pure elimination on paper) | 9.62 ms/hit but hits ≤15.5 % ceiling; measured net **B +4.4…+4.8 s** | identical (0.0) | **9.6–10.8 GB VRAM** for K=40 | **REJECT** |
| 4 | init PLY tensor cache (future) | A residency | ~0.4–0.6 s init | identical | ~0.1 GB | optional future |
| 5 | per-step micro-eliminations (task-18) | A | ≤0.05 s | identical | 0 | negligible (already default-off flags) |

## 5. Caveats

* GT-cache A/B is single-run per arm; the effect is dominated by structural misses, not noise
  (hits 9/200 and 0/200 are recorded counters, and B rises by >4 s in both scenes).
* The identity test used 3 real bicycle GT views and random renders; the wrapper is
  operation-for-operation the same math as the pinned package, and all 200 tail losses use it.
* Adam compatibility cells are single-run; the −6.4…−6.7 % B effect is 2–3× the B noise
  (±0.5–1.5 s), but quality needs paired repeats (and more cases) before adoption.
* All paths default off; delivery remains bit-identical when flags are absent.

## 6. Exact commands / artifacts

```bash
# GT feature cache
python /tmp/stagedloss/gtcache_identity.py   # identity + timing + weight hashes
python /tmp/stagedloss/gtcache_ab.py         # 4 cells: {bicycle,0b5635} x {ctrl, gtcache K=40}
# official CUDA Adam under the D3500 recipe
python /tmp/stagedloss/gtcache_adam_d3500.py  # env FFSPLAT_OFFICIAL_ADAM=1
# per-cell recipe: D3500 + VGG200 full-res 4801-5000 + scene cache + M1 gate (0.6 bicycle),
#   --lpips-timing, official metrics via research.run_sweep
```

Artifacts: `records/framework-opt-20261030/{gtcache_identity.json, gtcache_ab/{bicycle,0b5635}/
{ctrl,gtcache}/{summary.json,train_record.json}, adam_d3500/{bicycle,0b5635}/{summary.json,
train_record.json, smoke.json}, run.sh, scripts/}`; N1 official-Adam identity/smoke under
`/root/gpufree-data/outputs/d4000-13-20261030/` (task-22) and
`/root/gpufree-data/outputs/framework-opt-20261030/`. No git push; defaults unchanged.
