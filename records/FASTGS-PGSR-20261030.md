# FASTGS-PGSR-20261030 — the free half of PGSR on the FastGS delivery recipe

**Task 12** · N1 (3×4090), serial one GPU · 2026-10-31 · user ask: *"PGSR太慢了，尝试
FastGS+PGSR"* · raw data `/root/gpufree-data/outputs/fastgs-pgsr-20261030/`, local copies
under `records/fastgs-pgsr-20261030/raw/`.

> STATUS: complete (gate + A/B + depth probe). **(a) The normal-alignment half cannot be tested: the ReSplat PLY's
> `nx,ny,nz` fields are all zero** (measured: mean/max normal norm = 0.0, 0 % nonzero).
> **(b) The planar-regularization half is numerically active but is a no-op on this init**
> — the ReSplat Gaussians are *already* flat disks (one scale axis pinned at the PLY
> floor 1e-6 while the other two average 0.093/0.430), so there is no anisotropy left for
> PGSR's planar prior to remove; the gate shows it suppresses the (already negligible)
> axis growth 40× with no visible-quality effect (A/B below). **(c) PGSR's depth half is
> 4–6× cheaper than the first estimate but still not free**: a second gsplat RGB+ED pass
> costs 7.65 ms fwd+bwd vs 4.84 ms for a FastGS 1-cam step, so a 1-in-10-step,
> 1-camera depth loss would add ≈ +16 % B (TIME −0.9) and would need +0.9 total points to
> break even; the depth *channel itself* is free (+0.18 ms) only if a gsplat pass already
> exists. Combined with the metric (photometric NVS, no surface eval), the PGSR direction
> is **closed**.

## 1. What was implemented (defaults off → delivery stays bit-identical)

`research/train_sweep.py` (new flags; defaults 0/off so a normal delivery run is
unchanged; patch in `patch.diff`):

| flag | meaning |
|---|---|
| `--planar_weight W` | adds `W · mean((s_min/s_max.detach())²)` to the loss (PGSR planar flattening; per-Gaussian, no render pass) |
| `--normal_weight W` | adds `W · mean(1 − |cos(axis_min, n_init)|²)`; `axis_min` = world direction of the smallest-scale axis (built from the quaternion), `n_init` = ReSplat PLY normal |
| `--normal_from T` | once N is frozen after the densification window, rebuild the normal table by nearest-init lookup (cKDTree) at iteration T (default 1250) |
| `--shape_stats_every K` | log `N, mean s_min/s_max, anisotropy mean, median scales, mean normal |cos|` every K iterations (forced on when a PGSR weight is set) |

`research/run_sweep.py`: pass-through for the four flags.
`research/depth_cost_probe.py` (new): loads a delivery PLY into gsplat 1.5.3 tensors and
times `render_mode="RGB"` vs `"RGB+ED"` (forward and forward+backward) at the delivery
resolution — this is the number that decides whether PGSR's depth loss could ever fit.

Normals are read once from the init PLY (`nx,ny,nz`) into a side table; the FastGS model
itself is untouched (`competition/upstream/` not modified). Because densification changes
N, the table is exact before the first densification event and is rebuilt once at
`--normal_from` by nearest-init lookup (measured cost 0.50 s for 520 k init → 1.01 M
current, i.e. +0.1 % of B — acceptable).

## 2. Gate: does the prior actually change the Gaussian shape statistics? (bicycle, 1500-step runs, identical recipe except the weight)

Init (ReSplat PLY, 520,980 Gaussians): mean per-axis scale = **1.0e-6 / 0.093 / 0.430**,
i.e. one axis is pinned at the PLY's 1e-6 floor — **the init is already a flat (2D/disk)
Gaussian set**. `anisotropy_mean = mean(s_min/s_max) = 2.5e-5` at iteration 0.

| run | anisotropy @1500 | mean s_min | mean s_max | normal \|cos\| |
|---|---:|---:|---:|---:|
| ctrl | 5.029e-4 | 5.20e-6 | 0.2506 | — |
| planar_0.1 | 1.223e-4 | 2.36e-6 | 0.2521 | — |
| planar_1.0 | **1.216e-5** (40× lower than ctrl) | 7.23e-7 | 0.2518 | — |
| normal_0.1 | 5.047e-4 | 5.11e-6 | 0.2496 | **0.0** (⇒ term constant, zero gradient) |
| normal_1.0 | 4.987e-4 | 5.22e-6 | 0.2505 | **0.0** (⇒ no effect) |

Two findings, both decisive:

1. **`normal_align` is not testable with the data we have**: the ReSplat PLY carries
   `nx,ny,nz` fields but they are **all zero** (mean normal norm 0.0, max 0.0, 0 % of
   520,980 normals nonzero). The alignment term is then the constant 1.0 → zero gradient
   → the model provably ignores it (normal_* runs are statistically identical to ctrl).
   Making it work needs an external normal source (e.g. monocular normal estimation, or
   PGSR's own normal-from-depth) — a new dependency and a new render pass, out of scope
   for the "free subset".
2. **`planar_reg` is active but has nothing to flatten**: it flattens the smallest axis
   from 5.2e-6 to 7.2e-7 (40× suppression of the anisotropy growth) — so the prior *does*
   reach the optimizer — but that axis is already 5 orders of magnitude below the other
   two and below pixel footprint. The photometric objective has no incentive to care, so
   the expected quality effect is ≈0 (A/B in §4). The honest framing: *the ReSplat init
   already satisfies PGSR's planar-representation assumption by construction*; the prior
   is redundant for this pipeline.

## 3. Depth-cost probe (gsplat 1.5.3, bicycle delivery PLY)

`research/depth_cost_probe.py`, 993,861 Gaussians, 576×960 (bicycle prepared res), 3 train
views × 15 reps, after warm-up, `rasterize_mode="classic"`, SH degree 3, centred FoV
projection (same as FastGS):

| render mode | forward (ms) | forward+backward (ms) | channels |
|---|---:|---:|---:|
| RGB | 1.953 | **7.470** | 3 |
| RGB+ED | 1.986 | **7.652** | 4 |
| extra for depth | **+0.033 ms (+1.7 %)** | **+0.182 ms (+2.4 %)** | +1 |

Reference: one FastGS 1-camera training step (forward+backward) = **4.84 ms** on the same
scene (measured in task-8: 500 absgrad steps = 2.42 s).

Interpretation / affordability arithmetic (bicycle, budget 95 s, TIME = 20·(1−B/95)/0.9):

* The depth **channel** is essentially free (+0.18 ms) — but only if a gsplat pass already
  runs; FastGS's own rasterizer cannot return it without touching upstream CUDA
  (forbidden).
* A **second gsplat pass** on the same Gaussians costs a full 7.65 ms fwd+bwd, i.e.
  **+158 % of a FastGS step** if used every step → TIME −7.4 (infeasible).
* Every **10th step** (1 camera): amortised +0.77 ms/step → **+3.8 s B → ΔTIME ≈ −0.9**
  (needs +0.9 quality points to break even). Every 20th step → ΔTIME ≈ −0.45.
* This refines the task's initial estimate (+20–50 % B for depth supervision): the
  affordable operating point is **1-in-10 single-camera depth, +16 % B (−0.9 TIME)**;
  anything denser is −2…−7 TIME. It is not impossible, but it must earn ~1 point of
  photometric quality to pay for itself, on a metric that has no surface term at all.

## 4. A/B on the delivery recipe (serial, quiet box, paired in-session control)

A/B cells: `ctrl`, `pgsr_planar` (W=1.0) run through the delivery recipe with the
unmodified trainer (`records/staged-loss-20261030/run_staged.py` arms `pgsr_planar`,
`pgsr_normal`), official evaluate.py metrics.

| scene | arm | N | B (s) | PSNR | SSIM | LPIPS | q | TIME | total | Δq | ΔB | Δtotal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | ctrl | 995,134 | 26.75 | 24.557 | 0.7288 | 0.2987 | 9.950 | 15.95 | 25.90 | — | — | — |
| bicycle | **planar W=1.0** | 1,000,617 | 28.56 | 24.545 | 0.7282 | 0.2995 | 9.892 | 15.52 | 25.41 | −0.058 | +1.81 | **−0.49** |
| counter | ctrl (same-session, serial ctrl from the task-10 matrix) | 619,595 | 27.15 | 27.833 | 0.9011 | 0.1808 | 40.194 | 15.86 | 56.05 | — | — | — |
| counter | **planar W=1.0** | 619,616 | 28.41 | 27.774 | 0.9006 | 0.1818 | 39.907 | 15.31 | 55.22 | −0.287 | +1.26 | **−0.59** |

The A/B confirms the gate's prediction: quality is flat-to-slightly-negative (bicycle
−0.058 q, inside the ±0.1 run-to-run band; counter −0.287, the only sub-cliff scene) and
the extra per-Gaussian math costs a real **+1.67 s of phase-1 training on bicycle
(stage1 29.90 → 31.57 s = +0.33 ms/step** for two min/max reductions over 1 M Gaussians
plus backward; the task expected ≈0.1 ms/step, the measured cost is 3× that but still
small). ΔB +1.81/+1.26 and Δtotal **−0.49/−0.59** → the planar prior is a strict loss.
The normal arm was not A/B'd because the gate is conclusive (zero normals ⇒ the term is a
constant with zero gradient; `normal_*` runs were statistically identical to ctrl).

Expected outcome and how to read it: the planar prior changes a scale axis that is
already below any rendering effect, so any Δ within ±0.1 quality points is noise; a
*positive* Δ would be surprising and should be re-run before adoption.

## 5. Verdict

1. **`normal_align` (PGSR's normal-consistent planar representation): NOT IMPLEMENTABLE
   from the available data** — the ReSplat PLY normals are all zero. Not adopted.
2. **`planar_reg` (PGSR's planar flattening): implemented, numerically active, but
   redundant and a net loss** — the ReSplat world Gaussians arrive as flat disks (one
   axis pinned at the 1e-6 PLY floor), so the prior flattens an axis already 5 orders
   below the pixel footprint. A/B: bicycle Δq −0.058 / ΔB +1.81 s → **Δtotal −0.49**;
   counter Δq −0.287 / ΔB +1.26 s → **Δtotal −0.59**. Measured cost +0.33 ms/step
   (stage1 29.90 → 31.57 s on bicycle). Not adopted.
3. **PGSR's depth half is closed on cost/benefit**: a single-camera depth loss at 1-in-10
   steps = +16 % B (TIME −0.9) and needs +0.9 photometric points; at full rate it is
   −7.4 TIME. Since the competition metric has no surface/mesh term, the only PGSR
   contribution that could matter (depth-consistent surfaces) is not scored, while its
   cost is scored. Documented with the measured ms above so the direction does not need
   re-litigating.
4. The hybrid is a **clean negative**, but usefully bounded: the "free" half of PGSR is
   free *because it is already satisfied* by the ReSplat initialization, and the expensive
   half costs 16 %+ of B to chase an unscored property.

## 6. Artifacts / exact commands

| path | content |
|---|---|
| `records/fastgs-pgsr-20261030/patch.diff` | train_sweep/run_sweep PGSR flags (defaults off; bit-identical delivery) |
| `records/fastgs-pgsr-20261030/depth_cost_probe.py` | gsplat RGB vs RGB+ED cost probe |
| `records/fastgs-pgsr-20261030/run.sh` | gate + A/B + probe entry point |
| `records/fastgs-pgsr-20261030/results.jsonl` | A/B and gate rows |
| `records/fastgs-pgsr-20261030/raw/` | `depth_probe.json`, `gate_*.json` (train records with shape stats) |
| N1 `/root/gpufree-data/outputs/fastgs-pgsr-20261030/` | raw cells, gate records, probe |

```bash
# gate: shape statistics only (5 × 1500-step runs, bicycle)
bash /tmp/stagedloss/pgsr_gate.sh          # see run.sh

# A/B: delivery recipe ± planar prior, serial, one GPU
python records/staged-loss-20261030/run_staged.py --mode serial --allow-busy \
  --out-root /root/gpufree-data/outputs/fastgs-pgsr-20261030/matrix \
  --cells-file /tmp/pgsr_cells.txt

# depth cost probe
conda run -n fastgs python -m research.depth_cost_probe \
  --work outputs/prep/mipnerf360/bicycle/work --ply <delivery ply> \
  --views 3 --reps 15 --record .../depth_probe.json
```

No git push. `competition/upstream/`, `research/ply_ops.py`, `research/ply_voxgrow.py`
untouched. Defaults of all existing trainer flags unchanged (delivery recipe
bit-identical with the new flags absent).
