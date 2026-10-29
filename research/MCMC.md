# Module-B: explicit wall-clock MCMC (`research/mcmc_budget.py`)

Standalone research trainer for the gsplat MCMC + coarse-to-fine continuation.
It is the time-budgeted counterpart of `competition/mcmc.py`:

* **not** an E2E fraction of the FastGS reference: `--budget-seconds` is the
  module-B wall clock we are willing to spend;
* the held-out views are never read during optimization;
* setup, optimization and PLY export are all charged to the same clock;
* the trainer writes a PLY, `module_b.json`, and a `progress.json` trace so a
  single run exposes its quality/time curve.

## Design

Pipeline inside module B:

1. `competition.bridge.read_training_views(work)` returns the frozen split and
   **train-only** camera poses/intrinsics (`c2w`, `K`). The trainer asserts that
   this list is disjoint from `split["test"]`.
2. The initial Gaussians come from `--init`:
   * `.npz` through `competition.bridge.load_world`;
   * `.ply` through a local reader for the classic 3DGS layout written by
     `run_resplat.py` (channel-major `f_rest`, `rot_0..3` = wxyz, log scales,
     logit opacity). It is round-tripped against
     `competition.bridge.export_ply` in `--self-check`.
3. `competition.bridge.parameter_arrays` recenters the world Gaussians and
   normalizes camera centers to a unit scene extent; cameras are normalized with
   the same `center`/`scale`.
4. `competition.train_utils.make_parameters` builds the six gsplat parameters
   (`means`, `scales`, `quats`, `dc`, `rest`, `opacities`) and
   `make_optimizers` builds one Adam per parameter. `--sh-degree` truncates the
   initial SH coefficients; lower degrees are zero-padded back to SH3 on export.
5. Each step:
   * picks the next training view from a seeded permutation;
   * renders one view with gsplat `rasterization` (`sh_degree`, classic,
     `eps2d=.3`, `packed=False`) at the current resolution stage;
   * loss = `0.8 * L1 + 0.2 * SSIM + 0.01 * mean(opacity) + 0.01 * mean(scale)`
     (the migrated MCMC recipe; no LPIPS term);
   * `loss.backward()`, Adam step;
   * gsplat `MCMCStrategy.step_post_backward(...)` after the first epoch
     (`steps > len(train views)`) and while progress <
     `--topology-stop-fraction` (default 0.8). `cap_max =
     --cap-mult * initial_count`.
6. Resolution stages come from `--res-scales/--res-untils`. Their boundaries
   are fractions of the **time** allowance, not of iteration count, so a fast
   or slow GPU still spends the intended share of seconds at each scale. The
   position LR decays from `1.6e-4` to `1.6e-6` over `0.9` of the allowance,
   matching the frozen `competition/policies/mcmc-ctf.json` recipe.
7. Export uses `competition.bridge.export_ply` at
   `<out>/point_cloud/iteration_<steps>/point_cloud.ply`, in the FastGS/classic
   channel-major layout expected by `research/render_eval.py`.

**Reuse vs local code**

| part | source |
|---|---|
| train split, poses, intrinsics | `competition.bridge.read_training_views` |
| NPZ world, parameter conversion, PLY export | `competition.bridge` |
| parameters, optimizers, SSIM/regularization losses | `competition.train_utils` |
| rasterizer and MCMC strategy | gsplat 1.5.3, same calls as `competition/mcmc.py` |
| PLY initial-state reader | local (bridge only reads NPZ) |
| image decode | local PIL RGB + LANCZOS + uint8 HWC cache |

The last line is deliberate. The organizer's
`competition/upstream/resplat/scripts/infer_colmap.py::load_and_preprocess_images`
is the semantic reference (RGB, LANCZOS, `[0, 1]`), but importing that module
costs about 3.5 s startup time inside a 30 s budget and keeps float copies of
every view. The local loader decodes only the views that are actually visited
and caches uint8 tensors. Everything else (poses, world, export, losses) is
reused.

## Time-budget contract

```
allocation       = --budget-seconds * --budget-fraction        (default fraction 0.9)
export_reserve   = clamp(0.1 * allocation, 1 s, 10 s, <= 0.25 * allocation)
training_allow.  = allocation - setup_seconds - export_reserve
```

The clock starts **before** scene/init loading and stops after `export_ply`
returns, so `module_b.json.total_seconds` covers setup + training + export.
Interpreter/import startup is excluded, matching the timer convention in
`competition/mcmc.py`; if a scorer uses subprocess wall time, account for that
separately.

After every step the trainer keeps `(step_seconds, resolution_factor,
count_before)`. The next-step estimate is the max over the last 128 steps of

```
step_seconds * max(1, (new_factor / old_factor)^2) * max(1, new_count / old_count)
```

multiplied by `--step-safety` (default 1.5). On a step where MCMC relocate/add
will run, the estimate is additionally multiplied by 2 before the stop test,
so a first refinement burst cannot silently overrun the allocation. Before
starting another step:

```
if total_elapsed + predicted_next_step > allocation - export_reserve:
    stop and export
```

`--max-steps` (default 200000) is only a runaway guard; the wall clock is the
normal stop. The PLY export happens after the stop, so its measured time is in
`total_seconds` and in `allocation_met` (also exposed as `budget_met` against
the raw `--budget-seconds`).

`--budget-seconds 30 --budget-fraction 0.9` therefore targets a total of about
27 s, not 30 s; the 10% slack absorbs the final step and export. For the proxy
TIME points, score the PLY with the **per-case FastGS random-init 30K**
reference seconds (99.49 for hw/billboard, 92.57 for mipnerf360/counter in the
current O0 calibration), not with 30/60.

## Outputs

* `point_cloud/iteration_<steps>/point_cloud.ply` — final Gaussians.
* `module_b.json` — `total_seconds`, `setup_seconds`, `train_seconds`,
  `export_seconds`, `steps`, `num_init_gaussians`, `num_final_gaussians`,
  `stop_reason`, `allocation_met`, `budget_met`, `cap`, `init_format`, and
  `schedule` with the actual per-stage steps/seconds/gaussians plus the SH,
  refine, LR and MCMC settings.
* `progress.json` — trace rows `(step, seconds, train_seconds, gaussians,
  scale, resolution, step_seconds, loss)` at step 1, every 100 steps, at every
  resolution change, and at the final step.
* Fresh output directory is enforced: an existing non-empty `--out` is refused.

## Exact GPU commands

From the repository root on the Linux GPU host (Conda env with gsplat 1.5.3;
`[mcmc]` in `competition/profiles/compshare-20261029.json` is
`ffsplat-enhance`):

```bash
export PYTHONPATH=$PWD

# The four round-5 runs (30 s / 60 s on hw/billboard and mipnerf360/counter),
# training then official scoring, logs in outputs/experiments/round5/run.log.
bash research/experiments/round5_mcmc.sh
```

One case by hand:

```bash
/usr/local/miniconda3/bin/conda run --no-capture-output -n ffsplat-enhance \
  python -m research.mcmc_budget \
  --work outputs/O0-20261030/work/hw/billboard \
  --init outputs/O0-20261030/results/hw/billboard/resplat/resplat_world.ply \
  --out outputs/experiments/round5/hw-billboard-b30 \
  --budget-seconds 30 --budget-fraction 0.9 \
  --seed 20261015 --device cuda:0 \
  --res-scales 0.5,0.75,1.0 --res-untils 0.4,0.7 \
  --cap-mult 2.0 --refine-every 100 --sh-degree 3
```

Score that PLY with the exact `score_ply.py` flags (read `steps` and
`total_seconds` from `module_b.json`; `--budget` is the FastGS reference, not
the trainer budget):

```bash
STEPS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["steps"])' \
  outputs/experiments/round5/hw-billboard-b30/module_b.json)
TOTAL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_seconds"])' \
  outputs/experiments/round5/hw-billboard-b30/module_b.json)

/usr/local/miniconda3/bin/conda run --no-capture-output -n ffsplat-enhance \
  python -m research.score_ply \
  --work outputs/O0-20261030/work/hw/billboard \
  --out outputs/experiments/round5/hw-billboard-b30/score \
  --ply "outputs/experiments/round5/hw-billboard-b30/point_cloud/iteration_${STEPS}/point_cloud.ply" \
  --name b30 --budget 99.49 --b-seconds "$TOTAL"
```

`research/experiments/round5_mcmc.sh` does the same four times, using
`--budget 92.57` for counter and the same trainer settings. It reuses a
completed `module_b.json` and an existing `score_summary.json`; partial trainer
directories are skipped rather than overwritten.

CPU-only checks (argument parsing, allocation math, stage boundaries, step
prediction, PLY/NPZ round-trip; no torch/CUDA import):

```bash
python -m research.mcmc_budget --self-check
# e.g. on the Mac control env:
/opt/anaconda3/envs/ffsplat-enhance-macos/bin/python -m research.mcmc_budget --self-check
python -c "import research.mcmc_budget"   # lazy: no CUDA/torch needed
```

## Verification status

* API signatures were checked against the installed gsplat 1.5.3
  (`MCMCStrategy.__init__`, `step_post_backward`, `rasterization`) and mirror
  the known-working `competition/mcmc.py` calls.
* `--self-check` passes locally; the runner was exercised end-to-end with a
  fake `conda` that creates synthetic `module_b.json`/`score_summary.json`.
* **No CUDA training was run.** GPU throughput, actual step-time prediction,
  memory at large `--cap-mult`, and the 30/60 s quality curve are unmeasured.
  Do not treat the time estimates or scores as GPU-validated.
* The PLY reader is tested against `competition.bridge.export_ply`; actual
  ReSplat `resplat_world.ply` files were not available locally, but the
  `run_resplat.py` writer uses the same channel-major/wxyz layout.
* The image loader matches the organizer's decode semantics but is a local
  five-line implementation (see the reuse note); it was not compared
  numerically against the upstream helper on real images.
