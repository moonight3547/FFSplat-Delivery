"""Standalone, explicitly wall-clock-budgeted gsplat MCMC module-B trainer.

This is the research counterpart of ``competition/mcmc.py``.  It keeps the
same gsplat MCMC + coarse-to-fine recipe but replaces the E2E-derived time
fraction with an explicit module-B contract:

    allocation = --budget-seconds * --budget-fraction
    the allocation covers setup + optimization + PLY export
    training stops when elapsed + predicted_next_step > allocation - export_reserve

Everything after the argument parse is module-B time (scene/init loading,
parameter construction, training, export).  Interpreter/import startup is not
charged, matching the timer convention in ``competition/mcmc.py``.

Reuse (no re-derivation):
  * ``competition.bridge.read_training_views`` for train-only poses/intrinsics;
  * ``competition.bridge.load_world`` / ``parameter_arrays`` / ``export_ply``;
  * ``competition.train_utils`` for parameters, optimizers and losses;
  * gsplat ``rasterization`` + ``MCMCStrategy`` exactly as in
    ``competition/mcmc.py`` (checked against gsplat 1.5.3).

The held-out split is never read for optimization.  Image decoding is the
same PIL RGB + LANCZOS + [0,1] convention as the organizer's
``load_and_preprocess_images``; it is kept local because importing that
ReSplat module costs ~3.5 s inside a 30 s budget, and the local cache stores
uint8 HWC tensors instead of three float copies of every view.

Examples (from the repository root, CUDA host):

  python -m research.mcmc_budget --work outputs/O0-20261030/work/hw/billboard \
      --init outputs/O0-20261030/results/hw/billboard/resplat/resplat_world.ply \
      --out outputs/experiments/round5/hw-billboard-b30 \
      --budget-seconds 30 --budget-fraction 0.9 \
      --res-scales 0.5,0.75,1.0 --res-untils 0.4,0.7 --cap-mult 2.0

  python -m research.mcmc_budget --self-check     # CPU only, no CUDA/torch

See ``research/MCMC.md`` for the design and the exact GPU commands.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Recipe constants copied from the known-working competition/policies/mcmc-ctf.json.
POSITION_LR_INITIAL = 1.6e-4
POSITION_LR_FINAL = 1.6e-6
LR_END_FRACTION = 0.9
ADAM_LRS = dict(scales=0.005, quats=0.001, dc=0.0025, rest=0.000125, opacities=0.05)
SSIM_WEIGHT = 0.2
OPACITY_REGULARIZATION = 0.01
SCALE_REGULARIZATION = 0.01
NOISE_LR = 500000.0
MIN_OPACITY = 0.005
EXPORT_RESERVE_SECONDS_MAX = 10.0
REFINE_STEP_PENALTY = 2.0


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class Budget:
    """Module-B wall-clock contract, including setup and export."""

    seconds: float
    fraction: float = 0.9
    export_reserve_fraction: float = 0.1
    allocation: float = field(init=False)
    reserve: float = field(init=False)

    def __post_init__(self):
        seconds = _finite(self.seconds, "budget_seconds")
        fraction = _finite(self.fraction, "budget_fraction")
        reserve_fraction = _finite(self.export_reserve_fraction, "export_reserve_fraction")
        if seconds <= 0:
            raise ValueError("budget_seconds must be positive")
        if not 0 < fraction <= 1:
            raise ValueError("budget_fraction must be in (0, 1]")
        if not 0 < reserve_fraction < 1:
            raise ValueError("export_reserve_fraction must be in (0, 1)")
        allocation = seconds * fraction
        reserve = min(EXPORT_RESERVE_SECONDS_MAX, max(1.0, allocation * reserve_fraction))
        # Never reserve more than a quarter of a very small allocation.
        reserve = min(reserve, 0.25 * allocation)
        object.__setattr__(self, "seconds", seconds)
        object.__setattr__(self, "fraction", fraction)
        object.__setattr__(self, "export_reserve_fraction", reserve_fraction)
        object.__setattr__(self, "allocation", allocation)
        object.__setattr__(self, "reserve", reserve)

    def allowance(self, setup_seconds):
        setup = _finite(setup_seconds, "setup_seconds")
        if setup < 0:
            raise ValueError("setup_seconds must be non-negative")
        return self.allocation - self.reserve - setup

    def stop(self, total_elapsed, predicted_next_step):
        """True when one more estimated step would not fit before export."""
        elapsed = _finite(total_elapsed, "total_elapsed")
        predicted = _finite(predicted_next_step, "predicted_next_step")
        return elapsed + predicted > self.allocation - self.reserve

    def progress(self, total_elapsed, setup_seconds):
        """Fraction of the post-setup optimization allowance already used."""
        allowance = self.allowance(setup_seconds)
        if allowance <= 0:
            return 1.0
        elapsed = _finite(total_elapsed, "total_elapsed")
        return min(1.0, max(0.0, (elapsed - setup_seconds) / allowance))

    def met(self, total_seconds):
        return _finite(total_seconds, "total_seconds") <= self.allocation + 1e-9


def predict_next_step(observations, factor, count, safety):
    """Conservative prediction from observed steps, scaled by res/count jumps.

    ``observations`` holds ``(step_seconds, resolution_factor, count_before)``
    tuples.  A resolution increase costs ~factor^2 and more Gaussians cost
    ~count, so every past observation is scaled before taking the max.
    """
    factor = _finite(factor, "factor")
    count = _finite(count, "count")
    safety = _finite(safety, "step_safety")
    if factor <= 0 or count < 1 or safety < 1:
        raise ValueError("Invalid step-prediction inputs")
    if not observations:
        return 0.0
    predicted = 0.0
    for seconds, old_factor, old_count in observations:
        seconds = _finite(seconds, "observed step_seconds")
        old_factor = _finite(old_factor, "observed factor")
        old_count = _finite(old_count, "observed count")
        if seconds <= 0 or old_factor <= 0 or old_count < 1:
            raise ValueError("Invalid observed step")
        scaled = seconds * max(1.0, (factor / old_factor) ** 2) * max(1.0, count / old_count)
        predicted = max(predicted, scaled)
    return predicted * safety


def _parse_csv(text, name):
    if text is None:
        return []
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    values = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as error:
            raise ValueError(f"{name} must be comma-separated numbers") from error
        _finite(value, name)
        values.append(value)
    return values


def parse_schedule(scales_text, untils_text):
    """Parse and validate the coarse-to-fine schedule.

    ``--res-scales 0.5,0.75,1.0 --res-untils 0.4,0.7`` means: use 0.5 until
    40% of the optimization allowance, 0.75 until 70%, then full resolution.
    """
    scales = _parse_csv(scales_text, "res_scales") or [1.0]
    untils = _parse_csv(untils_text, "res_untils")
    if any(not 0 < scale <= 1 for scale in scales):
        raise ValueError("res_scales values must be in (0, 1]")
    if len(untils) != len(scales) - 1:
        raise ValueError("res_untils needs exactly len(res_scales)-1 cuts")
    if any(not 0 < cut < 1 for cut in untils):
        raise ValueError("res_untils values must be in (0, 1)")
    if any(b <= a for a, b in zip(scales, scales[1:])):
        raise ValueError("res_scales must increase from coarse to full resolution")
    if any(b <= a for a, b in zip(untils, untils[1:])):
        raise ValueError("res_untils must increase")
    if not math.isclose(scales[-1], 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("the last res_scales entry must be 1.0")
    return scales, untils


def stage_scale(progress, scales, untils):
    if not math.isfinite(progress):
        raise ValueError("progress must be finite")
    progress = min(1.0, max(0.0, progress))
    for cut, scale in zip(untils, scales[:-1]):
        if progress < cut:
            return scale
    return scales[-1]


def position_lr(progress):
    progress = min(1.0, max(0.0, progress))
    return math.exp(math.log(POSITION_LR_INITIAL)
                    + min(1.0, progress / LR_END_FRACTION)
                    * math.log(POSITION_LR_FINAL / POSITION_LR_INITIAL))


def image_dims(height, width, factor):
    if not (height >= 8 and width >= 8) or not 0 < factor <= 1:
        raise ValueError("Invalid image dimensions/factor")
    return max(8, round(height * factor)), max(8, round(width * factor))


def pad_rest_for_export(rest):
    rest = np.asarray(rest)
    if rest.ndim != 3 or rest.shape[2] != 3 or rest.shape[1] > 15:
        raise ValueError("rest must have shape [N, <=15, 3]")
    if rest.shape[1] == 15:
        return rest
    padding = np.zeros((len(rest), 15 - rest.shape[1], 3), dtype=rest.dtype)
    return np.concatenate((rest, padding), axis=1)


def _world_from_ply(path):
    """Read the classic 3DGS layout written by run_resplat.py/export_ply."""
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or ())
    needed = {"x", "y", "z", "opacity"} | {f"scale_{i}" for i in range(3)}
    needed |= {f"rot_{i}" for i in range(4)} | {f"f_dc_{i}" for i in range(3)}
    if not needed <= names:
        raise ValueError(f"PLY is missing classic 3DGS properties: {sorted(needed - names)}")

    def column(name):
        return np.asarray(vertex[name], dtype=np.float64)

    means = np.column_stack([column(k) for k in ("x", "y", "z")])
    n = len(means)
    if n == 0:
        raise ValueError("PLY contains no Gaussians")
    log_scales = np.column_stack([column(f"scale_{i}") for i in range(3)])
    scales = np.exp(log_scales)
    quats_wxyz = np.column_stack([column(f"rot_{i}") for i in range(4)])
    norms = np.linalg.norm(quats_wxyz, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("PLY contains zero-length rotations")
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]] / norms[:, None]
    dc = np.column_stack([column(f"f_dc_{i}") for i in range(3)])
    opacities = 1.0 / (1.0 + np.exp(-column("opacity")))

    rest_fields = []
    for name in names:
        if name.startswith("f_rest_"):
            try:
                index = int(name[len("f_rest_"):])
            except ValueError:
                continue
            rest_fields.append((index, name))
    rest_fields.sort()
    if rest_fields:
        if [index for index, _ in rest_fields] != list(range(len(rest_fields))):
            raise ValueError("PLY f_rest fields are not contiguous from zero")
        if len(rest_fields) % 3:
            raise ValueError("PLY f_rest field count is not divisible by three")
        per_channel = len(rest_fields) // 3
        if per_channel + 1 not in (1, 4, 9, 16):
            raise ValueError("Only SH degree 0..3 PLYs are supported")
        rest = np.stack([column(name) for _, name in rest_fields], axis=1)
        rest = rest.reshape(n, 3, per_channel)
        harmonics = np.concatenate((dc[:, :, None], rest), axis=2)
    else:
        harmonics = dc[:, :, None]

    arrays = dict(means=means, scales=scales, rotations_xyzw=quats_xyzw,
                  harmonics=harmonics, opacities=opacities)
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("PLY contains nonfinite Gaussian values")
    if np.any(scales <= 0):
        raise ValueError("PLY contains nonpositive scales")
    if np.any((opacities < 0) | (opacities > 1)):
        raise ValueError("PLY contains invalid opacities")
    return {key: np.ascontiguousarray(value, dtype=np.float32) for key, value in arrays.items()}


def read_world(path):
    """Load ``resplat_world.npz`` via bridge, or parse ``resplat_world.ply``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        from competition.bridge import load_world
        return load_world(path)
    if suffix == ".ply":
        return _world_from_ply(path)
    raise ValueError("--init must be a .npz or .ply Gaussian file")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Standalone wall-clock-budgeted gsplat MCMC module-B trainer (research).")
    parser.add_argument("--work", type=Path, help="prepared scene dir (module-A output; required)")
    parser.add_argument("--init", type=Path, help="ReSplat world Gaussians (.ply or .npz; required)")
    parser.add_argument("--out", type=Path, help="fresh module-B output dir (required)")
    parser.add_argument("--budget-seconds", type=float, default=None,
                        help="module-B wall-clock budget (required; not the FastGS 30K reference)")
    parser.add_argument("--budget-fraction", type=float, default=0.9,
                        help="usable fraction of --budget-seconds (default: 0.9)")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--res-scales", default="1.0",
                        help="comma-separated coarse-to-fine scales, e.g. 0.5,0.75,1.0")
    parser.add_argument("--res-untils", default="",
                        help="one progress cut per resolution transition, e.g. 0.4,0.7")
    parser.add_argument("--cap-mult", type=float, default=2.0,
                        help="MCMC cap = cap_mult * initial Gaussian count")
    parser.add_argument("--refine-every", type=int, default=100)
    parser.add_argument("--sh-degree", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--export-reserve-fraction", type=float, default=0.1,
                        help="fraction of the allocation reserved for PLY export")
    parser.add_argument("--step-safety", type=float, default=1.5,
                        help="multiplier on the observed step-time prediction")
    parser.add_argument("--topology-stop-fraction", type=float, default=0.8,
                        help="stop MCMC relocate/add after this fraction of the allowance")
    parser.add_argument("--snap-seconds", default="",
                        help="comma list of train-seconds at which to dump a PLY (curve)")
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--self-check", action="store_true",
                        help="run CPU-only checks of parsing/allocation math/PLY round-trip")
    return parser


def validate_args(args):
    if args.work is None or args.init is None or args.out is None:
        raise ValueError("--work, --init and --out are required")
    if args.budget_seconds is None:
        raise ValueError("--budget-seconds is required")
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError("--seed must be a non-negative integer")
    if type(args.refine_every) is not int or args.refine_every < 1:
        raise ValueError("--refine-every must be a positive integer")
    if type(args.max_steps) is not int or args.max_steps < 1:
        raise ValueError("--max-steps must be a positive integer")
    if not math.isfinite(args.cap_mult) or args.cap_mult < 1:
        raise ValueError("--cap-mult must be finite and >= 1")
    if not math.isfinite(args.step_safety) or args.step_safety < 1:
        raise ValueError("--step-safety must be finite and >= 1")
    if not math.isfinite(args.topology_stop_fraction) or not 0 < args.topology_stop_fraction <= 1:
        raise ValueError("--topology-stop-fraction must be in (0, 1]")
    if not isinstance(args.device, str) or not args.device:
        raise ValueError("--device must be a non-empty torch device string")
    scales, untils = parse_schedule(args.res_scales, args.res_untils)
    args.res_scales, args.res_untils = scales, untils
    args.budget = Budget(args.budget_seconds, args.budget_fraction, args.export_reserve_fraction)
    return args


def _run(args):
    import torch
    from gsplat.rendering import rasterization
    from gsplat.strategy import MCMCStrategy
    from competition.bridge import export_ply, parameter_arrays, read_training_views
    from competition.contracts import write_json
    from competition.train_utils import (
        make_optimizers, make_parameters, regularization_loss, ssim_loss)

    if not args.work.is_dir():
        raise FileNotFoundError(f"work dir not found: {args.work}")
    if not args.init.is_file():
        raise FileNotFoundError(f"init Gaussians not found: {args.init}")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"choose a fresh output dir: {out}")

    if not torch.cuda.is_available():
        raise RuntimeError("gsplat training needs CUDA; --self-check is the CPU-only path")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device for gsplat rasterization")
    torch.cuda.set_device(device)
    out.mkdir(parents=True, exist_ok=True)

    import importlib.metadata
    try:
        gsplat_version = importlib.metadata.version("gsplat")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - environment issue
        gsplat_version = "unknown"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.cuda.synchronize()
    start = time.perf_counter()

    # ---- setup: train-only views, world Gaussians, gsplat parameters ----
    split, views = read_training_views(args.work)
    if set(view["name"] for view in views) & set(split["test"]):
        raise ValueError("Held-out view leaked into the training view list")
    world = read_world(args.init)
    needed_sh = (args.sh_degree + 1) ** 2
    if world["harmonics"].shape[-1] < needed_sh:
        raise ValueError(f"init has SH{world['harmonics'].shape[-1]} but --sh-degree "
                         f"{args.sh_degree} needs {needed_sh} coefficients")
    if world["harmonics"].shape[-1] > needed_sh:
        world = dict(world, harmonics=world["harmonics"][:, :, :needed_sh])

    poses = np.stack([view["c2w"] for view in views]).astype(np.float32)
    center = poses[:, :3, 3].mean(0)
    extent = float(np.max(np.abs(poses[:, :3, 3] - center)))
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("Nondegenerate training cameras required")
    scene_scale = 1.0 / extent
    params = make_parameters(parameter_arrays(world, center, scene_scale), device)
    initial_count = int(params["means"].shape[0])
    cap = max(initial_count, int(round(args.cap_mult * initial_count)))
    cap_mult_actual = cap / initial_count

    poses[:, :3, 3] = (poses[:, :3, 3] - center) * scene_scale
    viewmats = torch.linalg.inv(torch.from_numpy(poses).to(device))

    height, width = split["resolution"]
    base_k = np.asarray(views[0]["K"], dtype=np.float32)
    intrinsics, dims = {}, {}
    for factor in sorted(set(args.res_scales)):
        sh, sw = image_dims(height, width, factor)
        k = base_k.copy()
        k[0] *= sw / width
        k[1] *= sh / height
        # The organizer FastGS camera uses a FoV projection, i.e. a centered
        # principal point; match it instead of COLMAP's cx/cy (as competition/mcmc.py).
        k[0, 2], k[1, 2] = sw / 2, sh / 2
        intrinsics[factor] = torch.from_numpy(np.ascontiguousarray(k)).to(device)
        dims[factor] = (sh, sw)

    opts = make_optimizers(params, dict(adam_lrs=ADAM_LRS, position_lr_initial=POSITION_LR_INITIAL))
    strategy = MCMCStrategy(
        cap_max=cap, min_opacity=MIN_OPACITY, noise_lr=NOISE_LR,
        refine_start_iter=0, refine_stop_iter=args.max_steps + 1,
        refine_every=args.refine_every)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state()
    torch.cuda.synchronize()
    setup_s = time.perf_counter() - start

    allowance = args.budget.allowance(setup_s)
    if allowance <= 0:
        raise RuntimeError(
            f"setup used {setup_s:.3f}s of allocation {args.budget.allocation:.3f}s "
            f"(export reserve {args.budget.reserve:.3f}s); no optimization time left")

    # ---- host image cache: uint8 HWC, decoded on first visit ----
    image_cache = {}

    def image_for(index, factor):
        key = (index, factor)
        cached = image_cache.get(key)
        if cached is None:
            from PIL import Image
            sh, sw = dims[factor]
            with Image.open(views[index]["path"]) as image:
                image = image.convert("RGB").resize((sw, sh), Image.Resampling.LANCZOS)
                cached = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
            image_cache[key] = cached
        return cached

    trace = []
    stage_steps = {scale: 0 for scale in args.res_scales}
    stage_seconds = {scale: 0.0 for scale in args.res_scales}
    stage_gaussians = {scale: initial_count for scale in args.res_scales}

    def flush_progress(step):
        write_json(out / "progress.json", dict(
            trainer="research-mcmc-budget",
            steps=step,
            budget_seconds=args.budget.seconds,
            allocation_seconds=round(args.budget.allocation, 6),
            setup_seconds=round(setup_s, 6),
            res_scales=args.res_scales,
            res_untils=args.res_untils,
            trace=trace))

    trace.append(dict(step=0, seconds=round(setup_s, 4), train_seconds=0.0,
                      gaussians=initial_count, scale=args.res_scales[0],
                      resolution=list(dims[args.res_scales[0]]), step_seconds=0.0, loss=None))
    flush_progress(0)

    # ---- wall-clock-budgeted optimization ----
    observations = deque(maxlen=128)
    queue = []
    steps = 0
    stop_reason = None
    train_start = time.perf_counter()
    snap_targets = sorted(float(x) for x in str(getattr(args, "snap_seconds", "") or "").split(",") if x.strip())
    while True:
        total_elapsed = time.perf_counter() - start
        progress = args.budget.progress(total_elapsed, setup_s)
        factor = stage_scale(progress, args.res_scales, args.res_untils)
        predicted = predict_next_step(list(observations), factor, len(params["means"]),
                                      args.step_safety)
        # The first relocate/sample-add step can be much slower than a normal
        # step; be conservative before MCMC topology updates.
        if ((steps + 1) % args.refine_every == 0 and steps + 1 > len(views)
                and progress < args.topology_stop_fraction):
            predicted *= REFINE_STEP_PENALTY
        if args.budget.stop(total_elapsed, predicted):
            stop_reason = "time_budget"
            break
        if steps >= args.max_steps:
            stop_reason = "max_steps"
            break

        if not queue:
            queue = [int(i) for i in rng.permutation(len(views))]
        index = queue.pop()
        sh, sw = dims[factor]
        before_count = len(params["means"])
        lr = position_lr(progress)
        opts["means"].param_groups[0]["lr"] = lr
        for opt in opts.values():
            opt.zero_grad(set_to_none=True)

        tick = time.perf_counter()
        gt = image_for(index, factor).to(device=device, non_blocking=True).float().div_(255)
        rendered, _, info = rasterization(
            means=params["means"], quats=params["quats"],
            scales=params["scales"].exp(), opacities=params["opacities"].sigmoid(),
            colors=torch.cat((params["dc"][:, None], params["rest"]), dim=1),
            viewmats=viewmats[index:index + 1], Ks=intrinsics[factor][None],
            width=sw, height=sh, sh_degree=args.sh_degree,
            near_plane=0.01 * scene_scale, far_plane=200.0 * scene_scale,
            eps2d=0.3, rasterize_mode="classic", packed=False)
        loss = ((1 - SSIM_WEIGHT) * (rendered[0] - gt).abs().mean()
                + SSIM_WEIGHT * ssim_loss(rendered[0], gt))
        loss = loss + regularization_loss(
            params, dict(opacity=OPACITY_REGULARIZATION, scale=SCALE_REGULARIZATION))
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise ValueError(f"nonfinite training loss at step {steps + 1}")
        loss.backward()
        for opt in opts.values():
            opt.step()
        steps += 1
        if steps > len(views) and progress < args.topology_stop_fraction:
            strategy.step_post_backward(params, opts, state, steps, info, lr=lr)
        torch.cuda.synchronize()
        step_s = time.perf_counter() - tick

        after_count = len(params["means"])
        observations.append((step_s, factor, before_count))
        stage_steps[factor] += 1
        stage_seconds[factor] += step_s
        stage_gaussians[factor] = after_count

        while snap_targets and time.perf_counter() - train_start >= snap_targets[0]:
            _secs = time.perf_counter() - train_start
            _arrs = [params[k].detach().cpu().numpy()
                     for k in ("means", "scales", "quats", "dc", "rest", "opacities")]
            _arrs[4] = pad_rest_for_export(_arrs[4])
            _sp = out / "snaps" / f"snap_{int(snap_targets[0]):04d}s.ply"
            _sp.parent.mkdir(parents=True, exist_ok=True)
            _t0 = time.perf_counter()
            export_ply(_sp, _arrs, center, scene_scale)
            print(f"[curve] snapshot {snap_targets[0]:.0f}s (true {_secs:.2f}s, write {time.perf_counter()-_t0:.2f}s) -> {_sp}")
            snap_targets.pop(0)

        # Progress trace: first step, every 100 steps, and every stage change.
        if steps == 1 or steps % 100 == 0 or stage_steps[factor] == 1:
            row = dict(step=steps, seconds=round(time.perf_counter() - start, 4),
                       train_seconds=round(time.perf_counter() - train_start, 4),
                       gaussians=after_count, scale=factor, resolution=[sh, sw],
                       step_seconds=round(step_s, 4), loss=loss_value)
            trace.append(row)
            flush_progress(steps)

    train_s = time.perf_counter() - train_start
    if steps == 0:  # pragma: no cover - only with a mis-sized budget
        raise RuntimeError("no optimization step fit inside the allocation")
    if not trace or trace[-1]["step"] != steps:
        trace.append(dict(step=steps, seconds=round(time.perf_counter() - start, 4),
                          train_seconds=round(train_s, 4),
                          gaussians=int(len(params["means"])), scale=factor,
                          resolution=list(dims[factor]), step_seconds=round(step_s, 4),
                          loss=loss_value))
        flush_progress(steps)

    # ---- export: count the final CPU copies and PLY write as module-B time ----
    export_start = time.perf_counter()
    arrays = [params[key].detach().cpu().numpy()
              for key in ("means", "scales", "quats", "dc", "rest", "opacities")]
    arrays[4] = pad_rest_for_export(arrays[4])
    ply_rel = Path("point_cloud") / f"iteration_{steps}" / "point_cloud.ply"
    export_ply(out / ply_rel, arrays, center, scene_scale)
    export_s = time.perf_counter() - export_start

    total_s = time.perf_counter() - start
    stages = []
    for i, scale in enumerate(args.res_scales):
        stages.append(dict(scale=scale, until=(args.res_untils[i]
                                               if i < len(args.res_untils) else 1.0),
                           steps=stage_steps[scale],
                           seconds=round(stage_seconds[scale], 4),
                           gaussians_end=stage_gaussians[scale]))
    resolved = [row["scale"] for row in stages if row["steps"] > 0]
    schedule = dict(
        res_scales=args.res_scales, res_untils=args.res_untils,
        stages=stages, resolved_scales=resolved,
        sh_degree=args.sh_degree, refine_every=args.refine_every,
        cap_mult=round(cap_mult_actual, 6), cap=cap,
        topology_stop_fraction=args.topology_stop_fraction,
        step_safety=args.step_safety, max_steps=args.max_steps,
        position_lr_initial=POSITION_LR_INITIAL,
        position_lr_final=POSITION_LR_FINAL, lr_end_fraction=LR_END_FRACTION,
        ssim_weight=SSIM_WEIGHT,
        opacity_regularization=OPACITY_REGULARIZATION,
        scale_regularization=SCALE_REGULARIZATION,
        noise_lr=NOISE_LR, min_opacity=MIN_OPACITY)

    record = dict(
        schema_version=1,
        trainer="research-mcmc-budget",
        validation_status="CUDA execution pending; gsplat 1.5.3 API mirrored from competition/mcmc.py",
        work=str(args.work), init=str(args.init), init_format=args.init.suffix.lower().lstrip("."),
        ply=str(ply_rel), progress_trace=str(out / "progress.json"),
        seed=args.seed, device=str(device),
        gsplat_version=gsplat_version, torch_version=torch.__version__,
        budget_seconds=args.budget.seconds, budget_fraction=args.budget.fraction,
        allocation_seconds=round(args.budget.allocation, 6),
        export_reserve_seconds=round(args.budget.reserve, 6),
        total_seconds=round(total_s, 6),
        setup_seconds=round(setup_s, 6),
        train_seconds=round(train_s, 6),
        export_seconds=round(export_s, 6),
        steps=steps, iterations=steps,
        num_init_gaussians=initial_count,
        num_final_gaussians=int(len(params["means"])),
        num_train_views=len(views),
        cap=cap, cap_mult=round(cap_mult_actual, 6),
        stop_reason=stop_reason,
        allocation_met=args.budget.met(total_s),
        budget_met=total_s <= args.budget.seconds + 1e-9,
        schedule=schedule)
    write_json(out / "module_b.json", record)

    summary = {key: record[key] for key in (
        "steps", "total_seconds", "setup_seconds", "train_seconds", "export_seconds",
        "num_init_gaussians", "num_final_gaussians", "allocation_seconds",
        "stop_reason", "allocation_met", "budget_met")}
    print(f"[mcmc-budget] {out} " + str(summary), flush=True)
    if not record["allocation_met"]:
        print("[mcmc-budget] WARNING: measured total exceeded the allocation", flush=True)
    return record


def _expect_value_error(call, label, checks):
    try:
        call()
    except ValueError:
        return checks + 1
    raise AssertionError(f"self-check: expected ValueError for {label}")


def self_check():
    """CPU-only unit checks; no torch/CUDA import."""
    from competition.bridge import export_ply, parameter_arrays

    checks = 0

    def check(condition, message):
        nonlocal checks
        if not condition:
            raise AssertionError(f"self-check: {message}")
        checks += 1

    def close(a, b, tol=1e-9):
        return abs(a - b) <= tol

    budget = Budget(30, 0.9, 0.1)
    check(close(budget.allocation, 27.0), "30 s * 0.9 allocation")
    check(close(budget.reserve, 2.7), "10% export reserve")
    check(close(budget.allowance(3.0), 21.3), "setup is deducted")
    check(budget.stop(24.4, 0.0), "stop exactly past allocation-reserve")
    check(not budget.stop(24.2, 0.0), "do not stop with room left")
    check(close(budget.progress(3.0 + 10.65, 3.0), 0.5), "progress uses post-setup allowance")
    check(Budget(60, 0.9).met(53.0) and not Budget(60, 0.9).met(54.1), "allocation_met")
    checks = _expect_value_error(lambda: Budget(0, 0.9), "zero budget", checks)
    checks = _expect_value_error(lambda: Budget(30, 1.5), "fraction > 1", checks)
    checks = _expect_value_error(lambda: Budget(30, 0.9, 0.0), "reserve fraction", checks)

    observation = [(0.2, 0.5, 1000)]
    check(close(predict_next_step(observation, 0.5, 1000, 1.5), 0.3),
          "same-resolution prediction applies safety")
    check(close(predict_next_step(observation, 1.0, 1000, 1.5), 0.2 * 4 * 1.5),
          "resolution jump scales quadratically")
    check(close(predict_next_step(observation, 0.5, 2000, 1.5), 0.2 * 2 * 1.5),
          "Gaussian growth scales linearly")
    check(predict_next_step([], 1.0, 1000, 1.5) == 0.0, "first step has no estimate")

    scales, untils = parse_schedule("0.5,0.75,1.0", "0.4,0.7")
    check(scales == [0.5, 0.75, 1.0] and untils == [0.4, 0.7], "schedule parse")
    check(stage_scale(0.0, scales, untils) == 0.5, "stage at start")
    check(stage_scale(0.399, scales, untils) == 0.5, "stage before first cut")
    check(stage_scale(0.4, scales, untils) == 0.75, "stage at first cut")
    check(stage_scale(0.7, scales, untils) == 1.0, "stage at second cut")
    check(stage_scale(1.0, scales, untils) == 1.0, "stage at end")
    check(close(position_lr(0.0), POSITION_LR_INITIAL), "initial position LR")
    check(close(position_lr(1.0), POSITION_LR_FINAL), "final position LR")
    for bad_scales, bad_untils, label in (
            ("0.5,0.75,1.0", "0.4", "missing cut"),
            ("0.5,0.75", "0.4,0.7", "schedule must end at 1.0"),
            ("0.5,0.75,0.9", "0.4,0.7", "final scale not 1.0"),
            ("0.5,0.5,1.0", "0.4,0.7", "non-increasing scales"),
            ("0.5,1.0", "0.7,0.9", "too many cuts"),
            ("0.5,1.0", "0.0", "cut below zero"),
            ("0.5,1.0", "1.0", "cut at one"),
            ("1.2", "", "scale above one")):
        checks = _expect_value_error(
            lambda s=bad_scales, u=bad_untils: parse_schedule(s, u), label, checks)

    check(image_dims(128, 192, 0.5) == (64, 96), "half-resolution dims")
    check(image_dims(64, 64, 0.01) == (8, 8), "minimum dims")
    rest = np.zeros((4, 9, 3), dtype=np.float32)
    padded = pad_rest_for_export(rest)
    check(padded.shape == (4, 15, 3) and np.all(padded[:, 9:] == 0), "SH rest padding")

    argv = ["--work", "/tmp/work", "--init", "/tmp/init.ply", "--out", "/tmp/out",
            "--budget-seconds", "30", "--res-scales", "0.5,0.75,1.0",
            "--res-untils", "0.4,0.7", "--cap-mult", "2.0",
            "--refine-every", "100", "--sh-degree", "3", "--seed", "7"]
    parsed = validate_args(build_parser().parse_args(argv))
    check(parsed.budget.seconds == 30.0 and close(parsed.budget.allocation, 27.0),
          "CLI budget parse")
    check(parsed.res_scales == [0.5, 0.75, 1.0] and parsed.res_untils == [0.4, 0.7],
          "CLI schedule parse")
    check(parsed.cap_mult == 2.0 and parsed.refine_every == 100 and parsed.sh_degree == 3,
          "CLI knobs parse")
    default = validate_args(build_parser().parse_args(
        ["--work", "/tmp/w", "--init", "/tmp/i.npz", "--out", "/tmp/o",
         "--budget-seconds", "60"]))
    check(default.res_scales == [1.0] and default.res_untils == [], "single-resolution default")
    checks = _expect_value_error(lambda: validate_args(build_parser().parse_args(
        ["--work", "/tmp/w", "--init", "/tmp/i.ply", "--out", "/tmp/o"])),
        "missing budget", checks)

    # PLY/NPZ round-trip against the pinned bridge exporter/loader.
    import importlib.util
    import tempfile
    rng = np.random.default_rng(7)
    count = 6
    quats = rng.normal(size=(count, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    world = dict(
        means=rng.normal(size=(count, 3)).astype(np.float32),
        scales=np.exp(rng.normal(size=(count, 3))).astype(np.float32),
        rotations_xyzw=quats,
        harmonics=rng.normal(size=(count, 3, 16)).astype(np.float32),
        opacities=np.linspace(0.1, 0.9, count).astype(np.float32))
    center = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    scene_scale = 0.7
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.savez(tmp / "world.npz", **world)
        loaded_npz = read_world(tmp / "world.npz")
        for key in ("means", "scales", "rotations_xyzw", "harmonics", "opacities"):
            np.testing.assert_allclose(loaded_npz[key], world[key], rtol=1e-6, atol=1e-6)
            checks += 1
        checks = _expect_value_error(lambda: read_world(tmp / "world.txt"),
                                     "unsupported init suffix", checks)
        if importlib.util.find_spec("plyfile") is None:
            print("note: plyfile not installed; skipped the PLY round-trip")
        else:
            arrays = parameter_arrays(world, center, scene_scale)
            export_ply(tmp / "world.ply", arrays, center, scene_scale)
            loaded = read_world(tmp / "world.ply")
            for key in ("means", "scales", "rotations_xyzw", "harmonics", "opacities"):
                np.testing.assert_allclose(loaded[key], world[key], rtol=1e-5, atol=1e-6)
                checks += 1
            recovered = parameter_arrays(loaded, center, scene_scale)
            for expected, actual in zip(arrays, recovered):
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                checks += 1

    print(f"self-check OK: {checks} checks "
          "(arguments, allocation math, schedule, step prediction, PLY/NPZ round-trip)")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    try:
        args = validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    _run(args)
    return 0


if __name__ == "__main__":
    main()
