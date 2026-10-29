"""Experimental MCMC/CTF transfer to organizer world-space artifacts.

CUDA acceptance and comparison to the organizer FastGS control are still required.
All preprocessing, training, control and PLY export below count toward B.
"""
import argparse
import math
from pathlib import Path
import time
from .contracts import read_json, remaining_b_seconds, validate_reference, write_json
from .budget import Deadline, validate_settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--resplat-record", type=Path, required=True)
    p.add_argument("--settings", type=Path, default=Path(__file__).parent/"policies/mcmc-ctf.json")
    args = p.parse_args()
    import importlib.metadata
    import numpy as np
    import torch
    from PIL import Image
    from gsplat.rendering import rasterization
    from gsplat.strategy import MCMCStrategy
    from .train_utils import make_parameters, make_optimizers, ssim_loss, regularization_loss
    from .bridge import load_world, parameter_arrays, read_training_views, export_ply

    if importlib.metadata.version("gsplat") != "1.5.3":
        raise RuntimeError("Transfer candidate requires gsplat 1.5.3")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Choose a fresh B output directory")
    cfg = validate_settings(read_json(args.settings))
    ref = validate_reference(read_json(args.reference))
    limit = remaining_b_seconds(ref["total_seconds"], read_json(args.work/"module_a.json")["total_seconds"],
                               read_json(args.resplat_record)["total_seconds"], cfg["e2e_fraction"])
    torch.cuda.synchronize()
    start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    split, views = read_training_views(args.work)
    world = load_world(args.initial)
    poses = np.stack([v["c2w"] for v in views])
    center = poses[:, :3, 3].mean(0)
    extent = np.max(np.abs(poses[:, :3, 3]-center))
    if not math.isfinite(float(extent)) or extent <= 0:
        raise ValueError("Nondegenerate training cameras required")
    scale = float(1/extent)
    params = make_parameters(parameter_arrays(world, center, scale), "cuda")
    initial_count = len(params["means"])
    cap = max(initial_count, cfg["cap_floor"])
    poses[:, :3, 3] = (poses[:, :3, 3]-center)*scale
    viewmats = torch.linalg.inv(torch.from_numpy(poses).cuda())
    h, w = split["resolution"]
    # Host cache avoids holding all three resolution image tensors on the GPU.
    images = {}
    intrinsics = {}
    for factor in sorted({s["scale"] for s in cfg["ctf"]}):
        sh, sw = max(8, round(h*factor)), max(8, round(w*factor))
        batch = []
        for view in views:
            with Image.open(view["path"]) as im:
                batch.append(torch.from_numpy(np.array(im.convert("RGB").resize((sw, sh), Image.Resampling.LANCZOS), dtype=np.uint8)))
        images[factor] = batch
        K = views[0]["K"].copy(); K[0] *= sw/w; K[1] *= sh/h
        # Official FastGS constructs projection from FoV and ignores COLMAP cx/cy.
        K[0, 2], K[1, 2] = sw/2, sh/2
        intrinsics[factor] = torch.from_numpy(K).cuda()
    opts = make_optimizers(params, cfg)
    maximum = cfg["max_updates"]
    strategy = MCMCStrategy(cap_max=cap, min_opacity=cfg["min_opacity"], noise_lr=cfg["noise_lr"],
        refine_start_iter=0, refine_stop_iter=maximum+1, refine_every=cfg["refine_every"])
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state()
    rng = np.random.default_rng(cfg["seed"])
    queue, trace = [], []
    stage_steps = {str(s["scale"]): 0 for s in cfg["ctf"]}
    torch.cuda.synchronize()
    setup_s = time.perf_counter()-start
    deadline = Deadline(limit, setup_s, cfg)
    train_start = time.perf_counter()
    steps = 0
    last_row = None
    while True:
        elapsed = time.perf_counter()-train_start
        plan = deadline.plan(elapsed, steps, len(params["means"]))
        if plan["stop_reason"]: break
        progress, factor, lr = (plan[k] for k in ("progress", "scale", "position_lr"))
        if not queue: queue = list(rng.permutation(len(views)))
        index = queue.pop()
        tick = time.perf_counter()
        before_count = len(params["means"])
        gt = images[factor][index].cuda().float()/255
        sh, sw = gt.shape[:2]
        opts["means"].param_groups[0]["lr"] = lr
        for opt in opts.values(): opt.zero_grad(set_to_none=True)
        rendered, _, info = rasterization(means=params["means"], quats=params["quats"],
            scales=params["scales"].exp(), opacities=params["opacities"].sigmoid(),
            colors=torch.cat((params["dc"][:, None], params["rest"]), dim=1),
            viewmats=viewmats[index:index+1], Ks=intrinsics[factor][None], width=sw, height=sh,
            sh_degree=3, near_plane=.01*scale, far_plane=200.*scale,
            eps2d=.3, rasterize_mode="classic", packed=False)
        loss = (1-cfg["ssim_weight"])*(rendered[0]-gt).abs().mean()+cfg["ssim_weight"]*ssim_loss(rendered[0], gt)
        loss = loss+regularization_loss(params, dict(opacity=cfg["opacity_regularization"], scale=cfg["scale_regularization"]))
        if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
        loss.backward()
        for opt in opts.values(): opt.step()
        steps += 1
        if steps > len(views) and progress < cfg["topology_stop_fraction"]:
            strategy.step_post_backward(params, opts, state, steps, info, lr=lr)
        torch.cuda.synchronize()
        step_s = time.perf_counter()-tick
        deadline.observe(step_s, factor, before_count)
        stage_steps[str(factor)] += 1
        last_row = dict(step=steps, **plan, gaussians=len(params["means"]),
                        step_seconds=step_s, elapsed_b_s=time.perf_counter()-start)
        if steps == 1 or steps % 100 == 0 or (trace and trace[-1]["scale"] != factor):
            trace.append(last_row)
            write_json(args.out/"progress.json", dict(steps=steps, trace=trace))
    if steps == 0:
        raise RuntimeError("B setup exhausted the allocation; no optimization result")
    if not trace or trace[-1]["step"] != steps: trace.append(last_row)
    arrays = [params[k].detach().cpu().numpy() for k in ("means", "scales", "quats", "dc", "rest", "opacities")]
    export_ply(args.out/f"point_cloud/iteration_{steps}/point_cloud.ply", arrays, center, scale)
    write_json(args.out/"training_trace.json", dict(settings=cfg, trace=trace, initial_count=initial_count,
        cap=cap, allocation_seconds=limit, training_steps=steps, stage_steps=stage_steps,
        stop_reason=plan["stop_reason"], export_reserve_seconds=deadline.reserve,
        renderer="gsplat classic eps2d=.3; FastGS parity pending"))
    torch.cuda.synchronize()
    total = time.perf_counter()-start
    write_json(args.out/"module_b.json", dict(trainer="experimental-organizer-mcmc-ctf", iterations=steps,
        num_init_gaussians=initial_count, num_final_gaussians=len(params["means"]), init_mode="organizer_world_npz",
        total_seconds=total, setup_seconds=setup_s, allocation_seconds=limit,
        allocation_met=total < limit, stop_reason=plan["stop_reason"], stage_steps=stage_steps,
        validation_status="unvalidated transfer; final rendering uses organizer FastGS"))


if __name__ == "__main__":
    main()
