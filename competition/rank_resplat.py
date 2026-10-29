"""16-view model-ranked A + unchanged organizer export, one shared model load."""
import argparse
from pathlib import Path
import sys
import time
from .contracts import read_json, validate_split, write_json
from .resplat import load_runner, seed_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path(__file__).parent/"upstream")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--settings", type=Path, default=Path(__file__).parent/"policies/ranked16.json")
    args = parser.parse_args()
    from .bridge import read_training_views
    from .selection import proposals, evaluate_context, rank_candidates
    import torch
    if (args.work/"selection.json").exists() or (args.out.exists() and any(args.out.iterdir())):
        raise FileExistsError("Refuse to repeat/overwrite a completed or partial A/R stage")
    cfg = read_json(args.settings)
    if cfg != dict(policy="ranked16-v1", probe_count=8, max_changes=2, relative_gain=.02, seed=20261015):
        raise ValueError("Use the frozen bounded ranked16-v1 settings for this transfer")
    upstream = load_runner(args.upstream)
    torch.cuda.synchronize(); start = time.perf_counter()
    split, views = read_training_views(args.work)
    probes, candidates = proposals(views, split["context"], cfg["probe_count"], cfg["max_changes"])
    model = None
    if len(candidates) > 1 and probes:
        seed_inference(cfg["seed"])
        model = upstream.build_model(experiment="dl3dv", checkpoint=str(args.checkpoint), num_refine=2,
            image_shape=tuple(split["resolution"]), overrides=list(upstream.PRESET_OVERRIDES), device=args.device)
        def evaluate(context):
            seed_inference(cfg["seed"])
            return evaluate_context(upstream, model, views, context, probes, split["resolution"], args.device)
        selected, rows = rank_candidates(candidates, evaluate, cfg["relative_gain"])
    else:
        selected, rows = 0, []
    original = list(split["context"])
    split["context"] = candidates[selected]
    validate_split(split)
    write_json(args.work/"split.json", split)
    write_json(args.work/"selection.json", dict(settings=cfg, probes=probes, proposals=candidates, selected=selected,
        metrics=rows, original_context=original, selected_context=split["context"],
        network_refinement_target="selected context only; probes used for post-inference rendering",
        model_builds_in_A=1 if model is not None else 0))
    torch.cuda.synchronize(); selection_s = time.perf_counter()-start
    info = read_json(args.work/"module_a.json")
    info.update(preprocessing_seconds=info["total_seconds"], selection_seconds=selection_s,
                total_seconds=info["total_seconds"]+selection_s, a_policy=cfg["policy"])
    write_json(args.work/"module_a.json", info)
    # Delegate final inference and exact world/SH/PLY export to the fixed source.
    # Only model construction is cached. The winning context gets a fresh forward.
    original_build, argv = upstream.build_model, sys.argv
    if model is not None:
        def reuse_model(**kwargs):
            if (kwargs["checkpoint"] != str(args.checkpoint) or kwargs["num_refine"] != 2 or
                    tuple(kwargs["image_shape"]) != tuple(split["resolution"]) or
                    kwargs["overrides"] != list(upstream.PRESET_OVERRIDES) or kwargs["device"] != args.device):
                raise ValueError("Cached model construction contract changed")
            seed_inference(cfg["seed"])
            return model
        upstream.build_model = reuse_model
    try:
        seed_inference(cfg["seed"])
        sys.argv = ["organizer_run_resplat", "--work", str(args.work), "--out", str(args.out),
                    "--checkpoint", str(args.checkpoint), "--num_refine", "2", "--device", args.device]
        upstream.main()
    finally:
        upstream.build_model, sys.argv = original_build, argv


if __name__ == "__main__":
    main()
