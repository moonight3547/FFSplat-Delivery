"""Seeded invocation of the unchanged organizer inference/export entrypoint."""
import argparse
import importlib.util
from pathlib import Path
import random
import sys

SEED = 20261015


def seed_inference(seed=SEED):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_runner(upstream):
    path = Path(upstream)/"baseline/run_resplat.py"
    spec = importlib.util.spec_from_file_location("organizer_run_resplat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.build_model
    def seeded_model(**kwargs):
        seed_inference()
        result = original(**kwargs)
        # The same inference RNG state with and without A's shared model load.
        seed_inference()
        return result
    module.build_model = seeded_model
    return module


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--upstream", type=Path, default=Path(__file__).parent/"upstream")
    args, rest = parser.parse_known_args()
    upstream = load_runner(args.upstream)
    argv = sys.argv
    try:
        sys.argv = ["organizer_run_resplat", *rest]
        upstream.main()
    finally:
        sys.argv = argv


if __name__ == "__main__":
    main()
