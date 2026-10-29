#!/usr/bin/env bash
# Golden standard depth dump: re-run ReSplat with save_depth and store the
# predicted depth of the 16 context views next to resplat_world.ply.
# Gating step for the depth-regularized module-B experiments (research/depth_reg.py).
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-/usr/local/miniconda3/bin/conda}
ENV=${ENV:-ffsplat-enhance}
for case in hw/billboard mipnerf360/counter; do
  work="outputs/O0-20261030/work/$case"
  out="outputs/O0-20261030/results/$case/resplat"
  [ -f "$out/depth_raw.npz" ] && { echo "skip $case"; continue; }
  [ -f "$work/split.json" ] || { echo "missing $case"; continue; }
  echo "=== depth $case $(date -Is)"
  CASE_WORK="$work" CASE_OUT="$out" $PY run --no-capture-output -n "$ENV" python - <<'PY'
import json, os, sys
import numpy as np
sys.path.insert(0, "competition/upstream/baseline")
import run_resplat
work, out = os.environ["CASE_WORK"], os.environ["CASE_OUT"]
context = json.load(open(work + "/split.json", encoding="utf-8"))["context"]
original = run_resplat.run_inference
def capture(*args, **kwargs):
    kwargs["save_depth"] = True
    gaussians, rendered, rendered_depth, dump = original(*args, **kwargs)
    depth = dump["depth_fullres"][0].detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(out + "/depth_raw.npz", depth=depth, names=np.asarray(context))
    print("[depth] dumped", depth.shape, "->", out)
    return gaussians, rendered, rendered_depth, dump
run_resplat.run_inference = capture
sys.argv = ["run_resplat.py", "--work", work, "--out", out,
            "--checkpoint", "/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth",
            "--num_refine", "2", "--device", "cuda:0"]
run_resplat.main()
PY
  [ -f "$out/depth_raw.npz" ] && $PY run --no-capture-output -n ffsplat-enhance python -m research.export_resplat_depth \
    --raw "$out/depth_raw.npz" --split "$work/split.json" --out "$out/resplat_depth.npz" 2>&1 | tail -2
done
echo done
