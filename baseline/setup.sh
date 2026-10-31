#!/usr/bin/env bash
# Same classic-3DGS extension builds as upstream, using a portable conda command.
# Install the FastGS renderer/environment first: see docs/ENVIRONMENT.md.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CONDA_BIN="${FFSPLAT_CONDA:-conda}"
FASTGS_ENV="${FFSPLAT_FASTGS_ENV:-fastgs}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
for mod in diff-gaussian-rasterization simple-knn fused-ssim; do
  cd "$HERE/3dgs/submodules/$mod"
  "$CONDA_BIN" run --no-capture-output -n "$FASTGS_ENV" python setup.py build_ext --inplace
done
