#!/usr/bin/env bash
# One-time setup: build the vendored 3DGS CUDA submodules in the fastgs env.
# (The resplat env is assumed to be already installed per resplat/README.md.)
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

source /home/guojunfu/anaconda3/etc/profile.d/conda.sh
conda activate fastgs


export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
export CC=${CC:-/usr/bin/gcc}
export CXX=${CXX:-/usr/bin/g++}

for mod in diff-gaussian-rasterization simple-knn fused-ssim; do
  echo "=== building $mod ==="
  cd "$HERE/3dgs/submodules/$mod"
  python setup.py build_ext --inplace
done

echo "setup done."
