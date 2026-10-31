#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
for arg in "$@"; do
  if [[ "$arg" == "--scene" || "$arg" == --scene=* ]]; then
    exec "${PYTHON:-python}" "$HERE/../run_pipeline.py" "$@"
  fi
done
exec "${PYTHON:-python}" "$HERE/run_suite.py" "$@"
