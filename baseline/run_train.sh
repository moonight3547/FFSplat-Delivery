#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ "${1:-}" == "--scene" ]]; then
  exec "${PYTHON:-python}" "$HERE/../run_pipeline.py" "$@"
fi
exec "${PYTHON:-python}" "$HERE/run_suite.py" "$@"
