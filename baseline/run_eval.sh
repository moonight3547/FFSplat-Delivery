#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python}" "$HERE/evaluate_delivery.py" "$@"
