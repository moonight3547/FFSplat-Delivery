#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# No arguments: evaluate all three groups from scenes.json and write summaries.
exec "${PYTHON:-python3}" "$HERE/evaluate_delivery.py" "$@"
