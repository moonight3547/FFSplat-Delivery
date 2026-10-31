#!/usr/bin/env bash
# Create both isolated runtimes; --plan only prints the installation plan.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python3}" "$HERE/../scripts/setup_delivery.py" "$@"
