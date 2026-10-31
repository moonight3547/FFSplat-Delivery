#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python}" "$HERE/module_b/train_native_reference.py" "$@"
