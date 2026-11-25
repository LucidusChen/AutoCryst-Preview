#!/bin/sh
# Supported local entry point: repair only this project's editable-import flags before Python site.
set -eu
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
RUNTIME_PYTHON="$REPO/.venv/bin/python"
if [ ! -x "$RUNTIME_PYTHON" ]; then
    echo "Project Python is missing. Run: sh crystalx_web/scripts/bootstrap_web.sh" >&2
    exit 2
fi
# The repair must run before Python processes .pth files. Never mask a broken editable install.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
exec "$RUNTIME_PYTHON" -S "$REPO/scripts/local_runtime.py" "$@"
