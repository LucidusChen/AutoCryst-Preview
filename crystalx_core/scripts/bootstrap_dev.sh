#!/bin/sh
# Create or repair the local development environment, then prove its CLI works.
#
# This script is deliberately local-only. It may download normal Python dependencies through pip,
# but it never downloads model weights and never runs SHELX, PLATON, or any other crystallographic
# executable.
#
# Usage from the repository root:
#   sh crystalx_core/scripts/bootstrap_dev.sh
#
# Optional overrides:
#   PYTHON=/path/to/python3 CRYSTALX_VENV=/path/to/venv sh crystalx_core/scripts/bootstrap_dev.sh

set -eu

REPO=$(cd "$(dirname "$0")/../.." && pwd)
CORE="$REPO/crystalx_core"
VENV="${CRYSTALX_VENV:-$REPO/.venv}"
PYTHON="${PYTHON:-python3}"

if [ ! -x "$VENV/bin/python" ]; then
    # The packages declare requires-python >= 3.10. The python3 that ships with the macOS command
    # line tools is older, so refuse it here with a clear next step instead of failing inside pip.
    if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        echo "Python 3.10 or newer is required (3.12 recommended); '$PYTHON' is not usable:" >&2
        "$PYTHON" -V >&2 2>&1 || echo "  '$PYTHON' was not found" >&2
        echo "Install Python 3.12 (python.org installer, or Homebrew: brew install python@3.12)," >&2
        echo "then run the bootstrap again with PYTHON=python3.12 in front of the command." >&2
        exit 2
    fi
    echo "Creating virtual environment: $VENV"
    "$PYTHON" -m venv "$VENV"
fi

# Python 3.12 on macOS deliberately skips .pth files carrying the UF_HIDDEN flag. A Finder-hidden
# .venv can pass that flag to the editable-install .pth file, making the installed `crystalx`
# command fail to import the package. Clear the flag before and after pip writes site-packages.
repair_macos_hidden_flags() {
    if [ "$(uname -s)" = "Darwin" ] && command -v chflags >/dev/null 2>&1; then
        chflags -R nohidden "$VENV"
    fi
}

repair_macos_hidden_flags

echo "Installing editable development package"
# The editable installs below use --no-build-isolation, so the build backend must be in the venv.
# A working development venv already has it, and nothing is fetched then. A fresh venv (Python 3.12
# no longer seeds setuptools) gets exactly the backend that every pyproject.toml here declares.
if ! "$VENV/bin/python" -c 'import setuptools' 2>/dev/null; then
    echo "Installing the declared build backend (setuptools>=68) into the new environment"
    "$VENV/bin/python" -m pip install "setuptools>=68"
fi
"$VENV/bin/python" -m pip install --no-build-isolation -e "$CORE[dev,geo]"
repair_macos_hidden_flags

echo "Verifying import and console command"
"$VENV/bin/python" -c "import crystalx_core; print('crystalx-core', crystalx_core.__version__)"
"$VENV/bin/crystalx" --version

echo "Development environment ready. No crystallographic executable was run."
