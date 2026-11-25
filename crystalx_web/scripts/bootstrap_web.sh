#!/bin/sh
# Add the local Web interface to the development environment, then prove it starts.
#
# Runs the core bootstrap first (it creates or repairs .venv and installs crystalx-core), then
# installs crystalx-native-tools (standard library only), the CrystalX Apple engine (brings PyTorch)
# and crystalx-web (brings Gradio). pip may download ordinary Python packages; nothing here
# downloads CrystalX source or model weights, and no SHELX, PLATON or other crystallographic
# executable is run.
#
# Usage from the repository root:
#   sh crystalx_web/scripts/bootstrap_web.sh
# Then check and start the interface:
#   sh scripts/crystalx.sh check
#   sh scripts/crystalx.sh web
#
# Optional overrides (same as the core bootstrap):
#   PYTHON=/path/to/python3 CRYSTALX_VENV=/path/to/venv sh crystalx_web/scripts/bootstrap_web.sh
# CRYSTALX_SKIP_APPLE=1 leaves out the Apple engine and PyTorch. The synthetic demo still works;
# `crystalx.sh check` then reports crystalx-apple as missing, and CrystalX inference is unavailable.

set -eu

REPO=$(cd "$(dirname "$0")/../.." && pwd)
VENV="${CRYSTALX_VENV:-$REPO/.venv}"

sh "$REPO/crystalx_core/scripts/bootstrap_dev.sh"

repair_macos_hidden_flags() {
    if [ "$(uname -s)" = "Darwin" ] && command -v chflags >/dev/null 2>&1; then
        chflags -R nohidden "$VENV"
    fi
}

echo "Installing native tool management (crystalx-native-tools; standard library only)"
"$VENV/bin/python" -m pip install --no-build-isolation -e "$REPO/crystalx_native_tools[dev]"
repair_macos_hidden_flags

if [ "${CRYSTALX_SKIP_APPLE:-0}" = "1" ]; then
    echo "Skipping the CrystalX Apple engine (CRYSTALX_SKIP_APPLE=1): no PyTorch, no CrystalX inference"
else
    echo "Installing the CrystalX Apple engine (crystalx-engine-apple-silicon, with PyTorch; no model weights)"
    "$VENV/bin/python" -m pip install --no-build-isolation -e "$REPO/crystalx_engine_apple_silicon[dev]"
    repair_macos_hidden_flags
fi

echo "Installing the Web interface (crystalx-web, with Gradio)"
"$VENV/bin/python" -m pip install --no-build-isolation -e "$REPO/crystalx_web[dev]"
repair_macos_hidden_flags

echo "Verifying the installed launch command (no server is started)"
cd "$REPO"
"$VENV/bin/crystalx-web" --check

echo
echo "Ready. Start the interface with:"
echo "    $VENV/bin/crystalx-web"
echo "then open the address it prints (http://127.0.0.1:7860 unless that port is busy)."
