#!/bin/sh
# CrystalX Demo: start (or reuse) the local Web demo and open it in the browser.
#
# Run by the application's native main executable (Contents/MacOS/CrystalX Demo). It uses only
# the Python runtime inside this bundle and writes only to the user's own folders, never into the
# bundle, which may be read-only:
#   macOS:  ~/Library/Application Support/CrystalX Demo   runs, records, instance state
#           ~/Library/Logs/CrystalX Demo                   logs
#           ~/Library/Caches/CrystalX Demo                 Python byte code
#   other:  the XDG data/state/cache folders (used by the Linux stand-in test of this layout)
# It does not install or download anything and changes no system, proxy, DNS, quarantine or
# Gatekeeper setting. The server listens on 127.0.0.1 only; stop it from the page.
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
python="$here/python/bin/python3"
case "$(uname -s)" in
    Darwin)
        logs="$HOME/Library/Logs/CrystalX Demo"
        cache="$HOME/Library/Caches/CrystalX Demo"
        ;;
    *)
        logs="${XDG_STATE_HOME:-$HOME/.local/state}/crystalx-demo"
        cache="${XDG_CACHE_HOME:-$HOME/.cache}/crystalx-demo"
        ;;
esac
mkdir -p "$logs" "$cache" 2>/dev/null
log="$logs/launcher.log"

tell() {
    message=$(printf '%s' "$1" | tr '"\\' "'/")
    if [ -x /usr/bin/osascript ]; then
        /usr/bin/osascript -e "display alert \"CrystalX Demo\" message \"$message\" as critical" >/dev/null 2>&1 || true
    fi
    printf 'CrystalX Demo: %s\n' "$1" >&2
}

if ! { : >> "$log"; } 2>/dev/null; then
    tell "Cannot write the log file $log. Check that your home folder's Library is writable."
    exit 1
fi
if [ ! -x "$python" ]; then
    tell "This copy of CrystalX Demo is incomplete: its Python runtime is missing. Please use a complete copy of the application."
    exit 1
fi

# The language of the user's macOS session, for the start messages and the page.
if [ -z "${CRYSTALX_LANG:-}" ] && [ -x /usr/bin/defaults ]; then
    first=$(/usr/bin/defaults read -g AppleLanguages 2>/dev/null | sed -n '2p' | tr -d ' ",')
    case "$first" in
        ja*) CRYSTALX_LANG=ja ;;
        en*) CRYSTALX_LANG=en ;;
        zh*) CRYSTALX_LANG=zh ;;
    esac
fi
[ -n "${CRYSTALX_LANG:-}" ] && export CRYSTALX_LANG

# A clean interpreter: nothing from the user's shell set-up reaches it (-I also ignores PYTHON*).
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV
printf '\n== %s ==\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$log" 2>/dev/null
"$python" -I -X "pycache_prefix=$cache/pycache" -m crystalx_web.desktop launch --profile app >> "$log" 2>&1
status=$?
if [ "$status" -ne 0 ]; then
    reason=$(grep -v '^[[:space:]]*$' "$log" 2>/dev/null | tail -n 3 | tr '\n' ' ' | cut -c1-500)
    tell "CrystalX Demo could not start (exit $status). $reason Log: $log"
fi
exit "$status"
