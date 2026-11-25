#!/bin/sh
# CrystalX Web demo -- developer launcher for macOS Finder (double-click this file).
#
# Prerequisite: this checkout's project environment (.venv), prepared once with
#     sh crystalx_web/scripts/bootstrap_web.sh
# This is NOT the portable application: it uses that environment in place. The packaged,
# relocatable "CrystalX Demo.app" is built separately (see crystalx_web/packaging/README.md).
#
# It goes through the project's supported entry point (scripts/crystalx.sh), which checks the
# environment, then starts this checkout's own local server (127.0.0.1 only) or reuses it when it is
# already running and proves its identity, waits until it answers, and opens it in the browser.
# This window is the server's console: close it or press Ctrl-C to stop the server. Run records
# stay in runs/ of this checkout. Nothing here installs, downloads or changes system settings.
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
repo=$(CDPATH= cd -- "$here/.." && pwd) || exit 1
entry="$repo/scripts/crystalx.sh"

pause() {
    printf '\n%s ' "$1"
    read -r _ || true
}

if [ ! -f "$entry" ]; then
    echo "找不到项目启动入口 / Cannot find the project start script: $entry" >&2
    pause "按回车键关闭此窗口 / Press Return to close this window."
    exit 2
fi
/bin/sh "$entry" web --desktop --attach
status=$?
case "$status" in
    0|130|143) ;;
    *) pause "启动没有成功（退出码 $status）。按回车键关闭此窗口 / Start failed (exit $status). Press Return to close." ;;
esac
exit "$status"
