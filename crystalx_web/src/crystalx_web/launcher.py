"""``crystalx-web``: start the local Web interface.

    crystalx-web                      # http://127.0.0.1:7860 (or the next free port)
    crystalx-web --runs-dir ~/runs    # keep run records elsewhere
    crystalx-web --check              # verify the installation and exit; starts no server
    crystalx-web --tools-config FILE  # tool locations (default: ./config.local.json if present)
    crystalx-web --allow-translation  # start with strict native mode off (Rosetta 2 allowed)

Desktop start (what the double-click launchers use; see :mod:`crystalx_web.desktop`)::

    crystalx-web --desktop [--attach]  # start or reuse the owned server, open it in the browser
    crystalx-web --desktop-stop        # stop the owned server (identity verified)
    crystalx-web --desktop-status      # is it running, where, which log

The server binds to the loopback interface only; there is deliberately no option to expose it.
Gradio's telemetry is switched off before Gradio is imported, its upload and cache directory is
placed inside the runs directory (an ignored location), loopback is exempted from any proxy for
this process only, and nothing is shared.
"""

from __future__ import annotations

import argparse
import atexit
import ipaddress
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

__all__ = ["main", "build_parser", "prepare_environment", "build_service_and_settings"]

_LOOPBACK_NAMES = {"localhost"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crystalx-web",
        description="Local Web interface for a preliminary CrystalX solve (SHELXT, SHELXL, PLATON).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Loopback address to bind (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=None, help="Port (default: 7860 or the next free one).")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Run records (default ./runs).")
    parser.add_argument("--toolbox", type=Path, default=None, help="Directory holding SHELX/PLATON executables.")
    parser.add_argument("--tools-config", type=Path, default=None,
                        help="Tool configuration file (default: ./config.local.json when it exists).")
    parser.add_argument("--allow-translation", action="store_true",
                        help="Start with strict native mode off. The page has a switch either way.")
    parser.add_argument("--check", action="store_true", help="Verify the installation and exit.")
    desktop = parser.add_argument_group("desktop start")
    action = desktop.add_mutually_exclusive_group()
    action.add_argument("--desktop", action="store_true",
                        help="Start or reuse this checkout's own local server and open it in the browser.")
    action.add_argument("--desktop-stop", action="store_true", help="Stop the owned desktop server.")
    action.add_argument("--desktop-status", action="store_true", help="Report the owned desktop server.")
    desktop.add_argument("--attach", action="store_true",
                         help="With --desktop: keep this terminal as the server's console (Ctrl-C stops it).")
    desktop.add_argument("--profile", choices=("dev", "app"), default="dev",
                         help="dev: this checkout's runs/ (default); app: per-user folders of CrystalX Demo.")
    desktop.add_argument("--data-dir", type=Path, default=None, help="Keep runs, records and logs in this folder.")
    desktop.add_argument("--no-browser", action="store_true", help="Do not open the browser.")
    desktop.add_argument("--startup-timeout", type=float, default=120.0,
                         help="Seconds to wait for the server to answer (default 120).")
    return parser


def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def prepare_environment(runs_dir: Path) -> Path:
    """Set Gradio's environment before it is imported. Returns the interface's state directory.

    Gradio checks its own startup through HTTP to 127.0.0.1; an inherited proxy setting without a
    loopback exception made that fail or hang (review D2). Loopback is exempted here, for this
    process and its children only.
    """
    from crystalx_web.desktop import apply_loopback_proxy_exemption

    runs_dir = runs_dir.expanduser().resolve()
    state_dir = runs_dir / "_web"
    temp_dir = runs_dir / "_gradio_tmp"
    for directory in (runs_dir, state_dir, temp_dir):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["GRADIO_TEMP_DIR"] = str(temp_dir)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("GRADIO_SSR_MODE", "False")
    apply_loopback_proxy_exemption()
    return state_dir


def _load_tools_config(path: Path | None, toolbox: Path | None):
    """The native-tools configuration, or ``(None, reason)`` when the package is missing."""
    try:
        from crystalx_native_tools.config import ToolsConfig, default_config_path, load_config
    except ImportError:
        return None, "crystalx-native-tools is not installed (run sh crystalx_web/scripts/bootstrap_web.sh)"
    source = path if path is not None else default_config_path()
    config = load_config(source)
    if toolbox is None and config.toolbox is None and Path("toolbox").is_dir():
        toolbox = Path(os.path.abspath("toolbox"))
    config = config.with_overrides(toolbox=toolbox)
    note = str(config.source) if config.source else f"none ({source} not found); defaults apply"
    if config.problems:
        note += " -- problems: " + "; ".join(config.problems)
    return config, note


def build_service_and_settings(
    runs_dir: Path,
    *,
    toolbox: Path | None = None,
    tools_config: Path | None = None,
    allow_translation: bool = False,
    desktop_stop: Callable[[], None] | None = None,
    quick_start: bool = False,
    default_lang: str | None = None,
) -> tuple[Any, Any, list[str]]:
    """The run service and page settings every start shares. Call after :func:`prepare_environment`."""
    from crystalx_core.app import RunService
    from crystalx_web.app import WebSettings

    runs_dir = runs_dir.expanduser().resolve()
    state_dir = runs_dir / "_web"
    try:  # the same deeper check the tool panel uses, so a run selects the file the panel shows
        from crystalx_native_tools.inspect import candidate_check_for

        candidate_check = candidate_check_for()
    except ImportError:
        candidate_check = None
    service = RunService(runs_dir, bundles_dir=state_dir / "bundles", candidate_check=candidate_check)
    # Absolute against the launch directory, symlinks kept: programs run inside run directories,
    # where a relative toolbox would point somewhere else (Codex review R1, 2026-10-14).
    toolbox = Path(os.path.abspath(toolbox.expanduser())) if toolbox else None
    config, config_note = _load_tools_config(tools_config, toolbox)
    if config is not None and toolbox is None:
        toolbox = getattr(config, "toolbox", None)
    strict_default = not allow_translation
    if config is not None and getattr(config, "strict_native", None) is not None and not allow_translation:
        strict_default = bool(config.strict_native)
    checkout = runs_dir.parent
    repository = checkout if (runs_dir.name == "runs" and (checkout / ".git").exists()
                              and (checkout / "crystalx_core").is_dir()) else None
    settings = WebSettings(
        state_dir=state_dir,
        repository=repository,
        toolbox_dir=toolbox,
        tools_config=config,
        strict_native_default=strict_default,
        probe_store=runs_dir / "_tools" / "launch_probes.json",
        quick_start=quick_start,
        desktop_stop=desktop_stop,
        **({"default_lang": default_lang} if default_lang else {}),
    )
    notes = [f"tool configuration: {config_note}", f"strict native mode at start: {'on' if strict_default else 'off'}",
             "iterative reconstruction: " + (f"runs in {repository / 'runs' / 'reconstruction'}" if repository
                                             else "unavailable (no developer checkout)")]
    return service, settings, notes


def _print_native_summary(tools_config, strict: bool) -> None:
    if tools_config is None:
        return
    from crystalx_native_tools.inventory import build_inventory

    inventory = build_inventory(tools_config, strict_native=strict)
    print(f"native tools (static check, nothing run): pipeline {inventory.pipeline}"
          + (f" -- blocked: {', '.join(inventory.pipeline_blockers)}" if inventory.pipeline_blockers else ""))
    for status in inventory.tools:
        selected = status.selected
        print(f"  {status.tool:<15} static={status.static:<9} launch={status.launch.get('status'):<11} "
              f"{selected.native + ' ' + selected.path if selected else '-'}")


def _desktop(args: argparse.Namespace) -> int:
    """The desktop actions run without importing Gradio in this process (the server imports it)."""
    from crystalx_web import desktop

    if args.port is not None and not 1 <= args.port <= 65535:
        print("error: --port must be between 1 and 65535", file=sys.stderr)
        return 2
    paths = desktop.paths_for(args.profile, args.data_dir)
    for name in ("toolbox", "tools_config"):  # absolute: the server runs from the data folder
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(os.path.abspath(value.expanduser())))
    if args.desktop_stop:
        return desktop.stop(paths)
    if args.desktop_status:
        return desktop.status(paths)
    return desktop.launch(
        paths,
        open_browser=not args.no_browser,
        attach=args.attach,
        startup_timeout=args.startup_timeout,
        port=args.port,
        toolbox=args.toolbox,
        tools_config=args.tools_config,
        allow_translation=args.allow_translation,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not _is_loopback(args.host):
        print(
            f"error: {args.host} is not a loopback address. This interface only serves the local "
            "machine (127.0.0.1 or ::1).",
            file=sys.stderr,
        )
        return 2
    if args.desktop or args.desktop_stop or args.desktop_status:
        return _desktop(args)
    if args.attach:
        print("error: --attach needs --desktop", file=sys.stderr)
        return 2
    state_dir = prepare_environment(args.runs_dir)
    runs_dir = args.runs_dir.expanduser().resolve()

    import gradio  # noqa: E402 - imported only after the environment above is set

    from crystalx_core.app import RunConfig, RunMode
    from crystalx_core.version import CORE_VERSION
    from crystalx_web.app import build_app, launch_options, run_tool_kwargs

    service, settings, notes = build_service_and_settings(
        runs_dir, toolbox=args.toolbox, tools_config=args.tools_config, allow_translation=args.allow_translation
    )
    toolbox = settings.toolbox_dir
    strict_default = settings.strict_native_default

    if args.check:
        report = service.capabilities(RunMode.REAL, config=RunConfig(**run_tool_kwargs(settings, strict_default)))
        build_app(service, settings)  # proves the page can be constructed
        host = report["host"]
        print(f"crystalx-web ready: crystalx-core {CORE_VERSION}, gradio {gradio.__version__}")
        print(f"host: {host['system']} {host['arch']}, python {host['python']}")
        print(f"runs directory: {runs_dir}")
        print(f"toolbox: {toolbox if toolbox else '(not set; PATH and known locations are searched)'}")
        for note in notes:
            print(note)
        _print_native_summary(settings.tools_config, strict_default)
        for name, tool in report["tools"].items():
            if tool["used_by_pipeline"]:
                verdict = "usable (static check)" if tool["usable_by_static_inspection"] else tool["diagnostic"]
                print(f"  {name:<15} {verdict}")
        print("No server was started and no crystallographic program was run.")
        return 0

    app = build_app(service, settings)
    atexit.register(service.shutdown)
    app.queue(default_concurrency_limit=4)
    print(f"CrystalX Web: run records in {runs_dir}")
    try:
        app.launch(**launch_options(args.host, args.port))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        service.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
