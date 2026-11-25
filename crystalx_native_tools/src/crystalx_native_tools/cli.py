"""``crystalx-tools``: inventory, launch probe, toolchain and build plans for the owner's own tools.

    crystalx-tools inventory                     # what a run would use; reads files only
    crystalx-tools inventory --allow-translation # judge as if Rosetta 2 were allowed
    crystalx-tools probe platon_special --authorize-launch   # RUNS the program once, isolated
    crystalx-tools toolchain                     # compilers present? (nothing is executed)
    crystalx-tools build-plan platon_special SRC --licence-ack "..."   # a plan, never a build

Exit status: 0 ready / started / plan ready; 3 blocked; 4 the probe ran but the program did not
start; 2 usage errors or a refused probe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from crystalx_native_tools import __version__
from crystalx_native_tools.build import DEFAULT_BUILD_ROOT, plan_build
from crystalx_native_tools.config import ToolsConfig, default_config_path, load_config
from crystalx_native_tools.host import detect_host
from crystalx_native_tools.inventory import PIPELINE_TOOLS, Inventory, build_inventory
from crystalx_native_tools.probe import ProbeRefused, ProbeStore, probe_launch
from crystalx_native_tools.toolchain import inspect_toolchain

__all__ = ["main", "build_parser", "load_effective_config", "DEFAULT_PROBE_STORE"]

DEFAULT_PROBE_STORE = Path("runs") / "_tools" / "launch_probes.json"
ALL_TOOLS = ("shelxt", "shelxl", "shelxs", "shelxc", "shelxd", "shelxe", "anode", "ciftab", "shredcif",
             "platon_special", "platon")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crystalx-tools", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"crystalx-native-tools {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=Path, default=None, help="Tool configuration (default ./config.local.json).")
        p.add_argument("--toolbox", type=Path, default=None, help="Toolbox directory (default from config, else ./toolbox).")
        policy = p.add_mutually_exclusive_group()
        policy.add_argument("--strict-native", dest="strict", action="store_true", default=None,
                            help="Refuse x86_64-only programs on Apple Silicon (default).")
        policy.add_argument("--allow-translation", dest="strict", action="store_false",
                            help="Judge x86_64 programs as runnable through Rosetta 2.")
        p.add_argument("--probes", type=Path, default=DEFAULT_PROBE_STORE, help="Launch-probe record file.")
        p.add_argument("--json", action="store_true")

    inventory = sub.add_parser("inventory", help="Inspect every candidate; runs nothing.")
    common(inventory)
    inventory.add_argument("--all", action="store_true", help="Include the SHELX programs the pipeline does not run.")

    probe = sub.add_parser("probe", help="Start one program once, isolated, to see whether it launches.")
    common(probe)
    probe.add_argument("tool", choices=ALL_TOOLS)
    probe.add_argument("--path", type=Path, default=None, help="A specific file (default: the selected candidate).")
    probe.add_argument("--authorize-launch", action="store_true", help="Required: this runs the program.")
    probe.add_argument("--timeout", type=float, default=10.0)

    toolchain = sub.add_parser("toolchain", help="Compilers present for a local arm64 build? Nothing is executed.")
    toolchain.add_argument("--json", action="store_true")

    build = sub.add_parser("build-plan", help="Plan an isolated arm64 build from supplied source. Never builds.")
    build.add_argument("tool", choices=ALL_TOOLS)
    build.add_argument("source", type=Path, nargs="?", default=None)
    build.add_argument("--licence-ack", default=None, help="Your statement that the source's terms allow this build.")
    build.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    build.add_argument("--json", action="store_true")
    return parser


def load_effective_config(config_path: Path | None, toolbox: Path | None, strict: bool | None) -> ToolsConfig:
    path = config_path if config_path is not None else default_config_path()
    config = load_config(path)
    if toolbox is None and config.toolbox is None and Path("toolbox").is_dir():
        toolbox = Path("toolbox")
    return config.with_overrides(toolbox=toolbox, strict_native=strict)


def _print_inventory(inventory: Inventory) -> None:
    host = inventory.host
    print(f"host: {host.system} {host.hardware_arch} (python {host.python}"
          f"{', translated' if host.process_translated else ''}); macOS {host.macos_version or '-'}")
    print(f"policy: {'strict native (no Rosetta 2)' if inventory.strict_native else 'translation allowed'}")
    if inventory.config.source:
        print(f"config: {inventory.config.source}")
    for status in inventory.tools:
        selected = status.selected
        where = f"{selected.path} [{selected.origin}, {selected.native}]" if selected else "-"
        print(f"\n{status.tool:<15} static={status.static} launch={status.launch.get('status')} "
              f"validated={status.validated}")
        print(f"  role: {status.role}")
        print(f"  selected: {where}")
        for candidate in status.candidates:
            if selected is not None and candidate.path == selected.path:
                continue
            print(f"  other candidate: {candidate.path} [{candidate.origin}, {candidate.native}] "
                  f"-> {candidate.resolver_verdict}")
        for blocker in status.blockers:
            print(f"  BLOCKED ({blocker.code}): {blocker.message}")
            if blocker.next_step:
                print(f"    next: {blocker.next_step}")
        if selected:
            for warning in selected.warnings:
                print(f"  warning: {warning}")
        if status.resources is not None:
            res = status.resources
            found = [c for c in res.checkdef if c.exists]
            found_text = ", ".join(f"{c.path} (v{c.version_date or '?'})" for c in found) or "none found"
            print(f"  check.def: {found_text}; version match: {res.version_match} (heuristic)")
            for note in res.notes:
                print(f"  note: {note}")
        if status.availability is not None:
            print(f"  official arm64 build: {status.availability.native_arm64_build}; source: "
                  f"{status.availability.source_code}")
    print(f"\npipeline: {inventory.pipeline}" + (f" ({', '.join(inventory.pipeline_blockers)})" if inventory.pipeline_blockers else ""))
    for note in inventory.notes:
        print(f"note: {note}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "toolchain":
        report = inspect_toolchain(detect_host())
        if args.json:
            print(json.dumps(report.to_record(), indent=2))
        else:
            for key, value in report.to_record().items():
                print(f"{key}: {value}")
        return 0

    if args.command == "build-plan":
        plan = plan_build(args.tool, args.source, toolchain=inspect_toolchain(detect_host()),
                          licence_ack=args.licence_ack, build_root=args.build_root)
        if args.json:
            print(json.dumps(plan.to_record(), indent=2))
        else:
            print(f"plan for {plan.tool}: recipe {plan.recipe}; executed: no (a plan only)")
            for step in plan.steps:
                print(f"  step: {step.purpose}: {' '.join(step.argv)}")
            for blocker in plan.blockers:
                print(f"  BLOCKED: {blocker}")
            for note in plan.notes:
                print(f"  note: {note}")
        return 0 if plan.ready else 3

    config = load_effective_config(args.config, args.toolbox, args.strict)
    store = ProbeStore(args.probes)
    if args.command == "inventory":
        tools = ALL_TOOLS if args.all else PIPELINE_TOOLS
        inventory = build_inventory(config, tools=tools, probe_store=store)
        if args.json:
            print(json.dumps(inventory.to_record(), indent=2))
        else:
            _print_inventory(inventory)
        return 0 if inventory.pipeline == "ready_static" else 3

    # probe
    strict = True if config.strict_native is None else config.strict_native
    if args.path is not None:
        target, expected = args.path, None
    else:
        status = build_inventory(config, tools=(args.tool,), probe_store=store).get(args.tool)
        if status is None or status.selected is None:
            print(f"error: no selected candidate for {args.tool}; see 'crystalx-tools inventory'", file=sys.stderr)
            return 2
        target, expected = Path(status.selected.path), status.selected.inspection.sha256
    try:
        result = probe_launch(args.tool, target, authorize=args.authorize_launch, expected_sha256=expected,
                              allow_translation=not strict, timeout_s=args.timeout)
    except ProbeRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    store.append(result)
    if args.json:
        print(json.dumps(result.to_record(), indent=2))
    else:
        print(f"{result.tool}: {result.outcome} (exit {result.exit_code}, signal {result.signal}, "
              f"{result.duration_s} s) -- {result.message}")
        print("This shows whether the program starts, not whether it computes correctly.")
    return 0 if result.started else 4


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
