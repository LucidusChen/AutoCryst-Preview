"""``crystalx solve``: the command-line client of the application service."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

from crystalx_core.app import (
    TERMINAL_PHASES,
    ChemicalPriorInput,
    RequestInvalid,
    RunConfig,
    RunMode,
    RunPhase,
    RunService,
    SolveRequest,
    write_demo_case,
)

__all__ = ["run_solve", "build_request"]

_EXIT = {
    RunPhase.COMPLETED.value: 0,
    RunPhase.FAILED.value: 1,
    RunPhase.BLOCKED.value: 1,
    RunPhase.INTERRUPTED.value: 1,
    RunPhase.CANCELLED.value: 130,
}


def build_request(args: argparse.Namespace) -> SolveRequest:
    """Turn parsed arguments into a request. Raises ``ValueError`` with a message for the user."""
    composition = ChemicalPriorInput.parse_composition(args.composition or "")
    metals = tuple(args.metal)
    if args.demo:
        if args.ins or args.hkl:
            raise ValueError("--demo runs the built-in synthetic case; do not pass --ins/--hkl with it")
        ins, hkl = write_demo_case(Path(args.runs_dir) / "_demo_input")
        mode, label = RunMode.MOCK_DEMO, "demo"
        metals = metals or ("Zn",)
    else:
        if not args.ins or not args.hkl:
            raise ValueError("pass --ins and --hkl, or --demo for the synthetic demo")
        ins, hkl = Path(args.ins), Path(args.hkl)
        mode, label = RunMode.REAL, Path(args.ins).stem
    config = RunConfig(
        shelxt_timeout_s=args.shelxt_timeout,
        shelxl_cycles=args.shelxl_cycles,
        shelxl_timeout_s=args.shelxl_timeout,
        platon_timeout_s=args.platon_timeout,
        cpu_seconds=args.cpu_limit,
        toolbox_dir=Path(os.path.abspath(args.toolbox)) if args.toolbox else None,
        strict_native=bool(getattr(args, "strict_native", False)),
    )
    return SolveRequest(
        ins_path=ins,
        hkl_path=hkl,
        prior=ChemicalPriorInput(
            metals=metals,
            elements=tuple(args.element),
            peptide=args.peptide,
            racemic=args.racemic,
            composition=composition,
        ),
        config=config,
        mode=mode,
        label=label,
    )


def run_solve(args: argparse.Namespace) -> int:
    try:
        request = build_request(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    service = RunService(args.runs_dir)
    try:
        if args.preview:
            return _preview(service, request, as_json=args.json)
        handle = service.submit(request)
    except RequestInvalid as exc:
        for issue in exc.issues:
            print(f"error: [{issue.code}] {issue.field}: {issue.message}", file=sys.stderr)
        return 2

    if request.mode is RunMode.MOCK_DEMO:
        print("MOCK DEMO: synthetic data and stand-in programs. Nothing here is a real result.")
    print(f"run {handle.run_id}")
    print(f"record: {handle.run_dir}")
    seen = 0
    try:
        while True:
            if not args.quiet:
                for event in service.events(handle.run_id, after=seen):
                    seen = event["index"] + 1
                    line = _format_event(event)
                    if line:
                        print(line, flush=True)
            state = service.status(handle.run_id)
            if state.get("phase") in {p.value for p in TERMINAL_PHASES}:
                break
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\ncancelling: stopping the running program and keeping its logs ...", flush=True)
        service.cancel(handle.run_id, "interrupted from the keyboard")
        state = service.wait(handle.run_id, timeout=30)

    summary = service.summary(handle.run_id) or {}
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        _print_summary(summary, handle.run_dir)
    return _EXIT.get(state.get("phase", ""), 1)


def _preview(service: RunService, request: SolveRequest, *, as_json: bool) -> int:
    preview = service.preview(request)
    if as_json:
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return 0 if preview["preconditions"]["may_start"] else 2
    diagnosis = preview["diagnosis"]
    cell = diagnosis["cell"]
    if cell:
        print(
            f"cell      : {cell['a']:.4f} {cell['b']:.4f} {cell['c']:.4f} "
            f"{cell['alpha']:.3f} {cell['beta']:.3f} {cell['gamma']:.3f}  V={cell['volume']} A^3"
        )
    print(f"symmetry  : {diagnosis['space_group']['summary']} -- {diagnosis['space_group']['explanation']}")
    print(f"SFAC/UNIT : {' '.join(diagnosis['composition']['sfac'])} / "
          f"{' '.join(f'{u:g}' for u in diagnosis['composition']['unit'])}")
    refl = diagnosis["reflections"]
    d_min = f"{refl['d_min_a']:.2f} A" if refl["d_min_a"] else "unknown"
    print(f"reflections: {refl['count']} (HKLF {refl['hklf']}, {refl['anomalous']}, d_min {d_min})")
    for warning in diagnosis["warnings"]:
        print(f"  warning: {warning}")
    print(f"preconditions: {preview['preconditions']['summary']}")
    for item in preview["plan"]["nominated"]:
        verdict = "runs" if item["will_run"] else f"not run ({item['readiness']})"
        print(f"  route {item['solver']:<16} {verdict}")
    for item in preview["plan"]["excluded"]:
        print(f"  route {item['solver']:<16} excluded: {item['reason']}")
    for step in preview["plan"]["downstream"]:
        print(f"  then  {step['step']:<16} {step['description']}")
    return 0 if preview["preconditions"]["may_start"] else 2


def _format_event(event: dict) -> str:
    kind = event.get("kind", "")
    stamp = event.get("ts", "")
    try:
        stamp = _dt.datetime.fromisoformat(stamp).strftime("%H:%M:%S")
    except ValueError:
        pass
    message = event.get("message", "")
    if kind == "app.stage":
        return f"[{stamp}] {event.get('stage', ''):<13} {event.get('state', ''):<16} {message}"
    if kind == "warning":
        return f"[{stamp}] warning: {message}"
    if kind == "command.started":
        return f"[{stamp}] exec: {message}"
    if kind == "command.finished":
        return f"[{stamp}] exit: {event.get('diagnostic', '')} ({event.get('duration_s', '?')} s)"
    if kind == "app.phase":
        return f"[{stamp}] run: {event.get('phase', '')} -- {message}"
    return ""


def _fmt(value, spec: str = "") -> str:
    if value is None:
        return "unknown"
    return format(value, spec) if spec else str(value)


def _print_summary(summary: dict, run_dir: Path) -> None:
    print()
    if summary.get("mock"):
        print("== MOCK DEMO -- synthetic data and stand-in programs; not a real result ==")
    process = summary.get("process", {})
    print(f"phase              : {process.get('phase')} -- {process.get('message')}")
    for stage in summary.get("stages", []):
        print(f"  {stage['name']:<13} {stage['state']:<16} {stage.get('message', '')}")
    print(f"model availability : {summary.get('model_availability')}")
    print(f"refinement status  : {summary.get('refinement_status')}")
    refinement = summary.get("refinement") or {}
    listing = refinement.get("listing") or {}
    if listing:
        print(
            f"  R1(obs) {_fmt(listing.get('r1_observed'))}  wR2 {_fmt(listing.get('wr2'))}  "
            f"GooF {_fmt(listing.get('goof'))}  shift/su mean {_fmt(listing.get('mean_shift_su'))} "
            f"max {_fmt(listing.get('max_shift_su'))} (convergence not assessed)  "
            f"peak/hole {_fmt(listing.get('highest_peak'))}/{_fmt(listing.get('deepest_hole'))}"
        )
    platon = summary.get("platon") or {}
    if platon.get("report"):
        counts = platon["report"]["counts"]
        print("PLATON alerts      : " + "  ".join(f"{k} {counts[k]}" for k in "ABCG"))
    elif platon:
        print(f"PLATON             : {platon.get('reason') or ' '.join(platon.get('reasons', []))}")
    eligibility = summary.get("scientific_eligibility", {})
    print(f"eligibility        : {eligibility.get('status')} -- {eligibility.get('reason')}")
    for warning in summary.get("warnings", []):
        print(f"  warning: {warning}")
    print(f"run directory      : {run_dir}")
