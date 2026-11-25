"""The ``crystalx`` command.

Ships before any GUI, on purpose: the CLI is the reference client, and every interface -- Swift on
macOS, the Gradio development UI, a later PySide6 client -- consumes the same core API and the same
run record.

``doctor`` reads header bytes and ``plan`` prints the argument arrays that *would* be run; neither
executes anything. ``solve`` is the one command that runs programs -- through the same application
service the Web interface uses (:mod:`crystalx_core.app`), with timeouts, resource limits and a full
run record. ``solve --demo`` runs the synthetic demo on the mock toolchain and touches no real
program.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from crystalx_core.domain.chemistry import ChemicalPrior, MetalPrior
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.radiation import (
    DatasetFingerprint,
    RadiationAssumption,
    assess_merge_compatibility,
)
from crystalx_core.domain.ids import make_stage_id
from crystalx_core.domain.onion import StageKind
from crystalx_core.domain.tools import ToolDiagnosticKind, ToolKind
from crystalx_core.errors import CrystalXError
from crystalx_core.io.shelx_reader import read_ins_header, scan_hkl
from crystalx_core.domain.solving import SolverKind
from crystalx_core.workflow.preconditions import check_preconditions
from crystalx_core.solvers.selector import SelectionPolicy, StrategySelector, summarise_trace
from crystalx_core.tools.resolver import ResolverConfig, ToolResolver
from crystalx_core.version import CORE_VERSION
from crystalx_core.workflow.ranking import describe_ordering

__all__ = ["main", "build_parser"]

_STATUS_MARK = {
    ToolDiagnosticKind.OK: "ok",
    ToolDiagnosticKind.OK_VIA_TRANSLATION: "Rosetta marker",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crystalx",
        description=(
            "AutoCryst platform-neutral core. Resolves local SHELX/PLATON installations, "
            "diagnoses diffraction data, and plans an initial-solution portfolio."
        ),
    )
    parser.add_argument("--version", action="version", version=f"crystalx-core {CORE_VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser(
        "doctor",
        help="Report the host platform and which crystallographic programs are usable.",
    )
    doctor.add_argument("--toolbox", type=Path, default=None, help="Project toolbox directory.")
    doctor.add_argument("--json", action="store_true", help="Machine-readable output.")

    diagnose = sub.add_parser(
        "diagnose", help="Describe a reflection file and an instruction file."
    )
    diagnose.add_argument("--hkl", type=Path, required=True)
    diagnose.add_argument("--ins", type=Path, required=True)
    diagnose.add_argument("--json", action="store_true")

    plan = sub.add_parser(
        "plan",
        help="Choose the initial-solution portfolio and explain why, without running anything.",
    )
    plan.add_argument("--hkl", type=Path, required=True)
    plan.add_argument("--ins", type=Path, required=True)
    plan.add_argument(
        "--metal",
        action="append",
        default=[],
        metavar="ELEMENT",
        help="Declare an expected metal, e.g. --metal Ag. May be repeated.",
    )
    plan.add_argument(
        "--element",
        action="append",
        default=[],
        metavar="SYMBOL",
        help=(
            "Declare a non-metal element you expect, e.g. --element S --element F for an NTf2 "
            "anion. A solution program places atoms only for elements it was told about, so an "
            "element missing from SFAC simply will not appear. May be repeated."
        ),
    )
    plan.add_argument(
        "--peptide",
        action="store_true",
        help="Declare a peptide-like system. Only this enables SHELXE main-chain autotracing.",
    )
    plan.add_argument(
        "--racemic",
        action="store_true",
        help=(
            "Declare the preparation racemic or meso. Lifts the chirality block, which otherwise "
            "refuses a space group containing an improper operation for a homochiral material."
        ),
    )
    plan.add_argument(
        "--force-solver",
        action="append",
        default=[],
        choices=[k.value for k in SolverKind],
        help="Override the selector and always nominate this route. Recorded as an override.",
    )
    plan.add_argument(
        "--exclude-solver",
        action="append",
        default=[],
        choices=[k.value for k in SolverKind],
        help="Override the selector and never nominate this route. Recorded as an override.",
    )
    plan.add_argument("--toolbox", type=Path, default=None)
    plan.add_argument("--json", action="store_true")

    merge = sub.add_parser(
        "merge-check",
        help="Decide whether several datasets may be merged. Refuses mixed radiation sources.",
    )
    merge.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("INS", "HKL"),
        required=True,
        help="An instruction file and its reflection file. Repeat for each dataset.",
    )
    merge.add_argument("--json", action="store_true")

    sub.add_parser("policy", help="Print the candidate ordering policy.")

    solve = sub.add_parser(
        "solve",
        help=(
            "Run a preliminary solve: SHELXT, a bounded isotropic SHELXL refinement and PLATON "
            "diagnostics, recorded in an immutable run directory."
        ),
        description=(
            "Runs the same application service as the Web interface. The result is a "
            "preliminary, unvalidated model: it can be inspected and exported, never ranked."
        ),
    )
    solve.add_argument("--ins", type=Path, help="Instruction file (.ins or .res).")
    solve.add_argument("--hkl", type=Path, help="Reflection file (.hkl).")
    solve.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run the built-in synthetic demo case with the mock toolchain. No real program runs; "
            "every output is marked MOCK."
        ),
    )
    solve.add_argument("--metal", action="append", default=[], metavar="ELEMENT")
    solve.add_argument("--element", action="append", default=[], metavar="SYMBOL")
    solve.add_argument("--peptide", action="store_true")
    solve.add_argument("--racemic", action="store_true")
    solve.add_argument(
        "--composition",
        default="",
        metavar="'EL N EL N ...'",
        help=(
            "Replace SFAC/UNIT in the SHELXT input copy, e.g. 'C 40 H 52 N 4 O 8 Zn 2'. "
            "Recorded line by line as an override; the original file is never changed."
        ),
    )
    solve.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Default: ./runs")
    solve.add_argument("--toolbox", type=Path, default=None)
    solve.add_argument(
        "--strict-native",
        action="store_true",
        help=(
            "Refuse x86_64-only executables on Apple Silicon instead of planning them through "
            "Rosetta 2; the search continues to native arm64 or universal builds."
        ),
    )
    solve.add_argument("--shelxt-timeout", type=float, default=3600.0, metavar="SECONDS")
    solve.add_argument("--shelxl-cycles", type=int, default=4)
    solve.add_argument("--shelxl-timeout", type=float, default=900.0, metavar="SECONDS")
    solve.add_argument("--platon-timeout", type=float, default=600.0, metavar="SECONDS")
    solve.add_argument(
        "--cpu-limit",
        type=int,
        default=None,
        metavar="SECONDS",
        help=(
            "CPU-time cap per program. Default: the program's wall-clock budget times the number "
            "of logical CPUs, because SHELXT is multi-threaded."
        ),
    )
    solve.add_argument(
        "--preview", action="store_true", help="Diagnose and plan only; start nothing."
    )
    solve.add_argument("--json", action="store_true", help="Print the final summary as JSON.")
    solve.add_argument("--quiet", action="store_true", help="Do not stream progress events.")

    verify = sub.add_parser(
        "verify-run",
        help="Check a finished run record for real-tool acceptance. Starts nothing, writes nothing.",
        description=(
            "Exit 0 only if every required stage ran on real programs, exited 0, still matches its "
            "recorded executable digest and left the files the next step needs. A run that merely "
            "completed (for example with PLATON blocked) is not accepted. Never reads a reference "
            "structure."
        ),
    )
    verify.add_argument("run_dir", type=Path, help="The run directory (runs/r-...).")
    verify.add_argument("--json", action="store_true", help="Print the verdict as JSON.")
    verify.add_argument(
        "--expect-input",
        action="append",
        default=[],
        metavar="ROLE=SHA256",
        help="Require the run to have read this input, e.g. ins=<sha256>; repeatable.",
    )
    verify.add_argument(
        "--allow-mock",
        action="store_true",
        help="Do not require real programs (for testing the checker on the demo only).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "diagnose":
            return _cmd_diagnose(args)
        if args.command == "plan":
            return _cmd_plan(args)
        if args.command == "merge-check":
            return _cmd_merge_check(args)
        if args.command == "policy":
            print(describe_ordering())
            return 0
        if args.command == "solve":
            from crystalx_core.cli.solve import run_solve

            return run_solve(args)
        if args.command == "verify-run":
            return _cmd_verify_run(args)
    except CrystalXError as exc:
        print(f"error: {exc}", file=sys.stderr)
        for path in exc.evidence_paths:
            print(f"  evidence: {path}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command!r}")
    return 2


def _cmd_verify_run(args: argparse.Namespace) -> int:
    from crystalx_core.app.acceptance import verify_run

    if not args.run_dir.is_dir():
        print(f"error: {args.run_dir} is not a directory", file=sys.stderr)
        return 2
    expected: dict[str, str] = {}
    for item in args.expect_input:
        role, sep, digest = item.partition("=")
        if not sep or role not in ("ins", "hkl") or len(digest) != 64:
            print(f"error: --expect-input wants ins=<sha256> or hkl=<sha256>, not {item!r}", file=sys.stderr)
            return 2
        expected[role] = digest.lower()
    verdict = verify_run(args.run_dir, require_real=not args.allow_mock, expected_inputs=expected)
    if args.json:
        print(json.dumps(verdict.to_record(), indent=2, ensure_ascii=False))
    else:
        for check in verdict.checks:
            print(f"{'PASS' if check.passed else 'FAIL'}  {check.name:<18} {check.detail}")
        for item in verdict.open_items:
            print(f"open  {item}")
        if verdict.accepted and args.allow_mock:
            print("CHECKER TEST: PASS with --allow-mock -- every stage ran and left its files; "
                  "this is NOT real-tool acceptance")
        elif verdict.accepted:
            print("ACCEPTANCE: PASS -- every required stage ran on real programs and left its files "
                  "(mechanical check; not a judgement of the structure)")
        else:
            print(f"ACCEPTANCE: FAIL -- {len(verdict.failed)} of {len(verdict.checks)} checks failed")
    return 0 if verdict.accepted else 1


def _make_resolver(toolbox: Path | None) -> ToolResolver:
    return ToolResolver(ResolverConfig(toolbox_dir=toolbox))


def _cmd_doctor(args: argparse.Namespace) -> int:
    resolver = _make_resolver(args.toolbox)
    report = resolver.resolve_all()

    if args.json:
        payload = {
            "host": {
                "system": report.host.system,
                "arch": report.host.arch.value,
                "rosetta_available": report.host.rosetta_available,
            },
            "tools": {
                name: {
                    "usable": res.is_usable,
                    "diagnostic": res.diagnostic.kind.value,
                    "message": res.diagnostic.message,
                    "remedy": res.diagnostic.remedy,
                    "path": str(res.identity.path) if res.identity else None,
                    "arch": res.identity.arch.value if res.identity else None,
                    "sha256": res.identity.sha256 if res.identity else None,
                    "source": res.identity.source.value if res.identity else None,
                }
                for name, res in sorted(report.resolutions.items())
            },
        }
        print(json.dumps(payload, indent=2))
        return 0

    host = report.host
    print(f"host        : {host.system} {host.arch.value}")
    if host.rosetta_available is not None:
        print(
            "rosetta 2   : "
            + (
                "marker found (actual x86_64 launch is verified only by an opt-in tool run)"
                if host.rosetta_available
                else "NOT detected"
            )
        )
    print()
    for name, resolution in sorted(report.resolutions.items()):
        if resolution.is_usable and resolution.identity is not None:
            mark = _STATUS_MARK.get(resolution.diagnostic.kind, "ok")
            print(
                f"  [{mark:>12}] {name:<16} {resolution.identity.arch.value:<10} "
                f"{resolution.identity.path}"
            )
            print(f"                 source={resolution.identity.source.value} "
                  f"sha256={resolution.identity.sha256[:16]}")
        else:
            print(f"  [{resolution.diagnostic.kind.value:>12}] {name}")
            print(f"                 {resolution.diagnostic.message}")
            if resolution.diagnostic.remedy:
                for line in resolution.diagnostic.remedy.splitlines():
                    print(f"                 {line}")
    print()
    platon = resolver.resolve_platon()
    if platon.is_usable and platon.identity is not None:
        print(f"PLATON for automation: {platon.identity.kind.value} at {platon.identity.path}")
    else:
        print(f"PLATON for automation: unavailable ({platon.diagnostic.message})")
    return 0


def _load_diagnosis(hkl: Path, ins: Path) -> DataDiagnosis:
    header = read_ins_header(ins)
    reflections = scan_hkl(hkl, cell=header.cell, hklf_hint=header.hklf_format)
    return DataDiagnosis(reflections=reflections, ins=header)


def _cmd_diagnose(args: argparse.Namespace) -> int:
    diagnosis = _load_diagnosis(args.hkl, args.ins)
    reflections = diagnosis.reflections
    header = diagnosis.ins

    if args.json:
        from crystalx_core._serde import to_jsonable

        print(json.dumps(to_jsonable(diagnosis), indent=2))
        return 0

    print(f"title            : {header.title or '(none)'}")
    if header.cell:
        cell = header.cell
        print(
            f"cell             : {cell.a:.4f} {cell.b:.4f} {cell.c:.4f} "
            f"{cell.alpha:.3f} {cell.beta:.3f} {cell.gamma:.3f}  V={cell.volume:.1f} A^3"
        )
    print(f"wavelength       : {header.wavelength_a if header.wavelength_a else 'unknown'} A")
    print(f"space group      : {header.space_group_confidence.value} "
          f"(LATT={header.latt}, {len(header.symm)} SYMM)")
    print(f"composition      : SFAC {' '.join(header.sfac) or '(none)'}")
    print(f"                   UNIT {' '.join(f'{u:g}' for u in header.unit) or '(none)'}")
    print()
    print(f"reflections      : {reflections.n_reflections}")
    print(f"HKLF             : {int(reflections.hklf_format)}")
    print(f"anomalous        : {reflections.anomalous.value} "
          f"({reflections.n_friedel_pairs} Friedel pairs)")
    if reflections.d_min_a is not None:
        print(f"resolution       : {reflections.d_max_a:.2f} - {reflections.d_min_a:.2f} A")
    if reflections.mean_i_over_sigma is not None:
        print(f"mean I/sigma     : {reflections.mean_i_over_sigma:.1f}")
    if reflections.completeness_unavailable is not None:
        print(f"completeness     : not computed - {reflections.completeness_unavailable.reason}")
    for warning in (*header.warnings, *reflections.warnings):
        print(f"  warning: {warning}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    diagnosis = _load_diagnosis(args.hkl, args.ins)
    prior = ChemicalPrior(
        metals=tuple(MetalPrior(element=m) for m in args.metal),
        declared_elements=tuple(args.element),
        is_peptide_like=args.peptide,
        enantiopure=False if args.racemic else None,
    )
    report = check_preconditions(diagnosis, prior)
    for precondition in report.preconditions:
        print("BLOCKED" if precondition.blocking else "NOTE")
        print(precondition.render())
        print()
    if not report.may_start:
        print("Refusing to plan a solve that cannot produce a correct answer.")
        return 2

    availability = None
    if args.toolbox is not None:
        availability = _make_resolver(args.toolbox).resolve_all()

    selector = StrategySelector(
        SelectionPolicy(
            forced_kinds=tuple(SolverKind(k) for k in args.force_solver),
            excluded_kinds=tuple(SolverKind(k) for k in args.exclude_solver),
        )
    )
    stage_id = make_stage_id(1, StageKind.INITIAL_SOLUTION.value)
    trace = selector.select(diagnosis, prior, stage_id=stage_id, availability=availability)

    if args.json:
        from crystalx_core._serde import to_jsonable

        print(json.dumps(to_jsonable(trace), indent=2))
        return 0

    for line in summarise_trace(trace):
        print(line)
    if not trace.nominated:
        print()
        print("No initial-solution branch is applicable. That is a reportable result, not a crash.")
        return 1
    return 0


def _cmd_merge_check(args: argparse.Namespace) -> int:
    """Refuse to merge datasets that do not describe the same experiment.

    The case this exists for: someone drops a molybdenum dataset in among copper ones. The
    wavelength changes the dispersion corrections, the absorption and the accessible resolution, so
    the merge is meaningless -- and it would look fine. Exit code 1 means the merge is refused.
    """
    fingerprints = []
    for ins_path, hkl_path in args.dataset:
        diagnosis = _load_diagnosis(Path(hkl_path), Path(ins_path))
        header = diagnosis.ins
        assumption = RadiationAssumption.from_wavelength(header.wavelength_a)
        cell = header.cell
        fingerprints.append(
            DatasetFingerprint(
                label=Path(ins_path).stem,
                sha256=diagnosis.reflections.sha256,
                radiation=assumption.source,
                wavelength_a=header.wavelength_a,
                cell=(
                    (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma) if cell else None
                ),
                hklf=int(diagnosis.reflections.hklf_format),
                latt=header.latt,
                n_symm=len(header.symm),
                has_unmerged_friedel_pairs=diagnosis.reflections.has_unmerged_friedel_pairs,
                d_min_a=diagnosis.reflections.d_min_a,
            )
        )

    report = assess_merge_compatibility(fingerprints)

    if args.json:
        from crystalx_core._serde import to_jsonable

        print(json.dumps(to_jsonable(report), indent=2))
        return 0 if report.may_merge else 1

    for fingerprint in report.fingerprints:
        wavelength = (
            f"{fingerprint.wavelength_a:.5f} A" if fingerprint.wavelength_a else "unknown"
        )
        print(f"  {fingerprint.label:<20} {fingerprint.radiation.value:<10} {wavelength}")
    print()
    if report.may_merge:
        print(f"MERGE ALLOWED: {report.summary()}")
    else:
        print("MERGE REFUSED")
        for blocker in report.blockers:
            print(f"  - [{blocker.kind.value}] {blocker.message}")
    for warning in report.warnings:
        print(f"  ! [{warning.kind.value}] {warning.message}")
    for note in report.notes:
        print(f"  note: {note}")
    return 0 if report.may_merge else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
