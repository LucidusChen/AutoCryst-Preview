"""Offline private acceptance CLI. Deliberately contains no launch/probe subcommand."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from crystalx_core.errors import CrystalXError
from crystalx_core.app.models import RunConfig

from crystalx_core.private_acceptance.grading import (
    GradingRequest, ScientificThresholds, grade_sealed_run, save_grading_report,
)
from crystalx_core.private_acceptance.harness import prepare_plan, private_directory, private_output_path
from crystalx_core.private_acceptance.privacy import audit_repository
from crystalx_core.private_acceptance.sealing import verify_seal, write_private_json
from crystalx_core.private_acceptance.specs import load_grader_spec, synthetic_solver_template


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local offline private acceptance; never launches proprietary tools")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("template", help="print a hand-authored synthetic solver contract")
    plan = sub.add_parser("plan", help="save a private, bounded plan; BLOCKED exits 3")
    plan.add_argument("--solver-spec", type=Path, required=True)
    plan.add_argument("--case-root", type=Path, required=True)
    plan.add_argument("--private-root", type=Path)
    plan.add_argument("--checkdef", type=Path)
    plan.add_argument("--toolbox", type=Path, help="explicit local search directory; resolved files are inspected and pinned")
    privacy = sub.add_parser("privacy-audit", help="save a private Git/content review report")
    privacy.add_argument("--private-root", type=Path)
    grade = sub.add_parser("grade", help="verify an externally pinned seal before independent grading")
    grade.add_argument("--run", type=Path, required=True)
    grade.add_argument("--seal-sha256", required=True)
    grade.add_argument("--grader-spec", type=Path, required=True)
    grade.add_argument("--case-root", type=Path, required=True)
    grade.add_argument("--private-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "template":
        print(json.dumps(synthetic_solver_template(), indent=2))
        return 0
    try:
        repository = args.repository.resolve(strict=True)
        root = args.private_root or repository / "runs" / "private_acceptance"
        base = private_directory(repository, root)
        if args.command == "plan":
            config = RunConfig(shelxt_timeout_s=900, shelxl_timeout_s=600,
                               platon_timeout_s=600, shelxl_cycles=4, cpu_seconds=3600,
                               strict_native=True, toolbox_dir=args.toolbox)
            path, _digest = prepare_plan(args.solver_spec, case_root=args.case_root,
                                        repository=repository, private_root=base,
                                        checkdef_path=args.checkdef, config=config)
            print(f"BLOCKED: offline plan saved at {path}")
            print("Real invocations: 0. Review the private plan and its pinned digest locally.")
            return 3
        if args.command == "privacy-audit":
            report = audit_repository(repository)
            folder = base / ("privacy-" + uuid.uuid4().hex)
            folder.mkdir(mode=0o700)
            private_output_path(repository, folder / "audit.json")
            write_private_json(folder, private_root=base, relative="audit.json", payload=report)
            print(f"{report['status']}: local report saved at {folder / 'audit.json'}")
            return 3 if report["status"] == "REVIEW_REQUIRED" else 0
        # A pin is verified before even resolving grader paths. No provider is supplied by
        # this CLI: absence of audited crystallographic matching remains NOT_EVALUATED.
        verify_seal(args.run, private_root=base, expected_seal_sha256=args.seal_sha256)
        spec = load_grader_spec(args.grader_spec, case_root=args.case_root)
        thresholds = None
        if spec.site_match_tolerance_angstrom is not None:
            thresholds = ScientificThresholds(
                spec.site_match_tolerance_angstrom, spec.heavy_atomic_number_min,
                spec.thresholds_owner_approved, spec.thresholds_approval_reference,
            )
        request = GradingRequest(spec.case_id, spec.track.value, spec.case_root,
                                 spec.answer_path, spec.reference_paths)
        report = grade_sealed_run(args.run, private_root=base, specification=request,
                                 expected_seal_sha256=args.seal_sha256, thresholds=thresholds)
        filename = "grading-" + uuid.uuid4().hex + ".json"
        private_output_path(repository, args.run / filename)
        save_grading_report(report, args.run, private_root=base, relative=filename)
        print(report.status.value + ": see the private grading record")
        return 3
    except (OSError, ValueError, CrystalXError) as exc:
        # Parser/OS messages may contain scientific values, paths, or excerpts. Keep errors
        # generic on shared stdout; inspect the explicit local inputs when diagnosis is needed.
        print(f"BLOCKED: {type(exc).__name__}; no proprietary program was started", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
