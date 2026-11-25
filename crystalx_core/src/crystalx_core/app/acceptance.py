"""Mechanical acceptance check of a finished run record (``crystalx verify-run``).

A run that *completed* is not an accepted run. The service ends a run ``completed`` when, for
example, PLATON was blocked because SHELXL wrote no FCF: that is an honest record of a partial
result, and the exit status of ``crystalx solve`` reflects only the phase. Accepting the real-tool
workflow needs more, and this module checks it from the record alone:

* the run used real programs (not the demo toolchain), and every stage succeeded;
* each program has a provenance record, exited 0, and is still the file whose digest was recorded;
* every file a later step depends on exists and still has the digest recorded when it was written;
* the reflections the refinement used follow the shared contract in
  :mod:`crystalx_core.app.reflections` -- SHELXT's matching ``.hkl``, or the original reflections
  under the evidenced "Orientation as input" fallback -- with the digest chain intact;
* the refinement listing reported no errors, and a refined model and a PLATON report exist;
* the retained staged copies of the inputs are re-hashed *now* and still match the recorded digests
  (current archive integrity), separately from the run's own end-of-run observation (historical).
  The original upload path is not required to exist any more: the staged copy is the durable
  evidence.

It starts no program and writes nothing. It never opens a reference ("answer") structure: scoring
against one is a separate step for the reviewer, after the run. Passing here means the pipeline ran
end to end on real programs; it says nothing about whether the structure is right.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from crystalx_core.app.models import (
    STAGE_ORDER,
    ModelAvailability,
    RefinementStatus,
    RunMode,
    RunPhase,
    StageState,
)
from crystalx_core.app.reflections import (
    ReflectionSource,
    decide_from_files,
    input_reflections_name,
    required_shelxt_outputs,
    solution_model_name,
    solution_reflections_name,
)
from crystalx_core.workspace.hashing import sha256_file

__all__ = ["AcceptanceCheck", "AcceptanceVerdict", "verify_run", "REQUIRED_OUTPUTS"]

SCHEMA = "crystalx.acceptance/1"

#: Files SHELXL and PLATON must leave in their working directories, by stem: PLATON reads SHELXL's
#: CIF and FCF. What SHELXT must have written depends on the recorded reflection source and comes
#: from :func:`crystalx_core.app.reflections.required_shelxt_outputs` (the chosen solution's
#: ``.res`` always; its matching ``.hkl`` unless the evidenced fallback was used).
REQUIRED_OUTPUTS: dict[str, tuple[str, ...]] = {
    "shelxl": ("{stem}.res", "{stem}.lst", "{stem}.cif", "{stem}.fcf"),
    "platon": ("{stem}.chk",),
}

_INPUT_ROLES = ("ins", "hkl")

_PROVENANCE_GLOBS = {
    "shelxt": "stages/*/branches/*-shelxt/provenance_*.json",
    "shelxl": "refinement/shelxl-01/provenance.json",
    "platon": "validation/platon-01/provenance.json",
}


@dataclass(frozen=True)
class AcceptanceCheck:
    name: str
    passed: bool
    detail: str

    def to_record(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class AcceptanceVerdict:
    run_dir: Path
    checks: tuple[AcceptanceCheck, ...]
    open_items: tuple[str, ...] = ()
    """What the reviewer still has to do or look at. Not failures; never a reason to pass either."""

    require_real: bool = True

    @property
    def accepted(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def failed(self) -> tuple[AcceptanceCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def to_record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "run_dir": str(self.run_dir),
            "accepted": self.accepted,
            "require_real_programs": self.require_real,
            "meaning": (
                "accepted = every required stage ran on real programs and produced its files; "
                "not a statement that the structure is correct"
            ),
            "checks": [check.to_record() for check in self.checks],
            "open_items": list(self.open_items),
            "checked_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }


def _load(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"{path.name} is missing"
    except (OSError, ValueError) as exc:
        return None, f"{path.name} is unreadable: {exc}"
    if not isinstance(value, dict):
        return None, f"{path.name} is not a JSON object"
    return value, ""


def verify_run(
    run_dir: Path,
    *,
    require_real: bool = True,
    expected_inputs: Mapping[str, str] | None = None,
) -> AcceptanceVerdict:
    """Check one run directory.

    ``expected_inputs`` maps an input role (``ins``, ``hkl``) to the SHA-256 the run must have
    read, tying the record to the dataset under test. ``require_real=False`` exists only for tests
    of the checker itself.
    """
    run_dir = Path(run_dir)
    state, state_problem = _load(run_dir / "state.json")
    summary, summary_problem = _load(run_dir / "summary.json")
    if state is None or summary is None:
        detail = "; ".join(p for p in (state_problem, summary_problem) if p)
        return AcceptanceVerdict(
            run_dir, (AcceptanceCheck("record_readable", False, detail),), require_real=require_real
        )

    checks: list[AcceptanceCheck] = [AcceptanceCheck("record_readable", True, "state and summary read")]
    open_items: list[str] = []
    stem = str(summary.get("stem") or "")

    mode, mock = summary.get("mode"), summary.get("mock")
    if require_real:
        real = mode == RunMode.REAL.value and mock is False
        checks.append(
            AcceptanceCheck(
                "real_programs",
                real,
                "real programs" if real else f"mode {mode!r}, mock {mock!r}: not real-tool evidence",
            )
        )

    phase = state.get("phase")
    checks.append(
        AcceptanceCheck("phase_completed", phase == RunPhase.COMPLETED.value, f"phase {phase!r}")
    )

    stage_states = {s.get("name"): s for s in state.get("stages", []) if isinstance(s, dict)}
    not_succeeded = []
    for name in (stage.value for stage in STAGE_ORDER):
        entry = stage_states.get(name)
        if entry is None:
            not_succeeded.append(f"{name}: not recorded")
        elif entry.get("state") != StageState.SUCCEEDED.value:
            not_succeeded.append(f"{name}: {entry.get('state')} ({entry.get('message', '')})")
    checks.append(
        AcceptanceCheck(
            "stages_succeeded",
            not not_succeeded,
            "all stages succeeded" if not not_succeeded else "; ".join(not_succeeded),
        )
    )

    availability = summary.get("model_availability")
    checks.append(
        AcceptanceCheck(
            "refined_model",
            availability == ModelAvailability.PRELIMINARY_REFINED_MODEL.value,
            f"model availability {availability!r}",
        )
    )

    refinement = summary.get("refinement") or {}
    listing = refinement.get("listing") or {}
    status = summary.get("refinement_status")
    errors = list(listing.get("errors") or [])
    clean = status == RefinementStatus.RAN_CONVERGENCE_NOT_ASSESSED.value and not errors
    detail = f"refinement status {status!r}"
    if errors:
        detail += "; listing errors: " + "; ".join(str(e) for e in errors)
    checks.append(AcceptanceCheck("refinement_clean", clean, detail))

    checks.append(_check_inputs_at_run_end(summary))
    staged_check, current_inputs = _check_staged_inputs(run_dir, summary)
    checks.append(staged_check)

    if expected_inputs:
        recorded = summary.get("inputs") or {}
        mismatched = []
        for role, digest in sorted(expected_inputs.items()):
            at_record = ((recorded.get(role) or {}) if isinstance(recorded, dict) else {}).get("sha256")
            now = current_inputs.get(role)
            if at_record != digest or now != digest:
                mismatched.append(
                    f"{role}: expected {digest[:16]}, recorded {str(at_record)[:16]}, "
                    f"staged copy now {str(now)[:16]}"
                )
        checks.append(
            AcceptanceCheck(
                "inputs_expected",
                not mismatched,
                "the retained staged copies are the expected files (re-hashed now)"
                if not mismatched
                else "; ".join(mismatched),
            )
        )

    lineage = refinement.get("reflections_lineage") if isinstance(refinement, dict) else None
    lineage = lineage if isinstance(lineage, dict) else {}
    letter = lineage.get("solution_letter")
    letter = letter if isinstance(letter, str) and re.fullmatch(r"[a-z]", letter) else "a"
    required = {
        "shelxt": required_shelxt_outputs(lineage.get("source"), stem, letter),
        "shelxl": tuple(pattern.format(stem=stem) for pattern in REQUIRED_OUTPUTS["shelxl"]),
        "platon": tuple(pattern.format(stem=stem) for pattern in REQUIRED_OUTPUTS["platon"]),
    }

    for tool in ("shelxt", "shelxl", "platon"):
        check, notes = _check_program(run_dir, tool, required[tool], require_real)
        checks.append(check)
        open_items.extend(notes)

    lineage_check, lineage_notes = _check_reflection_lineage(run_dir, summary, stem)
    checks.append(lineage_check)
    open_items.extend(lineage_notes)

    platon = summary.get("platon") or {}
    report = platon.get("report") if isinstance(platon.get("report"), dict) else None
    report_ok = bool(report) and int(report.get("lines_read") or 0) > 0
    checks.append(
        AcceptanceCheck(
            "platon_report",
            report_ok,
            f"{report.get('lines_read')} line(s) read, counts {report.get('counts')}" if report_ok
            else "no PLATON report was read",
        )
    )

    refined_view = (summary.get("viewer") or {}).get("refined")
    view_ok = bool(refined_view) and (run_dir / str(refined_view)).is_file()
    checks.append(
        AcceptanceCheck(
            "viewer_refined",
            view_ok,
            f"{refined_view}" if view_ok else "no refined-model view was written",
        )
    )

    if listing and not listing.get("format_verified", False):
        open_items.append(
            f"The SHELXL listing parser is unverified against this installation: compare R1, wR2, "
            f"GooF and shift/su in the summary with {stem}.lst."
        )
    if report and not report.get("format_verified", False):
        open_items.append(
            f"The PLATON report parser is unverified against this installation: compare the alert "
            f"counts with {stem}.chk."
        )
    if listing.get("notices"):
        open_items.append(
            "SHELXL printed messages this build cannot classify (review them in the .lst): "
            + "; ".join(str(n) for n in listing["notices"][:5])
        )
    if listing:
        open_items.append(
            "Convergence is not assessed by this build: read the final cycle's maximum shift/su in "
            f"{stem}.lst (summary: mean {listing.get('mean_shift_su')}, max {listing.get('max_shift_su')})."
        )
    open_items.append(
        "Scientific eligibility is not evaluated (no chemical-validity or coordination calculator): "
        "this check is mechanical."
    )
    open_items.append(
        "Scoring against a reference structure is a separate, later step for the reviewer; this "
        "check never reads one."
    )
    return AcceptanceVerdict(run_dir, tuple(checks), tuple(open_items), require_real=require_real)


def _latest_provenance(run_dir: Path, tool: str) -> Path | None:
    matches = sorted(run_dir.glob(_PROVENANCE_GLOBS[tool]))
    return matches[-1] if matches else None


def _check_program(
    run_dir: Path, tool: str, required: tuple[str, ...], require_real: bool
) -> tuple[AcceptanceCheck, list[str]]:
    name = f"{tool}_run"
    path = _latest_provenance(run_dir, tool)
    if path is None:
        return AcceptanceCheck(name, False, "no provenance record: the program did not run"), []
    record, problem = _load(path)
    if record is None:
        return AcceptanceCheck(name, False, problem), []

    problems: list[str] = []
    notes: list[str] = []
    if record.get("succeeded") is not True:
        problems.append(
            f"did not succeed ({record.get('diagnostic')}: {record.get('diagnostic_message', '')})"
        )
    if record.get("exit_code") != 0:
        problems.append(f"exit code {record.get('exit_code')!r}")
    if record.get("missing_outputs"):
        problems.append("expected outputs missing: " + ", ".join(record["missing_outputs"]))

    executable = str(record.get("executable_path") or "")
    if require_real:
        if record.get("executable_source") == "synthetic_demo":
            problems.append("a stand-in from the demo toolchain, not a program")
        exe = Path(executable)
        if not exe.is_absolute():
            problems.append(f"executable path {executable!r} is not absolute")
        elif not exe.is_file():
            problems.append(f"the executable is no longer at {executable}")
        else:
            try:
                current = sha256_file(exe)
            except OSError as exc:
                current = f"unreadable ({exc})"
            if current != record.get("executable_sha256"):
                problems.append(f"{executable} no longer matches the sha256 recorded for the run")

    work = path.parent
    written = {
        entry.get("relative_path"): entry
        for entry in record.get("outputs") or []
        if isinstance(entry, dict)
    }
    for relative in required:
        target, why = _inside(run_dir, str((work / relative).relative_to(run_dir)))
        if target is None:
            problems.append(f"{relative}: {why}")
            continue
        now, why = _digest(target)
        if now is None:
            problems.append(f"{relative} is {why}")
            continue
        entry = written.get(relative)
        if entry is None:
            problems.append(f"{relative} was not recorded as written by this program")
        elif now != entry.get("sha256"):
            problems.append(f"{relative} was changed after the program wrote it")

    if record.get("required_translation"):
        notes.append(f"{tool} ran through Rosetta 2 (x86_64 translation).")
    if record.get("group_cleanup") == "killed":
        notes.append(f"{tool}: a process left in its group after it ended had to be killed.")

    detail = (
        f"exit 0, {executable} (sha256 {str(record.get('executable_sha256'))[:16]}), outputs present"
        if not problems
        else "; ".join(problems)
    )
    return AcceptanceCheck(name, not problems, detail), notes


# --------------------------------------------------------------------------------------
# inputs: the run's historical observation, and the archive's integrity now
# --------------------------------------------------------------------------------------


def _inside(run_dir: Path, relative: str) -> tuple[Path | None, str]:
    """Resolve a recorded path; it must be relative and stay inside the run directory."""
    candidate = Path(relative)
    if not relative or candidate.is_absolute():
        return None, f"recorded path {relative!r} is not relative to the run directory"
    try:
        root = run_dir.resolve()
        target = (root / candidate).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"recorded path {relative!r} cannot be resolved: {exc}"
    try:
        target.relative_to(root)
    except ValueError:
        return None, f"recorded path {relative!r} leaves the run directory"
    return target, ""


def _digest(path: Path) -> tuple[str | None, str]:
    """The file's SHA-256 now, or ``None`` and why not."""
    try:
        if not path.exists():
            return None, "missing"
        if not path.is_file():
            return None, "not a readable regular file"
        return sha256_file(path), ""
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"unreadable ({exc})"


def _check_inputs_at_run_end(summary: dict[str, Any]) -> AcceptanceCheck:
    """What the run itself observed when it ended. Historical: nothing is re-read here."""
    verification = summary.get("input_verification") or {}
    if not isinstance(verification, dict):
        verification = {}
    problems = []
    for role in _INPUT_ROLES:
        entry = verification.get(role)
        if not isinstance(entry, dict):
            problems.append(f"{role}: no end-of-run observation recorded")
        elif entry.get("original_unchanged") is not True or entry.get("staged_copy_unchanged") is not True:
            problems.append(
                f"{role}: at the end of the run the original was "
                f"{'unchanged' if entry.get('original_unchanged') is True else 'changed or unreadable'} and "
                f"the staged copy {'unchanged' if entry.get('staged_copy_unchanged') is True else 'changed or unreadable'}"
            )
    detail = (
        "historical observation made when the run ended: originals and staged copies unchanged "
        "(the original upload path is not re-checked now and need not still exist)"
        if not problems
        else "historical observation made when the run ended: " + "; ".join(problems)
    )
    return AcceptanceCheck("inputs_at_run_end", not problems, detail)


def _check_staged_inputs(run_dir: Path, summary: dict[str, Any]) -> tuple[AcceptanceCheck, dict[str, str | None]]:
    """Re-hash the retained staged INS and HKL now. Both roles are required."""
    recorded = summary.get("inputs") or {}
    if not isinstance(recorded, dict):
        recorded = {}
    current: dict[str, str | None] = {}
    intact: list[str] = []
    problems: list[str] = []
    for role in _INPUT_ROLES:
        entry = recorded.get(role)
        staged = entry.get("staged") if isinstance(entry, dict) else None
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        current[role] = None
        if not isinstance(staged, str) or not staged or not isinstance(digest, str) or not digest:
            problems.append(f"{role}: no staged copy and digest recorded")
            continue
        target, why = _inside(run_dir, str(staged))
        if target is None:
            problems.append(f"{role}: {why}")
            continue
        now, why = _digest(target)
        current[role] = now
        if now is None:
            problems.append(f"{role}: {staged} is {why}")
        elif now != digest:
            problems.append(f"{role}: {staged} changed since the run (recorded {digest[:16]}, now {now[:16]})")
        else:
            intact.append(f"{role} {staged} sha256 {digest[:16]}")
    detail = (
        "re-hashed now and unchanged: " + "; ".join(intact) if not problems else "; ".join(problems)
    )
    return AcceptanceCheck("staged_inputs_intact", not problems, detail), current


# --------------------------------------------------------------------------------------
# reflections: the contract shared with the pipeline
# --------------------------------------------------------------------------------------


def _by_path(record: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    return {
        str(entry.get("relative_path")): entry
        for entry in record.get(key) or []
        if isinstance(entry, dict)
    }


def _check_reflection_lineage(
    run_dir: Path, summary: dict[str, Any], stem: str
) -> tuple[AcceptanceCheck, list[str]]:
    """Re-make the pipeline's reflection decision from the retained files and follow the digests.

    Chain: the run's staged reflections -> SHELXT's input copy -> (SHELXT's matching ``.hkl`` | the
    same input copy under the fallback) -> the refinement copy -> what SHELXL recorded reading.
    """
    name = "reflection_lineage"
    refinement = summary.get("refinement") or {}
    lineage = refinement.get("reflections_lineage") if isinstance(refinement, dict) else None
    if not isinstance(lineage, dict):
        return AcceptanceCheck(name, False, "no reflection lineage was recorded for the refinement"), []
    try:
        source = ReflectionSource(lineage.get("source"))
    except ValueError:
        return AcceptanceCheck(name, False, f"unknown reflection source {lineage.get('source')!r}"), []
    letter = lineage.get("solution_letter")
    if not (isinstance(letter, str) and re.fullmatch(r"[a-z]", letter)):
        return AcceptanceCheck(name, False, "no solution letter was recorded"), []

    shelxt_path = _latest_provenance(run_dir, "shelxt")
    shelxl_path = _latest_provenance(run_dir, "shelxl")
    shelxt, problem_t = _load(shelxt_path) if shelxt_path else (None, "no SHELXT provenance")
    shelxl, problem_l = _load(shelxl_path) if shelxl_path else (None, "no SHELXL provenance")
    if shelxt is None or shelxl is None or shelxt_path is None or shelxl_path is None:
        detail = "; ".join(p for p in (problem_t, problem_l) if p)
        return AcceptanceCheck(name, False, f"cannot follow the reflections: {detail}"), []

    branch = shelxt_path.parent
    matching = solution_reflections_name(stem, letter)
    reflections_name = matching if source is ReflectionSource.SHELXT_MATCHING else input_reflections_name(stem)
    problems: list[str] = []

    retained: dict[str, Path] = {}
    for key, expected in (
        ("solution_file", branch / solution_model_name(stem, letter)),
        ("reflections_file", branch / reflections_name),
        ("refinement_copy", shelxl_path.parent / input_reflections_name(stem)),
    ):
        target, why = _inside(run_dir, str(lineage.get(key) or ""))
        expected_target, expected_why = _inside(run_dir, str(expected.relative_to(run_dir)))
        if target is None:
            problems.append(f"{key}: {why}")
        elif expected_target is None:
            problems.append(f"{key}: {expected_why}")
        elif target != expected_target:
            problems.append(f"{key} {lineage.get(key)!r} is not the expected retained {expected.name}")
        else:
            retained[key] = target

    # Check paths before the decision reader can follow any retained symlink outside the run.
    for role in _INPUT_ROLES:
        relative = str((branch / f"{stem}.{role}").relative_to(run_dir))
        target, why = _inside(run_dir, relative)
        if target is None:
            problems.append(f"SHELXT input {role}: {why}")
        else:
            retained[f"shelxt_{role}"] = target
    if problems:
        return AcceptanceCheck(name, False, "; ".join(problems)), []

    decision = decide_from_files(branch, stem, letter)
    if decision.source is not source:
        found = decision.source.value if decision.source else f"refusal ({decision.reason})"
        problems.append(f"the retained files do not support the recorded source {source.value}: they give {found}")
    else:
        # The archived statement and CELL comparison must describe these same retained files.
        for key, value in decision.to_record().items():
            if key != "evidence" and lineage.get(key) != value:
                problems.append(f"recorded reflection decision {key} differs from the retained files")

    digest = lineage.get("reflections_sha256")
    inputs = summary.get("inputs") if isinstance(summary.get("inputs"), dict) else {}
    staged_hkl = (inputs.get("hkl") if isinstance(inputs.get("hkl"), dict) else {}).get("sha256")
    shelxt_inputs = _by_path(shelxt, "inputs")
    shelxt_outputs = _by_path(shelxt, "outputs")
    for role in _INPUT_ROLES:
        file_name = f"{stem}.{role}"
        recorded_digest = (shelxt_inputs.get(file_name) or {}).get("sha256")
        now, why = _digest(retained[f"shelxt_{role}"])
        if now is None:
            problems.append(f"SHELXT input {file_name} is {why}")
        elif recorded_digest is None or now != recorded_digest:
            problems.append(f"SHELXT input {file_name} changed since the run or has no recorded digest")

    model_name = solution_model_name(stem, letter)
    model_digest = (shelxt_outputs.get(model_name) or {}).get("sha256")
    model_now, why = _digest(retained["solution_file"])
    if model_now is None:
        problems.append(f"chosen solution {model_name} is {why}")
    elif model_digest is None or model_now != model_digest or lineage.get("solution_sha256") != model_digest:
        problems.append("the chosen solution digest differs from the lineage or SHELXT's provenance")
    shelxt_read = (shelxt_inputs.get(input_reflections_name(stem)) or {}).get("sha256")
    if shelxt_read is None or shelxt_read != staged_hkl:
        problems.append("the reflections SHELXT recorded reading are not the run's staged reflections")
    if source is ReflectionSource.SHELXT_MATCHING:
        written = (shelxt_outputs.get(matching) or {}).get("sha256")
        if written is None:
            problems.append(f"SHELXT's provenance does not record writing {matching}")
        elif written != digest:
            problems.append(f"the lineage digest of {matching} differs from SHELXT's provenance")
    else:
        if matching in shelxt_outputs:
            problems.append(
                f"SHELXT recorded writing {matching}, so the 'Orientation as input' fallback was not permitted"
            )
        if digest != shelxt_read:
            problems.append("the fallback reflections are not the file SHELXT read")

    now, why = _digest(retained["reflections_file"])
    if now is None:
        problems.append(f"{reflections_name} is {why}")
    elif now != digest:
        problems.append(f"{reflections_name} changed since the run")

    copy_name = input_reflections_name(stem)
    read_by_shelxl = (_by_path(shelxl, "inputs").get(copy_name) or {}).get("sha256")
    if read_by_shelxl is None or read_by_shelxl != digest:
        problems.append("the reflections SHELXL recorded reading differ from the chosen source")
    if lineage.get("refinement_copy_sha256") != digest:
        problems.append("the recorded refinement copy digest differs from the chosen source")
    copy_now, why = _digest(retained["refinement_copy"])
    if copy_now is None:
        problems.append(f"the refinement copy {copy_name} is {why}")
    elif copy_now != digest:
        problems.append(f"the refinement copy {copy_name} changed since the run")

    notes: list[str] = []
    if source is ReflectionSource.ORIGINAL_ORIENTATION_AS_INPUT:
        cell = lineage.get("cell_check") or {}
        notes.append(
            "The refinement used the original reflections under the 'Orientation as input' fallback "
            f"(statement {lineage.get('orientation_statement')!r}; cell difference "
            f"{cell.get('max_length_difference_a')} A / {cell.get('max_angle_difference_deg')} deg). "
            "The statement's text format is not verified against real SHELXT output."
        )
    if problems:
        return AcceptanceCheck(name, False, "; ".join(problems)), notes
    if source is ReflectionSource.SHELXT_MATCHING:
        detail = f"SHELXT's {matching} (sha256 {str(digest)[:16]}) was copied unchanged and read by SHELXL"
    else:
        detail = (
            f"original reflections (sha256 {str(digest)[:16]}) under the evidenced 'Orientation as "
            "input' fallback, copied unchanged and read by SHELXL"
        )
    return AcceptanceCheck(name, True, detail), notes
