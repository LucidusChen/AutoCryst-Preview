"""Expert-model component regression using the same service refinement/validation stages.

There is no initial-solution search here. A supplied model/reflection pair is copied, hashed and
fed to the ordinary bounded SHELXL/PLATON pipeline. Pair compatibility is an explicit owner
assumption, not established by matching CELL parameters. The independent mechanical verifier
never treats this track as successful de novo discovery or scientific review.

Real component dispatch is deliberately disabled for this offline milestone. Injecting this
pipeline into RunService cannot bypass that gate. Generated demo-only toolchains exercise the
shared software stages in synthetic tests; a future bounded pinned-plan authority mechanism must
replace the gate before any real component process is allowed.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from crystalx_core.app.acceptance import (
    AcceptanceCheck, _by_path, _check_inputs_at_run_end, _check_program,
    _check_staged_inputs, _digest, _inside, _load,
)
from crystalx_core.app.models import (
    STAGE_ORDER, ModelAvailability, RefinementStatus, RunMode, RunPhase, SolveRequest, StageName, StageState,
)
from crystalx_core.app.pipeline import SolvePipeline, _Halt
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.result import StopReasonKind
from crystalx_core.domain.symmetry import ChiralityCompatibility, analyse_symmetry
from crystalx_core.errors import CrystalXError
from crystalx_core.geometry.model import build_viewer_payload
from crystalx_core.io.shelx_reader import logical_lines, read_ins_header, scan_hkl
from crystalx_core.io.shelx_res import ResFile, read_res
from crystalx_core.private_acceptance.specs import AcceptanceTrack
from crystalx_core.refinement.shelxl import RefinementSettings, prepare_refinement_ins
from crystalx_core.workflow.preconditions import check_preconditions
from crystalx_core.workspace.hashing import sha256_file

__all__ = ["ComponentRegressionPipeline", "ComponentAcceptanceVerdict", "verify_component_run",
           "preview_component_request"]

_MODEL_DIR = "component/input_model"
_REFINE_DIR = "refinement/shelxl-01"
_COMPONENT_SOURCE = "expert_model_paired_hkl"


def _contained_record(root: Path, relative: str):
    target, why = _inside(root, relative)
    return _load(target) if target is not None else (None, why)


def _inspect_component_input(request: SolveRequest, *, ins_path: Path, hkl_path: Path,
                             diagnosis: DataDiagnosis | None = None) -> dict[str, Any]:
    """One fixed-model policy shared by offline preview and the staged execution path."""
    issues: list[str] = []
    codes: list[str] = []
    items: list[dict[str, Any]] = []

    def block(code: str, explanation: str) -> None:
        codes.append(code)
        issues.append(explanation)

    if request.mode is not RunMode.REAL:
        block("real_program_mode_required", "private component regression requires real-program mode")
    if request.prior.composition:
        block("composition_reassignment_forbidden", "component regression cannot replace SFAC/UNIT or reassign expert atom indices")
    try:
        if any(line.instruction.startswith("+") for line in logical_lines(ins_path)):
            block("instruction_include_forbidden", "external instruction includes are not permitted in the two-file component contract")
        if diagnosis is None:
            header = read_ins_header(ins_path)
            reflections = scan_hkl(hkl_path, cell=header.cell, hklf_hint=header.hklf_format)
            diagnosis = DataDiagnosis(reflections=reflections, ins=header)
        report = check_preconditions(diagnosis, request.prior.to_prior(), space_group_from_solver=False)
        items = [{"kind": p.kind.value, "blocking": p.blocking, "summary": p.summary,
                  "evidence": p.evidence, "remedy": p.remedy} for p in report.preconditions]
        for p in report.blockers:
            block(p.kind.value, p.summary)
        if diagnosis.reflections.n_reflections == 0:
            block("no_reflections", "the selected HKL contains no readable reflections")
        model = read_res(ins_path)
        if model.n_atoms == 0:
            block("no_model_atoms", "the supplied expert model contains no atom sites")
        if model.header.cell is None or model.header.latt is None or model.header.hklf_format is None:
            block("fixed_model_declarations_missing", "the fixed model must explicitly declare CELL, LATT and HKLF; no symmetry or reflection format is inferred")
        if any(not 1 <= atom.sfac_index <= len(model.header.sfac) for atom in model.atoms):
            block("invalid_sfac_index", "the supplied model has an atom with an invalid SFAC index")
        if any(not math.isfinite(v) for atom in model.atoms
               for v in (atom.x, atom.y, atom.z, atom.occupancy_code, *atom.u_values)):
            block("nonfinite_atom_values", "the supplied model has nonfinite atom values")
        enantiopure, _ = request.prior.to_prior().is_enantiopure()
        symmetry = analyse_symmetry(model.header.latt, model.header.symm)
        if enantiopure and symmetry.compatibility is ChiralityCompatibility.UNKNOWN:
            block("fixed_model_chirality_unresolved", "the fixed model's symmetry cannot be checked for the independently declared homochiral chemistry")
    except (CrystalXError, OSError, ValueError, OverflowError):
        # Do not propagate parser excerpts, coordinates, pathnames or private chemistry into a
        # generic planning error. Detailed inputs are retained locally, never written here.
        block("component_input_unreadable", "the selected component model/reflections could not be safely checked")
    return {
        "may_start": not issues, "fixed_model": True, "space_group_from_solver": False,
        "summary": "; ".join(issues) if issues else "fixed-model component preconditions satisfied",
        "items": items, "component_blockers": issues, "blocker_codes": codes,
        "initial_solution_performed": False,
        "pair_compatibility_scientifically_verified": False,
    }


def preview_component_request(request: SolveRequest) -> dict[str, Any]:
    """Read INS/HKL only; no workspace, writes, tool resolution or process launch.

    This reports software preconditions on the supplied model, not whether it is scientifically
    correct or matched to the selected reflections. Returned diagnostics belong in private plans.
    """
    issues = request.validate()
    if issues:
        return {
            "may_start": False, "fixed_model": True, "space_group_from_solver": False,
            "summary": "component request validation failed", "items": [],
            "component_blockers": ["component request validation failed"],
            "blocker_codes": sorted({item.code for item in issues}),
            "initial_solution_performed": False,
            "pair_compatibility_scientifically_verified": False,
        }
    return _inspect_component_input(request, ins_path=request.ins_path, hkl_path=request.hkl_path)


@dataclass(frozen=True)
class _SuppliedModel:
    res_path: Path
    res: ResFile
    letter: str = "component"


@dataclass(frozen=True)
class _ComponentCandidate:
    stem: str
    work_dir: Path
    solution: _SuppliedModel
    display: list[dict[str, Any]]
    alternatives: tuple = ()


@dataclass(frozen=True)
class _ComponentReflectionDecision:
    file_name: str
    model_sha256: str
    hkl_sha256: str
    allowed: bool = True
    source: str = _COMPONENT_SOURCE

    @property
    def note(self) -> str:
        return "explicitly supplied expert-model/reflection pair; scientific compatibility requires owner review"

    def to_record(self) -> dict[str, Any]:
        return {
            "contract": "crystalx.component_reflections/1", "source": self.source,
            "allowed": True, "file": self.file_name, "model_sha256": self.model_sha256,
            "paired_hkl_sha256": self.hkl_sha256,
            "evidence": "owner-selected model/HKL pair copied unchanged; no SHELXT reindexing or orientation inference",
            "compatibility_scientifically_verified": False,
        }


class ComponentRegressionPipeline(SolvePipeline):
    """Inject with ``RunService(..., pipeline_factory=ComponentRegressionPipeline)``."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.summary["acceptance_track"] = AcceptanceTrack.COMPONENT_REGRESSION.value
        self.summary["mock"] = self.request.mode is RunMode.MOCK_DEMO or self.toolchain.mode is RunMode.MOCK_DEMO
        self.summary["component_regression"] = {
            "initial_solution_performed": False,
            "meaning": "bounded component regression from a supplied expert model, not initial structure solution",
            "pair_compatibility": "owner-selected assumption; not independently scientifically verified",
            "real_launch_enabled": False,
        }

    def request_record(self) -> dict[str, Any]:
        return {**super().request_record(), "acceptance_track": AcceptanceTrack.COMPONENT_REGRESSION.value,
                "initial_solution_performed": False}

    def run(self) -> dict[str, Any]:
        self._set_phase(RunPhase.RUNNING, "running expert-model component regression")
        self.state["started_at"] = self.clock().isoformat()
        self._write_state()
        halt = None
        try:
            self._check_cancel()
            self._stage_inputs()
            self._diagnose()
            self._preconditions()
            self._component_plan()
            candidate = self._stage_supplied_model()
            refinement = self._shelxl(candidate)
            if isinstance(self.summary.get("refinement"), dict):
                self.summary["refinement"]["component_note"] = (
                    "A fixed bounded refinement of a supplied expert model. Its R values do not "
                    "measure initial-solution recovery or match an expert's completed refinement."
                )
                self.summary["refinement"]["preliminary_note"] = (
                    "This component regression uses the shared isotropic preparation policy, with "
                    "all derivative instruction edits recorded; the expert original is unchanged."
                )
            self._platon(refinement)
            self._viewer(candidate, refinement)
        except _Halt as stop:
            halt = stop
        except Exception as exc:  # same durable failure contract as SolvePipeline
            halt = self._internal_error(exc)
        return self._finish(halt)

    def _preconditions(self) -> None:
        self._check_cancel()
        self._begin(StageName.PRECONDITIONS, "checking the fixed supplied expert model")
        assert self._diagnosis is not None
        report = _inspect_component_input(self.request, ins_path=self._staged["ins"],
                                          hkl_path=self._staged["hkl"], diagnosis=self._diagnosis)
        self.summary["preconditions"] = report
        self.workspace.write_json("component/preconditions.json", report)
        for advisory in report["items"]:
            if not advisory["blocking"]:
                self._warn(advisory["summary"])
        if not report["may_start"]:
            message = report["summary"]
            self._end(StageName.PRECONDITIONS, StageState.BLOCKED, message)
            raise _Halt(RunPhase.BLOCKED, message, StopReasonKind.INPUT_INVALID)
        self._end(StageName.PRECONDITIONS, StageState.SUCCEEDED, report["summary"])

    def _component_plan(self) -> None:
        self._check_cancel()
        self._begin(StageName.PLAN, "using the fixed supplied expert model; no initial-solution search")
        if self.toolchain.mode is not RunMode.MOCK_DEMO:
            message = "real component launches are disabled in this offline milestone; bounded pinned-plan authority is not implemented"
            self._end(StageName.PLAN, StageState.BLOCKED, message)
            raise _Halt(RunPhase.BLOCKED, message, StopReasonKind.INPUT_INVALID)
        self.summary["plan"] = {
            "track": AcceptanceTrack.COMPONENT_REGRESSION.value,
            "initial_solution_performed": False, "stages": ["shelxl", "platon", "viewer"],
            "shelxl_cycles": self.request.config.shelxl_cycles,
            "preparation_policy": "shared bounded isotropic preparation; derivative edits recorded",
        }
        self.workspace.write_json("component/plan.json", self.summary["plan"])
        self._end(StageName.PLAN, StageState.SUCCEEDED, "bounded SHELXL then PLATON; no SHELXT")
        self._begin(StageName.SHELXT, "initial solution is outside this component regression")
        self._end(StageName.SHELXT, StageState.SKIPPED, "not performed: supplied expert model is the input")

    def _stage_supplied_model(self) -> _ComponentCandidate:
        self._check_cancel()
        directory = self.workspace.artefact_path(*_MODEL_DIR.split("/"))
        directory.mkdir(parents=True, exist_ok=False)
        model_path = directory / f"{self.stem}.res"
        hkl_path = directory / f"{self.stem}.hkl"
        shutil.copyfile(self._staged["ins"], model_path)
        shutil.copyfile(self._staged["hkl"], hkl_path)
        model = read_res(model_path)
        hashes = {"ins": sha256_file(model_path), "hkl": sha256_file(hkl_path)}
        for role, digest in hashes.items():
            if digest != self.summary["inputs"][role]["sha256"]:
                raise _Halt(RunPhase.FAILED, "supplied component pair copy failed integrity check", StopReasonKind.INPUT_INVALID)
        symmetry = analyse_symmetry(model.header.latt, model.header.symm)
        self.summary["component_regression"]["input_model"] = {
            "file": f"{_MODEL_DIR}/{model_path.name}", "sha256": hashes["ins"],
            "paired_hkl": f"{_MODEL_DIR}/{hkl_path.name}", "hkl_sha256": hashes["hkl"],
            "n_atoms": model.n_atoms,
        }
        self.summary["model_availability"] = "supplied_expert_model"
        self.summary["solution"] = None
        self.workspace.write_json("component/input_pair.json", self.summary["component_regression"]["input_model"])
        self._add_artefacts(directory, (model_path.name, hkl_path.name))
        return _ComponentCandidate(self.stem, directory, _SuppliedModel(model_path, model), [{
            "chirality": symmetry.compatibility.value, "symmetry_explanation": symmetry.explain(),
        }])

    def _reflection_decision(self, solution: _ComponentCandidate) -> _ComponentReflectionDecision:
        return _ComponentReflectionDecision(
            file_name=f"{solution.stem}.hkl", model_sha256=sha256_file(solution.solution.res_path),
            hkl_sha256=sha256_file(solution.work_dir / f"{solution.stem}.hkl"),
        )

    def _viewer(self, solution: _ComponentCandidate, refinement) -> None:
        self._begin(StageName.VIEWER, "building the supplied-model component views")
        sources = [("expert_model", solution.solution.res, "supplied expert model (component input)")]
        if refinement is not None and refinement.succeeded and refinement.res is not None:
            sources.append(("refined", refinement.res, "bounded expert-model component regression"))
        for name, model, label in sources:
            try:
                payload = build_viewer_payload(model, stage=name, source_label=label)
            except ValueError as exc:
                self._warn(f"component viewer: {name}: {exc}")
                continue
            payload["mock"] = self.summary["mock"]
            payload["acceptance_track"] = AcceptanceTrack.COMPONENT_REGRESSION.value
            self.workspace.write_json(f"viewer/{name}.json", payload)
            self.summary["viewer"][name] = f"viewer/{name}.json"
        state = StageState.SUCCEEDED if self.summary["viewer"] else StageState.FAILED
        self._end(StageName.VIEWER, state, "component views: " + ", ".join(self.summary["viewer"]))

    def _finish(self, halt):
        if halt is None:
            blocked = [s for s in self.state["stages"] if s["name"] in {"shelxl", "platon"}
                       and s["state"] in {StageState.BLOCKED.value, StageState.TOOL_UNAVAILABLE.value}]
            if blocked:
                halt = _Halt(RunPhase.BLOCKED, blocked[0]["message"], StopReasonKind.TOOL_UNAVAILABLE)
        return super()._finish(halt)

    def _completion_message(self) -> str:
        return "component regression finished from a supplied expert model; no initial solution or scientific validation is claimed"


@dataclass(frozen=True)
class ComponentAcceptanceVerdict:
    run_dir: Path
    checks: tuple[AcceptanceCheck, ...]
    open_items: tuple[str, ...] = ()
    require_real: bool = True

    @property
    def accepted(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def failed(self) -> tuple[AcceptanceCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": "crystalx.component_acceptance/1", "run_dir": str(self.run_dir),
            "track": AcceptanceTrack.COMPONENT_REGRESSION.value, "accepted": self.accepted,
            "require_real_programs": self.require_real, "initial_solution_verified": False,
            "scientifically_reviewed": False,
            "meaning": "mechanical supplied-model SHELXL/PLATON component regression only; no initial solution or scientific correctness claim",
            "checks": [check.to_record() for check in self.checks], "open_items": list(self.open_items),
        }


def _component_lineage(run_dir: Path, summary: dict[str, Any]) -> AcceptanceCheck:
    """Retained expert copy/HKL -> declared bounded INS/HKL -> SHELXL-read digests."""
    problems: list[str] = []
    stem = summary.get("stem")
    if not isinstance(stem, str) or not stem or Path(stem).name != stem:
        return AcceptanceCheck("component_lineage", False, "invalid component stem")
    refinement = summary.get("refinement") or {}
    lineage = refinement.get("reflections_lineage") or {}
    pair = (summary.get("component_regression") or {}).get("input_model") or {}
    provenance, _ = _contained_record(run_dir, f"{_REFINE_DIR}/provenance.json")
    if not provenance:
        return AcceptanceCheck("component_lineage", False, "no SHELXL provenance")
    if lineage.get("source") != _COMPONENT_SOURCE or lineage.get("contract") != "crystalx.component_reflections/1":
        problems.append("reflection source is not the explicit expert-model pair contract")
    inputs = summary.get("inputs") or {}
    expected_paths = {
        "model": (f"{_MODEL_DIR}/{stem}.res", pair.get("file")),
        "paired_hkl": (f"{_MODEL_DIR}/{stem}.hkl", pair.get("paired_hkl")),
        "refined_hkl": (f"{_REFINE_DIR}/{stem}.hkl", lineage.get("refinement_copy")),
    }
    retained: dict[str, Path] = {}
    for role, (expected, recorded) in expected_paths.items():
        if recorded != expected:
            problems.append(f"{role}: recorded path differs from its fixed component location")
            continue
        target, why = _inside(run_dir, expected)
        if target is None:
            problems.append(f"{role}: {why}")
        else:
            retained[role] = target
    if len(retained) != 3:
        return AcceptanceCheck("component_lineage", False, "; ".join(problems))
    digests = {role: _digest(path)[0] for role, path in retained.items()}
    model_expected = (inputs.get("ins") or {}).get("sha256")
    hkl_expected = (inputs.get("hkl") or {}).get("sha256")
    if not model_expected or any(value != model_expected for value in (
            digests["model"], pair.get("sha256"), lineage.get("model_sha256"), lineage.get("solution_sha256"))):
        problems.append("supplied model copy digest differs from the retained staged input or lineage")
    if not hkl_expected or any(value != hkl_expected for value in (
            digests["paired_hkl"], digests["refined_hkl"], pair.get("hkl_sha256"),
            lineage.get("paired_hkl_sha256"), lineage.get("reflections_sha256"), lineage.get("refinement_copy_sha256"))):
        problems.append("paired HKL digest differs along the staged/copy/refinement chain")
    if lineage.get("solution_file") != expected_paths["model"][0] or lineage.get("reflections_file") != expected_paths["paired_hkl"][0]:
        problems.append("lineage model/HKL paths differ from the fixed component pair")
    read_inputs = _by_path(provenance, "inputs")
    if (read_inputs.get(f"{stem}.hkl") or {}).get("sha256") != hkl_expected:
        problems.append("SHELXL did not record reading the paired HKL digest")
    ins_path, why = _inside(run_dir, f"{_REFINE_DIR}/{stem}.ins")
    actual_ins, _ = _digest(ins_path) if ins_path is not None else (None, why)
    if not actual_ins or (read_inputs.get(f"{stem}.ins") or {}).get("sha256") != actual_ins:
        problems.append("SHELXL instruction input is changed or missing its read digest")
    settings = refinement.get("settings") or {}
    try:
        if (type(settings.get("cycles")) is not int
                or any(settings.get(field) is not True for field in
                       ("isotropic", "write_cif", "write_fcf", "strip_q_peaks"))
                or settings.get("plan_peaks") != 20):
            problems.append("refinement settings differ from the fixed bounded component policy")
        expected_ins, changes, atoms, removed = prepare_refinement_ins(
            retained["model"].read_text(encoding="utf-8", errors="replace"), RefinementSettings(
                cycles=settings["cycles"], isotropic=settings["isotropic"], plan_peaks=settings["plan_peaks"],
                write_cif=settings["write_cif"], write_fcf=settings["write_fcf"], strip_q_peaks=settings["strip_q_peaks"],
            ),
        )
        if ins_path is None or ins_path.read_text(encoding="utf-8") != expected_ins:
            problems.append("SHELXL INS does not match deterministic bounded preparation of the supplied model")
        if refinement.get("changes") != list(changes):
            problems.append("recorded derivative instruction edits differ from retained input preparation")
        if atoms < 1:
            problems.append("bounded component preparation contains no atom sites")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        problems.append(f"cannot reconstruct bounded component preparation: {exc}")
    return AcceptanceCheck("component_lineage", not problems,
                           "; ".join(problems) if problems else "retained supplied model/HKL and bounded SHELXL read hashes agree")


def _verify_component_run(run_dir: Path, *, require_real: bool,
                          expected_inputs: Mapping[str, str] | None) -> ComponentAcceptanceVerdict:
    root = Path(run_dir)
    state, problem_s = _contained_record(root, "state.json")
    summary, problem_m = _contained_record(root, "summary.json")
    if state is None or summary is None:
        return ComponentAcceptanceVerdict(root, (AcceptanceCheck("record_readable", False,
                                          "; ".join(filter(None, (problem_s, problem_m)))),), require_real=require_real)
    checks = [AcceptanceCheck("record_readable", True, "state and summary read")]
    component = summary.get("component_regression") or {}
    checks.append(AcceptanceCheck("component_track", summary.get("acceptance_track") == AcceptanceTrack.COMPONENT_REGRESSION.value
                                  and component.get("initial_solution_performed") is False and summary.get("solution") is None,
                                  "explicit supplied-model component track; no initial solution is claimed"))
    if require_real:
        checks.append(AcceptanceCheck("real_programs", summary.get("mode") == "real" and summary.get("mock") is False,
                                      "real mode required; stand-ins cannot establish real component process evidence"))
    checks.append(AcceptanceCheck("phase_completed", state.get("phase") == RunPhase.COMPLETED.value,
                                  f"phase {state.get('phase')!r}"))
    states = {s.get("name"): s.get("state") for s in state.get("stages", []) if isinstance(s, dict)}
    required = [stage.value for stage in STAGE_ORDER if stage is not StageName.SHELXT]
    bad_stages = [name for name in required if states.get(name) != StageState.SUCCEEDED.value]
    checks.append(AcceptanceCheck("component_stages_succeeded", not bad_stages, ", ".join(bad_stages) or "required component stages succeeded"))
    no_shelxt = states.get("shelxt") == StageState.SKIPPED.value and not list(root.glob("stages/*/branches/*-shelxt/provenance_*.json"))
    checks.append(AcceptanceCheck("shelxt_not_run", no_shelxt, "SHELXT must be explicitly skipped and have no execution provenance"))
    checks.append(_check_inputs_at_run_end(summary))
    staged, current = _check_staged_inputs(root, summary)
    checks.append(staged)
    if expected_inputs:
        matches = set(expected_inputs) == {"ins", "hkl"} and all(current.get(role) == digest for role, digest in expected_inputs.items())
        checks.append(AcceptanceCheck("inputs_expected", matches, "both expected original input digests must match retained staged copies"))
    checks.append(_component_lineage(root, summary))
    notes = []
    stem = str(summary.get("stem") or "")
    for tool, required_outputs in (("shelxl", (f"{stem}.res", f"{stem}.lst", f"{stem}.cif", f"{stem}.fcf")),
                                   ("platon", (f"{stem}.chk",))):
        relative = f"{_REFINE_DIR if tool == 'shelxl' else 'validation/platon-01'}/provenance.json"
        record, why = _contained_record(root, relative)
        if record is None:
            check, items = AcceptanceCheck(f"{tool}_run", False, why), []
        else:
            check, items = _check_program(root, tool, required_outputs, require_real)
        checks.append(check)
        notes.extend(items)
    refinement = summary.get("refinement") or {}
    listing = refinement.get("listing") or {}
    refined = summary.get("model_availability") == ModelAvailability.PRELIMINARY_REFINED_MODEL.value
    checks.append(AcceptanceCheck("refinement_clean", refined and summary.get("refinement_status") == RefinementStatus.RAN_CONVERGENCE_NOT_ASSESSED.value
                                  and not listing.get("errors"), "readable preliminary component model and no recognised listing errors required"))
    report = (summary.get("platon") or {}).get("report") or {}
    checks.append(AcceptanceCheck("platon_report", isinstance(report.get("lines_read"), int) and report["lines_read"] > 0,
                                  "a nonempty PLATON report must be parsed"))
    viewer = (summary.get("viewer") or {}).get("refined")
    target, _ = _inside(root, viewer) if isinstance(viewer, str) else (None, "missing")
    checks.append(AcceptanceCheck("viewer_refined", target is not None and target.is_file(), "a contained refined-model view is required"))
    notes.extend(("SHELXL/PLATON parser dialects remain unverified until reviewed against actual local outputs.",
                  "Expert-model component regression does not verify initial solution, pair compatibility or scientific correctness."))
    return ComponentAcceptanceVerdict(root, tuple(checks), tuple(notes), require_real)


def verify_component_run(run_dir: Path, *, require_real: bool = True,
                         expected_inputs: Mapping[str, str] | None = None) -> ComponentAcceptanceVerdict:
    """Read-only component evidence check. Malformed archives fail closed, never pass or launch.

    The ordinary de novo verifier is deliberately not called: this track has neither a SHELXT
    process nor a solver-discovered model. No expert reference outside the run is opened.
    """
    try:
        return _verify_component_run(Path(run_dir), require_real=require_real, expected_inputs=expected_inputs)
    except (KeyError, TypeError, ValueError, AttributeError, OSError, RuntimeError) as exc:
        return ComponentAcceptanceVerdict(Path(run_dir), (AcceptanceCheck(
            "component_record_valid", False, f"component evidence is malformed or unreadable: {type(exc).__name__}",
        ),), require_real=require_real)
