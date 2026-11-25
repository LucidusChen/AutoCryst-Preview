"""One preliminary solve, stage by stage: inputs, diagnosis, plan, SHELXT, SHELXL, PLATON, viewer.

This is the job the application service runs on a worker thread and the command line runs in the
foreground. It composes existing core pieces -- the workspace, the diagnosis readers, the
preconditions, the strategy selector, the branch machine with the SHELXT backend -- with the two
new bounded stages (preliminary SHELXL refinement, PLATON diagnostics). It adds no science of its
own beyond two conservative checks, both recorded:

* a SHELXT solution whose chosen space group contains an improper operation is **not refined**
  when the declared chemistry is homochiral (the same rule the input precondition applies);
* the reflections SHELXT wrote for its solution (``<stem>_a.hkl``) are used for refinement; the
  original file is used only when SHELXT states the orientation is unchanged.

Four facts are recorded separately in ``summary.json`` and never collapsed: whether each stage ran
to completion (*process*), what kind of model exists (*model availability*), what the refinement
did (*refinement status*), and whether the model may be ranked or called validated (*scientific
eligibility* -- in this milestone, never).
"""

from __future__ import annotations

import datetime as _dt
import os
import shutil
import socket
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from crystalx_core.app.models import (
    STAGE_ORDER,
    ChemicalPriorInput,
    RunConfig,
    ModelAvailability,
    RefinementStatus,
    RunMode,
    RunPhase,
    SolveRequest,
    StageName,
    StageState,
)
from crystalx_core.app.reflections import ReflectionSource, decide_from_files, orientation_statement
from crystalx_core.app.staging import safe_stem, stage_case
from crystalx_core.app.toolchain import Toolchain
from crystalx_core.domain.chemistry import ChemicalPrior
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.ids import make_stage_id
from crystalx_core.domain.onion import OnionStage, StageKind
from crystalx_core.domain.onion import StageState as OnionStageState
from crystalx_core.domain.radiation import RadiationAssumption
from crystalx_core.domain.result import (
    NoTrustworthyCandidate,
    ReviewCheckpoint,
    ReviewCheckpointKind,
    RunResult,
    StopReason,
    StopReasonKind,
)
from crystalx_core.domain.solving import (
    BackendReadiness,
    BranchBudget,
    BranchState,
    InitialSolutionBranch,
    SolverKind,
)
from crystalx_core.domain.symmetry import ChiralityCompatibility, analyse_symmetry
from crystalx_core.domain.tools import ToolDiagnosticKind, ToolKind
from crystalx_core.errors import CrystalXError
from crystalx_core.geometry.model import build_viewer_payload
from crystalx_core.io.shelx_reader import read_ins_header, scan_hkl
from crystalx_core.io.shelx_res import ResFile, read_res
from crystalx_core.io.shelxl_lst import parse_lst
from crystalx_core.io.shelxt_output import parse_solution_rems, scan_lxt
from crystalx_core.refinement.shelxl import RefinementSettings, stage_refinement
from crystalx_core.solvers.base import BackendRegistry, backend_readiness
from crystalx_core.solvers.context import SolverContext, StagedCase
from crystalx_core.solvers.heavy_atom import heavy_atom_backends
from crystalx_core.solvers.machine import InitialSolutionMachine
from crystalx_core.solvers.selector import SelectionPolicy, StrategySelector
from crystalx_core.solvers.shelxt import ShelxtBackend, ShelxtSolution
from crystalx_core.tools.runner import CancelToken
from crystalx_core.validation.platon import PlatonSettings, parse_chk, validation_prerequisites
from crystalx_core.workflow.preconditions import check_preconditions
from crystalx_core.workspace.events import EventKind
from crystalx_core.workspace.hashing import sha256_file
from crystalx_core.workspace.store import RunWorkspace

__all__ = [
    "SolvePipeline",
    "diagnosis_display",
    "build_registry",
    "plan_display",
    "STATE_FILE",
    "SUMMARY_FILE",
    "REQUEST_FILE",
    "APP_EVENT",
]

STATE_FILE = "state.json"
SUMMARY_FILE = "summary.json"
REQUEST_FILE = "request.json"

#: Event names the application adds to the core's vocabulary.
APP_EVENT = {
    "stage": "app.stage",
    "phase": "app.phase",
    "warning": EventKind.WARNING,
}

_REFINE_DIR = "refinement/shelxl-01"
_VALIDATE_DIR = "validation/platon-01"


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class _Halt(Exception):
    """Stops the stage sequence with a decided phase. Not an error in the program sense."""

    def __init__(self, phase: RunPhase, message: str, stop: StopReasonKind) -> None:
        super().__init__(message)
        self.phase = phase
        self.message = message
        self.stop = stop


@dataclass
class _Solution:
    stem: str
    work_dir: Path
    solution: ShelxtSolution
    alternatives: tuple[ShelxtSolution, ...]
    display: list[dict[str, Any]]


@dataclass
class _Refinement:
    stem: str
    work_dir: Path
    succeeded: bool
    res: ResFile | None


@dataclass
class SolvePipeline:
    """Runs one :class:`SolveRequest` inside an already created :class:`RunWorkspace`."""

    workspace: RunWorkspace
    request: SolveRequest
    toolchain: Toolchain
    cancel: CancelToken = field(default_factory=CancelToken)
    clock: Callable[[], _dt.datetime] = _now
    on_change: Callable[[dict[str, Any]], None] | None = None
    """Called with the new state after every change, for a frontend that wants a push."""

    def __post_init__(self) -> None:
        self.prior: ChemicalPrior = self.request.prior.to_prior()
        self.stem = safe_stem(Path(self.request.ins_path).name)
        self.state: dict[str, Any] = {
            "schema": "crystalx.app.state/1",
            "run_id": str(self.workspace.run_id),
            "mode": self.request.mode.value,
            "label": self.request.label,
            "phase": RunPhase.QUEUED.value,
            "message": "waiting to start",
            "created_at": self.workspace.created_at.isoformat(),
            "started_at": None,
            "finished_at": None,
            "current_stage": None,
            "current_log": None,
            "cancel_requested": False,
            "service": {"pid": os.getpid(), "host": socket.gethostname()},
            "stages": [
                {"name": name.value, "state": StageState.PENDING.value, "message": ""}
                for name in STAGE_ORDER
            ],
        }
        self.summary: dict[str, Any] = {
            "schema": "crystalx.app.summary/1",
            "run_id": str(self.workspace.run_id),
            "mode": self.request.mode.value,
            "mock": self.request.mode is RunMode.MOCK_DEMO,
            "label": self.request.label,
            "stem": self.stem,
            "request": self.request_record(),
            "process": {"phase": RunPhase.QUEUED.value, "message": ""},
            "model_availability": ModelAvailability.NONE.value,
            "refinement_status": RefinementStatus.NOT_RUN.value,
            "scientific_eligibility": {
                "eligible_for_ranking": False,
                "status": "not_evaluated",
                "reason": (
                    "The chemical-validity and coordination gates have no calculator in this "
                    "build and stay NOT_EVALUATED, so no model from this run can be ranked, "
                    "accepted or called validated. It can be inspected and exported."
                ),
                "gates_not_evaluated": ["g4_chemical_validity", "g5_coordination_plausible"],
            },
            "inputs": {},
            "diagnosis": None,
            "preconditions": None,
            "plan": None,
            "solution": None,
            "refinement": None,
            "platon": None,
            "viewer": {},
            "warnings": [],
            "artefacts": [],
        }
        self._diagnosis: DataDiagnosis | None = None
        self._staged: dict[str, Path] = {}
        self._failed_stage: StageName | None = None

    # ------------------------------------------------------------------ public
    def request_record(self) -> dict[str, Any]:
        return {
            "ins": str(self.request.ins_path),
            "hkl": str(self.request.hkl_path),
            "prior": self.request.prior.to_dict(),
            "config": self.request.config.to_dict(),
            "mode": self.request.mode.value,
            "label": self.request.label,
        }

    def write_initial_state(self) -> None:
        """Persist the queued state; the service calls this before the worker starts."""
        self._write_state()

    def run(self) -> dict[str, Any]:
        """Run every stage that can run. Never raises for a scientific or tool failure."""
        self._set_phase(RunPhase.RUNNING, "running")
        self.state["started_at"] = self.clock().isoformat()
        self._write_state()
        halt: _Halt | None = None
        try:
            if self.cancel.cancelled:
                raise _Halt(RunPhase.CANCELLED, "cancelled before start", StopReasonKind.USER_ABORTED)
            self._stage_inputs()
            self._diagnose()
            self._preconditions()
            selection = self._plan()
            solution = self._shelxt(*selection)
            refinement = self._shelxl(solution)
            self._platon(refinement)
            self._viewer(solution, refinement)
        except _Halt as stop:
            halt = stop
        except Exception as exc:  # noqa: BLE001 - a defect must leave a record, not a dead thread
            halt = self._internal_error(exc)
        return self._finish(halt)

    # ------------------------------------------------------------------ stages
    def _stage_inputs(self) -> None:
        self._begin(StageName.STAGE_INPUTS, "copying the inputs into the run")
        try:
            ins = self.workspace.stage_input("ins", self.request.ins_path)
            hkl = self.workspace.stage_input("hkl", self.request.hkl_path)
        except CrystalXError as exc:
            self._end(StageName.STAGE_INPUTS, StageState.FAILED, str(exc))
            raise _Halt(RunPhase.FAILED, f"inputs could not be staged: {exc}", StopReasonKind.INPUT_INVALID)
        self._staged = {"ins": ins.staged_path, "hkl": hkl.staged_path}
        self.summary["inputs"] = {
            role: {
                "original": str(item.original_path),
                "staged": str(item.staged_path.relative_to(self.workspace.layout.root)),
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for role, item in (("ins", ins), ("hkl", hkl))
        }
        self._end(StageName.STAGE_INPUTS, StageState.SUCCEEDED, "read-only copies made and hashed")

    def _diagnose(self) -> None:
        self._check_cancel()
        self._begin(StageName.DIAGNOSE, "reading the instruction and reflection files")
        stage = OnionStage(stage_id=make_stage_id(0, StageKind.DIAGNOSE.value), kind=StageKind.DIAGNOSE, index=0)
        self.workspace.open_stage(stage)
        try:
            header = read_ins_header(self._staged["ins"])
            reflections = scan_hkl(self._staged["hkl"], cell=header.cell, hklf_hint=header.hklf_format)
        except CrystalXError as exc:
            self.workspace.update_stage(stage.with_state(OnionStageState.ABANDONED, summary=str(exc)))
            self._end(StageName.DIAGNOSE, StageState.FAILED, str(exc))
            raise _Halt(RunPhase.FAILED, f"the input could not be read: {exc}", StopReasonKind.INPUT_INVALID)
        diagnosis = DataDiagnosis(reflections=reflections, ins=header)
        self._diagnosis = diagnosis
        self.workspace.write_json(f"stages/{stage.stage_id}/diagnosis.json", diagnosis)
        display = diagnosis_display(diagnosis, self.request.prior)
        self.workspace.write_json(f"stages/{stage.stage_id}/display.json", display)
        self.summary["diagnosis"] = display
        for warning in display["warnings"]:
            self._warn(warning)
        self.workspace.update_stage(
            stage.with_state(OnionStageState.COMPLETED, summary=f"{reflections.n_reflections} reflections")
        )
        self._end(
            StageName.DIAGNOSE,
            StageState.SUCCEEDED,
            f"{reflections.n_reflections} reflections, {display['space_group']['summary']}",
        )

    def _preconditions(self) -> None:
        self._check_cancel()
        self._begin(StageName.PRECONDITIONS, "checking that a solve can give a correct answer")
        assert self._diagnosis is not None
        # Route context: the only route this application executes is SHELXT, which determines the
        # space group within the Laue group, so an impossible declared group is an unresolved
        # starting assumption here; SHELXT's chosen group is checked before refinement.
        report = check_preconditions(self._diagnosis, self.prior, space_group_from_solver=True)
        self.workspace.write_json("stages/s0-diagnose/preconditions.json", report)
        self.summary["preconditions"] = {
            "may_start": report.may_start,
            "summary": report.summary(),
            "items": [
                {
                    "kind": p.kind.value,
                    "blocking": p.blocking,
                    "summary": p.summary,
                    "evidence": p.evidence,
                    "remedy": p.remedy,
                    "override_hint": p.override_hint,
                }
                for p in report.preconditions
            ],
        }
        for advisory in report.advisories:
            self._warn(f"{advisory.summary} -- {advisory.remedy}")
        if not report.may_start:
            self._end(StageName.PRECONDITIONS, StageState.BLOCKED, report.summary())
            raise _Halt(RunPhase.BLOCKED, report.summary(), StopReasonKind.INPUT_INVALID)
        self._end(StageName.PRECONDITIONS, StageState.SUCCEEDED, report.summary())

    def _plan(self) -> tuple[OnionStage, Any, BackendRegistry, Any]:
        self._check_cancel()
        self._begin(StageName.PLAN, "choosing the initial-solution route")
        assert self._diagnosis is not None
        availability = self.toolchain.availability()
        config = self.request.config
        policy = SelectionPolicy(default_branch_budget=BranchBudget(max_wall_clock_s=config.shelxt_timeout_s))
        stage = OnionStage(
            stage_id=make_stage_id(1, StageKind.INITIAL_SOLUTION.value),
            kind=StageKind.INITIAL_SOLUTION,
            index=1,
        )
        trace = StrategySelector(policy).select(
            self._diagnosis, self.prior, stage_id=stage.stage_id, availability=availability
        )
        self.workspace.open_stage(stage)
        self.workspace.write_selection_trace(stage.stage_id, trace)

        shelxt_adapter = self.toolchain.adapter(ToolKind.SHELXT)
        context = SolverContext(
            case=StagedCase(stem=self.stem),
            diagnosis=self._diagnosis,
            prior=self.prior,
            radiation=RadiationAssumption.from_wavelength(self._diagnosis.wavelength_a),
            adapters={ToolKind.SHELXT: shelxt_adapter} if shelxt_adapter else {},
        )
        registry = build_registry(context, config)
        display = plan_display(trace, registry, config)
        self.workspace.write_json(f"stages/{stage.stage_id}/plan.json", display)
        self.summary["plan"] = display
        runnable = [n for n in display["nominated"] if n["will_run"]]
        message = (
            "will run: " + ", ".join(n["solver"] for n in runnable)
            if runnable
            else "no runnable initial-solution route"
        )
        self._end(StageName.PLAN, StageState.SUCCEEDED, message)
        return stage, trace, registry, availability

    def _shelxt(self, stage: OnionStage, trace, registry: BackendRegistry, availability) -> _Solution:
        self._check_cancel()
        config = self.request.config
        shelxt_branch = next((b for b in trace.nominated if b.kind is SolverKind.SHELXT), None)
        if shelxt_branch is None:
            reason = next((r for k, r in trace.excluded if k is SolverKind.SHELXT), "")
            resolution = self.toolchain.resolution(ToolKind.SHELXT)
            if resolution is not None and not resolution.is_usable:
                reason = resolution.diagnostic.message + (
                    f" Remedy: {resolution.diagnostic.remedy}" if resolution.diagnostic.remedy else ""
                )
            self.workspace.update_stage(stage.with_state(OnionStageState.ABANDONED, summary=reason))
            self._begin(StageName.SHELXT, "")
            self._end(StageName.SHELXT, StageState.TOOL_UNAVAILABLE, reason or "SHELXT was not nominated")
            raise _Halt(
                RunPhase.BLOCKED,
                f"SHELXT cannot be used on this machine: {reason}",
                StopReasonKind.TOOL_UNAVAILABLE,
            )

        self._begin(StageName.SHELXT, "SHELXT is searching for an initial solution")
        branch_dir = self.workspace.layout.branch_dir(stage.stage_id, shelxt_branch.branch_id)
        adapters = {}
        adapter = self.toolchain.adapter(ToolKind.SHELXT)
        if adapter is not None:
            adapters[ToolKind.SHELXT] = adapter

        def prepare(branch: InitialSolutionBranch, work_dir: Path) -> None:
            staged = stage_case(
                work_dir=work_dir,
                stem=self.stem,
                ins_source=self._staged["ins"],
                hkl_source=self._staged["hkl"],
                composition=self.request.prior.composition,
            )
            self.workspace.write_json(
                f"stages/{branch.stage_id}/branches/{branch.branch_id}/staging.json", staged.to_record()
            )
            for change in staged.changes:
                self._warn(f"composition override applied to the SHELXT input: {change}")
            if branch.kind is SolverKind.SHELXT:
                self.state["current_log"] = str(
                    (work_dir / "shelxt.stdout.log").relative_to(self.workspace.layout.root)
                )
                self._write_state()

        machine = InitialSolutionMachine(
            registry,
            self.workspace,
            availability=availability,
            adapters=adapters,
            cancel=self.cancel,
            limits=config.limits_for(config.shelxt_timeout_s),
            prepare=prepare,
        )
        portfolio = machine.run_portfolio(stage, trace)
        self.state["current_log"] = None
        report = next((r for r in portfolio.reports if r.branch.kind is SolverKind.SHELXT), None)
        for other in portfolio.reports:
            if other.branch.kind is not SolverKind.SHELXT:
                self._note(f"{other.branch.kind.value}: {other.outcome.summary}")

        if report is None:  # pragma: no cover - nominated above
            raise RuntimeError("SHELXT branch missing from the portfolio")
        state = report.branch.state
        if state is BranchState.ABORTED:
            self.workspace.update_stage(stage.with_state(OnionStageState.ABANDONED, summary="cancelled"))
            self._end(StageName.SHELXT, StageState.CANCELLED, report.outcome.summary)
            raise _Halt(RunPhase.CANCELLED, "cancelled during SHELXT", StopReasonKind.USER_ABORTED)
        if state is BranchState.BLOCKED_TOOL_UNAVAILABLE:
            self.workspace.update_stage(stage.with_state(OnionStageState.ABANDONED, summary=report.outcome.summary))
            self._end(StageName.SHELXT, StageState.TOOL_UNAVAILABLE, report.outcome.summary)
            raise _Halt(RunPhase.BLOCKED, report.outcome.summary, StopReasonKind.TOOL_UNAVAILABLE)
        if state is not BranchState.SUCCEEDED:
            message = report.outcome.failure_reason or report.outcome.summary
            code = report.outcome.diagnostic_code
            if code == ToolDiagnosticKind.ARCH_MISMATCH.value:
                message += " -- " + _launch_refusal_classification(self.toolchain)
            self.workspace.update_stage(stage.with_state(OnionStageState.ABANDONED, summary=message))
            self._end(StageName.SHELXT, StageState.FAILED, message)
            raise _Halt(RunPhase.FAILED, f"SHELXT did not produce a solution: {message}", StopReasonKind.ALL_BRANCHES_EXHAUSTED)

        backend = registry.get(SolverKind.SHELXT)
        assert isinstance(backend, ShelxtBackend)
        solutions, read_warnings = backend.collect_solutions(branch_dir)
        for warning in read_warnings:
            self._warn(f"SHELXT output: {warning}")
        root = self.workspace.layout.root
        display = []
        for sol in solutions:
            rem = parse_solution_rems(sol.res.rem_lines)
            symmetry = analyse_symmetry(sol.res.header.latt, sol.res.header.symm)
            display.append(
                {
                    "letter": sol.letter,
                    "is_primary": sol.is_primary,
                    "file": str(sol.res_path.relative_to(root)),
                    "space_group": rem.space_group,
                    "r1": rem.r1,
                    "r_weak": rem.r_weak,
                    "alpha": rem.alpha,
                    "absences_mean_i_over_sigma": rem.mean_i_over_sigma_absences,
                    "n_systematic_absences": rem.n_systematic_absences,
                    "formula": rem.formula,
                    "n_atoms": sol.n_atoms,
                    "elements": list(sol.elements),
                    "concerns": list(rem.concerns()),
                    "chirality": symmetry.compatibility.value,
                    "symmetry_explanation": symmetry.explain(),
                    "orientation_as_input": orientation_statement(sol.res.rem_lines) is not None,
                }
            )
        lxt = scan_lxt(branch_dir / f"{self.stem}.lxt")
        self.summary["solution"] = {
            "branch": str(shelxt_branch.branch_id),
            "work_dir": str(branch_dir.relative_to(root)),
            "solutions": display,
            "ordering_note": "Solutions are listed in SHELXT's own order (a = SHELXT's best).",
            "lxt": {
                "found": lxt.source.is_file(),
                "trials": [
                    {"space_group": t.space_group, "r1": t.r1, "r_weak": t.r_weak, "alpha": t.alpha, "cfom": t.cfom}
                    for t in lxt.trials[:20]
                ],
                "format_verified": lxt.format_verified,
                "warnings": list(lxt.warnings),
            },
            "element_note": (
                "SHELXT assigns elements from peak heights. They are a starting guess, not a "
                "determination."
            ),
        }
        primary = solutions[0]
        self.summary["model_availability"] = ModelAvailability.SOLUTION_MODEL.value
        self._add_artefacts(branch_dir, (f"{self.stem}_{s.letter}.res" for s in solutions))
        self._add_artefacts(branch_dir, (f"{self.stem}.lxt", "shelxt.stdout.log", "shelxt.stderr.log"))
        self.workspace.update_stage(
            stage.with_state(OnionStageState.COMPLETED, summary=f"{len(solutions)} SHELXT solution(s)")
        )
        sg = display[0]["space_group"] or "space group not stated"
        self._end(
            StageName.SHELXT,
            StageState.SUCCEEDED,
            f"{len(solutions)} solution(s); primary in {sg} with {primary.n_atoms} atoms",
        )
        return _Solution(self.stem, branch_dir, primary, tuple(solutions[1:]), display)

    def _reflection_decision(self, solution: _Solution):
        """Select reflection lineage for this pipeline's declared source model."""
        return decide_from_files(solution.work_dir, solution.stem, solution.solution.letter)

    def _shelxl(self, solution: _Solution) -> _Refinement | None:
        self._check_cancel()
        config = self.request.config
        self._begin(StageName.SHELXL, "preparing a bounded isotropic refinement")
        root = self.workspace.layout.root

        enantiopure, why = self.prior.is_enantiopure()
        primary_display = solution.display[0]
        if enantiopure and primary_display["chirality"] == ChiralityCompatibility.UNKNOWN.value:
            message = (
                "The space group SHELXT chose could not be classified "
                f"({primary_display['symmetry_explanation']}), so it cannot be confirmed to admit a "
                f"homochiral structure ({why}). The chirality check stays unresolved and the solution "
                "is not refined."
            )
            self.summary["refinement"] = {"blocked": True, "reason": message}
            self._end(StageName.SHELXL, StageState.BLOCKED, message)
            self._warn(message)
            return None
        if enantiopure and primary_display["chirality"] == ChiralityCompatibility.CONTAINS_IMPROPER.value:
            sohncke = [d["letter"] for d in solution.display[1:] if d["chirality"] == ChiralityCompatibility.SOHNCKE.value]
            message = (
                f"SHELXT's primary solution is in a space group with an improper operation "
                f"({primary_display['symmetry_explanation']}), which a homochiral material cannot "
                f"occupy ({why}). It is not refined. "
                + (
                    f"SHELXT also wrote Sohncke-compatible solution(s) {', '.join(sohncke)}; "
                    "review them, or declare the preparation racemic if it is."
                    if sohncke
                    else "No Sohncke-compatible alternative was written; review the space group."
                )
            )
            self.summary["refinement"] = {"blocked": True, "reason": message}
            self._end(StageName.SHELXL, StageState.BLOCKED, message)
            self._warn(message)
            return None

        # One contract with `crystalx verify-run` (app/reflections.py): SHELXT's matching .hkl, or
        # the original reflections only under the evidenced 'Orientation as input' fallback.
        decision = self._reflection_decision(solution)
        if not decision.allowed:
            self.summary["refinement"] = {
                "blocked": True,
                "reason": decision.reason,
                "reflections_decision": decision.to_record(),
            }
            self._end(StageName.SHELXL, StageState.BLOCKED, decision.reason)
            return None
        hkl_for_refinement = solution.work_dir / decision.file_name
        hkl_note = decision.note
        if decision.source is ReflectionSource.ORIGINAL_ORIENTATION_AS_INPUT:
            self._warn(f"SHELXT wrote no matching .hkl; refining against the {hkl_note}")

        adapter = self.toolchain.adapter(ToolKind.SHELXL)
        if adapter is None:
            resolution = self.toolchain.resolution(ToolKind.SHELXL)
            message = resolution.diagnostic.message if resolution else "shelxl was not checked"
            if resolution is not None and resolution.diagnostic.remedy:
                message += f" Remedy: {resolution.diagnostic.remedy}"
            self.summary["refinement"] = {"blocked": True, "reason": message, "tool_unavailable": True}
            self._end(StageName.SHELXL, StageState.TOOL_UNAVAILABLE, message)
            return None

        refine_dir = self.workspace.artefact_path(*_REFINE_DIR.split("/"))
        refine_dir.mkdir(parents=True, exist_ok=False)
        settings = RefinementSettings(
            cycles=config.shelxl_cycles,
            timeout_s=config.shelxl_timeout_s,
            limits=config.limits_for(config.shelxl_timeout_s),
        )
        prepared = stage_refinement(
            solution_res=solution.solution.res_path,
            hkl=hkl_for_refinement,
            workdir=refine_dir,
            stem=solution.stem,
            settings=settings,
        )
        lineage = {
            **decision.to_record(),
            "solution_letter": solution.solution.letter,
            "solution_file": str(solution.solution.res_path.relative_to(root)),
            "solution_sha256": sha256_file(solution.solution.res_path),
            "reflections_file": str(hkl_for_refinement.relative_to(root)),
            "reflections_sha256": sha256_file(hkl_for_refinement),
            "refinement_copy": str(prepared.hkl_path.relative_to(root)),
            "refinement_copy_sha256": sha256_file(prepared.hkl_path),
        }
        self.workspace.write_json(
            f"{_REFINE_DIR}/prepared.json",
            {
                "settings": settings.to_record(),
                "source_solution": str(solution.solution.res_path.relative_to(root)),
                "reflections": str(hkl_for_refinement.relative_to(root)),
                "reflections_note": hkl_note,
                "reflections_lineage": lineage,
                "changes": list(prepared.changes),
                "atoms_kept": prepared.atoms_kept,
                "q_peaks_removed": prepared.q_peaks_removed,
            },
        )
        if lineage["refinement_copy_sha256"] != lineage["reflections_sha256"]:
            message = "the refinement copy of the reflections does not match its source; not refined"
            self.summary["refinement"] = {"blocked": True, "reason": message, "reflections_lineage": lineage}
            self._end(StageName.SHELXL, StageState.FAILED, message)
            return None
        plan = adapter.build_plan(
            args=(solution.stem,),
            cwd=refine_dir,
            timeout_s=config.shelxl_timeout_s,
            expected_outputs=prepared.expected_outputs,
            declared_inputs=(prepared.ins_path.name, prepared.hkl_path.name),
            label="shelxl",
            limits=settings.limits,
        )
        self.workspace.write_json(f"{_REFINE_DIR}/command.json", plan.to_record())
        self._set_stage_message(StageName.SHELXL, f"SHELXL: {config.shelxl_cycles} isotropic cycles")
        self.state["current_log"] = f"{_REFINE_DIR}/shelxl.stdout.log"
        self._write_state()
        self.workspace.events.emit(
            EventKind.COMMAND_STARTED, " ".join(plan.argv), argv=list(plan.argv), cwd=str(plan.cwd)
        )
        outcome, record = adapter.invoke(plan, log_dir=refine_dir, cancel=self.cancel)
        self.state["current_log"] = None
        self.workspace.write_json(f"{_REFINE_DIR}/provenance.json", record)
        self.workspace.write_json(f"{_REFINE_DIR}/result.json", outcome.to_record())
        self.workspace.events.emit(
            EventKind.COMMAND_FINISHED,
            outcome.diagnostic.message,
            diagnostic=outcome.diagnostic.kind.value,
            exit_code=outcome.exit_code,
            duration_s=round(outcome.duration_s, 3),
        )
        self._add_artefacts(
            refine_dir,
            (f"{solution.stem}.ins", f"{solution.stem}.res", f"{solution.stem}.lst",
             f"{solution.stem}.cif", f"{solution.stem}.fcf", "shelxl.stdout.log", "shelxl.stderr.log"),
        )
        if outcome.cancelled:
            self._end(StageName.SHELXL, StageState.CANCELLED, outcome.diagnostic.message)
            raise _Halt(RunPhase.CANCELLED, "cancelled during SHELXL", StopReasonKind.USER_ABORTED)

        listing = parse_lst(refine_dir / f"{solution.stem}.lst")
        res: ResFile | None = None
        res_error = ""
        if (refine_dir / f"{solution.stem}.res").is_file():
            try:
                res = read_res(refine_dir / f"{solution.stem}.res")
            except CrystalXError as exc:
                res_error = str(exc)
        agreement = _compare_listing_and_res(listing, res)
        detail: dict[str, Any] = {
            "settings": settings.to_record(),
            "work_dir": _REFINE_DIR,
            "reflections_note": hkl_note,
            "reflections_lineage": lineage,
            "changes": list(prepared.changes),
            "diagnostic": outcome.diagnostic.kind.value,
            "diagnostic_message": outcome.diagnostic.message,
            "exit_code": outcome.exit_code,
            "duration_s": round(outcome.duration_s, 3),
            "listing": listing.to_record(),
            "res_r_factors": _res_r_factors(res),
            "res_parse_error": res_error,
            "agreement": agreement,
            "preliminary_note": (
                "Isotropic, fixed cycle count, no hydrogens, no restraints, no disorder. A low "
                "R1 here says the heavy atoms are roughly right; it says nothing about the light "
                "atoms, the hand or the space group."
            ),
        }
        succeeded = outcome.succeeded and res is not None and res.n_atoms > 0
        if not succeeded or listing.errors:
            if not succeeded:
                status = RefinementStatus.FAILED
                message = outcome.diagnostic.message
            else:
                status = RefinementStatus.ERRORS_REPORTED
                message = (
                    "SHELXL reported a recognised error; the refined model is not offered (message "
                    "classification is a keyword heuristic, not yet verified against real output)"
                )
            if listing.errors:
                message += " | SHELXL reported: " + "; ".join(listing.errors[:3])
            if res_error:
                message += f" | the .res could not be read: {res_error}"
            detail["status"] = status.value
            self.summary["refinement"] = detail
            self.summary["refinement_status"] = status.value
            self._end(StageName.SHELXL, StageState.FAILED, message)
            return _Refinement(solution.stem, refine_dir, False, res)

        status = RefinementStatus.RAN_CONVERGENCE_NOT_ASSESSED
        for item in agreement.get("disagreements", []):
            self._warn(f"SHELXL listing and .res disagree: {item}")
        for item in listing.warnings[:5]:
            self._warn(f"SHELXL advice/warning: {item}")
        for item in listing.notices[:5]:
            self._warn(f"SHELXL message not classified (review the .lst): {item}")
        detail["status"] = status.value
        self.summary["refinement"] = detail
        self.summary["refinement_status"] = status.value
        self.summary["model_availability"] = ModelAvailability.PRELIMINARY_REFINED_MODEL.value
        r1 = listing.r1_observed if listing.r1_observed is not None else (res.r_factors.r1_observed if res else None)
        shifts = f"mean shift/su {listing.mean_shift_su}, max {listing.max_shift_su}"
        self._end(
            StageName.SHELXL,
            StageState.SUCCEEDED,
            f"{status.value}; R1 = {r1 if r1 is not None else 'unknown'}; {shifts}",
        )
        return _Refinement(solution.stem, refine_dir, True, res)

    def _platon(self, refinement: _Refinement | None) -> None:
        self._check_cancel()
        config = self.request.config
        self._begin(StageName.PLATON, "checking the refinement output with PLATON")
        if refinement is None or not refinement.succeeded:
            self.summary["platon"] = {"skipped": True, "reason": "no successful refinement to check"}
            self._end(StageName.PLATON, StageState.SKIPPED, "no successful refinement to check")
            return
        missing = validation_prerequisites(refinement.work_dir, refinement.stem)
        if missing:
            self.summary["platon"] = {"blocked": True, "reasons": missing}
            self._end(StageName.PLATON, StageState.BLOCKED, " ".join(missing))
            return
        adapter = self.toolchain.platon_adapter()
        if adapter is None:
            resolution = self.toolchain.platon()
            message = resolution.diagnostic.message if resolution else "PLATON was not checked"
            self.summary["platon"] = {"blocked": True, "reasons": [message], "tool_unavailable": True}
            self._end(StageName.PLATON, StageState.TOOL_UNAVAILABLE, message)
            return

        val_dir = self.workspace.artefact_path(*_VALIDATE_DIR.split("/"))
        val_dir.mkdir(parents=True, exist_ok=False)
        stem = refinement.stem
        for suffix in (".cif", ".fcf"):
            shutil.copyfile(refinement.work_dir / f"{stem}{suffix}", val_dir / f"{stem}{suffix}")
        settings = PlatonSettings(
            timeout_s=config.platon_timeout_s, limits=config.limits_for(config.platon_timeout_s)
        )
        plan = adapter.build_plan(
            args=("-u", f"{stem}.cif"),
            cwd=val_dir,
            timeout_s=config.platon_timeout_s,
            expected_outputs=(f"{stem}.chk",),
            declared_inputs=(f"{stem}.cif", f"{stem}.fcf"),
            env_unset=("DISPLAY", "CHECKDEF"),
            label="platon",
            limits=settings.limits,
        )
        self.workspace.write_json(f"{_VALIDATE_DIR}/command.json", plan.to_record())
        self.workspace.write_json(f"{_VALIDATE_DIR}/settings.json", settings.to_record())
        self.state["current_log"] = f"{_VALIDATE_DIR}/platon.stdout.log"
        self._write_state()
        self.workspace.events.emit(
            EventKind.COMMAND_STARTED, " ".join(plan.argv), argv=list(plan.argv), cwd=str(plan.cwd)
        )
        outcome, record = adapter.invoke(plan, log_dir=val_dir, cancel=self.cancel)
        self.state["current_log"] = None
        self.workspace.write_json(f"{_VALIDATE_DIR}/provenance.json", record)
        self.workspace.write_json(f"{_VALIDATE_DIR}/result.json", outcome.to_record())
        self.workspace.events.emit(
            EventKind.COMMAND_FINISHED,
            outcome.diagnostic.message,
            diagnostic=outcome.diagnostic.kind.value,
            exit_code=outcome.exit_code,
            duration_s=round(outcome.duration_s, 3),
        )
        self._add_artefacts(val_dir, (f"{stem}.chk", "platon.stdout.log", "platon.stderr.log"))
        if outcome.cancelled:
            self._end(StageName.PLATON, StageState.CANCELLED, outcome.diagnostic.message)
            raise _Halt(RunPhase.CANCELLED, "cancelled during PLATON", StopReasonKind.USER_ABORTED)
        report = parse_chk(val_dir / f"{stem}.chk")
        self.workspace.write_json(f"{_VALIDATE_DIR}/report.json", report.to_record())
        detail = {
            "program": adapter.kind.value,
            "diagnostic": outcome.diagnostic.kind.value,
            "diagnostic_message": outcome.diagnostic.message,
            "duration_s": round(outcome.duration_s, 3),
            "report": report.to_record(),
            "note": (
                "An ALERT is a prompt to look, not a verdict. Zero level-A alerts on a preliminary "
                "isotropic model is not validation."
            ),
        }
        self.summary["platon"] = detail
        if not outcome.succeeded:
            self._end(StageName.PLATON, StageState.FAILED, outcome.diagnostic.message)
            return
        counts = report.counts
        self._end(
            StageName.PLATON,
            StageState.SUCCEEDED,
            "alerts: " + ", ".join(f"{level} {counts[level]}" for level in "ABCG"),
        )

    def _viewer(self, solution: _Solution, refinement: _Refinement | None) -> None:
        self._begin(StageName.VIEWER, "building the 3-D model views")
        written = []
        sources = [("solution", solution.solution.res, "SHELXT solution (unrefined)")]
        if refinement is not None and refinement.succeeded and refinement.res is not None:
            sources.append(("refined", refinement.res, "preliminary SHELXL refinement"))
        for name, res, label in sources:
            try:
                payload = build_viewer_payload(res, stage=name, source_label=label)
            except ValueError as exc:
                self._warn(f"viewer: {name}: {exc}")
                continue
            payload["mock"] = self.request.mode is RunMode.MOCK_DEMO
            self.workspace.write_json(f"viewer/{name}.json", payload)
            self.summary["viewer"][name] = f"viewer/{name}.json"
            written.append(name)
        if not written:
            self._end(StageName.VIEWER, StageState.FAILED, "no model could be prepared for display")
            return
        self._end(StageName.VIEWER, StageState.SUCCEEDED, "views: " + ", ".join(written))

    # ------------------------------------------------------------------ finishing
    def _internal_error(self, exc: Exception) -> _Halt:
        text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        try:
            self.workspace.write_text("errors/internal_error.txt", text, allow_replace=True)
        except Exception:  # noqa: BLE001 - the record of the record failing is the state file
            pass
        current = self.state.get("current_stage")
        if current:
            self._end(StageName(current), StageState.FAILED, f"internal error: {exc}")
        return _Halt(RunPhase.FAILED, f"internal error: {exc}", StopReasonKind.INTERNAL_ERROR)

    def _finish(self, halt: _Halt | None) -> dict[str, Any]:
        if halt is None:
            failed = [s for s in self.state["stages"] if s["state"] == StageState.FAILED.value]
            phase = RunPhase.FAILED if failed else RunPhase.COMPLETED
            message = (
                f"{failed[0]['name']} failed: {failed[0]['message']}" if failed else self._completion_message()
            )
            stop = StopReasonKind.REVIEW_CHECKPOINT
        else:
            phase, message, stop = halt.phase, halt.message, halt.stop
        for stage in self.state["stages"]:
            if stage["state"] == StageState.PENDING.value:
                stage["state"] = StageState.SKIPPED.value
                stage["message"] = stage["message"] or "not reached"
            elif stage["state"] == StageState.RUNNING.value:
                stage["state"] = (
                    StageState.CANCELLED.value if phase is RunPhase.CANCELLED else StageState.FAILED.value
                )
        self._verify_inputs()
        self.summary["process"] = {"phase": phase.value, "message": message, "stop_reason": stop.value}
        self.summary["stages"] = [dict(s) for s in self.state["stages"]]
        self.state["finished_at"] = self.clock().isoformat()
        self.state["current_stage"] = None
        self.state["current_log"] = None
        self._set_phase(phase, message)
        self._write_summary()
        self._write_state()
        result = RunResult(
            run_id=self.workspace.run_id,
            stop_reason=StopReason(
                kind=stop,
                message=message,
                suggested_next_action=(
                    "review the preliminary model; it is not validated"
                    if phase is RunPhase.COMPLETED
                    else ""
                ),
            ),
            review_checkpoints=(
                (
                    ReviewCheckpoint(
                        kind=ReviewCheckpointKind.SEVERE_VALIDATION_ALERT
                        if self._has_severe_alerts()
                        else ReviewCheckpointKind.DATA_QUALITY,
                        question=(
                            "This run produced a preliminary, unvalidated model. A crystallographer "
                            "must review it before anything is concluded from it."
                        ),
                        blocking=True,
                    ),
                )
                if self.summary["model_availability"] != ModelAvailability.NONE.value
                else ()
            ),
            no_trustworthy_candidate=NoTrustworthyCandidate(
                reason=self.summary["scientific_eligibility"]["reason"], requires_human_expert=True
            ),
            stages_completed=tuple(
                s for s in (make_stage_id(0, "diagnose"), make_stage_id(1, "initial_solution"))
                if self.workspace.layout.stage_json(s).exists()
            ),
            notes=tuple(self.summary["warnings"][:50]),
            metadata={
                "app_phase": phase.value,
                "mode": self.request.mode.value,
                "model_availability": self.summary["model_availability"],
                "refinement_status": self.summary["refinement_status"],
            },
        )
        try:
            self.workspace.finalize(result)
        except CrystalXError as exc:  # pragma: no cover - finalize twice is a defect
            self._warn(f"could not finalize the run record: {exc}")
        return self.summary

    def _completion_message(self) -> str:
        model = self.summary["model_availability"]
        if model == ModelAvailability.PRELIMINARY_REFINED_MODEL.value:
            return "finished: preliminary refined model available (not validated)"
        if model == ModelAvailability.SOLUTION_MODEL.value:
            return "finished: SHELXT solution available; no refined model"
        return "finished without a model"

    def _has_severe_alerts(self) -> bool:
        platon = self.summary.get("platon") or {}
        counts = (platon.get("report") or {}).get("counts") or {}
        return bool(counts.get("A") or counts.get("B"))

    def _verify_inputs(self) -> None:
        """Re-hash the user's originals and the run's copies; report any change, never repair it."""
        checks = {}
        for role, info in self.summary.get("inputs", {}).items():
            original = Path(info["original"])
            staged = self.workspace.layout.root / info["staged"]
            entry = {"recorded_sha256": info["sha256"]}
            try:
                entry["original_unchanged"] = sha256_file(original) == info["sha256"]
            except OSError:
                entry["original_unchanged"] = None
                entry["original_note"] = "the original could not be read at the end of the run"
            try:
                entry["staged_copy_unchanged"] = sha256_file(staged) == info["sha256"]
            except OSError:
                entry["staged_copy_unchanged"] = False
            if entry.get("original_unchanged") is False:
                self._warn(f"the original {role} file changed during the run; the run used its copy")
            if entry["staged_copy_unchanged"] is False:
                self._warn(f"the run's copy of the {role} file changed; this run's record is suspect")
            checks[role] = entry
        self.summary["input_verification"] = checks

    # ------------------------------------------------------------------ state helpers
    def _check_cancel(self) -> None:
        if self.cancel.cancelled:
            raise _Halt(RunPhase.CANCELLED, self.cancel.reason or "cancelled", StopReasonKind.USER_ABORTED)

    def _stage_entry(self, name: StageName) -> dict[str, Any]:
        return next(s for s in self.state["stages"] if s["name"] == name.value)

    def _begin(self, name: StageName, message: str) -> None:
        entry = self._stage_entry(name)
        entry.update(state=StageState.RUNNING.value, message=message, started_at=self.clock().isoformat())
        self.state["current_stage"] = name.value
        self.workspace.events.emit(APP_EVENT["stage"], message or name.value, stage=name.value, state="running")
        self._write_state()

    def _set_stage_message(self, name: StageName, message: str) -> None:
        self._stage_entry(name)["message"] = message
        self._write_state()

    def _end(self, name: StageName, state: StageState, message: str) -> None:
        entry = self._stage_entry(name)
        entry.update(state=state.value, message=message, finished_at=self.clock().isoformat())
        if self.state.get("current_stage") == name.value:
            self.state["current_stage"] = None
        self.workspace.events.emit(APP_EVENT["stage"], message, stage=name.value, state=state.value)
        self._write_state()
        self._write_summary()

    def _set_phase(self, phase: RunPhase, message: str) -> None:
        self.state["phase"] = phase.value
        self.state["message"] = message
        self.summary["process"]["phase"] = phase.value
        self.summary["process"]["message"] = message
        self.workspace.events.emit(APP_EVENT["phase"], message, phase=phase.value)

    def _warn(self, message: str) -> None:
        if message not in self.summary["warnings"]:
            self.summary["warnings"].append(message)
            self.workspace.events.emit(APP_EVENT["warning"], message)

    def _note(self, message: str) -> None:
        self.summary.setdefault("notes", []).append(message)

    def _add_artefacts(self, directory: Path, names) -> None:
        root = self.workspace.layout.root
        for name in names:
            path = directory / name
            if path.is_file():
                relative = str(path.relative_to(root))
                if relative not in self.summary["artefacts"]:
                    self.summary["artefacts"].append(relative)

    def _write_state(self) -> None:
        self.state["updated_at"] = self.clock().isoformat()
        self.state["cancel_requested"] = self.cancel.cancelled
        if not self.workspace.is_finalized:
            self.workspace.write_json(STATE_FILE, self.state, allow_replace=True)
        if self.on_change is not None:
            try:
                self.on_change(dict(self.state))
            except Exception:  # noqa: BLE001 - a frontend callback must not break the run
                pass

    def _write_summary(self) -> None:
        if not self.workspace.is_finalized:
            self.workspace.write_json(SUMMARY_FILE, self.summary, allow_replace=True)

def diagnosis_display(diagnosis: DataDiagnosis, prior_input: ChemicalPriorInput) -> dict[str, Any]:
    """The facts a person needs about the input, with every warning stated in words."""
    prior = prior_input.to_prior()
    header = diagnosis.ins
    refl = diagnosis.reflections
    symmetry = header.symmetry()
    trust = header.composition_trust(
        declared_metals=prior.metal_elements, declared_elements=prior.declared_elements
    )
    radiation = RadiationAssumption.from_wavelength(header.wavelength_a)
    warnings = [*header.warnings, *refl.warnings]
    composition_override = bool(prior_input.composition)
    missing = [*trust.missing_declared_metals, *trust.missing_declared_elements]
    if missing and not composition_override:
        warnings.append(
            f"Declared element(s) {', '.join(missing)} are not in the file's SFAC. SHELXT places "
            "atoms only for elements it is told about, so they cannot appear in the solution. "
            "Add them with a composition override if they are really present."
        )
    if symmetry.is_trivially_sohncke:
        warnings.append(
            "The file declares P1 (LATT -1, no SYMM). SHELXT will consider only Laue group -1; "
            "if the true symmetry is higher, the instruction file must say so."
        )
    sg_parts = []
    if header.latt is not None:
        sg_parts.append(f"LATT {header.latt}")
    sg_parts.append(f"{len(header.symm)} SYMM")
    cell = header.cell
    return {
        "title": header.title,
        "cell": (
            {"a": cell.a, "b": cell.b, "c": cell.c, "alpha": cell.alpha, "beta": cell.beta,
             "gamma": cell.gamma, "volume": round(cell.volume, 2)}
            if cell
            else None
        ),
        "wavelength_a": header.wavelength_a,
        "radiation": radiation.source.value,
        "radiation_assumed": radiation.is_assumed,
        "space_group": {
            "summary": ", ".join(sg_parts),
            "latt": header.latt,
            "symm": list(header.symm),
            "confidence": header.space_group_confidence.value,
            "chirality": symmetry.compatibility.value,
            "explanation": symmetry.explain(),
            "centrosymmetric": symmetry.is_centrosymmetric,
        },
        "composition": {
            "sfac": list(header.sfac),
            "unit": list(header.unit),
            "trusted": trust.trusted,
            "detail": trust.detail,
            "override": [list(pair) for pair in prior_input.composition],
        },
        "reflections": {
            "count": refl.n_reflections,
            "hklf": int(refl.hklf_format),
            "anomalous": refl.anomalous.value,
            "friedel_pairs": refl.n_friedel_pairs,
            "d_min_a": refl.d_min_a,
            "d_max_a": refl.d_max_a,
            "mean_i_over_sigma": refl.mean_i_over_sigma,
            "completeness": refl.completeness,
            "completeness_unavailable": (
                refl.completeness_unavailable.reason if refl.completeness_unavailable else None
            ),
        },
        "prior": prior_input.to_dict(),
        "enantiopure": dict(zip(("value", "reason"), prior.is_enantiopure())),
        "warnings": warnings,
    }


def build_registry(context: SolverContext, config: RunConfig) -> BackendRegistry:
    """SHELXT (production) plus the plan-only heavy-atom routes, so the plan can say which is which."""
    registry = BackendRegistry()
    registry.register(ShelxtBackend(context, timeout_s=config.shelxt_timeout_s))
    for backend in heavy_atom_backends(context):
        registry.register(backend)
    return registry


def plan_display(trace, registry: BackendRegistry, config: RunConfig) -> dict[str, Any]:
    """The selector's decision, each route's readiness, and the downstream bounded steps."""

    def readiness(kind: SolverKind) -> str:
        if not registry.has(kind):
            return "no_backend"
        return backend_readiness(registry.get(kind)).value

    return {
        "nominated": [
            {
                "branch_id": str(b.branch_id),
                "solver": b.kind.value,
                "priority": b.priority,
                "readiness": readiness(b.kind),
                "will_run": readiness(b.kind) == BackendReadiness.PRODUCTION.value,
                "reasons": list(b.nominated_because),
                "time_budget_s": b.budget.max_wall_clock_s,
            }
            for b in trace.nominated
        ],
        "excluded": [{"solver": k.value, "reason": r} for k, r in trace.excluded],
        "rules": [
            {"rule_id": r.rule_id, "verdict": r.verdict.value, "detail": r.detail} for r in trace.rules
        ],
        "notes": list(trace.notes),
        "downstream": [
            {
                "step": "shelxl",
                "description": (
                    f"{config.shelxl_cycles} isotropic least-squares cycles (L.S. "
                    f"{config.shelxl_cycles}), LIST 4, ACTA, PLAN 20 -- a preliminary "
                    "refinement, not a refinement strategy"
                ),
                "time_budget_s": config.shelxl_timeout_s,
            },
            {
                "step": "platon",
                "description": "platon -u <stem>.cif, only when the CIF and the FCF both exist",
                "time_budget_s": config.platon_timeout_s,
            },
        ],
        "limits": {
            "shelxt": config.limits_for(config.shelxt_timeout_s).to_record(),
            "shelxl": config.limits_for(config.shelxl_timeout_s).to_record(),
            "platon": config.limits_for(config.platon_timeout_s).to_record(),
            "cpu_note": (
                "CPU time is capped per program at its wall-clock budget times the number of "
                "logical CPUs unless set explicitly, because SHELXT runs several threads."
            ),
        },
        "laue_note": (
            "SHELXT reads the Laue group from LATT/SYMM and tests every space group within it, "
            "so the declared space group is a starting point, not a constraint. The "
            "instruction file is passed to SHELXT unmodified."
        ),
    }


def _launch_refusal_classification(toolchain: Toolchain) -> str:
    """Name the obstacle when the operating system refused to execute a program."""
    host = toolchain.availability().host
    if host.system == "Darwin" and host.arch.value == "arm64":
        marker = (
            "a Rosetta 2 marker file is present, so the static check passed, but translation did "
            "not happen"
            if host.rosetta_available
            else "no Rosetta 2 marker was found"
        )
        return (
            "classification: host translation. macOS refused to execute an x86_64 program on "
            f"this arm64 Mac ({marker}). Installing or repairing Rosetta 2 is a system change "
            "for the owner to decide; a native arm64 build would remove the dependency."
        )
    return (
        "classification: binary format. The operating system refused to execute the program on "
        f"this {host.system} {host.arch.value} host; it was built for another platform."
    )


def _res_r_factors(res: ResFile | None) -> dict[str, Any] | None:
    if res is None:
        return None
    r = res.r_factors
    return {
        "r1_observed": r.r1_observed,
        "r1_all": r.r1_all,
        "n_observed": r.n_observed,
        "n_all": r.n_all,
        "wr2": r.wr2,
        "goof": r.goof,
        "n_parameters": r.n_parameters,
        "highest_peak": r.highest_peak,
        "deepest_hole": r.deepest_hole,
        "q_peaks": len(res.q_peaks),
        "atoms": res.n_atoms,
    }


def _compare_listing_and_res(listing, res: ResFile | None) -> dict[str, Any]:
    """Compare the headline numbers SHELXL wrote in two places. Report, never reconcile."""
    if res is None:
        return {"compared": False, "reason": "no .res to compare with"}
    pairs = (
        ("R1(obs)", listing.r1_observed, res.r_factors.r1_observed),
        ("wR2", listing.wr2, res.r_factors.wr2),
        ("GooF", listing.goof, res.r_factors.goof),
    )
    disagreements = []
    compared = []
    for name, from_lst, from_res in pairs:
        if from_lst is None or from_res is None:
            continue
        compared.append(name)
        if abs(from_lst - from_res) > 5e-4:
            disagreements.append(f"{name}: listing {from_lst} vs .res {from_res}")
    return {"compared": True, "fields": compared, "disagreements": disagreements}
