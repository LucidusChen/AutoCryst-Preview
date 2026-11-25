"""Small reversible reconstruction steps, executed by the existing OnionWorkflow.

This module never opens an answer and never starts a crystallographic program. Providers own
observations, inference and refinement; the stage executor enforces their contracts and retains
actual model/instruction files. Synthetic providers are explicitly software evidence. Real tool
dispatch is deliberately unavailable in this milestone.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Protocol

from crystalx_core._serde import from_jsonable, to_jsonable
from crystalx_core.domain.candidate import Candidate, CandidateProvenance, Mutation, MutationKind
from crystalx_core.domain.chemistry import ATOMIC_NUMBERS
from crystalx_core.domain.hypothesis import LockDirective, LockMode
from crystalx_core.domain.hypothesis import AnchorSet, HeavyAtomHypothesis, HeavyAtomSite
from crystalx_core.domain.ids import make_candidate_id
from crystalx_core.domain.inference import ElementInferenceRequest, ElementInferenceResult, InferenceBackend
from crystalx_core.domain.metrics import GeometryReport, RefinementMetrics, ValidationReport
from crystalx_core.domain.onion import StageKind
from crystalx_core.domain.result import StopReasonKind
from crystalx_core.io.shelx_res import read_res
from crystalx_core.refinement.shelxl import RefinementSettings, prepare_refinement_ins
from crystalx_core.tools.runner import CancelToken
from crystalx_core.workflow.onion import StageControlStop, StageRequest, StageResult
from crystalx_core.workspace.hashing import sha256_file


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(to_jsonable(value), sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


class ReconstructionBlocked(ValueError):
    """A missing provider or incompatible observation cannot be replaced with a guess."""


@dataclass(frozen=True)
class Site:
    site_id: str
    element: str
    fractional: tuple[float, float, float]
    occupancy: float
    u_iso: float


@dataclass(frozen=True)
class ModelSnapshot:
    text: str
    sites: tuple[Site, ...]
    sfac: tuple[str, ...]
    basis_key: str
    settings_sha256: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    @property
    def structural_key(self) -> str:
        # Atom labels are stable identities; text REM/format changes do not constitute progress.
        return digest([(s.site_id, s.element, tuple(round(x % 1, 6) for x in s.fractional),
                        s.occupancy, s.u_iso) for s in self.sites])


_SETTING_WORDS = {"CELL", "LATT", "SYMM", "HKLF", "TWIN", "BASF", "MERG", "OMIT", "SHEL", "WGHT",
                  "UNIT", "SFAC", "DFIX", "DANG", "SADI", "SAME", "EADP", "EXYZ", "PART", "RESI",
                  "AFIX", "HFIX", "SIMU", "DELU", "RIGU", "ISOR", "SUMP", "CHIV", "FLAT"}


def read_snapshot(path: Path, *, reflection_sha256: str, coordinate_bound: float = 1.0,
                  allow_nonpositive_u: bool = False,allow_anisotropic_metals:bool=False) -> ModelSnapshot:
    if type(allow_nonpositive_u) is not bool or type(allow_anisotropic_metals) is not bool:
        raise ReconstructionBlocked("nonpositive_displacement_policy_must_be_explicit_boolean")
    if (isinstance(coordinate_bound, bool) or not math.isfinite(coordinate_bound)
            or not 0 < coordinate_bound <= 5):
        raise ReconstructionBlocked("invalid_explicit_fractional_coordinate_bound")
    parsed = read_res(path)
    text = path.read_text(encoding="utf-8")
    if parsed.sha256 != hashlib.sha256(text.encode()).hexdigest():
        raise ReconstructionBlocked("model_changed_during_parse")
    if not parsed.atoms or not parsed.header.cell or not parsed.header.hklf_format:
        raise ReconstructionBlocked("model_atoms_cell_and_hklf_required")
    if "CRYSTALX ACTIVE CONSTRAINTS BEGIN" in text.upper():
        raise ReconstructionBlocked("refinement_provider_must_return_an_explicit_lock_free_model")
    sites = []
    for atom in parsed.atoms:
        element = parsed.element_of(atom)
        value=atom.u_iso
        if allow_anisotropic_metals and atom.is_anisotropic:
            from crystalx_core.refinement.cycle import METALS,displacement_value
            if element not in METALS:raise ReconstructionBlocked("anisotropic_nonmetal_outside_explicit_metal_policy")
            value=displacement_value(parsed,atom)
        if (not element or atom.part not in (None, 0) or atom.residue is not None
                or atom.afix not in (None, 0) or len(atom.u_values) not in ((1,6) if allow_anisotropic_metals else (1,))
                or not 0 < atom.occupancy <= 1 or abs(atom.occupancy_code) >= 20
                or value is None or not allow_nonpositive_u and value <= 0
                or any(not math.isfinite(v) for v in (atom.x, atom.y, atom.z, value,*atom.u_values))
                or any(not -coordinate_bound <= v < coordinate_bound for v in (atom.x, atom.y, atom.z))
                or coordinate_bound == 5 and any(v <= -5 for v in (atom.x, atom.y, atom.z))):
            raise ReconstructionBlocked("unsupported_disorder_free_variable_or_encoded_coordinate_model")
        sites.append(Site(atom.label, element, (atom.x, atom.y, atom.z), atom.occupancy, value))
    if len({s.site_id.casefold() for s in sites}) != len(sites):
        raise ReconstructionBlocked("duplicate_site_identity")
    setting = tuple(line.strip() for line in text.splitlines()
                    if line.split() and line.split()[0].upper() in _SETTING_WORDS)
    settings_hash = digest(setting)
    return ModelSnapshot(text, tuple(sites), parsed.header.sfac,
                         digest((reflection_sha256, settings_hash)), settings_hash)


@dataclass(frozen=True)
class DensityPeak:
    peak_id: str
    fractional: tuple[float, float, float]
    height: float


@dataclass(frozen=True)
class DensityObservation:
    model_sha256: str
    basis_key: str
    source_sha256: str
    kind: str  # Q_PEAK_LIST, VERIFIED_FULL_MAP, or SYNTHETIC_PEAK_LIST
    peaks: tuple[DensityPeak, ...]
    coverage: str = "partial"

    def validate_for(self, model: ModelSnapshot, *, synthetic: bool) -> None:
        if self.model_sha256 != model.sha256 or self.basis_key != model.basis_key:
            raise ReconstructionBlocked("density_model_or_reflection_identity_mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ReconstructionBlocked("density_source_identity_required")
        allowed = {"Q_PEAK_LIST", "VERIFIED_FULL_MAP"} | ({"SYNTHETIC_PEAK_LIST"} if synthetic else set())
        if self.kind not in allowed or (self.coverage == "full" and self.kind != "VERIFIED_FULL_MAP"):
            raise ReconstructionBlocked("density_capability_mislabelled")
        if len({p.peak_id for p in self.peaks}) != len(self.peaks):
            raise ReconstructionBlocked("duplicate_density_peak_identity")
        for peak in self.peaks:
            if (not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", peak.peak_id)
                    or len(peak.fractional) != 3
                    or any(not math.isfinite(v) or not -1 <= v < 1 for v in peak.fractional)
                    or not math.isfinite(peak.height) or peak.height <= 0):
                raise ReconstructionBlocked("invalid_density_peak")


_CHECKS = frozenset({"geometry", "coordination", "occupancy_adp", "chirality_symmetry",
                     "restraints", "heavy_site_drift", "residual_distribution"})


@dataclass(frozen=True)
class Assessment:
    basis_key: str
    r1: float
    wr2: float
    goof: float
    residual_regions: tuple[float, ...]
    n_reflections: int
    n_parameters: int
    checks: tuple[tuple[str, str], ...]  # PASS, FAIL, NOT_EVALUATED
    source_sha256: str
    heavy_drift_angstrom: float = 0.0
    requires_review: bool = False
    scientific_validation: bool = False
    least_squares_converged: bool | None = None
    model_sha256: str = ""

    def blockers(self) -> tuple[str, ...]:
        status = dict(self.checks)
        failures = [name + ":" + status.get(name, "NOT_EVALUATED")
                    for name in sorted(_CHECKS) if status.get(name) != "PASS"]
        values = (self.r1, self.wr2, self.goof, self.heavy_drift_angstrom, *self.residual_regions)
        if (not self.residual_regions or any(not math.isfinite(v) or v < 0 for v in values)
                or not 0 <= self.r1 <= 1 or not 0 <= self.wr2 <= 2 or self.goof <= 0
                or type(self.n_reflections) is not int or type(self.n_parameters) is not int
                or self.n_reflections <= 0 or self.n_parameters <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256)
                or not re.fullmatch(r"[0-9a-f]{64}", self.model_sha256)):
            failures.append("invalid_assessment")
        if self.least_squares_converged is not True:
            failures.append("least_squares_convergence:NOT_EVALUATED" if self.least_squares_converged is None
                            else "least_squares_convergence:FAIL")
        return tuple(failures)


@dataclass(frozen=True)
class InferenceAudit:
    weights_kind: str  # synthetic or official_trained
    weights_sha256: str
    source_files: tuple[tuple[str, str], ...]
    preprocessing_policy_sha256: str
    reference_comparison_sha256: str = ""
    provenance_reference: str = ""

    def blockers(self, *, synthetic: bool) -> tuple[str, ...]:
        failures = []
        if not re.fullmatch(r"[0-9a-f]{64}", self.weights_sha256):
            failures.append("checkpoint_digest_missing")
        if not self.source_files or any(not re.fullmatch(r"[0-9a-f]{64}", sha) for _, sha in self.source_files):
            failures.append("inference_source_pins_missing")
        if not re.fullmatch(r"[0-9a-f]{64}", self.preprocessing_policy_sha256):
            failures.append("preprocessing_policy_not_pinned")
        if not synthetic:
            if self.weights_kind != "official_trained" or not self.provenance_reference:
                failures.append("official_trained_checkpoint_provenance_missing")
            if not re.fullmatch(r"[0-9a-f]{64}", self.reference_comparison_sha256):
                failures.append("reference_inference_comparison_missing")
        elif self.weights_kind != "synthetic":
            failures.append("synthetic_demo_requires_explicit_synthetic_weights")
        return tuple(failures)


@dataclass(frozen=True)
class PreparedInference:
    request: ElementInferenceRequest
    # Physical peak coordinates come only from DensityObservation, never from inference output.
    source_model_sha256: str
    preprocessing_policy_sha256: str


class DensityProvider(Protocol):
    def observe(self, model: ModelSnapshot) -> DensityObservation: ...


class InferencePreparer(Protocol):
    def prepare(self, model: ModelSnapshot, observation: DensityObservation) -> PreparedInference: ...


@dataclass(frozen=True)
class RefinementTrial:
    input_path: Path
    hkl_path: Path
    work_directory: Path
    proposed_model: ModelSnapshot
    active_locks: Mapping[str, LockDirective]
    cycles: int
    timeout_s: float
    cancel: CancelToken
    synthetic: bool


@dataclass(frozen=True)
class RefinementOutcome:
    lock_free_model_text: str
    assessment: Assessment
    process_kind: str
    succeeded: bool = True
    review_required: bool = False


class RefinementProvider(Protocol):
    def refine(self, request: RefinementTrial) -> RefinementOutcome: ...


@dataclass(frozen=True)
class ReconstructionPolicy:
    max_rounds: int = 6
    max_proposals: int = 6
    max_added_atoms: int = 3
    max_tool_calls: int = 7
    max_inference_calls: int = 6
    max_failures: int = 2
    max_stagnant_rounds: int = 2
    max_wall_seconds: float = 60
    trial_wall_seconds: float = 10
    shelxl_cycles_per_trial: int = 2
    maximum_parameters_per_added_atom: int = 5
    maximum_heavy_drift_angstrom: float = 0.15
    ai_required: bool = True
    require_full_map: bool = False

    def __post_init__(self):
        for field_name in ("max_rounds", "max_proposals", "max_added_atoms", "max_tool_calls",
                           "max_inference_calls",
                           "max_failures", "max_stagnant_rounds", "shelxl_cycles_per_trial",
                           "maximum_parameters_per_added_atom"):
            if type(getattr(self, field_name)) is not int or getattr(self, field_name) < 1:
                raise ValueError("reconstruction budgets must be positive integers")
        for value in (self.max_wall_seconds, self.trial_wall_seconds, self.maximum_heavy_drift_angstrom):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("reconstruction limits must be finite and positive")


@dataclass
class Usage:
    rounds: int = 0
    proposals: int = 0
    added_atoms: int = 0
    tool_calls: int = 0
    inference_calls: int = 0
    failures: int = 0
    stagnant: int = 0
    elapsed_seconds: float = 0


@dataclass(frozen=True)
class Proposal:
    kind: str
    site_id: str
    element: str
    peak_id: str = ""
    fractional: tuple[float, float, float] | None = None
    alternatives: tuple[tuple[int, float], ...] = ()


def mutate_model(model: ModelSnapshot, proposal: Proposal) -> str:
    """Exactly one SFAC reassignment or one observed-peak addition; no masks/settings edits."""
    if proposal.element not in model.sfac:
        raise ReconstructionBlocked("proposal_element_outside_declared_sfac")
    index = model.sfac.index(proposal.element) + 1
    lines = model.text.splitlines()
    if proposal.kind == "element_flip":
        found = 0
        for n, line in enumerate(lines):
            tokens = line.split()
            if tokens and tokens[0] == proposal.site_id:
                if len(tokens) != 7:
                    raise ReconstructionBlocked("unsupported_non_isotropic_atom_mutation")
                tokens[1] = str(index)
                lines[n] = " ".join(tokens)
                found += 1
        if found != 1:
            raise ReconstructionBlocked("mutation_site_not_unique")
    elif proposal.kind == "atom_added" and proposal.fractional is not None and proposal.peak_id:
        if any(site.site_id == proposal.site_id for site in model.sites):
            raise ReconstructionBlocked("new_site_identity_already_exists")
        row = (f"{proposal.site_id} {index} " + " ".join(f"{v:.8f}" for v in proposal.fractional)
               + " 11.00000 0.05000")
        location = next((n for n, line in enumerate(lines) if line.split() and line.split()[0].upper() == "HKLF"), None)
        if location is None:
            raise ReconstructionBlocked("hklf_required_for_mutation")
        lines.insert(location, row)
    else:
        raise ReconstructionBlocked("unsupported_proposal")
    return "\n".join(lines) + "\n"


def decide(parent: Assessment, trial: Assessment, *, added: int,
           policy: ReconstructionPolicy) -> tuple[str, tuple[str, ...]]:
    blockers = trial.blockers()
    if blockers or trial.requires_review:
        missing = any("NOT_EVALUATED" in item for item in blockers)
        return ("REVIEW" if missing or trial.requires_review else "REJECT", blockers or ("human_review",))
    if parent.basis_key != trial.basis_key or parent.n_reflections != trial.n_reflections:
        return "REVIEW", ("incompatible_reflections_or_settings",)
    if trial.heavy_drift_angstrom > policy.maximum_heavy_drift_angstrom:
        return "REJECT", ("heavy_scaffold_drift",)
    if trial.n_parameters > parent.n_parameters + added * policy.maximum_parameters_per_added_atom:
        return "REJECT", ("uncontrolled_parameter_growth",)
    if (trial.r1 > parent.r1 + 1e-6 or trial.wr2 > parent.wr2 + 1e-6
            or abs(trial.goof - 1) > abs(parent.goof - 1) + 1e-6):
        return "REJECT", ("refinement_metrics_regressed",)
    if len(parent.residual_regions) != len(trial.residual_regions):
        return "REVIEW", ("incompatible_residual_regions",)
    if any(t > p + 1e-6 for p, t in zip(parent.residual_regions, trial.residual_regions)):
        return "REJECT", ("residual_distribution_regressed",)
    if trial.r1 >= parent.r1 - 1e-6 or sum(trial.residual_regions) >= sum(parent.residual_regions) - 1e-6:
        return "REJECT", ("no_joint_refinement_density_progress",)
    return "ACCEPT_PROVISIONAL", ("compatible_multi_check_improvement",)


@dataclass
class ReconstructionState:
    model: ModelSnapshot
    assessment: Assessment | None = None
    active_locks: dict[str, LockDirective] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    seen_structures: list[str] = field(default_factory=list)
    attempted_proposals: list[str] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    checkpoints: list[tuple[str, str]] = field(default_factory=list)
    stop_code: str = ""
    parent_checkpoint_sha256: str = ""


class ReconstructionExecutor:
    """One shared StageExecutor for initial, scaffold and repeated model-edit stages."""

    def __init__(self, state: ReconstructionState, *, hkl_path: Path,
                 density: DensityProvider | None, inference: InferenceBackend | None,
                 preparer: InferencePreparer | None, inference_audit: InferenceAudit | None,
                 refinement: RefinementProvider | None, policy: ReconstructionPolicy,
                 synthetic: bool, cancel: CancelToken, clock: Callable[[], float] = time.monotonic,
                 initial_locks: Mapping[str, LockDirective] | None = None):
        self.state, self.hkl_path = state, hkl_path
        self.hkl_sha256 = sha256_file(hkl_path)
        self.density, self.inference, self.preparer = density, inference, preparer
        self.audit, self.refinement, self.policy = inference_audit, refinement, policy
        self.synthetic, self.cancel, self.clock = synthetic, cancel, clock
        self.initial_locks = dict(initial_locks or {})
        self.started = clock()
        self.prior_elapsed = state.usage.elapsed_seconds
        self._pending: tuple[ModelSnapshot, Assessment, dict[str, LockDirective], str] | None = None
        self._pending_decision: dict | None = None
        self._before = None
        self._transaction_stage: str | None = None

    def _check(self):
        if self.cancel.cancelled:
            raise ReconstructionBlocked("cancelled")
        elapsed = self.clock() - self.started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ReconstructionBlocked("resource_clock_invalid")
        self.state.usage.elapsed_seconds = self.prior_elapsed + elapsed
        if self.state.usage.elapsed_seconds >= self.policy.max_wall_seconds:
            raise ReconstructionBlocked("total_wall_budget_exhausted")
        if sha256_file(self.hkl_path) != self.hkl_sha256:
            raise ReconstructionBlocked("reflection_input_changed")

    def _checkpoint(self, request: StageRequest, label: str, *, check_resources: bool = True):
        if check_resources:
            self._check()
        number = len(self.state.checkpoints)
        relative = f"reconstruction/checkpoint-{number:04d}-{label}.json"
        body = {"schema_version": 1, "policy_sha256": digest(self.policy),
                "reflection_sha256": self.hkl_sha256, "synthetic": self.synthetic,
                "state": to_jsonable(replace(self.state, checkpoints=[])),
                "inference_audit": to_jsonable(self.audit)}
        path = request.workspace.write_json(relative, body)
        pin = sha256_file(path)
        request.workspace.write_json(relative + ".pin.json", {"sha256": pin})
        self.state.checkpoints.append((relative, pin))

    def _stop(self, request: StageRequest, code: str) -> StageResult:
        self.state.stop_code = code
        # A cancellation/deadline still gets a final immutable record; no new external work.
        request.workspace.write_json(f"reconstruction/stop-{request.stage.stage_id}.json",
                                     {"code": code, "usage": self.state.usage,
                                      "accepted_model_sha256": self.state.model.sha256 if self.state.assessment is not None else None,
                                      "retained_model_sha256": self.state.model.sha256})
        return StageResult(requires_human_review=True, summary=code, notes=(code,), attempts_used=0)

    def execute(self, request: StageRequest) -> StageResult:
        self._pending = None
        self._pending_decision = None
        self._before = (self.state.model, self.state.assessment, dict(self.state.active_locks),
                        list(self.state.seen_structures))
        self._transaction_stage = str(request.stage.stage_id)
        try:
            self._check()
            request.checkpoint()
            if not self.synthetic:
                raise ReconstructionBlocked("real_dispatch_not_authorized_or_implemented")
            if self.policy.ai_required:
                failures = (self.audit.blockers(synthetic=self.synthetic) if self.audit else ("inference_audit_missing",))
                if self.inference is None or self.preparer is None or failures:
                    raise ReconstructionBlocked("ai_required_provider_or_checkpoint_missing:" + ",".join(failures))
            if self.density is None or self.refinement is None:
                raise ReconstructionBlocked("density_or_refinement_provider_missing")
            if request.stage.kind is StageKind.INITIAL_SOLUTION:
                return self._initial(request)
            if request.stage.kind is StageKind.ANCHOR:
                # Provisional restraints are hypotheses, not a validated metal scaffold.
                directives = tuple(self.initial_locks.values())
                for directive in directives:
                    if directive.mode is LockMode.HARD_LOCK:
                        raise ReconstructionBlocked("hard_lock_requires_separate_owner_review")
                heavy = tuple(HeavyAtomSite(s.site_id, s.element, *s.fractional,
                                           occupancy=s.occupancy, u_iso=s.u_iso, element_is_assumed=True)
                              for s in self.state.model.sites if ATOMIC_NUMBERS[s.element] >= 20)
                hypothesis = HeavyAtomHypothesis("provisional-scaffold", heavy,
                                                  notes="synthetic refinement support only; not experimentally validated")
                request.workspace.write_json("reconstruction/scaffold-hypothesis.json", hypothesis)
                return StageResult(anchor=AnchorSet("provisional", hypothesis, validated=False),
                                   lock_directives=directives, attempts_used=0,
                                   summary="provisional reversible scaffold hypotheses")
            if request.stage.kind is StageKind.FIRST_SHELL:
                return StageResult(attempts_used=0, summary="seed retained; trial edits use repeated S4 stages")
            if request.stage.kind is StageKind.FINALIZE:
                return self._stop(request, "scientific_review_not_performed")
            return self._iterate(request)
        except ReconstructionBlocked as exc:
            self.state.stop_code = str(exc)
            if "budget_exhausted" in str(exc):
                raise StageControlStop(StopReasonKind.BUDGET_EXCEEDED, str(exc)) from exc
            if str(exc) == "cancelled":
                raise StageControlStop(StopReasonKind.USER_ABORTED, "cancelled") from exc
            return self._stop(request, str(exc))

    def commit_stage(self, request: StageRequest, chosen: Candidate | None,
                     active_lock_ids: tuple[str, ...]) -> None:
        """Only the controller's acceptance acknowledgment can promote a pending model."""
        if self._pending is not None:
            model, assessment, locks, candidate_id = self._pending
            if chosen is None or str(chosen.candidate_id) != candidate_id:
                self.abort_stage(request, "controller_did_not_accept_pending_candidate")
                return
            if set(locks) != set(active_lock_ids):
                raise ReconstructionBlocked("controller_lock_set_differs_from_executed_instructions")
            self.state.model, self.state.assessment = model, assessment
            self.state.active_locks = dict(locks)
            self.state.seen_structures.append(model.structural_key)
            self.state.usage.stagnant = 0
            if self._pending_decision is not None:
                self._pending_decision["decision"] = "ACCEPT_PROVISIONAL"
            self._checkpoint(request, "accepted", check_resources=False)
            request.workspace.write_json(f"reconstruction/commit-{request.stage.stage_id}.json",
                                         {"candidate_id": candidate_id, "model_sha256": model.sha256,
                                          "active_lock_ids": active_lock_ids})
        self._pending = None
        self._pending_decision = None
        self._before = None
        self._transaction_stage = None

    def abort_stage(self, request: StageRequest, reason: str) -> None:
        """The accepted model and active lock set stay unchanged; budgets are never refunded."""
        if self._transaction_stage != str(request.stage.stage_id):
            return  # A preflight stop before execute must never undo the previous committed stage.
        if self._before is not None:
            model, assessment, locks, seen = self._before
            self.state.model, self.state.assessment = model, assessment
            self.state.active_locks, self.state.seen_structures = locks, seen
        if self._pending_decision is not None and self._pending_decision["decision"] in {"ACCEPT_PENDING_CONTROLLER", "ACCEPT_PROVISIONAL"}:
            self._pending_decision["decision"] = "REJECTED_BY_CONTROLLER"
            self._pending_decision["controller_reason"] = reason
            self.state.usage.failures += 1
            self.state.usage.stagnant += 1
        elif reason.startswith("stage_transaction_failed"):
            self.state.usage.failures += 1
        self._pending = None
        if self.state.assessment is not None:
            self._checkpoint(request, "rolled-back", check_resources=False)
        self._pending_decision = None
        self._before = None
        self._transaction_stage = None

    def _candidate(self, request: StageRequest, model: ModelSnapshot, assessment: Assessment,
                   path: Path, mutation: Mutation | None = None) -> Candidate:
        candidate_id = make_candidate_id(request.workspace.next_candidate_index())
        return Candidate(candidate_id, CandidateProvenance(
            created_at=datetime.now(timezone.utc).isoformat(), stage_id=request.stage.stage_id,
            parent_id=request.input_candidate.candidate_id if request.input_candidate else None,
            reflection_file_sha256=self.hkl_sha256,
            input_hashes={"model": model.sha256, **({"parent_checkpoint": self.state.parent_checkpoint_sha256}
                          if self.state.parent_checkpoint_sha256 else {})},
            model_identities={"weights": self.audit.weights_sha256} if self.audit else {},
            notes="SYNTHETIC SOFTWARE EVIDENCE; provisional, not scientifically validated"),
            mutations=(mutation,) if mutation else (), label="provisional reconstruction",
            res_path=str(path.relative_to(request.workspace.layout.root)),
            n_non_hydrogen_atoms=len(model.sites),
            metrics=RefinementMetrics(r1=assessment.r1, wr2=assessment.wr2, goof=assessment.goof,
                                      n_reflections_observed=assessment.n_reflections,
                                      n_parameters=assessment.n_parameters, converged=assessment.least_squares_converged,
                                      reflection_basis_key=assessment.basis_key),
            geometry=GeometryReport(evaluated=not assessment.blockers()),
            validation=ValidationReport("synthetic_assessment", completed=not assessment.blockers()))

    def _refine(self, request: StageRequest, model: ModelSnapshot, locks: dict[str, LockDirective],
                directory: Path) -> tuple[ModelSnapshot, Assessment, Path]:
        from crystalx_core.refinement.locks import apply_locks
        self._check()
        request.checkpoint()
        if self.state.usage.tool_calls >= self.policy.max_tool_calls:
            raise ReconstructionBlocked("total_tool_call_budget_exhausted")
        request.before_call()
        self.state.usage.tool_calls += 1  # Count reserved calls even after interruption/failure.
        directory.mkdir(parents=True, exist_ok=False)
        baseline, changes, _, _ = prepare_refinement_ins(model.text, RefinementSettings(
            cycles=self.policy.shelxl_cycles_per_trial, timeout_s=self.policy.trial_wall_seconds))
        application = apply_locks(baseline, locks)
        ins = directory / "trial.ins"
        ins.write_text(application.text)
        hkl = directory / "trial.hkl"
        shutil.copyfile(self.hkl_path, hkl)
        hkl.chmod(0o444)
        request.workspace.write_json(str((directory / "instructions.json").relative_to(request.workspace.layout.root)),
                                     {"locks": application, "changes": changes,
                                      "hkl_sha256": self.hkl_sha256, "cycles": self.policy.shelxl_cycles_per_trial})
        self._checkpoint(request, "in-flight")  # The reserved call survives interruption/restart.
        before = self.clock()
        outcome = self.refinement.refine(RefinementTrial(
            ins, hkl, directory, model, locks, self.policy.shelxl_cycles_per_trial,
            min(self.policy.trial_wall_seconds, self.policy.max_wall_seconds - self.state.usage.elapsed_seconds),
            self.cancel, self.synthetic))
        self._check()
        if self.clock() - before > self.policy.trial_wall_seconds:
            raise ReconstructionBlocked("trial_wall_budget_exhausted")
        if sha256_file(hkl) != self.hkl_sha256 or sha256_file(ins) != application.output_sha256:
            raise ReconstructionBlocked("refinement_provider_changed_declared_inputs")
        if not outcome.succeeded or outcome.process_kind != "synthetic":
            raise ReconstructionBlocked("provider_process_failed_or_mislabelled")
        # A file-backed provider owns raw trial.res. Never overwrite its scientific output
        # with the separate, audited model from which managed constraints were removed.
        output = directory / "lock-free-model.res"
        if output.is_symlink():
            raise ReconstructionBlocked("refinement_provider_lock_free_artifact_must_be_regular")
        if output.exists():
            if output.read_text() != outcome.lock_free_model_text:
                raise ReconstructionBlocked("refinement_provider_lock_free_artifact_mismatch")
        else:
            output.write_text(outcome.lock_free_model_text)
        returned = read_snapshot(output, reflection_sha256=self.hkl_sha256)
        if (returned.basis_key != model.basis_key or returned.sfac != model.sfac
                or [(s.site_id, s.element) for s in returned.sites] != [(s.site_id, s.element) for s in model.sites]
                or outcome.assessment.basis_key != model.basis_key
                or outcome.assessment.model_sha256 != returned.sha256):
            raise ReconstructionBlocked("provider_changed_setting_site_identity_or_reflection_basis")
        assessment = replace(outcome.assessment, requires_review=(outcome.review_required or outcome.assessment.requires_review))
        request.workspace.write_json(str((directory / "assessment.json").relative_to(request.workspace.layout.root)), assessment)
        return returned, assessment, output

    def _initial(self, request: StageRequest) -> StageResult:
        calls_before = self.state.usage.tool_calls
        if self.state.assessment is None:
            model, assessment, path = self._refine(request, self.state.model, self.state.active_locks,
                                                   request.workspace.layout.root / "reconstruction/baseline")
            if assessment.blockers() or assessment.requires_review:
                raise ReconstructionBlocked("initial_assessment_incomplete_or_failed")
        else:
            model, assessment = self.state.model, self.state.assessment
            path = request.workspace.write_text("reconstruction/resumed-seed.res", self.state.model.text)
        candidate = self._candidate(request, model, assessment, path)
        self._pending = (model, assessment, dict(self.state.active_locks), str(candidate.candidate_id))
        return StageResult(candidates=(candidate,), new_atoms_added=0,
                           attempts_used=0, tool_calls_used=self.state.usage.tool_calls - calls_before,
                           summary="automatic initial model retained as seed, not final answer")

    def _select(self, model: ModelSnapshot, observation: DensityObservation,
                result: ElementInferenceResult | None) -> Proposal | None:
        if result is None:
            return None  # Explicit SHELX-only mode performs no AI element proposal.
        if result.model_identity.weights_sha256 != self.audit.weights_sha256 or tuple(result.model_identity.source_files) != self.audit.source_files:
            raise ReconstructionBlocked("inference_checkpoint_or_source_identity_mismatch")
        predictions = {p.site_id: p for p in result.predictions}
        expected = {s.site_id for s in model.sites} | {p.peak_id for p in observation.peaks}
        if set(predictions) != expected or len(predictions) != len(result.predictions):
            raise ReconstructionBlocked("inference_site_identity_mismatch")
        atomic_to_symbol = {v: k for k, v in ATOMIC_NUMBERS.items()}
        options = []
        for key, prediction in predictions.items():
            top = prediction.topk
            if (len(top) < 2 or len({p.atomic_number for p in top}) != len(top)
                    or any(type(p.atomic_number) is not int or p.atomic_number not in result.candidate_atomic_numbers for p in top)
                    or any(not math.isfinite(p.probability) or not 0 <= p.probability <= 1 for p in top)
                    or sum(p.probability for p in top) > 1 + 1e-6
                    or any(top[i].probability < top[i + 1].probability for i in range(len(top) - 1))):
                raise ReconstructionBlocked("invalid_inference_alternatives")
            existing = next((s for s in model.sites if s.site_id == key), None)
            peak = next((p for p in observation.peaks if p.peak_id == key), None)
            for rank, alternative in enumerate(top):
                if existing and rank:
                    continue  # Keep alternatives as uncertainty; do not force a flip from a matching top label.
                element = atomic_to_symbol.get(alternative.atomic_number)
                if element not in model.sfac or existing and element == existing.element:
                    continue
                proposal = (Proposal("element_flip", key, element,
                                     alternatives=tuple((p.atomic_number, p.probability) for p in top)) if existing
                            else Proposal("atom_added", f"X{self.state.usage.proposals + 1:03d}", element,
                                          peak.peak_id, peak.fractional,
                                          tuple((p.atomic_number, p.probability) for p in top)))
                signature = digest((model.structural_key, proposal.kind, key, element))
                if signature not in self.state.attempted_proposals:
                    options.append((rank, -alternative.probability, key, proposal, signature))
        if not options:
            return None
        _, _, _, proposal, signature = min(options, key=lambda item: item[:3])
        self.state.attempted_proposals.append(signature)
        return proposal

    def _iterate(self, request: StageRequest) -> StageResult:
        usage = self.state.usage
        if usage.rounds >= self.policy.max_rounds or usage.proposals >= self.policy.max_proposals:
            raise ReconstructionBlocked("iteration_or_proposal_budget_exhausted")
        if usage.failures >= self.policy.max_failures:
            raise ReconstructionBlocked("failure_budget_exhausted")
        usage.rounds += 1
        # Ledger is authoritative: rejected trial locks do not survive the next call.
        if request.ledger is not None:
            self.state.active_locks = {key: request.ledger.get(key) for key in request.active_lock_ids}
        observation = self.density.observe(self.state.model)
        self._check()
        observation.validate_for(self.state.model, synthetic=self.synthetic)
        if self.policy.require_full_map and observation.kind != "VERIFIED_FULL_MAP":
            raise ReconstructionBlocked("full_density_map_provider_missing")
        inference_result = None
        if self.policy.ai_required:
            prepared = self.preparer.prepare(self.state.model, observation)
            if prepared.source_model_sha256 != self.state.model.sha256 or prepared.preprocessing_policy_sha256 != self.audit.preprocessing_policy_sha256:
                raise ReconstructionBlocked("inference_preprocessing_identity_mismatch")
            expected_ids = tuple(s.site_id for s in self.state.model.sites) + tuple(p.peak_id for p in observation.peaks)
            if tuple(prepared.request.site_ids[:len(expected_ids)]) != expected_ids:
                raise ReconstructionBlocked("prepared_inference_site_identity_mismatch")
            self._check()
            request.checkpoint()
            if usage.inference_calls >= self.policy.max_inference_calls:
                raise ReconstructionBlocked("inference_call_budget_exhausted")
            usage.inference_calls += 1
            inference_result = self.inference.predict_elements(prepared.request, allow_cpu_fallback=False)
            self._check()
            request.checkpoint()
            if inference_result.fell_back:
                raise ReconstructionBlocked("inference_forbidden_fallback")
            if (inference_result.input_sha256 != digest(prepared.request)
                    or tuple(inference_result.candidate_atomic_numbers) != tuple(prepared.request.candidate_atomic_numbers or ())):
                raise ReconstructionBlocked("inference_input_identity_mismatch")
        proposal = self._select(self.state.model, observation, inference_result)
        if proposal is None:
            usage.stagnant += 1
            self._checkpoint(request, "no-progress")
            if usage.stagnant >= self.policy.max_stagnant_rounds or not self.policy.ai_required:
                return self._stop(request, "no_progress_not_convergence")
            return StageResult(should_rollback=True, rollback_reason="no_supported_proposal", attempts_used=0)
        added = int(proposal.kind == "atom_added")
        if usage.added_atoms + added > self.policy.max_added_atoms:
            raise ReconstructionBlocked("total_added_atom_budget_exhausted")
        request.meter.reserve(attempts=1, new_atoms=added, candidates=1)
        usage.proposals += 1
        usage.added_atoms += added  # Attempts count even when rejected; rollback does not refund.
        trial_dir = request.workspace.layout.root / f"reconstruction/trial-{usage.proposals:04d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        proposed_path = trial_dir / "proposal.res"
        proposed_path.write_text(mutate_model(self.state.model, proposal))
        proposed = read_snapshot(proposed_path, reflection_sha256=self.hkl_sha256)
        if proposed.structural_key in self.state.seen_structures:
            raise ReconstructionBlocked("oscillation_detected")
        proposed_locks = dict(self.state.active_locks)
        new_directives = ()
        if added and self.state.model.sites:
            first = self.state.model.sites[0].site_id
            directive = LockDirective(LockMode.SOFT_RESTRAINT, (first, proposal.site_id),
                                      "provisional synthetic coordination hypothesis; reversible trial",
                                      instruction_lines=(f"DFIX 2.0 0.2 {first} {proposal.site_id}",))
            key = request.ledger.next_id()
            proposed_locks[key] = directive
            new_directives = (directive,)
        self._checkpoint(request, "before-call")
        returned, assessment, path = self._refine(request, proposed, proposed_locks, trial_dir / "refine")
        request.checkpoint()
        decision, reasons = decide(self.state.assessment, assessment, added=added, policy=self.policy)
        mutation = Mutation(MutationKind.ATOM_ADDED if added else MutationKind.ELEMENT_FLIP,
                            "one density-supported site or element hypothesis", atoms=(proposal.site_id,),
                            to_value=proposal.element, driver="synthetic_density_and_inference")
        candidate = self._candidate(request, returned, assessment, path, mutation)
        entry = {"run_id": str(request.workspace.run_id), "round": usage.rounds, "proposal": to_jsonable(proposal),
                 "proposal_verdict": decision,
                 "decision": "ACCEPT_PENDING_CONTROLLER" if decision == "ACCEPT_PROVISIONAL" else decision,
                 "reasons": reasons, "parent_sha256": self.state.model.sha256,
                 "trial_sha256": returned.sha256, "parent_locks": list(self.state.active_locks),
                 "trial_locks": list(proposed_locks), "density": to_jsonable(observation),
                 "inference": to_jsonable(inference_result), "candidate_id": str(candidate.candidate_id)}
        self.state.decisions.append(entry)
        request.workspace.write_json(f"reconstruction/trial-{usage.proposals:04d}/decision.json", entry)
        if decision == "ACCEPT_PROVISIONAL":
            entry["decision"] = "ACCEPT_PENDING_CONTROLLER"
            self._pending_decision = entry
            self._pending = (returned, assessment, proposed_locks, str(candidate.candidate_id))
        else:
            usage.stagnant += 1
            usage.failures += 1
        return StageResult(candidates=(candidate,), lock_directives=new_directives,
                           should_rollback=decision != "ACCEPT_PROVISIONAL", rollback_reason=", ".join(reasons),
                           requires_human_review=decision == "REVIEW", attempts_used=1,
                           tool_calls_used=1, failures_used=int(decision != "ACCEPT_PROVISIONAL"),
                           new_atoms_added=added, summary=decision)


def load_checkpoint(path: Path, *, expected_sha256: str, policy: ReconstructionPolicy,
                    hkl_path: Path, expected_synthetic: bool,
                    expected_inference_audit: InferenceAudit | None = None) -> ReconstructionState:
    """Restart in a fresh workspace; never overwrite the parent history or read an answer."""
    if path.is_symlink() or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ReconstructionBlocked("checkpoint_pin_required")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ReconstructionBlocked("checkpoint_size_exceeded")
    with path.open("rb") as source:
        data = source.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ReconstructionBlocked("checkpoint_digest_mismatch")
    def invalid_constant(_):
        raise ReconstructionBlocked("nonfinite_checkpoint")
    payload = json.loads(data, parse_constant=invalid_constant)
    if (set(payload) != {"schema_version", "policy_sha256", "reflection_sha256", "synthetic", "state", "inference_audit"}
            or payload["schema_version"] != 1 or payload["policy_sha256"] != digest(policy)
            or payload["reflection_sha256"] != sha256_file(hkl_path)
            or payload["synthetic"] is not expected_synthetic):
        raise ReconstructionBlocked("checkpoint_policy_or_input_identity_mismatch")
    if payload["inference_audit"] != to_jsonable(expected_inference_audit):
        raise ReconstructionBlocked("checkpoint_inference_provenance_changed")
    state = from_jsonable(ReconstructionState, payload["state"])
    if to_jsonable(state) != payload["state"]:
        raise ReconstructionBlocked("unknown_checkpoint_state_fields")
    counters = (state.usage.rounds, state.usage.proposals, state.usage.added_atoms,
                state.usage.tool_calls, state.usage.inference_calls, state.usage.failures, state.usage.stagnant)
    if (any(type(value) is not int or value < 0 for value in counters)
            or not math.isfinite(state.usage.elapsed_seconds) or state.usage.elapsed_seconds < 0):
        raise ReconstructionBlocked("invalid_checkpoint_resource_counters")
    if (state.assessment is None or state.assessment.blockers()
            or state.assessment.basis_key != state.model.basis_key):
        raise ReconstructionBlocked("checkpoint_not_an_accepted_model")
    state.parent_checkpoint_sha256 = expected_sha256
    state.stop_code = ""
    return state
