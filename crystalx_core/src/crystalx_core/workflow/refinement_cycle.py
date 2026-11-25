"""Pretrained assignment/refinement StageExecutor for the existing OnionWorkflow.

The owner-selected development mode defers scientific gates and stops on numerical stability.
It retains unvalidated candidates, never converts that stopping rule into scientific convergence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from copy import deepcopy
from datetime import datetime, timezone
import math
import os
import shutil
from pathlib import Path
import time
from collections import Counter

from crystalx_core.domain.candidate import Candidate, CandidateProvenance, Mutation, MutationKind
from crystalx_core.domain.chemistry import ATOMIC_NUMBERS
from crystalx_core.domain.ids import make_candidate_id
from crystalx_core.domain.metrics import RefinementMetrics
from crystalx_core.domain.onion import StageKind
from crystalx_core.domain.result import StopReasonKind
from crystalx_core.refinement.cycle import (
    FileRefinementResult, assign_to_model, checked_res, important_movement, important_sites,
    inference_input, refine_file, setting_key,prune_model, periodic_distance, cell_values,
    displacement_value,METALS,
)
from crystalx_core.refinement.adaptive import (AdaptivePolicy,propose_assignments,propose_pruning,
    select_growth_peaks,geometry_summary, AssignmentProposal,metal_ripple_rejections)
from crystalx_core.tools.runner import CancelToken, ResourceLimits
from crystalx_core.workflow.onion import StageControlStop, StageRequest, StageResult
from crystalx_core.workflow.reconstruction import ReconstructionBlocked, digest
from crystalx_core.workspace.hashing import sha256_file


@dataclass(frozen=True)
class CyclePolicy:
    iterations: int | None = None
    max_rounds: int = 20
    stable_transitions: int = 2
    r1_tolerance: float = 0.001
    coordinate_tolerance_angstrom: float = 0.02
    shelxl_cycles: int = 4
    plan_peaks: int = 20
    fourier_map: int = 2
    anisotropic_metals: bool = False
    metal_ripple_exclusion_angstrom: float = 0.0
    refinement_method: str = "ls"
    early_cgls_rounds: int = 0
    damp_factor: float | None = None
    damp_shift_esd_limit: float = 15.0
    refinement_displacement_threshold_angstrom: float = 0.5
    max_refinement_net_shift_angstrom: float | None = None
    rejection_backoff: bool = True
    rejection_budget_scope: str = "base_model_streak"
    backoff_failure_budget: int = 4
    rejected_action_ttl_rounds: int = 3
    anchor_uiso_guard: bool = False
    anchor_uiso_cap: float = 2.0
    anchor_uiso_policy: str = "unlock_or_block"
    anchor_min_supported: int = 2
    max_new_peaks_per_round: int = 12
    minimum_peak_height: float = 0.5
    minimum_peak_distance_angstrom: float = 0.8
    minimum_assignment_probability: float = 0.0
    max_r1_regression: float = 0.02
    max_failures: int = 2
    max_wall_seconds: float = 7200.
    trial_timeout_s: float = 300.
    important_atom_ids: tuple[str, ...] = ()
    inference_protocol: str = "current_labels"
    inference_seed: int = 42
    stopping_policy: str = "movable_and_growth"
    require_no_element_changes_for_stop: bool = True
    record_diagnostics: bool = False
    adaptive: AdaptivePolicy = field(default_factory=AdaptivePolicy)

    def __post_init__(self):
        from crystalx_core.refinement.shelxl import RefinementSettings
        RefinementSettings(method=self.refinement_method, damp_factor=self.damp_factor,
                           damp_shift_esd_limit=self.damp_shift_esd_limit)
        if type(self.early_cgls_rounds) is not int or self.early_cgls_rounds < 0:
            raise ValueError("early CGLS round count must be nonnegative")
        limit = self.max_refinement_net_shift_angstrom
        if limit is not None and (type(limit) not in (int,float) or not math.isfinite(limit) or limit <= 0):
            raise ValueError("net refinement shift limit must be finite and positive or None")
        if type(self.refinement_displacement_threshold_angstrom) not in (int,float) or not math.isfinite(self.refinement_displacement_threshold_angstrom) or self.refinement_displacement_threshold_angstrom <= 0:
            raise ValueError("displacement reporting threshold must be finite and positive")
        if self.rejection_budget_scope not in ("base_model_streak", "run_total"):
            raise ValueError("invalid rejection budget scope")
        if type(self.backoff_failure_budget) is not int or self.backoff_failure_budget < 4:
            raise ValueError("backoff failure budget must allow all four action levels")
        if type(self.rejected_action_ttl_rounds) is not int or self.rejected_action_ttl_rounds < 0:
            raise ValueError("rejected action lifetime must be a nonnegative integer")
        if self.anchor_uiso_policy not in ("unlock_or_block", "block_any", "report_only"):
            raise ValueError("invalid anchor Uiso policy")
        if type(self.anchor_min_supported) is not int or self.anchor_min_supported < 1:
            raise ValueError("anchor minimum support must be positive")
        if type(self.plan_peaks) is not int or not 0 <= self.plan_peaks <= 999:
            raise ValueError("plan_peaks must be an integer in [0,999]")
        if type(self.fourier_map) is not int or self.fourier_map not in (2,3,4):
            raise ValueError("fourier_map must be 2, 3 or 4")
        if type(self.anisotropic_metals) is not bool:
            raise ValueError("anisotropic_metals must be boolean")
        if (type(self.metal_ripple_exclusion_angstrom) not in (int,float)
                or not math.isfinite(self.metal_ripple_exclusion_angstrom) or self.metal_ripple_exclusion_angstrom < 0):
            raise ValueError("metal ripple exclusion must be finite and nonnegative")
        if type(self.rejection_backoff) is not bool or type(self.anchor_uiso_guard) is not bool:
            raise ValueError("backoff and anchor guard switches must be boolean")
        if type(self.anchor_uiso_cap) not in (int, float) or not math.isfinite(self.anchor_uiso_cap) or self.anchor_uiso_cap <= 0:
            raise ValueError("anchor Uiso cap must be finite and positive")
        if self.inference_protocol not in ("upstream_ins","initial_labels","current_labels"):
            raise ValueError("invalid inference input protocol")
        if type(self.inference_seed) is not int or not 0<=self.inference_seed<2**32:
            raise ValueError("invalid recorded inference seed")
        if self.stopping_policy not in ("legacy_important","movable_and_growth"):
            raise ValueError("invalid numerical stopping policy")
        if (type(self.require_no_element_changes_for_stop) is not bool or type(self.record_diagnostics) is not bool
                or not isinstance(self.adaptive,AdaptivePolicy)):
            raise ValueError("typed adaptive policy and explicit stop flag required")
        object.__setattr__(self, "important_atom_ids", tuple(self.important_atom_ids))
        for name in ("max_rounds", "stable_transitions", "shelxl_cycles", "max_failures"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("cycle integer budgets must be positive")
        if self.iterations is not None and (type(self.iterations) is not int or not 1 <= self.iterations <= self.max_rounds):
            raise ValueError("explicit iterations must fit the outer-round safety budget")
        if type(self.max_new_peaks_per_round) is not int or not 0 <= self.max_new_peaks_per_round <= 200:
            raise ValueError("new-peak budget must be in [0,200]")
        for name in ("r1_tolerance", "coordinate_tolerance_angstrom", "minimum_peak_height",
                     "minimum_peak_distance_angstrom", "max_r1_regression", "max_wall_seconds", "trial_timeout_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("cycle numerical limits must be finite and positive")
        if (type(self.minimum_assignment_probability) not in (int,float)
                or not math.isfinite(self.minimum_assignment_probability)
                or not 0 <= self.minimum_assignment_probability <= 1 or self.shelxl_cycles > 50):
            raise ValueError("invalid assignment probability or refinement-cycle bound")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("unsupported cycle policy fields")
        value=dict(value)
        if "adaptive" in value and isinstance(value["adaptive"],dict):value["adaptive"]=AdaptivePolicy.from_dict(value["adaptive"])
        return cls(**value)

    def to_dict(self):
        return asdict(self)


@dataclass
class CycleState:
    model_path: Path
    hkl_path: Path
    locked_sites: tuple[str, ...] = ()
    important_sites: tuple[str, ...] = ()
    result: FileRefinementResult | None = None
    attempts: int = 0
    inference_calls: int = 0
    tool_calls: int = 0
    failures: int = 0
    stable_count: int = 0
    stop_code: str = ""
    history: list[dict] = field(default_factory=list)
    adaptive_state: dict = field(default_factory=dict)
    rounds_without_accepted_sites: int = 0
    rejected_proposals: dict = field(default_factory=dict)
    rejected_action_sets: dict = field(default_factory=dict)
    base_rejection_count: int = 0
    repeated_rejected_actions: int = 0
    skipped_rejected_actions: int = 0
    skipped_rejected_proposals: int = 0
    rejection_base_key: str = ""
    rejection_backoff_level: int = 0


class RefinementCycleExecutor:
    def __init__(self, state: CycleState, *, assigner, adapter, policy: CyclePolicy,
                 candidate_elements: tuple[str, ...], cancel: CancelToken,
                 synthetic: bool, clock=time.monotonic, started: float | None = None,
                 original_ins_path:Path|None=None,initial_labels:dict[str,str]|None=None,
                 original_ins_sha256:str|None=None):
        self.state, self.assigner, self.adapter, self.policy = state, assigner, adapter, policy
        self.candidate_elements, self.cancel, self.synthetic = candidate_elements, cancel, synthetic
        self.clock, self.started = clock, clock() if started is None else started
        self.hkl_sha = sha256_file(state.hkl_path)
        self.original_ins_path=original_ins_path
        self.original_ins_sha=original_ins_sha256 or (sha256_file(original_ins_path) if original_ins_path else None)
        if original_ins_path and sha256_file(original_ins_path)!=self.original_ins_sha:
            raise ReconstructionBlocked("original_instruction_staged_hash_differs_from_upload")
        self.initial_labels=dict(initial_labels or {})
        self.initial_labels_sha=digest(self.initial_labels)
        self.baseline_site_ids=tuple(a.label for a in self._read_model(state.model_path).atoms)
        self.current_sha = sha256_file(state.model_path)
        self.pending = None
        self._transaction_stage = None
        self._accepted_before = None
        self._stage_entry = None
        self._transaction_aborted = False
        self._commit_started = False

    def check(self):
        if self.cancel.cancelled:
            raise StageControlStop(StopReasonKind.USER_ABORTED, "cancelled")
        elapsed = self.clock()-self.started
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed >= self.policy.max_wall_seconds:
            raise StageControlStop(StopReasonKind.BUDGET_EXCEEDED, "whole_chain_wall_budget_exhausted")
        if sha256_file(self.state.hkl_path) != self.hkl_sha or sha256_file(self.state.model_path) != self.current_sha:
            raise ReconstructionBlocked("accepted_model_or_reflection_input_changed")
        if ((self.original_ins_path is not None and sha256_file(self.original_ins_path)!=self.original_ins_sha)
                or digest(self.initial_labels)!=self.initial_labels_sha):
            raise ReconstructionBlocked("original_instruction_or_initial_descriptor_source_changed")
        return elapsed

    def execute(self, request: StageRequest) -> StageResult:
        self._transaction_stage = str(request.stage.stage_id)
        self._accepted_before = (self.state.model_path, self.state.result, self.current_sha,
                                 self.state.stable_count, self.state.stop_code,
                                 deepcopy(self.state.adaptive_state),self.state.rounds_without_accepted_sites,
                                 self.state.locked_sites,self.state.rejection_base_key,
                                 self.state.rejection_backoff_level,self.state.base_rejection_count)
        self._stage_entry = None
        self._transaction_aborted = False
        self._commit_started = False
        try:
            return self._execute(request)
        except ReconstructionBlocked as error:
            self.state.stop_code=str(error)
            return StageResult(attempts_used=0,requires_human_review=True,summary=self.state.stop_code)

    def _read_model(self,path):
        model=checked_res(path,allow_anisotropic=self.policy.anisotropic_metals)
        if any(a.is_anisotropic and model.element_of(a) not in METALS for a in model.atoms):
            raise ReconstructionBlocked("anisotropic_nonmetal_outside_explicit_metal_policy")
        return model

    def _prepare_trial(self, request, current, directory, level):
        state = self.state
        policy = self.policy
        if level:
            policy = replace(policy, adaptive=replace(policy.adaptive, pruning_enabled=False),
                             max_new_peaks_per_round=0 if level >= 2 else policy.max_new_peaks_per_round)
        basis = setting_key(current, self.hkl_sha)
        if level == 3:
            request.workspace.write_json(str((directory/"growth-selection.json").relative_to(request.workspace.layout.root)),
                {"enabled":False,"mode":"backoff_pure_refinement","decisions":[
                    {"peak_id":p.label,"height":p.height,"decision":"reject","reason":"backoff_pure_refinement"} for p in current.q_peaks]})
            pruning = propose_pruning(current, policy=replace(policy.adaptive, pruning_enabled=False), state=state.adaptive_state)
            inference_path = directory / "inference-input.res"
            selected = inference_input(current, inference_path, max_new_peaks=0,
                min_peak_height=policy.minimum_peak_height, min_distance=policy.minimum_peak_distance_angstrom)
            proposed_path = directory / "assigned.res"
            shutil.copyfile(inference_path, proposed_path)
            proposal = AssignmentProposal({}, deepcopy(state.adaptive_state), {"mode": "pure_refinement_no_assignment"})
            return pruning, None, proposal, proposed_path, {"changes": []}, selected, inference_path
        pruning=propose_pruning(current,policy=policy.adaptive,state=state.adaptive_state,
            protected_site_ids=state.locked_sites,baseline_site_ids=self.baseline_site_ids)
        request.workspace.write_json(str((directory/"pruning-proposal.json").relative_to(request.workspace.layout.root)),pruning.evidence)
        working=current
        if pruning.site_ids:
            pruned_path=directory/"pruned-input.res"
            archive=prune_model(current,pruned_path,pruning.site_ids,protected_site_ids=state.locked_sites,
                allow_anisotropic_metals=policy.anisotropic_metals)
            request.workspace.write_json(str((directory/"deleted-sites.json").relative_to(request.workspace.layout.root)),archive)
            working=self._read_model(pruned_path)
        growth_ids,growth_evidence=(),{"enabled":False,"mode":"legacy_absolute_height_and_separation"}
        if policy.adaptive.growth_filter_enabled:
            growth_ids,growth_evidence=select_growth_peaks(working,policy=policy.adaptive,
                round_number=state.attempts,limit=policy.max_new_peaks_per_round,
                minimum_height=policy.minimum_peak_height,minimum_distance=policy.minimum_peak_distance_angstrom,
                metal_ripple_exclusion_angstrom=policy.metal_ripple_exclusion_angstrom)
        excluded={};legacy_decisions=[]
        if not policy.adaptive.growth_filter_enabled and policy.metal_ripple_exclusion_angstrom:
            excluded=metal_ripple_rejections(working,policy=policy.adaptive,radius=policy.metal_ripple_exclusion_angstrom)
        peak_limit=max(policy.max_new_peaks_per_round,len(growth_ids))
        inference_path = directory/"inference-input.res"
        selected = inference_input(working,inference_path,max_new_peaks=peak_limit,
            min_peak_height=policy.minimum_peak_height,min_distance=policy.minimum_peak_distance_angstrom,
            selected_peak_ids=growth_ids if policy.adaptive.growth_filter_enabled else None,
            excluded_peak_ids=tuple(excluded),decision_records=legacy_decisions if not policy.adaptive.growth_filter_enabled else None)
        if not policy.adaptive.growth_filter_enabled:
            growth_evidence.update(decisions=[{**row,**excluded.get(row["peak_id"],{})} for row in legacy_decisions],
                metal_ripple_exclusion_angstrom=policy.metal_ripple_exclusion_angstrom)
        request.workspace.write_json(str((directory/"growth-selection.json").relative_to(request.workspace.layout.root)),growth_evidence)
        inference_sha=sha256_file(inference_path)
        basis=setting_key(current,self.hkl_sha)
        if setting_key(self._read_model(inference_path),self.hkl_sha)!=basis:
            raise ReconstructionBlocked("inference_preparation_changed_reflection_settings")
        state.inference_calls += 1
        request.workspace.write_json(str((directory/"in-flight.json").relative_to(request.workspace.layout.root)),
            {"attempts":state.attempts,"inference_calls":state.inference_calls,"tool_calls":state.tool_calls,
             "accepted_model_sha256":self.current_sha,"scientifically_validated":False})
        protocol_kwargs={}
        if policy.inference_protocol!="current_labels":
            if self.original_ins_path is None or sha256_file(self.original_ins_path)!=self.original_ins_sha:
                raise ReconstructionBlocked("original_instruction_protocol_source_changed_or_missing")
            protocol_kwargs={"input_protocol":policy.inference_protocol,"original_ins_path":self.original_ins_path,
                "initial_labels":dict(self.initial_labels),"protocol_seed":policy.inference_seed}
        assignment = self.assigner.assign(inference_path,candidate_elements=self.candidate_elements,include_q_peaks=True,**protocol_kwargs)
        self.check(); request.checkpoint()
        if (assignment.model_sha256 != inference_sha or sha256_file(inference_path)!=inference_sha
                or assignment.result.input_sha256 != digest(assignment.request)
                or assignment.result.fell_back):
            raise ReconstructionBlocked("pretrained_request_model_or_device_identity_mismatch")
        if not self.synthetic and assignment.result.probability_kind.startswith("synthetic"):
            raise ReconstructionBlocked("synthetic_predictions_for_measured_data_refused")
        request.workspace.write_json(str((directory/"inference.json").relative_to(request.workspace.layout.root)),
                                     assignment.to_record())
        provenance=getattr(assignment,"provenance",{})
        if not self.synthetic and provenance.get("input_protocol","current_labels")!=policy.inference_protocol:
            raise ReconstructionBlocked("actual_inference_protocol_differs_from_frozen_policy")
        protocol_choices=provenance.get("protocol_element_choices")
        proposal=propose_assignments(self._read_model(inference_path),assignment.result.predictions,
            policy=policy.adaptive,state=pruning.state,protected_site_ids=state.locked_sites,
            proposed_elements=protocol_choices)
        request.workspace.write_json(str((directory/"assignment-policy.json").relative_to(request.workspace.layout.root)),proposal.evidence)
        overrides={**(protocol_choices or {}),**proposal.overrides} if (policy.adaptive.hysteresis_enabled or policy.adaptive.uiso_checks_enabled) else protocol_choices
        retired=tuple(sorted(set(state.adaptive_state.get("retired_site_ids",()))|set(pruning.site_ids)))
        proposal.state["retired_site_ids"]=retired
        proposed_path = directory/"assigned.res"
        edits = assign_to_model(self._read_model(inference_path),assignment.result.predictions,proposed_path,
            candidate_elements=self.candidate_elements,locked_sites=state.locked_sites,
            minimum_probability=policy.minimum_assignment_probability,element_overrides=overrides,reserved_site_ids=retired,
            allow_anisotropic_metals=policy.anisotropic_metals)
        if setting_key(self._read_model(proposed_path),self.hkl_sha)!=basis:
            raise ReconstructionBlocked("assignment_changed_reflection_settings")
        return pruning, assignment, proposal, proposed_path, edits, selected, inference_path

    def _model_key(self, model):
        # Exclude comments and replaced controls, but retain scale/weights and all
        # other consumed instructions: equal coordinates do not imply equal LS input.
        ignored = {"TITL", "REM", "L.S.", "CGLS", "PLAN", "LIST", "ACTA", "FMAP"}
        if self.policy.damp_factor is not None:
            ignored.add("DAMP")
        atom_ids = {a.label.upper() for a in model.atoms} | {q.label.upper() for q in model.q_peaks}
        context = []
        from crystalx_core.io.shelx_reader import logical_lines
        for line in logical_lines(model.source):
            if line.instruction == "END":break
            if line.instruction not in ignored | atom_ids:
                context.append(line.tokens)
            if line.instruction == "HKLF":break
        return digest({"setting": setting_key(model, self.hkl_sha), "retained_instructions": context,
            "atoms": [(a.label, model.element_of(a), a.x, a.y, a.z, a.occupancy_code, a.u_values)
                      for a in sorted(model.atoms, key=lambda a: a.label)]})

    def _failure_budget(self):
        return max(self.policy.max_failures, self.policy.backoff_failure_budget) if self.policy.rejection_backoff else self.policy.max_failures

    @staticmethod
    def _action_key(action_sets):
        changes = action_sets["element_and_growth_changes"]
        payload = {"deleted": sorted(action_sets["deleted_site_ids"]),
            "elements": sorted((row["site_id"], row["from"], row["to"]) for row in changes if row["from"] != "Q_PEAK"),
            "growth": sorted((row.get("peak_id", row["site_id"]), row["to"]) for row in changes if row["from"] == "Q_PEAK")}
        return digest(payload) if any(payload.values()) else None

    def _remember_rejection(self, request, entry, reason):
        state = self.state
        key = entry["proposal_sha256"]
        if state.rejection_base_key != entry["base_model_key"]:
            state.base_rejection_count = 0
        if key not in state.rejected_proposals:
            state.failures += 1
            state.base_rejection_count += 1
            state.rejected_proposals[key] = {"proposal_sha256": key, "base_model_key": entry["base_model_key"],
                "action_sets": entry["action_sets"], "backoff_level": entry["backoff_level"],
                "first_round": state.attempts, "reason": reason, "r1": entry.get("r1")}
        action_key = entry["action_set_sha256"]
        if action_key:
            previous = state.rejected_action_sets.get(action_key, {})
            repeated = bool(previous)
            state.repeated_rejected_actions += int(repeated)
            state.rejected_action_sets[action_key] = {"action_sets": entry["action_sets"],
                "last_rejected_round": state.attempts, "rejection_count": previous.get("rejection_count", 0) + 1}
            entry["repeated_rejected_action_set"] = repeated
        entry["distinct_rejections_total"] = state.failures
        entry["base_rejection_count"] = state.base_rejection_count
        state.rejection_base_key = entry["base_model_key"]
        if self.policy.rejection_backoff:
            state.rejection_backoff_level = entry["backoff_level"] + 1
            if state.rejection_backoff_level > 3:
                state.stop_code = "no_acceptable_step_in_current_action_set"
        request.workspace.write_json("rejected-proposals.json", state.rejected_proposals, allow_replace=True)
        request.workspace.write_json("rejected-action-sets.json", state.rejected_action_sets, allow_replace=True)

    def _execute(self, request: StageRequest) -> StageResult:
        self.pending = None
        self.check(); request.checkpoint()
        if request.stage.kind in (StageKind.ANCHOR, StageKind.FIRST_SHELL):
            return StageResult(attempts_used=0, summary="retained explicit provisional scaffold; scientific checks deferred")
        if self.state.stop_code:
            return StageResult(attempts_used=0, requires_human_review=True,
                               summary=self.state.stop_code, notes=("candidate is unvalidated; separate checkCIF available",))
        budget_count = self.state.base_rejection_count if self.policy.rejection_backoff and self.policy.rejection_budget_scope == "base_model_streak" else self.state.failures
        if budget_count >= self._failure_budget():
            self.state.stop_code = "failure_budget_exhausted"
            return StageResult(attempts_used=0, requires_human_review=True, summary=self.state.stop_code)
        if self.state.attempts >= (self.policy.iterations or self.policy.max_rounds):
            self.state.stop_code = "fixed_rounds_completed" if self.policy.iterations else "round_safety_budget_exhausted"
            return StageResult(attempts_used=0, requires_human_review=True, summary=self.state.stop_code)
        if request.stage.kind is StageKind.FINALIZE:
            return StageResult(attempts_used=0, requires_human_review=True, summary="candidate_ready_for_separate_checkcif")
        state, policy = self.state, self.policy
        state.attempts += 1
        directory = request.workspace.layout.root / f"cycles/round-{state.attempts:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        current = self._read_model(state.model_path)
        if not state.important_sites:
            state.important_sites = important_sites(current, policy.important_atom_ids)
        basis = setting_key(current, self.hkl_sha)
        base_key = self._model_key(current)
        level = (state.rejection_backoff_level if policy.rejection_backoff and state.rejection_base_key == base_key else 0)
        method = "cgls" if state.attempts <= policy.early_cgls_rounds else policy.refinement_method
        skipped = []
        while True:
            if level > 3:
                state.stop_code = "no_acceptable_step_in_current_action_set"
                request.workspace.write_json(str((directory/"skipped-rejected-proposals.json").relative_to(request.workspace.layout.root)), skipped)
                return StageResult(attempts_used=0, requires_human_review=True, summary=state.stop_code)
            trial_directory = directory if level == 0 else directory / f"backoff-{level}"
            trial_directory.mkdir(parents=True, exist_ok=True)
            pruning, assignment, proposal, proposed_path, edits, selected, inference_path = self._prepare_trial(
                request, current, trial_directory, level)
            proposal_key = digest({"base": base_key, "proposed": self._model_key(self._read_model(proposed_path)),
                                   "cycles": policy.shelxl_cycles, "plan_peaks": policy.plan_peaks,
                                   "fourier_map":policy.fourier_map,"anisotropic_metals":policy.anisotropic_metals,
                                   "metal_ripple_exclusion_angstrom":policy.metal_ripple_exclusion_angstrom,
                                   "method": method, "locked_sites": state.locked_sites, "damp_factor": policy.damp_factor,
                                   "damp_shift_esd_limit": policy.damp_shift_esd_limit})
            action_sets = {"deleted_site_ids": pruning.site_ids,
                           "element_and_growth_changes": edits["changes"]}
            action_key = self._action_key(action_sets)
            old_action = state.rejected_action_sets.get(action_key, {}) if action_key else {}
            recent_action = bool(old_action and state.attempts - old_action["last_rejected_round"] <= policy.rejected_action_ttl_rounds)
            exact_duplicate = proposal_key in state.rejected_proposals
            if not policy.rejection_backoff or not (exact_duplicate or recent_action):
                break
            state.skipped_rejected_actions += int(recent_action)
            state.skipped_rejected_proposals += int(exact_duplicate)
            skipped.append({"proposal_sha256": proposal_key, "action_set_sha256": action_key,
                "backoff_level": level, "action_sets": action_sets,
                "reason": "recent_rejected_action_set" if recent_action else "exact_rejected_proposal"})
            self.check(); request.checkpoint()
            level += 1
        request.workspace.write_json(str((directory/"proposal-identity.json").relative_to(request.workspace.layout.root)),
            {"proposal_sha256": proposal_key, "base_model_key": base_key, "action_sets": action_sets,
             "backoff_level": level, "skipped_duplicates": skipped, "action_set_sha256": action_key, "inference_executed": assignment is not None})
        added = sum(item["from"] == "Q_PEAK" for item in edits["changes"])
        request.meter.reserve(attempts=1,candidates=1,new_atoms=added)
        request.workspace.write_json(str((directory/"assignments.json").relative_to(request.workspace.layout.root)),edits)
        request.before_call(); state.tool_calls += 1
        request.workspace.write_json(str((directory/"tool-call-reservation.json").relative_to(request.workspace.layout.root)),
            {"tool_calls":state.tool_calls,"attempts":state.attempts,"input_sha256":sha256_file(proposed_path)})
        result = refine_file(model_path=proposed_path,hkl_path=state.hkl_path,directory=directory/"shelxl",
            adapter=self.adapter,cycles=policy.shelxl_cycles,plan_peaks=policy.plan_peaks,
            fourier_map=policy.fourier_map,anisotropic_metals=policy.anisotropic_metals,
            method=method,damp_factor=policy.damp_factor,damp_shift_esd_limit=policy.damp_shift_esd_limit,
            timeout_s=min(policy.trial_timeout_s,policy.max_wall_seconds-self.check()),cancel=self.cancel,
            locked_sites=state.locked_sites,limits=ResourceLimits(cpu_seconds=max(1,math.ceil(policy.trial_timeout_s*(os.cpu_count() or 1))),
                max_file_bytes=128*1024*1024,max_open_files=256))
        self.check(); request.checkpoint()
        returned = self._read_model(result.model_path)
        trial_model = self._read_model(proposed_path)
        before_refinement = {a.label: a for a in trial_model.atoms}
        cell = cell_values(trial_model.header.cell)
        shifts = {a.label: periodic_distance(cell,
            (before_refinement[a.label].x,before_refinement[a.label].y,before_refinement[a.label].z), (a.x,a.y,a.z))
            for a in returned.atoms if a.label in before_refinement}
        limit = policy.max_refinement_net_shift_angstrom
        over_limit_sites = sorted(key for key,value in shifts.items() if limit is not None and value > limit)
        displacement = {"maximum_angstrom": max(shifts.values(), default=None),
            "threshold_angstrom": policy.refinement_displacement_threshold_angstrom,
            "above_threshold_site_ids": sorted(key for key,value in shifts.items() if value > policy.refinement_displacement_threshold_angstrom),
            "above_threshold_count": sum(value > policy.refinement_displacement_threshold_angstrom for value in shifts.values()),
            "compared_site_count": len(shifts), "scope": "proposed_model_to_refined_model_periodic_same_site_ids",
            "fixed_site_ids": state.locked_sites, "diagnostic_only": limit is None,
            "rejection_limit_angstrom": limit, "over_rejection_limit_site_ids": over_limit_sites}
        request.workspace.write_json(f"cycles/round-{state.attempts:03d}/refinement-displacements.json", displacement)
        if policy.record_diagnostics:
            from crystalx_core.refinement.diagnostics import model_diagnostics
            from crystalx_core.refinement.adaptive import uiso_consistency
            metrics=model_diagnostics(returned,policy=policy.adaptive)
            request.workspace.write_json(f"cycles/round-{state.attempts:03d}/answer-free-metrics.json",metrics)
            request.workspace.write_json(f"cycles/round-{state.attempts:03d}/post-refinement-u.json",
                uiso_consistency(returned,policy=policy.adaptive,state=proposal.state))
        if setting_key(returned,self.hkl_sha)!=basis:
            raise ReconstructionBlocked("refinement_changed_accepted_reflection_settings")
        delta_r = abs(result.r1-state.result.r1) if state.result else None
        common={a.label for a in current.atoms}&{a.label for a in returned.atoms}
        monitored=tuple(key for key in state.important_sites if key in common)
        if policy.stopping_policy=="movable_and_growth":
            monitored=tuple(key for key in monitored if key not in state.locked_sites)
            if not monitored:
                monitored=tuple(a.label for a in returned.atoms if a.label in common and a.label not in state.locked_sites
                    and returned.element_of(a)!="H")
        movement = important_movement(current,returned,monitored) if state.result and monitored else None
        before_elements={a.label:current.element_of(a) for a in current.atoms}
        after_elements={a.label:returned.element_of(a) for a in returned.atoms}
        same_important_elements=all(before_elements[key]==after_elements[key] for key in monitored)
        changed=sum(item["from"]!="Q_PEAK" for item in edits["changes"])
        quiet_rounds=state.rounds_without_accepted_sites+1 if added==0 and not pruning.site_ids else 0
        stable = (delta_r is not None and delta_r <= policy.r1_tolerance and movement is not None
                  and movement <= policy.coordinate_tolerance_angstrom and same_important_elements)
        if policy.stopping_policy=="movable_and_growth":
            stable=stable and quiet_rounds>=2 and not pruning.site_ids
        if policy.require_no_element_changes_for_stop:stable=stable and changed==0
        reject_r1 = state.result is not None and result.r1 > state.result.r1 + policy.max_r1_regression
        reject_displacement = bool(over_limit_sites)
        reject = reject_r1 or reject_displacement
        inference_model=self._read_model(inference_path) if assignment is not None else None
        import json
        growth_record=json.loads((trial_directory/"growth-selection.json").read_text())
        entry = {"round":state.attempts,"model_sha256":sha256_file(result.model_path),"r1":result.r1,
            "r1_change":delta_r,"important_coordinate_max_change_angstrom":movement,
            "important_site_ids":state.important_sites,"selected_partial_q_peaks":selected,
            "monitored_movable_site_ids":monitored,"excluded_fixed_site_ids":state.locked_sites,
            "inference_protocol":policy.inference_protocol,"inference_seed":policy.inference_seed,
            "nonpositive_u_site_ids":sorted(a.label for a in inference_model.atoms
                if (value:=displacement_value(inference_model,a)) is not None and value<=0) if assignment is not None else [],
            "proposal_sha256":proposal_key,"base_model_key":base_key,"action_sets":action_sets,"backoff_level":level,
            "action_set_sha256":action_key,"effective_failure_budget":self._failure_budget(),
            "rejection_budget_scope":policy.rejection_budget_scope,
            "plan_peaks":policy.plan_peaks,"inference_executed":assignment is not None,
            "fourier_map":policy.fourier_map,"anisotropic_metals":policy.anisotropic_metals,
            "metal_ripple_exclusion_angstrom":policy.metal_ripple_exclusion_angstrom,
            "listed_input_peak_count":len(current.q_peaks),"listed_output_peak_count":len(returned.q_peaks),
            "selected_peak_count":len(selected),"accepted_new_site_count":0,
            "peak_rejection_reason_counts":dict(Counter(row['reason'] for row in growth_record.get('decisions',[]) if row['decision']!='select')),
            "refinement_method":method,"damp_factor":policy.damp_factor,"damp_shift_esd_limit":policy.damp_shift_esd_limit,
            "refinement_max_displacement_angstrom":displacement["maximum_angstrom"],
            "refinement_displacement_above_threshold_count":displacement["above_threshold_count"],
            "refinement_displacement_threshold_angstrom":policy.refinement_displacement_threshold_angstrom,
            "max_refinement_net_shift_angstrom":limit,
            "refinement_displacement_limit_exceeded":reject_displacement,
            "stopping_policy":policy.stopping_policy,"accepted_new_sites_if_retained":added,
            "deleted_site_ids_if_retained":pruning.site_ids,"element_changes":changed,
            "rounds_without_accepted_sites_if_retained":quiet_rounds,
            "wR2":result.wr2,"goof":result.goof,"n_parameters":result.n_parameters,
            "stable_transition":stable,"scientifically_validated":False,
            "decision":"REJECT_REFINEMENT_DISPLACEMENT" if reject_displacement else "REJECT_R_REGRESSION" if reject_r1 else "RETAIN_UNVALIDATED_PENDING_CONTROLLER",
            "model_path":str(result.model_path.relative_to(request.workspace.layout.root))}
        state.history.append(entry)
        self._stage_entry = entry
        request.workspace.write_json(str((directory/"decision.json").relative_to(request.workspace.layout.root)),entry)
        next_locked_sites = state.locked_sites
        if policy.anchor_uiso_guard and state.locked_sites:
            anchors = {a.label: displacement_value(returned,a) for a in returned.atoms if a.label in state.locked_sites}
            if any(value is None for value in anchors.values()):
                raise ReconstructionBlocked("anchor_displacement_scalar_unavailable")
            bad = {key: value for key, value in anchors.items()
                   if value >= policy.anchor_uiso_cap or value > policy.adaptive.u_abs_high}
            supported = tuple(key for key in state.locked_sites if key not in bad)
            blocked = bool(policy.anchor_uiso_policy != "report_only" and
                (len(supported) < policy.anchor_min_supported or policy.anchor_uiso_policy == "block_any" and bad))
            if bad and not blocked and policy.anchor_uiso_policy == "unlock_or_block":
                next_locked_sites = supported
            entry["anchor_sites_to_unlock_if_retained"] = sorted(set(state.locked_sites) - set(next_locked_sites))
            request.workspace.write_json(f"cycles/round-{state.attempts:03d}/anchor-u-check.json",
                {"observed_u": anchors, "unsupported_u": bad, "u_abs_high": policy.adaptive.u_abs_high,
                 "upper_cap": policy.anchor_uiso_cap, "minimum_supported": policy.anchor_min_supported,
                 "policy": policy.anchor_uiso_policy, "blocked": blocked,
                 "unlock_if_retained": entry["anchor_sites_to_unlock_if_retained"],
                 "atom_records_deleted": False, "expert_used": False})
            if blocked:
                entry["decision"] = "BLOCK_UNSUPPORTED_ANCHORS"
                state.stop_code = "anchor_displacement_unsupported"
                request.workspace.write_json(f"cycles/round-{state.attempts:03d}/decision.json", entry, allow_replace=True)
                return StageResult(attempts_used=1, tool_calls_used=1, requires_human_review=True, summary=state.stop_code)
        candidate_id = make_candidate_id(request.workspace.next_candidate_index())
        candidate = Candidate(candidate_id,CandidateProvenance(datetime.now(timezone.utc).isoformat(),
            request.stage.stage_id,parent_id=request.input_candidate.candidate_id if request.input_candidate else None,
            reflection_file_sha256=self.hkl_sha,input_hashes={"model":sha256_file(result.model_path)},
            model_identities={"weights":assignment.result.model_identity.weights_sha256} if assignment is not None else {},
            notes="Owner-requested development reconstruction. Scientific gates deferred; not a validated structure."),
            mutations=(Mutation(MutationKind.ELEMENT_FLIP,"pretrained assignment and bounded density-supported growth",
                                driver="pretrained_crystalx"),) if assignment is not None else (),
            res_path=str(result.model_path.relative_to(request.workspace.layout.root)),
            n_non_hydrogen_atoms=sum(returned.element_of(a)!="H" for a in returned.atoms),
            metrics=RefinementMetrics(r1=result.r1,wr2=result.wr2,goof=result.goof,parse_ok=True,
                converged=None,n_reflections_observed=result.n_reflections,n_parameters=result.n_parameters,
                reflection_basis_key=self.hkl_sha))
        if reject:
            self._remember_rejection(request, entry, "net_refinement_displacement" if reject_displacement else "R1_regression")
            state.stable_count = 0
        else:
            self.pending = (result,entry,stable,str(candidate_id),proposal.state,quiet_rounds,next_locked_sites)
        return StageResult(candidates=(candidate,),accepted_candidate_id=None if reject else candidate_id,
            should_rollback=reject,rollback_reason=("net refinement displacement exceeded the configured experimental limit" if reject_displacement else "R1 regressed beyond the configured development limit" if reject_r1 else ""),
            attempts_used=1,tool_calls_used=1,failures_used=int(reject),new_atoms_added=added,
            summary="rejected trial" if reject else "retained unvalidated reconstruction iteration")

    def commit_stage(self, request, chosen, active_lock_ids):
        if self.pending is None:
            return
        result,entry,stable,candidate_id,next_policy_state,quiet_rounds,next_locked_sites = self.pending
        if chosen is None or str(chosen.candidate_id) != candidate_id:
            self.abort_stage(request,"controller_did_not_select_model")
            raise ReconstructionBlocked("controller_did_not_select_pending_model")
        self._commit_started = True
        self.state.result = result;self.state.model_path = result.model_path
        self.current_sha = sha256_file(result.model_path)
        self.state.adaptive_state=deepcopy(next_policy_state)
        self.state.rounds_without_accepted_sites=quiet_rounds
        self.state.locked_sites=next_locked_sites
        self.state.stable_count = self.state.stable_count+1 if stable else 0
        self.state.rejection_backoff_level = 0
        self.state.rejection_base_key = ""
        self.state.base_rejection_count = 0
        entry["decision"] = "RETAIN_UNVALIDATED"
        entry["accepted_new_site_count"] = entry["accepted_new_sites_if_retained"]
        entry["stable_transition_count"] = self.state.stable_count
        entry["adaptive_state_sha256"]=digest(self.state.adaptive_state)
        request.workspace.write_json(f"cycles/round-{self.state.attempts:03d}/policy-state.json",self.state.adaptive_state)
        if self.policy.iterations is not None and self.state.attempts >= self.policy.iterations:
            self.state.stop_code = "fixed_rounds_completed"
        elif self.policy.iterations is None and self.state.stable_count >= self.policy.stable_transitions:
            self.state.stop_code = ("movable_sites_r1_and_growth_stable"
                if self.policy.stopping_policy == "movable_and_growth" else "important_coordinates_and_r1_stable")
        request.workspace.write_json(f"cycles/round-{self.state.attempts:03d}/commit.json",entry)
        self.pending = None
        # The service wraps this hook with status persistence. Keep the matching stage's
        # prior accepted snapshot until the next execute, so an outer write may still abort.

    def abort_stage(self, request, reason):
        if (self._transaction_stage != str(request.stage.stage_id)
                or self._accepted_before is None or self._transaction_aborted):
            return
        self._transaction_aborted = True
        path,result,current_sha,stability,stop_code,adaptive_state,quiet_rounds,locked_sites,old_base_key,old_level,old_count = self._accepted_before
        self.state.model_path,self.state.result,self.current_sha = path,result,current_sha
        self.state.stable_count = stability
        self.state.adaptive_state=deepcopy(adaptive_state)
        self.state.rounds_without_accepted_sites=quiet_rounds
        self.state.locked_sites=locked_sites
        if self._commit_started:
            self.state.stop_code = stop_code
            self.state.rejection_base_key=old_base_key
            self.state.rejection_backoff_level=old_level
            self.state.base_rejection_count=old_count
        entry = self._stage_entry
        if entry is not None and entry["decision"] in ("REJECT_R_REGRESSION", "REJECT_REFINEMENT_DISPLACEMENT"):
            # execute() already charged this rejection. It breaks consecutive stability
            # even though the previous accepted model and constraints remain active.
            self.state.stable_count = 0
        elif entry is not None and entry["decision"] == "BLOCK_UNSUPPORTED_ANCHORS":
            self.state.stable_count = 0
        elif entry is not None:
            entry["decision"] = "REJECTED_BY_CONTROLLER"
            entry["reason"] = reason
            entry["accepted_model_restored_sha256"] = self.current_sha
            entry["accepted_stable_transition_count"] = self.state.stable_count
            self._remember_rejection(request, entry, "controller: " + reason)
        elif reason.startswith("stage_transaction_failed"):
            self.state.failures += 1
        self.pending = None
        if entry is not None:
            entry["accepted_new_site_count"]=0
            request.workspace.write_json(f"cycles/round-{self.state.attempts:03d}/abort.json",
                {"reason":reason,"restored_model_path":str(self.state.model_path),
                 "restored_model_sha256":self.current_sha,"stable_transitions":self.state.stable_count,
                 "stop_code":self.state.stop_code,"attempts":self.state.attempts,
                 "restored_adaptive_state_sha256":digest(self.state.adaptive_state),
                 "restored_quiet_rounds":self.state.rounds_without_accepted_sites,
                 "restored_fixed_site_ids":self.state.locked_sites,
                 "restored_base_rejection_count":self.state.base_rejection_count,
                 "inference_calls":self.state.inference_calls,"tool_calls":self.state.tool_calls,
                 "failures":self.state.failures,"scientifically_validated":False})
