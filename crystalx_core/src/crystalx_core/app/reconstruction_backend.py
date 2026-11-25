"""Frontend-facing multi-route pretrained reconstruction with separate validation.

No expert/reference model is accepted in a solve request. All numerical candidates are
explicitly unvalidated; checkCIF and expert comparison consume a separately pinned output seal.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Callable

from crystalx_core._serde import to_jsonable
from crystalx_core.app.demo import is_demo_input
from crystalx_core.app.reconstruction_device import resolve_inference_device
from crystalx_core.app.models import RunConfig, RunMode
from crystalx_core.app.reconstruction_stops import STOP_CODES, empty_stop_record, stop_record
from crystalx_core.app.solve_routes import (
    InitialRouteConfig, execute_initial_route, plan_initial_route,
)
from crystalx_core.app.toolchain import RealToolchain
from crystalx_core.domain.chemistry import ATOMIC_NUMBERS, ChemicalPrior
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.gates import GateKind, RankingPolicy
from crystalx_core.domain.onion import StageBudget, StageKind
from crystalx_core.domain.radiation import RadiationAssumption
from crystalx_core.domain.tools import ToolKind
from crystalx_core.io.shelx_reader import logical_lines, read_ins_header, scan_hkl
from crystalx_core.private_acceptance.harness import private_directory
from crystalx_core.private_acceptance.sealing import seal_run
from crystalx_core.refinement.cycle import (
    METALS, assign_to_model, checked_res, heavy_only_seed, pinned_bytes, promote_initial_peaks,
)
from crystalx_core.tools.runner import CancelToken, ResourceLimits
from crystalx_core.workflow.onion import OnionPolicy, OnionWorkflow
from crystalx_core.workflow.reconstruction import ReconstructionBlocked, digest
from crystalx_core.workflow.refinement_cycle import CyclePolicy, CycleState, RefinementCycleExecutor
from crystalx_core.workspace.hashing import sha256_file
from crystalx_core.workspace.store import RunWorkspace


FRONTEND_CONTRACT = "2026-11-20"


@dataclass(frozen=True)
class ReconstructionRequest:
    ins_path: Path
    hkl_path: Path
    route: InitialRouteConfig = field(default_factory=InitialRouteConfig)
    cycle: CyclePolicy = field(default_factory=CyclePolicy)
    metal_only: bool = False
    anchor_elements: tuple[str, ...] = ()
    candidate_elements: tuple[str, ...] = ()
    candidate_element_source: str = "original_ins"
    mode: RunMode = RunMode.REAL
    strict_native: bool = True
    tool_paths: tuple[tuple[str, Path], ...] = ()
    checkpoint: Path | None = None
    checkpoint_sha256: str | None = None
    device: str = "auto"
    heavy_preparation: str = "current_loop"
    label: str = "reconstruction"

    def __post_init__(self):
        object.__setattr__(self,"ins_path",Path(self.ins_path).absolute())
        object.__setattr__(self,"hkl_path",Path(self.hkl_path).absolute())
        object.__setattr__(self,"mode",RunMode(self.mode))
        object.__setattr__(self,"anchor_elements",tuple(self.anchor_elements))
        object.__setattr__(self,"candidate_elements",tuple(self.candidate_elements))
        object.__setattr__(self,"tool_paths",tuple((name,Path(path).absolute()) for name,path in self.tool_paths))
        if self.checkpoint is not None:object.__setattr__(self,"checkpoint",Path(self.checkpoint).absolute())
        if not isinstance(self.route,InitialRouteConfig) or not isinstance(self.cycle,CyclePolicy):
            raise ValueError("typed initial route and cycle policy required")
        if self.heavy_preparation not in ("current_loop","upstream_heavy_v1"):
            raise ValueError("unknown heavy preparation route")
        if self.heavy_preparation=="upstream_heavy_v1" and (self.route.strategy.value!="shelxt" or self.metal_only):
            raise ValueError("upstream heavy stage requires a complete initial SHELXT route without metal pruning")
        if type(self.metal_only) is not bool or type(self.strict_native) is not bool or self.device not in ("auto","cpu","mps"):
            raise ValueError("invalid anchor, native or inference-device option")
        if len(self.candidate_elements) != len(set(self.candidate_elements)) or any(e not in ATOMIC_NUMBERS for e in self.candidate_elements):
            raise ValueError("candidate elements must be unique declared chemical elements")
        if self.candidate_element_source not in ("original_ins","initial_seed"):
            raise ValueError("candidate element source must be original_ins or initial_seed")
        if not set(self.anchor_elements) <= METALS:
            raise ValueError("anchor elements must be metals")
        if not isinstance(self.label,str) or len(self.label)>128 or any(ord(c)<32 for c in self.label):
            raise ValueError("invalid run label")

    def to_record(self):
        return to_jsonable(self)

    @classmethod
    def from_record(cls,value):
        if not isinstance(value,dict) or set(value)-set(cls.__dataclass_fields__) or not {"ins_path","hkl_path"} <= set(value):
            raise ValueError("unknown or missing reconstruction request fields; references are not solve inputs")
        obj = dict(value)
        if "route" in obj: obj["route"] = InitialRouteConfig.from_dict(obj["route"])
        if "cycle" in obj: obj["cycle"] = CyclePolicy.from_dict(obj["cycle"])
        return cls(**obj)


@dataclass(frozen=True)
class ReconstructionHandle:
    run_id: str
    run_root: Path


class ReconstructionBackendService:
    """One worker at a time; frontend status survives service restarts as private JSON."""

    def __init__(self,repository:Path,runs_root:Path,*,toolchain_factory=None,assigner_factory=None,
                 protein_converter=None,clock=time.monotonic):
        self.repository = Path(repository).resolve(strict=True)
        self.runs_root = private_directory(self.repository,runs_root)
        self.toolchain_factory, self.assigner_factory = toolchain_factory,assigner_factory
        self.protein_converter, self.clock = protein_converter,clock
        self.active = {};self.lock=threading.Lock();self.worker_lock=threading.Lock()

    def _toolchain(self,request):
        if self.toolchain_factory:
            value = self.toolchain_factory(request)
        elif request.mode is RunMode.MOCK_DEMO:
            raise ReconstructionBlocked("synthetic_route_toolchain_must_be_explicitly_injected")
        else:
            try:
                from crystalx_native_tools.inspect import candidate_check_for
                check = candidate_check_for()
            except ImportError:
                check = None
            value = RealToolchain.for_config(RunConfig(toolbox_dir=self.repository/"toolbox",
                strict_native=request.strict_native,tool_paths=request.tool_paths),candidate_check=check)
        if value.mode != request.mode:
            raise ReconstructionBlocked("toolchain_mode_differs_from_request")
        return value

    def _assigner(self,request):
        if self.assigner_factory:
            return self.assigner_factory(request)
        if request.mode is RunMode.MOCK_DEMO:
            raise ReconstructionBlocked("synthetic_inference_must_be_explicitly_injected")
        from crystalx_core.app.reconstruction_runtime_inference import RuntimeElementAssigner, OFFICIAL_HEAVY_SHA256
        from crystalx_core.domain.inference import InferenceDevice
        checkpoint = request.checkpoint or self.repository/"crystalx_engine/inference_repo/weights/crystalx-heavy.pth"
        return RuntimeElementAssigner(self.repository/"crystalx_engine/inference_repo",checkpoint,
            expected_checkpoint_sha256=request.checkpoint_sha256 or OFFICIAL_HEAVY_SHA256,
            device=InferenceDevice(request.device),allow_anisotropic_metals=request.cycle.anisotropic_metals)

    def preview(self,request:ReconstructionRequest,*,_toolchain=None,_device_decision=None) -> dict:
        if request.mode is RunMode.MOCK_DEMO and not is_demo_input(request.ins_path,request.hkl_path):
            raise ReconstructionBlocked("synthetic_providers_cannot_be_used_on_measured_inputs")
        device = _device_decision if _device_decision is not None else resolve_inference_device(
            request.device, synthetic=request.mode is RunMode.MOCK_DEMO)
        plan=plan_initial_route(request.route,ins_path=request.ins_path,hkl_path=request.hkl_path,
                                work_dir=self.runs_root/"preview-initial-route")
        toolchain=_toolchain or self._toolchain(request)
        requirements=tuple(dict.fromkeys((*[step.tool for step in plan.steps],ToolKind.SHELXL)))
        tools=[]; blockers=list(plan.blockers)
        if device["device_blocker"]:blockers.append(device["device_blocker"])
        for kind in requirements:
            resolution=toolchain.resolution(kind)
            usable=bool(resolution and resolution.is_usable)
            tools.append({"tool":kind.value,"usable_static":usable,
                "diagnostic":resolution.diagnostic.kind.value if resolution else "not_found",
                "message":resolution.diagnostic.message if resolution else "tool not found",
                "path":str(resolution.identity.path) if resolution and resolution.identity else None})
            if not usable:blockers.append("tool_unavailable:"+kind.value)
        if not self.assigner_factory:
            checkpoint=request.checkpoint or self.repository/"crystalx_engine/inference_repo/weights/crystalx-heavy.pth"
            if not checkpoint.is_file():blockers.append("pretrained_checkpoint_missing")
        terminal=stop_record("preview_blocked",detail="; ".join(blockers)) if blockers else {
            "phase":"ready",**empty_stop_record()}
        return {"schema":"crystalx.reconstruction.preview/1","frontend_contract":FRONTEND_CONTRACT,
            "route":plan.to_dict(),"tools":tools,**terminal,**device,
            "blockers":blockers,"ready_static":not blockers,"cycle":request.cycle.to_dict(),
            "metal_only":request.metal_only,"checkcif":"SEPARATE_NOT_EVALUATED",
            "heavy_preparation":request.heavy_preparation,
            "expert_reference_used":False,"scientifically_validated":False}

    def _new(self,request):
        if request.mode is RunMode.MOCK_DEMO and not is_demo_input(request.ins_path,request.hkl_path):
            raise ReconstructionBlocked("synthetic_providers_cannot_be_used_on_measured_inputs")
        workspace=RunWorkspace.create(self.runs_root,label=request.label,
            metadata={"mode":request.mode.value,"workflow":"OnionWorkflow","scientific_checks":"deferred_by_owner_20261110"})
        device=resolve_inference_device(request.device,synthetic=request.mode is RunMode.MOCK_DEMO)
        workspace.write_json("inference-device.json",device)
        # The persisted request is directly replayable and pins an explicit
        # device. The original selector is preserved in inference-device.json.
        effective=replace(request,device=device["device_used"]) if device["device_used"] else request
        workspace.write_json("request.json",effective.to_record())
        self._state(workspace,"queued",None)
        return workspace

    def _state(self,workspace,phase,state,**extra):
        device=json.loads(workspace.layout.root.joinpath("inference-device.json").read_text())
        body={"schema":"crystalx.reconstruction.state/1","frontend_contract":FRONTEND_CONTRACT,**device,
            "run_id":str(workspace.run_id),"phase":phase,
            "owner_pid":os.getpid(),
            "history":state.history if state else [],"round":state.attempts if state else 0,
            "inference_calls":state.inference_calls if state else 0,"tool_calls":state.tool_calls if state else 0,
            "stable_transitions":state.stable_count if state else 0,**empty_stop_record(),
            "rounds_without_accepted_sites":state.rounds_without_accepted_sites if state else 0,
            "distinct_rejections_total":state.failures if state else 0,
            "base_rejection_count":state.base_rejection_count if state else 0,
            "repeated_rejected_actions":state.repeated_rejected_actions if state else 0,
            "skipped_rejected_actions":state.skipped_rejected_actions if state else 0,
            "skipped_rejected_proposals":state.skipped_rejected_proposals if state else 0,
            "scientifically_validated":False,"checkcif_status":"NOT_EVALUATED",**extra}
        workspace.write_json("state.json",body,allow_replace=True)
        return body

    def run(self,request:ReconstructionRequest,*,cancel:CancelToken|None=None) -> dict:
        workspace=self._new(request)
        with self.worker_lock:
            return self._run(workspace,request,cancel or CancelToken())

    def run_from_generated_initial(self,request:ReconstructionRequest,*,source_run_root:Path,
                                   expected_initial_result_sha256:str,cancel:CancelToken|None=None) -> dict:
        from crystalx_core.app.generated_initial import verify_generated_initial
        initial,receipt=verify_generated_initial(source_run_root,private_root=self.runs_root,request=request,
            expected_result_sha256=expected_initial_result_sha256)
        workspace=self._new(request)
        with self.worker_lock:
            return self._run(workspace,request,cancel or CancelToken(),_generated_initial=(initial,receipt))

    def submit(self,request:ReconstructionRequest) -> ReconstructionHandle:
        workspace=self._new(request);cancel=CancelToken()
        handle=ReconstructionHandle(str(workspace.run_id),workspace.layout.root)
        def work():
            try:
                with self.worker_lock:self._run(workspace,request,cancel)
            finally:
                with self.lock:self.active.pop(handle.run_id,None)
        thread=threading.Thread(target=work,name="crystalx-reconstruction-"+handle.run_id,daemon=True)
        with self.lock:self.active[handle.run_id]=(thread,cancel)
        thread.start();return handle

    def _path(self,run_id):
        if not isinstance(run_id,str) or re.fullmatch(r"r-\d{8}T\d{6}Z-[a-f0-9]{8}",run_id) is None:
            raise ValueError("invalid reconstruction run id")
        matches=[entry for entry in self.runs_root.iterdir() if (entry.name==run_id or entry.name.startswith(run_id+"_"))
                 and not entry.is_symlink() and entry.is_dir()]
        if len(matches)!=1:raise ValueError("unknown or ambiguous reconstruction run id")
        path=matches[0]
        if path.is_symlink() or path.resolve().parent!=self.runs_root.resolve():raise ValueError("unsafe run path")
        return path

    def run_root(self, run_id: str) -> Path:
        """Return the validated local run directory; arbitrary paths are never accepted."""
        return self._path(run_id)

    def status(self,run_id):
        state=json.loads((self._path(run_id)/"state.json").read_text())
        if state.get("phase") in {"running","queued"}:
            pid=state.get("owner_pid")
            try:
                if type(pid) is not int or pid<1:raise ProcessLookupError
                os.kill(pid,0)
            except ProcessLookupError:
                state.update(stop_record("owner_process_interrupted"),
                    message="owner process is no longer running; no automatic replay")
            except PermissionError:
                pass
        return self._response_contract(state)
    def summary(self,run_id):
        return self._response_contract(json.loads((self._path(run_id)/"summary.json").read_text()))

    @staticmethod
    def _response_contract(record):
        """Enrich responses without rewriting immutable historical run evidence."""
        # Keep a known historical marker. Missing markers describe this enriched
        # response, and do not rewrite or re-date the persisted source evidence.
        record={"frontend_contract":FRONTEND_CONTRACT,**empty_stop_record(),
            "device_requested":None,"device_used":None,"device_selection_reason":None,
            "device_blocker":None,"operator_cpu_fallback_allowed":False,**record}
        if record.get("phase") not in {"queued","running","ready"}:
            fallback={"blocked":"reconstruction_blocked","cancelled":"cancelled",
                "interrupted":"owner_process_interrupted"}.get(record.get("phase"),"backend_error")
            record.update(stop_record(record.get("stop_code") or fallback,detail=record.get("stop_detail"),
                fallback_code=fallback))
        return record
    def cancel(self,run_id):
        with self.lock:
            job=self.active.get(run_id)
            if job:job[1].cancel("cancelled by frontend")
        return bool(job)

    def _run(self,workspace,request,cancel,*,_generated_initial=None):
        started=self.clock();root=workspace.layout.root;cycle_state=None
        device=json.loads((root/"inference-device.json").read_text())
        if _generated_initial is not None:started-=_generated_initial[0].duration_s
        summary={"schema":"crystalx.reconstruction.summary/1","frontend_contract":FRONTEND_CONTRACT,**device,
            "run_id":str(workspace.run_id),**empty_stop_record(),
            "run_root":str(root),"phase":"running","mode":request.mode.value,"scientifically_validated":False,
            "expert_reference_used":False,"ai_solved_res":None,"seal_sha256":None,"blockers":[],
            "checkcif_status":"NOT_EVALUATED","automatic_stop_is_scientific_convergence":False}
        self._state(workspace,"running",None)
        try:
            if device["device_blocker"]:raise ReconstructionBlocked(device["device_blocker"])
            request=replace(request,device=device["device_used"])
            def checkpoint():
                if cancel.cancelled:raise ReconstructionBlocked("cancelled")
                elapsed=self.clock()-started
                if elapsed<0 or elapsed>=request.cycle.max_wall_seconds:
                    raise ReconstructionBlocked("whole_chain_wall_budget_exhausted")
            toolchain=self._toolchain(request)
            preview=self.preview(request,_toolchain=toolchain,_device_decision=device);workspace.write_json("preview.json",preview)
            if preview["blockers"]:
                summary.update(stop_record("preview_blocked",detail="; ".join(preview["blockers"])),
                    blockers=preview["blockers"])
                return self._finish(workspace,summary,cycle_state)
            if cancel.cancelled:raise ReconstructionBlocked("cancelled")
            ins=workspace.stage_input("instrument_instructions",request.ins_path)
            hkl=workspace.stage_input("measured_reflections",request.hkl_path)
            if dict(preview["route"]["input_hashes"])!={"ins":ins.sha256,"hkl":hkl.sha256}:
                raise ReconstructionBlocked("inputs_changed_after_preview")
            route=plan_initial_route(request.route,ins_path=ins.staged_path,hkl_path=hkl.staged_path,
                                     work_dir=root/"initial")
            if _generated_initial is None:
                initial=execute_initial_route(route,toolchain=toolchain,cancel=cancel,
                    limits=ResourceLimits(cpu_seconds=max(1,int(request.route.timeout_s*(os.cpu_count() or 1))),max_file_bytes=32*1024*1024,
                        max_open_files=256),protein_converter=self.protein_converter or convert_protein_trace,
                    deadline=started+request.cycle.max_wall_seconds,clock=self.clock)
            else:
                initial,receipt=_generated_initial
                generated=workspace.stage_input("automatic_initial_model",initial.seed_path)
                paired=workspace.stage_input("initial_matching_reflections",initial.paired_hkl_path)
                if generated.sha256!=initial.seed_sha256 or paired.sha256!=initial.paired_hkl_sha256:
                    raise ReconstructionBlocked("generated_initial_changed_during_staging")
                initial=replace(initial,seed_path=generated.staged_path,paired_hkl_path=paired.staged_path)
                workspace.write_json("initial-reuse.json",receipt)
                summary["initial_reuse"]=receipt
            summary["initial"]=initial.to_dict();workspace.write_json("initial-result.json",initial.to_dict())
            summary["tool_calls"]=initial.tool_calls
            if initial.status!="INITIAL_MODEL_READY":
                code=("cancelled" if cancel.cancelled or initial.status=="CANCELLED" else
                    "whole_chain_wall_budget_exhausted" if "initial_route_deadline_exhausted" in initial.blockers else
                    "initial_solution_blocked" if initial.status in ("BLOCKED","INITIAL_ARTIFACTS_READY") else
                    "initial_solution_failed")
                summary.update(stop_record(code,detail=initial.status+": "+"; ".join(initial.blockers)),
                    blockers=list(initial.blockers))
                return self._finish(workspace,summary,cycle_state)
            if request.heavy_preparation=="upstream_heavy_v1":
                checkpoint()
                stage_result,stage=self._heavy_stage(workspace,request,initial,toolchain,cancel,
                    deadline=started+request.cycle.max_wall_seconds,checkpoint=checkpoint)
                cycle_state=CycleState(stage_result.model_path,initial.paired_hkl_path,result=stage_result,
                    inference_calls=1,tool_calls=initial.tool_calls+1)
                summary.update(stop_record("heavy_stage_completed"),stages=[stage],history=[],
                    heavy_preparation=request.heavy_preparation,inference_calls=1,tool_calls=initial.tool_calls+1)
                if sha256_file(request.ins_path)!=ins.sha256 or sha256_file(request.hkl_path)!=hkl.sha256:
                    raise ReconstructionBlocked("original_input_changed_during_run")
                self._seal_candidate(workspace,request,summary,stage_result)
                return self._finish(workspace,summary,cycle_state)
            model_path=root/"seed.res"
            promotion=promote_initial_peaks(initial.seed_path,model_path,source_program=request.route.strategy.value)
            workspace.write_json("seed-promotion.json",promotion)
            parsed=checked_res(model_path)
            original_header=read_ins_header(ins.staged_path)
            if original_header.sha256!=ins.sha256:
                raise ReconstructionBlocked("original_instruction_staged_hash_differs_from_upload")
            initial_labels={a.label:parsed.element_of(a) for a in parsed.atoms}
            candidates=request.candidate_elements or tuple(original_header.sfac if request.candidate_element_source=="original_ins" else parsed.header.sfac)
            workspace.write_json("candidate-element-source.json",{"effective_source":"explicit_request" if request.candidate_elements else request.candidate_element_source,
                "elements":candidates,"original_ins_sha256":ins.sha256,"original_sfac":original_header.sfac,
                "initial_seed_sfac":parsed.header.sfac,"expert_used":False})
            if len([e for e in candidates if e!="H"])<2:
                raise ReconstructionBlocked("at_least_two_declared_nonhydrogen_candidate_elements_required")
            assigner=self._assigner(request)
            if sha256_file(ins.staged_path)!=ins.sha256:
                raise ReconstructionBlocked("original_instruction_staged_hash_differs_from_upload")
            checkpoint()
            extra_inferences=0
            locked=()
            if request.metal_only:
                if not any(parsed.element_of(a) in METALS for a in parsed.atoms):
                    checkpoint();extra_inferences=1
                    summary["inference_calls"]=extra_inferences
                    workspace.write_json("metal-seed-inference-reservation.json",{"inference_calls":extra_inferences,
                        "source_model_sha256":sha256_file(model_path)})
                    metal_input_sha=sha256_file(model_path)
                    protocol_kwargs={"input_protocol":request.cycle.inference_protocol,"original_ins_path":ins.staged_path,
                        "initial_labels":initial_labels,"protocol_seed":request.cycle.inference_seed} if request.cycle.inference_protocol!="current_labels" else {}
                    assigned=assigner.assign(model_path,candidate_elements=candidates,include_q_peaks=False,**protocol_kwargs)
                    checkpoint()
                    if sha256_file(ins.staged_path)!=ins.sha256:
                        raise ReconstructionBlocked("original_instruction_staged_hash_differs_from_upload")
                    if (assigned.model_sha256!=metal_input_sha or sha256_file(model_path)!=metal_input_sha
                            or assigned.result.input_sha256!=digest(assigned.request)):
                        raise ReconstructionBlocked("metal_seed_inference_identity_mismatch")
                    workspace.write_json("metal-seed-inference.json",assigned.to_record())
                    assigned_path=root/"metal-assigned-seed.res"
                    assign_to_model(parsed,assigned.result.predictions,assigned_path,candidate_elements=candidates,
                        element_overrides=getattr(assigned,"provenance",{}).get("protocol_element_choices"))
                    parsed=checked_res(assigned_path)
                model_path=root/"metal-only-seed.res"
                locked=heavy_only_seed(parsed,model_path,request.anchor_elements)
                workspace.write_json("metal-anchor.json",{"retained_sites":locked,"source_model_sha256":parsed.sha256,
                    "removed_nonmetal_atoms":len(parsed.atoms)-len(locked),"validated_scaffold":False,
                    "phase_policy":"partial-model phases recomputed each round; not measured-phase constraints"})
            cycle_state=CycleState(model_path,initial.paired_hkl_path,locked_sites=locked,
                inference_calls=extra_inferences,tool_calls=initial.tool_calls,
                adaptive_state={"retired_site_ids":tuple(sorted(set(initial_labels)-set(locked)))} if request.metal_only else {})
            adapter=toolchain.adapter(ToolKind.SHELXL)
            if adapter is None:raise ReconstructionBlocked("shelxl_became_unavailable")
            from crystalx_core.tools.adapter import ExternalToolAdapter
            adapter=ExternalToolAdapter(adapter.resolution,runner=adapter.runner,strict_regular_files=True)
            executor=RefinementCycleExecutor(cycle_state,assigner=assigner,adapter=adapter,policy=request.cycle,
                candidate_elements=candidates,cancel=cancel,synthetic=request.mode is RunMode.MOCK_DEMO,
                clock=self.clock,started=started,original_ins_path=ins.staged_path,initial_labels=initial_labels,
                original_ins_sha256=ins.sha256)
            original_commit=executor.commit_stage
            def commit(*args):
                original_commit(*args)
                self._state(workspace,"running",cycle_state)
            executor.commit_stage=commit
            header=read_ins_header(ins.staged_path)
            diagnosis=DataDiagnosis(ins=header,reflections=scan_hkl(hkl.staged_path,cell=header.cell,hklf_hint=header.hklf_format))
            waived={kind.value:"Owner 2026-11-10: develop numerical reconstruction first; separate checkCIF/review afterwards"
                    for kind in (GateKind.G2_REFINEMENT_CONVERGED,GateKind.G4_CHEMICAL_VALIDITY,
                                 GateKind.G5_COORDINATION_PLAUSIBLE,GateKind.G6_DATA_PARAMETER_RATIO,
                                 GateKind.G7_NO_UNRESOLVED_SEVERE_ALERT)}
            peak_budget=max(request.cycle.max_new_peaks_per_round,request.cycle.adaptive.early_peak_limit) if request.cycle.adaptive.growth_filter_enabled else request.cycle.max_new_peaks_per_round
            budget=StageBudget(max_attempts=1,max_candidates=1,max_new_atoms=peak_budget,
                max_tool_calls=1,max_wall_clock_s=request.cycle.max_wall_seconds,max_failures=1)
            workflow=OnionWorkflow(workspace,{kind:executor for kind in (StageKind.INITIAL_SOLUTION,StageKind.ANCHOR,
                StageKind.FIRST_SHELL,StageKind.SHELL_EXPANSION,StageKind.FINALIZE)},
                policy=OnionPolicy(max_shells=request.cycle.max_rounds+1,require_anchor_before_expansion=False,
                    ranking_policy=RankingPolicy(waived_gates=waived),stage_budgets={kind.value:budget for kind in StageKind}),
                clock=self.clock,cancel_token=cancel)
            result=workflow.run(diagnosis=diagnosis,prior=ChemicalPrior(),
                radiation=RadiationAssumption.from_wavelength(diagnosis.wavelength_a))
            workspace.write_json("workflow-result.json",result)
            reason=result.stop_reason
            if cancel.cancelled or reason.kind.value=="user_aborted":
                terminal=stop_record("cancelled",detail=reason.message or None)
            elif cycle_state.stop_code:
                terminal=stop_record(cycle_state.stop_code)
            else:
                code={"budget_exceeded":"workflow_budget_exhausted","internal_error":"backend_error"}.get(
                    reason.kind.value,"workflow_stopped")
                if reason.message in STOP_CODES and STOP_CODES[reason.message].category=="budget_exhausted":
                    code=reason.message
                terminal=stop_record(code,detail=reason.kind.value+": "+reason.message)
            phase=terminal["phase"]
            summary.update(terminal,
                history=cycle_state.history,inference_calls=cycle_state.inference_calls,tool_calls=cycle_state.tool_calls,
                stable_transitions=cycle_state.stable_count,scientific_gate_waivers=waived,
                distinct_rejections_total=cycle_state.failures,base_rejection_count=cycle_state.base_rejection_count,
                repeated_rejected_actions=cycle_state.repeated_rejected_actions,
                skipped_rejected_actions=cycle_state.skipped_rejected_actions,
                skipped_rejected_proposals=cycle_state.skipped_rejected_proposals)
            if cycle_state.result is not None:
                if cycle_state.result.method == "cgls" or cycle_state.result.fourier_map!=2:
                    reason = ("cgls_native_cif_requires_recorded_nonzero_ls_refinement" if cycle_state.result.method=="cgls" else
                        "non_difference_map_native_cif_requires_recorded_fmap2_refinement")
                    workspace.write_json("archive-blocked.json", {"reason":reason,
                        "model_preserved":True,"new_native_calls":0,
                        "native_limits":["CGLS incompatible with ACTA", "FMAP 3/4 incompatible with ACTA", "L.S. 0 incompatible with ACTA",
                                         "Second DAMP parameter too small for ACTA"],
                        "available_roles":["model","fcf"],"cif_status":"NOT_EVALUATED",
                        "fourier_map":cycle_state.result.fourier_map,
                        "next_step":"Any new LS/FMAP2 refinement requires explicit budget and acceptance checks"})
                    cycle_state.result = replace(cycle_state.result,cif_path=None)
                    summary["blockers"].append(reason)
                    if not cancel.cancelled:
                        prior_stop = summary["stop_code"]
                        summary.update(stop_record("reconstruction_blocked",
                            detail=reason+"; retained_numerical_stop="+str(prior_stop)))
                self._seal_candidate(workspace,request,summary,cycle_state.result)
            if sha256_file(request.ins_path)!=ins.sha256 or sha256_file(request.hkl_path)!=hkl.sha256:
                raise ReconstructionBlocked("original_input_changed_during_run")
            return self._finish(workspace,summary,cycle_state)
        except Exception as error:
            terminal=(stop_record("cancelled",detail=str(error)) if cancel.cancelled else
                stop_record(str(error)) if isinstance(error,ReconstructionBlocked) else
                stop_record("backend_error",detail=str(error)))
            summary.update(terminal,error_type=type(error).__name__)
            return self._finish(workspace,summary,cycle_state)

    def _heavy_stage(self,workspace,request,initial,toolchain,cancel,*,deadline,checkpoint):
        from crystalx_core.app.upstream_heavy import write_heavy_stage_model
        from crystalx_core.refinement.cycle import refine_file
        from crystalx_core.tools.adapter import ExternalToolAdapter
        stage_root=workspace.layout.root/"heavy-stage";stage_root.mkdir()
        assigner=self._assigner(request)
        if not callable(getattr(assigner,"assign_upstream_heavy",None)):
            raise ReconstructionBlocked("upstream_heavy_preparer_not_supported_by_injected_assigner")
        checkpoint()
        assignment=assigner.assign_upstream_heavy(initial.seed_path)
        checkpoint()
        workspace.write_json("heavy-stage/inference.json",assignment.to_record())
        assigned_path=stage_root/"assigned.res"
        changes=write_heavy_stage_model(initial.seed_path,assigned_path,assignment)
        workspace.write_json("heavy-stage/preparation.json",changes)
        adapter=toolchain.adapter(ToolKind.SHELXL)
        if adapter is None:raise ReconstructionBlocked("shelxl_became_unavailable")
        adapter=ExternalToolAdapter(adapter.resolution,runner=adapter.runner,strict_regular_files=True)
        checkpoint()
        result=refine_file(model_path=assigned_path,hkl_path=initial.paired_hkl_path,
            directory=stage_root/"anisotropic-ls16",adapter=adapter,cycles=16,
            timeout_s=min(request.cycle.trial_timeout_s,deadline-self.clock()),cancel=cancel,isotropic=False)
        checkpoint()
        stage={"stage":"upstream_heavy_v1","inference_calls":1,"shelxl_cycles":16,
            "refinement_method":"ls","anisotropic":True,"r1":result.r1,"wr2":result.wr2,
            "goof":result.goof,"parameters":result.n_parameters,
            "preparation_sha256":sha256_file(stage_root/"preparation.json"),
            "inference_sha256":sha256_file(stage_root/"inference.json"),
            "upstream_prediction_parity_verified":False,"hydrogen_stage_performed":False,
            "scientifically_validated":False}
        workspace.write_json("heavy-stage/result.json",stage)
        return result,stage

    def _seal_candidate(self,workspace,request,summary,result):
        root=workspace.layout.root;outputs={};exports=root/"outputs";exports.mkdir()
        for role,path in (("model",result.model_path),("cif",result.cif_path),("fcf",result.fcf_path)):
            if path:
                name="ai_solved.res" if role=="model" else "ai_solved."+role
                expected=dict(result.artifact_sha256s)[path.name]
                data=pinned_bytes(path,expected,maximum=128*1024*1024)
                target=exports/name;target.write_bytes(data);outputs[role]=str(target.relative_to(root))
        seal=seal_run(root,private_root=self.runs_root,outputs=outputs,track="de_novo",case_id=request.label,
            terminal_state=summary["phase"],completion={"ai_reference_used":False,"scientifically_validated":False,
                "pretrained_inference_calls":summary.get("inference_calls",0),"stop_code":summary["stop_code"]})
        summary.update(ai_solved_res=str(exports/"ai_solved.res"),seal_sha256=seal.seal_sha256,
            candidate_sha256=seal.file_for_role("model").sha256,candidate_available=True)
        receipt=self.runs_root/(str(workspace.run_id)+".seal-pin.json")
        receipt.write_text(json.dumps({"run_root":str(root),"seal_sha256":seal.seal_sha256})+"\n")

    def _finish(self,workspace,summary,state):
        summary=self._response_contract(summary)
        extra={"tool_calls":summary.get("tool_calls",0),"inference_calls":summary.get("inference_calls",0)} if state is None else {}
        # A poller may request the summary as soon as it sees a terminal state.
        # Publish the complete summary before making that state observable.
        workspace.write_json("summary.json",summary,allow_replace=True)
        self._state(workspace,summary["phase"],state,message=summary.get("stop_code"),
            blockers=summary.get("blockers",[]),stages=summary.get("stages",[]),
            **{key:summary[key] for key in empty_stop_record()},**extra)
        return to_jsonable(summary)


def convert_protein_trace(request) -> Path:
    """Explicit same-cell/space-group PDB trace conversion; no substructure substitution."""
    import gemmi
    structure=gemmi.read_structure(str(request.pdb_path))
    header=read_ins_header(request.original_ins_path)
    if header.cell is None or not structure.cell.is_crystal():
        raise ReconstructionBlocked("protein_trace_and_instrument_cells_required")
    values=(header.cell.a,header.cell.b,header.cell.c,header.cell.alpha,header.cell.beta,header.cell.gamma)
    if any(abs(a-b)>max(1e-3,abs(a)*1e-5) for a,b in zip(values,structure.cell.parameters)):
        raise ReconstructionBlocked("protein_trace_cell_differs_from_native_reflections")
    expected=gemmi.find_spacegroup_by_name(request.declared_space_group)
    actual=gemmi.find_spacegroup_by_name(structure.spacegroup_hm)
    if expected is None or actual is None or expected.number!=actual.number:
        raise ReconstructionBlocked("protein_trace_space_group_or_hand_requires_explicit_reindexing")
    if len(structure)!=1:raise ReconstructionBlocked("protein_trace_requires_one_coordinate_model")
    sfac=list(header.sfac);sites=[]
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                if atom.altloc not in ("\0"," ","") or atom.element.name in ("X","H"):
                    continue
                if atom.element.name not in sfac:sfac.append(atom.element.name)
                xyz=structure.cell.fractionalize(atom.pos)
                sites.append((atom.element.name,xyz))
    if not sites or len(sites)>999:raise ReconstructionBlocked("protein_trace_size_requires_residue_aware_adapter")
    records=[line.text for line in logical_lines(request.original_ins_path)
             if line.instruction in {"TITL","CELL","ZERR","LATT","SYMM"}]
    units=list(header.unit or ())
    if len(units)!=len(header.sfac):units=[0]*len(header.sfac)
    units.extend([0]*(len(sfac)-len(units)))
    records.extend(["SFAC "+" ".join(sfac),"UNIT "+" ".join(str(n) for n in units),"FVAR 1"])
    for i,(element,xyz) in enumerate(sites,1):
        records.append(f"P{i:03d} {sfac.index(element)+1} {xyz.x:.8f} {xyz.y:.8f} {xyz.z:.8f} 11 0.05")
    records.extend(["HKLF 4","END"])
    output=request.work_dir/"trace-converted.res";output.write_text("\n".join(records)+"\n")
    checked_res(output);return output
