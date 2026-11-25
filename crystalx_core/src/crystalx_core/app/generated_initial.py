"""Reuse only a pinned, independently generated initial output from this backend.

This is not an arbitrary RES/answer input. It verifies source requests, tool invocation,
original inputs and generated output roles before returning an automatic model receipt.
"""

from pathlib import Path
import json
import math

from crystalx_core._serde import from_jsonable, to_jsonable
from crystalx_core.app.solve_routes import InitialRouteResult, InitialStrategy
from crystalx_core.refinement.cycle import pinned_bytes
from crystalx_core.workflow.reconstruction import ReconstructionBlocked
from crystalx_core.workspace.hashing import sha256_file


def verify_generated_initial(source_root: Path, *, private_root: Path, request,
                             expected_result_sha256: str) -> tuple[InitialRouteResult,dict]:
    source=Path(source_root).absolute();private=Path(private_root).resolve(strict=True)
    if source.is_symlink() or source.resolve(strict=True)!=source or source.parent!=private:
        raise ReconstructionBlocked("generated_initial_source_must_be_a_direct_private_run")
    raw=pinned_bytes(source/"initial-result.json",expected_result_sha256,maximum=16*1024*1024)
    payload=json.loads(raw)
    fields={name:payload[name] for name in InitialRouteResult.__dataclass_fields__ if name in payload}
    result=from_jsonable(InitialRouteResult,fields)
    if to_jsonable(result.to_dict())!=payload or result.status!="INITIAL_MODEL_READY" or result.seed_format!="shelx":
        raise ReconstructionBlocked("source_is_not_an_explicit_successful_initial_model_record")
    if result.strategy is InitialStrategy.PROTEIN_CDE:
        raise ReconstructionBlocked("protein_trace_reuse_requires_a_separate_conversion_receipt")
    if result.strategy!=request.route.strategy or not math.isfinite(result.duration_s) or result.duration_s<0:
        raise ReconstructionBlocked("generated_initial_route_or_resource_identity_mismatch")
    source_request_path=source/"request.json"
    source_request_sha=sha256_file(source_request_path)
    source_request=json.loads(pinned_bytes(source_request_path,source_request_sha))
    if source_request.get("mode")!=request.mode.value or source_request.get("route")!=request.route.to_dict():
        raise ReconstructionBlocked("generated_initial_source_mode_or_route_changed")
    if any(name in source_request for name in ("expert_path","answer_path","reference_res")):
        raise ReconstructionBlocked("reference_seeded_source_is_not_an_independent_initial_result")
    plan_path=source/"initial/route-plan.json";plan_sha=sha256_file(plan_path)
    plan=json.loads(pinned_bytes(plan_path,plan_sha))
    expected_inputs={"ins":sha256_file(request.ins_path),"hkl":sha256_file(request.hkl_path)}
    if plan.get("input_hashes")!=expected_inputs:
        raise ReconstructionBlocked("generated_initial_original_input_pins_changed")
    for role in ("ins","hkl"):
        pinned_bytes(source/"initial"/("original."+role),expected_inputs[role],maximum=512*1024*1024)
    expected_tool="shelxt" if result.strategy is InitialStrategy.SHELXT else request.route.small_solver
    if len(result.invocations)!=1 or result.tool_calls!=1:
        raise ReconstructionBlocked("generated_initial_requires_one_complete_selected_initial_invocation")
    invocation=result.invocations[0]
    if (invocation.get("tool")!=expected_tool or invocation.get("succeeded") is not True
            or invocation.get("exit_code")!=0 or invocation.get("cancelled")
            or request.mode.value=="real" and invocation.get("executable_source")=="synthetic_demo"):
        raise ReconstructionBlocked("generated_initial_tool_evidence_is_not_successful_or_mode_consistent")
    root=source/"initial"
    if result.seed_path.parent!=root or result.paired_hkl_path.parent!=root:
        raise ReconstructionBlocked("generated_initial_output_roles_escape_source_stage")
    outputs={item["relative_path"]:item for item in invocation.get("outputs",[])}
    model_entry=outputs.get(result.seed_path.name,{})
    if model_entry.get("role")!="output" or model_entry.get("sha256")!=result.seed_sha256:
        raise ReconstructionBlocked("automatic_model_was_not_a_new_recorded_tool_output")
    if result.strategy is InitialStrategy.SHELXT:
        entry=outputs.get(result.paired_hkl_path.name,{})
        if entry.get("role")!="output" or entry.get("sha256")!=result.paired_hkl_sha256:
            raise ReconstructionBlocked("matching_reflections_were_not_a_new_recorded_tool_output")
    elif result.paired_hkl_sha256!=expected_inputs["hkl"]:
        raise ReconstructionBlocked("fixed_setting_solver_reflections_changed")
    pinned_bytes(result.seed_path,result.seed_sha256)
    pinned_bytes(result.paired_hkl_path,result.paired_hkl_sha256,maximum=512*1024*1024)
    if pinned_bytes(source/"initial-result.json",expected_result_sha256)!=raw:
        raise ReconstructionBlocked("generated_initial_record_changed_during_verification")
    return result,{"source_run_root":str(source),"initial_result_sha256":expected_result_sha256,
        "source_request_sha256":source_request_sha,"source_plan_sha256":plan_sha,
        "selected_source_tool_calls":result.tool_calls,"selected_source_duration_s":result.duration_s,
        "expert_reference_used":False,"new_initial_tool_invocations":0}
