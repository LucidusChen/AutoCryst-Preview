"""Offline preparation of bounded, private acceptance plans.

No proprietary process can be started through this module. Plans describe the existing
RunService route, with its actual resolved executables, rather than a second solve engine.
The absence of a launch command is intentional: reviewed instruction preparation, validation
resource binding and an enforceable memory policy are not delivered execution guarantees.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import uuid
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from crystalx_core.app.models import RunConfig
from crystalx_core.app.service import RunService
from crystalx_core.app.staging import safe_stem
from crystalx_core.app.toolchain import RealToolchain, Toolchain
from crystalx_core.domain.tools import ToolKind
from crystalx_core.errors import CrystalXError
from crystalx_core.io.shelx_reader import read_ins_header
from crystalx_core.private_acceptance.preflight import PreflightPolicy, PreflightReport, evaluate_toolchain
from crystalx_core.private_acceptance.sealing import write_private_json
from crystalx_core.private_acceptance.specs import AcceptanceTrack, SolverSpec, contained_path, load_solver_spec
from crystalx_core.workspace.hashing import sha256_file

if TYPE_CHECKING:
    from crystalx_core.workflow.reconstruction import ReconstructionPolicy

__all__ = ["PlanError", "private_directory", "private_output_path", "build_plan", "save_plan", "prepare_plan"]


class PlanError(ValueError):
    """A local plan could not be prepared without weakening its input boundary."""


def private_output_path(repository: Path, path: Path) -> Path:
    """Check the exact generated filename; a directory-level probe cannot cover exceptions."""
    repo = repository.resolve(strict=True)
    target = Path(os.path.abspath(path))
    try:
        target.relative_to(repo / "runs")
    except ValueError as exc:
        raise PlanError("private output must remain below ignored runs") from exc
    for parent in (target, *target.parents):
        if parent == repo:
            break
        if parent.is_symlink():
            raise PlanError("private output cannot use symlink paths")
    result = subprocess.run(["git", "check-ignore", "--quiet", "--no-index", "--", str(target)],
                            cwd=repo, capture_output=True, check=False)
    tracked = subprocess.run(["git", "ls-files", "-z", "--", str(target)],
                             cwd=repo, capture_output=True, check=False)
    if result.returncode or tracked.returncode or tracked.stdout:
        raise PlanError("the exact private output filename is unignored or tracked")
    return target


def private_directory(repository: Path | str, directory: Path | str) -> Path:
    """Require an ignored directory below this checkout's runs/, with no symlink parents.

    This protects the normal repository workflow; it cannot stop force-add, filesystem reads
    by another same-user process, copying files elsewhere, or changes to ignore rules later.
    """
    repo = Path(repository).resolve(strict=True)
    raw = Path(os.path.abspath(directory))
    try:
        relative = raw.relative_to(repo / "runs")
    except ValueError as exc:
        raise PlanError("private output must be below this checkout's ignored runs directory") from exc
    if not relative.parts:
        raise PlanError("choose a dedicated subdirectory below runs")
    for parent in (raw, *raw.parents):
        if parent == repo:
            break
        if parent.is_symlink():
            raise PlanError("private output cannot use symlink directories")
    probe = raw / ".private-output-probe"
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", "--", str(probe)],
        cwd=repo, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise PlanError("private output is not covered by Git ignore rules")
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", str(raw)], cwd=repo,
        capture_output=True, check=False,
    )
    if tracked.returncode or tracked.stdout:
        raise PlanError("private output has tracked files or Git tracking could not be checked")
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    return raw.resolve(strict=True)


def _input_record(path: Path, spec: SolverSpec) -> dict[str, Any]:
    info = path.stat()
    return {"path": str(path), "relative_to_case": str(path.relative_to(spec.case_root)),
            "sha256": sha256_file(path), "bytes": info.st_size}


def _selected(toolchain: Toolchain, kind: ToolKind) -> dict[str, Any]:
    resolution = toolchain.platon() if kind is ToolKind.PLATON else toolchain.resolution(kind)
    identity = resolution.identity if resolution else None
    return {"requested_tool": kind.value,
            "selected_tool": resolution.kind.value if resolution else None,
            "path": str(identity.path) if identity else None,
            "sha256": identity.sha256 if identity else None,
            "architecture": identity.arch.value if identity else None,
            "source": identity.source.value if identity else None,
            "static_usable_in_core": bool(resolution and resolution.is_usable)}


def build_plan(spec: SolverSpec, *, output_root: Path, config: RunConfig | None = None,
               checkdef_path: Path | None = None, toolchain: Toolchain | None = None,
               repository: Path | None = None,
               reconstruction_policy: ReconstructionPolicy | None = None) -> dict[str, Any]:
    """Read only the solver contract and its allowlisted files; never load a grader.

    Dependency injection exists for synthetic unit tests. It cannot enable execution.
    All paths, priors, diagnostics and hashes returned here are private evidence.
    """
    output_root = private_directory(repository or Path.cwd(), output_root)
    config = config or RunConfig(shelxt_timeout_s=900, shelxl_timeout_s=600,
                                platon_timeout_s=600, shelxl_cycles=4, cpu_seconds=3600,
                                strict_native=True)
    if not config.strict_native:
        raise PlanError("this private acceptance milestone requires strict native mode")
    if (type(config.shelxl_cycles) is not int or not 1 <= config.shelxl_cycles <= 50
            or config.cpu_seconds is not None and (type(config.cpu_seconds) is not int or config.cpu_seconds <= 0)):
        raise PlanError("refinement cycles and explicit CPU seconds must be positive integers")
    for field in ("shelxt_timeout_s", "shelxl_timeout_s", "platon_timeout_s"):
        value = getattr(config, field)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise PlanError("program budgets must be finite positive seconds")
    if spec.track is AcceptanceTrack.INITIAL_MODEL_RECOVERY:
        return _build_recovery_plan(spec, output_root=output_root, config=config,
                                    checkdef_path=checkdef_path, toolchain=toolchain,
                                    policy=reconstruction_policy)
    request = spec.to_solve_request(config)
    issues = request.validate()
    if issues:
        raise PlanError("solver inputs fail request validation")
    chain = toolchain or RealToolchain.for_config(config)
    required = ((ToolKind.SHELXL,) if spec.track is AcceptanceTrack.COMPONENT_REGRESSION
                else (ToolKind.SHELXT, ToolKind.SHELXL))
    preflight = evaluate_toolchain(chain, policy=PreflightPolicy(
        required_tools=required, checkdef_path=checkdef_path,
        validation_resource_binding_verified=False,
    ))
    inputs = {"ins": _input_record(spec.ins_path, spec), "hkl": _input_record(spec.hkl_path, spec)}
    try:
        header = read_ins_header(spec.ins_path)
    except CrystalXError as exc:
        raise PlanError("instruction input could not be safely diagnosed") from exc
    source_elements = set(header.sfac)
    required_elements = set(spec.prior.elements) | set(spec.prior.metals)
    absent = sorted(required_elements - source_elements)
    composition = dict(spec.preparation.composition)
    effective_elements = set(composition) if composition else source_elements
    missing_effective = sorted(required_elements - effective_elements)
    blockers = list(spec.blockers) + list(preflight.blockers)
    if missing_effective:
        blockers.append("declared_prior_not_in_effective_sfac")
    if spec.preparation.policy != "preserve_input":
        blockers.append("composition_preparation_review_not_authenticated")
    # The shared runner deliberately has no memory limit: macOS RLIMIT_AS is not reliable.
    # Recording null is honest; this plan is not eligible for execution until a policy exists.
    blockers.extend(("memory_enforcement_policy_not_implemented",
                     "bounded_owner_launch_authority_not_attached",
                     "offline_milestone_no_launch_entrypoint"))
    stem = safe_stem(spec.ins_path.name)
    steps = []
    kinds = ([ToolKind.SHELXL, ToolKind.PLATON] if spec.track is AcceptanceTrack.COMPONENT_REGRESSION
             else [ToolKind.SHELXT, ToolKind.SHELXL, ToolKind.PLATON])
    for kind in kinds:
        selected = _selected(chain, kind)
        args = ["-u", stem + ".cif"] if kind is ToolKind.PLATON else [stem]
        timeout = getattr(config, kind.value + "_timeout_s")
        steps.append({**selected, "argv": [selected["path"], *args] if selected["path"] else None,
                      "argv_unresolved_reason": None if selected["path"] else "executable_not_resolved",
                      "max_invocations": 1, "wall_seconds": timeout,
                      "limits": {**config.limits_for(timeout).to_record(),
                                 "memory_bytes": None, "memory_enforced": False},
                      "shelxl_cycles": config.shelxl_cycles if kind is ToolKind.SHELXL else None,
                      "working_directory": {
                          ToolKind.SHELXT: "<new-run>/stages/<selected-stage>/branches/<selected-branch>",
                          ToolKind.SHELXL: "<new-run>/refinement/shelxl-01",
                          ToolKind.PLATON: "<new-run>/validation/platon-01",
                      }[kind],
                      "prerequisite": "successful refinement plus both CIF and FCF" if kind is ToolKind.PLATON
                      else "selected model with verified reflection lineage" if kind is ToolKind.SHELXL
                      else "contained staged input pair and all execution gates"})
    # Shared preview may inspect allowlisted INS/HKL, but never answers or references.
    preview = None
    component_preview = None
    if spec.track is AcceptanceTrack.COMPONENT_REGRESSION:
        from crystalx_core.private_acceptance.component import preview_component_request
        try:
            component_preview = preview_component_request(request)
            if not component_preview["may_start"]:
                blockers.append("component_fixed_model_preconditions_blocked")
        except (OSError, ValueError, CrystalXError) as exc:
            blockers.append("component_preview_failed:" + type(exc).__name__)
    else:
        service = RunService(output_root / "service-preview", toolchain_factory=lambda _: chain)
        try:
            preview = service.preview(request)
            if not preview["preconditions"]["may_start"]:
                blockers.append("shared_service_input_preconditions_blocked")
            if not any(item["solver"] == "shelxt" and item["will_run"]
                       for item in preview["plan"]["nominated"]):
                blockers.append("shared_service_shelxt_route_not_runnable")
        except (OSError, ValueError, CrystalXError) as exc:
            blockers.append("shared_service_preview_failed:" + type(exc).__name__)
    after = {"ins": _input_record(spec.ins_path, spec), "hkl": _input_record(spec.hkl_path, spec)}
    if inputs != after:
        raise PlanError("an original input changed during planning")
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "case_id": spec.case_id, "track": spec.track.value, "status": "BLOCKED",
            "inputs": inputs, "case_root": str(spec.case_root), "config": config.to_dict(),
            "priors": [{"field": p.field, "value": p.value, "provenance": p.provenance,
                        "source": p.source} for p in spec.prior_records],
            "preparation": {"policy": spec.preparation.policy,
                            "composition": list(spec.preparation.composition),
                            "owner_review_reference": spec.preparation.owner_review_reference,
                            "official_semantics_reference": spec.preparation.official_semantics_reference,
                            "missing_prior_elements_in_original_sfac": absent,
                            "missing_prior_elements_in_effective_sfac": missing_effective,
                            "metal_and_element_flags_rewrite_sfac_unit": False,
                            "original_latt_symm_hklf_preserved": True},
            "steps": steps, "max_real_invocations": len(steps),
            "maximum_total_program_wall_seconds": sum(s["wall_seconds"] for s in steps),
            "output_root": str(output_root), "preflight": preflight.to_record(),
            "shared_service_preview": preview, "component_preview": component_preview,
            "blockers": sorted(set(blockers)),
            "stop_conditions": ["cancellation or wall/CPU/file limit", "changed input/executable/dependency/resource",
                                "missing required output or failed process evidence",
                                "ambiguous reflection lineage", "missing CIF or FCF prevents PLATON"],
            "launch_requirements": {"environment": "CRYSTALX_REAL_TOOLS=1",
                                    "authority": "explicit owner approval of this exact case, track and pinned plan hash",
                                    "implemented_launch_entrypoint": False},
            "status_layers": {"SOFTWARE_TESTED": "SEE_ACCEPTANCE_RECORD",
                              "REAL_PROCESS_VERIFIED": False, "SCIENTIFICALLY_REVIEWED": False},
            "real_invocations": 0,
            "limitations": ["No memory ceiling is enforced by the shared macOS runner.",
                            "No OS filesystem sandbox is provided; same-user programs can access other files.",
                            "Prior declaration provenance is supplied by the owner, not independently authenticated.",
                            "A plan is not a successful process run or scientific validation."]}


def _recovery_seed_record(spec: SolverSpec, hkl_sha256: str) -> dict[str, Any] | None:
    """Recheck only the explicitly declared seed; never discover substitutes or open a grader."""
    if spec.seed is None:
        return None
    try:
        relative = str(spec.seed.path.relative_to(spec.case_root))
        path = contained_path(spec.case_root, relative, must_exist=False)
        if not path.exists():
            return None
        if not path.is_file() or path in {spec.ins_path, spec.hkl_path}:
            raise PlanError("automatic seed is no longer a distinct regular model input")
        current = sha256_file(path)
        if current != spec.seed.sha256 or hkl_sha256 != spec.seed.compatible_hkl_sha256:
            raise PlanError("automatic seed or its paired reflection identity changed before planning")
        header = read_ins_header(path)
        if header.sha256 != current:
            raise PlanError("automatic seed changed while its source setting was read")
        cell = header.cell
        return {"role": spec.seed.role.value, "path": str(path), "relative_to_case": relative,
                "sha256": current, "compatible_hkl_sha256": spec.seed.compatible_hkl_sha256,
                "provenance_source": spec.seed.provenance_source,
                "owner_declaration": spec.seed.owner_declaration,
                "compatibility_scientifically_verified": False,
                "sfac": list(header.sfac),
                "source_setting": {"cell": [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma]
                                   if cell is not None else None,
                                   "latt": header.latt, "symm": list(header.symm),
                                   "hklf": int(header.hklf_format) if header.hklf_format is not None else None},
                "used_as_ground_truth": False}
    except (OSError, ValueError, RuntimeError, CrystalXError) as exc:
        raise PlanError("automatic seed identity/path could not be safely retained") from exc


def _build_recovery_plan(spec: SolverSpec, *, output_root: Path, config: RunConfig,
                         checkdef_path: Path | None, toolchain: Toolchain | None,
                         policy: ReconstructionPolicy | None) -> dict[str, Any]:
    """Provisional iterative plan, isolated from the one-shot RunService solve route."""
    from crystalx_core.workflow.reconstruction import ReconstructionPolicy, digest

    policy = ReconstructionPolicy() if policy is None else policy
    if not isinstance(policy, ReconstructionPolicy):
        raise PlanError("recovery planning requires the shared ReconstructionPolicy")
    inputs = {"ins": _input_record(spec.ins_path, spec), "hkl": _input_record(spec.hkl_path, spec)}
    seed = _recovery_seed_record(spec, inputs["hkl"]["sha256"])
    blockers = list(spec.blockers)
    if seed is None:
        blockers.append("automatic_initial_model_missing")
        preflight = PreflightReport("BLOCKED", (), ("automatic_initial_model_missing",),
                                    policy={"strict_native": True, "required_tools": [],
                                            "require_platon": False})
        selected: dict[ToolKind, dict[str, Any]] = {}
    else:
        chain = toolchain or RealToolchain.for_config(config)
        preflight = evaluate_toolchain(chain, policy=PreflightPolicy(
            required_tools=(ToolKind.SHELXL,), require_platon=True, checkdef_path=checkdef_path,
            validation_resource_binding_verified=False,
        ))
        selected = {kind: _selected(chain, kind) for kind in (ToolKind.SHELXL, ToolKind.PLATON)}
        blockers.extend(preflight.blockers)
    missing_seed_elements = (sorted((set(spec.prior.elements) | set(spec.prior.metals)) - set(seed["sfac"]))
                             if seed is not None else [])
    if missing_seed_elements:
        blockers.append("declared_prior_not_in_seed_sfac")
    if not policy.ai_required:
        blockers.append("initial_model_recovery_requires_trained_inference")
    blockers.extend(("official_trained_checkpoint_and_reference_inference_not_bound",
                     "audited_density_geometry_and_refinement_providers_not_bound",
                     "iterative_policy_requires_owner_review",
                     "iterative_platon_checkpoint_binding_not_implemented",
                     "memory_enforcement_policy_not_implemented",
                     "bounded_owner_launch_authority_not_attached",
                     "real_reconstruction_dispatch_disabled",
                     "offline_milestone_no_launch_entrypoint"))
    # The shared executor budgets baseline/trial refinement calls. This provisional plan also
    # reserves one call for the required PLATON checkpoint within the SAME aggregate tool cap.
    # Actual PLATON binding/dispatch remains absent and therefore blocks the whole real track.
    platon_calls = 1 if seed is not None else 0
    refinement_calls = min(1 + policy.max_rounds, max(0, policy.max_tool_calls - platon_calls)) if seed else 0
    if seed is not None and refinement_calls < 1:
        blockers.append("iterative_tool_budget_cannot_fit_baseline_and_required_checkpoint")
    steps: list[dict[str, Any]] = []
    if seed is not None:
        # ReconstructionExecutor writes each derivative instruction pair as trial.ins/trial.hkl.
        # The supplied model keeps its original filename and pinned role in the private record.
        stem = "trial"
        for kind, calls, wall, workdir in (
            (ToolKind.SHELXL, refinement_calls,
             min(config.shelxl_timeout_s, policy.trial_wall_seconds, policy.max_wall_seconds),
             "<new-run>/reconstruction/<baseline-or-trial>"),
            (ToolKind.PLATON, platon_calls, min(config.platon_timeout_s, policy.max_wall_seconds),
             "<new-run>/reconstruction/<review-checkpoint>/platon"),
        ):
            executable = selected[kind]
            args = ["-u", stem + ".cif"] if kind is ToolKind.PLATON else [stem]
            steps.append({**executable,
                          "step": "bounded_refinement" if kind is ToolKind.SHELXL else "platon_review_checkpoint",
                          "implementation_status": "PLAN_ONLY_REAL_DISPATCH_DISABLED",
                          "argv": [executable["path"], *args] if executable["path"] else None,
                          "argv_unresolved_reason": None if executable["path"] else "executable_not_resolved",
                          "max_invocations": calls, "wall_seconds": wall,
                          "working_directory": workdir,
                          "shelxl_cycles": policy.shelxl_cycles_per_trial if kind is ToolKind.SHELXL else None,
                          "limits": {**config.limits_for(wall).to_record(),
                                     "memory_bytes": None, "memory_enforced": False},
                          "prerequisite": "owner-declared compatible seed/HKL plus all iterative scientific providers"
                          if kind is ToolKind.SHELXL else "accepted provisional checkpoint and compatible CIF plus FCF"})
    workflow = {
        "controller": "OnionWorkflow", "application": "ReconstructionService",
        "policy": asdict(policy), "policy_sha256": digest(policy),
        "budgets_provisional": True, "owner_approved": False, "real_dispatch_enabled": False,
        "initial_solution_performed": False, "expert_component_regression": False,
        "baseline": ["replay_explicit_automatic_seed", "bounded_refinement", "initial_assessment"],
        "loop": ["density_observation", "trained_element_hypotheses", "single_auditable_mutation",
                 "bounded_refinement", "compatible_evidence_reassessment", "accept_reject_or_review"],
        "rejection_restores": ["accepted_parent_model", "branch_active_locks"],
        "final": ["platon_review_checkpoint", "seal_outputs", "separate_scientific_grading"],
        "total_program_call_cap": policy.max_tool_calls,
        "reserved_platon_calls": platon_calls, "refinement_call_cap": refinement_calls,
        "max_trial_proposals": min(policy.max_proposals, policy.max_rounds, max(0, refinement_calls - 1)),
        "workflow_rounds_are_not_shelxl_cycles": True,
    }
    after = {"ins": _input_record(spec.ins_path, spec), "hkl": _input_record(spec.hkl_path, spec)}
    if inputs != after or seed != _recovery_seed_record(spec, after["hkl"]["sha256"]):
        raise PlanError("a recovery input changed during planning")
    potential_program_wall = sum(step["wall_seconds"] * step["max_invocations"] for step in steps)
    return {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": spec.case_id, "track": spec.track.value, "status": "BLOCKED",
        "inputs": inputs, "case_root": str(spec.case_root), "config": config.to_dict(),
        "input_roles": [{"role": item.role.value, "path": str(item.path)} for item in spec.input_roles],
        "seed": seed, "shared_service_preview": None, "component_preview": None,
        "preparation": {"policy": spec.preparation.policy, "originals_unchanged": True,
                        "missing_prior_elements_in_seed_sfac": missing_seed_elements,
                        "metal_and_element_flags_rewrite_sfac_unit": False},
        "priors": [{"field": item.field, "value": item.value, "provenance": item.provenance,
                    "source": item.source} for item in spec.prior_records],
        "workflow": workflow, "steps": steps,
        "max_real_invocations": sum(step["max_invocations"] for step in steps),
        "maximum_total_program_wall_seconds": min(potential_program_wall, policy.max_wall_seconds),
        "output_root": str(output_root), "preflight": preflight.to_record(),
        "blockers": sorted(set(blockers)),
        "stop_conditions": ["cancellation or total wall/tool/inference/round/proposal/atom/failure budget",
                            "missing scientific provider or contradictory setting/hand/geometry evidence",
                            "oscillation, stagnation, unknown required checks or human review",
                            "changed seed/input/executable/dependency/resource",
                            "missing required output or process evidence", "missing CIF or FCF prevents PLATON"],
        "launch_requirements": {"environment": "CRYSTALX_REAL_TOOLS=1",
                                "authority": "explicit owner approval of exact case, seed/HKL pins, policy and plan hash",
                                "implemented_launch_entrypoint": False},
        "status_layers": {"SOFTWARE_TESTED": "SEE_ACCEPTANCE_RECORD",
                          "REAL_PROCESS_VERIFIED": False, "SCIENTIFICALLY_REVIEWED": False},
        "real_invocations": 0,
        "limitations": ["A supplied automatic model is a starting checkpoint, never final ground truth.",
                        "Seed/HKL compatibility is an owner declaration plus identity pins, not a scientific proof.",
                        "The default iteration limits are provisional software defaults requiring case-specific owner review.",
                        "PLATON integration and real inference/refinement dispatch are not implemented for this iterative track.",
                        "No operating-system filesystem sandbox or enforced macOS memory ceiling is supplied."]}


def save_plan(plan: dict[str, Any], *, repository: Path, private_root: Path) -> tuple[Path, str]:
    base = private_directory(repository, private_root)
    folder = base / ("plan-" + uuid.uuid4().hex)
    folder.mkdir(mode=0o700)
    private_output_path(repository, folder / "plan.json")
    private_output_path(repository, folder / "plan.sha256.json")
    write_private_json(folder, private_root=base, relative="plan.json", payload=plan)
    path = folder / "plan.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    write_private_json(folder, private_root=base, relative="plan.sha256.json", payload={"sha256": digest})
    return path, digest


def prepare_plan(spec_path: Path, *, case_root: Path, repository: Path, private_root: Path,
                 checkdef_path: Path | None = None, config: RunConfig | None = None,
                 reconstruction_policy: ReconstructionPolicy | None = None) -> tuple[Path, str]:
    base = private_directory(repository, private_root)
    spec = load_solver_spec(spec_path, case_root=case_root)
    plan = build_plan(spec, output_root=base, config=config, checkdef_path=checkdef_path,
                      repository=repository, reconstruction_policy=reconstruction_policy)
    return save_plan(plan, repository=repository, private_root=base)
