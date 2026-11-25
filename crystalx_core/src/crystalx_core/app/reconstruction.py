"""Application entry for the existing onion controller's iterative vertical slice.

The current runnable mode uses explicit synthetic providers and the byte-identical built-in
synthetic reflection case. It cannot accidentally run mock providers on private measurements.
Real mode returns concrete blockers before loading a seed or calling an inference/tool provider.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from crystalx_core.app.demo import is_demo_input, demo_digests
from crystalx_core.domain.result import StopReasonKind
from crystalx_core.domain.chemistry import ChemicalPrior
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.inference import InferenceBackend
from crystalx_core.domain.onion import StageBudget, StageKind
from crystalx_core.domain.radiation import RadiationAssumption
from crystalx_core.io.shelx_reader import read_ins_header, scan_hkl
from crystalx_core.private_acceptance.harness import private_directory
from crystalx_core.private_acceptance.specs import AcceptanceTrack, SolverSpec
from crystalx_core.tools.runner import CancelToken
from crystalx_core.workflow.onion import LockLedger, OnionPolicy, OnionWorkflow
from crystalx_core.workflow.reconstruction import (
    DensityProvider, InferenceAudit, InferencePreparer, ReconstructionBlocked,
    ReconstructionExecutor, ReconstructionPolicy, ReconstructionState, RefinementProvider,
    Usage, digest, load_checkpoint, read_snapshot,
)
from crystalx_core.workspace.store import RunWorkspace
from crystalx_core.workspace.hashing import sha256_file


@dataclass(frozen=True)
class ReconstructionProviders:
    density: DensityProvider | None
    inference: InferenceBackend | None
    preparer: InferencePreparer | None
    inference_audit: InferenceAudit | None
    refinement: RefinementProvider | None


def real_prerequisites(providers: ReconstructionProviders) -> tuple[str, ...]:
    missing = list(providers.inference_audit.blockers(synthetic=False)
                   if providers.inference_audit else ("official_trained_checkpoint_provenance_missing",
                                                      "reference_inference_comparison_missing"))
    missing.extend(("actual_native_tools_and_validation_resources_not_verified",
                    "audited_density_and_geometry_providers_not_bound",
                    "memory_enforcement_policy_not_implemented",
                    "bounded_owner_exact_plan_authority_missing",
                    "real_reconstruction_dispatch_disabled"))
    return tuple(missing)


class ReconstructionService:
    """Runs StageExecutors through OnionWorkflow; there is no competing iteration driver."""

    def __init__(self, repository: Path, runs_root: Path):
        self.repository = repository.resolve(strict=True)
        self.runs_root = private_directory(self.repository, runs_root)
        self.last_workflow: OnionWorkflow | None = None
        self.last_executor: ReconstructionExecutor | None = None

    def run(self, *, ins_path: Path, hkl_path: Path, initial_model: Path | None,
            providers: ReconstructionProviders, policy: ReconstructionPolicy | None = None,
            synthetic: bool = False, cancel: CancelToken | None = None,
            checkpoint: tuple[Path, str] | None = None,
            clock: Callable[[], float] = time.monotonic,
            _initial_handoff: tuple[object, str] | None = None) -> dict:
        policy = policy or ReconstructionPolicy()
        cancel = cancel or CancelToken()
        if not synthetic:
            return {"status": "BLOCKED", "blockers": real_prerequisites(providers),
                    "real_invocations": 0, "scientifically_reviewed": False}
        handoff = None
        if _initial_handoff is not None:
            from crystalx_core.app.reconstruction_handoff import verify_initial_model_receipt
            if checkpoint is not None:
                raise ReconstructionBlocked("initial_handoff_and_restart_are_distinct_entries")
            handoff = verify_initial_model_receipt(_initial_handoff[0],
                expected_receipt_sha256=_initial_handoff[1])
            if (not handoff.synthetic or ins_path != handoff.instrument_ins_path
                    or hkl_path != handoff.paired_hkl_path or initial_model != handoff.model_path):
                raise ReconstructionBlocked("initial_handoff_mode_or_selected_paths_mismatch")
        if not is_demo_input(ins_path, hkl_path):
            raise ReconstructionBlocked("synthetic_providers_only_accept_builtin_synthetic_inputs")
        workspace = RunWorkspace.create(self.runs_root, label="synthetic-iterative-reconstruction",
                                        metadata={"synthetic": True, "workflow": "OnionWorkflow"})
        staged_ins = workspace.stage_input("instrument_instructions", ins_path)
        staged_hkl = workspace.stage_input("measured_reflections", hkl_path)
        if handoff is not None:
            workspace.write_json("reconstruction/initial-model-handoff.json", _initial_handoff[0].to_record()
                if hasattr(_initial_handoff[0], "to_record") else _initial_handoff[0])
        if (staged_ins.sha256, staged_hkl.sha256) != demo_digests() or (
                sha256_file(staged_ins.staged_path), sha256_file(staged_hkl.staged_path)) != demo_digests():
            raise ReconstructionBlocked("staged_synthetic_input_identity_mismatch")
        original = {ins_path: sha256_file(ins_path), hkl_path: sha256_file(hkl_path)}
        if checkpoint is not None:
            parent, pin = checkpoint
            parent_absolute = parent.resolve(strict=True)
            try:
                parent_absolute.relative_to(self.repository / "runs")
            except ValueError as exc:
                raise ReconstructionBlocked("restart_checkpoint_must_remain_private") from exc
            state = load_checkpoint(parent, expected_sha256=pin, policy=policy,
                                    hkl_path=staged_hkl.staged_path, expected_synthetic=True,
                                    expected_inference_audit=providers.inference_audit)
            # Independently reparse text; a pinned record may not supply inconsistent site fields.
            restored_path = workspace.write_text("reconstruction/restart-model.res", state.model.text)
            restored = read_snapshot(restored_path, reflection_sha256=staged_hkl.sha256)
            if restored != state.model:
                raise ReconstructionBlocked("checkpoint_model_fields_do_not_match_model_text")
        else:
            if initial_model is None:
                raise ReconstructionBlocked("automatic_initial_model_missing")
            from crystalx_core.app.reconstruction_demo import synthetic_seed_text
            import hashlib
            if sha256_file(initial_model) != hashlib.sha256(synthetic_seed_text().encode()).hexdigest():
                raise ReconstructionBlocked("synthetic_providers_require_the_pinned_synthetic_seed")
            seed = workspace.stage_input("automatic_initial_model", initial_model)
            expected_seed = hashlib.sha256(synthetic_seed_text().encode()).hexdigest()
            if seed.sha256 != expected_seed or sha256_file(seed.staged_path) != expected_seed:
                raise ReconstructionBlocked("staged_synthetic_seed_identity_mismatch")
            original[initial_model] = seed.sha256
            state = ReconstructionState(read_snapshot(seed.staged_path, reflection_sha256=staged_hkl.sha256),
                usage=Usage(tool_calls=handoff.source_tool_calls, elapsed_seconds=handoff.source_duration_s)
                if handoff else Usage())
        header = read_ins_header(staged_ins.staged_path)
        diagnosis = DataDiagnosis(ins=header, reflections=scan_hkl(staged_hkl.staged_path,
                                  cell=header.cell, hklf_hint=header.hklf_format))
        executor = ReconstructionExecutor(state, hkl_path=staged_hkl.staged_path,
            density=providers.density, inference=providers.inference, preparer=providers.preparer,
            inference_audit=providers.inference_audit, refinement=providers.refinement,
            policy=policy, synthetic=True, cancel=cancel, clock=clock)
        remaining = max(1, policy.max_rounds - state.usage.rounds)
        stage_budget = StageBudget(max_attempts=1, max_candidates=1,
                                   max_new_atoms=1, max_wall_clock_s=policy.max_wall_seconds,
                                   max_tool_calls=1, max_failures=1)
        workflow = OnionWorkflow(workspace, {kind: executor for kind in (
            StageKind.INITIAL_SOLUTION, StageKind.ANCHOR, StageKind.FIRST_SHELL,
            StageKind.SHELL_EXPANSION, StageKind.FINALIZE)},
            policy=OnionPolicy(max_shells=remaining,
                               # A provisional scaffold is explicitly allowed for software trials.
                               # It is never relabelled validated; real dispatch remains blocked.
                               require_anchor_before_expansion=False,
                               stage_budgets={kind.value: stage_budget for kind in StageKind}),
            clock=clock, cancel_token=cancel)
        workflow.ledger = LockLedger.from_snapshot(state.active_locks,
                                                   active_ids=tuple(state.active_locks))
        self.last_workflow, self.last_executor = workflow, executor
        result = workflow.run(diagnosis=diagnosis, prior=ChemicalPrior(),
                              radiation=RadiationAssumption.from_wavelength(diagnosis.wavelength_a))
        # The driver, rather than a returned provider claim, decides accepted active locks.
        state.active_locks = {key: workflow.ledger.get(key) for key in workflow.ledger.active_ids()}
        if any(sha256_file(path) != sha for path, sha in original.items()):
            raise ReconstructionBlocked("original_input_changed")
        if handoff is not None:
            if verify_initial_model_receipt(_initial_handoff[0],
                    expected_receipt_sha256=_initial_handoff[1]) != handoff:
                raise ReconstructionBlocked("initial_handoff_source_changed")
        accepted_available = state.assessment is not None
        blocked = not accepted_available or any(token in state.stop_code for token in ("missing", "mismatch", "not_pinned", "not_authorized"))
        status = "FAILED" if result.stop_reason.kind is StopReasonKind.INTERNAL_ERROR else "BLOCKED" if blocked else "SOFTWARE_EXECUTED"
        summary = {"status": status,
                   "synthetic": True, "workflow": "OnionWorkflow", "run_root": str(workspace.layout.root),
                   "capability_mode": "AI_REQUIRED" if policy.ai_required else "SHELX_ONLY_DIAGNOSTIC",
                   "stop_code": state.stop_code or result.stop_reason.kind.value,
                   "stop_reason": result.stop_reason, "usage": state.usage,
                   "accepted_model_sha256": state.model.sha256 if accepted_available else None,
                   "accepted_model_available": accepted_available,
                   "retained_model_sha256": state.model.sha256,
                   "active_locks": tuple(state.active_locks), "decisions": state.decisions,
                   "checkpoints": state.checkpoints,
                   "parent_checkpoint_sha256": state.parent_checkpoint_sha256,
                   "real_invocations": 0, "scientifically_reviewed": False,
                   "policy_sha256": digest(policy)}
        if handoff is not None:
            summary["source_initial_handoff"] = {"receipt_sha256": handoff.receipt_sha256,
                "source_run_id": handoff.source_run_id, "source_tool_calls_charged": handoff.source_tool_calls,
                "source_duration_seconds_charged": handoff.source_duration_s,
                "scope": "selected synthetic initial-solution invocation; not whole-source-run authorization",
                "real_dispatch_allowed": False}
        workspace.write_text("reconstruction/accepted-model.res" if accepted_available
                             else "reconstruction/retained-initial-model.res", state.model.text)
        workspace.write_json("reconstruction/summary.json", summary)
        workspace.finalize(result)
        return summary

    def run_from_initial_handoff(self, receipt, *, expected_receipt_sha256: str,
            providers: ReconstructionProviders, policy: ReconstructionPolicy | None = None,
            synthetic: bool = False, cancel: CancelToken | None = None) -> dict:
        """Stage a proven automatic output; real whole-chain dispatch remains disabled."""
        if not synthetic:
            return {"status": "BLOCKED", "blockers": real_prerequisites(providers),
                    "real_invocations": 0, "scientifically_reviewed": False}
        from crystalx_core.app.reconstruction_handoff import verify_initial_model_receipt
        verified = verify_initial_model_receipt(receipt, expected_receipt_sha256=expected_receipt_sha256)
        return self.run(ins_path=verified.instrument_ins_path, hkl_path=verified.paired_hkl_path,
            initial_model=verified.model_path, providers=providers, policy=policy, synthetic=True,
            cancel=cancel, _initial_handoff=(receipt, expected_receipt_sha256))

    def from_private_spec(self, spec: SolverSpec, *, providers: ReconstructionProviders,
                          policy: ReconstructionPolicy | None = None) -> dict:
        """Resolve the explicit seed role only; real dispatch remains behind future authority."""
        if spec.track is not AcceptanceTrack.INITIAL_MODEL_RECOVERY:
            raise ReconstructionBlocked("explicit_initial_model_recovery_track_required")
        blockers = tuple(spec.blockers) + real_prerequisites(providers)
        return {"status": "BLOCKED", "blockers": blockers, "real_invocations": 0,
                "seed_role": "AUTOMATIC_INITIAL_MODEL", "expert_answer_used": False}
