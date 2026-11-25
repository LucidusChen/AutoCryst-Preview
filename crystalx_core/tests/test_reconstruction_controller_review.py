"""Independent injected-provider review cases, using only generated synthetic inputs."""

import hashlib
import json
import subprocess
from dataclasses import replace

import pytest

from crystalx_core.app.demo import write_demo_case
from crystalx_core.app.reconstruction import ReconstructionService
from crystalx_core.app.reconstruction_demo import synthetic_providers, synthetic_seed_text
from crystalx_core.domain.result import StopReasonKind
from crystalx_core.tools.runner import CancelToken
from crystalx_core.workflow.reconstruction import ReconstructionBlocked, ReconstructionPolicy, digest


@pytest.fixture
def review_case(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    (repo / ".gitignore").write_text("/runs/\n")
    source = tmp_path / "synthetic-inputs"
    ins, hkl = write_demo_case(source)
    seed = source / "synthetic-seed.res"
    seed.write_text(synthetic_seed_text())
    service = ReconstructionService(repo, repo / "runs" / "review")
    return service, ins, hkl, seed


def _run(case, providers=None, **kwargs):
    service, ins, hkl, seed = case
    return service.run(ins_path=ins, hkl_path=hkl, initial_model=seed,
                       providers=providers or synthetic_providers(), synthetic=True, **kwargs)


class RefinementWrapper:
    def __init__(self, base, transform):
        self.base, self.transform = base, transform
        self.calls = 0

    def refine(self, request):
        self.calls += 1
        outcome = self.base.refine(request)
        return self.transform(request, outcome, self.calls)


def _refinement(providers, transform):
    wrapped = RefinementWrapper(providers.refinement, transform)
    return replace(providers, refinement=wrapped), wrapped


def _moved_text(request):
    lines = request.proposed_model.text.splitlines()
    target = request.proposed_model.sites[-1].site_id
    for n, line in enumerate(lines):
        fields = line.split()
        if fields and fields[0] == target:
            fields[2] = f"{float(fields[2]) + 0.001:.8f}"
            lines[n] = " ".join(fields)
    return "\n".join(lines) + "\n"


def test_ai_required_missing_inference_blocks_before_refinement(review_case):
    providers = synthetic_providers()
    summary = _run(review_case, replace(providers, inference=None))
    assert summary["status"] == "BLOCKED"
    assert providers.refinement.calls == 0
    assert summary["usage"].tool_calls == 0
    assert summary["real_invocations"] == 0 and not summary["scientifically_reviewed"]


def test_missing_density_blocks_before_refinement(review_case):
    providers = synthetic_providers()
    summary = _run(review_case, replace(providers, density=None))
    assert summary["status"] == "BLOCKED"
    assert providers.refinement.calls == 0


def test_unpinned_seed_never_reaches_synthetic_providers(review_case):
    service, ins, hkl, seed = review_case
    seed.write_text(synthetic_seed_text().replace("C1 ", "Unapproved1 "))
    providers = synthetic_providers()
    with pytest.raises(ReconstructionBlocked, match="pinned_synthetic_seed"):
        _run(review_case, providers)
    assert providers.refinement.calls == 0


def test_onion_gate_failure_does_not_commit_pending_baseline_model(review_case):
    service, _, _, seed = review_case
    expected_seed = hashlib.sha256(seed.read_bytes()).hexdigest()
    def invalid_ratio(request, outcome, number):
        return replace(outcome, lock_free_model_text=_moved_text(request),
                       assessment=replace(outcome.assessment, n_parameters=100000))
    providers, _ = _refinement(synthetic_providers(), invalid_ratio)
    summary = _run(review_case, providers)
    assert summary["accepted_model_sha256"] is None
    assert summary["retained_model_sha256"] == expected_seed
    assert not summary["accepted_model_available"]
    assert service.last_executor.state.assessment is None
    assert not summary["decisions"]
    assert not summary["active_locks"]


def test_provider_exception_keeps_previous_accepted_model_and_charged_call(review_case):
    service, _, _, seed = review_case
    expected_seed = hashlib.sha256(seed.read_bytes()).hexdigest()
    def fail_second(request, outcome, number):
        if number == 2:
            raise RuntimeError("synthetic confidential provider message")
        return outcome
    providers, wrapper = _refinement(synthetic_providers(), fail_second)
    summary = _run(review_case, providers)
    assert wrapper.calls == 2
    assert summary["usage"].tool_calls == 2
    assert summary["stop_reason"].kind is StopReasonKind.INTERNAL_ERROR
    assert summary["accepted_model_sha256"] == expected_seed
    assert "confidential provider message" not in repr(summary)


def test_post_refinement_cancellation_never_commits_trial(review_case):
    _, _, _, seed = review_case
    cancel = CancelToken()
    expected_seed = hashlib.sha256(seed.read_bytes()).hexdigest()
    def cancel_second(request, outcome, number):
        if number == 2:
            cancel.cancel("synthetic cancellation")
        return outcome
    providers, wrapper = _refinement(synthetic_providers(), cancel_second)
    summary = _run(review_case, providers, cancel=cancel)
    assert wrapper.calls == 2
    assert summary["stop_reason"].kind is StopReasonKind.USER_ABORTED
    assert summary["accepted_model_sha256"] == expected_seed
    assert not any(d["decision"] == "ACCEPT_PROVISIONAL" for d in summary["decisions"])


def test_post_refinement_wall_deadline_never_commits_trial(review_case):
    _, _, _, seed = review_case
    clock = [0.0]
    expected_seed = hashlib.sha256(seed.read_bytes()).hexdigest()
    def delay_second(request, outcome, number):
        if number == 2:
            clock[0] = 2.0
        return outcome
    providers, wrapper = _refinement(synthetic_providers(), delay_second)
    summary = _run(review_case, providers, policy=ReconstructionPolicy(max_wall_seconds=1),
                   clock=lambda: clock[0])
    assert wrapper.calls == 2
    assert summary["stop_reason"].kind is StopReasonKind.BUDGET_EXCEEDED
    assert summary["accepted_model_sha256"] == expected_seed


class DensityWrapper:
    def __init__(self, base, transform):
        self.base, self.transform = base, transform

    def observe(self, model):
        return self.transform(self.base.observe(model))


@pytest.mark.parametrize("change,code", [
    ({"source_sha256": "untrusted-source"}, "density_source_identity"),
    ({"basis_key": "wrong-basis"}, "density_model_or_reflection"),
    ({"coverage": "full"}, "density_capability"),
])
def test_invalid_density_evidence_is_not_used_for_model_changes(review_case, change, code):
    providers = synthetic_providers()
    providers = replace(providers, density=DensityWrapper(providers.density,
                                                         lambda value: replace(value, **change)))
    summary = _run(review_case, providers)
    assert code in summary["stop_code"]
    assert providers.inference.calls == 0
    assert providers.refinement.calls == 1
    assert not summary["decisions"]


def test_inference_checkpoint_identity_mismatch_blocks_proposal(review_case):
    providers = synthetic_providers()
    class WrongIdentity:
        def predict_elements(self, request, **kwargs):
            result = providers.inference.predict_elements(request, **kwargs)
            return replace(result, model_identity=replace(result.model_identity, weights_sha256=digest("other-weights")))
    summary = _run(review_case, replace(providers, inference=WrongIdentity()))
    assert "inference_checkpoint_or_source_identity_mismatch" in summary["stop_code"]
    assert providers.refinement.calls == 1
    assert not summary["decisions"]


def test_changed_declared_trial_input_is_not_accepted(review_case):
    _, _, _, seed = review_case
    expected_seed = hashlib.sha256(seed.read_bytes()).hexdigest()
    def corrupt_copy(request, outcome, number):
        if number == 2:
            request.hkl_path.chmod(0o644)
            request.hkl_path.write_text("synthetic tampered copy\n")
        return outcome
    providers, wrapper = _refinement(synthetic_providers(), corrupt_copy)
    summary = _run(review_case, providers)
    assert wrapper.calls == 2
    assert "changed_declared_inputs" in summary["stop_code"]
    assert summary["accepted_model_sha256"] == expected_seed
    assert not summary["active_locks"]


def test_provider_review_does_not_commit_trial_or_trial_locks(review_case):
    _, _, _, seed = review_case
    expected_seed = hashlib.sha256(seed.read_bytes()).hexdigest()
    def review_second(request, outcome, number):
        return replace(outcome, review_required=True) if number == 2 else outcome
    providers, wrapper = _refinement(synthetic_providers(), review_second)
    summary = _run(review_case, providers)
    assert wrapper.calls == 2
    assert summary["stop_reason"].kind is StopReasonKind.REVIEW_CHECKPOINT
    assert summary["accepted_model_sha256"] == expected_seed
    assert summary["decisions"][-1]["decision"] == "REVIEW"
    assert not summary["active_locks"]


def test_inflight_checkpoint_records_reserved_call_before_provider_entry(review_case):
    checked = []
    def inspect_checkpoint(request, outcome, number):
        reconstruction = next(parent for parent in request.work_directory.parents if parent.name == "reconstruction")
        checkpoints = sorted(reconstruction.glob("checkpoint-*-in-flight.json"))
        assert checkpoints
        payload = json.loads(checkpoints[-1].read_text())
        assert payload["state"]["usage"]["tool_calls"] == number
        checked.append(number)
        if number == 2:
            raise RuntimeError("simulated provider interruption")
        return outcome
    providers, _ = _refinement(synthetic_providers(), inspect_checkpoint)
    summary = _run(review_case, providers)
    assert checked == [1, 2]
    assert summary["usage"].tool_calls == 2


def test_restart_rejects_changed_inference_audit(review_case):
    service, ins, hkl, _ = review_case
    policy = ReconstructionPolicy(max_rounds=8, max_inference_calls=8)
    summary = _run(review_case, policy=policy)
    relative, pin = summary["checkpoints"][-1]
    from pathlib import Path
    checkpoint = Path(summary["run_root"]) / relative
    providers = synthetic_providers(restart=True)
    providers = replace(providers, inference_audit=replace(providers.inference_audit,
                                                          weights_sha256=digest("changed-on-restart")))
    with pytest.raises(ReconstructionBlocked, match="inference_provenance_changed"):
        service.run(ins_path=ins, hkl_path=hkl, initial_model=None, providers=providers,
                    synthetic=True, policy=policy, checkpoint=(checkpoint, pin))


def test_cancel_before_next_execute_preserves_already_committed_trial(review_case, monkeypatch):
    from crystalx_core.domain.onion import StageKind
    from crystalx_core.workflow.reconstruction import ReconstructionExecutor
    cancel = CancelToken()
    committed = []
    original = ReconstructionExecutor.commit_stage
    def cancel_after_commit(self, request, chosen, active_lock_ids):
        original(self, request, chosen, active_lock_ids)
        if request.stage.kind is StageKind.SHELL_EXPANSION and chosen is not None:
            committed.append(self.state.model.sha256)
            cancel.cancel("stop before next execute")
    monkeypatch.setattr(ReconstructionExecutor, "commit_stage", cancel_after_commit)
    summary = _run(review_case, cancel=cancel)
    assert committed
    assert summary["stop_reason"].kind is StopReasonKind.USER_ABORTED
    assert summary["accepted_model_sha256"] == committed[-1]
    assert summary["decisions"][-1]["decision"] == "ACCEPT_PROVISIONAL"


@pytest.mark.parametrize("change", ["input_hash", "forbidden_fallback"])
def test_inference_result_must_match_request_and_no_fallback_contract(review_case, change):
    providers = synthetic_providers()
    class WrongResult:
        def predict_elements(self, request, **kwargs):
            result = providers.inference.predict_elements(request, **kwargs)
            if change == "input_hash":
                return replace(result, input_sha256=digest("stale coordinate request"))
            return replace(result, fell_back=True, fallback_reason="unapproved fallback")
    summary = _run(review_case, replace(providers, inference=WrongResult()))
    assert providers.refinement.calls == 1, "unbound predictions must not reach trial refinement"
    assert not summary["decisions"]
    assert not summary["scientifically_reviewed"]
