"""Genuine model-file mutations through OnionWorkflow, using only generated providers/data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from crystalx_core.app.demo import write_demo_case
from crystalx_core.app.reconstruction import ReconstructionProviders, ReconstructionService
from crystalx_core.app.reconstruction_demo import (
    SYNTHETIC_AUDIT, SyntheticDensity, SyntheticInference, SyntheticPreparer, SyntheticRefinement,
    run_demo, synthetic_providers, synthetic_seed_text,
)
from crystalx_core.app.service import RunService
from crystalx_core.domain.candidate import CandidateStatus
from crystalx_core.domain.result import StopReasonKind
from crystalx_core.workflow.reconstruction import (
    ReconstructionBlocked, ReconstructionPolicy, decide, digest, load_checkpoint,
)


@pytest.fixture
def local(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("/runs/\n")
    source = repo / "runs/generated-inputs"
    source.mkdir(parents=True)
    ins, hkl = write_demo_case(source)
    seed = source / "automatic.res"
    seed.write_text(synthetic_seed_text())
    return repo, ins, hkl, seed


def run(local, *, providers=None, policy=None, **kwargs):
    repo, ins, hkl, seed = local
    service = ReconstructionService(repo, repo / "runs/iterations")
    result = service.run(ins_path=ins, hkl_path=hkl, initial_model=seed,
                         providers=providers or synthetic_providers(),
                         policy=policy or ReconstructionPolicy(), synthetic=True, **kwargs)
    return service, result


def test_actual_multiround_models_are_mutated_and_bad_low_r_is_rejected(local):
    _, _, _, seed = local
    before = seed.read_bytes()
    service, summary = run(local)
    decisions = summary["decisions"]
    assert [d["decision"] for d in decisions] == ["ACCEPT_PROVISIONAL", "REJECT", "ACCEPT_PROVISIONAL"]
    assert len({d["trial_sha256"] for d in decisions}) == 3
    assert decisions[1]["parent_sha256"] == decisions[2]["parent_sha256"]
    assert decisions[1]["trial_locks"] and not decisions[1]["parent_locks"]
    assert "lock-000" not in summary["active_locks"]  # rejected trial's restraint stays inactive
    assert summary["active_locks"] == ("lock-001",)
    assert summary["stop_code"] == "no_progress_not_convergence"
    assert summary["stop_reason"].kind is StopReasonKind.REVIEW_CHECKPOINT
    assert not summary["scientifically_reviewed"] and summary["real_invocations"] == 0
    root = Path(summary["run_root"])
    final = (root / "reconstruction/accepted-model.res").read_text()
    assert "C1 3 " in final  # stable label; SFAC changes from carbon to nitrogen
    assert "X003 3 " in final
    assert "X002 " not in final
    rejected = json.loads((root / "reconstruction/trial-0002/refine/assessment.json").read_text())
    accepted = json.loads((root / "reconstruction/trial-0001/refine/assessment.json").read_text())
    assert rejected["r1"] < accepted["r1"]
    assert "DFIX 2 0.2 Zn1 X002" in (root / "reconstruction/trial-0002/refine/trial.ins").read_text()
    assert "DFIX 2 0.2 Zn1 X003" in (root / "reconstruction/trial-0003/refine/trial.ins").read_text()
    assert "X002" not in (root / "reconstruction/trial-0003/refine/trial.ins").read_text()
    states = [c.status for c in service.last_workflow.workspace.lineage.iter_all()]
    assert CandidateStatus.ABANDONED in states
    assert seed.read_bytes() == before


def test_restart_keeps_hash_pin_usage_locks_and_parent_history(local):
    repo, ins, hkl, _ = local
    policy = ReconstructionPolicy(max_rounds=8, max_inference_calls=8)
    service, first = run(local, policy=policy)
    relative, pin = first["checkpoints"][-1]
    checkpoint = Path(first["run_root"]) / relative
    before = checkpoint.read_bytes()
    second = service.run(ins_path=ins, hkl_path=hkl, initial_model=None,
                         providers=synthetic_providers(restart=True), policy=policy, synthetic=True,
                         checkpoint=(checkpoint, pin))
    assert second["run_root"] != first["run_root"]
    assert second["parent_checkpoint_sha256"] == pin
    assert second["usage"].proposals == first["usage"].proposals + 1
    assert second["usage"].tool_calls == first["usage"].tool_calls + 1
    assert len(second["active_locks"]) == 2
    assert second["decisions"][-1]["decision"] == "ACCEPT_PROVISIONAL"
    assert second["decisions"][-1]["run_id"] != first["decisions"][-1]["run_id"]
    assert service.last_workflow.workspace.lineage.iter_all()[0].provenance.input_hashes["parent_checkpoint"] == pin
    assert second["accepted_model_sha256"] != first["accepted_model_sha256"]
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("change", ["pin", "policy", "hkl", "audit", "unknown_field", "model_fields", "negative_usage"])
def test_restart_rejects_changed_pins_basis_policy_and_overfull_state(local, change):
    repo, ins, hkl, _ = local
    policy = ReconstructionPolicy(max_rounds=8, max_inference_calls=8)
    _, first = run(local, policy=policy)
    relative, pin = first["checkpoints"][-1]
    checkpoint = Path(first["run_root"]) / relative
    providers = synthetic_providers()
    if change == "pin":
        pin = "0" * 64
    elif change == "policy":
        policy = replace(policy, max_tool_calls=8)
    elif change == "hkl":
        hkl.write_text(hkl.read_text() + "changed")
    elif change == "audit":
        providers = replace(providers, inference_audit=replace(SYNTHETIC_AUDIT, weights_sha256="1" * 64))
    else:
        payload = json.loads(checkpoint.read_text())
        if change == "unknown_field":
            payload["state"]["answer_path"] = "never-open.res"
        elif change == "model_fields":
            payload["state"]["model"]["sites"][0]["fractional"][0] += .1
        else:
            payload["state"]["usage"]["tool_calls"] = -1
        checkpoint = checkpoint.parent / ("altered-" + change + ".json")
        checkpoint.write_text(json.dumps(payload))
        pin = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    service = ReconstructionService(repo, repo / "runs/restart")
    with pytest.raises((ValueError, ReconstructionBlocked)):
        service.run(ins_path=ins, hkl_path=hkl, initial_model=None, providers=providers,
                    policy=policy, synthetic=True, checkpoint=(checkpoint, pin))


@pytest.mark.parametrize("field,limit,maximum", [
    ("max_rounds", 1, 1), ("max_proposals", 1, 1), ("max_tool_calls", 1, 1),
    ("max_inference_calls", 1, 1), ("max_added_atoms", 1, 1), ("max_failures", 1, 1),
])
def test_global_budgets_never_exceed_their_limits(local, field, limit, maximum):
    policy = replace(ReconstructionPolicy(), **{field: limit})
    _, result = run(local, policy=policy)
    usage = result["usage"]
    name = {"max_rounds": "rounds", "max_proposals": "proposals", "max_tool_calls": "tool_calls",
            "max_inference_calls": "inference_calls", "max_added_atoms": "added_atoms", "max_failures": "failures"}[field]
    assert getattr(usage, name) <= maximum
    assert result["stop_reason"].kind is StopReasonKind.BUDGET_EXCEEDED


def test_internal_cycles_differ_from_outer_iterations_and_real_mode_never_calls(local):
    providers = synthetic_providers()
    _, result = run(local, providers=providers)
    assert providers.refinement.calls == 4
    assert all(t.cycles == 2 for t in providers.refinement.trials)
    assert result["usage"].rounds > 1
    repo, _, _, _ = local
    real = ReconstructionService(repo, repo / "runs/real-plan").run(
        ins_path=Path("do-not-open.ins"), hkl_path=Path("do-not-open.hkl"),
        initial_model=Path("expert-answer-trap.res"), providers=providers)
    assert real["status"] == "BLOCKED" and real["real_invocations"] == 0
    assert providers.refinement.calls == 4
    assert "official_trained_checkpoint_provenance_missing" in real["blockers"]


def test_existing_runservice_application_api_calls_onion(local):
    repo, ins, hkl, seed = local
    result = RunService(repo / "runs/app").reconstruct(repository=repo, ins_path=ins, hkl_path=hkl,
        initial_model=seed, providers=synthetic_providers(), synthetic=True)
    assert result["workflow"] == "OnionWorkflow"
    assert len(result["decisions"]) == 3


def test_answer_access_trap_covers_solver_and_restart(local, monkeypatch):
    repo, ins, hkl, seed = local
    answer = repo / "held-out-answer.res"
    answer.write_text("do not open")
    original = Path.open

    def guard(path, *args, **kwargs):
        assert path != answer
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    summary = run_demo(repo, repo / "runs/full-demo")
    assert summary["restart_decisions"][-1] == "ACCEPT_PROVISIONAL"
    assert summary["real_invocations"] == 0


def test_q_peak_list_is_not_a_full_map(local):
    _, result = run(local, policy=replace(ReconstructionPolicy(), require_full_map=True))
    assert result["stop_code"] == "full_density_map_provider_missing"
    assert result["usage"].proposals == 0


def test_shelx_only_diagnostic_is_explicit_and_does_not_claim_ai(local):
    providers = replace(synthetic_providers(), inference=None, inference_audit=None, preparer=None)
    _, result = run(local, providers=providers, policy=replace(ReconstructionPolicy(), ai_required=False))
    assert result["capability_mode"] == "SHELX_ONLY_DIAGNOSTIC"
    assert result["usage"].inference_calls == 0
    assert result["decisions"] == []
    assert result["stop_code"] == "no_progress_not_convergence"


@pytest.mark.parametrize("missing_check", ["geometry", "coordination", "occupancy_adp", "chirality_symmetry",
                                          "restraints", "heavy_site_drift", "residual_distribution"])
def test_missing_required_trial_check_stops_without_promoting_model(local, missing_check):
    base = SyntheticRefinement()

    class MissingCheck:
        def refine(self, request):
            outcome = base.refine(request)
            if base.calls == 2:
                checks = tuple((name, "NOT_EVALUATED" if name == missing_check else result)
                               for name, result in outcome.assessment.checks)
                return replace(outcome, assessment=replace(outcome.assessment, checks=checks))
            return outcome

    providers = replace(synthetic_providers(), refinement=MissingCheck())
    _, summary = run(local, providers=providers)
    assert base.calls == 2
    assert summary["decisions"][-1]["decision"] == "REVIEW"
    assert summary["accepted_model_sha256"] == hashlib.sha256(local[3].read_bytes()).hexdigest()
    assert not summary["active_locks"]


def test_r_improvement_without_residual_distribution_progress_is_rejected(local):
    base = SyntheticRefinement()
    baseline = []

    class LowerROnly:
        def refine(self, request):
            outcome = base.refine(request)
            if base.calls == 1:
                baseline.append(outcome.assessment.residual_regions)
            else:
                outcome = replace(outcome, assessment=replace(outcome.assessment, residual_regions=baseline[0]))
            return outcome

    _, summary = run(local, providers=replace(synthetic_providers(), refinement=LowerROnly()),
                     policy=replace(ReconstructionPolicy(), max_failures=1))
    assert summary["decisions"][-1]["decision"] == "REJECT"
    assert summary["decisions"][-1]["reasons"] == ("no_joint_refinement_density_progress",)
    assert summary["accepted_model_sha256"] == hashlib.sha256(local[3].read_bytes()).hexdigest()


def test_oscillation_stops_before_revisiting_a_previous_model(local):
    base = SyntheticInference()

    class ToggleExistingSite:
        def predict_elements(self, request, **kwargs):
            result = base.predict_elements(request, **kwargs)
            if base.calls > 1:
                from crystalx_core.domain.inference import ClassProbability, ElementPrediction
                result = replace(result, predictions=tuple(
                    ElementPrediction(p.site_id, (ClassProbability(6, .99), ClassProbability(7, .01)))
                    if p.site_id == "C1" else p for p in result.predictions))
            return result

    providers = replace(synthetic_providers(), inference=ToggleExistingSite())
    _, summary = run(local, providers=providers)
    assert summary["stop_code"] == "oscillation_detected"
    assert providers.refinement.calls == 2  # baseline plus first accepted flip
    assert len(summary["decisions"]) == 1
    assert summary["decisions"][0]["decision"] == "ACCEPT_PROVISIONAL"


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -1, 0])
def test_invalid_bounds_rejected(bad):
    with pytest.raises(ValueError):
        ReconstructionPolicy(max_wall_seconds=bad)
