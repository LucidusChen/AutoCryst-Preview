"""Generated demo artifacts only; no SHELX/PLATON executable or real dataset is accessed."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from crystalx_core.app import RunConfig, RunMode, RunService, SolveRequest, write_demo_case
from crystalx_core.app.acceptance import verify_run
from crystalx_core.app.mock_toolchain import MockToolchain, MockToolchainExecutor
from crystalx_core.app.reconstruction_demo import synthetic_seed_text
from crystalx_core.app.reconstruction import ReconstructionService
from crystalx_core.app.reconstruction_file_demo import file_demo_providers
from crystalx_core.app.reconstruction_handoff import (
    HandoffError, InitialModelReceipt, create_initial_model_receipt, verify_initial_model_receipt,
)
from crystalx_core.domain.tools import ToolKind
from crystalx_core.solvers.mock import ScriptedExecutor
from crystalx_core.tools.runner import ProcessRunner
from crystalx_core.workflow.reconstruction import ReconstructionPolicy


BRANCH = "stages/s1-initial_solution/branches/b00-shelxt"
PROVENANCE = BRANCH + "/provenance_00.json"
STEM = "crystalx_demo"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExactSeedExecutor(MockToolchainExecutor):
    """The stand-in generates the exact synthetic iterative seed before provenance is recorded."""

    def _shelxt(self, plan):
        messages = super()._shelxt(plan)
        (plan.cwd / f"{plan.argv[1]}_a.res").write_text(synthetic_seed_text())
        return messages


class ExactSeedToolchain(MockToolchain):
    def _runner(self):
        return ProcessRunner(ExactSeedExecutor(step_seconds=0.001), base_env={})


def make_source(tmp_path, *, exact_seed=False, **mock):
    ins, hkl = write_demo_case(tmp_path / "generated-inputs")
    expected = {"ins": sha(ins), "hkl": sha(hkl)}
    factory = (lambda request: ExactSeedToolchain(step_seconds=0.001)) if exact_seed else (
        lambda request: MockToolchain(step_seconds=0.001, **mock))
    service = RunService(tmp_path / "runs", toolchain_factory=factory)
    handle = service.submit(SolveRequest(ins, hkl, mode=RunMode.MOCK_DEMO,
                                         config=RunConfig(mock_step_seconds=0.001)))
    service.wait(handle.run_id, timeout=10)
    return handle.run_dir, expected, (ins, hkl)


def receipt(root, expected, **kwargs):
    return create_initial_model_receipt(root, branch=BRANCH, provenance=PROVENANCE,
                                        solution_letter="a", expected_inputs=expected,
                                        require_real=False, **kwargs)


def read_json(path):
    return json.loads(path.read_text())


def update_json(path, change):
    data = read_json(path)
    change(data)
    path.write_text(json.dumps(data))


def mutate(path):
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"\nREM synthetic tamper\n")


def test_receipt_is_narrow_readonly_and_preserves_source_roles(tmp_path):
    root, expected, _ = make_source(tmp_path)
    before = {str(path.relative_to(root)): sha(path) for path in root.rglob("*") if path.is_file()}
    result = receipt(root, expected)
    assert result.original_instructions.role == "INSTRUMENT_INSTRUCTIONS"
    assert result.measured_reflections.role == "MEASURED_REFLECTIONS"
    assert result.automatic_model.role == "AUTOMATIC_INITIAL_MODEL"
    assert result.paired_reflections.role == "SHELXT_MATCHING_REFLECTIONS"
    assert result.original_instructions.sha256 == expected["ins"]
    assert result.measured_reflections.sha256 == expected["hkl"]
    assert result.shelxt_reflections.sha256 == expected["hkl"]
    assert result.automatic_model.relative_path == f"{BRANCH}/{STEM}_a.res"
    assert result.paired_reflections.relative_path == f"{BRANCH}/{STEM}_a.hkl"
    assert result.synthetic
    assert len(result.records) == 6
    restored = InitialModelReceipt.from_record(result.to_record())
    assert restored == result
    handoff = verify_initial_model_receipt(restored, expected_receipt_sha256=result.receipt_sha256)
    assert handoff.model_path == root / BRANCH / f"{STEM}_a.res"
    assert handoff.paired_hkl_path == root / BRANCH / f"{STEM}_a.hkl"
    assert handoff.instrument_ins_path == root / result.original_instructions.relative_path
    assert handoff.model_sha256 == sha(handoff.model_path)
    assert handoff.paired_hkl_sha256 == sha(handoff.paired_hkl_path)
    assert not handoff.real_dispatch_allowed
    assert not handoff.full_workflow_verified
    assert not handoff.scientifically_reviewed
    assert "expert" not in result.to_record()
    assert before == {str(path.relative_to(root)): sha(path) for path in root.rglob("*") if path.is_file()}
    with pytest.raises(FrozenInstanceError):
        result.stem = "changed"


def test_exact_generated_seed_fixture_matches_iterative_synthetic_gate(tmp_path):
    root, expected, _ = make_source(tmp_path, exact_seed=True)
    result = receipt(root, expected)
    handoff = verify_initial_model_receipt(result.to_record(), expected_receipt_sha256=result.receipt_sha256)
    assert handoff.model_path.read_bytes() == synthetic_seed_text().encode()
    assert handoff.instrument_ins_path.read_bytes() != handoff.model_path.read_bytes()
    assert handoff.paired_hkl_sha256 == expected["hkl"]


def test_missing_original_uploads_do_not_remove_retained_input_evidence(tmp_path):
    root, expected, originals = make_source(tmp_path)
    for path in originals:
        path.unlink()
    result = receipt(root, expected)
    assert verify_initial_model_receipt(result, expected_receipt_sha256=result.receipt_sha256)


def test_successful_initial_stage_can_be_handed_off_after_downstream_failure(tmp_path):
    root, expected, _ = make_source(tmp_path, fail_tool=ToolKind.PLATON_SPECIAL)
    assert read_json(root / "state.json")["phase"] == "failed"
    result = receipt(root, expected)
    assert result.automatic_model.role == "AUTOMATIC_INITIAL_MODEL"
    assert not verify_run(root, require_real=False).accepted
    assert not result.to_record()["full_workflow_verified"]


def test_synthetic_stage_cannot_establish_real_receipt_evidence(tmp_path):
    root, expected, _ = make_source(tmp_path)
    with pytest.raises(HandoffError, match="not real-process evidence"):
        create_initial_model_receipt(root, branch=BRANCH, provenance=PROVENANCE,
                                     solution_letter="a", expected_inputs=expected)


@pytest.mark.parametrize("where", ["state", "branch", "outcome", "invocation", "exit", "cancelled"])
def test_failed_or_cancelled_initial_stage_is_never_a_generated_checkpoint(tmp_path, where):
    root, expected, _ = make_source(tmp_path)
    if where == "state":
        update_json(root / "state.json", lambda data: next(s for s in data["stages"] if s["name"] == "shelxt").update(state="failed"))
    elif where in {"branch", "outcome"}:
        update_json(root / BRANCH / (where + ".json"), lambda data: data.update(state="failed"))
    else:
        key, value = {"invocation": ("succeeded", False), "exit": ("exit_code", 1), "cancelled": ("cancelled", True)}[where]
        update_json(root / PROVENANCE, lambda data: data.update({key: value}))
    with pytest.raises(HandoffError):
        receipt(root, expected)


def test_live_source_is_not_sealed_as_a_checkpoint(tmp_path):
    root, expected, _ = make_source(tmp_path)
    update_json(root / "state.json", lambda data: data.update(phase="running"))
    update_json(root / "summary.json", lambda data: data["process"].update(phase="running"))
    with pytest.raises(HandoffError, match="terminal"):
        receipt(root, expected)


@pytest.mark.parametrize("missing", ["branch.json", "outcome.json", "staging.json", "provenance_00.json"])
def test_missing_source_provenance_blocks_receipt(tmp_path, missing):
    root, expected, _ = make_source(tmp_path)
    (root / BRANCH / missing).unlink()
    with pytest.raises(HandoffError):
        receipt(root, expected)


@pytest.mark.parametrize("role", ["ins", "hkl"])
def test_expected_original_input_identity_is_required(tmp_path, role):
    root, expected, _ = make_source(tmp_path)
    expected[role] = "0" * 64
    with pytest.raises(HandoffError, match="expected input"):
        receipt(root, expected)


def test_both_original_input_pins_are_mandatory(tmp_path):
    root, expected, _ = make_source(tmp_path)
    del expected["ins"]
    with pytest.raises(HandoffError, match="both expected"):
        receipt(root, expected)


@pytest.mark.parametrize("artifact", ["original_ins", "original_hkl", "branch_ins", "branch_hkl", "model", "paired_hkl"])
def test_changed_source_or_paired_reflections_invalidates_independently_pinned_receipt(tmp_path, artifact):
    root, expected, _ = make_source(tmp_path)
    result = receipt(root, expected)
    paths = {"original_ins": result.original_instructions.relative_path,
             "original_hkl": result.measured_reflections.relative_path,
             "branch_ins": result.shelxt_instructions.relative_path,
             "branch_hkl": result.shelxt_reflections.relative_path,
             "model": result.automatic_model.relative_path,
             "paired_hkl": result.paired_reflections.relative_path}
    mutate(root / paths[artifact])
    with pytest.raises(HandoffError):
        verify_initial_model_receipt(result, expected_receipt_sha256=result.receipt_sha256)


def test_swapped_matching_output_is_not_accepted_merely_because_filename_and_cell_match(tmp_path):
    root, expected, _ = make_source(tmp_path)
    (root / BRANCH / f"{STEM}_a.hkl").write_text("synthetic unrelated reflection set")
    with pytest.raises(HandoffError, match="written provenance"):
        receipt(root, expected)


@pytest.mark.parametrize("statement", [False, True])
def test_original_hkl_fallback_remains_blocked_even_with_equal_cell_and_orientation_phrase(tmp_path, statement):
    root, expected, _ = make_source(tmp_path, shelxt_write_hkl=False, shelxt_orientation_statement=statement)
    with pytest.raises(HandoffError, match="fallback is unverified"):
        receipt(root, expected)


@pytest.mark.parametrize("role", ["modified", "input", "log"])
def test_model_must_be_a_newly_written_output_not_a_preexisting_or_expert_file(tmp_path, role):
    root, expected, _ = make_source(tmp_path)
    update_json(root / PROVENANCE, lambda data: next(item for item in data["outputs"] if item["relative_path"] == f"{STEM}_a.res").update(role=role))
    with pytest.raises(HandoffError, match="newly written"):
        receipt(root, expected)


def test_expert_component_summary_is_not_an_automatic_solution(tmp_path):
    root, expected, _ = make_source(tmp_path)
    update_json(root / "summary.json", lambda data: data.update(acceptance_track="COMPONENT_REGRESSION"))
    with pytest.raises(HandoffError, match="expert component"):
        receipt(root, expected)


def test_wrong_receipt_role_is_refused_even_if_the_wrong_record_is_repinned(tmp_path):
    root, expected, _ = make_source(tmp_path)
    result = receipt(root, expected)
    wrong = replace(result, automatic_model=replace(result.automatic_model, role="EXPERT_FINAL_ANSWER"))
    with pytest.raises(HandoffError, match="file roles"):
        verify_initial_model_receipt(wrong, expected_receipt_sha256=wrong.receipt_sha256)


def test_mutated_receipt_does_not_match_external_pin(tmp_path):
    root, expected, _ = make_source(tmp_path)
    result = receipt(root, expected)
    wrong = replace(result, solution_letter="b")
    with pytest.raises(HandoffError, match="independently pinned"):
        verify_initial_model_receipt(wrong, expected_receipt_sha256=result.receipt_sha256)


@pytest.mark.parametrize("change", [lambda data: data.update(real_dispatch_allowed=True),
                                    lambda data: data.update(scientifically_reviewed=True),
                                    lambda data: data.update(expert_answer="held-out.res")])
def test_receipt_record_cannot_grant_privileges_or_add_expert_inputs(tmp_path, change):
    root, expected, _ = make_source(tmp_path)
    data = receipt(root, expected).to_record()
    change(data)
    with pytest.raises(HandoffError):
        InitialModelReceipt.from_record(data)


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_model_alias_to_an_expert_or_external_file_is_not_opened(tmp_path, alias):
    root, expected, _ = make_source(tmp_path)
    model = root / BRANCH / f"{STEM}_a.res"
    outside = tmp_path / "synthetic-expert-answer.res"
    outside.write_bytes(model.read_bytes())
    model.unlink()
    if alias == "symlink":
        model.symlink_to(outside)
    else:
        model.hardlink_to(outside)
    with pytest.raises(HandoffError):
        receipt(root, expected)


@pytest.mark.parametrize("field,value", [("branch", "../expert"), ("provenance", "../expert.res"),
                                        ("solution_letter", "../answer"), ("solution_letter", "")])
def test_explicit_artifact_selection_cannot_escape_or_discover_files(tmp_path, field, value):
    root, expected, _ = make_source(tmp_path)
    options = dict(branch=BRANCH, provenance=PROVENANCE, solution_letter="a", expected_inputs=expected, require_real=False)
    options[field] = value
    with pytest.raises(HandoffError):
        create_initial_model_receipt(root, **options)


@pytest.mark.parametrize("duration", [-1, True, "1", None, float("nan"), float("inf")])
def test_selected_invocation_cost_is_explicit_and_finite(tmp_path, duration):
    root, expected, _ = make_source(tmp_path)
    update_json(root / PROVENANCE, lambda data: data.update(duration_s=duration))
    with pytest.raises(HandoffError):
        receipt(root, expected)


def reconstruction_repository(tmp_path):
    repo = tmp_path.resolve() / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("/runs/\n")
    return repo


@pytest.mark.parametrize("entry", ["direct", "service_facade"])
def test_verified_initial_checkpoint_runs_real_file_boundary_with_provisional_accept_reject(tmp_path, entry):
    repo = reconstruction_repository(tmp_path)
    source, expected, _ = make_source(repo / "runs/source", exact_seed=True)
    checkpoint = receipt(source, expected)
    pin = checkpoint.receipt_sha256
    source_before = {str(path.relative_to(source)): sha(path) for path in source.rglob("*") if path.is_file()}
    if entry == "direct":
        service = ReconstructionService(repo, repo / "runs/iterative-children")
        result = service.run_from_initial_handoff(checkpoint, expected_receipt_sha256=pin,
            providers=file_demo_providers(), synthetic=True)
    else:
        service = RunService(repo / "runs/iterative-children")
        result = service.reconstruct(repository=repo, initial_handoff=checkpoint,
            expected_receipt_sha256=pin, providers=file_demo_providers(), synthetic=True)
    assert [item["decision"] for item in result["decisions"]] == ["ACCEPT_PROVISIONAL", "REJECT", "ACCEPT_PROVISIONAL"]
    assert result["usage"].tool_calls == 5  # one retained SHELXT job plus four controlled child refinements
    assert result["usage"].elapsed_seconds >= checkpoint.source_duration_s
    assert result["source_initial_handoff"]["source_tool_calls_charged"] == 1
    assert result["source_initial_handoff"]["source_duration_seconds_charged"] == checkpoint.source_duration_s
    assert result["source_initial_handoff"]["receipt_sha256"] == pin
    assert result["real_invocations"] == 0 and not result["scientifically_reviewed"]
    child = Path(result["run_root"])
    retained = read_json(child / "reconstruction/initial-model-handoff.json")
    assert retained["receipt_sha256"] == pin
    assert retained["source_cost_scope"] == "selected_shelxt_invocation_only"
    assert not retained["full_source_run_cost_included"]
    assert subprocess.run(["git", "check-ignore", "--quiet", "--no-index", "--", str(child / "reconstruction/initial-model-handoff.json")], cwd=repo).returncode == 0
    assert source_before == {str(path.relative_to(source)): sha(path) for path in source.rglob("*") if path.is_file()}
    assert verify_initial_model_receipt(checkpoint, expected_receipt_sha256=pin)


def test_retained_initial_call_consumes_tight_budget_before_any_refinement_child(tmp_path):
    repo = reconstruction_repository(tmp_path)
    source, expected, _ = make_source(repo / "runs/source", exact_seed=True)
    checkpoint = receipt(source, expected)
    executor = ScriptedExecutor()
    service = ReconstructionService(repo, repo / "runs/iterative-children")
    result = service.run_from_initial_handoff(checkpoint, expected_receipt_sha256=checkpoint.receipt_sha256,
        providers=file_demo_providers(runner=ProcessRunner(executor)),
        policy=ReconstructionPolicy(max_tool_calls=1), synthetic=True)
    assert result["usage"].tool_calls == 1
    assert result["stop_code"] == "total_tool_call_budget_exhausted"
    assert result["decisions"] == []
    assert result["accepted_model_sha256"] is None and not result["accepted_model_available"]
    assert not (Path(result["run_root"]) / "reconstruction/accepted-model.res").exists()
    assert executor.calls == []


def test_invalid_receipt_pin_never_creates_a_child_run(tmp_path):
    repo = reconstruction_repository(tmp_path)
    source, expected, _ = make_source(repo / "runs/source", exact_seed=True)
    checkpoint = receipt(source, expected)
    service = ReconstructionService(repo, repo / "runs/iterative-children")
    with pytest.raises(HandoffError):
        service.run_from_initial_handoff(checkpoint, expected_receipt_sha256="0" * 64,
            providers=file_demo_providers(), synthetic=True)
    assert list(service.runs_root.iterdir()) == []


def test_real_service_handoff_stays_blocked_before_receipt_or_provider_reads(tmp_path, monkeypatch):
    repo = reconstruction_repository(tmp_path)
    service = ReconstructionService(repo, repo / "runs/iterative-children")
    monkeypatch.setattr("crystalx_core.app.reconstruction_handoff.verify_initial_model_receipt",
                        lambda *a, **k: pytest.fail("real dispatch must stop before receipt reads"))
    result = service.run_from_initial_handoff({"not": "an accepted receipt"}, expected_receipt_sha256="0" * 64,
        providers=file_demo_providers(), synthetic=False)
    assert result["status"] == "BLOCKED" and result["real_invocations"] == 0
    assert list(service.runs_root.iterdir()) == []
