"""Recovery plans are private and offline; every file and selected identity is synthetic."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from crystalx_core.app.models import RunMode
from crystalx_core.app.toolchain import Toolchain
from crystalx_core.domain.tools import ToolKind
from crystalx_core.private_acceptance import harness
from crystalx_core.private_acceptance.preflight import PreflightReport
from crystalx_core.private_acceptance.specs import SolverSpec, parse_solver_spec
from crystalx_core.workflow.reconstruction import ReconstructionPolicy


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path.resolve() / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("/runs/\n")
    return repo


@pytest.fixture
def case(tmp_path):
    root = tmp_path.resolve() / "fictional-measurement"
    root.mkdir()
    header = "TITL Synthetic original\nCELL 0.71073 10 11 12 90 90 90\nLATT -1\nSFAC C O\nUNIT 2 2\nHKLF 4\nEND\n"
    (root / "original.ins").write_text(header)
    (root / "measured.hkl").write_text("   1   0   0  100.00    1.00\n")
    (root / "automatic.res").write_text(header.replace("HKLF 4", "FVAR 1.0\nC1 1 0.1 0.2 0.3 11.0 0.04\nHKLF 4"))
    (root / "held-out.res").write_text("Synthetic answer trap")
    return root


@pytest.fixture
def spec_json(case):
    def sha(name):
        return hashlib.sha256((case / name).read_bytes()).hexdigest()

    return {"schema_version": 1, "case_id": "fictional", "track": "INITIAL_MODEL_RECOVERY",
            "inputs": {"ins": "original.ins", "hkl": "measured.hkl"},
            "seed": {"path": "automatic.res", "role": "AUTOMATIC_INITIAL_MODEL",
                     "sha256": sha("automatic.res"), "compatible_hkl_sha256": sha("measured.hkl"),
                     "provenance": {"kind": "owner_confirmed_automatic_initial_model",
                                    "source": "Synthetic checkpoint source",
                                    "owner_declaration": "Synthetic automatic-model provenance approval"}}}


class SyntheticSelectedChain(Toolchain):
    mode = RunMode.REAL

    @staticmethod
    def _selection(kind):
        identity = SimpleNamespace(path=Path("/fictional-tools") / kind.value, sha256="a" * 64,
                                   arch=SimpleNamespace(value="arm64"), source=SimpleNamespace(value="synthetic-test-identity"))
        return SimpleNamespace(kind=kind, identity=identity, is_usable=True)

    def resolution(self, kind):
        assert kind is ToolKind.SHELXL, "recovery plan must never select an initial-solution program"
        return self._selection(kind)

    def platon(self):
        return self._selection(ToolKind.PLATON_SPECIAL)


@pytest.fixture
def static_inspection(monkeypatch):
    policies = []

    def inspect(chain, *, policy):
        policies.append(policy)
        return PreflightReport("READY_STATIC", (), (), policy={
            "required_tools": [kind.value for kind in policy.required_tools],
            "require_platon": policy.require_platon,
        })

    monkeypatch.setattr(harness, "evaluate_toolchain", inspect)
    monkeypatch.setattr("crystalx_core.tools.runner.ProcessRunner.run", lambda *a, **k: pytest.fail("planning must never launch"))
    return policies


def plan(spec_json, case, repository, **kwargs):
    return harness.build_plan(parse_solver_spec(spec_json, case_root=case),
                              output_root=repository / "runs/private", repository=repository,
                              toolchain=SyntheticSelectedChain(), **kwargs)


def test_recovery_plan_pins_automatic_role_and_uses_shared_iterative_policy(case, spec_json, repository, static_inspection, monkeypatch):
    monkeypatch.setattr(SolverSpec, "to_solve_request", lambda *a, **k: pytest.fail("seed track must not enter one-shot solve API"))
    monkeypatch.setattr(harness.RunService, "preview", lambda *a, **k: pytest.fail("seed track must not preview one-shot SHELXT"))
    read = Path.open

    def guarded(self, *args, **kwargs):
        assert self != case / "held-out.res"
        return read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    result = plan(spec_json, case, repository)
    assert result["track"] == "INITIAL_MODEL_RECOVERY"
    assert result["status"] == "BLOCKED" and result["real_invocations"] == 0
    assert result["seed"]["role"] == "AUTOMATIC_INITIAL_MODEL"
    assert result["seed"]["sha256"] == spec_json["seed"]["sha256"]
    assert result["seed"]["compatible_hkl_sha256"] == result["inputs"]["hkl"]["sha256"]
    assert result["seed"]["source_setting"]["latt"] == -1
    assert result["seed"]["used_as_ground_truth"] is False
    assert result["seed"]["compatibility_scientifically_verified"] is False
    assert {row["role"] for row in result["input_roles"]} == {
        "INSTRUMENT_INSTRUCTIONS", "MEASURED_REFLECTIONS", "AUTOMATIC_INITIAL_MODEL"}
    assert [step["requested_tool"] for step in result["steps"]] == ["shelxl", "platon"]
    assert result["steps"][0]["argv"] == ["/fictional-tools/shelxl", "trial"]
    assert result["steps"][1]["argv"] == ["/fictional-tools/platon_special", "-u", "trial.cif"]
    assert result["shared_service_preview"] is result["component_preview"] is None
    assert len(static_inspection) == 1
    assert static_inspection[0].required_tools == (ToolKind.SHELXL,)
    assert static_inspection[0].require_platon
    workflow = result["workflow"]
    assert workflow["controller"] == "OnionWorkflow"
    assert workflow["budgets_provisional"] and not workflow["owner_approved"]
    assert workflow["initial_solution_performed"] is False
    assert workflow["expert_component_regression"] is False
    assert workflow["workflow_rounds_are_not_shelxl_cycles"]
    assert "trained_element_hypotheses" in workflow["loop"]
    assert workflow["rejection_restores"] == ["accepted_parent_model", "branch_active_locks"]
    assert "real_reconstruction_dispatch_disabled" in result["blockers"]
    assert "official_trained_checkpoint_and_reference_inference_not_bound" in result["blockers"]
    assert result["launch_requirements"]["implemented_launch_entrypoint"] is False
    assert not (repository / "runs/private/service-preview").exists()


@pytest.mark.parametrize("missing", ["object", "declared_file", "deleted_after_parse"])
def test_missing_seed_has_null_model_no_tool_selection_and_no_answer_fallback(case, spec_json, repository, static_inspection, monkeypatch, missing):
    if missing == "object":
        del spec_json["seed"]
    elif missing == "declared_file":
        spec_json["seed"]["path"] = "not-yet-supplied.res"
    spec = parse_solver_spec(spec_json, case_root=case)
    if missing == "deleted_after_parse":
        (case / "automatic.res").unlink()
    monkeypatch.setattr(SolverSpec, "to_solve_request", lambda *a, **k: pytest.fail("one-shot recovery fallback forbidden"))
    monkeypatch.setattr(SyntheticSelectedChain, "resolution", lambda *a, **k: pytest.fail("missing seed must not select tools"))
    monkeypatch.setattr(SyntheticSelectedChain, "platon", lambda *a, **k: pytest.fail("missing seed must not select PLATON"))
    result = harness.build_plan(spec, output_root=repository / "runs/private", repository=repository,
                                toolchain=SyntheticSelectedChain())
    assert result["status"] == "BLOCKED"
    assert result["seed"] is None
    assert result["steps"] == []
    assert result["max_real_invocations"] == result["real_invocations"] == 0
    assert "automatic_initial_model_missing" in result["blockers"]
    assert result["preflight"]["policy"]["required_tools"] == []
    assert result["preflight"]["policy"]["require_platon"] is False
    assert not static_inspection


def test_recovery_budget_caps_include_baseline_trials_and_required_checkpoint(case, spec_json, repository, static_inspection):
    policy = ReconstructionPolicy(max_rounds=3, max_proposals=2, max_tool_calls=4,
                                  trial_wall_seconds=5, max_wall_seconds=30,
                                  shelxl_cycles_per_trial=3)
    result = plan(spec_json, case, repository, reconstruction_policy=policy)
    assert result["workflow"]["refinement_call_cap"] == 3
    assert result["workflow"]["reserved_platon_calls"] == 1
    assert result["workflow"]["max_trial_proposals"] == 2
    assert result["max_real_invocations"] == 4
    assert sum(step["max_invocations"] for step in result["steps"]) <= policy.max_tool_calls
    assert result["maximum_total_program_wall_seconds"] <= policy.max_wall_seconds
    assert result["steps"][0]["shelxl_cycles"] == 3
    assert result["workflow"]["policy"]["max_rounds"] == 3
    assert result["steps"][0]["wall_seconds"] <= policy.trial_wall_seconds
    assert all(step["implementation_status"] == "PLAN_ONLY_REAL_DISPATCH_DISABLED" for step in result["steps"])


def test_too_small_tool_cap_blocks_instead_of_omitting_required_checkpoint(case, spec_json, repository, static_inspection):
    result = plan(spec_json, case, repository, reconstruction_policy=ReconstructionPolicy(max_tool_calls=1))
    assert "iterative_tool_budget_cannot_fit_baseline_and_required_checkpoint" in result["blockers"]
    assert result["workflow"]["refinement_call_cap"] == 0
    assert result["workflow"]["reserved_platon_calls"] == 1
    assert result["max_real_invocations"] == 1


def test_ai_required_track_cannot_become_shelx_only_plan(case, spec_json, repository, static_inspection):
    result = plan(spec_json, case, repository, reconstruction_policy=ReconstructionPolicy(ai_required=False))
    assert "initial_model_recovery_requires_trained_inference" in result["blockers"]
    assert result["status"] == "BLOCKED"


@pytest.mark.parametrize("policy", [False, 0, {}])
def test_invalid_policy_never_silently_uses_default_budgets(case, spec_json, repository, static_inspection, policy):
    with pytest.raises(harness.PlanError, match="shared ReconstructionPolicy"):
        plan(spec_json, case, repository, reconstruction_policy=policy)
    assert not static_inspection


def test_declared_prior_absent_from_seed_sfac_is_an_explicit_blocker(case, spec_json, repository, static_inspection):
    spec_json["priors"] = {"metals": [{"value": "Cu", "provenance": "owner_confirmed", "source": "Synthetic owner statement"}]}
    before = (case / "automatic.res").read_bytes()
    result = plan(spec_json, case, repository)
    assert "declared_prior_not_in_seed_sfac" in result["blockers"]
    assert result["preparation"]["missing_prior_elements_in_seed_sfac"] == ["Cu"]
    assert result["preparation"]["metal_and_element_flags_rewrite_sfac_unit"] is False
    assert (case / "automatic.res").read_bytes() == before


def test_changed_seed_after_contract_is_rejected(case, spec_json, repository, static_inspection):
    spec = parse_solver_spec(spec_json, case_root=case)
    (case / "automatic.res").write_text("changed synthetic model")
    with pytest.raises(harness.PlanError):
        harness.build_plan(spec, output_root=repository / "runs/private", repository=repository,
                            toolchain=SyntheticSelectedChain())


def test_changed_seed_during_preflight_is_detected(case, spec_json, repository, monkeypatch):
    def changing(chain, *, policy):
        (case / "automatic.res").write_text("changed synthetic model")
        return PreflightReport("READY_STATIC", (), ())

    monkeypatch.setattr(harness, "evaluate_toolchain", changing)
    with pytest.raises(harness.PlanError):
        plan(spec_json, case, repository)


def test_recovery_plan_saved_only_under_existing_private_guard(case, spec_json, repository, static_inspection):
    result = plan(spec_json, case, repository)
    path, pin = harness.save_plan(result, repository=repository, private_root=repository / "runs/private")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pin
    assert json.loads(path.read_text())["seed"]["role"] == "AUTOMATIC_INITIAL_MODEL"
    assert subprocess.run(["git", "check-ignore", "--quiet", "--no-index", "--", str(path)], cwd=repository).returncode == 0
