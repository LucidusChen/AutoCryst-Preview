"""Offline integration boundaries; all files and identities here are fictional."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from crystalx_core.app.models import RunConfig
from crystalx_core.app.toolchain import RealToolchain
from crystalx_core.private_acceptance import harness
from crystalx_core.private_acceptance.__main__ import main
from crystalx_core.private_acceptance.preflight import PreflightReport
from crystalx_core.private_acceptance.sealing import seal_run
from crystalx_core.private_acceptance.specs import parse_solver_spec, synthetic_solver_template


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("/runs/\n/realdata_examples/**\n")
    return repo


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "fictional_case"
    root.mkdir()
    (root / "original.ins").write_text("TITL Synthetic\nCELL 0.71073 10 11 12 90 90 90\n"
                                       "LATT -1\nSFAC C O\nUNIT 2 2\nHKLF 4\nEND\n")
    (root / "original.hkl").write_text("   1   0   0  100.00    1.00\n")
    (root / "held-out.res").write_text("Private synthetic answer trap")
    return root


@pytest.fixture
def chain():
    # No guessed toolbox or host PATH is needed for deterministic missing-tool planning.
    from crystalx_core.tools.resolver import ResolverConfig, ToolResolver
    return RealToolchain(resolver=ToolResolver(ResolverConfig(
        search_path=[], include_olex2=False, include_known_locations=False)))


def test_private_output_requires_ignored_untracked_subdirectory(repository, tmp_path):
    assert harness.private_directory(repository, repository / "runs/private") .is_dir()
    for location in (repository / "public", tmp_path / "outside", repository / "runs"):
        with pytest.raises(harness.PlanError):
            harness.private_directory(repository, location)
    (repository / ".gitignore").write_text("")
    with pytest.raises(harness.PlanError, match="ignore"):
        harness.private_directory(repository, repository / "runs/new")


def test_private_output_rejects_symlink_and_tracked_files(repository, tmp_path):
    (repository / "runs").mkdir()
    (repository / "runs/link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(harness.PlanError, match="symlink"):
        harness.private_directory(repository, repository / "runs/link/child")
    location = repository / "runs/tracked"
    location.mkdir()
    (location / "bad.json").write_text("{}")
    subprocess.run(["git", "add", "-f", "runs/tracked/bad.json"], cwd=repository, check=True)
    with pytest.raises(harness.PlanError, match="tracked"):
        harness.private_directory(repository, location)


def test_plan_is_offline_blocks_missing_tools_and_never_opens_answer(case, chain, repository, monkeypatch):
    spec = parse_solver_spec(synthetic_solver_template(), case_root=case)
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.name != "held-out.res"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr("crystalx_core.tools.runner.ProcessRunner.run",
                        lambda *a, **k: pytest.fail("offline planning attempted a launch"))
    plan = harness.build_plan(spec, output_root=repository / "runs/private", repository=repository, toolchain=chain)
    assert plan["status"] == "BLOCKED" and plan["real_invocations"] == 0
    assert plan["launch_requirements"]["environment"] == "CRYSTALX_REAL_TOOLS=1"
    assert plan["launch_requirements"]["implemented_launch_entrypoint"] is False
    assert all(step["argv"] is None for step in plan["steps"])
    assert plan["max_real_invocations"] == 3
    assert plan["maximum_total_program_wall_seconds"] == 2100
    assert all(not step["limits"]["memory_enforced"] for step in plan["steps"])
    assert "memory_enforcement_policy_not_implemented" in plan["blockers"]
    assert "expected_answer" not in json.dumps(plan)
    assert all(value is False for key, value in plan["status_layers"].items() if key != "SOFTWARE_TESTED")


def test_prior_flags_do_not_rewrite_instructions_and_missing_sfac_blocks(case, chain, repository):
    value = synthetic_solver_template()
    value["track"] = "UNSOLVED"
    value["priors"] = {"metals": [{"value": "Cu", "provenance": "owner_confirmed",
                                    "source": "Fictional owner evidence"}]}
    before = (case / "original.ins").read_bytes()
    spec = parse_solver_spec(value, case_root=case)
    plan = harness.build_plan(spec, output_root=repository / "runs/private", repository=repository, toolchain=chain)
    assert plan["preparation"]["missing_prior_elements_in_effective_sfac"] == ["Cu"]
    assert "declared_prior_not_in_effective_sfac" in plan["blockers"]
    assert not plan["preparation"]["metal_and_element_flags_rewrite_sfac_unit"]
    assert (case / "original.ins").read_bytes() == before
    assert spec.to_solve_request().prior.metals == ("Cu",)
    assert spec.to_solve_request().prior.composition == ()


def test_component_plan_has_no_initial_solver_step(case, chain, repository):
    value = synthetic_solver_template()
    value["track"] = "COMPONENT_REGRESSION"
    spec = parse_solver_spec(value, case_root=case)
    plan = harness.build_plan(spec, output_root=repository / "runs/private", repository=repository, toolchain=chain)
    assert [s["requested_tool"] for s in plan["steps"]] == ["shelxl", "platon"]
    assert plan["max_real_invocations"] == 2
    assert plan["shared_service_preview"] is None
    assert not plan["component_preview"]["may_start"]  # synthetic fixture has no model atoms
    assert "component_fixed_model_preconditions_blocked" in plan["blockers"]
    assert plan["preflight"]["policy"]["required_tools"] == ["shelxl"]


def test_ready_static_does_not_authorize_a_launch(case, chain, repository, monkeypatch):
    monkeypatch.setattr(harness, "evaluate_toolchain", lambda *a, **k: PreflightReport("READY_STATIC", (), ()))
    plan = harness.build_plan(parse_solver_spec(synthetic_solver_template(), case_root=case),
                              output_root=repository / "runs/private", repository=repository, toolchain=chain)
    assert plan["status"] == "BLOCKED"
    assert "bounded_owner_launch_authority_not_attached" in plan["blockers"]


def test_plan_checks_original_hash_after_preview(case, chain, repository, monkeypatch):
    spec = parse_solver_spec(synthetic_solver_template(), case_root=case)
    original = harness.RunService.preview

    def changing(service, request):
        result = original(service, request)
        request.hkl_path.write_text("changed synthetic input")
        return result

    monkeypatch.setattr(harness.RunService, "preview", changing)
    with pytest.raises(harness.PlanError, match="changed"):
        harness.build_plan(spec, output_root=repository / "runs/private", repository=repository, toolchain=chain)


def test_plan_rejects_translation_policy(case, chain, repository):
    spec = parse_solver_spec(synthetic_solver_template(), case_root=case)
    with pytest.raises(harness.PlanError, match="strict native"):
        harness.build_plan(spec, output_root=repository / "runs/private", repository=repository, toolchain=chain,
                           config=RunConfig(strict_native=False))


def test_save_plan_uses_new_folder_and_pinned_digest(repository):
    plan = {"schema_version": 1, "case_id": "synthetic"}
    first, digest = harness.save_plan(plan, repository=repository, private_root=repository / "runs/private")
    second, _ = harness.save_plan(plan, repository=repository, private_root=repository / "runs/private")
    assert first.parent != second.parent
    import hashlib
    assert hashlib.sha256(first.read_bytes()).hexdigest() == digest
    assert json.loads((first.parent / "plan.sha256.json").read_text())["sha256"] == digest


def test_actual_json_target_ignore_exception_is_rejected_before_write(repository):
    (repository / ".gitignore").write_text("/runs/**\n!/runs/**/\n!/runs/**/*.json\n")
    with pytest.raises(harness.PlanError, match="exact private output"):
        harness.save_plan({"fictional": True}, repository=repository,
                          private_root=repository / "runs/private")
    assert not list((repository / "runs/private").rglob("*.json"))


@pytest.mark.parametrize("budget", [float("nan"), float("inf")])
def test_nonfinite_plan_budgets_are_rejected(case, chain, repository, budget):
    spec = parse_solver_spec(synthetic_solver_template(), case_root=case)
    with pytest.raises(harness.PlanError, match="finite"):
        harness.build_plan(spec, output_root=repository / "runs/private", repository=repository,
                           toolchain=chain, config=RunConfig(strict_native=True, shelxt_timeout_s=budget))


def test_cli_template_and_no_launch_command(capsys):
    assert main(["template"]) == 0
    assert json.loads(capsys.readouterr().out)["case_id"] == "synthetic_example"
    with pytest.raises(SystemExit):
        main(["run"])


def test_cli_missing_repository_has_no_traceback(tmp_path, capsys):
    assert main(["--repository", str(tmp_path / "missing"), "privacy-audit"]) == 3
    error = capsys.readouterr().err
    assert "BLOCKED" in error and "Traceback" not in error


@pytest.mark.parametrize("cpu", [True, 1.5, float("nan")])
def test_explicit_cpu_seconds_must_be_integer(case, chain, repository, cpu):
    spec = parse_solver_spec(synthetic_solver_template(), case_root=case)
    with pytest.raises(harness.PlanError, match="integers"):
        harness.build_plan(spec, output_root=repository / "runs/private", repository=repository,
                           toolchain=chain, config=RunConfig(strict_native=True, cpu_seconds=cpu))


def test_cli_grade_verifies_pin_before_resolving_answer_config(repository, capsys):
    assert main(["--repository", str(repository), "grade", "--run", str(repository / "runs/missing"),
                 "--seal-sha256", "0" * 64, "--grader-spec", "do-not-open.json",
                 "--case-root", str(repository)]) == 3
    assert "BLOCKED" in capsys.readouterr().err


def test_cli_grade_no_provider_keeps_answer_closed(repository, monkeypatch, capsys):
    base = harness.private_directory(repository, repository / "runs/private_acceptance")
    run = base / "sealed"
    run.mkdir()
    (run / "model.res").write_text("synthetic model")
    pin = seal_run(run, private_root=base, case_id="fictional", track="DE_NOVO",
                   outputs={"model": "model.res"})
    (repository / "held-out.res").write_text("answer trap")
    spec = repository / "grader.json"
    spec.write_text(json.dumps({"schema_version": 1, "case_id": "fictional", "track": "DE_NOVO",
                                "answer": "held-out.res"}))
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.name != "held-out.res"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert main(["--repository", str(repository), "grade", "--run", str(run),
                 "--seal-sha256", pin.seal_sha256, "--grader-spec", str(spec),
                 "--case-root", str(repository)]) == 3
    assert "NOT_EVALUATED" in capsys.readouterr().out
