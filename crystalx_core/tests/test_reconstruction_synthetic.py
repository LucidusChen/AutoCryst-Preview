"""Supported demo injection, with no optional dependencies or real tool executions."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import pytest

from crystalx_core.app.demo import DEMO_MARKER, demo_digests, write_demo_case
from crystalx_core.app.models import RunMode
from crystalx_core.app.reconstruction_backend import ReconstructionBackendService, ReconstructionRequest
from crystalx_core.app.reconstruction_demo import synthetic_seed_text
from crystalx_core.app.reconstruction_synthetic import assigner_factory, toolchain_factory
from crystalx_core.app.solve_routes import (
    InitialRouteConfig, InitialStrategy, PhasingDataset, ProteinPhasingConfig, plan_initial_route,
)
from crystalx_core.domain.tools import ResolutionSource, ToolKind
from crystalx_core.io.shelx_res import read_res
from crystalx_core.tools.runner import CancelToken
from crystalx_core.workflow.reconstruction import ReconstructionBlocked
from crystalx_core.workflow.refinement_cycle import CyclePolicy
from crystalx_core.workspace.hashing import sha256_file


@pytest.fixture
def demo(tmp_path):
    repo = tmp_path.resolve() / "synthetic-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("/runs/\n")
    ins, hkl = write_demo_case(repo / "runs/inputs")
    return repo, ReconstructionRequest(ins, hkl, mode=RunMode.MOCK_DEMO,
                                       cycle=CyclePolicy(iterations=2))


def _service(repo):
    return ReconstructionBackendService(repo, repo / "runs/reconstruction",
                                        toolchain_factory=toolchain_factory,
                                        assigner_factory=assigner_factory)


@pytest.mark.parametrize("route,tool", [
    (InitialRouteConfig(), ToolKind.SHELXT),
    (InitialRouteConfig(InitialStrategy.SMALL_MOLECULE), ToolKind.SHELXS),
    (InitialRouteConfig(InitialStrategy.SMALL_MOLECULE, small_solver="shelxd",
                        find_sites=2, plop=(10, 2)), ToolKind.SHELXD),
])
def test_supported_factories_complete_all_three_small_routes_and_are_honest(demo, route, tool):
    repo, request = demo
    request = replace(request, route=route)
    result = _service(repo).run(request)
    assert result["phase"] == "completed", result
    assert result["stop_code"] == "fixed_rounds_completed"
    assert result["mode"] == RunMode.MOCK_DEMO.value
    assert result["inference_calls"] == 2 and result["tool_calls"] == 3
    assert not result["scientifically_validated"] and not result["expert_reference_used"]
    assert result["checkcif_status"] == "NOT_EVALUATED"
    root = Path(result["run_root"])
    initial = json.loads((root / "initial/synthetic-provider.json").read_text())
    assert initial["synthetic"] and not initial["actual_pretrained_weights_executed"]
    assert initial["original_hkl_sha256"] == demo_digests()[1]
    assert result["initial"]["invocations"][0]["tool"] == tool.value
    for evidence in root.glob("cycles/round-*/inference.json"):
        record = json.loads(evidence.read_text())
        assert record["provenance"]["synthetic"]
        assert not record["provenance"]["actual_pretrained_weights_executed"]
        assert record["provenance"]["input_protocol"] == "current_labels"
        assert record["result"]["probability_kind"].startswith("synthetic")
    for suffix in ("res", "cif", "fcf"):
        output = root / "outputs" / ("ai_solved." + suffix)
        assert DEMO_MARKER in output.read_text() and "SYNTHETIC" in output.read_text()
    for evidence in root.glob("cycles/round-*/shelxl/invocation.json"):
        record = json.loads(evidence.read_text())
        assert record["executable_source"] == ResolutionSource.SYNTHETIC_DEMO.value
        assert not record["required_translation"]
    assert (sha256_file(request.ins_path), sha256_file(request.hkl_path)) == demo_digests()


@pytest.mark.parametrize("factory", [toolchain_factory, assigner_factory])
@pytest.mark.parametrize("change,code", [
    ({"mode": RunMode.REAL}, "requires_mock_demo_mode"),
    ({"checkpoint": Path("does-not-exist.pth")}, "does_not_load_weights"),
    ({"checkpoint_sha256": "a" * 64}, "does_not_load_weights"),
    ({"device": "mps"}, "does_not_load_weights"),
    ({"candidate_elements": ("C", "N", "Zn")}, "builtin_candidate_elements"),
    ({"anchor_elements": ("Fe",)}, "builtin_zinc_anchor"),
    ({"cycle": CyclePolicy(inference_protocol="initial_labels")}, "current_labels_only"),
    ({"cycle": CyclePolicy(inference_protocol="upstream_ins")}, "current_labels_only"),
])
def test_both_factories_reject_unsupported_mode_weights_protocol_and_chemistry(demo, factory, change, code):
    _, request = demo
    with pytest.raises(ReconstructionBlocked, match=code):
        factory(replace(request, **change))


@pytest.mark.parametrize("factory", [toolchain_factory, assigner_factory])
@pytest.mark.parametrize("strategy", [InitialStrategy.PROTEIN_CDE, InitialStrategy.SHELXT])
def test_both_factories_reject_protein_context_without_reading_extra_dataset(demo, factory, strategy):
    _, request = demo
    protein = ProteinPhasingConfig(datasets=(PhasingDataset("SAD", Path("never-read-extra.hkl")),))
    request = replace(request, route=InitialRouteConfig(strategy, protein=protein))
    with pytest.raises(ReconstructionBlocked, match="protein_or_extra_datasets"):
        factory(request)


@pytest.mark.parametrize("factory", [toolchain_factory, assigner_factory])
@pytest.mark.parametrize("input_name", ["ins_path", "hkl_path"])
def test_both_factories_pin_both_originals_and_reject_marker_only_spoofs(demo, factory, input_name):
    _, request = demo
    source = getattr(request, input_name)
    source.write_bytes(source.read_bytes() + b"\n")
    assert DEMO_MARKER in request.ins_path.read_text()
    with pytest.raises(ReconstructionBlocked, match="not_pinned_builtin_data"):
        factory(request)


@pytest.mark.parametrize("factory", [toolchain_factory, assigner_factory])
@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_factory_rejects_input_aliases_even_with_exact_content(demo, factory, alias):
    _, request = demo
    aliased = request.ins_path.with_name("alias.ins")
    if alias == "symlink":
        aliased.symlink_to(request.ins_path)
    else:
        os.link(request.ins_path, aliased)
    with pytest.raises(ReconstructionBlocked, match="not_pinned_builtin_data"):
        factory(replace(request, ins_path=aliased))


def _staged_initial(request, directory):
    directory.mkdir()
    route = plan_initial_route(request.route, ins_path=request.ins_path, hkl_path=request.hkl_path,
                               work_dir=directory)
    (directory / "original.ins").write_bytes(request.ins_path.read_bytes())
    (directory / "original.hkl").write_bytes(request.hkl_path.read_bytes())
    (directory / (route.stem + ".ins")).write_text(route.prepared_ins)
    (directory / (route.stem + ".hkl")).write_bytes(request.hkl_path.read_bytes())
    chain = toolchain_factory(request)
    step = route.steps[0]
    adapter = chain.adapter(step.tool)
    plan = adapter.build_plan(args=step.args, cwd=directory, declared_inputs=step.declared_inputs,
                              expected_outputs=step.expected_outputs)
    return chain, plan


@pytest.mark.parametrize("changed", ["source_ins", "source_hkl", "staged_ins", "staged_hkl", "original_ins"])
def test_executor_rechecks_source_and_initial_stage_pins_after_construction(demo, changed):
    repo, request = demo
    chain, plan = _staged_initial(request, repo / "runs/staged")
    paths = {"source_ins": request.ins_path, "source_hkl": request.hkl_path,
             "staged_ins": plan.cwd / (plan.argv[1] + ".ins"),
             "staged_hkl": plan.cwd / (plan.argv[1] + ".hkl"),
             "original_ins": plan.cwd / "original.ins"}
    path = paths[changed]
    path.write_bytes(path.read_bytes() + b"\n")
    result = chain.executor.execute(plan, {})
    assert result.exit_code == 64
    assert "synthetic_provider_" in result.stderr
    assert not chain.executor.calls
    assert all(not (plan.cwd / name).exists() for name in plan.expected_outputs)


def _trial(request, directory):
    directory.mkdir()
    (directory / "trial.ins").write_text(synthetic_seed_text())
    (directory / "trial.hkl").write_bytes(request.hkl_path.read_bytes())
    chain = toolchain_factory(request)
    adapter = chain.adapter(ToolKind.SHELXL)
    plan = adapter.build_plan(args=("trial",), cwd=directory,
                              declared_inputs=("trial.ins", "trial.hkl"),
                              expected_outputs=("trial.res", "trial.lst"))
    return chain, plan


@pytest.mark.parametrize("change", ["reflections", "coordinates", "instructions", "extra_site", "mode_tool"])
def test_refinement_executor_cannot_process_arbitrary_marker_bearing_models(demo, change):
    repo, request = demo
    chain, plan = _trial(request, repo / "runs/trial")
    path = plan.cwd / "trial.ins"
    if change == "reflections":
        (plan.cwd / "trial.hkl").write_text("not built-in\n")
    elif change == "coordinates":
        path.write_text(path.read_text().replace("0.25000000", "0.26000000"))
    elif change == "instructions":
        path.write_text(path.read_text().replace("FVAR 1", "+external.ins\nFVAR 1"))
    elif change == "extra_site":
        path.write_text(path.read_text().replace("HKLF 4", "C9 1 0.1 0.2 0.3 11 0.03\nHKLF 4"))
    else:
        plan = replace(plan, tool_kind=ToolKind.SHELXC)
    result = chain.executor.execute(plan, {})
    assert result.exit_code == 64
    assert not (plan.cwd / "trial.res").exists()


def test_assigner_rechecks_originals_model_geometry_and_protocol_per_call(demo):
    repo, request = demo
    assigner = assigner_factory(request)
    model = repo / "runs/seed.res"
    model.write_text(synthetic_seed_text())
    elements = ("C", "H", "N", "O", "Zn")
    assignment = assigner.assign(model, candidate_elements=elements)
    assert assignment.provenance["synthetic"]
    assert not assignment.provenance["actual_pretrained_weights_executed"]
    with pytest.raises(ReconstructionBlocked, match="current_labels_only"):
        assigner.assign(model, candidate_elements=elements, input_protocol="upstream_ins")
    model.write_text(model.read_text().replace("0.25000000", "0.26000000"))
    with pytest.raises(ReconstructionBlocked, match="not_a_builtin_derivative"):
        assigner.assign(model, candidate_elements=elements)
    model.write_text(synthetic_seed_text())
    request.hkl_path.write_bytes(request.hkl_path.read_bytes() + b"\n")
    with pytest.raises(ReconstructionBlocked, match="not_pinned_builtin_data"):
        assigner.assign(model, candidate_elements=elements)


def test_demo_supports_fixed_metal_encoding_without_claiming_convergence(demo):
    repo, request = demo
    result = _service(repo).run(replace(request, metal_only=True))
    assert result["phase"] == "completed", result
    assert result["stop_code"] == "fixed_rounds_completed"
    assert result["automatic_stop_is_scientific_convergence"] is False
    model = read_res(Path(result["ai_solved_res"]))
    assert [a.label for a in model.atoms] == ["Zn1"]
    assert model.atoms[0].x == .25


def test_executor_and_backend_honor_precancel_without_writing_mock_outputs(demo):
    repo, request = demo
    chain, plan = _trial(request, repo / "runs/cancel-trial")
    cancel = CancelToken()
    cancel.cancel("demo cancellation")
    raw = chain.executor.execute(plan, {}, cancel=cancel)
    assert raw.cancelled and raw.exit_code is None
    assert not (plan.cwd / "trial.res").exists()
    result = _service(repo).run(request, cancel=cancel)
    assert result["phase"] == "cancelled" and result["ai_solved_res"] is None


def test_toolchain_advertises_only_supported_synthetic_routes(demo):
    _, request = demo
    chain = toolchain_factory(request)
    for kind in ToolKind:
        resolution = chain.resolution(kind)
        if kind in (ToolKind.SHELXT, ToolKind.SHELXS, ToolKind.SHELXD, ToolKind.SHELXL):
            assert resolution.is_usable
            assert resolution.identity.source is ResolutionSource.SYNTHETIC_DEMO
            assert "SYNTHETIC" in resolution.diagnostic.message
        else:
            assert not resolution.is_usable and chain.adapter(kind) is None
