"""Initial strategies run through real adapters with scripted, generated outputs only."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from conftest import write_hkl, write_ins
from crystalx_core.app.models import RunMode
from crystalx_core.app.mock_toolchain import _mock_identity
from crystalx_core.app.solve_routes import (
    InitialRouteConfig, InitialStrategy, PhasingDataset, ProteinPhasingConfig, RouteConfigError,
    execute_initial_route, plan_initial_route,
)
from crystalx_core.app.toolchain import Toolchain
from crystalx_core.domain.tools import (
    BinaryArch, HostPlatform, ToolAvailabilityReport, ToolDiagnostic, ToolDiagnosticKind,
    ToolKind, ToolResolution,
)
from crystalx_core.solvers.mock import ScriptedExecutor, ScriptedResponse
from crystalx_core.tools.runner import CancelToken, ProcessRunner
from crystalx_core.workspace.hashing import sha256_file


MODEL = """TITL Synthetic generated initial model
CELL 0.71073 10 10 10 90 90 90
LATT -1
SFAC C N O Se
UNIT 4 2 2 1
FVAR 1.0
C1 1 0.1 0.2 0.3 11.0 0.05
HKLF 4
END
"""


class ScriptedRouteToolchain(Toolchain):
    mode = RunMode.MOCK_DEMO

    def __init__(self, responses=(), unavailable=()):
        self.executor = ScriptedExecutor(responses)
        self.unavailable = set(unavailable)

    def availability(self):
        resolutions = {}
        for kind in (ToolKind.SHELXT, ToolKind.SHELXS, ToolKind.SHELXC, ToolKind.SHELXD, ToolKind.SHELXE):
            missing = kind in self.unavailable
            resolutions[kind.value] = ToolResolution(kind, None if missing else _mock_identity(kind),
                ToolDiagnostic(ToolDiagnosticKind.NOT_FOUND if missing else ToolDiagnosticKind.OK, "Synthetic contract fixture"))
        return ToolAvailabilityReport(HostPlatform("demo", BinaryArch.UNKNOWN), resolutions)

    def _runner(self):
        return ProcessRunner(self.executor, base_env={})


@pytest.fixture
def original(tmp_path):
    ins = write_ins(tmp_path / "original.ins", latt=-1, symm=(), sfac=("C", "N", "O", "Se"), unit=(4, 2, 2, 1))
    hkl = write_hkl(tmp_path / "original.hkl", include_friedel=True)
    return ins, hkl


def make_plan(original, tmp_path, config):
    return plan_initial_route(config, ins_path=original[0], hkl_path=original[1], work_dir=tmp_path / "route")


def response(files, **kwargs):
    return ScriptedResponse(creates=tuple(files), file_contents=files, **kwargs)


def protein_config(original, **changes):
    protein = ProteinPhasingConfig(True, "SAD", "P1", 1, "Se", 0.5,
        (PhasingDataset("SAD", original[1], friedel_pairs_retained=True),))
    return InitialRouteConfig(InitialStrategy.PROTEIN_CDE, protein=replace(protein, **changes))


def protein_responses(plan):
    s = plan.stem
    return [response({s + ".hkl": plan.hkl_path.read_text(), s + "_fa.ins": MODEL,
                      s + "_fa.hkl": "synthetic FA and alpha\n"}),
            response({s + "_fa.res": MODEL}),
            response({s + ".phs": "synthetic phases", s + ".pdb": "REMARK CC = 20.0\nATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C\nEND\n"}),
            response({s + "_i.phs": "synthetic inverted phases", s + "_i.pdb": "REMARK CC = 30.0\nATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C\nEND\n"})]


@pytest.mark.parametrize("strategy,options,tool", [
    (InitialStrategy.SHELXT, {}, ToolKind.SHELXT),
    (InitialStrategy.SMALL_MOLECULE, {}, ToolKind.SHELXS),
    (InitialStrategy.SMALL_MOLECULE, {"small_method": "patterson"}, ToolKind.SHELXS),
    (InitialStrategy.SMALL_MOLECULE, {"small_solver": "shelxd", "find_sites": 2, "plop": (3, 5)}, ToolKind.SHELXD),
])
def test_small_initial_routes_execute_existing_adapters_and_preserve_originals(original, tmp_path, strategy, options, tool):
    before = [path.read_bytes() for path in original]
    plan = make_plan(original, tmp_path, InitialRouteConfig(strategy, **options))
    assert not plan.blockers
    files = {plan.stem + ("_a.res" if tool is ToolKind.SHELXT else ".res"): MODEL}
    if tool is ToolKind.SHELXT:
        files[plan.stem + "_a.hkl"] = original[1].read_text()
    chain = ScriptedRouteToolchain([response(files)])
    result = execute_initial_route(plan, toolchain=chain)
    assert result.status == "INITIAL_MODEL_READY"
    assert result.seed_format == "shelx"
    assert result.seed_sha256 == sha256_file(result.seed_path)
    assert result.paired_hkl_sha256 == sha256_file(original[1])
    assert chain.executor.calls[0].tool_kind is tool
    assert result.invocations[0]["executable_source"] == "synthetic_demo"
    assert result.invocations[0]["inputs"][0]["sha256"] == sha256_file(plan.work_dir / (plan.stem + ".ins"))
    assert [path.read_bytes() for path in original] == before
    assert (plan.work_dir / "original.ins").read_bytes() == before[0]
    assert json.loads((plan.work_dir / "route-result.json").read_text())["assignment_performed"] is False
    if tool is ToolKind.SHELXS:
        keyword = "PATT" if options else "TREF"
        assert f"{keyword} 100" in (plan.work_dir / (plan.stem + ".ins")).read_text()
    if tool is ToolKind.SHELXD:
        assert "FIND 2\nPLOP 3 5\nNTRY 100" in (plan.work_dir / (plan.stem + ".ins")).read_text()


def test_protein_route_runs_c_d_and_both_e_hands_without_mislabeling_substructure(original, tmp_path):
    plan = make_plan(original, tmp_path, protein_config(original))
    assert not plan.blockers
    chain = ScriptedRouteToolchain(protein_responses(plan))
    result = execute_initial_route(plan, toolchain=chain)
    assert [call.tool_kind for call in chain.executor.calls] == [ToolKind.SHELXC, ToolKind.SHELXD, ToolKind.SHELXE, ToolKind.SHELXE]
    assert result.status == "INITIAL_ARTIFACTS_READY" and result.seed_format == "pdb"
    assert result.seed_path.name == plan.stem + "_i.pdb"
    assert result.evidence["hand"] == "inverted"
    assert result.evidence["hand_selection"] == "reported_trace_cc"
    assert result.evidence["substructure_is_not_full_seed"]
    assert result.evidence["cc_parser_format_verified"] is False
    assert result.blockers == ("pdb_to_shelxl_conversion_required",)
    script = chain.executor.calls[0].stdin_script
    assert "CELL " in script and "SPAG P1" in script and "SAD phasing_00.hkl" in script
    assert "FIND 1" in script and "NTRY 100" in script
    assert "-i" not in chain.executor.calls[2].argv and "-i" in chain.executor.calls[3].argv
    assert all("-x" not in call.argv for call in chain.executor.calls)


def test_protein_converter_gets_trace_and_generated_native_reflections_not_heavy_atom_res(original, tmp_path):
    plan = make_plan(original, tmp_path, protein_config(original))
    requests = []

    def convert(request):
        requests.append(request)
        assert request.pdb_path.suffix == ".pdb"
        assert request.paired_hkl_path.name == plan.stem + ".hkl"
        output = request.work_dir / "converted-seed.res"
        output.write_text(MODEL)
        return output

    result = execute_initial_route(plan, toolchain=ScriptedRouteToolchain(protein_responses(plan)), protein_converter=convert)
    assert result.status == "INITIAL_MODEL_READY" and result.seed_path.name == "converted-seed.res"
    assert len(requests) == 1 and requests[0].hand == "inverted"
    assert requests[0].declared_space_group == "P1"


def test_protein_missing_configuration_never_uses_generic_hkl_as_anomalous_data(original, tmp_path):
    plan = make_plan(original, tmp_path, InitialRouteConfig(InitialStrategy.PROTEIN_CDE))
    result = execute_initial_route(plan, toolchain=ScriptedRouteToolchain())
    assert result.status == "BLOCKED"
    assert "protein_phasing_configuration_missing" in result.blockers
    assert not plan.work_dir.exists()


@pytest.mark.parametrize("change,blocker", [
    ({"protein_confirmed": False}, "protein_context_not_declared"),
    ({"space_group": ""}, "protein_space_group_missing"),
    ({"find_sites": None}, "protein_substructure_sites_and_element_required"),
    ({"solvent_fraction": None}, "protein_solvent_fraction_missing"),
    ({"datasets": ()}, "sad_requires_explicit_anomalous_dataset"),
])
def test_protein_prerequisites_are_reported_per_route(original, tmp_path, change, blocker):
    plan = make_plan(original, tmp_path, protein_config(original, **change))
    assert blocker in plan.blockers
    chain = ScriptedRouteToolchain()
    assert execute_initial_route(plan, toolchain=chain).status == "BLOCKED"
    assert not chain.executor.calls


def test_merged_sad_data_blocks_even_if_declared_as_anomalous(original, tmp_path):
    write_hkl(original[1], include_friedel=False)
    plan = make_plan(original, tmp_path, protein_config(original))
    assert "anomalous_friedel_pairs_not_present:SAD" in plan.blockers


def test_mad_cannot_reuse_one_small_measurement_as_two_wavelengths(original, tmp_path):
    datasets = tuple(PhasingDataset(role, original[1], friedel_pairs_retained=True) for role in ("PEAK", "INFL"))
    plan = make_plan(original, tmp_path, protein_config(original, experiment="MAD", datasets=datasets))
    assert "mad_requires_distinct_wavelength_measurements" in plan.blockers


def test_distinct_pinned_datasets_with_explicit_mad_roles_are_planned(original, tmp_path):
    second = tmp_path / "infl.hkl"
    second.write_text(original[1].read_text() + "\n")
    datasets = (PhasingDataset("PEAK", original[1], friedel_pairs_retained=True),
                PhasingDataset("INFL", second, friedel_pairs_retained=True))
    plan = make_plan(original, tmp_path, protein_config(original, experiment="MAD", datasets=datasets))
    assert not plan.blockers
    assert "PEAK phasing_00.hkl" in plan.steps[0].stdin_script
    assert "INFL phasing_01.hkl" in plan.steps[0].stdin_script


def test_mtz_is_not_fed_to_shelxc_without_conversion(original, tmp_path):
    mtz = tmp_path / "data.mtz"
    mtz.write_bytes(b"synthetic MTZ not parsed")
    plan = make_plan(original, tmp_path, protein_config(original, datasets=(PhasingDataset("SAD", mtz, "MTZ", True),)))
    assert "mtz_to_supported_reflections_conversion_required" in plan.blockers


def test_small_shelxd_never_guesses_find_plop_or_cell_contents(original, tmp_path):
    plan = make_plan(original, tmp_path, InitialRouteConfig(InitialStrategy.SMALL_MOLECULE, small_solver="shelxd"))
    assert "shelxd_small_requires_explicit_find_and_plop_counts" in plan.blockers
    assert execute_initial_route(plan, toolchain=ScriptedRouteToolchain()).status == "BLOCKED"


def test_missing_executable_does_not_start_another_route_or_mutate_work_folder(original, tmp_path):
    plan = make_plan(original, tmp_path, InitialRouteConfig())
    chain = ScriptedRouteToolchain(unavailable=(ToolKind.SHELXT,))
    result = execute_initial_route(plan, toolchain=chain)
    assert result.blockers == ("tool_unavailable:shelxt",)
    assert not chain.executor.calls and not plan.work_dir.exists()


def test_failed_c_does_not_continue_to_d_or_fake_a_seed(original, tmp_path):
    plan = make_plan(original, tmp_path, protein_config(original))
    chain = ScriptedRouteToolchain([ScriptedResponse(exit_code=1)])
    result = execute_initial_route(plan, toolchain=chain)
    assert result.status == "FAILED" and result.seed_path is None
    assert len(chain.executor.calls) == 1


def test_cancellation_before_execution_starts_nothing(original, tmp_path):
    plan = make_plan(original, tmp_path, InitialRouteConfig())
    token = CancelToken()
    token.cancel()
    chain = ScriptedRouteToolchain()
    assert execute_initial_route(plan, toolchain=chain, cancel=token).status == "CANCELLED"
    assert not chain.executor.calls


def test_changed_source_input_never_reaches_external_program(original, tmp_path):
    plan = make_plan(original, tmp_path, InitialRouteConfig())
    original[1].write_text("changed source")
    chain = ScriptedRouteToolchain()
    result = execute_initial_route(plan, toolchain=chain)
    assert result.status == "BLOCKED" and not chain.executor.calls


def test_old_output_directory_is_never_reused(original, tmp_path):
    plan = make_plan(original, tmp_path, InitialRouteConfig())
    plan.work_dir.mkdir()
    old = plan.work_dir / "old-result.res"
    old.write_text("keep previous artifact")
    with pytest.raises(RouteConfigError, match="fresh empty"):
        execute_initial_route(plan, toolchain=ScriptedRouteToolchain())
    assert old.read_text() == "keep previous artifact"


@pytest.mark.parametrize("where", ["config", "protein", "dataset"])
def test_strategy_serializers_reject_unknown_fields_at_all_levels(original, where):
    data = protein_config(original).to_dict()
    target = data if where == "config" else data["protein"] if where == "protein" else data["protein"]["datasets"][0]
    target["expert_answer"] = "do-not-read.res"
    with pytest.raises(RouteConfigError):
        InitialRouteConfig.from_dict(data)


def test_strategy_serialization_roundtrip_preserves_explicit_roles_and_options(original):
    config = protein_config(original)
    assert InitialRouteConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


def test_shelxs_q_only_output_remains_provisional_sites_for_pretrained_assignment(original, tmp_path):
    plan = make_plan(original, tmp_path, InitialRouteConfig(InitialStrategy.SMALL_MOLECULE))
    q_model = MODEL.replace("C1 1 0.1 0.2 0.3 11.0 0.05", "Q1 1 0.1 0.2 0.3 11.0 0.05 2.0")
    result = execute_initial_route(plan, toolchain=ScriptedRouteToolchain([response({plan.stem + ".res": q_model})]))
    assert result.status == "INITIAL_MODEL_READY"
    assert result.evidence["n_atoms"] == 0 and result.evidence["n_peaks"] == 1
    assert result.evidence["element_assignments_are_provisional"]
    assert result.tool_calls == 1
    assert result.to_dict()["assignment_performed"] is False


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_generated_model_alias_is_rejected_before_output_parsing(original, tmp_path, alias):
    plan = make_plan(original, tmp_path, InitialRouteConfig(InitialStrategy.SMALL_MOLECULE))
    outside = tmp_path / "unrelated-private-model.res"
    outside.write_text(MODEL)

    class AliasingExecutor(ScriptedExecutor):
        def execute(self, command, env, **options):
            raw = super().execute(command, env, **options)
            target = command.cwd / (plan.stem + ".res")
            if alias == "symlink":
                target.symlink_to(outside)
            else:
                target.hardlink_to(outside)
            return raw

    chain = ScriptedRouteToolchain()
    chain.executor = AliasingExecutor()
    result = execute_initial_route(plan, toolchain=chain)
    assert result.status == "FAILED"
    assert result.tool_calls == 1
    assert "unsafe_or_unreadable_initial_solver_artifact" in result.blockers
    assert outside.read_text() == MODEL


def test_generated_output_cannot_exceed_32_mib(original, tmp_path):
    from crystalx_core.app.solve_routes import MAX_GENERATED_ARTIFACT_BYTES
    plan = make_plan(original, tmp_path, InitialRouteConfig(InitialStrategy.SMALL_MOLECULE))

    class OversizedExecutor(ScriptedExecutor):
        def execute(self, command, env, **options):
            raw = super().execute(command, env, **options)
            with (command.cwd / (plan.stem + ".res")).open("wb") as stream:
                stream.truncate(MAX_GENERATED_ARTIFACT_BYTES + 1)
            return raw

    chain = ScriptedRouteToolchain()
    chain.executor = OversizedExecutor()
    result = execute_initial_route(plan, toolchain=chain)
    assert result.status == "FAILED" and result.tool_calls == 1
    assert chain.executor.calls[0].limits.max_file_bytes <= MAX_GENERATED_ARTIFACT_BYTES


def test_small_solver_control_continuations_are_replaced_as_whole_instructions(original, tmp_path):
    original[0].write_text(original[0].read_text().replace("HKLF 4", "FIND 5\nPLOP 6 7 =\n    8 9\nNTRY 999\nHKLF 4"))
    plan = make_plan(original, tmp_path, InitialRouteConfig(InitialStrategy.SMALL_MOLECULE,
        small_solver="shelxd", find_sites=2, plop=(3, 5), trials=10))
    assert "PLOP 3 5\nNTRY 10" in plan.prepared_ins
    assert "    8 9" not in plan.prepared_ins
    assert "NTRY 999" not in plan.prepared_ins


def test_only_one_failed_shelxe_hand_does_not_discard_the_other_automatic_trace(original, tmp_path):
    plan = make_plan(original, tmp_path, protein_config(original))
    responses = protein_responses(plan)
    responses[2] = ScriptedResponse(exit_code=1)
    result = execute_initial_route(plan, toolchain=ScriptedRouteToolchain(responses))
    assert result.status == "INITIAL_ARTIFACTS_READY"
    assert result.evidence["hand"] == "inverted"
    assert result.tool_calls == 4


def test_trace_cc_can_be_read_from_retained_shelxe_stdout_when_pdb_has_no_score(original, tmp_path):
    plan = make_plan(original, tmp_path, protein_config(original))
    responses = protein_responses(plan)
    for response_item, suffix, score in ((responses[2], "", 35.0), (responses[3], "_i", 25.0)):
        name = plan.stem + suffix + ".pdb"
        response_item.file_contents[name] = "\n".join(response_item.file_contents[name].splitlines()[1:])
        response_item.stdout = f"CC for partial structure against native data = {score}%\n"
    result = execute_initial_route(plan, toolchain=ScriptedRouteToolchain(responses))
    assert result.evidence["hand"] == "original"
    assert result.evidence["trace_cc"]["original"] == 35.0
    assert result.evidence["trace_cc_sources"]["original"].endswith(".stdout.log")
    assert result.evidence["hand_selection_is_provisional"]


def test_protein_converter_output_cannot_alias_an_external_or_expert_file(original, tmp_path):
    plan = make_plan(original, tmp_path, protein_config(original))
    outside = tmp_path / "external-model.res"
    outside.write_text(MODEL)

    def convert(request):
        pointer = request.work_dir / "converted.res"
        pointer.symlink_to(outside)
        return pointer

    result = execute_initial_route(plan, toolchain=ScriptedRouteToolchain(protein_responses(plan)), protein_converter=convert)
    assert result.status == "FAILED"
    assert "protein_converter_output_not_contained" in result.blockers


def test_whole_chain_deadline_is_not_reset_between_c_d_and_e(original, tmp_path):
    plan = make_plan(original, tmp_path, protein_config(original))
    now = [0.0]

    class AdvancingExecutor(ScriptedExecutor):
        def execute(self, command, env, **options):
            raw = super().execute(command, env, **options)
            now[0] += 4
            return raw

    chain = ScriptedRouteToolchain()
    chain.executor = AdvancingExecutor(protein_responses(plan))
    result = execute_initial_route(plan, toolchain=chain, deadline=7.0, clock=lambda: now[0])
    assert result.status == "FAILED"
    assert result.blockers == ("initial_route_deadline_exhausted",)
    assert [call.timeout_s for call in chain.executor.calls] == [7.0, 3.0]
    assert result.tool_calls == 2
    assert len(result.invocations) == 2


def test_expired_whole_chain_deadline_starts_no_tool_or_new_artifacts(original, tmp_path):
    plan = make_plan(original, tmp_path, InitialRouteConfig())
    chain = ScriptedRouteToolchain()
    result = execute_initial_route(plan, toolchain=chain, deadline=3.0, clock=lambda: 4.0)
    assert result.status == "BLOCKED"
    assert result.tool_calls == 0 and not chain.executor.calls
    assert not plan.work_dir.exists()


@pytest.mark.parametrize("behavior", ["exception", "none", "malformed_model"])
def test_converter_failure_retains_all_raw_protein_artifacts(original, tmp_path, behavior):
    plan = make_plan(original, tmp_path, protein_config(original))

    def convert(request):
        if behavior == "exception":
            raise RuntimeError("private exception text must not become a public report")
        if behavior == "none":
            return None
        path = request.work_dir / "bad-conversion.res"
        path.write_text("not a structure")
        return path

    result = execute_initial_route(plan, toolchain=ScriptedRouteToolchain(protein_responses(plan)), protein_converter=convert)
    assert result.status == "FAILED" and result.tool_calls == 4
    assert (plan.work_dir / (plan.stem + "_fa.res")).is_file()
    assert (plan.work_dir / (plan.stem + ".pdb")).is_file()
    assert (plan.work_dir / (plan.stem + "_i.pdb")).is_file()
    assert "private exception text" not in json.dumps(result.to_dict())


def test_conversion_also_respects_shared_deadline(original, tmp_path):
    plan = make_plan(original, tmp_path, protein_config(original))
    now = [0.0]

    def convert(request):
        assert request.deadline == 5.0 and request.cancel is not None
        output = request.work_dir / "converted.res"
        output.write_text(MODEL)
        now[0] = 6.0
        return output

    result = execute_initial_route(plan, toolchain=ScriptedRouteToolchain(protein_responses(plan)),
        protein_converter=convert, deadline=5.0, clock=lambda: now[0])
    assert result.status == "FAILED" and result.blockers == ("initial_route_deadline_exhausted",)
    assert result.tool_calls == 4


@pytest.mark.parametrize("options", [{"trials": 0}, {"timeout_s": float("nan")}, {"trials": True}, {"find_sites": -1}, {"plop": (1, 0)}])
def test_route_resource_and_trial_values_are_bounded(options):
    with pytest.raises(RouteConfigError):
        InitialRouteConfig(**options)
