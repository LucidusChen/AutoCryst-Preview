"""The application service: upload to result, failure, cancellation, isolation, durability, safety.

Every run here uses either the mock toolchain on the generated synthetic demo case, or the real
toolchain pointed at an empty or synthetic toolbox -- so a real resolver and a real process runner
are exercised, but no crystallographic program exists to be launched. The one synthetic
"executable" is a header-only Mach-O file written by the test, used to prove that an operating
system's refusal to execute is classified rather than reported as a missing file.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from conftest import CPU_X86_64, macho_thin, write_executable, write_hkl, write_ins
from crystalx_core.app import (
    ChemicalPriorInput,
    ModelAvailability,
    RefinementStatus,
    RequestInvalid,
    RunConfig,
    RunMode,
    RunPhase,
    RunService,
    SolveRequest,
    StageState,
    UnknownRun,
    write_demo_case,
)
from crystalx_core.app.demo import DEMO_MARKER, demo_case_texts, demo_digests, is_demo_input
from crystalx_core.app.mock_toolchain import MockToolchain, MockToolchainExecutor
from crystalx_core.app.pipeline import _launch_refusal_classification
from crystalx_core.app.staging import (
    CompositionOverrideError,
    apply_composition_override,
    safe_stem,
)
from crystalx_core.app.toolchain import RealToolchain
from crystalx_core.domain.tools import BinaryArch, HostPlatform, ToolKind
from crystalx_core.tools.resolver import ResolverConfig, ToolResolver
from crystalx_core.tools.runner import ExecutionPlan
from crystalx_core.workspace.store import RunWorkspace

FAST = 0.05


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def empty_resolver(tmp_path: Path, *, host: HostPlatform | None = None) -> ToolResolver:
    toolbox = tmp_path / "toolbox"
    toolbox.mkdir(exist_ok=True)
    return ToolResolver(
        ResolverConfig(
            toolbox_dir=toolbox, search_path=[], include_known_locations=False, include_olex2=False
        ),
        host=host,
        quarantine_checker=lambda _p: False,
    )


def make_service(
    tmp_path: Path, *, resolver: ToolResolver | None = None, step_seconds: float = FAST, **mock
) -> RunService:
    def factory(request: SolveRequest):
        if request.mode is RunMode.MOCK_DEMO:
            return MockToolchain(step_seconds=step_seconds, **mock)
        return RealToolchain(resolver=resolver or empty_resolver(tmp_path))

    return RunService(tmp_path / "runs", toolchain_factory=factory)


def demo_request(tmp_path: Path, *, prior: ChemicalPriorInput | None = None, **config) -> SolveRequest:
    ins, hkl = write_demo_case(tmp_path / "demo_input")
    return SolveRequest(
        ins_path=ins,
        hkl_path=hkl,
        prior=prior or ChemicalPriorInput(metals=("Zn",)),
        config=RunConfig(mock_step_seconds=FAST, **config),
        mode=RunMode.MOCK_DEMO,
        label="demo",
    )


def stage_states(state: dict) -> dict[str, str]:
    return {s["name"]: s["state"] for s in state["stages"]}


def wait_until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached in time")


# --------------------------------------------------------------------------------------
# the synthetic demo case
# --------------------------------------------------------------------------------------


def test_the_demo_case_is_deterministic_and_recognised_exactly(tmp_path: Path):
    ins, hkl = write_demo_case(tmp_path)
    assert is_demo_input(ins, hkl)
    assert hashlib.sha256(ins.read_bytes()).hexdigest() == demo_digests()[0]
    assert DEMO_MARKER in ins.read_text()
    altered = tmp_path / "altered.hkl"
    altered.write_bytes(hkl.read_bytes().replace(b"   0   0   0", b"   0   0   1", 1))
    assert not is_demo_input(ins, altered)


def test_writing_the_demo_never_overwrites_a_different_file(tmp_path: Path):
    (tmp_path / "crystalx_demo.ins").write_text("TITL someone else's file\n")
    with pytest.raises(FileExistsError):
        write_demo_case(tmp_path)
    assert (tmp_path / "crystalx_demo.ins").read_text() == "TITL someone else's file\n"


def test_the_demo_reflections_are_a_plausible_merged_p21_dataset():
    ins_text, hkl_text = demo_case_texts()
    assert "LATT -1" in ins_text and "SYMM -X, 1/2+Y, -Z" in ins_text
    rows = hkl_text.splitlines()
    assert rows[-1].split() == ["0", "0", "0", "0.00", "0.00"]
    assert len(rows) > 1000


# --------------------------------------------------------------------------------------
# upload to result
# --------------------------------------------------------------------------------------


def test_mock_demo_runs_from_upload_to_result(tmp_path: Path):
    service = make_service(tmp_path)
    handle = service.submit(demo_request(tmp_path))
    state = service.wait(handle.run_id, timeout=30)

    assert state["phase"] == RunPhase.COMPLETED.value
    assert set(stage_states(state).values()) == {StageState.SUCCEEDED.value}
    summary = service.summary(handle.run_id)
    assert summary["mock"] is True and summary["mode"] == "mock_demo"
    assert summary["model_availability"] == ModelAvailability.PRELIMINARY_REFINED_MODEL.value
    assert summary["refinement_status"] == RefinementStatus.RAN_CONVERGENCE_NOT_ASSESSED.value
    # the four facts stay separate: a refinement that ran is still not eligible for ranking
    assert summary["scientific_eligibility"]["eligible_for_ranking"] is False
    assert summary["scientific_eligibility"]["status"] == "not_evaluated"
    listing = summary["refinement"]["listing"]
    assert listing["r1_observed"] == pytest.approx(0.0412)
    assert listing["cycles_run"] == 4
    assert summary["refinement"]["agreement"]["disagreements"] == []
    assert summary["platon"]["report"]["counts"] == {"A": 0, "B": 1, "C": 1, "G": 1}
    assert all(a["token"].startswith("MOCK") for a in summary["platon"]["report"]["alerts"])

    refined = service.viewer_payload(handle.run_id, "refined")
    assert refined["mock"] is True
    assert refined["counts"]["boundary_bonds_not_drawn"] == 1
    assert refined["boundary_bonds"][0]["a"] == "C5"
    assert refined["confidence"]["available"] is False

    root = handle.run_dir
    assert (root / "result.json").is_file()
    assert json.loads((root / "result.json").read_text())["stop_reason"]["kind"] == "review_checkpoint"
    provenance = json.loads((root / "refinement/shelxl-01/provenance.json").read_text())
    assert provenance["executable_source"] == "synthetic_demo"
    assert provenance["limits"]["cpu_seconds"] == RunConfig().limits_for(900.0).cpu_seconds
    shelxt = json.loads(
        (root / "stages/s1-initial_solution/branches/b00-shelxt/provenance_00.json").read_text()
    )
    assert shelxt["limits"]["cpu_seconds"] == RunConfig().limits_for(3600.0).cpu_seconds
    assert shelxt["limits"]["max_file_bytes"] == 2 * 1024**3


def test_cpu_limits_scale_with_the_wall_budget_and_the_cpu_count():
    cores = os.cpu_count() or 1
    assert RunConfig().limits_for(100.0).cpu_seconds == 100 * cores
    assert RunConfig(cpu_seconds=50).limits_for(100.0).cpu_seconds == 50
    with pytest.raises(ValueError):
        RunConfig(cpu_seconds=0)


def test_the_shelxt_input_is_passed_through_verbatim(tmp_path: Path):
    """LATT/SYMM reach SHELXT unchanged: it takes the Laue group from them."""
    service = make_service(tmp_path)
    handle = service.submit(demo_request(tmp_path))
    service.wait(handle.run_id, timeout=30)
    branch = handle.run_dir / "stages/s1-initial_solution/branches/b00-shelxt"
    staging = json.loads((branch / "staging.json").read_text())
    assert staging["ins_is_verbatim"] is True and staging["changes"] == []
    assert (branch / "crystalx_demo.ins").read_bytes() == demo_case_texts()[0].encode()


def test_the_refinement_uses_shelxt_reflections_and_records_every_change(tmp_path: Path):
    service = make_service(tmp_path)
    handle = service.submit(demo_request(tmp_path, shelxl_cycles=3))
    service.wait(handle.run_id, timeout=30)
    prepared = json.loads((handle.run_dir / "refinement/shelxl-01/prepared.json").read_text())
    assert prepared["reflections"].endswith("crystalx_demo_a.hkl")
    assert any(change.startswith("inserted control block: L.S. 3") for change in prepared["changes"])
    ins = (handle.run_dir / "refinement/shelxl-01/crystalx_demo.ins").read_text()
    assert "LIST 4" in ins and "ACTA" in ins and "ANIS" not in ins


def test_preview_diagnoses_and_plans_without_writing_anything(tmp_path: Path):
    service = make_service(tmp_path)
    preview = service.preview(demo_request(tmp_path))
    assert preview["preconditions"]["may_start"] is True
    assert [n["solver"] for n in preview["plan"]["nominated"] if n["will_run"]] == ["shelxt"]
    assert preview["diagnosis"]["reflections"]["count"] > 1000
    assert [p for p in (tmp_path / "runs").iterdir() if p.name.startswith("r-")] == []


def test_heavy_atom_routes_stay_plan_only_in_the_plan(tmp_path: Path):
    """With every tool reported usable, a heavy-atom route is nominated but marked plan-only."""
    from crystalx_core.app.pipeline import build_registry, plan_display
    from crystalx_core.domain.data import DataDiagnosis
    from crystalx_core.domain.radiation import RadiationAssumption
    from crystalx_core.io.shelx_reader import read_ins_header, scan_hkl
    from crystalx_core.solvers.context import SolverContext, StagedCase
    from crystalx_core.solvers.selector import StrategySelector

    ins = write_ins(tmp_path / "x.ins")
    hkl = write_hkl(tmp_path / "x.hkl")
    header = read_ins_header(ins)
    diagnosis = DataDiagnosis(reflections=scan_hkl(hkl, cell=header.cell), ins=header)
    prior = ChemicalPriorInput(metals=("Ag",)).to_prior()
    trace = StrategySelector().select(diagnosis, prior, stage_id="s1-initial_solution")
    context = SolverContext(
        case=StagedCase("x"), diagnosis=diagnosis, prior=prior,
        radiation=RadiationAssumption.from_wavelength(0.71073),
    )
    display = plan_display(trace, build_registry(context, RunConfig()), RunConfig())
    routes = {n["solver"]: n for n in display["nominated"]}
    assert routes["shelxs_patt"]["readiness"] == "plan_only"
    assert routes["shelxs_patt"]["will_run"] is False
    assert routes["shelxt"]["will_run"] is True


# --------------------------------------------------------------------------------------
# failures
# --------------------------------------------------------------------------------------


def test_a_failed_refinement_keeps_the_solution_model_and_says_so(tmp_path: Path):
    service = make_service(tmp_path, fail_tool=ToolKind.SHELXL)
    handle = service.submit(demo_request(tmp_path))
    state = service.wait(handle.run_id, timeout=30)
    stages = stage_states(state)
    assert state["phase"] == RunPhase.FAILED.value
    assert stages["shelxt"] == "succeeded"
    assert stages["shelxl"] == "failed"
    assert stages["platon"] == "skipped"
    summary = service.summary(handle.run_id)
    assert summary["model_availability"] == ModelAvailability.SOLUTION_MODEL.value
    assert summary["refinement_status"] == RefinementStatus.FAILED.value
    assert "injected failure" in summary["refinement"]["diagnostic_message"]
    assert service.viewer_payload(handle.run_id, "solution") is not None
    assert service.viewer_payload(handle.run_id, "refined") is None


def test_a_clean_exit_without_a_solution_is_a_failure(tmp_path: Path):
    service = make_service(tmp_path, fail_tool=ToolKind.SHELXT, fail_mode="no_output")
    handle = service.submit(demo_request(tmp_path))
    state = service.wait(handle.run_id, timeout=30)
    assert state["phase"] == RunPhase.FAILED.value
    assert stage_states(state)["shelxt"] == "failed"
    summary = service.summary(handle.run_id)
    assert summary["model_availability"] == ModelAvailability.NONE.value
    assert "did not produce" in summary["process"]["message"]


def test_a_program_that_exceeds_its_time_budget_is_stopped(tmp_path: Path):
    service = make_service(tmp_path, fail_tool=ToolKind.SHELXT, fail_mode="hang")
    handle = service.submit(demo_request(tmp_path, shelxt_timeout_s=0.4))
    state = service.wait(handle.run_id, timeout=30)
    assert state["phase"] == RunPhase.FAILED.value
    assert "budget" in service.summary(handle.run_id)["process"]["message"]


def test_platon_is_blocked_when_the_fcf_is_missing(tmp_path: Path):
    service = make_service(tmp_path, omit_suffixes=(".fcf",))
    handle = service.submit(demo_request(tmp_path))
    state = service.wait(handle.run_id, timeout=30)
    assert state["phase"] == RunPhase.COMPLETED.value
    assert stage_states(state)["platon"] == "blocked"
    reasons = " ".join(service.summary(handle.run_id)["platon"]["reasons"])
    assert "external SHELXL" in reasons
    assert not (handle.run_dir / "validation").exists()


def test_a_missing_tool_blocks_the_run_before_anything_starts(tmp_path: Path):
    service = make_service(tmp_path)
    ins, hkl = write_ins(tmp_path / "c.ins"), write_hkl(tmp_path / "c.hkl")
    handle = service.submit(SolveRequest(ins_path=ins, hkl_path=hkl, label="real"))
    state = service.wait(handle.run_id, timeout=30)
    assert state["phase"] == RunPhase.BLOCKED.value
    assert stage_states(state)["shelxt"] == StageState.TOOL_UNAVAILABLE.value
    assert "not found" in state["message"]
    assert not (handle.run_dir / "refinement").exists()


def test_an_x86_64_build_without_rosetta_is_explained_not_attempted(tmp_path: Path):
    host = HostPlatform(system="Darwin", arch=BinaryArch.ARM64, rosetta_available=False)
    resolver = empty_resolver(tmp_path, host=host)
    write_executable(tmp_path / "toolbox" / "shelxt", macho_thin(CPU_X86_64))
    service = make_service(tmp_path, resolver=resolver)
    ins, hkl = write_ins(tmp_path / "c.ins"), write_hkl(tmp_path / "c.hkl")
    handle = service.submit(SolveRequest(ins_path=ins, hkl_path=hkl, label="real"))
    state = service.wait(handle.run_id, timeout=30)
    assert state["phase"] == RunPhase.BLOCKED.value
    assert "Rosetta 2" in state["message"]
    assert "softwareupdate --install-rosetta" in state["message"]


@pytest.mark.skipif(os.name != "posix", reason="launch refusal is exercised through POSIX exec")
def test_an_operating_system_refusal_to_execute_is_classified(tmp_path: Path):
    """Static inspection passes, the launch is refused: the diagnosis must say why."""
    host = HostPlatform(system="Linux", arch=BinaryArch.X86_64)
    resolver = empty_resolver(tmp_path, host=host)
    write_executable(tmp_path / "toolbox" / "shelxt", macho_thin(CPU_X86_64))
    service = make_service(tmp_path, resolver=resolver)
    ins, hkl = write_ins(tmp_path / "c.ins"), write_hkl(tmp_path / "c.hkl")
    handle = service.submit(SolveRequest(ins_path=ins, hkl_path=hkl, label="real"))
    state = service.wait(handle.run_id, timeout=30)
    assert state["phase"] == RunPhase.FAILED.value
    message = service.summary(handle.run_id)["process"]["message"]
    assert "architecture problem" in message
    assert "classification: binary format" in message
    outcome = json.loads(
        (handle.run_dir / "stages/s1-initial_solution/branches/b00-shelxt/outcome.json").read_text()
    )
    assert outcome["diagnostic_code"] == "arch_mismatch"


def test_a_refused_launch_on_apple_silicon_names_host_translation():
    class _Toolchain:
        def availability(self):
            from crystalx_core.domain.tools import ToolAvailabilityReport

            return ToolAvailabilityReport(
                host=HostPlatform(system="Darwin", arch=BinaryArch.ARM64, rosetta_available=True)
            )

    text = _launch_refusal_classification(_Toolchain())
    assert text.startswith("classification: host translation")
    assert "marker file is present" in text


# --------------------------------------------------------------------------------------
# scientific boundaries are not relaxed for the demo
# --------------------------------------------------------------------------------------


def test_an_impossible_declared_group_is_an_unresolved_start_for_shelxt_not_a_pass(tmp_path: Path):
    """Route context: SHELXT searches the Laue group, so the declaration does not block the solve;
    it is recorded, and the group SHELXT chooses is checked before refinement (next test)."""
    service = make_service(tmp_path)
    ins = write_ins(tmp_path / "c.ins", latt=1)  # centrosymmetric declaration
    hkl = write_hkl(tmp_path / "c.hkl")
    handle = service.submit(
        SolveRequest(ins_path=ins, hkl_path=hkl, prior=ChemicalPriorInput(peptide=True), label="pep")
    )
    state = service.wait(handle.run_id, timeout=30)
    assert stage_states(state)["preconditions"] == "succeeded"
    items = service.summary(handle.run_id)["preconditions"]["items"]
    assert [(i["kind"], i["blocking"]) for i in items] == [("chirality_unresolved_start", False)]
    assert stage_states(state)["shelxt"] == StageState.TOOL_UNAVAILABLE.value  # reached SHELXT


def test_a_shelxt_space_group_impossible_for_the_declared_chemistry_is_not_refined(tmp_path: Path):
    service = make_service(tmp_path, shelxt_centrosymmetric=True)
    request = demo_request(tmp_path, prior=ChemicalPriorInput(metals=("Zn",), peptide=True))
    handle = service.submit(request)
    state = service.wait(handle.run_id, timeout=30)
    assert state["phase"] == RunPhase.COMPLETED.value
    assert stage_states(state)["shelxl"] == "blocked"
    summary = service.summary(handle.run_id)
    assert summary["model_availability"] == ModelAvailability.SOLUTION_MODEL.value
    assert "homochiral" in summary["refinement"]["reason"]
    assert not (handle.run_dir / "refinement").exists()


def test_the_mock_toolchain_refuses_anything_but_the_demo(tmp_path: Path):
    service = make_service(tmp_path)
    ins, hkl = write_ins(tmp_path / "c.ins"), write_hkl(tmp_path / "c.hkl")
    with pytest.raises(RequestInvalid) as caught:
        service.submit(SolveRequest(ins_path=ins, hkl_path=hkl, mode=RunMode.MOCK_DEMO))
    assert caught.value.issues[0].code == "mock_requires_demo"

    work = tmp_path / "work"
    work.mkdir()
    (work / "x.ins").write_text(f"TITL {DEMO_MARKER}\n")
    (work / "x.hkl").write_text("   1   0   0    1.00    1.00\n")
    plan = ExecutionPlan(argv=("<mock-shelxt>", "x"), cwd=work, tool_kind=ToolKind.SHELXT)
    raw = MockToolchainExecutor(step_seconds=FAST).execute(plan, {})
    assert raw.exit_code == 64 and "not the synthetic demo" in raw.stderr
    assert sorted(p.name for p in work.iterdir()) == ["x.hkl", "x.ins"]


# --------------------------------------------------------------------------------------
# cancellation
# --------------------------------------------------------------------------------------


def test_cancelling_a_running_program_stops_it_and_keeps_the_record(tmp_path: Path):
    service = make_service(tmp_path, step_seconds=30.0)
    handle = service.submit(demo_request(tmp_path))
    # wait until the program has really started and written output, not merely been staged
    wait_until(lambda: "[mock] program" in service.log_tail(handle.run_id)[1])
    started = time.monotonic()
    assert service.cancel(handle.run_id)
    state = service.wait(handle.run_id, timeout=10)
    assert time.monotonic() - started < 5
    stages = stage_states(state)
    assert state["phase"] == RunPhase.CANCELLED.value
    assert stages["shelxt"] == "cancelled"
    assert stages["shelxl"] == stages["platon"] == "skipped"
    outcome = json.loads(
        (handle.run_dir / "stages/s1-initial_solution/branches/b00-shelxt/outcome.json").read_text()
    )
    assert outcome["state"] == "aborted" and outcome["diagnostic_code"] == "cancelled"
    log = handle.run_dir / "stages/s1-initial_solution/branches/b00-shelxt/shelxt.stdout.log"
    assert "stopped on request" in log.read_text()
    assert json.loads((handle.run_dir / "result.json").read_text())["stop_reason"]["kind"] == "user_aborted"


def test_a_run_cancelled_before_its_program_starts_records_that_nothing_ran(tmp_path: Path):
    """Cancelling between staging and launch: the runner refuses to start the program at all."""
    service = make_service(tmp_path, step_seconds=30.0)
    handle = service.submit(demo_request(tmp_path))
    wait_until(lambda: service.status(handle.run_id).get("current_log"))
    service.cancel(handle.run_id)
    state = service.wait(handle.run_id, timeout=10)
    assert state["phase"] == RunPhase.CANCELLED.value
    result = json.loads(
        (handle.run_dir / "stages/s1-initial_solution/branches/b00-shelxt/result_00.json").read_text()
    )
    assert result["diagnostic"] == "cancelled"


def test_a_queued_run_can_be_cancelled_before_it_starts(tmp_path: Path):
    service = make_service(tmp_path, step_seconds=30.0)
    first = service.submit(demo_request(tmp_path))
    wait_until(lambda: service.status(first.run_id).get("current_log"))
    second = service.submit(demo_request(tmp_path))
    assert service.status(second.run_id)["phase"] == RunPhase.QUEUED.value
    assert service.cancel(second.run_id)
    state = service.wait(second.run_id, timeout=10)
    assert state["phase"] == RunPhase.CANCELLED.value
    assert set(stage_states(state).values()) == {"skipped"}
    assert not (second.run_dir / "stages" / "s1-initial_solution").exists()
    service.cancel(first.run_id)
    assert service.wait(first.run_id, timeout=10)["phase"] == RunPhase.CANCELLED.value


def test_shutdown_cancels_active_runs(tmp_path: Path):
    service = make_service(tmp_path, step_seconds=30.0)
    handle = service.submit(demo_request(tmp_path))
    wait_until(lambda: service.status(handle.run_id).get("current_log"))
    service.shutdown(timeout=10)
    assert service.status(handle.run_id)["phase"] == RunPhase.CANCELLED.value


# --------------------------------------------------------------------------------------
# isolation, immutability and durability
# --------------------------------------------------------------------------------------


def test_source_files_are_never_modified(tmp_path: Path):
    service = make_service(tmp_path)
    request = demo_request(tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (request.ins_path, request.hkl_path)}
    handle = service.submit(request)
    service.wait(handle.run_id, timeout=30)
    for path, (data, mtime) in before.items():
        assert path.read_bytes() == data and path.stat().st_mtime_ns == mtime
    for staged in (handle.run_dir / "inputs").glob("crystalx_demo.*"):
        assert stat.S_IMODE(staged.stat().st_mode) == 0o444
    checks = service.summary(handle.run_id)["input_verification"]
    assert all(c["original_unchanged"] and c["staged_copy_unchanged"] for c in checks.values())


def test_runs_are_isolated_from_each_other(tmp_path: Path):
    service = make_service(tmp_path)
    a = service.submit(demo_request(tmp_path))
    service.wait(a.run_id, timeout=30)
    b = service.submit(demo_request(tmp_path))
    service.wait(b.run_id, timeout=30)
    assert a.run_id != b.run_id and a.run_dir != b.run_dir
    for handle in (a, b):
        artefacts = service.artefacts(handle.run_id)
        assert all(not name.startswith("..") for name in artefacts)
        assert (handle.run_dir / "refinement/shelxl-01/crystalx_demo.res").is_file()
    assert service.summary(a.run_id)["run_id"] == a.run_id


def test_a_run_left_running_by_a_dead_service_is_reported_interrupted(tmp_path: Path):
    workspace = RunWorkspace.create(tmp_path / "runs", label="left")
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    state = {"phase": "running", "message": "SHELXT", "service": {"pid": process.pid, "host": socket.gethostname()}}
    workspace.write_json("state.json", state)
    before = (workspace.layout.root / "state.json").read_bytes()

    status = RunService(tmp_path / "runs").status(str(workspace.run_id))
    assert status["phase"] == RunPhase.INTERRUPTED.value
    assert status["recorded_phase"] == "running"
    assert (workspace.layout.root / "state.json").read_bytes() == before


def test_a_run_owned_by_another_live_process_is_not_called_interrupted(tmp_path: Path):
    workspace = RunWorkspace.create(tmp_path / "runs", label="elsewhere")
    workspace.write_json(
        "state.json",
        {"phase": "running", "service": {"pid": os.getppid(), "host": socket.gethostname()}},
    )
    status = RunService(tmp_path / "runs").status(str(workspace.run_id))
    assert status["phase"] == "running"
    assert "another process" in status["note"]


def test_a_new_service_reads_finished_runs_from_disk(tmp_path: Path):
    service = make_service(tmp_path)
    handle = service.submit(demo_request(tmp_path))
    service.wait(handle.run_id, timeout=30)
    fresh = RunService(tmp_path / "runs")
    assert fresh.status(handle.run_id)["phase"] == RunPhase.COMPLETED.value
    assert [row["run_id"] for row in fresh.list_runs()] == [handle.run_id]
    kinds = {e["kind"] for e in fresh.events(handle.run_id)}
    assert {"run.started", "app.stage", "command.started", "command.finished", "run.finished"} <= kinds


# --------------------------------------------------------------------------------------
# downloads
# --------------------------------------------------------------------------------------


def test_downloads_are_restricted_to_the_runs_own_artefacts(tmp_path: Path):
    service = make_service(tmp_path)
    handle = service.submit(demo_request(tmp_path))
    service.wait(handle.run_id, timeout=30)
    good = "refinement/shelxl-01/crystalx_demo.cif"
    assert service.artefact_path(handle.run_id, good) == handle.run_dir / good
    for bad in ("../../etc/passwd", "/etc/passwd", "inputs/crystalx_demo.hkl", "refinement/../run.json",
                "refinement\\shelxl-01\\crystalx_demo.cif", ""):
        with pytest.raises(PermissionError):
            service.artefact_path(handle.run_id, bad)
    for bad_id in ("../x", "r-20260101T000000Z-zzzzzzzz", "", "r-20260101T000000Z-1234567"):
        with pytest.raises(UnknownRun):
            service.status(bad_id)


def test_the_export_bundle_contains_this_run_only(tmp_path: Path):
    service = make_service(tmp_path)
    handle = service.submit(demo_request(tmp_path))
    service.wait(handle.run_id, timeout=30)
    bundle = service.export_bundle(handle.run_id)
    assert bundle.parent == tmp_path / "runs" / "_bundles"
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
    prefix = handle.run_dir.name + "/"
    assert names and all(name.startswith(prefix) for name in names)
    assert prefix + "summary.json" in names
    assert not any(".." in name for name in names)


def test_an_unfinished_run_cannot_be_exported(tmp_path: Path):
    service = make_service(tmp_path, step_seconds=30.0)
    handle = service.submit(demo_request(tmp_path))
    wait_until(lambda: service.status(handle.run_id).get("current_log"))
    with pytest.raises(RuntimeError):
        service.export_bundle(handle.run_id)
    service.cancel(handle.run_id)
    service.wait(handle.run_id, timeout=10)


# --------------------------------------------------------------------------------------
# requests and staging
# --------------------------------------------------------------------------------------


def test_bad_requests_are_refused_with_stable_codes(tmp_path: Path):
    service = make_service(tmp_path)
    ins = write_ins(tmp_path / "c.ins")
    wrong = tmp_path / "c.txt"
    wrong.write_text("x")
    with pytest.raises(RequestInvalid) as caught:
        service.submit(
            SolveRequest(
                ins_path=ins,
                hkl_path=wrong,
                prior=ChemicalPriorInput(metals=("Xx",), composition=(("C", 0.0),)),
            )
        )
    codes = {issue.code for issue in caught.value.issues}
    assert codes == {"wrong_extension", "unknown_element", "non_positive_count"}


def test_a_composition_override_changes_only_the_shelxt_copy(tmp_path: Path):
    service = make_service(tmp_path)
    composition = ChemicalPriorInput.parse_composition("C 10 H 28 N 4 O 12 Zn 2 Cl 1")
    request = demo_request(tmp_path, prior=ChemicalPriorInput(metals=("Zn",), composition=composition))
    handle = service.submit(request)
    state = service.wait(handle.run_id, timeout=30)
    assert state["phase"] == RunPhase.COMPLETED.value
    staging = json.loads(
        (handle.run_dir / "stages/s1-initial_solution/branches/b00-shelxt/staging.json").read_text()
    )
    assert staging["ins_is_verbatim"] is False
    assert staging["changes"][0] == "SFAC C H N O Zn -> SFAC C H N O Zn Cl"
    assert any("composition override" in w for w in service.summary(handle.run_id)["warnings"])
    assert request.ins_path.read_bytes() == demo_case_texts()[0].encode()


def test_composition_override_refuses_what_it_cannot_do_safely():
    pairs = (("C", 4.0), ("Zn", 1.0))
    with pytest.raises(CompositionOverrideError, match="atom records"):
        apply_composition_override("SFAC C Zn\nUNIT 4 1\nC1 1 0.1 0.2 0.3 11.0 0.05\nEND\n", pairs)
    with pytest.raises(CompositionOverrideError, match="long form"):
        apply_composition_override("SFAC C 2.31 20.8 1.02 10.2 1.59 0.57 0.87 51.7 0.22\nEND\n", pairs)
    with pytest.raises(CompositionOverrideError, match="continued"):
        apply_composition_override("SFAC C =\n  Zn\nEND\n", pairs)
    text, changes = apply_composition_override(
        "TITL x\nSFAC C H\nDISP H 0 0 0\nDISP C 0.003 0.002 10\nHKLF 4\nEND\nREM tail\n", pairs
    )
    assert "SFAC C Zn" in text and "UNIT 4 1" in text and "DISP C" in text and "DISP H" not in text
    assert text.endswith("END\nREM tail\n")
    assert "inserted a UNIT record (the file had none)" in changes


@pytest.mark.parametrize(
    "name, expected",
    [
        ("DEV-CASE-A.ins", "DEV-CASE-A"),
        ("my file (2).ins", "my_file_2"),
        ("../../etc/passwd", "passwd"),
        ("....ins", "case"),
        ("_x.ins", "x"),
        ("数据.ins", "case"),
        ("a" * 80 + ".ins", "a" * 32),
    ],
)
def test_the_shelx_stem_never_comes_from_the_user_verbatim(name, expected):
    assert safe_stem(name) == expected


# --------------------------------------------------------------------------------------
# the command line uses the same service
# --------------------------------------------------------------------------------------


def test_crystalx_solve_demo_runs_end_to_end(tmp_path: Path, capsys):
    from crystalx_core.cli.main import main

    code = main(["solve", "--demo", "--runs-dir", str(tmp_path / "runs"), "--quiet"])
    output = capsys.readouterr().out
    assert code == 0
    assert "MOCK DEMO" in output
    assert "preliminary_refined_model" in output
    assert "not_evaluated" in output


def test_crystalx_solve_needs_inputs_or_demo(tmp_path: Path, capsys):
    from crystalx_core.cli.main import main

    assert main(["solve", "--runs-dir", str(tmp_path / "runs")]) == 2
    assert "--demo" in capsys.readouterr().err


def test_the_application_layer_imports_no_heavy_dependency():
    code = (
        "import sys, crystalx_core.app, crystalx_core.cli.solve; "
        "banned = {'torch', 'numpy', 'scipy', 'gradio', 'gemmi', 'rdkit', 'cctbx'}; "
        "print(sorted(banned & set(sys.modules)))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "[]"
