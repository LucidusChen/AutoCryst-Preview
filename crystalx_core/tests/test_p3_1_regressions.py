"""P3.1 regressions for the Codex review of 2026-10-14 (R1-R4 and the follow-up items).

The "executables" here are test-authored shell or Python scripts that imitate only the file
contract of SHELXT (write ``_a.res``/``_a.hkl``) or fail on purpose. No crystallographic program
is involved, and every resolver below is confined to the test's own toolbox directory, so a real
SHELX or PLATON installation on the machine running the tests can never be picked up.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from conftest import write_hkl, write_ins
from crystalx_core.app import (
    ChemicalPriorInput,
    ModelAvailability,
    RefinementStatus,
    RunConfig,
    RunMode,
    RunPhase,
    RunService,
    SolveRequest,
    write_demo_case,
)
from crystalx_core.app.mock_toolchain import MockToolchain
from crystalx_core.app.toolchain import RealToolchain
from crystalx_core.domain.chemistry import ChemicalPrior
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.ids import make_stage_id
from crystalx_core.domain.onion import StageKind
from crystalx_core.domain.solving import SolverKind
from crystalx_core.domain.tools import ToolDiagnosticKind, ToolKind
from crystalx_core.io.shelx_reader import read_ins_header, scan_hkl
from crystalx_core.solvers.selector import SelectionPolicy, StrategySelector
from crystalx_core.tools import runner as runner_module
from crystalx_core.tools.adapter import ExternalToolAdapter
from crystalx_core.tools.resolver import ResolverConfig, ToolResolver
from crystalx_core.tools.runner import CancelToken, ExecutionPlan, ProcessRunner
from crystalx_core.workflow.preconditions import PreconditionKind, check_preconditions

POSIX = os.name == "posix"
posix_only = pytest.mark.skipif(not POSIX, reason="POSIX scripts, process groups and signals")
FAST = 0.05
STAGE = make_stage_id(1, StageKind.INITIAL_SOLUTION.value)

# A stand-in with SHELXT's file contract only: one trial structure in P2(1) (or with no LATT
# record, which the symmetry analysis cannot classify) and a copy of the reflections.
FAKE_SHELXT = """\
stem="$1"
{
  echo "TITL ${stem}_a in P2(1)"
  echo "REM SHELXT solution in P2(1): R1 0.100, Rweak 0.050, Alpha 0.000"
  echo "REM <I/s> 0.000 for 0 systematic absences, Orientation as input"
  grep -E '^(CELL|ZERR)' "${stem}.ins"
  __LATT__
  echo "SYMM -X, 1/2+Y, -Z"
  echo "SFAC C N O Ag"
  echo "UNIT 24 4 8 2"
  echo "Ag1 4 0.1 0.2 0.3 11.0 0.05"
  echo "O1 3 0.3 0.2 0.3 11.0 0.05"
  echo "HKLF 4"
  echo "END"
} > "${stem}_a.res"
cp "${stem}.hkl" "${stem}_a.hkl"
echo "test stand-in for shelxt finished"
"""
FAILING_TOOL = 'echo "test stand-in refuses to run" >&2\nexit 3\n'


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toolbox(root: Path, *, latt: bool = True) -> Path:
    """Stand-ins for every program the pipeline uses, so that no search can reach a real one."""
    box = root / "tb"
    _script(box / "shelxt", FAKE_SHELXT.replace("__LATT__", 'echo "LATT -1"' if latt else ":"))
    for name in ("shelxl", "platon_special", "platon"):
        _script(box / name, FAILING_TOOL)
    return box


def _confined_resolver(toolbox_dir: Path | None) -> ToolResolver:
    return ToolResolver(
        ResolverConfig(
            toolbox_dir=toolbox_dir, search_path=[], include_known_locations=False, include_olex2=False
        ),
        quarantine_checker=lambda _p: False,
    )


def _service(tmp_path: Path, **mock) -> RunService:
    def factory(request: SolveRequest):
        if request.mode is RunMode.MOCK_DEMO:
            return MockToolchain(step_seconds=FAST, **mock)
        return RealToolchain(resolver=_confined_resolver(request.config.toolbox_dir))

    return RunService(tmp_path / "runs", toolchain_factory=factory)


def _demo_request(tmp_path: Path, prior: ChemicalPriorInput | None = None) -> SolveRequest:
    ins, hkl = write_demo_case(tmp_path / "demo_input")
    return SolveRequest(
        ins_path=ins,
        hkl_path=hkl,
        prior=prior or ChemicalPriorInput(metals=("Zn",)),
        config=RunConfig(mock_step_seconds=FAST),
        mode=RunMode.MOCK_DEMO,
        label="demo",
    )


def _stages(state: dict) -> dict[str, str]:
    return {s["name"]: s["state"] for s in state["stages"]}


def _shelxt_provenance(run_dir: Path) -> dict:
    files = sorted(run_dir.glob("stages/*/branches/*-shelxt/provenance_*.json"))
    assert files, "no SHELXT provenance written"
    return json.loads(files[-1].read_text())


def _diagnose(tmp_path: Path, **ins_kwargs) -> DataDiagnosis:
    ins = write_ins(tmp_path / "case.ins", **ins_kwargs)
    hkl = write_hkl(tmp_path / "case.hkl", include_friedel=True)
    header = read_ins_header(ins)
    return DataDiagnosis(reflections=scan_hkl(hkl, cell=header.cell, hklf_hint=header.hklf_format), ins=header)


def _is_running(pid: int) -> bool:
    """True only for a live process; a zombie has already died (see test_runner_control)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            return stat.read_text().rsplit(")", 1)[-1].split()[0] != "Z"
        except OSError:
            return False
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def _wait_for(path: Path, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"{path.name} did not appear")


# --------------------------------------------------------------------------------------
# R1 - the executable path survives the change of working directory and matches its digest
# --------------------------------------------------------------------------------------


def test_a_relative_toolbox_is_made_absolute_when_the_request_is_built(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert RunConfig(toolbox_dir=Path("tb")).toolbox_dir == tmp_path / "tb"
    assert RunConfig(toolbox_dir=Path("a/../tb")).toolbox_dir == tmp_path / "tb"
    assert RunConfig().toolbox_dir is None


@posix_only
def test_a_relative_toolbox_still_runs_from_a_nested_working_directory(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    program = _script(tmp_path / "tb" / "shelxt", 'echo ran > marker\n')
    resolution = _confined_resolver(Path("tb")).resolve(ToolKind.SHELXT)
    assert resolution.is_usable
    assert resolution.identity.path == tmp_path / "tb" / "shelxt"
    assert resolution.identity.path.is_absolute()

    work = tmp_path / "runs" / "r1" / "stages" / "branch"
    work.mkdir(parents=True)
    outcome, record = ExternalToolAdapter(resolution).run(
        args=["case"], cwd=work, timeout_s=20, expected_outputs=["marker"]
    )
    assert outcome.succeeded, outcome.diagnostic.message
    assert record.executable_path == str(program)
    assert record.executable_sha256 == _sha256(program)
    assert outcome.plan.argv[0] == str(program)


@posix_only
def test_an_executable_replaced_after_resolution_is_not_started(tmp_path: Path):
    program = _script(tmp_path / "tb" / "shelxt", "exit 0\n")
    resolution = _confined_resolver(tmp_path / "tb").resolve(ToolKind.SHELXT)
    _script(program, "touch started\nexit 0\n")  # different bytes at the same path
    work = tmp_path / "work"
    work.mkdir()
    outcome, record = ExternalToolAdapter(resolution).run(args=["case"], cwd=work, timeout_s=20)
    assert outcome.diagnostic.kind is ToolDiagnosticKind.EXECUTABLE_CHANGED
    assert not outcome.succeeded
    assert not (work / "started").exists(), "the replaced program must not run"
    assert record.executable_sha256 == resolution.identity.sha256  # what was resolved, not run


@posix_only
def test_crystalx_solve_with_a_relative_toolbox_runs_and_records_those_programs(
    tmp_path: Path, monkeypatch, capsys
):
    """Codex attempt A: ``--toolbox toolbox`` gave not_found once the program ran in its run dir."""
    from crystalx_core.cli.main import main

    monkeypatch.chdir(tmp_path)
    box = _toolbox(tmp_path)
    write_ins(tmp_path / "c.ins", latt=-1, symm=("-X, 1/2+Y, -Z",))
    write_hkl(tmp_path / "c.hkl")
    code = main(
        ["solve", "--ins", "c.ins", "--hkl", "c.hkl", "--toolbox", "tb", "--runs-dir", "runs", "--quiet"]
    )
    capsys.readouterr()
    assert code == 1  # SHELXL's stand-in exits 3: a failed run, never "success"
    (run_dir,) = [p for p in (tmp_path / "runs").iterdir() if p.name.startswith("r-")]

    shelxt = _shelxt_provenance(run_dir)
    assert shelxt["succeeded"] is True
    assert shelxt["executable_path"] == str(box / "shelxt")
    assert shelxt["executable_sha256"] == _sha256(box / "shelxt")
    assert shelxt["executable_source"] == "toolbox"

    shelxl = json.loads((run_dir / "refinement/shelxl-01/provenance.json").read_text())
    assert shelxl["executable_path"] == str(box / "shelxl")
    assert shelxl["diagnostic"] == "nonzero_exit"  # it was found and started: not "not_found"
    assert str(box) in (run_dir / "request.json").read_text()


@posix_only
def test_the_service_runs_a_relative_toolbox_given_in_the_request(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    box = _toolbox(tmp_path)
    service = _service(tmp_path)
    ins = write_ins(tmp_path / "c.ins", latt=-1, symm=("-X, 1/2+Y, -Z",))
    hkl = write_hkl(tmp_path / "c.hkl")
    handle = service.submit(
        SolveRequest(ins_path=ins, hkl_path=hkl, config=RunConfig(toolbox_dir=Path("tb")), label="r1")
    )
    state = service.wait(handle.run_id, timeout=60)
    assert _stages(state)["shelxt"] == "succeeded"
    assert _stages(state)["shelxl"] == "failed"
    assert _shelxt_provenance(handle.run_dir)["executable_path"] == str(box / "shelxt")


# --------------------------------------------------------------------------------------
# R3 - cancellation and timeouts stop the whole process group, even after the parent exits
# --------------------------------------------------------------------------------------

_PARENT = """\
import os, subprocess, sys, time
CHILD = (
    "import signal, time\\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
    "open('child.ready', 'w').close()\\n"
    "time.sleep(120)\\n"
)
child = subprocess.Popen([sys.executable, "-c", CHILD])
while not os.path.exists("child.ready"):
    time.sleep(0.01)
with open("child.pid.tmp", "w") as handle:
    handle.write(str(child.pid))
os.replace("child.pid.tmp", "child.pid")
if sys.argv[1:] == ["exit"]:
    sys.exit(0)  # leave the child behind
time.sleep(120)
"""


def _parent_program(tmp_path: Path) -> Path:
    path = tmp_path / "parent.py"
    path.write_text(f"#!{sys.executable}\n" + _PARENT, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def short_grace(monkeypatch):
    monkeypatch.setattr(runner_module, "TERMINATE_GRACE_S", 0.5)


def _run_in_thread(plan: ExecutionPlan, tmp_path: Path, cancel: CancelToken | None = None):
    result: dict = {}

    def target() -> None:
        result["outcome"] = ProcessRunner().run(plan, log_dir=tmp_path / "logs", cancel=cancel)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, result


@posix_only
def test_cancel_kills_a_child_that_ignores_sigterm(tmp_path: Path, short_grace):
    work = tmp_path / "work"
    work.mkdir()
    plan = ExecutionPlan(argv=(str(_parent_program(tmp_path)), "wait"), cwd=work, timeout_s=60, label="tree")
    token = CancelToken()
    thread, result = _run_in_thread(plan, tmp_path, token)
    _wait_for(work / "child.pid")
    child = int((work / "child.pid").read_text())
    token.cancel("test")
    thread.join(30)
    outcome = result["outcome"]
    assert outcome.cancelled
    assert outcome.diagnostic.kind is ToolDiagnosticKind.CANCELLED
    assert outcome.group_cleanup == "killed"
    assert not _is_running(child)


@posix_only
def test_a_timeout_kills_a_child_that_ignores_sigterm(tmp_path: Path, short_grace):
    work = tmp_path / "work"
    work.mkdir()
    plan = ExecutionPlan(argv=(str(_parent_program(tmp_path)), "wait"), cwd=work, timeout_s=3.0, label="tree")
    thread, result = _run_in_thread(plan, tmp_path)
    _wait_for(work / "child.pid")
    child = int((work / "child.pid").read_text())
    thread.join(30)
    outcome = result["outcome"]
    assert outcome.timed_out
    assert outcome.diagnostic.kind is ToolDiagnosticKind.TIMEOUT
    assert outcome.group_cleanup == "killed"
    assert not _is_running(child)


@posix_only
def test_a_child_left_behind_by_a_parent_that_exited_is_stopped(tmp_path: Path, short_grace):
    work = tmp_path / "work"
    work.mkdir()
    plan = ExecutionPlan(argv=(str(_parent_program(tmp_path)), "exit"), cwd=work, timeout_s=60, label="tree")
    outcome = ProcessRunner().run(plan, log_dir=tmp_path / "logs")
    child = int((work / "child.pid").read_text())
    assert outcome.exit_code == 0
    assert outcome.group_cleanup == "killed"
    assert not _is_running(child)


@posix_only
def test_a_program_that_leaves_nothing_behind_needs_no_cleanup(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    program = _script(tmp_path / "quiet", "exit 0\n")
    outcome = ProcessRunner().run(ExecutionPlan(argv=(str(program),), cwd=work, timeout_s=20))
    assert outcome.succeeded and outcome.group_cleanup == "none"


# --------------------------------------------------------------------------------------
# fatal refinement errors are failures, not results
# --------------------------------------------------------------------------------------


def test_a_listing_with_shelxl_errors_is_a_failed_refinement(tmp_path: Path):
    service = _service(tmp_path, shelxl_listing_extra=(" ** CANNOT FIND ATOM ZN2 IN INSTRUCTION DFIX **",))
    handle = service.submit(_demo_request(tmp_path))
    state = service.wait(handle.run_id, timeout=30)
    stages = _stages(state)
    assert stages["shelxl"] == "failed"
    assert stages["platon"] == "skipped"
    assert state["phase"] != RunPhase.COMPLETED.value
    summary = service.summary(handle.run_id)
    assert summary["refinement_status"] == RefinementStatus.ERRORS_REPORTED.value
    assert summary["model_availability"] == ModelAvailability.SOLUTION_MODEL.value
    assert "CANNOT FIND ATOM ZN2 IN INSTRUCTION DFIX" in json.dumps(summary["refinement"])
    assert service.viewer_payload(handle.run_id, "refined") is None


# --------------------------------------------------------------------------------------
# chirality: route context for SHELXT, hard checks everywhere else
# --------------------------------------------------------------------------------------


def test_a_racemic_declaration_lets_a_centrosymmetric_choice_be_refined(tmp_path: Path):
    service = _service(tmp_path, shelxt_centrosymmetric=True)
    prior = ChemicalPriorInput(metals=("Zn",), peptide=True, racemic=True)
    handle = service.submit(_demo_request(tmp_path, prior))
    state = service.wait(handle.run_id, timeout=30)
    assert _stages(state)["shelxl"] == "succeeded"


@posix_only
def test_an_unclassifiable_shelxt_group_is_not_refined_for_a_homochiral_material(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _toolbox(tmp_path, latt=False)  # the trial structure has no LATT record
    service = _service(tmp_path)
    ins = write_ins(tmp_path / "c.ins", latt=-1, symm=("-X, 1/2+Y, -Z",))
    hkl = write_hkl(tmp_path / "c.hkl")
    handle = service.submit(
        SolveRequest(
            ins_path=ins,
            hkl_path=hkl,
            prior=ChemicalPriorInput(peptide=True),
            config=RunConfig(toolbox_dir=tmp_path / "tb"),
            label="pep",
        )
    )
    state = service.wait(handle.run_id, timeout=60)
    assert _stages(state)["shelxt"] == "succeeded"
    assert _stages(state)["shelxl"] == "blocked"
    assert "could not be classified" in service.summary(handle.run_id)["refinement"]["reason"]
    assert not (handle.run_dir / "refinement").exists()


def test_without_route_context_an_impossible_group_still_blocks(tmp_path: Path):
    diagnosis = _diagnose(tmp_path, latt=1, symm=("-x, y+1/2, -z",))
    prior = ChemicalPrior(is_peptide_like=True)
    report = check_preconditions(diagnosis, prior)
    assert not report.may_start
    assert report.blockers[0].kind is PreconditionKind.CHIRALITY_IMPOSSIBLE

    routed = check_preconditions(diagnosis, prior, space_group_from_solver=True)
    assert routed.may_start
    assert [i.kind for i in routed.preconditions] == [PreconditionKind.CHIRALITY_UNRESOLVED_START]
    assert routed.blockers == ()


def test_fixed_space_group_routes_are_excluded_by_r10_and_cannot_be_forced(tmp_path: Path):
    diagnosis = _diagnose(tmp_path, latt=1, symm=("-x, y+1/2, -z",))
    prior = ChemicalPrior(is_peptide_like=True)
    trace = StrategySelector().select(diagnosis, prior, stage_id=STAGE)
    reasons = dict(trace.excluded)
    assert SolverKind.SHELXT in trace.nominated_kinds
    for kind in (SolverKind.SHELXS_PATT, SolverKind.SHELXS_TREF):
        assert reasons[kind].startswith("R10:")

    forced = StrategySelector(SelectionPolicy(forced_kinds=(SolverKind.SHELXS_PATT,))).select(
        diagnosis, prior, stage_id=STAGE
    )
    assert SolverKind.SHELXS_PATT not in forced.nominated_kinds
    assert dict(forced.excluded)[SolverKind.SHELXS_PATT].startswith("R10:")


# --------------------------------------------------------------------------------------
# R4 follow-up and SHELXL message classification
# --------------------------------------------------------------------------------------


def test_a_missing_maximum_shift_is_unknown_not_converged():
    from crystalx_core.io.shelxl_lst import parse_lst_text

    listing = parse_lst_text(" Mean shift/esd =   0.001\n")
    assert listing.mean_shift_su == pytest.approx(0.001)
    assert listing.max_shift_su is None
    assert listing.converged is None
    assert listing.to_record()["convergence"] == "not_assessed"


def test_starred_shelxl_lines_are_classified_not_all_fatal():
    from crystalx_core.io.shelxl_lst import parse_lst_text

    listing = parse_lst_text(
        " ** Extinction (EXTI) or solvent water (SWAT) correction may be required **\n"
        " ** Absolute structure probably wrong - invert and repeat refinement **\n"
        " ** CANNOT OPEN FILE c.hkl **\n"
        " ** SOMETHING THIS BUILD HAS NEVER SEEN **\n"
    )
    assert listing.errors == ("CANNOT OPEN FILE c.hkl",)
    assert len(listing.warnings) == 2
    assert listing.notices == ("SOMETHING THIS BUILD HAS NEVER SEEN",)


def test_shelxl_advice_alone_does_not_fail_the_refinement(tmp_path: Path):
    advice = " ** Extinction (EXTI) or solvent water (SWAT) correction may be required **"
    service = _service(tmp_path, shelxl_listing_extra=(advice, " ** UNFAMILIAR MESSAGE **"))
    handle = service.submit(_demo_request(tmp_path))
    state = service.wait(handle.run_id, timeout=30)
    assert _stages(state)["shelxl"] == "succeeded"
    summary = service.summary(handle.run_id)
    assert summary["refinement_status"] == RefinementStatus.RAN_CONVERGENCE_NOT_ASSESSED.value
    warnings = " ".join(summary["warnings"])
    assert "Extinction" in warnings and "not classified" in warnings


@posix_only
def test_an_explicit_relative_tool_path_is_made_absolute(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    program = _script(tmp_path / "bin" / "my-shelxt", "exit 0\n")
    resolver = ToolResolver(
        ResolverConfig(
            explicit_paths={"shelxt": Path("bin/my-shelxt")},
            search_path=[],
            include_known_locations=False,
            include_olex2=False,
        ),
        quarantine_checker=lambda _p: False,
    )
    resolution = resolver.resolve(ToolKind.SHELXT)
    assert resolution.is_usable
    assert resolution.identity.path == program
    work = tmp_path / "elsewhere" / "deeper"
    work.mkdir(parents=True)
    outcome, _ = ExternalToolAdapter(resolution).run(args=[], cwd=work, timeout_s=20)
    assert outcome.succeeded, outcome.diagnostic.message


def test_the_contact_search_extent_follows_the_cell_metric():
    from crystalx_core.geometry.model import CONTACT_SEARCH_CAP, contact_search_extents

    # ordinary cell, sites inside it: one or two translations per axis
    (na, nb, nc), complete = contact_search_extents((10, 11, 12, 90, 95, 90), [(0.1, 0.2, 0.3), (0.9, 0.8, 0.7)], 3.5)
    assert complete and max(na, nb, nc) <= 2
    # a flat skew cell needs more along the short reciprocal direction, and gets it
    (na, nb, nc), complete = contact_search_extents((5, 5, 10, 90, 90, 30), [(0.0, 0.0, 0.0), (0.49, 0.49, 0.0)], 3.5)
    assert complete and na >= 2 and nb >= 2
    # a pathological cell is capped and says so
    _, complete = contact_search_extents((1.0, 1.0, 1.0, 90, 90, 1.0), [(0.0, 0.0, 0.0)], 3.5)
    assert not complete
    assert CONTACT_SEARCH_CAP >= 2


def test_a_contact_two_cells_away_in_a_very_skew_cell_is_found(tmp_path: Path):
    """A brute-force check over a wide translation range agrees with the bounded search."""
    import itertools
    import math as _m

    from crystalx_core.geometry.model import frac_to_cart, orthogonalisation_matrix
    from crystalx_core.io.shelx_res import read_res

    cell = (4.0, 4.0, 9.0, 90.0, 90.0, 8.0)
    text = (
        "TITL skew\nCELL 0.71073 4 4 9 90 90 8\nZERR 1 0 0 0 0 0 0\nLATT -1\nSFAC C\nUNIT 2\n"
        "C1 1 0.00 0.00 0.00 11.0 0.05\nC2 1 0.95 0.05 0.00 11.0 0.05\nHKLF 4\nEND\n"
    )
    from crystalx_core.geometry.model import build_viewer_payload

    res_path = tmp_path / "skew.res"
    res_path.write_text(text, encoding="utf-8")
    payload = build_viewer_payload(read_res(res_path), stage="solution")
    m = orthogonalisation_matrix(cell)
    expected = set()
    a, b = (0.0, 0.0, 0.0), (0.95, 0.05, 0.0)
    for t in itertools.product(range(-6, 7), repeat=3):
        if t == (0, 0, 0):
            continue
        d = _m.dist(frac_to_cart(m, a), frac_to_cart(m, (b[0] + t[0], b[1] + t[1], b[2] + t[2])))
        if 0.40 <= d <= 0.76 * 2 + 0.40:
            expected.add(tuple(t))
    found = {tuple(c["translation"]) for c in payload["boundary_bonds"] if {c["a"], c["b"]} == {"C1", "C2"}}
    assert payload["contact_search"]["complete"] is True
    assert any(max(abs(x) for x in t) >= 3 for t in expected), "the case must need |t| >= 3"
    assert expected <= found
