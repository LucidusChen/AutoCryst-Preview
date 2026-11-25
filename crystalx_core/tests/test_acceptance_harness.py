"""R2: the owner-run acceptance script and the run-record check it relies on.

The script is run for real here, end to end, against *stand-in* programs: small Python scripts in
a test toolbox that hand the job to the demo toolchain's writer, on the synthetic demo case. The
resolver finds them as ordinary programs, so the run record is a real-mode record -- which is
exactly what lets these tests prove the script's success path as well as its failures. No
crystallographic program is involved, and the stand-ins only work on the synthetic demo case.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import crystalx_core
from crystalx_core.app import ChemicalPriorInput, RunConfig, RunMode, RunService, SolveRequest, write_demo_case
from crystalx_core.app.acceptance import verify_run
from crystalx_core.app.demo import demo_digests
from crystalx_core.app.mock_toolchain import MockToolchain
from crystalx_core.domain.tools import ToolKind

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "crystalx_web" / "scripts" / "acceptance_private_case.sh"
SRC = Path(crystalx_core.__file__).resolve().parents[1]
STEM = "synthetic_acceptance_case"

posix_script = pytest.mark.skipif(
    os.name != "posix" or not SCRIPT.is_file() or shutil.which("sh") is None,
    reason="needs a POSIX shell and the crystalx_web scripts directory",
)

_STAND_IN = """\
import os, sys
sys.path.insert(0, {src!r})
from pathlib import Path
from crystalx_core.app.mock_toolchain import MockToolchainExecutor
from crystalx_core.domain.tools import ToolKind
from crystalx_core.tools.runner import ExecutionPlan

kind = ToolKind({kind!r})
plan = ExecutionPlan(argv=(kind.value, *sys.argv[1:]), cwd=Path.cwd(), tool_kind=kind)
executor = MockToolchainExecutor(step_seconds=0.0, omit_suffixes={omit!r}, fail_tool={fail})
result = executor.execute(plan, dict(os.environ))
sys.stdout.write(result.stdout + "\\n")
sys.stderr.write(result.stderr)
sys.exit(result.exit_code if result.exit_code is not None else 70)
"""


def _stand_in_toolbox(root: Path, *, omit: tuple[str, ...] = (), fail: str | None = None) -> Path:
    box = root / "toolbox"
    box.mkdir(parents=True, exist_ok=True)
    for kind in ("shelxt", "shelxl", "platon_special", "platon"):
        fail_expr = f"ToolKind({fail!r})" if fail == kind else "None"
        path = box / kind
        path.write_text(
            f"#!{sys.executable}\n" + _STAND_IN.format(src=str(SRC), kind=kind, omit=omit, fail=fail_expr),
            encoding="utf-8",
        )
        path.chmod(0o755)
    return box


def _harness(tmp_path: Path, **toolbox) -> dict[str, str]:
    source = tmp_path / "originals"
    ins, hkl = write_demo_case(tmp_path / "demo")
    source.mkdir()
    shutil.copyfile(ins, source / f"{STEM}.ins")
    shutil.copyfile(hkl, source / f"{STEM}.hkl")
    cli = tmp_path / "bin" / "crystalx"
    cli.parent.mkdir()
    cli.write_text(f'#!/bin/sh\nPYTHONPATH="{SRC}" exec "{sys.executable}" -m crystalx_core "$@"\n')
    cli.chmod(0o755)
    ins_digest, hkl_digest = demo_digests()
    env = dict(os.environ)
    env.update(
        CRYSTALX_ACCEPT_REPO=str(tmp_path),
        CRYSTALX_ACCEPT_CLI=str(cli),
        CRYSTALX_ACCEPT_PYTHON=sys.executable,
        CRYSTALX_ACCEPT_SOURCE_DIR=str(source),
        CRYSTALX_ACCEPT_STEM=STEM,
        CRYSTALX_ACCEPT_TOOLBOX=str(_stand_in_toolbox(tmp_path, **toolbox)),
        CRYSTALX_ACCEPT_RUNS_ROOT=str(tmp_path / "runs"),
        CRYSTALX_ACCEPT_EXPECT_INS_SHA256=ins_digest,
        CRYSTALX_ACCEPT_EXPECT_HKL_SHA256=hkl_digest,
    )
    return env


def _run_script(env: dict[str, str]) -> tuple[int, str, Path | None]:
    proc = subprocess.run(
        ["sh", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=300, check=False
    )
    runs = Path(env["CRYSTALX_ACCEPT_RUNS_ROOT"])
    dirs = sorted(runs.glob("acceptance_private_case_*")) if runs.is_dir() else []
    return proc.returncode, proc.stdout + proc.stderr, (dirs[-1] if dirs else None)


@posix_script
def test_the_script_passes_when_every_stage_ran_and_left_its_files(tmp_path: Path):
    code, output, directory = _run_script(_harness(tmp_path))
    assert code == 0, output
    assert "RESULT: PASS" in output
    assert directory is not None
    assert (directory / "status").read_text().strip() == "0"
    assert "RESULT: PASS" in (directory / "transcript.txt").read_text()
    verdict = json.loads((directory / "acceptance.json").read_text())
    assert verdict["accepted"] is True
    assert {c["name"] for c in verdict["checks"]} >= {"real_programs", "stages_succeeded", "inputs_expected"}


@posix_script
@pytest.mark.parametrize("required", ("CRYSTALX_ACCEPT_SOURCE_DIR", "CRYSTALX_ACCEPT_STEM"))
def test_private_inputs_must_be_supplied_explicitly(tmp_path: Path, required: str):
    env = _harness(tmp_path)
    env.pop(required)
    code, output, directory = _run_script(env)
    assert code == 2, output
    assert "CRYSTALX_ACCEPT_SOURCE_DIR and CRYSTALX_ACCEPT_STEM are required" in output
    assert directory is None
    assert not Path(env["CRYSTALX_ACCEPT_RUNS_ROOT"]).exists()


@posix_script
def test_omitted_expected_digests_capture_the_explicit_synthetic_inputs(tmp_path: Path):
    env = _harness(tmp_path)
    env.pop("CRYSTALX_ACCEPT_EXPECT_INS_SHA256")
    env.pop("CRYSTALX_ACCEPT_EXPECT_HKL_SHA256")
    code, output, directory = _run_script(env)
    assert code == 0, output
    assert directory is not None
    verdict = json.loads((directory / "acceptance.json").read_text())
    assert verdict["accepted"] is True
    assert next(c for c in verdict["checks"] if c["name"] == "inputs_expected")["passed"] is True


@posix_script
def test_a_failed_program_fails_the_acceptance_and_the_exit_status(tmp_path: Path):
    code, output, directory = _run_script(_harness(tmp_path, fail="shelxl"))
    assert code == 1, output
    assert "solve exit=1" in output and "RESULT: FAIL" in output
    assert (directory / "status").read_text().strip() == "1"
    assert "RESULT: FAIL" in (directory / "transcript.txt").read_text()  # kept on failure


@posix_script
def test_a_test_authored_failing_cli_fails_the_acceptance(tmp_path: Path):
    """The CLI itself fails (no run record at all); every other subcommand still works."""
    env = _harness(tmp_path)
    real = env["CRYSTALX_ACCEPT_CLI"]
    failing = tmp_path / "bin" / "crystalx-failing"
    failing.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = solve ]; then echo "test CLI: solve fails" >&2; exit 1; fi\n'
        f'exec "{real}" "$@"\n'
    )
    failing.chmod(0o755)
    env["CRYSTALX_ACCEPT_CLI"] = str(failing)
    code, output, directory = _run_script(env)
    assert code == 5, output
    assert "solve exit=1" in output
    transcript = (directory / "transcript.txt").read_text()
    assert "RESULT: FAIL" in transcript and "test CLI: solve fails" in transcript


@posix_script
def test_a_completed_run_with_a_blocked_stage_is_not_accepted(tmp_path: Path):
    """The first script's failure mode: the solve exits 0 (phase completed), PLATON was blocked."""
    code, output, directory = _run_script(_harness(tmp_path, omit=(".fcf",)))
    assert "solve exit=0" in output, output
    assert code == 1
    verdict = json.loads((directory / "acceptance.json").read_text())
    failed = {c["name"] for c in verdict["checks"] if not c["passed"]}
    assert {"stages_succeeded", "platon_run", "platon_report"} <= failed


@posix_script
def test_a_digest_mismatch_stops_before_anything_runs(tmp_path: Path):
    env = _harness(tmp_path)
    env["CRYSTALX_ACCEPT_EXPECT_HKL_SHA256"] = "0" * 64
    code, output, directory = _run_script(env)
    assert code == 4, output
    assert not (directory / "runs").exists()


@posix_script
def test_a_mac_that_cannot_launch_x86_64_stops_before_the_solve(tmp_path: Path):
    env = _harness(tmp_path)
    fake = tmp_path / "fakebin"
    fake.mkdir()
    for name, body in (("file", 'echo "Mach-O 64-bit executable x86_64"\n'), ("arch", "exit 1\n")):
        (fake / name).write_text("#!/bin/sh\n" + body)
        (fake / name).chmod(0o755)
    env["PATH"] = f"{fake}{os.pathsep}{env.get('PATH', '')}"
    code, output, directory = _run_script(env)
    assert code == 3, output
    assert "host translation" in output
    assert not (directory / "runs").exists()


@posix_script
def test_two_runs_never_share_a_directory(tmp_path: Path):
    env = _harness(tmp_path)
    env["CRYSTALX_ACCEPT_EXPECT_INS_SHA256"] = "0" * 64  # stop early; only the directory matters
    _run_script(env)
    _run_script(env)
    assert len(list((tmp_path / "runs").glob("acceptance_private_case_*"))) == 2


@posix_script
def test_a_missing_cli_is_a_setup_error(tmp_path: Path):
    env = _harness(tmp_path)
    env["CRYSTALX_ACCEPT_CLI"] = str(tmp_path / "nowhere" / "crystalx")
    code, output, _ = _run_script(env)
    assert code == 2 and "not found" in output


# --------------------------------------------------------------------------------------
# the run-record check itself
# --------------------------------------------------------------------------------------


def _mock_run(tmp_path: Path, **mock) -> Path:
    service = RunService(
        tmp_path / "runs", toolchain_factory=lambda _r: MockToolchain(step_seconds=0.02, **mock)
    )
    ins, hkl = write_demo_case(tmp_path / "demo_input")
    handle = service.submit(
        SolveRequest(
            ins_path=ins,
            hkl_path=hkl,
            prior=ChemicalPriorInput(metals=("Zn",)),
            config=RunConfig(mock_step_seconds=0.02),
            mode=RunMode.MOCK_DEMO,
            label="demo",
        )
    )
    service.wait(handle.run_id, timeout=30)
    return handle.run_dir


def test_mock_output_is_never_real_tool_acceptance(tmp_path: Path):
    run_dir = _mock_run(tmp_path)
    verdict = verify_run(run_dir)
    assert not verdict.accepted
    failed = {c.name for c in verdict.failed}
    assert {"real_programs", "shelxt_run", "shelxl_run", "platon_run"} <= failed
    # the checker itself passes the complete demo record when told to ignore the mock origin
    assert verify_run(run_dir, require_real=False).accepted


def test_partial_completion_is_not_acceptance_even_for_the_checker_test(tmp_path: Path):
    blocked = verify_run(_mock_run(tmp_path / "a", omit_suffixes=(".fcf",)), require_real=False)
    assert not blocked.accepted
    assert "platon: blocked" in next(c.detail for c in blocked.checks if c.name == "stages_succeeded")

    errors = verify_run(
        _mock_run(tmp_path / "b", shelxl_listing_extra=(" ** ILLEGAL INSTRUCTION AFIX 99 **",)),
        require_real=False,
    )
    assert not errors.accepted
    assert "ILLEGAL INSTRUCTION AFIX 99" in next(c.detail for c in errors.checks if c.name == "refinement_clean")


def test_an_output_changed_after_the_run_is_detected(tmp_path: Path):
    run_dir = _mock_run(tmp_path)
    lst = next(run_dir.glob("refinement/shelxl-01/*.lst"))
    lst.write_text(lst.read_text() + "edited\n")
    verdict = verify_run(run_dir, require_real=False)
    assert not verdict.accepted
    assert "changed after" in next(c.detail for c in verdict.checks if c.name == "shelxl_run")


def test_an_unexpected_input_digest_is_detected(tmp_path: Path):
    run_dir = _mock_run(tmp_path)
    ins_digest, _ = demo_digests()
    assert verify_run(run_dir, require_real=False, expected_inputs={"ins": ins_digest}).accepted
    verdict = verify_run(run_dir, require_real=False, expected_inputs={"ins": "0" * 64})
    assert [c.name for c in verdict.failed] == ["inputs_expected"]


def test_an_unreadable_record_is_a_failure_not_a_crash(tmp_path: Path):
    verdict = verify_run(tmp_path)
    assert not verdict.accepted
    assert verdict.checks[0].name == "record_readable"


def test_verify_run_cli_exit_codes(tmp_path: Path, capsys):
    from crystalx_core.cli.main import main

    run_dir = _mock_run(tmp_path)
    assert main(["verify-run", str(run_dir)]) == 1
    assert "ACCEPTANCE: FAIL" in capsys.readouterr().out
    assert main(["verify-run", str(run_dir), "--allow-mock"]) == 0
    assert "NOT real-tool acceptance" in capsys.readouterr().out
    assert main(["verify-run", str(tmp_path / "missing")]) == 2
    assert main(["verify-run", str(run_dir), "--expect-input", "ins=abc"]) == 2
