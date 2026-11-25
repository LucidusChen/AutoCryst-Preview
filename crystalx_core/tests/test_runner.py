"""Process invocation.

Two kinds of test here:

* **Scripted** tests, which replace the executor entirely and assert on the exact argument arrays.
* A small number of tests that really do run a POSIX shell script **written by the test itself**, to
  prove that timeout, exit-code and expected-output handling work against a real process rather than
  only against a mock.

No crystallographic program is executed anywhere in this suite.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from conftest import write_shell_stub
from crystalx_core.domain.tools import ToolDiagnosticKind
from crystalx_core.errors import ToolExecutionError
from crystalx_core.solvers.mock import ScriptedExecutor, ScriptedResponse
from crystalx_core.tools.runner import ExecutionPlan, ProcessRunner

HAVE_SH = shutil.which("sh") is not None
real_process = pytest.mark.skipif(not HAVE_SH, reason="POSIX shell not available")


# --------------------------------------------------------------------------------------
# plan construction
# --------------------------------------------------------------------------------------


def test_plan_requires_argv(tmp_path: Path):
    with pytest.raises(ValueError):
        ExecutionPlan(argv=(), cwd=tmp_path)


def test_plan_requires_positive_timeout(tmp_path: Path):
    with pytest.raises(ValueError):
        ExecutionPlan(argv=("x",), cwd=tmp_path, timeout_s=0)


def test_plan_record_is_serialisable(tmp_path: Path):
    plan = ExecutionPlan(
        argv=("platon_special", "-u", "case.cif"),
        cwd=tmp_path,
        stdin_script="CALC ADDSYM\nEND\n",
        expected_outputs=("case.chk",),
        label="platon_validate",
    )
    record = plan.to_record()
    assert record["argv"] == ["platon_special", "-u", "case.cif"]
    assert record["has_stdin_script"] is True
    assert record["expected_outputs"] == ["case.chk"]


# --------------------------------------------------------------------------------------
# scripted executor
# --------------------------------------------------------------------------------------


def test_scripted_success_with_expected_output(tmp_path: Path):
    executor = ScriptedExecutor([ScriptedResponse(creates=["case.chk"])])
    runner = ProcessRunner(executor, base_env={})
    plan = ExecutionPlan(argv=("platon", "-u", "case.cif"), cwd=tmp_path, expected_outputs=("case.chk",))

    outcome = runner.run(plan)

    assert outcome.succeeded
    assert outcome.diagnostic.kind is ToolDiagnosticKind.OK
    assert outcome.produced_outputs == ("case.chk",)
    assert outcome.missing_outputs == ()
    assert executor.argv_history == [("platon", "-u", "case.cif")]


def test_expected_output_missing_is_not_success(tmp_path: Path):
    """Exit code zero with no output is the failure mode that quietly corrupts a workflow."""
    executor = ScriptedExecutor([ScriptedResponse(exit_code=0)])
    runner = ProcessRunner(executor, base_env={})
    plan = ExecutionPlan(argv=("shelxl", "case"), cwd=tmp_path, expected_outputs=("case.res",))

    outcome = runner.run(plan)

    assert not outcome.succeeded
    assert outcome.diagnostic.kind is ToolDiagnosticKind.EXPECTED_OUTPUT_MISSING
    assert outcome.missing_outputs == ("case.res",)


def test_nonzero_exit_is_reported_with_output_tail(tmp_path: Path):
    executor = ScriptedExecutor([ScriptedResponse(exit_code=2, stderr="fatal: no cell\n")])
    runner = ProcessRunner(executor, base_env={})
    outcome = runner.run(ExecutionPlan(argv=("shelxt", "case"), cwd=tmp_path))

    assert outcome.diagnostic.kind is ToolDiagnosticKind.NONZERO_EXIT
    assert "fatal: no cell" in outcome.diagnostic.message
    assert outcome.exit_code == 2


def test_timeout_is_classified_and_advised(tmp_path: Path):
    executor = ScriptedExecutor([ScriptedResponse(timed_out=True)])
    runner = ProcessRunner(executor, base_env={})
    outcome = runner.run(ExecutionPlan(argv=("shelxd", "case"), cwd=tmp_path, timeout_s=1.0))

    assert outcome.timed_out
    assert outcome.diagnostic.kind is ToolDiagnosticKind.TIMEOUT
    assert "budget" in outcome.diagnostic.remedy


def test_bad_cpu_type_is_reported_as_architecture_not_missing_file(tmp_path: Path):
    """macOS raises OSError('Bad CPU type in executable') for an x86_64 binary without Rosetta."""
    executor = ScriptedExecutor(
        [ScriptedResponse(os_error="[Errno 86] Bad CPU type in executable: 'shelxt'")]
    )
    runner = ProcessRunner(executor, base_env={})
    outcome = runner.run(ExecutionPlan(argv=("shelxt", "case"), cwd=tmp_path))

    assert outcome.diagnostic.kind is ToolDiagnosticKind.ARCH_MISMATCH
    assert "architecture problem, not a missing file" in outcome.diagnostic.message
    assert "install-rosetta" in outcome.diagnostic.remedy


def test_exec_format_error_is_also_architecture(tmp_path: Path):
    executor = ScriptedExecutor([ScriptedResponse(os_error="[Errno 8] Exec format error")])
    runner = ProcessRunner(executor, base_env={})
    outcome = runner.run(ExecutionPlan(argv=("shelxt", "case"), cwd=tmp_path))
    assert outcome.diagnostic.kind is ToolDiagnosticKind.ARCH_MISMATCH


def test_permission_error_is_classified(tmp_path: Path):
    executor = ScriptedExecutor([ScriptedResponse(os_error="[Errno 13] Permission denied")])
    runner = ProcessRunner(executor, base_env={})
    outcome = runner.run(ExecutionPlan(argv=("shelxt", "case"), cwd=tmp_path))
    assert outcome.diagnostic.kind is ToolDiagnosticKind.NOT_EXECUTABLE


def test_stdin_script_is_passed_through(tmp_path: Path):
    """PLATON drives ADDSYM/SQUEEZE from stdin instructions, not flags."""
    executor = ScriptedExecutor([ScriptedResponse()])
    runner = ProcessRunner(executor, base_env={})
    plan = ExecutionPlan(
        argv=("platon", "case.res"), cwd=tmp_path, stdin_script="CALC ADDSYM\nEND\n"
    )
    runner.run(plan)
    assert executor.calls[0].stdin_script == "CALC ADDSYM\nEND\n"


def test_env_overrides_and_unset(tmp_path: Path):
    """Headless PLATON wants DISPLAY removed and CHECKDEF controlled."""
    executor = ScriptedExecutor([ScriptedResponse()])
    runner = ProcessRunner(executor, base_env={"DISPLAY": ":0", "HOME": "/root"})
    plan = ExecutionPlan(
        argv=("platon_special", "-u", "case.cif"),
        cwd=tmp_path,
        env_unset=("DISPLAY",),
        env_overrides={"CHECKDEF": ""},
    )
    runner.run(plan)
    env = executor.envs[0]
    assert "DISPLAY" not in env
    assert env["CHECKDEF"] == ""
    assert env["HOME"] == "/root"


def test_logs_are_written_when_a_log_dir_is_given(tmp_path: Path):
    executor = ScriptedExecutor([ScriptedResponse(stdout="hello", stderr="warn")])
    runner = ProcessRunner(executor, base_env={})
    logs = tmp_path / "logs"
    outcome = runner.run(
        ExecutionPlan(argv=("shelxl", "case"), cwd=tmp_path, label="refine"), log_dir=logs
    )
    assert outcome.stdout_path is not None and outcome.stdout_path.read_text() == "hello"
    assert outcome.stderr_path is not None and outcome.stderr_path.read_text() == "warn"


def test_missing_working_directory_raises(tmp_path: Path):
    runner = ProcessRunner(ScriptedExecutor(), base_env={})
    with pytest.raises(ToolExecutionError):
        runner.run(ExecutionPlan(argv=("x",), cwd=tmp_path / "absent"))


# --------------------------------------------------------------------------------------
# real subprocess, using shell scripts the test writes itself
# --------------------------------------------------------------------------------------


@real_process
def test_real_process_success_and_output_creation(tmp_path: Path):
    stub = write_shell_stub(tmp_path / "stub.sh", 'printf ok > "$1"\nexit 0\n')
    runner = ProcessRunner()
    outcome = runner.run(
        ExecutionPlan(argv=(str(stub), "made.txt"), cwd=tmp_path, expected_outputs=("made.txt",))
    )
    assert outcome.succeeded
    assert (tmp_path / "made.txt").read_text() == "ok"


@real_process
def test_real_process_nonzero_exit(tmp_path: Path):
    stub = write_shell_stub(tmp_path / "fail.sh", 'echo "boom" >&2\nexit 3\n')
    runner = ProcessRunner()
    outcome = runner.run(ExecutionPlan(argv=(str(stub),), cwd=tmp_path))
    assert outcome.exit_code == 3
    assert outcome.diagnostic.kind is ToolDiagnosticKind.NONZERO_EXIT
    assert "boom" in outcome.diagnostic.message


@real_process
def test_real_process_timeout(tmp_path: Path):
    stub = write_shell_stub(tmp_path / "slow.sh", "sleep 5\n")
    runner = ProcessRunner()
    outcome = runner.run(ExecutionPlan(argv=(str(stub),), cwd=tmp_path, timeout_s=0.5))
    assert outcome.timed_out
    assert outcome.diagnostic.kind is ToolDiagnosticKind.TIMEOUT


@real_process
def test_real_process_reads_stdin_script(tmp_path: Path):
    stub = write_shell_stub(tmp_path / "echo.sh", 'cat > "$1"\n')
    runner = ProcessRunner()
    outcome = runner.run(
        ExecutionPlan(
            argv=(str(stub), "captured.txt"),
            cwd=tmp_path,
            stdin_script="CALC SQUEEZE\nEND\n",
            expected_outputs=("captured.txt",),
        )
    )
    assert outcome.succeeded
    assert (tmp_path / "captured.txt").read_text() == "CALC SQUEEZE\nEND\n"


@real_process
def test_real_process_runs_in_the_given_directory(tmp_path: Path):
    """A program that writes beside its input must not be able to reach the user's originals."""
    work = tmp_path / "iteration"
    work.mkdir()
    stub = write_shell_stub(tmp_path / "pwd.sh", 'pwd > where.txt\n')
    runner = ProcessRunner()
    outcome = runner.run(
        ExecutionPlan(argv=(str(stub),), cwd=work, expected_outputs=("where.txt",))
    )
    assert outcome.succeeded
    assert (work / "where.txt").read_text().strip() == str(work)
    assert not (tmp_path / "where.txt").exists()


@real_process
def test_real_process_no_shell_interpretation(tmp_path: Path):
    """An argument containing shell metacharacters must be passed through literally."""
    stub = write_shell_stub(tmp_path / "arg.sh", 'printf "%s" "$1" > arg.txt\n')
    dangerous = "case; rm -rf /tmp/should-not-happen"
    runner = ProcessRunner()
    outcome = runner.run(
        ExecutionPlan(argv=(str(stub), dangerous), cwd=tmp_path, expected_outputs=("arg.txt",))
    )
    assert outcome.succeeded
    assert (tmp_path / "arg.txt").read_text() == dangerous
