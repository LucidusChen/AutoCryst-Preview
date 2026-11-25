"""Cancellation, wall-clock timeout and operating-system resource limits.

Programs here are test-authored shell scripts or an explicit invalid-format refusal fixture.
No crystallographic executable runs.
These are real processes on purpose: the behaviour under test -- process groups, signals, kernel
limits -- cannot be faked meaningfully.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from crystalx_core.domain.tools import ToolDiagnosticKind
from crystalx_core.tools.runner import (
    CancelToken,
    ExecutionPlan,
    ProcessRunner,
    ResourceLimits,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process groups and rlimits are POSIX")


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _is_running(pid: int) -> bool:
    """True only for a live process. A zombie has already died and is merely awaiting reaping.

    ``kill(pid, 0)`` succeeds on a zombie, and in a container whose PID 1 does not reap orphans a
    killed grandchild can stay a zombie indefinitely. Checking for "exists" would therefore report
    a correctly terminated process as still running.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        return stat.read_text().split()[2] != "Z"
    import subprocess

    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def _plan(tmp_path: Path, program: Path, **kwargs) -> ExecutionPlan:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return ExecutionPlan(argv=(str(program),), cwd=work, label="probe", **kwargs)


def test_cancellation_stops_the_program_and_everything_it_started(tmp_path: Path):
    """The whole process group goes, including a grandchild the program spawned."""
    marker = tmp_path / "child.pid"
    program = _script(
        tmp_path / "slow.sh",
        f'sleep 60 & echo $! > "{marker}"\necho started\nwait\n',
    )
    token = CancelToken()
    plan = _plan(tmp_path, program, timeout_s=60)

    def cancel_soon():
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        token.cancel("test asked to stop")

    threading.Thread(target=cancel_soon, daemon=True).start()
    started = time.monotonic()
    outcome = ProcessRunner().run(plan, log_dir=tmp_path / "logs", cancel=token)
    elapsed = time.monotonic() - started

    assert outcome.diagnostic.kind is ToolDiagnosticKind.CANCELLED
    assert outcome.cancelled
    assert elapsed < 15, "cancellation must not wait for the program to finish"
    child = int(marker.read_text().strip())
    time.sleep(0.2)
    assert not _is_running(child), "the grandchild sleep was in the same group and must be dead"
    assert (tmp_path / "logs" / "probe.stdout.log").read_text().strip() == "started"


def test_a_token_cancelled_before_start_runs_nothing(tmp_path: Path):
    witness = tmp_path / "ran"
    program = _script(tmp_path / "touch.sh", f'touch "{witness}"\n')
    token = CancelToken()
    token.cancel()
    outcome = ProcessRunner().run(_plan(tmp_path, program), cancel=token)
    assert outcome.diagnostic.kind is ToolDiagnosticKind.CANCELLED
    assert not witness.exists()


def test_the_wall_clock_timeout_kills_the_process_group(tmp_path: Path):
    program = _script(tmp_path / "hang.sh", "sleep 60\n")
    started = time.monotonic()
    outcome = ProcessRunner().run(_plan(tmp_path, program, timeout_s=0.5))
    assert outcome.diagnostic.kind is ToolDiagnosticKind.TIMEOUT
    assert time.monotonic() - started < 10


def test_the_cpu_limit_is_enforced_by_the_kernel(tmp_path: Path):
    program = _script(tmp_path / "spin.sh", "while :; do :; done\n")
    plan = _plan(tmp_path, program, timeout_s=30, limits=ResourceLimits(cpu_seconds=1))
    outcome = ProcessRunner().run(plan)
    assert outcome.diagnostic.kind is ToolDiagnosticKind.RESOURCE_LIMIT
    assert "CPU-time limit" in outcome.diagnostic.message
    assert outcome.to_record()["limits"]["cpu_seconds"] == 1


def test_the_file_size_limit_is_enforced_by_the_kernel(tmp_path: Path):
    program = _script(
        tmp_path / "fill.sh",
        "dd if=/dev/zero of=big.bin bs=1024 count=4096 2>/dev/null\n",
    )
    plan = _plan(tmp_path, program, limits=ResourceLimits(max_file_bytes=512 * 1024))
    outcome = ProcessRunner().run(plan)
    # dd itself is killed by SIGXFSZ; the shell then reports a non-zero status. Either way the
    # oversized file must not exist at full size.
    written = (plan.cwd / "big.bin").stat().st_size if (plan.cwd / "big.bin").exists() else 0
    assert written <= 512 * 1024
    assert outcome.diagnostic.kind in (
        ToolDiagnosticKind.RESOURCE_LIMIT,
        ToolDiagnosticKind.NONZERO_EXIT,
    )


def test_an_exec_format_refusal_is_still_an_architecture_diagnosis_under_limits(tmp_path: Path):
    """Regression guard for the launcher.

    Without care, a failed ``exec`` inside the limit-applying launcher would surface as a Python
    traceback and a generic non-zero exit, and the original exec-format error would stop being
    recognised. Bad-CPU errors are separately covered in test_runner.py with scripted outcomes.
    """
    fake = tmp_path / "foreign"
    # Malformed x86_64 Mach-O can instead reach Rosetta and receive SIGKILL. This
    # non-shebang invalid image deterministically exercises the kernel's ENOEXEC.
    fake.write_bytes(b"not an executable image\n")
    fake.chmod(0o755)
    plan = _plan(tmp_path, fake, limits=ResourceLimits(cpu_seconds=5))
    outcome = ProcessRunner().run(plan)
    assert outcome.diagnostic.kind is ToolDiagnosticKind.ARCH_MISMATCH, outcome.diagnostic.message


def test_logs_are_on_disk_while_the_program_is_still_running(tmp_path: Path):
    """What lets a user interface show progress during a long SHELXT run."""
    program = _script(tmp_path / "talk.sh", "echo early\nsleep 2\necho late\n")
    log_dir = tmp_path / "logs"
    seen_early: list[str] = []

    def peek():
        log = log_dir / "probe.stdout.log"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if log.exists() and "early" in log.read_text():
                seen_early.append(log.read_text())
                return
            time.sleep(0.05)

    watcher = threading.Thread(target=peek, daemon=True)
    watcher.start()
    outcome = ProcessRunner().run(_plan(tmp_path, program, timeout_s=10), log_dir=log_dir)
    watcher.join(timeout=1)
    assert outcome.succeeded
    assert seen_early and "late" not in seen_early[0]


def test_limits_are_recorded_even_when_the_program_succeeds(tmp_path: Path):
    program = _script(tmp_path / "ok.sh", "echo fine\n")
    limits = ResourceLimits(cpu_seconds=60, max_file_bytes=10 * 1024 * 1024, nice_increment=5)
    outcome = ProcessRunner().run(_plan(tmp_path, program, limits=limits))
    assert outcome.succeeded
    record = outcome.to_record()["limits"]
    assert record["cpu_seconds"] == 60 and record["nice_increment"] == 5
    assert "not enforced on macOS" in record["memory"]


def test_invalid_limits_are_refused():
    with pytest.raises(ValueError):
        ResourceLimits(cpu_seconds=0)
    with pytest.raises(ValueError):
        ResourceLimits(nice_increment=-1)


def test_the_launcher_never_builds_a_shell_string(tmp_path: Path):
    """A filename full of shell metacharacters must arrive as one literal argument."""
    weird = "a; rm -rf ~ $(echo pwned) 'q'"
    program = _script(tmp_path / "echo.sh", 'printf "%s" "$1" > arg.txt\n')
    plan = ExecutionPlan(
        argv=(str(program), weird),
        cwd=tmp_path,
        limits=ResourceLimits(cpu_seconds=5),
    )
    outcome = ProcessRunner().run(plan)
    assert outcome.succeeded
    assert (tmp_path / "arg.txt").read_text() == weird


def test_the_launcher_uses_the_running_interpreter_in_isolated_mode():
    from crystalx_core.tools.runner import _launcher_argv

    wrapped = _launcher_argv(["/bin/true"], ResourceLimits(cpu_seconds=1))
    assert wrapped[0] == sys.executable
    assert "-I" in wrapped and wrapped[-2:] == ["--", "/bin/true"]
    assert _launcher_argv(["/bin/true"], None) == ["/bin/true"]
