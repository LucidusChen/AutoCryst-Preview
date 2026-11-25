"""Scripted standard input must not hold off the wall-clock timeout or cancellation.

Codex recheck 2026-10-22 section 4: the runner wrote ``stdin_script`` synchronously before its
wait loop, so a program that never read a large script blocked timeout and cancellation. Every
program here is a test-authored ``#!/bin/sh`` script; no crystallographic executable runs.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from crystalx_core.domain.tools import ToolDiagnosticKind
from crystalx_core.tools import runner as runner_module
from crystalx_core.tools.runner import CancelToken, ExecutionPlan, ProcessRunner

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX shell scripts and process groups")

ONE_MIB = "A" * (1 << 20)
#: Far above the pipe buffer, far below anything a test machine would struggle with.
ALLOWANCE_S = 20.0


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _work(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return work


def _is_running(pid: int) -> bool:
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
    import subprocess

    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def _writer_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "crystalx-stdin-writer"]


@pytest.fixture
def short_grace(monkeypatch):
    monkeypatch.setattr(runner_module, "TERMINATE_GRACE_S", 0.5)


def test_a_program_that_never_reads_a_large_script_still_times_out(tmp_path: Path, short_grace):
    work = _work(tmp_path)
    program = _script(tmp_path / "deaf", 'echo $$ > pid\nexec sleep 60\n')
    plan = ExecutionPlan(argv=(str(program),), cwd=work, timeout_s=1.0, stdin_script=ONE_MIB, label="deaf")
    started = time.monotonic()
    outcome = ProcessRunner().run(plan, log_dir=tmp_path / "logs")
    elapsed = time.monotonic() - started
    assert elapsed < ALLOWANCE_S, f"timeout took {elapsed:.1f} s"
    assert outcome.timed_out
    assert outcome.diagnostic.kind is ToolDiagnosticKind.TIMEOUT
    assert not _is_running(int((work / "pid").read_text()))
    assert not _writer_threads()


def test_a_program_that_never_reads_a_large_script_can_be_cancelled(tmp_path: Path, short_grace):
    work = _work(tmp_path)
    program = _script(tmp_path / "deaf", 'echo $$ > pid\nexec sleep 60\n')
    plan = ExecutionPlan(argv=(str(program),), cwd=work, timeout_s=120, stdin_script=ONE_MIB, label="deaf")
    token = CancelToken()
    result: dict = {}
    thread = threading.Thread(
        target=lambda: result.update(outcome=ProcessRunner().run(plan, cancel=token)), daemon=True
    )
    started = time.monotonic()
    thread.start()
    deadline = time.monotonic() + 10
    while not (work / "pid").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    token.cancel("test")
    thread.join(ALLOWANCE_S)
    assert not thread.is_alive(), "cancellation did not return"
    assert time.monotonic() - started < ALLOWANCE_S
    outcome = result["outcome"]
    assert outcome.cancelled
    assert outcome.diagnostic.kind is ToolDiagnosticKind.CANCELLED
    assert outcome.group_cleanup in ("terminated", "killed")
    assert not _is_running(int((work / "pid").read_text()))
    assert not _writer_threads()


def test_a_child_that_ignores_sigterm_and_never_reads_is_still_cleaned_up(tmp_path: Path, short_grace):
    work = _work(tmp_path)
    program = _script(
        tmp_path / "stubborn",
        "trap '' TERM\n"
        "sh -c 'trap \"\" TERM; echo $$ > child; exec sleep 60' &\n"
        "sleep 60\n",
    )
    plan = ExecutionPlan(argv=(str(program),), cwd=work, timeout_s=1.0, stdin_script=ONE_MIB, label="stubborn")
    started = time.monotonic()
    outcome = ProcessRunner().run(plan)
    assert time.monotonic() - started < ALLOWANCE_S
    assert outcome.timed_out
    assert outcome.group_cleanup == "killed"
    assert not _is_running(int((work / "child").read_text()))


def test_a_small_script_is_still_delivered(tmp_path: Path):
    work = _work(tmp_path)
    program = _script(tmp_path / "echoer", "cat\n")
    outcome = ProcessRunner().run(
        ExecutionPlan(argv=(str(program),), cwd=work, timeout_s=20, stdin_script="SQUEEZE\nEND\n", label="echoer"),
        log_dir=tmp_path / "logs",
    )
    assert outcome.succeeded
    assert (tmp_path / "logs" / "echoer.stdout.log").read_text() == "SQUEEZE\nEND\n"


def test_a_large_script_is_delivered_completely_to_a_reader(tmp_path: Path):
    work = _work(tmp_path)
    program = _script(tmp_path / "counter", "wc -c > count\n")
    outcome = ProcessRunner().run(
        ExecutionPlan(argv=(str(program),), cwd=work, timeout_s=30, stdin_script=ONE_MIB, label="counter")
    )
    assert outcome.succeeded
    assert int((work / "count").read_text().split()[0]) == 1 << 20
    assert not _writer_threads()


def test_no_script_closes_stdin_at_once(tmp_path: Path):
    work = _work(tmp_path)
    program = _script(tmp_path / "reader", "cat > got\n")
    outcome = ProcessRunner().run(ExecutionPlan(argv=(str(program),), cwd=work, timeout_s=20))
    assert outcome.succeeded
    assert (work / "got").read_text() == ""
