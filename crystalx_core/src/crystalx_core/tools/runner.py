"""Safe invocation of external crystallographic programs.

Every rule here exists to protect the user's data and to make a run reproducible:

* argument **arrays**, never a shell string, and never ``shell=True``;
* a per-invocation working directory inside the run workspace, so a program that writes beside its
  input cannot touch the originals;
* a timeout, because SHELXD and SHELXE will happily run for hours;
* stdout and stderr captured to files and kept;
* an explicit list of expected output files, so "exited 0 but produced nothing" is caught here
  rather than three stages later;
* optional scripted stdin, because PLATON drives ADDSYM, SQUEEZE, ASYM and LEPAGE from interactive
  instructions rather than command-line flags;
* cancellation that terminates the whole process group, and operating-system resource limits
  (CPU time, largest file written, open files, scheduling priority) recorded with the run.

``CommandExecutor`` is a protocol so that unit tests never need a real program.

## Why resource limits use a launcher process rather than ``preexec_fn``

The Web service runs each job in a worker thread. The Python documentation states that
``preexec_fn`` "is not safe to use in the presence of threads" -- the forked child can deadlock
before ``exec``. Limits are therefore applied by a tiny, constant launcher: a fresh single-threaded
interpreter that calls ``setrlimit`` and then ``execv``-s the tool. No shell is involved, no string
is built from a filename, and because the launcher *replaces itself* with the tool, the process ID,
exit status and terminating signal are the tool's own.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from crystalx_core.domain.tools import ToolDiagnostic, ToolDiagnosticKind, ToolIdentity, ToolKind
from crystalx_core.errors import ToolExecutionError

__all__ = [
    "ExecutionPlan",
    "ExecutionOutcome",
    "CommandExecutor",
    "SubprocessExecutor",
    "ProcessRunner",
    "RawResult",
    "ResourceLimits",
    "CancelToken",
    "DEFAULT_LIMITS",
]

#: Seconds between the polite SIGTERM and the SIGKILL when a process group must be stopped.
TERMINATE_GRACE_S = 3.0

#: How much of a log is kept in memory for classification. The full file stays on disk.
_LOG_TAIL_BYTES = 2_000_000

_IS_POSIX = os.name == "posix"


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


@dataclass(frozen=True)
class ResourceLimits:
    """Operating-system limits applied to one external program.

    Each limit is optional and is recorded exactly as requested, so a provenance record shows what
    a program was *allowed* to do as well as what it did.

    Memory is deliberately absent. ``RLIMIT_AS`` is not enforced on macOS, and offering a limit the
    target platform silently ignores would put a false guarantee into the record.
    """

    cpu_seconds: int | None = None
    """``RLIMIT_CPU``. The kernel sends ``SIGXCPU`` when the program has used this much CPU."""

    max_file_bytes: int | None = None
    """``RLIMIT_FSIZE``. The largest single file the program may write; beyond it, ``SIGXFSZ``."""

    max_open_files: int | None = None
    """``RLIMIT_NOFILE``."""

    nice_increment: int | None = None
    """Added to the scheduling priority in the child. Lowers priority only; never raises it."""

    def __post_init__(self) -> None:
        for name in ("cpu_seconds", "max_file_bytes", "max_open_files"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"ResourceLimits.{name} must be positive, got {value}")
        if self.nice_increment is not None and not 0 <= self.nice_increment <= 19:
            raise ValueError("ResourceLimits.nice_increment must be between 0 and 19")

    @property
    def is_empty(self) -> bool:
        return all(
            getattr(self, n) is None
            for n in ("cpu_seconds", "max_file_bytes", "max_open_files", "nice_increment")
        )

    def to_record(self) -> dict[str, object]:
        return {
            "cpu_seconds": self.cpu_seconds,
            "max_file_bytes": self.max_file_bytes,
            "max_open_files": self.max_open_files,
            "nice_increment": self.nice_increment,
            "memory": "not limited (RLIMIT_AS is not enforced on macOS)",
            "enforced_by": "operating system, via a setrlimit-then-exec launcher" if _IS_POSIX else "not enforced on this platform",
        }

    def describe(self) -> str:
        parts = []
        if self.cpu_seconds is not None:
            parts.append(f"CPU {self.cpu_seconds} s")
        if self.max_file_bytes is not None:
            parts.append(f"largest file {self.max_file_bytes // (1024 * 1024)} MiB")
        if self.max_open_files is not None:
            parts.append(f"open files {self.max_open_files}")
        if self.nice_increment is not None:
            parts.append(f"nice +{self.nice_increment}")
        return ", ".join(parts) or "none"


#: Conservative defaults for a bounded local run. Generous enough for SHELXT on a large cell,
#: small enough that a runaway job cannot fill a disk or occupy a machine indefinitely.
DEFAULT_LIMITS = ResourceLimits(
    cpu_seconds=3600,
    max_file_bytes=2 * 1024 * 1024 * 1024,
    max_open_files=1024,
    nice_increment=5,
)


class CancelToken:
    """A thread-safe request to stop. Shared between a job's caller and its executor."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason = ""

    def cancel(self, reason: str = "cancelled by user") -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)


# A constant program, never formatted with user data. argv[1] is the limits as JSON; the tool's
# own argv follows a literal "--".
_LAUNCHER_SOURCE = r"""
import json, os, resource, sys
limits = json.loads(sys.argv[1])
assert sys.argv[2] == "--"
argv = sys.argv[3:]
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
def _cap(kind, value):
    soft, hard = resource.getrlimit(kind)
    if hard != resource.RLIM_INFINITY:
        value = min(value, hard)
    resource.setrlimit(kind, (value, hard))
if limits.get("cpu_seconds"):
    _cap(resource.RLIMIT_CPU, int(limits["cpu_seconds"]))
if limits.get("max_file_bytes"):
    _cap(resource.RLIMIT_FSIZE, int(limits["max_file_bytes"]))
if limits.get("max_open_files"):
    _cap(resource.RLIMIT_NOFILE, int(limits["max_open_files"]))
if limits.get("nice_increment"):
    os.nice(int(limits["nice_increment"]))
try:
    os.execvp(argv[0], argv)
except OSError as exc:
    # Report a failed exec in a form the parent can turn back into the same os_error it would have
    # seen without the launcher. "Bad CPU type in executable" must stay an architecture diagnosis.
    sys.stderr.write("CRYSTALX_LAUNCH_ERROR " + json.dumps(
        {"errno": exc.errno, "strerror": exc.strerror, "filename": argv[0]}) + "\n")
    sys.stderr.flush()
    os._exit(127)
"""

_LAUNCH_ERROR_MARKER = "CRYSTALX_LAUNCH_ERROR "


def _launcher_argv(argv: Sequence[str], limits: ResourceLimits | None) -> list[str]:
    """Wrap a tool's argv in the limit-applying launcher, or return it unchanged."""
    if limits is None or limits.is_empty or not _IS_POSIX:
        return list(argv)
    payload = json.dumps(
        {
            "cpu_seconds": limits.cpu_seconds,
            "max_file_bytes": limits.max_file_bytes,
            "max_open_files": limits.max_open_files,
            "nice_increment": limits.nice_increment,
        }
    )
    return [sys.executable, "-I", "-S", "-c", _LAUNCHER_SOURCE, payload, "--", *argv]


@dataclass(frozen=True)
class ExecutionPlan:
    """Exactly what will be executed. Serialised to ``command.json`` before the program starts.

    Writing the plan first means that if the process hangs or the machine dies, the record still
    shows what was attempted.
    """

    argv: tuple[str, ...]
    cwd: Path
    timeout_s: float = 600.0
    stdin_script: str | None = None
    """Instruction lines fed to the program's standard input, e.g. ``"CALC ADDSYM\\nEND\\n"``."""

    env_overrides: Mapping[str, str] = field(default_factory=dict)
    env_unset: tuple[str, ...] = ()
    """Variables to remove, e.g. ``DISPLAY`` for a headless PLATON run."""

    expected_outputs: tuple[str, ...] = ()
    """Paths relative to ``cwd`` that the program must produce for the step to count as done."""

    declared_inputs: tuple[str, ...] = ()
    """Paths relative to ``cwd`` the program is expected to read. Hashed before the run so the
    record can prove what went in, not only what came out."""

    label: str = ""
    tool_sha256: str = ""
    tool_kind: ToolKind | None = None
    """Which external program this is. Lets the executor pick the matching provenance adapter; left
    ``None`` by mock backends, which then take the plain runner path."""

    limits: ResourceLimits | None = None
    """Operating-system limits for this invocation. ``None`` means none were requested."""

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("ExecutionPlan.argv must not be empty")
        if self.timeout_s <= 0:
            raise ValueError("ExecutionPlan.timeout_s must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "cwd": str(self.cwd),
            "timeout_s": self.timeout_s,
            "has_stdin_script": self.stdin_script is not None,
            "stdin_script": self.stdin_script,
            "env_overrides": dict(self.env_overrides),
            "env_unset": list(self.env_unset),
            "expected_outputs": list(self.expected_outputs),
            "declared_inputs": list(self.declared_inputs),
            "label": self.label,
            "tool_sha256": self.tool_sha256,
            "tool_kind": self.tool_kind.value if self.tool_kind else None,
            "limits": self.limits.to_record() if self.limits else None,
        }


@dataclass(frozen=True)
class RawResult:
    """What the executor observed. No interpretation."""

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    os_error: str = ""
    cancelled: bool = False
    terminated_by_signal: int | None = None
    """The signal that ended the process, when it did not exit normally."""

    logs_written: bool = False
    """True when the executor already streamed stdout/stderr to the requested log files."""

    group_cleanup: str = "none"
    """What had to be done to the program's process group at the end: ``none``, ``terminated``
    or ``killed`` (members survived SIGTERM, or were left running after the program exited)."""


@dataclass(frozen=True)
class ExecutionOutcome:
    """Interpreted result of one invocation."""

    plan: ExecutionPlan
    diagnostic: ToolDiagnostic
    exit_code: int | None
    started_at: str
    ended_at: str
    duration_s: float
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    produced_outputs: tuple[str, ...] = ()
    missing_outputs: tuple[str, ...] = ()
    timed_out: bool = False
    cancelled: bool = False
    terminated_by_signal: int | None = None
    group_cleanup: str = "none"

    @property
    def succeeded(self) -> bool:
        return self.diagnostic.kind.is_usable

    def to_record(self) -> dict[str, object]:
        return {
            "diagnostic": self.diagnostic.kind.value,
            "message": self.diagnostic.message,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "stdout_path": str(self.stdout_path) if self.stdout_path else None,
            "stderr_path": str(self.stderr_path) if self.stderr_path else None,
            "produced_outputs": list(self.produced_outputs),
            "missing_outputs": list(self.missing_outputs),
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "terminated_by_signal": self.terminated_by_signal,
            "group_cleanup": self.group_cleanup,
            "limits": self.plan.limits.to_record() if self.plan.limits else None,
        }


class CommandExecutor(Protocol):
    """Runs a plan and reports what happened. Substituted wholesale in unit tests."""

    def execute(self, plan: ExecutionPlan, env: Mapping[str, str]) -> RawResult: ...


class SubprocessExecutor:
    """The real executor. ``shell=False`` always.

    Output goes straight to the log files rather than through pipes. That keeps a long SHELXT run
    from blocking on a full pipe buffer, and lets a user interface tail the log while the program
    is still running.
    """

    #: Accepts ``cancel``/``stdout_path``/``stderr_path``. Executors without it get the legacy call.
    supports_control = True

    #: How often the wait loop checks for cancellation and the wall-clock deadline.
    poll_interval_s = 0.2

    def execute(
        self,
        plan: ExecutionPlan,
        env: Mapping[str, str],
        *,
        cancel: CancelToken | None = None,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
    ) -> RawResult:
        argv = _launcher_argv(plan.argv, plan.limits)
        owns_files = stdout_path is None
        if owns_files:
            scratch = Path(tempfile.mkdtemp(prefix="crystalx-run-"))
            stdout_path = scratch / "stdout.log"
            stderr_path = scratch / "stderr.log"
        assert stdout_path is not None and stderr_path is not None
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)

        timed_out = cancelled = False
        with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
            try:
                process = subprocess.Popen(  # noqa: S603 - argv array, shell disabled on purpose
                    argv,
                    cwd=str(plan.cwd),
                    env=dict(env),
                    stdin=subprocess.PIPE,
                    stdout=out,
                    stderr=err,
                    shell=False,
                    start_new_session=_IS_POSIX,
                )
            except OSError as exc:
                # errno 8 (ENOEXEC) / macOS "Bad CPU type in executable" lands here when no
                # launcher is in use.
                return RawResult(exit_code=None, stdout="", stderr="", os_error=str(exc))

            deadline = time.monotonic() + plan.timeout_s
            writer = _feed_stdin(process, plan.stdin_script)
            cleanup = "none"
            while True:
                try:
                    process.wait(timeout=self.poll_interval_s)
                    # The program has exited; anything it left behind in its group is stopped too.
                    cleanup = _terminate_group(process)
                    break
                except subprocess.TimeoutExpired:
                    if cancel is not None and cancel.cancelled:
                        cancelled = True
                        cleanup = _terminate_group(process)
                        break
                    if time.monotonic() >= deadline:
                        timed_out = True
                        cleanup = _terminate_group(process)
                        break

        if writer is not None:
            # Normally immediate: the group is gone, so the pipe has no reader. A writer still
            # blocked here (a process escaped the group and keeps the pipe open) is a daemon
            # thread and is left alone; closing the stream under it could deadlock on its lock.
            writer.join(STDIN_WRITER_JOIN_S)

        stdout = _read_tail(stdout_path)
        stderr = _read_tail(stderr_path)
        returncode = process.returncode
        signal_number = -returncode if returncode is not None and returncode < 0 else None

        os_error = ""
        if returncode == 127 and _LAUNCH_ERROR_MARKER in stderr:
            os_error = _decode_launch_error(stderr)

        result = RawResult(
            exit_code=None if (os_error or timed_out or cancelled or signal_number) else returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            os_error=os_error,
            cancelled=cancelled,
            terminated_by_signal=signal_number,
            logs_written=not owns_files,
            group_cleanup=cleanup,
        )
        return result


#: How long to wait for the stdin writer after the program has ended or been stopped. Once every
#: member of the process group is gone the pipe has no reader, so a blocked write fails at once.
STDIN_WRITER_JOIN_S = 5.0


def _close_quietly(stream) -> None:
    try:
        stream.close()
    except (BrokenPipeError, OSError, ValueError):
        pass


def _feed_stdin(process: subprocess.Popen, script: str | None) -> threading.Thread | None:
    """Deliver ``script`` to the program's standard input without blocking the control loop.

    A pipe holds only a few tens of kilobytes. Writing a larger script synchronously would block
    until the program reads it, and a program that never reads would then hold off the wall-clock
    timeout and cancellation for ever. The write therefore happens on a daemon thread; the caller's
    wait loop starts at once. When the program ends or its group is stopped, the pipe loses its
    reader and the pending write fails with a broken pipe, which ends the thread. Returns the thread
    (``None`` when there is nothing to write; stdin is then simply closed).
    """
    stream = process.stdin
    if stream is None:
        return None
    data = script.encode("utf-8") if script else b""
    if not data:
        _close_quietly(stream)
        return None

    def write() -> None:
        try:
            stream.write(data)
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            _close_quietly(stream)

    writer = threading.Thread(target=write, name="crystalx-stdin-writer", daemon=True)
    writer.start()
    return writer


def _group_members_alive(pgid: int) -> bool:
    """True while any process of the group is still running. Zombies do not count.

    On Linux this reads ``/proc`` so that an orphan's unreaped zombie (common in containers whose
    init does not reap) is not mistaken for a live process. Elsewhere a signal-0 probe is used; on
    macOS orphans are reaped by launchd promptly.
    """
    if sys.platform.startswith("linux") and os.path.isdir("/proc"):
        wanted = str(pgid)
        try:
            entries = os.listdir("/proc")
        except OSError:
            entries = []
        for entry in entries:
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat", "rb") as handle:
                    fields = handle.read().decode("ascii", "replace").rsplit(")", 1)[-1].split()
            except OSError:
                continue
            if len(fields) >= 3 and fields[2] == wanted and fields[0] not in ("Z", "X"):
                return True
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - a member we may not signal still exists
        return True
    return True


def _terminate_group(process: subprocess.Popen) -> str:
    """Stop the program **and every other member of its process group**.

    SIGTERM to the group, then wait up to :data:`TERMINATE_GRACE_S` for *all* members -- not just
    the leader -- to go; any survivor (for example a child that ignores SIGTERM after its parent
    has exited) then gets SIGKILL. Returns ``"none"`` (nothing was left), ``"terminated"`` or
    ``"killed"``. POSIX process groups only; on other platforms only the direct child is stopped.
    """
    if not _IS_POSIX:  # pragma: no cover - non-POSIX hosts: no process-group control
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=TERMINATE_GRACE_S)
                return "terminated"
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return "killed"
        return "none"

    pgid = process.pid  # start_new_session=True makes the child its own group leader
    process.poll()
    if process.returncode is not None and not _group_members_alive(pgid):
        return "none"
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + TERMINATE_GRACE_S
    while time.monotonic() < deadline:
        process.poll()
        if process.returncode is not None and not _group_members_alive(pgid):
            return "terminated"
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
    settle = time.monotonic() + 2.0
    while time.monotonic() < settle and _group_members_alive(pgid):
        time.sleep(0.05)
    return "killed"


def _read_tail(path: Path) -> str:
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            if size > _LOG_TAIL_BYTES:
                handle.seek(size - _LOG_TAIL_BYTES)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _decode_launch_error(stderr: str) -> str:
    for line in stderr.splitlines():
        if line.startswith(_LAUNCH_ERROR_MARKER):
            try:
                info = json.loads(line[len(_LAUNCH_ERROR_MARKER):])
            except ValueError:
                break
            return f"[Errno {info.get('errno')}] {info.get('strerror')}: {info.get('filename')!r}"
    return "the launcher could not execute the program"


class ProcessRunner:
    """Executes plans, writes the logs, and classifies the result."""

    def __init__(
        self,
        executor: CommandExecutor | None = None,
        *,
        base_env: Mapping[str, str] | None = None,
        clock: Callable[[], _dt.datetime] = _utc_now,
    ) -> None:
        self.executor = executor or SubprocessExecutor()
        self.base_env = dict(base_env if base_env is not None else os.environ)
        self._clock = clock

    def build_env(self, plan: ExecutionPlan) -> dict[str, str]:
        env = dict(self.base_env)
        for name in plan.env_unset:
            env.pop(name, None)
        env.update(plan.env_overrides)
        return env

    def run(
        self,
        plan: ExecutionPlan,
        *,
        log_dir: Path | None = None,
        cancel: CancelToken | None = None,
    ) -> ExecutionOutcome:
        """Run one plan. Never raises for a program failure; the failure is in the outcome.

        ``cancel`` lets another thread stop the program; the outcome is then ``CANCELLED`` and the
        logs written up to that point are kept.
        """
        if not plan.cwd.is_dir():
            raise ToolExecutionError(
                f"Working directory does not exist: {plan.cwd}",
                diagnostic=ToolDiagnosticKind.NOT_FOUND.value,
                evidence_paths=[plan.cwd],
            )

        stdout_path: Path | None = None
        stderr_path: Path | None = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            stem = plan.label or Path(plan.argv[0]).name
            stdout_path = log_dir / f"{stem}.stdout.log"
            stderr_path = log_dir / f"{stem}.stderr.log"

        started = self._clock()
        if cancel is not None and cancel.cancelled:
            raw = RawResult(exit_code=None, stdout="", stderr="", cancelled=True)
        elif getattr(self.executor, "supports_control", False):
            raw = self.executor.execute(
                plan,
                self.build_env(plan),
                cancel=cancel,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        else:
            raw = self.executor.execute(plan, self.build_env(plan))
        ended = self._clock()
        duration = (ended - started).total_seconds()

        if stdout_path is not None and stderr_path is not None and not raw.logs_written:
            stdout_path.write_text(raw.stdout, encoding="utf-8")
            stderr_path.write_text(raw.stderr, encoding="utf-8")

        produced: list[str] = []
        missing: list[str] = []
        for relative in plan.expected_outputs:
            target = plan.cwd / relative
            (produced if target.exists() else missing).append(relative)

        diagnostic = self._classify(plan, raw, missing)
        return ExecutionOutcome(
            plan=plan,
            diagnostic=diagnostic,
            exit_code=raw.exit_code,
            started_at=started.isoformat(),
            ended_at=ended.isoformat(),
            duration_s=duration,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            produced_outputs=tuple(produced),
            missing_outputs=tuple(missing),
            timed_out=raw.timed_out,
            cancelled=raw.cancelled,
            terminated_by_signal=raw.terminated_by_signal,
            group_cleanup=raw.group_cleanup,
        )

    def _classify(
        self, plan: ExecutionPlan, raw: RawResult, missing: Sequence[str]
    ) -> ToolDiagnostic:
        program = plan.argv[0]
        if raw.cancelled:
            return ToolDiagnostic(
                kind=ToolDiagnosticKind.CANCELLED,
                message=f"{program} was stopped on request; its process group was terminated",
                path=Path(program),
            )
        if raw.timed_out:
            return ToolDiagnostic(
                kind=ToolDiagnosticKind.TIMEOUT,
                message=f"{program} exceeded its {plan.timeout_s:g}s budget and was stopped",
                path=Path(program),
                remedy="Raise the per-branch time budget, or reduce the number of solver trials.",
            )
        if raw.os_error:
            lowered = raw.os_error.lower()
            if "exec format" in lowered or "bad cpu type" in lowered:
                return ToolDiagnostic(
                    kind=ToolDiagnosticKind.ARCH_MISMATCH,
                    message=(
                        f"The operating system refused to execute {program}: {raw.os_error}. "
                        "This is an architecture problem, not a missing file."
                    ),
                    path=Path(program),
                    remedy=(
                        "On Apple Silicon install Rosetta 2 with "
                        "'softwareupdate --install-rosetta --agree-to-license', or obtain a native build."
                    ),
                )
            if "permission" in lowered:
                return ToolDiagnostic(
                    kind=ToolDiagnosticKind.NOT_EXECUTABLE,
                    message=f"Permission denied executing {program}: {raw.os_error}",
                    path=Path(program),
                    remedy=f"chmod +x {program}",
                )
            return ToolDiagnostic(
                kind=ToolDiagnosticKind.NOT_FOUND,
                message=f"Could not start {program}: {raw.os_error}",
                path=Path(program),
            )
        limit_hit = _signal_to_limit(raw.terminated_by_signal, plan.limits)
        if limit_hit:
            return ToolDiagnostic(
                kind=ToolDiagnosticKind.RESOURCE_LIMIT,
                message=f"{program} was stopped by the operating system: {limit_hit}",
                path=Path(program),
                remedy="Raise the corresponding limit only if the run is expected to need it.",
            )
        if raw.terminated_by_signal is not None:
            return ToolDiagnostic(
                kind=ToolDiagnosticKind.NONZERO_EXIT,
                message=f"{program} was terminated by signal {raw.terminated_by_signal}",
                path=Path(program),
            )
        if raw.exit_code != 0:
            tail = (raw.stderr or raw.stdout or "").strip().splitlines()[-3:]
            detail = " | ".join(tail) if tail else "no diagnostic output"
            return ToolDiagnostic(
                kind=ToolDiagnosticKind.NONZERO_EXIT,
                message=f"{program} exited with status {raw.exit_code}: {detail}",
                path=Path(program),
            )
        if missing:
            return ToolDiagnostic(
                kind=ToolDiagnosticKind.EXPECTED_OUTPUT_MISSING,
                message=(
                    f"{program} exited successfully but did not produce: {', '.join(missing)}. "
                    "Treating a silent no-op as success is how a wrong structure gets through."
                ),
                path=Path(program),
            )
        return ToolDiagnostic(
            kind=ToolDiagnosticKind.OK,
            message=f"{program} completed and produced all expected outputs",
            path=Path(program),
        )


def _signal_to_limit(signal_number: int | None, limits: ResourceLimits | None) -> str:
    """Name the limit a terminating signal corresponds to, or return an empty string."""
    if signal_number is None or not _IS_POSIX:
        return ""
    if signal_number == getattr(signal, "SIGXCPU", -1):
        budget = limits.cpu_seconds if limits else None
        return f"CPU-time limit exceeded ({budget} s)" if budget else "CPU-time limit exceeded"
    if signal_number == getattr(signal, "SIGXFSZ", -1):
        budget = limits.max_file_bytes if limits else None
        return (
            f"tried to write a file larger than the {budget}-byte limit"
            if budget
            else "file-size limit exceeded"
        )
    return ""


def plan_for_tool(
    identity: ToolIdentity,
    *,
    args: Sequence[str],
    cwd: Path,
    timeout_s: float = 600.0,
    stdin_script: str | None = None,
    expected_outputs: Sequence[str] = (),
    env_overrides: Mapping[str, str] | None = None,
    env_unset: Sequence[str] = (),
    label: str = "",
) -> ExecutionPlan:
    """Build a plan from a resolved tool identity, pinning the executable's digest."""
    return ExecutionPlan(
        argv=(str(identity.path), *args),
        cwd=cwd,
        timeout_s=timeout_s,
        stdin_script=stdin_script,
        env_overrides=dict(env_overrides or {}),
        env_unset=tuple(env_unset),
        expected_outputs=tuple(expected_outputs),
        label=label or identity.kind.value,
        tool_sha256=identity.sha256,
    )


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
