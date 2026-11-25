"""Opt-in launch probe: does this executable actually start on this machine?

This **runs the user's program**, so it never happens implicitly: :func:`probe_launch` refuses unless
``authorize=True`` is passed (the CLI flag is ``--authorize-launch``), and the Web interface never
calls it. The program gets an empty, isolated working directory, no arguments, a closed standard
input, ``DISPLAY`` and ``CHECKDEF`` unset, a short wall-clock limit and process-group cleanup.

What it can establish is only "the operating system started the program" -- or exactly why not
(wrong CPU, missing library, killed at start). It does **not** establish that the program computes
anything correctly; that needs a real run on authorised data.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from crystalx_core.domain.tools import BinaryArch, BinaryFormat, ToolDiagnosticKind
from crystalx_core.tools.binfmt import probe_binary
from crystalx_core.tools.runner import ExecutionPlan, ProcessRunner, ResourceLimits
from crystalx_core.workspace.hashing import sha256_file

from crystalx_native_tools.host import HostFacts, detect_host
from crystalx_native_tools.macho import read_macho

__all__ = ["ProbeRefused", "LaunchProbe", "probe_launch", "ProbeStore", "PROBE_SCHEMA"]

PROBE_SCHEMA = "crystalx.launch-probe/1"
_SAFE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


class ProbeRefused(RuntimeError):
    """The probe was not run (not authorised, file changed, or policy forbids it)."""


@dataclass(frozen=True)
class LaunchProbe:
    schema: str
    tool: str
    path: str
    sha256: str
    arches: tuple[str, ...]
    probed_at: str
    outcome: str
    """``launched``, ``running_when_stopped``, ``refused``, ``library_failure``, ``killed_at_start``
    or ``error``."""

    exit_code: int | None
    signal: int | None
    duration_s: float
    diagnostic: str
    message: str
    stdout_tail: str
    stderr_tail: str
    host: dict

    @property
    def started(self) -> bool:
        return self.outcome in ("launched", "running_when_stopped")

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["arches"] = list(self.arches)
        record["started"] = self.started
        record["meaning"] = (
            "whether the operating system started the program; not whether it computes correctly"
        )
        return record


def _tail(path: Path | None, limit: int = 2000) -> str:
    if path is None or not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def _classify(outcome, stderr: str) -> str:
    kind = outcome.diagnostic.kind
    if kind in (ToolDiagnosticKind.ARCH_MISMATCH, ToolDiagnosticKind.ROSETTA_MISSING,
                ToolDiagnosticKind.NOT_EXECUTABLE, ToolDiagnosticKind.NOT_FOUND):
        return "refused"
    if outcome.timed_out:
        return "running_when_stopped"
    lowered = stderr.lower()
    if "dyld" in lowered and ("library not loaded" in lowered or "symbol not found" in lowered):
        return "library_failure"
    if outcome.terminated_by_signal == 9 and outcome.duration_s < 2.0:
        return "killed_at_start"
    if outcome.exit_code is not None:
        return "launched"
    if outcome.terminated_by_signal is not None:
        return "launched"
    return "error"


def probe_launch(
    tool: str,
    path: Path | str,
    *,
    authorize: bool,
    expected_sha256: str | None = None,
    allow_translation: bool = False,
    timeout_s: float = 10.0,
    host: HostFacts | None = None,
    work_root: Path | None = None,
) -> LaunchProbe:
    if not authorize:
        raise ProbeRefused(
            "the launch probe runs the program; pass authorize=True (CLI: --authorize-launch) to allow it"
        )
    target = Path(os.path.abspath(Path(path).expanduser()))
    if not target.is_file():
        raise ProbeRefused(f"{target} is not a file")
    digest = sha256_file(target)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ProbeRefused(f"{target} changed since it was inspected (sha256 {digest[:16]}); not started")
    host = host or detect_host()
    macho = read_macho(target)
    arches = macho.arches if macho is not None else ()
    if host.is_apple_silicon and macho is not None and not macho.has_arm64 and not allow_translation:
        raise ProbeRefused(
            f"{target} has no arm64 code; strict native mode does not probe it through Rosetta 2"
        )
    kind = probe_binary(target)
    unverifiable = kind.fmt is BinaryFormat.SCRIPT or (kind.fmt is BinaryFormat.UNKNOWN and kind.arch is BinaryArch.UNKNOWN)
    if unverifiable and not allow_translation:
        what = "a script" if kind.fmt is BinaryFormat.SCRIPT else "a file of unrecognised format"
        raise ProbeRefused(
            f"{target} is {what}; strict native mode cannot verify what it would run (a wrapper may start "
            "an x86_64 program through Rosetta 2), so it is not probed. Probe the native executable itself."
        )

    root = Path(tempfile.mkdtemp(prefix=f"crystalx-probe-{tool}-", dir=str(work_root) if work_root else None))
    work = root / "work"
    work.mkdir()
    plan = ExecutionPlan(
        argv=(str(target),),
        cwd=work,
        timeout_s=timeout_s,
        env_unset=("DISPLAY", "CHECKDEF"),
        label=f"probe-{tool}",
        limits=ResourceLimits(cpu_seconds=max(1, int(timeout_s * 2)), max_file_bytes=64 * 1024 * 1024),
    )
    runner = ProcessRunner(base_env={"PATH": _SAFE_PATH, "HOME": str(root), "LANG": "C", "LC_ALL": "C"})
    outcome = runner.run(plan, log_dir=root / "logs")
    stdout = _tail(outcome.stdout_path)
    stderr = _tail(outcome.stderr_path)
    return LaunchProbe(
        schema=PROBE_SCHEMA,
        tool=tool,
        path=str(target),
        sha256=digest,
        arches=tuple(arches),
        probed_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        outcome=_classify(outcome, stderr),
        exit_code=outcome.exit_code,
        signal=outcome.terminated_by_signal,
        duration_s=round(outcome.duration_s, 3),
        diagnostic=outcome.diagnostic.kind.value,
        message=outcome.diagnostic.message,
        stdout_tail=stdout,
        stderr_tail=stderr,
        host=host.to_record(),
    )


class ProbeStore:
    """Append-only JSON file of probe records (kept in an ignored state directory)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [r for r in data if isinstance(r, dict) and r.get("schema") == PROBE_SCHEMA] if isinstance(data, list) else []

    def append(self, probe: LaunchProbe) -> None:
        records = self.load()[-199:]
        records.append(probe.to_record())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(records, indent=1), encoding="utf-8")
        os.replace(temporary, self.path)

    def latest(self, tool: str, sha256: str | None) -> dict | None:
        """The newest record for this tool **and this exact file**; a changed file has no record."""
        if not sha256:
            return None
        for record in reversed(self.load()):
            if record.get("tool") == tool and record.get("sha256") == sha256:
                return record
        return None
