"""Append-only structured event log.

One JSON object per line, flushed on every write. If the process dies mid-run the log is still a
valid, readable record up to the last completed line -- which is the point: a crashed run must leave
evidence, not an empty directory.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Callable, Iterator

__all__ = ["EventLog", "EventKind", "read_events"]


class EventKind:
    """Well-known event names.

    A plain namespace rather than an enum so that adapters can emit their own names without
    modifying the core, while the common ones stay spelled consistently.
    """

    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    INPUT_STAGED = "input.staged"
    STAGE_STARTED = "stage.started"
    STAGE_FINISHED = "stage.finished"
    STAGE_ROLLED_BACK = "stage.rolled_back"
    BRANCH_NOMINATED = "branch.nominated"
    BRANCH_STATE_CHANGED = "branch.state_changed"
    BRANCH_NOT_APPLICABLE = "branch.not_applicable"
    TOOL_RESOLVED = "tool.resolved"
    TOOL_UNAVAILABLE = "tool.unavailable"
    COMMAND_STARTED = "command.started"
    COMMAND_FINISHED = "command.finished"
    CANDIDATE_CREATED = "candidate.created"
    CANDIDATE_STATUS_CHANGED = "candidate.status_changed"
    LOCK_APPLIED = "lock.applied"
    GATE_EVALUATED = "gate.evaluated"
    REVIEW_CHECKPOINT = "review.checkpoint"
    POLICY_STOP = "policy.stop"
    WARNING = "warning"


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class EventLog:
    """Append-only NDJSON writer."""

    def __init__(self, path: Path | str, *, clock: Callable[[], _dt.datetime] = _utc_now) -> None:
        self.path = Path(path)
        self._clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def emit(self, kind: str, message: str = "", /, **fields: Any) -> dict[str, Any]:
        """Append one event and return it.

        ``kind`` and ``message`` are positional-only so that a caller may pass a field literally
        named ``kind`` (a stage kind, a solver kind) without colliding with the parameter.
        """
        record: dict[str, Any] = {
            "ts": self._clock().isoformat(),
            "kind": kind,
        }
        if message:
            record["message"] = message
        for key, value in fields.items():
            if value is None:
                continue
            record[key] = _jsonable(value)
        line = json.dumps(record, ensure_ascii=False, sort_keys=False)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        return record

    def read(self) -> list[dict[str, Any]]:
        return list(read_events(self.path))

    def __len__(self) -> int:
        return sum(1 for _ in read_events(self.path))


def read_events(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield events, skipping a torn final line rather than raising.

    A partially written last line is the normal signature of a crash; the rest of the log is still
    good and must remain readable.
    """
    target = Path(path)
    if not target.exists():
        return
    with open(target, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _jsonable(value: Any) -> Any:
    from crystalx_core._serde import to_jsonable

    try:
        return to_jsonable(value)
    except TypeError:
        return str(value)
