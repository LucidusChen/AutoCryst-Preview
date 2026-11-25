"""Immutable per-run workspace: staging, hashing, append-only events, candidate lineage."""

from __future__ import annotations

from crystalx_core.workspace.events import EventKind, EventLog, read_events
from crystalx_core.workspace.hashing import (
    FileDigest,
    digest_dir,
    manifest_for,
    sha256_bytes,
    sha256_file,
    sha256_object,
    sha256_text,
)
from crystalx_core.workspace.layout import RunLayout
from crystalx_core.workspace.store import RunWorkspace, StagedInput

__all__ = [
    "EventKind",
    "EventLog",
    "FileDigest",
    "RunLayout",
    "RunWorkspace",
    "StagedInput",
    "digest_dir",
    "manifest_for",
    "read_events",
    "sha256_bytes",
    "sha256_file",
    "sha256_object",
    "sha256_text",
]
