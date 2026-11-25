"""Bounded local privacy audit; detailed findings must stay under ignored runs/.

Filename/substring checks are a review aid, not proof that arbitrary paraphrased scientific
content is absent. This module never opens coordinate/measurement files or starts tools.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _git(repository: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=repository, capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Git privacy inspection failed")
    return result.stdout


def _strings(value: Any):
    if isinstance(value, str):
        if len(value.strip()) >= 24:
            yield value.strip()
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def audit_repository(repository: Path) -> dict[str, Any]:
    """Inspect Git paths/history and exact long metadata strings, without printing content."""
    repo = repository.resolve(strict=True)
    entries = _git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    names = sorted(set(p.decode("utf-8", errors="surrogateescape") for p in entries.split(b"\0") if p))
    forbidden = [n for n in names if n.startswith(("realdata_examples/", "runs/"))]
    history = _git(repo, "log", "--all", "--format=%H", "--", "realdata_examples").decode().splitlines()
    remotes = _git(repo, "remote").decode().splitlines()
    signatures: set[str] = set()
    archive = repo / "realdata_examples"
    for path in archive.rglob("case.json"):
        if path.is_symlink():
            continue
        try:
            path.resolve().relative_to(archive.resolve())
            value = json.loads(path.read_text(encoding="utf-8"))
            signatures.update(_strings(value))
        except (ValueError, OSError):
            continue
    hits = []
    for name in names:
        if name.startswith(("realdata_examples/", "runs/")):
            continue
        path = repo / name
        if path.suffix.lower() not in {".md", ".py", ".json", ".sh", ".txt", ".toml"}:
            continue
        try:
            if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
                continue
            contents = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        count = sum(1 for value in signatures if value in contents)
        if count:
            hits.append({"path": name, "matching_metadata_string_count": count})
    return {"schema_version": 1,
            "current_forbidden_paths": forbidden,
            "private_archive_history_commits": sorted(set(history)),
            "configured_remote_count": len(remotes),
            "exact_metadata_copy_candidates": hits,
            "status": "REVIEW_REQUIRED" if forbidden or history or hits else "BOUNDED_PATH_CHECK_PASSED",
            "limitations": ["No history was rewritten and no remote content was inspected.",
                            "Only long exact manifest strings were scanned; numbers, paraphrases and binaries need separate review.",
                            "Coordinate and reflection files were not opened.",
                            "Ignoring and untracking do not erase prior commits or external copies."]}
