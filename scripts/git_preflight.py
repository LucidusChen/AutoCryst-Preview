#!/usr/bin/env python3
"""Read-only audit of staged paths and blobs; local identifiers never enter output.

Checks the entire index, including a caller's GIT_INDEX_FILE, rather than worktree
bytes or only the diff against HEAD. Requires Python's standard library and an
existing local Git repository. History and scientific content need separate review.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys


LOCAL_IDENTIFIERS = ".private_identifiers"
MAX_BLOB_BYTES = 1024 * 1024
FORBIDDEN_ROOTS = frozenset({
    "runs", "realdata_examples", "dist", "paper_ref", "platon_instruction",
    "crystalx_engine", ".local-engines", ".native-engine-env",
})
FORBIDDEN_SUFFIXES = (".ckpt", ".pt", ".pth", ".pyc", ".tar.gz", ".tgz", ".zip")
FORBIDDEN_PARTS = frozenset({"__pycache__", ".venv", "venv", ".eggs"})
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"(?im)^\s*(?:export\s+)?[A-Za-z0-9_]*(?:api[_-]?key|access[_-]?token|"
        r"auth[_-]?token|client[_-]?secret|password|passwd)[A-Za-z0-9_]*"
        r"\s*[:=]\s*['\"]?([A-Za-z0-9_./+=-]{20,})"
    ),
)


class PreflightError(Exception):
    """Audit prerequisite failure; its message contains no scanned file data."""


def _git(repo: Path, *arguments: str, data: bytes | None = None) -> bytes:
    # Preserve GIT_INDEX_FILE and disable optional Git index refresh/writes.
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo), *arguments],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        check=False,
    )
    if result.returncode:
        raise PreflightError("a required read-only Git inspection failed")
    return result.stdout


def _identifiers(repo: Path) -> tuple[str, ...]:
    path = repo / LOCAL_IDENTIFIERS
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise PreflightError("the local identifier list must be a regular file")
        if metadata.st_size > MAX_BLOB_BYTES:
            raise PreflightError("the local identifier list exceeds its size bound")
        # Check the rule even if this local file was mistakenly force-added.
        _git(repo, "check-ignore", "--no-index", "--quiet", "--", LOCAL_IDENTIFIERS)
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise PreflightError("an ignored, UTF-8 local identifier list is required") from None
    identifiers = tuple(dict.fromkeys(
        line.strip().casefold() for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ))
    if not identifiers:
        raise PreflightError("the ignored local identifier list must contain literal identifiers")
    return identifiers


def _forbidden_path(path: str, *, allow_upstream_private: bool = False) -> bool:
    lowered = tuple(part.casefold() for part in PurePosixPath(path).parts)
    if not lowered:
        return True
    basename = lowered[-1]
    local_list = basename == LOCAL_IDENTIFIERS or (
        basename.startswith(LOCAL_IDENTIFIERS + ".") and basename != LOCAL_IDENTIFIERS + ".example"
    )
    excluded_root = lowered[0] in FORBIDDEN_ROOTS
    if lowered[0] == "crystalx_engine" and allow_upstream_private:
        excluded_root = False
    if excluded_root or local_list:
        return True
    if lowered[0] == "toolbox" and lowered not in {("toolbox", ".gitkeep"), ("toolbox", ".gitignore")}:
        return True
    if any(part in FORBIDDEN_PARTS or part.endswith(".egg-info") for part in lowered):
        return True
    if basename.endswith(FORBIDDEN_SUFFIXES):
        return True
    if basename == ".env" or (basename.startswith(".env.") and basename not in {".env.example", ".env.sample"}):
        return True
    return basename in {"id_rsa", "id_ed25519", "credentials.json", "service-account.json"}


def audit_index(repo: Path, *, allow_upstream_private: bool = False) -> tuple[int, int, list[str]]:
    """Return entry count, staged-byte count and non-disclosing refusal reasons."""
    top = Path(os.fsdecode(_git(repo, "rev-parse", "--show-toplevel")).strip()).resolve()
    if top != repo.resolve():
        raise PreflightError("--repo must name the repository root")
    identifiers = _identifiers(repo)
    entries = []
    for raw in _git(repo, "ls-files", "--stage", "-z").split(b"\0"):
        if not raw:
            continue
        try:
            descriptor, name = raw.split(b"\t", 1)
            mode, oid, stage = descriptor.split()
            entries.append((mode, oid, stage, os.fsdecode(name)))
        except ValueError:
            raise PreflightError("the index contains an unreadable entry") from None
    object_ids = tuple(dict.fromkeys(oid for _, oid, _, _ in entries))
    sizes = {}
    if object_ids:
        details = _git(repo, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                       data=b"\n".join(object_ids) + b"\n")
        for line in details.splitlines():
            fields = line.split()
            if len(fields) == 3 and fields[1] == b"blob" and fields[2].isdigit():
                sizes[fields[0]] = int(fields[2])
    failures = []
    total_bytes = 0
    for number, (mode, oid, stage, path) in enumerate(entries, 1):
        # Entry numbers make failures locatable without printing sensitive path names.
        prefix = f"entry {number}"
        if any(token in path.casefold() for token in identifiers):
            failures.append(f"{prefix}: private identifier in staged path")
        excluded = _forbidden_path(path, allow_upstream_private=allow_upstream_private)
        if excluded:
            failures.append(f"{prefix}: excluded staged path")
        if mode not in (b"100644", b"100755") or stage != b"0":
            failures.append(f"{prefix}: only resolved regular source files are allowed")
            continue
        size = sizes.get(oid)
        if size is None:
            failures.append(f"{prefix}: staged blob is unavailable")
            continue
        total_bytes += size
        if size > MAX_BLOB_BYTES:
            failures.append(f"{prefix}: staged blob exceeds the 1 MiB limit")
            continue
        if excluded:
            # An excluded file already fails. Do not open private data, upstream
            # material, or the local identifier list through their staged blobs.
            continue
        blob = _git(repo, "cat-file", "blob", oid.decode("ascii"))
        if len(blob) != size:
            raise PreflightError("a staged object changed during inspection")
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            failures.append(f"{prefix}: binary or non-UTF-8 staged blob")
            continue
        if any(ord(character) < 32 and character not in "\t\n\r" for character in text):
            failures.append(f"{prefix}: binary control bytes in staged blob")
        if any(token in text.casefold() for token in identifiers):
            failures.append(f"{prefix}: private identifier in staged blob")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            failures.append(f"{prefix}: possible credential in staged blob")
    return len(entries), total_bytes, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("visibility", nargs="?", choices=("private", "public"), default="private")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", help="emit a machine-readable receipt without scanned content")
    parser.add_argument("--allow-upstream-private", action="store_true",
                        help="owner-authorized upstream source retention for a private destination only; no weights or archives")
    arguments = parser.parse_args(argv)
    try:
        if arguments.allow_upstream_private and arguments.visibility != "private":
            raise PreflightError("upstream source retention is private-only")
        count, size, failures = audit_index(arguments.repo, allow_upstream_private=arguments.allow_upstream_private)
    except (PreflightError, OSError):
        # Do not echo Git stderr, paths, or identifiers when a prerequisite fails.
        count, size = 0, 0
        failures = ["preflight prerequisites failed; require an existing Git repository and an ignored regular .private_identifiers file"]
    if arguments.json:
        print(json.dumps({
            "schema": "crystalx.git-preflight/1", "passed": not failures,
            "entry_count": count, "staged_bytes": size, "violations": failures,
            "scope": "selected_index_staged_paths_and_blobs", "visibility": arguments.visibility,
            "index_modified": False, "remote_accessed": False,
            "upstream_private_retention": arguments.allow_upstream_private,
        }, sort_keys=True))
        return int(bool(failures))
    print(f"Inspected {count} staged entries ({size} bytes); visibility={arguments.visibility}; strict exclusions apply.")
    for failure in failures:
        print(f"REFUSE: {failure}")
    if failures:
        print("Preflight failed. No index, object, branch, remote or worktree changes were made.")
        return 1
    print("PREFLIGHT PASSED. History and scientific content require separate review. No changes were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
