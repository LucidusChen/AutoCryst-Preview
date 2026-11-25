"""Explicit output seals for local, private acceptance runs.

The caller must supply an ignored private results directory and wait for its run
to terminate. A seal records that declaration; it cannot prove process isolation
or stop another process modifying a file. Reads use directory descriptors and
``O_NOFOLLOW``; content, identity and metadata are checked before and after reads.
Pin the returned digest outside the mutable run directory before opening answers.
This is an integrity record, not a digital signature or a filesystem sandbox.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Iterator, Mapping


SEAL_FILENAME = ".acceptance-seal.json"
_ROLE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,79}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TRACKS = frozenset({"de_novo", "component_regression", "unsolved", "initial_model_recovery"})
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "blocked"})
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024


class SealError(ValueError):
    """An output, seal or private path failed closed."""


@dataclass(frozen=True)
class SealedFile:
    role: str
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class RunSeal:
    run_root: Path
    track: str
    terminal_state: str
    files: tuple[SealedFile, ...]
    seal_sha256: str
    created_utc: str
    case_id: str
    externally_pinned: bool = False

    def file_for_role(self, role: str) -> SealedFile:
        for item in self.files:
            if item.role == role:
                return item
        raise SealError("Required sealed output role is missing")


def _relative(value: str | Path) -> str:
    raw = str(value)
    if not raw or "\\" in raw or "\x00" in raw or ":" in raw:
        raise SealError("Private paths must be ordinary relative POSIX paths")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(p in {"", ".", ".."} for p in raw.split("/")):
        raise SealError("Private path traversal or an absolute path is forbidden")
    return path.as_posix()


def _canonical_track(value: str) -> str:
    if not isinstance(value, str) or value.lower() not in _TRACKS:
        raise SealError("Unknown acceptance track")
    return value.lower()


def _absolute(path: Path | str) -> Path:
    # Do not resolve here: doing so would hide a symlink supplied by the caller.
    value = Path(path).absolute()
    if ".." in value.parts:
        raise SealError("Private root traversal is forbidden")
    return value


@contextmanager
def _directory(path: Path | str) -> Iterator[int]:
    if not all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW")):
        raise SealError("Safe private descriptor operations are unsupported on this platform")
    absolute = _absolute(path)
    fd = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in absolute.parts[1:]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd
    except OSError as exc:
        raise SealError("Private directory is absent, unsafe or changed") from exc
    finally:
        os.close(fd)


@contextmanager
def _run_directory(run_root: Path | str, private_root: Path | str) -> Iterator[int]:
    private = _absolute(private_root)
    run = _absolute(run_root)
    try:
        relative = run.relative_to(private)
    except ValueError as exc:
        raise SealError("Run directory must remain inside the private results root") from exc
    if not relative.parts:
        raise SealError("Run directory must be a child of the private results root")
    with _directory(private) as root_fd:
        fd = os.dup(root_fd)
        try:
            for component in relative.parts:
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            yield fd
        except OSError as exc:
            raise SealError("Private run directory is absent, unsafe or changed") from exc
        finally:
            os.close(fd)


@contextmanager
def _parent(run_fd: int, relative: str) -> Iterator[tuple[int, str]]:
    parts = PurePosixPath(_relative(relative)).parts
    fd = os.dup(run_fd)
    try:
        for component in parts[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd, parts[-1]
    except OSError as exc:
        raise SealError("Private output parent is absent, unsafe or changed") from exc
    finally:
        os.close(fd)


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read(run_fd: int, relative: str, *, max_bytes: int | None = None,
          collect: bool = False) -> tuple[str, int, bytes]:
    with _parent(run_fd, relative) as (parent_fd, name):
        fd = None
        try:
            before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise SealError("Private evidence must be a regular file without hard links")
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
                raise SealError("Private evidence changed while opening")
            digest = hashlib.sha256()
            size = 0
            chunks = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if max_bytes is not None and size > max_bytes:
                    raise SealError("Private evidence exceeds the bounded read limit")
                digest.update(chunk)
                if collect:
                    chunks.append(chunk)
            after = os.fstat(fd)
            retained = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (_identity(before) != _identity(after) or _identity(before) != _identity(retained)
                    or not stat.S_ISREG(retained.st_mode) or retained.st_nlink != 1
                    or size != after.st_size):
                raise SealError("Private evidence changed while reading")
            return digest.hexdigest(), size, b"".join(chunks)
        except OSError as exc:
            raise SealError("Private evidence is missing, unsafe or unreadable") from exc
        finally:
            if fd is not None:
                os.close(fd)


def _write_exclusive(run_fd: int, relative: str, payload: bytes) -> None:
    with _parent(run_fd, relative) as (parent_fd, name):
        fd = None
        created = False
        try:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=parent_fd)
            created = True
            remaining = memoryview(payload)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise SealError("Private evidence could not be written completely")
                remaining = remaining[written:]
            os.fsync(fd)
            os.fsync(parent_fd)
        except (OSError, SealError) as exc:
            if created:
                os.unlink(name, dir_fd=parent_fd)
            raise SealError("Private evidence already exists or cannot be written safely") from exc
        finally:
            if fd is not None:
                os.close(fd)


def write_private_json(run_root: Path | str, *, private_root: Path | str,
                       relative: str, payload: Mapping[str, object]) -> Path:
    """Write a new local report exclusively; no parent creation or overwriting."""
    relative = _relative(relative)
    if relative == SEAL_FILENAME:
        raise SealError("The seal path is reserved")
    try:
        data = (json.dumps(dict(payload), sort_keys=True, ensure_ascii=True,
                           allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SealError("Private report must contain finite JSON values") from exc
    with _run_directory(run_root, private_root) as run_fd:
        _write_exclusive(run_fd, relative, data)
    return _absolute(run_root) / relative


def seal_run(run_root: Path | str, *, private_root: Path | str,
             outputs: Mapping[str, Path | str], track: str, case_id: str,
             terminal_state: str = "completed",
             completion: Mapping[str, str | int | float | bool | None] | None = None) -> RunSeal:
    """Seal only explicitly named output files after the caller stops its run.

    Output paths are relative to ``run_root``. No input, answer or reference is
    discovered by glob. All files to be graded must be explicitly included.
    """
    track = _canonical_track(track)
    if (not isinstance(case_id, str) or not case_id.strip() or len(case_id) > 128
            or any(ord(c) < 32 or ord(c) == 127 for c in case_id)):
        raise SealError("Seal requires an explicit local case identifier")
    if (not isinstance(terminal_state, str) or terminal_state not in _TERMINAL_STATES):
        raise SealError("Unknown acceptance track or nonterminal run state")
    if not isinstance(outputs, Mapping) or not outputs or len(outputs) > 10000:
        raise SealError("A seal requires a bounded, nonempty explicit output list")
    if completion is not None and (not isinstance(completion, Mapping) or any(
        not isinstance(k, str) or not k or type(v) not in {str, int, float, bool, type(None)}
        for k, v in completion.items()
    )):
        raise SealError("Completion evidence must contain scalar JSON fields")
    entries = []
    seen = set()
    with _run_directory(run_root, private_root) as run_fd:
        if any(not isinstance(role, str) for role in outputs):
            raise SealError("Invalid sealed output role")
        for role, path in sorted(outputs.items()):
            if not isinstance(role, str) or not _ROLE.fullmatch(role):
                raise SealError("Invalid sealed output role")
            relative = _relative(path)
            if relative == SEAL_FILENAME or relative in seen:
                raise SealError("Duplicate output path or reserved seal path")
            seen.add(relative)
            digest, size, _ = _read(run_fd, relative)
            entries.append({"role": role, "path": relative, "sha256": digest, "size": size})
        payload = {"schema_version": 1, "case_id": case_id, "track": track, "terminal_state": terminal_state,
                   "created_utc": datetime.now(timezone.utc).isoformat(), "files": entries,
                   "completion": dict(completion or {})}
        try:
            data = (json.dumps(payload, sort_keys=True, allow_nan=False,
                               separators=(",", ":")) + "\n").encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise SealError("Completion evidence must contain finite JSON values") from exc
        _write_exclusive(run_fd, SEAL_FILENAME, data)
    # A second pass detects changed or swapped files around manifest creation.
    return verify_seal(run_root, private_root=private_root,
                       expected_seal_sha256=hashlib.sha256(data).hexdigest())


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SealError("Duplicate keys in private seal")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise SealError("Nonfinite value in private seal")


def verify_seal(run_root: Path | str, *, private_root: Path | str,
                expected_seal_sha256: str | None = None) -> RunSeal:
    """Rehash every declared file, rejecting unsafe paths and symlink changes."""
    if expected_seal_sha256 is not None and (not isinstance(expected_seal_sha256, str)
                                           or not _SHA256.fullmatch(expected_seal_sha256)):
        raise SealError("Pinned seal digest must be a lowercase SHA-256 digest")
    with _run_directory(run_root, private_root) as run_fd:
        digest, _, data = _read(run_fd, SEAL_FILENAME, max_bytes=_MAX_MANIFEST_BYTES, collect=True)
        if expected_seal_sha256 is not None and digest != expected_seal_sha256:
            raise SealError("Seal differs from the independently retained digest")
        try:
            payload = json.loads(data, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        except (ValueError, UnicodeError) as exc:
            raise SealError("Private seal is malformed") from exc
        if (not isinstance(payload, dict) or set(payload) != {
                "schema_version", "case_id", "track", "terminal_state", "created_utc", "files", "completion"}
                or type(payload["schema_version"]) is not int or payload["schema_version"] != 1
                or not isinstance(payload["case_id"], str) or not payload["case_id"].strip()
                or len(payload["case_id"]) > 128
                or any(ord(c) < 32 or ord(c) == 127 for c in payload["case_id"])
                or not isinstance(payload["track"], str) or payload["track"] not in _TRACKS
                or not isinstance(payload["terminal_state"], str) or payload["terminal_state"] not in _TERMINAL_STATES
                or not isinstance(payload["created_utc"], str)
                or not isinstance(payload["completion"], dict)
                or any(not k or type(v) not in {str, int, float, bool, type(None)}
                       for k, v in payload["completion"].items())
                or not isinstance(payload["files"], list) or not 1 <= len(payload["files"]) <= 10000):
            raise SealError("Private seal schema is unsupported or invalid")
        entries = []
        roles: set[str] = set()
        paths: set[str] = set()
        for item in payload["files"]:
            if (not isinstance(item, dict) or set(item) != {"role", "path", "sha256", "size"}
                    or not isinstance(item["role"], str) or not _ROLE.fullmatch(item["role"])
                    or not isinstance(item["path"], str)
                    or not isinstance(item["sha256"], str) or not _SHA256.fullmatch(item["sha256"])
                    or type(item["size"]) is not int or item["size"] < 0):
                raise SealError("Private seal contains an invalid file entry")
            relative = _relative(item["path"])
            if relative == SEAL_FILENAME or relative in paths or item["role"] in roles:
                raise SealError("Private seal contains a duplicate or reserved output")
            paths.add(relative)
            roles.add(item["role"])
            actual_digest, size, _ = _read(run_fd, relative)
            if actual_digest != item["sha256"] or size != item["size"]:
                raise SealError("Sealed output integrity verification failed")
            entries.append(SealedFile(item["role"], relative, actual_digest, size))
        return RunSeal(_absolute(run_root), payload["track"], payload["terminal_state"],
                       tuple(entries), digest, payload["created_utc"], payload["case_id"],
                       expected_seal_sha256 is not None)


def read_sealed_output(seal: RunSeal, *, private_root: Path | str, role: str,
                       max_bytes: int = 32 * 1024 * 1024) -> bytes:
    """Read a bounded, reverified model snapshot for a separate grading adapter."""
    verified = verify_seal(seal.run_root, private_root=private_root,
                           expected_seal_sha256=seal.seal_sha256)
    entry = verified.file_for_role(role)
    with _run_directory(verified.run_root, private_root) as run_fd:
        digest, size, data = _read(run_fd, entry.path, max_bytes=max_bytes, collect=True)
    if digest != entry.sha256 or size != entry.size:
        raise SealError("Sealed model changed before grading")
    return data


def read_private_reference(path: Path | str, *, case_root: Path | str,
                            max_bytes: int = 32 * 1024 * 1024) -> tuple[bytes, str]:
    """Read an explicitly selected answer only after the grader verifies a seal.

    This low-level helper is intentionally never called by solver code. Callers
    outside the separate grader would not acquire isolation merely by using it.
    """
    root = _absolute(case_root)
    target = _absolute(path)
    try:
        relative = _relative(target.relative_to(root))
    except ValueError as exc:
        raise SealError("Reference must remain inside its explicit private case root") from exc
    with _directory(root) as fd:
        digest, _, data = _read(fd, relative, max_bytes=max_bytes, collect=True)
    return data, digest
