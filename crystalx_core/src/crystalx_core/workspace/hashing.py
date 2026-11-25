"""Content hashing for run provenance.

Everything that enters or leaves a run workspace is hashed so that a record can be checked later:
inputs, tool executables, model checkpoints, and produced artefacts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from crystalx_core._serde import canonical_json

__all__ = ["sha256_file", "sha256_bytes", "sha256_text", "sha256_object", "FileDigest", "digest_dir"]

_CHUNK = 1024 * 1024


def sha256_file(path: Path | str) -> str:
    """Streaming SHA-256 of a file, safe for large reflection files."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_object(value: Any) -> str:
    """Hash of a domain object's canonical JSON rendering.

    Deterministic across processes and orderings, so the same logical object always hashes the same.
    """
    return sha256_text(canonical_json(value))


class FileDigest:
    """A file's identity: path, size, mtime and digest."""

    __slots__ = ("path", "size_bytes", "mtime_epoch", "sha256")

    def __init__(self, path: Path, size_bytes: int, mtime_epoch: float, sha256: str) -> None:
        self.path = path
        self.size_bytes = size_bytes
        self.mtime_epoch = mtime_epoch
        self.sha256 = sha256

    @classmethod
    def of(cls, path: Path | str) -> "FileDigest":
        resolved = Path(path)
        stat = resolved.stat()
        return cls(resolved, stat.st_size, stat.st_mtime, sha256_file(resolved))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "mtime_epoch": self.mtime_epoch,
            "sha256": self.sha256,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FileDigest({self.path!s}, {self.size_bytes}B, {self.sha256[:12]}...)"


def digest_dir(directory: Path | str, *, relative_to: Path | None = None) -> list[dict[str, Any]]:
    """Digest every regular file under ``directory``, sorted by relative path.

    Sorting makes the manifest itself deterministic, which matters when the manifest is hashed.
    """
    root = Path(directory)
    base = relative_to or root
    entries: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        record = FileDigest.of(path).to_dict()
        record["path"] = str(path.relative_to(base))
        entries.append(record)
    return entries


def manifest_for(paths: Iterable[Path | str], *, relative_to: Path | None = None) -> list[dict[str, Any]]:
    """Digest an explicit list of files."""
    entries: list[dict[str, Any]] = []
    for path in sorted(Path(p) for p in paths):
        record = FileDigest.of(path).to_dict()
        if relative_to is not None:
            try:
                record["path"] = str(path.relative_to(relative_to))
            except ValueError:
                record["path"] = str(path)
        entries.append(record)
    return entries
