"""Supported, externally pinned candidate reads for local frontends.

No solver, grading reference, summary-selected file path or regeneration is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import TYPE_CHECKING

from crystalx_core.private_acceptance.sealing import RunSeal, SealError, read_sealed_output, verify_seal

if TYPE_CHECKING:
    from crystalx_core.app.reconstruction_backend import ReconstructionBackendService

__all__ = ["PinnedCandidate", "pinned_candidate", "export_candidate", "candidate_scene"]
OUTPUT_NAMES = {"model": "ai_solved.res", "cif": "ai_solved.cif", "fcf": "ai_solved.fcf"}


@dataclass(frozen=True)
class PinnedCandidate:
    seal: RunSeal
    private_root: Path

    def read(self, role: str, *, max_bytes: int = 32 * 1024 * 1024) -> bytes:
        if role not in OUTPUT_NAMES:
            raise SealError("Only candidate output roles may be read")
        return read_sealed_output(self.seal, private_root=self.private_root, role=role, max_bytes=max_bytes)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SealError("Duplicate external pin field")
        result[key] = value
    return result


def pinned_candidate(service: ReconstructionBackendService, run_id: str) -> PinnedCandidate:
    """Check the validated run id, external pin, terminal phase and every sealed hash."""
    root = service.run_root(run_id)
    state = service.status(run_id)
    if state.get("phase") not in {"completed", "failed", "blocked", "cancelled"}:
        raise SealError("Candidate requires a sealed terminal run")
    path = service.runs_root / f"{run_id}.seal-pin.json"
    try:
        named = path.lstat()
        if not stat.S_ISREG(named.st_mode) or named.st_nlink != 1 or named.st_size > 8192:
            raise SealError("External pin must be a bounded unaliased file")
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as source:
            opened = os.fstat(source.fileno())
            data = source.read(8193)
            after = os.fstat(source.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if len(data) > 8192 or not identity(named) == identity(opened) == identity(after) == identity(path.lstat()):
            raise SealError("External pin changed during read")
        pin = json.loads(data, object_pairs_hook=_unique)
        if (not isinstance(pin, dict) or set(pin) != {"run_root", "seal_sha256"}
                or not isinstance(pin["run_root"], str) or Path(pin["run_root"]) != root
                or not isinstance(pin["seal_sha256"], str) or re.fullmatch(r"[a-f0-9]{64}", pin["seal_sha256"]) is None):
            raise SealError("External candidate pin is malformed or names another run")
        seal = verify_seal(root, private_root=service.runs_root, expected_seal_sha256=pin["seal_sha256"])
        if seal.terminal_state != state["phase"]:
            raise SealError("Seal terminal state differs from durable run phase")
        return PinnedCandidate(seal, service.runs_root)
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise SealError("Candidate pin or seal verification failed") from exc


def export_candidate(service: ReconstructionBackendService, run_id: str, destination: Path) -> dict:
    """Write a fresh private directory, never overwrite a selected destination or follow aliases."""
    candidate = pinned_candidate(service, run_id)
    contents = {role: candidate.read(role) for role in OUTPUT_NAMES
                if any(item.role == role for item in candidate.seal.files)}
    if not contents.get("model"):
        raise SealError("Nonempty sealed model is required")
    target = Path(destination).absolute()
    if target.parent.resolve(strict=True) != target.parent or ".." in target.parts:
        raise SealError("Export parent must be canonical")
    # Seal checks precede any destination write; caller chooses a new directory.
    target.mkdir(mode=0o700, exist_ok=False)
    outputs = []
    for role, content in contents.items():
        path = target / OUTPUT_NAMES[role]
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
            stream.write(content)
        outputs.append({"role": role, "path": str(path), "size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()})
    return {"seal_sha256": candidate.seal.seal_sha256, "outputs": outputs,
            "missing_roles": [role for role in OUTPUT_NAMES if role not in contents]}


def candidate_scene(service: ReconstructionBackendService, run_id: str, *, max_sites: int = 1000) -> dict:
    """Use the core viewer on a temporary verified copy, with no persistent scene state."""
    from crystalx_core.geometry.model import build_viewer_payload, orthogonalisation_matrix
    from crystalx_core.io.shelx_res import read_res
    if type(max_sites) is not int or not 1 <= max_sites <= 2000:
        raise ValueError("Scene site bound must be in [1, 2000]")
    candidate = pinned_candidate(service, run_id)
    content = candidate.read("model", max_bytes=4 * 1024 * 1024)
    # A bounded private ephemeral copy lets the existing file parser retain its authority.
    with tempfile.TemporaryDirectory(prefix=".scene-", dir=service.runs_root) as temp:
        model = Path(temp) / "candidate.res"
        model.write_bytes(content)
        model.chmod(0o600)
        parsed = read_res(model)
        if parsed.n_atoms + len(parsed.q_peaks) > max_sites:
            raise ValueError("Scene site limit exceeded")
        payload = build_viewer_payload(parsed, stage="refined", source_label="Sealed unvalidated candidate")
    cell = parsed.header.cell
    matrix = orthogonalisation_matrix((cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma))
    return {"schema": "autocryst.scene/1", "run_id": run_id, "seal_sha256": candidate.seal.seal_sha256,
            "candidate_sha256": hashlib.sha256(content).hexdigest(), "units": "angstrom",
            "fractional_to_cartesian_rows": matrix, "scientifically_validated": False, "payload": payload}
