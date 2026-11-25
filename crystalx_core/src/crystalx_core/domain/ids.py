"""Stable, sortable, human-readable identifiers.

Identifiers are plain strings at runtime (``NewType``) so that they serialise transparently and can
be used as dict keys and filesystem path components without conversion. They are *not* opaque: a
reader should be able to look at ``s4-shell-02/b01-shelxs_patt`` in a directory listing and know
where they are.

All generators take their time and randomness from injected sources so that tests are deterministic.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import NewType

__all__ = [
    "RunId",
    "StageId",
    "BranchId",
    "CandidateId",
    "AtomId",
    "make_run_id",
    "make_stage_id",
    "make_branch_id",
    "make_candidate_id",
    "slugify",
    "RUN_ID_PATTERN",
]

RunId = NewType("RunId", str)
StageId = NewType("StageId", str)
BranchId = NewType("BranchId", str)
CandidateId = NewType("CandidateId", str)
AtomId = NewType("AtomId", str)

RUN_ID_PATTERN = re.compile(r"^r-\d{8}T\d{6}Z-[0-9a-f]{8}$")

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, max_length: int = 40) -> str:
    """Lowercase, filesystem-safe slug. Empty input yields ``"unnamed"``."""
    slug = _SLUG_STRIP.sub("-", text.strip().lower()).strip("-")
    slug = slug[:max_length].strip("-")
    return slug or "unnamed"


def make_run_id(created_at: _dt.datetime, entropy: str) -> RunId:
    """Build a run id from a timezone-aware UTC timestamp and 8 hex characters of entropy.

    The timestamp comes first so that a plain lexicographic sort of run directories is also a
    chronological sort.
    """
    if created_at.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")
    stamp = created_at.astimezone(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = entropy.lower()
    if not re.fullmatch(r"[0-9a-f]{8}", token):
        raise ValueError(f"entropy must be 8 hex characters, got {entropy!r}")
    return RunId(f"r-{stamp}-{token}")


def make_stage_id(index: int, kind: str, *, shell: int | None = None) -> StageId:
    """``s1-initial_solution`` or, for repeated shells, ``s4-shell_expansion-02``."""
    if index < 0:
        raise ValueError("stage index must be >= 0")
    base = f"s{index}-{kind}"
    if shell is not None:
        base = f"{base}-{shell:02d}"
    return StageId(base)


def make_branch_id(index: int, solver_kind: str) -> BranchId:
    """``b00-shelxt``. The numeric prefix preserves nomination order under sorting."""
    if index < 0:
        raise ValueError("branch index must be >= 0")
    return BranchId(f"b{index:02d}-{solver_kind.lower()}")


def make_candidate_id(index: int) -> CandidateId:
    """``c000``. Monotonic across the whole run, never reused, never renumbered."""
    if index < 0:
        raise ValueError("candidate index must be >= 0")
    return CandidateId(f"c{index:03d}")
