"""Candidate models and their lineage.

Every candidate the workflow ever creates is kept, including the ones it rejects. A crystallographer
must be able to ask "what else did you try, and why did you drop it" and get a complete answer.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from crystalx_core.domain.ids import AtomId, BranchId, CandidateId, StageId
from crystalx_core.domain.metrics import (
    ConfidenceReport,
    GeometryReport,
    RefinementMetrics,
    ValidationReport,
)

__all__ = [
    "MutationKind",
    "Mutation",
    "CandidateProvenance",
    "CandidateStatus",
    "Candidate",
    "Lineage",
]


class MutationKind(str, enum.Enum):
    """How a candidate differs from its parent."""

    INITIAL_SOLUTION = "initial_solution"
    """Root candidate: came straight out of a solver branch."""

    ELEMENT_FLIP = "element_flip"
    """One atom's element changed. The paper's margin-ranked single flip is this."""

    ATOM_ADDED = "atom_added"
    ATOM_REMOVED = "atom_removed"
    OCCUPANCY_CHANGED = "occupancy_changed"
    LOCK_CHANGED = "lock_changed"
    FRAGMENT_PLACED = "fragment_placed"
    """A known rigid group (triflate, NTf2, perchlorate, ...) fitted into residual density."""

    HYDROGEN_PLACED = "hydrogen_placed"
    DISORDER_MODEL_CHANGED = "disorder_model_changed"
    SPACE_GROUP_CHANGED = "space_group_changed"
    REFINEMENT_SETTINGS_CHANGED = "refinement_settings_changed"


@dataclass(frozen=True)
class Mutation:
    """The single, explicit difference between a candidate and its parent."""

    kind: MutationKind
    description: str
    atoms: tuple[AtomId, ...] = ()
    from_value: str = ""
    to_value: str = ""
    driver: str = ""
    """What proposed it: ``crystalx_margin``, ``difference_fourier``, ``user``, ``reference_cif``..."""

    margin: float | None = None
    """For an element flip: the top-1/top-2 probability margin that made this atom the choice."""


@dataclass(frozen=True)
class CandidateProvenance:
    """Everything needed to reproduce and audit one candidate.

    ``reference_structure_influenced`` exists so that a candidate steered by a user-supplied
    reference CIF can never later be mistaken for an independent result.
    """

    created_at: str
    stage_id: StageId
    branch_id: BranchId | None = None
    parent_id: CandidateId | None = None
    tool_identities: dict[str, str] = field(default_factory=dict)
    """``tool name -> sha256``, so a rerun on a different SHELXL build is detectable."""

    tool_versions: dict[str, str] = field(default_factory=dict)
    model_identities: dict[str, str] = field(default_factory=dict)
    input_hashes: dict[str, str] = field(default_factory=dict)
    commands: tuple[tuple[str, ...], ...] = ()
    """Argument arrays exactly as executed. Never a shell string."""

    working_directory: str = ""
    resolution_cut_a: float | None = None
    reflection_file_sha256: str = ""
    hklf_format: int | None = None
    twin_law: str = ""
    space_group: str = ""
    lock_directive_ids: tuple[str, ...] = ()
    reference_structure_influenced: bool = False
    reference_structure_labels: tuple[str, ...] = ()
    user_overrides: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def reflection_basis_key(self) -> str:
        """Identity of the reflection basis this candidate was refined against.

        Two candidates may only be compared when these keys match.
        """
        return "|".join(
            [
                self.reflection_file_sha256 or "nohkl",
                f"dmin={self.resolution_cut_a if self.resolution_cut_a is not None else 'none'}",
                f"hklf={self.hklf_format if self.hklf_format is not None else 'none'}",
                f"twin={self.twin_law or 'none'}",
            ]
        )


class CandidateStatus(str, enum.Enum):
    PROPOSED = "proposed"
    REFINING = "refining"
    REFINED = "refined"
    VALIDATED = "validated"
    REJECTED = "rejected"
    """Failed a hard gate. Kept, with the failing gate recorded."""

    SUPERSEDED = "superseded"
    """A descendant is strictly better; kept for the audit trail."""

    ABANDONED = "abandoned"
    """Dropped by a rollback. Kept."""


@dataclass(frozen=True)
class Candidate:
    """One structural model under consideration."""

    candidate_id: CandidateId
    provenance: CandidateProvenance
    status: CandidateStatus = CandidateStatus.PROPOSED
    mutations: tuple[Mutation, ...] = ()
    label: str = ""
    ins_path: str = ""
    res_path: str = ""
    cif_path: str = ""
    formula: str = ""
    n_non_hydrogen_atoms: int | None = None
    metrics: RefinementMetrics | None = None
    geometry: GeometryReport | None = None
    validation: ValidationReport | None = None
    confidence: ConfidenceReport | None = None
    rejection_reason: str = ""

    @property
    def parent_id(self) -> CandidateId | None:
        return self.provenance.parent_id

    @property
    def is_root(self) -> bool:
        return self.provenance.parent_id is None

    def with_status(self, status: CandidateStatus, *, reason: str = "") -> "Candidate":
        return Candidate(
            candidate_id=self.candidate_id,
            provenance=self.provenance,
            status=status,
            mutations=self.mutations,
            label=self.label,
            ins_path=self.ins_path,
            res_path=self.res_path,
            cif_path=self.cif_path,
            formula=self.formula,
            n_non_hydrogen_atoms=self.n_non_hydrogen_atoms,
            metrics=self.metrics,
            geometry=self.geometry,
            validation=self.validation,
            confidence=self.confidence,
            rejection_reason=reason or self.rejection_reason,
        )


@dataclass
class Lineage:
    """The candidate forest for a run.

    Mutable by design: it is the run's working index. It is serialised to
    ``candidates/index.json`` after every change so a crash still leaves a readable history.
    """

    candidates: dict[str, Candidate] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def add(self, candidate: Candidate) -> None:
        key = str(candidate.candidate_id)
        if key in self.candidates:
            raise ValueError(f"Candidate {key} already exists; candidates are never overwritten")
        parent = candidate.provenance.parent_id
        if parent is not None and str(parent) not in self.candidates:
            raise ValueError(f"Unknown parent {parent} for candidate {key}")
        self.candidates[key] = candidate
        self.order.append(key)

    def replace(self, candidate: Candidate) -> None:
        """Update a candidate in place (status, metrics, validation) keeping its identity."""
        key = str(candidate.candidate_id)
        if key not in self.candidates:
            raise KeyError(f"Unknown candidate {key}")
        self.candidates[key] = candidate

    def get(self, candidate_id: CandidateId | str) -> Candidate:
        return self.candidates[str(candidate_id)]

    def children(self, candidate_id: CandidateId | str) -> tuple[Candidate, ...]:
        key = str(candidate_id)
        return tuple(
            c for c in self.iter_all() if c.provenance.parent_id and str(c.provenance.parent_id) == key
        )

    def ancestry(self, candidate_id: CandidateId | str) -> tuple[Candidate, ...]:
        """Root-first chain of ancestors ending with the candidate itself."""
        chain: list[Candidate] = []
        cursor: str | None = str(candidate_id)
        seen: set[str] = set()
        while cursor is not None:
            if cursor in seen:
                raise ValueError(f"Cycle detected in candidate lineage at {cursor}")
            seen.add(cursor)
            candidate = self.candidates[cursor]
            chain.append(candidate)
            parent = candidate.provenance.parent_id
            cursor = str(parent) if parent is not None else None
        return tuple(reversed(chain))

    def roots(self) -> tuple[Candidate, ...]:
        return tuple(c for c in self.iter_all() if c.is_root)

    def iter_all(self) -> tuple[Candidate, ...]:
        return tuple(self.candidates[key] for key in self.order)

    def next_index(self) -> int:
        return len(self.order)

    def __len__(self) -> int:
        return len(self.order)
