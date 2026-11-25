"""Heavy-atom hypotheses, coordination layers, and explicit locking.

The peeling-onion workflow commits to the metal core first because that is where the evidence is
strongest: a heavy atom dominates the Patterson, dominates the anomalous signal, and dominates the
initial map. Everything weaker is then interpreted against a fixed, auditable reference frame.

"Locking" is the dangerous part of that idea, because an implicit lock is indistinguishable from
silently overwriting a crystallographer's coordinates. So a :class:`LockDirective` carries the mode,
the reason, the affected atoms, **and the exact SHELXL instruction lines it generates**. Those lines
are written into the run record as an inspectable artefact.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from crystalx_core.domain.ids import AtomId, BranchId, CandidateId

__all__ = [
    "EvidenceKind",
    "HypothesisEvidence",
    "HeavyAtomSite",
    "HeavyAtomHypothesis",
    "DonorAssignment",
    "CoordinationLayer",
    "AnchorSet",
    "LockMode",
    "LockDirective",
    "format_fixed_shelxl_value",
]


class EvidenceKind(str, enum.Enum):
    """Where a piece of support for a heavy-atom hypothesis came from.

    Kept as an enum rather than free text so that the workflow can require *independent* evidence:
    a CrystalX confidence plus a SHELXD correlation coefficient are two sources; a CrystalX
    confidence twice is one.
    """

    SHELXT_SOLUTION = "shelxt_solution"
    SHELXS_PATTERSON = "shelxs_patterson"
    SHELXD_SUBSTRUCTURE = "shelxd_substructure"
    SHELXE_HAND_TEST = "shelxe_hand_test"
    ANODE_ANOMALOUS_PEAK = "anode_anomalous_peak"
    """Anomalous difference density at the site. A physical observable, not a model score."""

    DIFFERENCE_FOURIER_PEAK = "difference_fourier_peak"
    REFINEMENT_BEHAVIOUR = "refinement_behaviour"
    """The site refines with a sensible displacement parameter and occupancy."""

    COORDINATION_GEOMETRY = "coordination_geometry"
    CRYSTALX_INFERENCE = "crystalx_inference"
    """Model prediction. Orders the search; never sufficient on its own."""

    USER_PRIOR = "user_prior"
    REFERENCE_STRUCTURE = "reference_structure"
    """Advisory only. Never counts towards acceptance."""

    @property
    def is_experimental(self) -> bool:
        """True for evidence grounded in the measured data rather than a model or a prior."""
        return self in (
            EvidenceKind.SHELXS_PATTERSON,
            EvidenceKind.SHELXD_SUBSTRUCTURE,
            EvidenceKind.SHELXE_HAND_TEST,
            EvidenceKind.ANODE_ANOMALOUS_PEAK,
            EvidenceKind.DIFFERENCE_FOURIER_PEAK,
            EvidenceKind.REFINEMENT_BEHAVIOUR,
        )


@dataclass(frozen=True)
class HypothesisEvidence:
    """One supporting or contradicting observation, with its numeric value where there is one."""

    kind: EvidenceKind
    summary: str
    value: float | None = None
    units: str = ""
    supports: bool = True
    source_branch: BranchId | None = None
    artefact_path: str = ""


@dataclass(frozen=True)
class HeavyAtomSite:
    """A candidate heavy-atom position in fractional coordinates."""

    atom_id: AtomId
    element: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    u_iso: float | None = None
    peak_height: float | None = None
    element_is_assumed: bool = False
    """True when the element was assigned by inference or prior rather than measured. Anomalous
    scatterers found by SHELXD, for instance, are *positions* first and *elements* only by
    assumption."""


@dataclass(frozen=True)
class HeavyAtomHypothesis:
    """One self-consistent proposal for the heavy-atom substructure.

    Several of these coexist by design. The workflow is required to carry alternatives forward until
    crystallographic validation can separate them, rather than picking the highest-scoring one early.
    """

    hypothesis_id: str
    sites: tuple[HeavyAtomSite, ...]
    origin_branch: BranchId | None = None
    evidence: tuple[HypothesisEvidence, ...] = ()
    confidence: float | None = None
    """Model confidence, if any. Explicitly *not* a probability that the hypothesis is correct."""

    notes: str = ""

    @property
    def n_sites(self) -> int:
        return len(self.sites)

    @property
    def elements(self) -> tuple[str, ...]:
        seen: list[str] = []
        for site in self.sites:
            if site.element not in seen:
                seen.append(site.element)
        return tuple(seen)

    def experimental_evidence(self) -> tuple[HypothesisEvidence, ...]:
        return tuple(e for e in self.evidence if e.kind.is_experimental and e.supports)

    def contradicting_evidence(self) -> tuple[HypothesisEvidence, ...]:
        return tuple(e for e in self.evidence if not e.supports)

    def has_independent_experimental_support(self, *, minimum_sources: int = 2) -> bool:
        """True when at least ``minimum_sources`` *distinct* experimental evidence kinds support it.

        Repeating the same kind of evidence does not count twice.
        """
        return len({e.kind for e in self.experimental_evidence()}) >= minimum_sources


@dataclass(frozen=True)
class DonorAssignment:
    """A donor atom attached to a metal node, with the distance that justified it."""

    donor_atom_id: AtomId
    donor_element: str
    metal_atom_id: AtomId
    distance_a: float
    within_expected_window: bool | None = None
    donor_group_label: str = ""
    assigned_by: EvidenceKind = EvidenceKind.DIFFERENCE_FOURIER_PEAK


@dataclass(frozen=True)
class CoordinationLayer:
    """The first coordination sphere around one metal node."""

    metal_atom_id: AtomId
    metal_element: str
    donors: tuple[DonorAssignment, ...] = ()
    geometry_label: str = ""
    """e.g. ``linear``, ``trigonal_planar``, ``tetrahedral``, ``square_planar``. Free text because
    the classifier that fills it in is a later slice."""

    geometry_is_plausible: bool | None = None
    notes: str = ""

    @property
    def coordination_number(self) -> int:
        return len(self.donors)


@dataclass(frozen=True)
class AnchorSet:
    """The validated metal core that the rest of the structure will be grown around."""

    anchor_id: str
    hypothesis: HeavyAtomHypothesis
    layers: tuple[CoordinationLayer, ...] = ()
    lock_directives: tuple[LockDirective, ...] = ()
    validated: bool = False
    validation_summary: str = ""

    @property
    def metal_atom_ids(self) -> tuple[AtomId, ...]:
        return tuple(site.atom_id for site in self.hypothesis.sites)


class LockMode(str, enum.Enum):
    """How firmly an atom group is held during the next refinement."""

    HARD_LOCK = "hard_lock"
    """Coordinates fixed. Reversible, recorded, and always accompanied by the generated
    instruction lines."""

    SOFT_RESTRAINT = "soft_restraint"
    """Held near a chemically sensible target but free to move, via distance and angle restraints
    to the coordination sphere."""

    EXPLORATORY = "exploratory"
    """Present but on probation: refined with a free occupancy so the data can vote on whether the
    site is real, and removable on rollback without disturbing the rest of the model."""

    FREE = "free"
    """Ordinary refinement. No directive is emitted."""


def format_fixed_shelxl_value(value: float, *, decimals: int = 5) -> str:
    """Render a coordinate so that SHELXL treats it as fixed.

    SHELXL's free-variable convention: a parameter written as ``10.xxxxx`` means "fix at
    ``0.xxxxx``" (free variable 1 multiplied by 10 is the 'fix' signal). This helper exists so that
    the convention lives in exactly one tested place instead of being reimplemented at each call
    site.

    Only fractional coordinates in ``[-1, 1)`` are supported, which is the range SHELXL atom records
    use; anything else raises rather than silently emitting a nonsensical instruction.
    """
    if not -1.0 <= value < 1.0:
        raise ValueError(
            f"Fixed SHELXL coordinates must lie in [-1, 1); got {value}. "
            "Wrap the coordinate into the unit cell before locking it."
        )
    return f"{10.0 + value:.{decimals}f}"


@dataclass(frozen=True)
class LockDirective:
    """An explicit, recorded decision to constrain part of the model.

    ``instruction_lines`` is the whole point: a crystallographer can read exactly what was imposed on
    the refinement without reverse-engineering it from the resulting ``.ins``.
    """

    mode: LockMode
    atom_ids: tuple[AtomId, ...]
    reason: str
    instruction_lines: tuple[str, ...] = ()
    target_distances: tuple[tuple[AtomId, AtomId, float, float], ...] = ()
    """``(atom_a, atom_b, target_a, esd_a)`` tuples backing any emitted ``DFIX``."""

    applied_at_stage: str = ""
    source_candidate: CandidateId | None = None
    reversible: bool = True
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode is LockMode.FREE and self.instruction_lines:
            raise ValueError("LockMode.FREE must not emit instruction lines")
        if not self.reason.strip():
            raise ValueError("A lock directive must state a reason")
