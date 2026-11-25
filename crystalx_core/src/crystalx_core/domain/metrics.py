"""Refinement metrics, validation output, and model confidence.

Three things are kept strictly apart:

* :class:`RefinementMetrics` and :class:`ValidationReport` are **crystallographic evidence**.
* :class:`GeometryReport` is a **derived chemical check** on the model.
* :class:`ConfidenceReport` is a **model output**. It may order the search. It never orders the
  result and it is never averaged into a score alongside the evidence.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from crystalx_core.domain.ids import AtomId

__all__ = [
    "AlertSeverity",
    "Alert",
    "ValidationReport",
    "ResidualDensityPeak",
    "ResidualDensity",
    "RefinementMetrics",
    "GeometryIssueKind",
    "GeometryIssue",
    "GeometryReport",
    "AtomConfidence",
    "ConfidenceReport",
]


class AlertSeverity(str, enum.Enum):
    """PLATON / CheckCIF alert levels, kept as levels rather than numbers.

    A level A alert is a review gate. It must not be turned into a score term that a good R factor
    can outweigh.
    """

    A = "A"
    B = "B"
    C = "C"
    G = "G"
    """Informational ('general information'), not a defect."""

    UNKNOWN = "unknown"

    @property
    def is_severe(self) -> bool:
        return self in (AlertSeverity.A, AlertSeverity.B)


@dataclass(frozen=True)
class Alert:
    """One validation finding."""

    code: str
    severity: AlertSeverity
    message: str
    alert_type: int | None = None
    """PLATON alert type: 1 CIF construction, 2 structural, 3 data quality, 4 geometry, 5 other."""

    value: str = ""
    acknowledged: bool = False
    acknowledgement_reason: str = ""

    @property
    def is_unresolved_severe(self) -> bool:
        return self.severity.is_severe and not self.acknowledged


@dataclass(frozen=True)
class ValidationReport:
    """Output of a validation tool run, with the raw file kept alongside."""

    tool: str
    tool_version: str = ""
    alerts: tuple[Alert, ...] = ()
    raw_path: str = ""
    check_def_version: str = ""
    """PLATON's ``check.def`` carries a yymmdd version that must match the executable. A mismatch is
    diagnosed rather than ignored."""

    completed: bool = True
    failure_reason: str = ""

    def by_severity(self, severity: AlertSeverity) -> tuple[Alert, ...]:
        return tuple(a for a in self.alerts if a.severity is severity)

    @property
    def unresolved_severe(self) -> tuple[Alert, ...]:
        return tuple(a for a in self.alerts if a.is_unresolved_severe)

    @property
    def has_unresolved_severe(self) -> bool:
        return bool(self.unresolved_severe)


@dataclass(frozen=True)
class ResidualDensityPeak:
    """A difference-Fourier peak. Height alone is not interesting; location is."""

    height_e_per_a3: float
    x: float
    y: float
    z: float
    nearest_atom: AtomId | None = None
    nearest_atom_element: str = ""
    distance_to_nearest_a: float | None = None

    def is_near_heavy_atom(self, heavy_elements: frozenset[str]) -> bool:
        """A large peak beside a heavy atom is usually absorption or termination ripple.

        The same peak in the middle of a void is usually a missing solvent or anion. Both are
        recorded; only the second normally means the model is incomplete.
        """
        return self.nearest_atom_element.strip().capitalize() in heavy_elements


@dataclass(frozen=True)
class ResidualDensity:
    """Residual electron density after refinement."""

    max_peak_e_per_a3: float | None = None
    min_hole_e_per_a3: float | None = None
    peaks: tuple[ResidualDensityPeak, ...] = ()
    rms_e_per_a3: float | None = None

    def significant_peaks_away_from_heavy(
        self, heavy_elements: frozenset[str], *, threshold_e_per_a3: float
    ) -> tuple[ResidualDensityPeak, ...]:
        """Peaks above ``threshold`` that are *not* adjacent to a heavy atom.

        These are the ones that usually mean something is missing from the model.
        """
        return tuple(
            p
            for p in self.peaks
            if p.height_e_per_a3 >= threshold_e_per_a3
            and not p.is_near_heavy_atom(heavy_elements)
        )


@dataclass(frozen=True)
class RefinementMetrics:
    """SHELXL refinement result.

    ``reflection_basis_key`` is what makes two candidates comparable: it combines the reflection file
    digest, the resolution cut, the HKLF format and any twin law. Candidates with different keys are
    refused comparison rather than ranked against each other.
    """

    r1: float | None = None
    wr2: float | None = None
    goof: float | None = None
    restrained_goof: float | None = None
    n_reflections_total: int | None = None
    n_reflections_observed: int | None = None
    n_parameters: int | None = None
    n_restraints: int | None = None
    n_constraints: int | None = None
    max_shift_over_esd: float | None = None
    converged: bool | None = None
    residual_density: ResidualDensity | None = None
    reflection_basis_key: str = ""
    i_over_sigma_cut: float | None = None
    raw_path: str = ""
    parse_ok: bool = True
    parse_failure_reason: str = ""

    @property
    def data_to_parameter_ratio(self) -> float | None:
        if not self.n_parameters:
            return None
        n_data = self.n_reflections_observed or self.n_reflections_total
        if not n_data:
            return None
        return n_data / self.n_parameters

    @property
    def goof_deviation(self) -> float | None:
        """``|GooF - 1|``. Used as an ordering key, never as a pass/fail on its own."""
        if self.goof is None:
            return None
        return abs(self.goof - 1.0)


class GeometryIssueKind(str, enum.Enum):
    BOND_TOO_SHORT = "bond_too_short"
    BOND_TOO_LONG = "bond_too_long"
    ATOM_CLASH = "atom_clash"
    IMPLAUSIBLE_COORDINATION_NUMBER = "implausible_coordination_number"
    METAL_DONOR_DISTANCE_OUT_OF_RANGE = "metal_donor_distance_out_of_range"
    ELEMENT_OUTSIDE_DECLARED_SFAC = "element_outside_declared_sfac"
    CHARGE_IMBALANCE = "charge_imbalance"
    DISCONNECTED_FRAGMENT = "disconnected_fragment"


@dataclass(frozen=True)
class GeometryIssue:
    kind: GeometryIssueKind
    message: str
    atoms: tuple[AtomId, ...] = ()
    value: float | None = None
    expected_min: float | None = None
    expected_max: float | None = None
    is_fatal: bool = True
    """Fatal issues fail the chemical-validity gate. Non-fatal ones are recorded and shown."""


@dataclass(frozen=True)
class GeometryReport:
    """Derived chemical sanity of the model."""

    issues: tuple[GeometryIssue, ...] = ()
    coordination_numbers: dict[str, int] = field(default_factory=dict)
    charge_balance_net: int | None = None
    charge_balance_checked: bool = False
    evaluated: bool = True
    not_evaluated_reason: str = ""

    @property
    def fatal_issues(self) -> tuple[GeometryIssue, ...]:
        return tuple(i for i in self.issues if i.is_fatal)

    @property
    def is_chemically_valid(self) -> bool:
        return not self.fatal_issues


@dataclass(frozen=True)
class AtomConfidence:
    """Per-atom model output, with the full top-k retained.

    The top-1 / top-2 margin is the quantity the CrystalX paper uses to generate an additional
    structure-level candidate by flipping the single most ambiguous assignment. Keeping the whole
    distribution is what makes that possible, and what stops the UI from implying more certainty
    than the model expressed.
    """

    atom_id: AtomId
    predicted: str
    topk: tuple[tuple[str, float], ...] = ()

    @property
    def top1_probability(self) -> float | None:
        return self.topk[0][1] if self.topk else None

    @property
    def margin(self) -> float | None:
        """``p(top1) - p(top2)``. Smaller means more ambiguous, so flip these first."""
        if len(self.topk) < 2:
            return None
        return self.topk[0][1] - self.topk[1][1]


@dataclass(frozen=True)
class ConfidenceReport:
    """Model confidence for a candidate.

    Reported in its own block, never merged into the crystallographic ordering key.
    """

    model_name: str = ""
    model_sha256: str = ""
    atoms: tuple[AtomConfidence, ...] = ()
    device: str = ""
    note: str = "Model confidence orders the search; it is not evidence that a structure is correct."

    def most_ambiguous(self, limit: int = 1) -> tuple[AtomConfidence, ...]:
        """Atoms with the smallest top-1 / top-2 margin, most ambiguous first."""
        scored = [a for a in self.atoms if a.margin is not None]
        scored.sort(key=lambda a: a.margin)  # type: ignore[arg-type,return-value]
        return tuple(scored[:limit])
