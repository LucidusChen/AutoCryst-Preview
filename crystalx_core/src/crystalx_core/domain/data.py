"""Diffraction-data description and the Stage-0 diagnosis.

Anything this module cannot determine from the user's files alone is reported as ``None`` together
with a stated reason, never approximated. Two examples that matter:

* **Completeness** needs the space-group symmetry to know how many unique reflections were possible.
  Until a symmetry provider is registered, completeness is ``None``.
* **Anomalous signal strength** needs ``f''(element, wavelength)``. Dispersion corrections are not
  hard-coded here; ``AnomalousScatteringTable`` is the injection point and ``cctbx.eltbx`` is the
  intended provider.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "HklfFormat",
    "AnomalousAvailability",
    "SpaceGroupConfidence",
    "UnitCell",
    "ResolutionPolicy",
    "ResolutionShellStats",
    "ReflectionDataSummary",
    "AnomalousScatteringTable",
    "InsHeader",
    "DataDiagnosis",
    "Unavailable",
]


class HklfFormat(enum.IntEnum):
    """SHELX ``HKLF`` reflection-file format code."""

    HKLF3 = 3
    """h k l F sigma(F)."""

    HKLF4 = 4
    """h k l F^2 sigma(F^2). The ordinary modern case."""

    HKLF5 = 5
    """Twinned data with a batch/component number. Constrains which backends are meaningful."""

    HKLF6 = 6
    """Twinned data, HKLF 5 variant with all components listed."""


class AnomalousAvailability(str, enum.Enum):
    """Whether the reflection file can support experimental phasing at all."""

    FRIEDEL_PAIRS_PRESENT = "friedel_pairs_present"
    """Both ``h k l`` and ``-h -k -l`` occur as separate records: SHELXC/D/E is possible."""

    FRIEDEL_PAIRS_SPARSE = "friedel_pairs_sparse"
    """A few mates exist, but far too few to be unmerged anomalous data.

    Re-measured standards, a duplicated batch and centric reflections all leave a handful of
    ``(h,k,l)``/``(-h,-k,-l)`` coincidences in a file that was otherwise merged. Treating those as
    anomalous data would open the experimental-phasing route on a dataset that cannot support it,
    so this is a distinct state and is **not** accepted by any rule that requires
    :attr:`FRIEDEL_PAIRS_PRESENT`. See ``shelx_reader.MIN_FRIEDEL_COVERAGE``.
    """

    MERGED_NO_PAIRS = "merged_no_pairs"
    """Friedel mates were merged away. The SHELXC-D-E route is not applicable to this file."""

    UNKNOWN = "unknown"
    """Not determined (file unreadable, or scan not run)."""


class SpaceGroupConfidence(str, enum.Enum):
    """How sure we are about the space group before solving."""

    USER_DECLARED = "user_declared"
    FROM_INS = "from_ins"
    """``LATT``/``SYMM`` present in the instruction file."""

    UNKNOWN = "unknown"
    """No symmetry declared. SHELXT can determine it; SHELXS/SHELXD cannot be run without it."""


@dataclass(frozen=True)
class Unavailable:
    """Explicit 'not computed, and here is why', used instead of a silent ``None``."""

    reason: str
    blocking_capability: str = ""


@dataclass(frozen=True)
class UnitCell:
    """Direct-space unit cell in angstrom and degrees."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float

    def __post_init__(self) -> None:
        for name in ("a", "b", "c"):
            if getattr(self, name) <= 0:
                raise ValueError(f"Cell edge {name} must be positive, got {getattr(self, name)}")
        for name in ("alpha", "beta", "gamma"):
            angle = getattr(self, name)
            if not 0.0 < angle < 180.0:
                raise ValueError(f"Cell angle {name} must be in (0, 180), got {angle}")

    @property
    def volume(self) -> float:
        """Cell volume in cubic angstrom."""
        ca, cb, cg = (math.cos(math.radians(x)) for x in (self.alpha, self.beta, self.gamma))
        radicand = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
        if radicand <= 0.0:
            raise ValueError("Degenerate unit cell: non-positive volume radicand")
        return self.a * self.b * self.c * math.sqrt(radicand)

    def reciprocal(self) -> tuple[float, float, float, float, float, float]:
        """Return ``(a*, b*, c*, cos(alpha*), cos(beta*), cos(gamma*))``."""
        sa, sb, sg = (math.sin(math.radians(x)) for x in (self.alpha, self.beta, self.gamma))
        ca, cb, cg = (math.cos(math.radians(x)) for x in (self.alpha, self.beta, self.gamma))
        volume = self.volume
        a_star = self.b * self.c * sa / volume
        b_star = self.a * self.c * sb / volume
        c_star = self.a * self.b * sg / volume
        cos_alpha_star = (cb * cg - ca) / (sb * sg)
        cos_beta_star = (ca * cg - cb) / (sa * sg)
        cos_gamma_star = (ca * cb - cg) / (sa * sb)
        return a_star, b_star, c_star, cos_alpha_star, cos_beta_star, cos_gamma_star

    def d_spacing(self, h: int, k: int, l: int) -> float:
        """Interplanar spacing ``d`` in angstrom for reflection ``(h k l)``.

        Uses the general triclinic reciprocal-metric expression

            1/d^2 = h^2 a*^2 + k^2 b*^2 + l^2 c*^2
                    + 2 k l b* c* cos(alpha*)
                    + 2 h l a* c* cos(beta*)
                    + 2 h k a* b* cos(gamma*)

        which reduces to ``d = a / sqrt(h^2 + k^2 + l^2)`` for a cubic cell.
        """
        if h == 0 and k == 0 and l == 0:
            raise ValueError("d-spacing is undefined for reflection (0 0 0)")
        a_s, b_s, c_s, cos_al_s, cos_be_s, cos_ga_s = self.reciprocal()
        inv_d2 = (
            (h * a_s) ** 2
            + (k * b_s) ** 2
            + (l * c_s) ** 2
            + 2 * k * l * b_s * c_s * cos_al_s
            + 2 * h * l * a_s * c_s * cos_be_s
            + 2 * h * k * a_s * b_s * cos_ga_s
        )
        if inv_d2 <= 0.0:
            raise ValueError(f"Non-positive 1/d^2 for reflection ({h} {k} {l})")
        return 1.0 / math.sqrt(inv_d2)


@dataclass(frozen=True)
class ResolutionPolicy:
    """The resolution cut a candidate was refined against.

    Part of the comparability gate: two candidates refined on different resolution cuts are not
    comparable, no matter how their R factors look.
    """

    d_min_a: float | None = None
    d_max_a: float | None = None
    applied_by: str = "data"

    def key(self) -> tuple[float | None, float | None]:
        return (self.d_min_a, self.d_max_a)


@dataclass(frozen=True)
class ResolutionShellStats:
    """Per-shell statistics, outermost shell being the interesting one."""

    d_min_a: float
    d_max_a: float
    n_reflections: int
    mean_i_over_sigma: float | None = None


@dataclass(frozen=True)
class ReflectionDataSummary:
    """What we could learn from the user's ``.hkl`` alone."""

    source: Path
    sha256: str
    n_reflections: int
    hklf_format: HklfFormat
    h_range: tuple[int, int]
    k_range: tuple[int, int]
    l_range: tuple[int, int]
    anomalous: AnomalousAvailability
    n_friedel_pairs: int = 0
    friedel_coverage: float = 0.0
    """Fraction of records that participate in a Friedel pair. 0.0 when there are none."""

    hklf_was_declared: bool = False
    """True when ``HKLF`` came from the instruction file rather than from record shape."""

    d_min_a: float | None = None
    d_max_a: float | None = None
    mean_i_over_sigma: float | None = None
    outer_shell: ResolutionShellStats | None = None
    completeness: float | None = None
    completeness_unavailable: Unavailable | None = None
    n_zero_sigma: int = 0
    truncated_at_terminator: bool = True
    warnings: tuple[str, ...] = ()

    @property
    def has_unmerged_friedel_pairs(self) -> bool:
        return self.anomalous is AnomalousAvailability.FRIEDEL_PAIRS_PRESENT


@runtime_checkable
class AnomalousScatteringTable(Protocol):
    """Provider of dispersion corrections.

    Intentionally a protocol with **no built-in implementation** in ``crystalx_core``. Anomalous
    signal strength depends on ``f''(element, wavelength)`` -- the same silver compound behaves very
    differently at Cu K-alpha and Mo K-alpha -- and inventing those numbers would be worse than
    admitting we do not have them. ``cctbx.eltbx.sasaki`` / ``henke`` is the intended provider and is
    already a dependency of the inference engine.

    Until a provider is registered, ``SelectionRule`` R07 reports ``INSUFFICIENT_INFORMATION``.
    """

    def f_double_prime(self, element: str, wavelength_a: float) -> float:
        """Imaginary dispersion correction ``f''`` in electrons."""
        ...


@dataclass(frozen=True)
class InsHeader:
    """The instruction-file header, read token-anchored (never by substring matching)."""

    source: Path
    sha256: str
    title: str = ""
    wavelength_a: float | None = None
    cell: UnitCell | None = None
    z: float | None = None
    latt: int | None = None
    symm: tuple[str, ...] = ()
    sfac: tuple[str, ...] = ()
    unit: tuple[float, ...] = ()
    hklf_format: HklfFormat | None = None
    space_group_confidence: SpaceGroupConfidence = SpaceGroupConfidence.UNKNOWN
    rem_lines: tuple[str, ...] = ()
    """``REM`` records, verbatim. Vendor pipelines announce themselves here, which is how a
    machine-written placeholder composition is recognised."""

    has_atom_records: bool = False
    """Whether the file contains atoms. Separates a refined model from a pre-solution input."""

    warnings: tuple[str, ...] = ()

    def symmetry(self):
        """Exact Sohncke / improper-operation analysis of ``LATT`` and ``SYMM``.

        Returns :class:`crystalx_core.domain.symmetry.SpaceGroupSymmetry`. Imported lazily to keep
        this module free of intra-domain import cycles.
        """
        from crystalx_core.domain.symmetry import analyse_symmetry

        return analyse_symmetry(self.latt, self.symm)

    def composition_trust(
        self, declared_metals: tuple[str, ...] = (), declared_elements: tuple[str, ...] = ()
    ):
        """Whether ``SFAC``/``UNIT`` carry information or are a filled-in field.

        Returns :class:`crystalx_core.domain.composition.CompositionTrust`.
        """
        from crystalx_core.domain.composition import assess_composition

        return assess_composition(
            sfac=self.sfac,
            unit=self.unit,
            z=int(self.z) if self.z else None,
            rem_lines=self.rem_lines,
            declared_metals=declared_metals,
            declared_elements=declared_elements,
            has_atom_records=self.has_atom_records,
        )

    @property
    def non_hydrogen_unit_total(self) -> float:
        total = 0.0
        for symbol, count in zip(self.sfac, self.unit):
            if symbol.strip().upper() != "H":
                total += count
        return total

    @property
    def is_centrosymmetric_latt(self) -> bool | None:
        """``LATT`` is positive for a centrosymmetric lattice and negative otherwise."""
        if self.latt is None:
            return None
        return self.latt > 0


@dataclass(frozen=True)
class DataDiagnosis:
    """The Stage-0 picture: what the data are, what the chemistry is said to be, what tools exist."""

    reflections: ReflectionDataSummary
    ins: InsHeader
    resolution_policy: ResolutionPolicy = field(default_factory=ResolutionPolicy)
    notes: tuple[str, ...] = ()

    @property
    def wavelength_a(self) -> float | None:
        return self.ins.wavelength_a

    @property
    def space_group_confidence(self) -> SpaceGroupConfidence:
        return self.ins.space_group_confidence
